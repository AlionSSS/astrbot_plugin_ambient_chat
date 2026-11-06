"""自主发言执行：统一接纳、并发、发送节奏、取消与有限收尾。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..speech.delivery import PreparedSpeech

if TYPE_CHECKING:
    from ..main import AmbientChatPlugin


@dataclass(frozen=True, slots=True)
class SpeechResult:
    state: str
    reason: str

    @property
    def ok(self) -> bool:
        """``partial`` 也算说过话：只发出一部分片段仍是确切的事实。"""
        return self.state in {"sent", "history_failed", "partial"}


def _partial_reason(outcome: Any) -> str:
    if outcome.stopped_by == "cancelled":
        return f"已发 {outcome.sent}/{outcome.total} 段，其余因停止或配置变更未发"
    return f"已发 {outcome.sent}/{outcome.total} 段，第 {outcome.sent + 1} 段发送失败"


@dataclass(slots=True)
class _Execution:
    policy: str
    model_id: str = ""
    phase: str = "queued"
    valid: bool = True
    task: asyncio.Task | None = None


def _binding_policy(plugin: AmbientChatPlugin, binding_id: str) -> str | None:
    """冻结执行策略；展示名称与排序时间不影响执行的有效性。"""
    binding = plugin.store.get_binding(binding_id)
    if not plugin.is_enabled() or binding is None or not binding.enabled:
        return None
    rule = plugin.store.get_rule(binding.rule_id)
    if rule is None:
        return None
    bound = binding.to_dict()
    bound.pop("name", None)
    content = rule.to_dict()
    content.pop("updated_at", None)
    content.pop("name", None)
    model = next(
        (
            m.to_dict()
            for m in plugin.settings.decision_models
            if m.ref == rule.intent.decision_provider_id
        ),
        None,
    )
    if model:
        model.pop("name", None)
    return json.dumps([bound, content, model], sort_keys=True, ensure_ascii=False)


class SpeechExecution:
    def __init__(
        self,
        plugin: AmbientChatPlugin,
        *,
        send_timeout: float = 60,
        record_timeout: float = 10,
        accepts: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._plugin = plugin
        self._send_timeout = send_timeout
        self._record_timeout = record_timeout
        self._accepts = accepts
        self._active: dict[tuple[str, str], _Execution] = {}
        self._results: dict[tuple[str, str], dict[str, Any]] = {}
        self._policies: dict[str, str | None] = {}
        self._syncing: dict[str, int] = {}
        self._syncing_models: dict[str, int] = {}
        self._models: dict[str, tuple] = {}
        self._closed = False
        self._inflight = 0
        self._last_send: float | None = None
        self._send_lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._reconcile_lock = asyncio.Lock()
        self._detached: set[asyncio.Task] = set()

    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    async def execute(
        self,
        binding_id: str,
        umo: str,
        *,
        on_admitted: Callable[[], None] | None = None,
    ) -> SpeechResult:
        """接纳后、首次让出控制权前通知入口；等待并发名额也算已接纳。"""
        key = (binding_id, umo)
        if key in self._active:
            return SpeechResult("busy", "忙碌跳过：该会话已有执行")
        policy = _binding_policy(self._plugin, binding_id)
        binding = self._plugin.store.get_binding(binding_id)
        if (
            self._closed
            or policy is None
            or binding is None
            or umo not in binding.targets
            or (self._accepts is not None and not self._accepts(binding_id, umo))
        ):
            return SpeechResult("cancelled", "已取消：绑定未运行或目标已移除")
        rule = self._plugin.store.get_rule(binding.rule_id)
        ref = rule.intent.decision_provider_id
        model_id = ref.removeprefix("custom:") if ref.startswith("custom:") else ""
        if (
            binding_id in self._syncing
            or model_id in self._syncing_models
            or self._policies.get(binding_id) != policy
        ):
            return SpeechResult("cancelled", "已取消：配置正在同步")
        run = _Execution(policy, model_id)
        self._active[key] = run
        run.task = asyncio.create_task(
            self._run(key, run), name=f"ambient_speech_{binding_id}"
        )
        if on_admitted is not None:
            on_admitted()
        try:
            return await asyncio.shield(run.task)
        except asyncio.CancelledError:
            # 调度循环/请求退出也必须遵守相同收尾，不能把实际发送一起取消。
            if run.phase not in {"sending", "recording"}:
                run.valid = False
                if run.phase != "queued":
                    run.task.cancel()
                self._notify()
            await self._join(run.task)
            raise

    async def reconcile(self) -> None:
        # 差分和接纳闸门在首次 await 前登记。各次同步只等待自身影响的执行，
        # 停止 B 不因 A 的发送收尾而排队；重叠变更用计数保留闸门。
        policies = self._current_policies()
        models = self._current_models()
        affected = {
            bid
            for bid in self._policies.keys() | policies.keys()
            if self._policies.get(bid) != policies.get(bid)
        }
        changed_models = {
            mid
            for mid in self._models.keys() | models.keys()
            if self._models.get(mid) != models.get(mid)
        }
        self._policies = policies
        self._models = models
        for bid in affected:
            self._syncing[bid] = self._syncing.get(bid, 0) + 1
        for mid in changed_models:
            self._syncing_models[mid] = self._syncing_models.get(mid, 0) + 1
        tasks = []
        for (bid, _), run in self._active.items():
            if bid not in affected and run.model_id not in changed_models:
                continue
            run.valid = False
            if run.phase not in {"queued", "sending", "recording"}:
                run.task.cancel()
            tasks.append(run.task)
        self._notify()
        try:
            for task in tasks:
                await self._join(task)
            # 只释放本次变化的模型，不替另一条尚在取消判断的同步提前关闭客户端。
            async with self._reconcile_lock:
                await self._plugin.router.reload(changed_models)
            if (
                self._policies != self._current_policies()
                or self._models != self._current_models()
            ):
                await self.reconcile()
        finally:
            for values, changed in (
                (self._syncing, affected),
                (self._syncing_models, changed_models),
            ):
                for key in changed:
                    values[key] -= 1
                    if not values[key]:
                        values.pop(key)
            self._notify()
        known = {
            (b.id, umo)
            for b in self._plugin.store.bindings.values()
            for umo in b.targets
        }
        self._results = {
            key: value for key, value in self._results.items() if key in known
        }

    def _current_models(self) -> dict[str, tuple]:
        return {
            model.id: (model.base_url, model.api_key, model.model)
            for model in self._plugin.settings.decision_models
        }

    def _current_policies(self) -> dict[str, str | None]:
        return {
            bid: _binding_policy(self._plugin, bid)
            for bid in self._plugin.store.bindings
        }

    async def close(self) -> None:
        self._closed = True
        tasks = []
        for run in self._active.values():
            run.valid = False
            if run.phase not in {"queued", "sending", "recording"}:
                run.task.cancel()
            tasks.append(run.task)
        self._notify()
        for task in tasks:
            await self._join(task)

    @staticmethod
    async def _join(task: asyncio.Task) -> None:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        # _run 自己将取消转换为结果。
        if not task.cancelled():
            task.result()

    def snapshot(self) -> dict[tuple[str, str], dict[str, Any]]:
        result = {key: dict(value) for key, value in self._results.items()}
        for key, run in self._active.items():
            result.setdefault(key, {})["phase"] = run.phase
        return result

    def is_idle(self, binding_id: str) -> bool:
        """该绑定已无执行（含发送和历史记录收尾），可以清理其旧目标缓冲。"""
        return not any(bid == binding_id for bid, _ in self._active)

    def _valid(self, key: tuple[str, str], run: _Execution) -> bool:
        return (
            run.valid
            and not self._closed
            and _binding_policy(self._plugin, key[0]) == run.policy
            and (self._accepts is None or self._accepts(*key))
        )

    async def _run(self, key: tuple[str, str], run: _Execution) -> SpeechResult:
        admitted = False
        outcome = SpeechResult("cancelled", "已取消：停止或配置变化")
        try:
            while self._inflight >= self._plugin.settings.concurrency:
                if not self._valid(key, run):
                    return outcome
                await self._changed.wait()
            self._inflight += 1
            admitted = True
            if self._valid(key, run):
                outcome = await self._perform(key, run)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Ambient] 自主发言执行异常（%s）：%s", key, exc)
            outcome = SpeechResult("skipped", f"执行失败：{type(exc).__name__}")
        finally:
            previous = self._results.get(key, {})
            self._results[key] = {
                "state": outcome.state,
                "reason": outcome.reason,
                "last_fire_at": datetime.now(self._plugin.timezone).isoformat(
                    timespec="seconds"
                )
                if outcome.ok
                else previous.get("last_fire_at", ""),
                "phase": "idle",
            }
            self._active.pop(key, None)
            if admitted:
                self._inflight -= 1
            self._notify()
        return outcome

    async def _perform(self, key: tuple[str, str], run: _Execution) -> SpeechResult:
        plugin = self._plugin
        bid, umo = key
        rule = plugin.store.get_rule(plugin.store.get_binding(bid).rule_id)
        run.phase = "judging"
        now = datetime.now(plugin.timezone)
        blocked = plugin.blockers.check(rule, umo, now)
        if blocked.blocked:
            return SpeechResult("skipped", blocked.reason)
        intent = await plugin.judge.judge(rule, umo, plugin.buffer, now)
        if not intent.speak:
            # 判定失败不是「不想说话」：单独的 state 让页面标红并提示排查方向。
            state = "decision_failed" if intent.failure else "skipped"
            return SpeechResult(state, intent.reason or "无意向")
        persona = await plugin.judge.persona_prompt(umo) if rule.inject_persona else ""
        run.phase = "generating"
        speech = await plugin.delivery.prepare(
            rule, umo, intent.hint, persona, datetime.now(plugin.timezone)
        )
        if speech is None:
            return SpeechResult("skipped", "无可投递内容：会话不可用或生成失败")
        run.phase = "pacing"
        async with self._send_lock:
            loop = asyncio.get_running_loop()
            while self._last_send is not None:
                wait = (
                    self._last_send
                    + plugin.settings.send_interval_seconds
                    - loop.time()
                )
                if wait <= 0:
                    break
                changed = self._changed
                try:
                    async with asyncio.timeout(wait):
                        await changed.wait()
                except TimeoutError:
                    pass
            started = asyncio.Event()
            send = asyncio.create_task(self._send(key, run, speech, started))
            try:
                await started.wait()
            except asyncio.CancelledError:
                send.cancel()
                raise
            # 发送锁罩住整次投递（含所有分段），间隔基准才会落在收尾时刻。
            deadline = loop.time() + self._send_timeout
            try:
                outcome = await self._bounded(send, max(0, deadline - loop.time()))
            except TimeoutError:
                outcome = SpeechResult(
                    "unknown", "发送结果未知（等待超过 60 秒，不自动重发）"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Ambient] 发送异常（%s）：%s", umo, exc)
                outcome = SpeechResult("unknown", "发送异常，结果未知（不自动重发）")
            finally:
                self._last_send = loop.time()

        if isinstance(outcome, SpeechResult):
            return outcome
        if not outcome.sent:
            return SpeechResult("skipped", "发送失败")
        run.phase = "recording"
        # 部分发送只记实际发出的片段，否则历史里会出现没说过的话。
        record_speech = (
            speech
            if outcome.complete
            else PreparedSpeech(speech.umo, speech.conversation_id, outcome.text)
        )
        try:
            recorded = await self._bounded(
                asyncio.create_task(plugin.delivery.record(record_speech)),
                self._record_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 已发送但历史记录失败（%s）：%s", umo, exc)
            recorded = False
        if not outcome.complete:
            reason = _partial_reason(outcome)
            if not recorded:
                reason = f"{reason}，会话历史记录失败"
            return SpeechResult("partial", reason)
        if not recorded:
            return SpeechResult("history_failed", "已发送，但会话历史记录失败")
        return SpeechResult("sent", f"已发言（{intent.reason}）")

    async def _send(
        self, key: tuple[str, str], run: _Execution, speech: Any, started: asyncio.Event
    ) -> bool | SpeechResult:
        started.set()
        if not self._valid(key, run):
            return SpeechResult("cancelled", "已取消：停止或配置变化")
        rule = self._plugin.store.get_rule(
            self._plugin.store.get_binding(key[0]).rule_id
        )
        blocked = self._plugin.blockers.check(
            rule, key[1], datetime.now(self._plugin.timezone)
        )
        if blocked.blocked:
            return SpeechResult("skipped", blocked.reason)
        # 从复核到实际调用发送不让出控制权；段间用 _valid 判断还要不要发下一段。
        run.phase = "sending"
        return await self._plugin.delivery.send(
            speech, should_continue=lambda: self._valid(key, run)
        )

    async def _bounded(self, task: asyncio.Task, timeout: float) -> Any:
        # wait_for 会等待协程响应取消，无法保证本地等待上限。
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done:
            if task.cancelled():
                raise RuntimeError("投递 adapter 自行取消")
            return task.result()
        task.cancel()
        self._detached.add(task)
        task.add_done_callback(self._forget_detached)
        raise TimeoutError

    def _forget_detached(self, task: asyncio.Task) -> None:
        self._detached.discard(task)
        if not task.cancelled():
            task.exception()
