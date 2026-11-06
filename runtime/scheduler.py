"""调度器：为每条「Ambient 绑定」维护触发循环。

调度语义见 ``docs/adr/0003``（引擎）、``docs/adr/0004``（时段随机）、
``docs/adr/0006``（绑定模型 / 一对多时的会话挑选）与 ``docs/adr/0024``
（最小间隔、确定性采样、触发合并、固定间隔）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from astrbot.api import logger

from ..common import TRIGGER_CRON, TRIGGER_INTERVAL, TRIGGER_ONCE, parse_hhmm
from ..rules.models import SELECTION_RANDOM, Binding, Rule, RuleError, Trigger
from .execution import SpeechResult

if TYPE_CHECKING:
    from ..main import AmbientChatPlugin

MERGE_WINDOW_SECONDS = 60.0
"""同一绑定的两次触发间隔小于该值时合并成一次。

cron 的最小粒度是分钟，多个触发器（或 cron 与时段随机）撞在同一分钟时，
上一版会立刻再跑一遍「拦截 → 意向 → 投递」；合并后同一分钟内只认真实触发一次。
"""

MAX_LOOKAHEAD_DAYS = 8
"""时段随机在限定周几时最多向后找几天（一周 + 1 天兜底）。"""


class AmbientScheduler:
    """按 Ambient 绑定（机器人应用 + 规则 + 目标会话）运行触发循环。

    一次触发可能覆盖多个目标会话：绑定为一对多且策略为随机时，先随机挑出
    N 个会话，再按挑选顺序逐个走过「拦截 → 意向 → 投递」。
    """

    def __init__(self, plugin: AmbientChatPlugin) -> None:
        self._plugin = plugin
        self._tasks: dict[str, asyncio.Task] = {}
        self._retiring: set[asyncio.Task] = set()
        self._window_cache: dict[tuple[str, int, date], list[datetime]] = {}
        self._consumed_once: set[tuple[str, int]] = set()
        self._expired_warned: set[tuple[str, int]] = set()
        self._interval_anchor: dict[tuple[str, int], datetime] = {}
        self._schedule_policy: dict[str, str] = {}
        self._triggers: dict[str, list] = {}
        self._next_fire_at: dict[str, datetime] = {}
        self._binding_fired_at: dict[str, datetime] = {}
        self._binding_result: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ 生命周期
    async def rebuild(self) -> None:
        """仅替换策略变化的绑定调度，保留无关任务及触发消耗状态。"""
        async with self._lock:
            policies = {
                binding.id: self._policy(binding)
                for binding in self._plugin.store.bindings.values()
            }
            changed = {
                bid
                for bid in self._schedule_policy.keys() | policies.keys()
                if self._schedule_policy.get(bid) != policies.get(bid)
            }
            tasks = [self._tasks.pop(bid) for bid in changed if bid in self._tasks]
            for task in tasks:
                self._retiring.add(task)
                task.add_done_callback(self._forget_retiring)
                task.cancel()
            # 执行 module 已管理取消和收尾。这里不等别条绑定仍在发送的旧循环，
            # 避免停止 B 的配置同步被 A 的有限发送收尾拖住。
            for bid in changed:
                new = policies.get(bid)
                # 启停/规则非触发字段变化不重置一次性消耗、间隔锚点或随机采样。
                old_triggers = self._triggers.get(bid)
                new_triggers = json.loads(new)[1] if new else None
                if (
                    old_triggers is not None
                    and new_triggers is not None
                    and old_triggers != new_triggers
                ):
                    self._consumed_once = {
                        key for key in self._consumed_once if key[0] != bid
                    }
                    self._expired_warned = {
                        key for key in self._expired_warned if key[0] != bid
                    }
                    self._interval_anchor = {
                        key: value
                        for key, value in self._interval_anchor.items()
                        if key[0] != bid
                    }
                    self._window_cache = {
                        key: value
                        for key, value in self._window_cache.items()
                        if key[0] != bid
                    }
                self._next_fire_at.pop(bid, None)
                if new is None:
                    continue
                self._triggers[bid] = new_triggers
                binding = self._plugin.store.get_binding(bid)
                rule = self._plugin.store.get_rule(binding.rule_id)
                task = asyncio.create_task(
                    self._run_binding(binding, rule), name=f"ambient_{bid}"
                )
                task.add_done_callback(
                    lambda done, binding_id=bid: self._forget_task(binding_id, done)
                )
                self._tasks[bid] = task
            self._schedule_policy = {
                bid: policy for bid, policy in policies.items() if policy is not None
            }
            known = set(self._plugin.store.bindings)
            self._triggers = {
                bid: value for bid, value in self._triggers.items() if bid in known
            }
            self._interval_anchor = {
                key: value
                for key, value in self._interval_anchor.items()
                if key[0] in known
            }
            self._consumed_once = {
                key for key in self._consumed_once if key[0] in known
            }
            self._expired_warned = {
                key for key in self._expired_warned if key[0] in known
            }
            self._window_cache = {
                key: value
                for key, value in self._window_cache.items()
                if key[0] in known
            }

    def _policy(self, binding: Binding) -> str | None:
        rule = self._plugin.store.get_rule(binding.rule_id)
        if not self._plugin.is_enabled() or not binding.enabled or rule is None:
            return None
        lifecycle = getattr(self._plugin, "lifecycle", None)
        if lifecycle is not None and not lifecycle.can_schedule(binding.id):
            return None
        bound = binding.to_dict()
        bound.pop("name", None)
        return json.dumps(
            [bound, [rule.id, [trigger.to_dict() for trigger in rule.triggers]]],
            sort_keys=True,
        )

    async def stop(self) -> None:
        async with self._lock:
            tasks, self._tasks = self._tasks, {}
            for task in tasks.values():
                task.cancel()
            for task in set(tasks.values()) | self._retiring:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._schedule_policy.clear()
            self._next_fire_at.clear()

    def _forget_retiring(self, task: asyncio.Task) -> None:
        self._retiring.discard(task)
        if not task.cancelled():
            task.exception()

    def _forget_task(self, binding_id: str, task: asyncio.Task) -> None:
        if self._tasks.get(binding_id) is task:
            self._tasks.pop(binding_id, None)

    # ------------------------------------------------------------------ 手动触发
    async def trigger_target(self, binding_id: str, umo: str) -> dict[str, Any]:
        """立即对一个目标会话执行一次「拦截 → 意向 → 投递」。

        只用于 WebUI「手动触发一次」：跳过入口触发、挑选策略与触发合并，
        其余判断与定时触发完全一致；被拦截或判定不发言时不发送任何消息。
        """
        binding = self._plugin.store.get_binding(binding_id)
        if binding is None:
            raise RuleError(f"绑定不存在：{binding_id}")
        rule = self._plugin.store.get_rule(binding.rule_id)
        if rule is None:
            raise RuleError(f"规则不存在：{binding.rule_id}")
        if not binding.enabled:
            raise RuleError("绑定未在运行，请先启动后再手动触发")
        if umo not in binding.targets:
            raise RuleError("该会话不在这条绑定的目标里")
        if not self._plugin.is_enabled():
            raise RuleError("「启用自主发言」已关闭，无法触发")

        admitted = False

        def on_admitted() -> None:
            nonlocal admitted
            admitted = True
            self._binding_fired_at[binding.id] = datetime.now(self._plugin.timezone)

        outcome = await self._plugin.execution.execute(
            binding.id, umo, on_admitted=on_admitted
        )
        if admitted:
            self._binding_result[binding.id] = (
                f"手动触发 {_short_umo(umo)}：{outcome.reason}"
            )
        return {
            "binding_id": binding.id,
            "umo": umo,
            "spoke": outcome.ok,
            "state": outcome.state,
            "reason": outcome.reason,
        }

    # ------------------------------------------------------------------ 单绑定
    async def _run_binding(self, binding: Binding, rule: Rule) -> None:
        tz = self._plugin.timezone
        try:
            while True:
                now = datetime.now(tz)
                upcoming = self._next_fire(binding, rule, now, tz)
                if upcoming is None:
                    self._next_fire_at.pop(binding.id, None)
                    if any(trigger.is_expired(now) for trigger in rule.triggers):
                        self._binding_result[binding.id] = (
                            "没有后续触发：一次性触发已过期"
                        )
                    return
                fire_at, trigger = upcoming
                self._next_fire_at[binding.id] = fire_at
                delay = (fire_at - now).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)
                self._consume_once(binding, rule, trigger)
                if self._should_merge(binding, datetime.now(tz)):
                    logger.info(
                        "[Ambient] 绑定 %s 在 %g 秒内重复触发，已合并。",
                        binding.id,
                        MERGE_WINDOW_SECONDS,
                    )
                    continue
                await self._fire(binding)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Ambient] 绑定 %s 调度异常：%s", binding.id, exc)
            self._binding_result[binding.id] = f"调度异常：{exc}"

    def _consume_once(self, binding: Binding, rule: Rule, trigger: Trigger) -> None:
        if trigger.type != TRIGGER_ONCE:
            return
        for index, item in enumerate(rule.triggers):
            if item is trigger:
                self._consumed_once.add((binding.id, index))
                return

    def _should_merge(self, binding: Binding, fire_at: datetime) -> bool:
        """与上一次真实触发间隔过近时合并，避免多个触发器撞在同一分钟连跑。"""
        last_at = self._binding_fired_at.get(binding.id)
        if last_at is None:
            return False
        return (fire_at - last_at).total_seconds() < MERGE_WINDOW_SECONDS

    def _next_fire(
        self,
        binding: Binding,
        rule: Rule,
        now: datetime,
        tz: ZoneInfo,
    ) -> tuple[datetime, Trigger] | None:
        candidates: list[tuple[datetime, Trigger]] = []
        for index, trigger in enumerate(rule.triggers):
            fire_at = self._trigger_next_fire(binding, index, trigger, now, tz)
            if fire_at is not None:
                candidates.append((fire_at, trigger))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])

    def _trigger_next_fire(
        self,
        binding: Binding,
        index: int,
        trigger: Trigger,
        now: datetime,
        tz: ZoneInfo,
    ) -> datetime | None:
        if trigger.type == TRIGGER_ONCE:
            if (binding.id, index) in self._consumed_once:
                return None
            if trigger.at is None:
                return None
            fire_at = trigger.at.astimezone(tz)
            if fire_at <= now:
                # 过期不丢弃：规则页会把它标成「已过期」，这里只吞掉这次触发。
                self._consumed_once.add((binding.id, index))
                if (binding.id, index) not in self._expired_warned:
                    self._expired_warned.add((binding.id, index))
                    logger.warning(
                        "[Ambient] 绑定 %s 的一次性触发已过期，不会触发：%s",
                        binding.id,
                        fire_at.isoformat(timespec="minutes"),
                    )
                return None
            return fire_at

        if trigger.type == TRIGGER_CRON:
            from apscheduler.triggers.cron import CronTrigger

            try:
                cron = CronTrigger.from_crontab(trigger.expr, timezone=tz)
            except (TypeError, ValueError):
                return None
            fire_at = cron.get_next_fire_time(None, now)
            if fire_at is not None and fire_at <= now:
                # get_next_fire_time 会返回「恰好等于 now」的时刻，
                # 跳过它以免同一分钟内重复触发。
                fire_at = cron.get_next_fire_time(
                    fire_at, now + timedelta(microseconds=1)
                )
            return fire_at

        if trigger.type == TRIGGER_INTERVAL:
            return self._interval_next_fire(binding, index, trigger, now)

        return self._window_next_fire(binding, index, trigger, now, tz)

    def _interval_next_fire(
        self,
        binding: Binding,
        index: int,
        trigger: Trigger,
        now: datetime,
    ) -> datetime | None:
        """锚定在「绑定首次调度时刻」的等间隔触发，抖动按次数确定性生成。

        锚点只存在进程内：插件重启后从当前时刻重新起算，
        但保存配置不会打乱节奏。重算时回查可能尚未到期的周期，
        并比较实际时刻，避免正向抖动漏触发或相邻周期换序。
        """
        key = (binding.id, index)
        period_seconds = max(1, trigger.every_minutes) * 60.0
        anchor = self._interval_anchor.get(key)
        if anchor is None:
            anchor = now
            self._interval_anchor[key] = anchor
        jitter_span = period_seconds * max(0, trigger.jitter_percent) / 100
        seed = f"{binding.id}:{index}"
        elapsed = (now - anchor).total_seconds()
        # nominal + span <= now 的周期一定已过期；其余周期仍需检查。
        occurrence = max(1, int((elapsed - jitter_span) // period_seconds) + 1)
        next_fire: datetime | None = None
        while True:
            nominal = occurrence * period_seconds
            earliest = anchor + timedelta(seconds=nominal - jitter_span)
            # 后续周期即使取最大负向抖动，也不会比已有候选更早。
            if next_fire is not None and earliest >= next_fire:
                return next_fire
            fire_at = anchor + timedelta(
                seconds=nominal + _interval_jitter(seed, occurrence, jitter_span)
            )
            if fire_at > now and (next_fire is None or fire_at < next_fire):
                next_fire = fire_at
            occurrence += 1

    def _window_next_fire(
        self,
        binding: Binding,
        index: int,
        trigger: Trigger,
        now: datetime,
        tz: ZoneInfo,
    ) -> datetime | None:
        for offset in range(MAX_LOOKAHEAD_DAYS):
            day = now.date() + timedelta(days=offset)
            for sample in self._window_samples(binding, index, trigger, day, tz):
                if sample > now:
                    return sample
        return None

    def _window_samples(
        self,
        binding: Binding,
        index: int,
        trigger: Trigger,
        day: date,
        tz: ZoneInfo,
    ) -> list[datetime]:
        """以绑定 + 日期为种子采样：不同绑定互不相同，同一天重启也不变。"""
        if trigger.days and day.isoweekday() not in trigger.days:
            return []
        key = (binding.id, index, day)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached

        start_hm = parse_hhmm(trigger.start)
        end_hm = parse_hhmm(trigger.end)
        if not start_hm or not end_hm:
            return []
        start_at = datetime.combine(day, time(*start_hm), tz)
        end_at = datetime.combine(day, time(*end_hm), tz)
        span = (end_at - start_at).total_seconds()
        if span <= 0:
            return []

        count = max(1, trigger.count)
        gap = max(1, trigger.gap_minutes) * 60.0
        rng = random.Random(f"{binding.id}:{index}:{day.isoformat()}")
        # 先按「次数 + 最小间隔」压缩可用区间，再把排序后的采样点依次错开一个间隔，
        # 这样既保证任意两次至少相隔 gap，又保留窗口内的随机感。
        slack = span - (count - 1) * gap
        if slack > 0:
            offsets = sorted(rng.uniform(0, slack) for _ in range(count))
            samples = [
                start_at + timedelta(seconds=value + order * gap)
                for order, value in enumerate(offsets)
            ]
        else:
            # 历史配置可能塞不下最小间隔：退回均匀采样，不让这个触发器彻底哑掉。
            offsets = sorted(rng.uniform(0, span) for _ in range(count))
            samples = [start_at + timedelta(seconds=value) for value in offsets]
        self._window_cache[key] = samples
        logger.info(
            "[Ambient] 绑定 %s 在 %s 的随机时刻：%s",
            binding.id,
            day.isoformat(),
            ", ".join(item.strftime("%H:%M:%S") for item in samples),
        )
        return samples

    # ------------------------------------------------------------------ 执行
    def _select_targets(self, binding: Binding) -> list[str]:
        """按挑选策略决定本次触发要处理的目标会话。"""
        targets = list(binding.targets)
        selection = binding.selection
        if selection.mode != SELECTION_RANDOM:
            return targets
        count = max(1, min(selection.count, len(targets)))
        if count >= len(targets):
            return targets
        return random.sample(targets, count)

    async def _fire(self, binding: Binding) -> None:
        selected = self._select_targets(binding)
        if not selected:
            self._binding_result[binding.id] = "没有可用目标会话"
            return

        spoken: list[str] = []
        skipped: list[str] = []
        outcomes: list[SpeechResult] = []
        admitted = False

        def on_admitted() -> None:
            nonlocal admitted
            if not admitted:
                admitted = True
                self._binding_fired_at[binding.id] = datetime.now(self._plugin.timezone)

        for umo in selected:
            outcome = await self._plugin.execution.execute(
                binding.id, umo, on_admitted=on_admitted
            )
            outcomes.append(outcome)
            if outcome.ok:
                spoken.append(umo)
            else:
                skipped.append(f"{_short_umo(umo)}：{outcome.reason}")

        if not admitted:
            return
        unknown = sum(outcome.state == "unknown" for outcome in outcomes)
        history_failed = sum(outcome.state == "history_failed" for outcome in outcomes)
        partial = sum(outcome.state == "partial" for outcome in outcomes)
        summary = f"触发 {len(selected)} 个，确认发言 {len(spoken)} 个"
        if partial:
            summary += f"，部分发送 {partial} 个"
        if unknown:
            summary += f"，发送结果未知 {unknown} 个"
        if history_failed:
            summary += f"，历史记录失败 {history_failed} 个"
        # 绑定摘要只留计数：每个会话的原因由「运行管理」卡片里的目标行逐条展示，
        # 同一批文案再拼进摘要只会让那格又长又难读。
        self._binding_result[binding.id] = summary
        if skipped:
            logger.info(
                "[Ambient] 绑定 %s %s（未发言：%s）。",
                binding.id,
                summary,
                "、".join(skipped),
            )
        else:
            logger.info("[Ambient] 绑定 %s %s。", binding.id, summary)

    # ------------------------------------------------------------------ 状态
    def status(self) -> list[dict]:
        """全部绑定的运行状态（含已停止的绑定）。

        启动 / 停止在「运行状态」页管理，因此这里必须覆盖未启动的绑定，
        否则停用后就再也点不到「启动」。
        """
        now = datetime.now(self._plugin.timezone)
        result: list[dict] = []
        executions = self._plugin.execution.snapshot()
        lifecycle = getattr(self._plugin, "lifecycle", None)
        lifecycle_states = lifecycle.snapshot() if lifecycle is not None else {}
        for binding in self._plugin.store.bindings.values():
            rule = self._plugin.store.get_rule(binding.rule_id)
            running = binding.id in self._tasks
            next_at = self._next_fire_at.get(binding.id) if running else None
            fired_at = self._binding_fired_at.get(binding.id)
            targets = []
            for umo in binding.targets:
                key = (binding.id, umo)
                execution = executions.get(key, {})
                targets.append(
                    {
                        "umo": umo,
                        "last_fire_at": execution.get("last_fire_at", ""),
                        "last_result": execution.get("reason", "等待首次触发"),
                        "state": execution.get("state", ""),
                        "phase": execution.get("phase", "idle"),
                    }
                )
            result.append(
                {
                    "binding_id": binding.id,
                    "platform_id": binding.platform_id,
                    "binding_name": binding.name,
                    "rule_id": binding.rule_id,
                    "rule_name": rule.name if rule else binding.rule_id,
                    "selection": binding.selection.to_dict(),
                    "target_count": len(binding.targets),
                    "enabled": binding.enabled,
                    **lifecycle_states.get(binding.id, {}),
                    "running": running,
                    "next_fire_at": next_at.isoformat(timespec="seconds")
                    if next_at
                    else "",
                    "next_fire_in": max(0, int((next_at - now).total_seconds()))
                    if next_at
                    else None,
                    "last_fire_at": fired_at.isoformat(timespec="seconds")
                    if fired_at
                    else "",
                    "last_result": self._binding_result.get(
                        binding.id,
                        "等待首次触发" if running else "已停止",
                    ),
                    "targets": targets,
                }
            )
        # 只按机器人应用归组、组内保持创建顺序：启停后卡片不再整体重排，
        # 否则同机器人同规则的多条绑定看起来像「一起被停掉了」。
        result.sort(key=lambda item: item["platform_id"])
        return result


def _interval_jitter(seed: str, occurrence: int, span: float) -> float:
    """按「绑定 + 触发器 + 第几次」确定性生成抖动，重复计算不会漂移。"""
    if span <= 0:
        return 0.0
    return random.Random(f"{seed}:{occurrence}").uniform(-span, span)


def _short_umo(umo: str) -> str:
    parts = umo.split(":", 2)
    if len(parts) != 3:
        return umo
    kind = "群" if parts[1] == "GroupMessage" else "私聊"
    return f"{kind}{parts[2]}"
