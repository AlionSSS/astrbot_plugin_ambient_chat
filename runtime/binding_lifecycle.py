"""Ambient 绑定的预热、目标接管、撤销和停止清理。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..conversation.messages import ChatMessage
from ..rules.models import Binding, RuleError

if TYPE_CHECKING:
    from ..main import AmbientChatPlugin

MAX_TRACKED_SESSIONS = 500


@dataclass(eq=False)
class _Preheat:
    binding: Binding
    policy: str
    invalid: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str = ""
    load: asyncio.Task | None = None


@dataclass
class _Prepared:
    binding: Binding
    policy: str
    messages: dict[str, list[ChatMessage]]


class BindingLifecycle:
    """调用方无需维护预热和配置提交之间的顺序，也不自行清理目标缓冲。

    预热只读取待发布结果；提交和运行应用成功后才接管目标。停止期间保留旧目标
    所有权，直到自主发言执行完成有限收尾，再清理并允许下一次操作。
    """

    def __init__(self, plugin: AmbientChatPlugin) -> None:
        self._plugin = plugin
        self._store = plugin.store
        self._buffer = plugin.buffer
        self._enabled = plugin.is_enabled()
        self._closed = False
        self._pending: dict[str, _Preheat] = {}
        self._prepared: dict[str, _Prepared] = {}
        self._ready: dict[str, tuple[str, ...]] = {}
        self._owned: dict[str, tuple[str, ...]] = {}
        self._cleanup: set[str] = set()
        self._stopping: set[str] = set()
        self._loads: set[asyncio.Task] = set()
        # 私有协作 seam：在同步提交内校验并观察成功发布，避免 Web 调用方漏通知。
        self._store._binding_guard = self._guard_edit
        self._store._on_commit = self._committed

    def _policy(self, binding: Binding) -> str:
        rule = self._store.get_rule(binding.rule_id)
        content = rule.to_dict() if rule else None
        if content is not None:
            content.pop("name", None)
            content.pop("updated_at", None)
        return json.dumps(content, sort_keys=True, ensure_ascii=False)

    def _guard_edit(self, binding_id: str) -> None:
        if binding_id in self._stopping:
            raise RuleError("绑定正在停止收尾，请等待停止完成后再操作")

    def _invalidate(self, operation: _Preheat, reason: str) -> None:
        if not operation.invalid.is_set():
            operation.reason = reason
            operation.invalid.set()
            if operation.load is not None:
                operation.load.cancel()
        bid = operation.binding.id
        if self._pending.get(bid) is operation:
            self._pending.pop(bid)

    def _committed(self) -> None:
        """观察已发布配置；改回原值也不能复活已在成功提交时撤销的启动。"""
        for operation in list(self._pending.values()):
            current = self._store.get_binding(operation.binding.id)
            if (
                current is not operation.binding
                or self._policy(current) != operation.policy
            ):
                self._invalidate(operation, "绑定或关联规则已变化，启动已取消")
        for bid, prepared in list(self._prepared.items()):
            current = self._store.get_binding(bid)
            if (
                current is not prepared.binding
                or self._policy(current) != prepared.policy
            ):
                self._prepared.pop(bid)
        for bid, targets in list(self._ready.items()):
            current = self._store.get_binding(bid)
            if (
                current is None
                or not current.enabled
                or tuple(current.targets) != targets
            ):
                self._ready.pop(bid)
                self._cleanup.add(bid)

    def _refresh(self) -> None:
        enabled = self._plugin.is_enabled() and not self._closed
        if not enabled:
            for operation in list(self._pending.values()):
                self._invalidate(operation, "自主发言已关闭，启动已取消")
            self._prepared.clear()
            self._cleanup.update(self._owned)
            self._ready.clear()
        self._enabled = enabled
        self._committed()

    def accepts(
        self, umo: str, *, binding_id: str | None = None, for_message: bool = False
    ) -> bool:
        self._refresh()
        if not self._enabled:
            return False
        binding = self._store.binding_for_umo(umo)
        if binding is None or (binding_id is not None and binding.id != binding_id):
            return False
        if binding.id in self._stopping or umo not in self._ready.get(binding.id, ()):
            return False
        return not (
            for_message
            and not self._buffer.has(umo)
            and self._buffer.tracked >= MAX_TRACKED_SESSIONS
        )

    def record(
        self,
        umo: str,
        message: ChatMessage,
        *,
        image_components: Sequence[Any] = (),
        audio_components: Sequence[Any] = (),
    ) -> bool:
        if not self.accepts(umo, for_message=True):
            return False
        self._buffer.append(
            umo,
            message,
            image_components=image_components,
            audio_components=audio_components,
        )
        return True

    def can_schedule(self, binding_id: str) -> bool:
        """允许登记即将就绪的调度；实际执行仍必须穿过 accepts。"""
        self._refresh()
        return (
            self._enabled
            and binding_id not in self._stopping
            and binding_id not in self._cleanup
            and (binding_id in self._ready or binding_id in self._prepared)
        )

    def save_binding(self, raw: Any) -> Binding:
        return self._store.upsert_binding(raw)

    def delete_binding(self, binding_id: str) -> bool:
        return self._store.delete_binding(binding_id)

    def _new(self, binding: Binding) -> _Preheat:
        operation = _Preheat(binding, self._policy(binding))
        self._pending[binding.id] = operation
        return operation

    def _forget_load(self, task: asyncio.Task) -> None:
        self._loads.discard(task)
        if not task.cancelled():
            task.exception()

    async def _read(self, operation: _Preheat) -> dict[str, list[ChatMessage]]:
        binding = operation.binding
        rule = self._store.get_rule(binding.rule_id)
        if rule is None:
            raise RuleError(f"规则不存在：{binding.rule_id}")
        # 先解析机器人昵称：预热的历史里 bot 条目要靠它覆盖核心写入的占位名，
        # 否则第一条消息到来前只能回退「我」。
        await asyncio.gather(
            *(self._plugin.account.name_for(umo) for umo in binding.targets)
        )
        operation.load = asyncio.create_task(
            self._buffer.load_history(
                self._plugin.context, dict.fromkeys(binding.targets, rule.context_turns)
            ),
            name=f"ambient_preheat_{binding.id}",
        )
        self._loads.add(operation.load)
        operation.load.add_done_callback(self._forget_load)
        cancelled = asyncio.create_task(operation.invalid.wait())
        try:
            await asyncio.wait(
                (operation.load, cancelled), return_when=asyncio.FIRST_COMPLETED
            )
            self._refresh()
            if operation.invalid.is_set():
                raise RuleError(operation.reason)
            return operation.load.result()
        finally:
            cancelled.cancel()
            if not operation.load.done():
                operation.load.cancel()

    async def start(self, binding_id: str) -> Binding:
        self._refresh()
        if self._closed:
            raise RuleError("插件已卸载，无法启动")
        self._guard_edit(binding_id)
        binding = self._store.get_binding(binding_id)
        if binding is None:
            raise RuleError(f"绑定不存在：{binding_id}")
        if binding_id in self._pending or binding_id in self._prepared:
            raise RuleError("绑定正在启动，请等待或取消启动")
        if binding.enabled:
            raise RuleError("绑定已在运行中，请先停止后再操作")
        if self._store.get_rule(binding.rule_id) is None:
            raise RuleError(f"规则不存在：{binding.rule_id}")
        operation = self._new(binding)
        try:
            messages = await self._read(operation) if self._enabled else {}
            # 关闭总开关时新点击启动可只保存启用状态；在途预热遇到关闭则已撤销。
            if operation.invalid.is_set():
                raise RuleError(operation.reason)
            self._pending.pop(binding_id, None)
            # 从复查到提交和登记待发布结果不让出控制权，失败提交不会触碰缓冲。
            binding = self._store.set_binding_enabled(binding_id, True)
            if self._enabled:
                self._prepared[binding_id] = _Prepared(
                    binding, operation.policy, messages
                )
            await self._plugin.configuration.apply()
            return binding
        finally:
            if self._pending.get(binding_id) is operation:
                self._pending.pop(binding_id)

    async def stop(self, binding_id: str) -> Binding:
        self._refresh()
        self._guard_edit(binding_id)
        binding = self._store.get_binding(binding_id)
        if binding is None:
            raise RuleError(f"绑定不存在：{binding_id}")
        operation = self._pending.get(binding_id)
        if operation is not None and not binding.enabled:
            self._invalidate(operation, "用户已取消启动")
            return binding
        if not binding.enabled:
            raise RuleError("绑定未在运行")
        self._stopping.add(binding_id)
        try:
            binding = self._store.set_binding_enabled(binding_id, False)
        except BaseException:
            self._stopping.discard(binding_id)
            raise
        self._owned.setdefault(binding_id, tuple(binding.targets))
        self._cleanup.add(binding_id)
        # 应用失败或请求中断时保留停止态的闸门，重新应用成功才释放。
        await self._plugin.configuration.apply()
        return binding

    async def _restore(self, operation: _Preheat) -> None:
        try:
            messages = await self._read(operation)
            binding = operation.binding
            self._prepared[binding.id] = _Prepared(binding, operation.policy, messages)
        except RuleError:
            if not operation.invalid.is_set():
                raise
        finally:
            bid = operation.binding.id
            if self._pending.get(bid) is operation:
                self._pending.pop(bid)

    async def reconcile(self) -> None:
        self._refresh()
        await self._plugin.execution.reconcile()
        self._refresh()
        if self._closed:
            return
        settings = self._store.settings
        self._buffer.set_maxlen(settings.buffer_size)
        self._buffer.set_ignored_senders(settings.buffer_ignored_sender_ids)
        self._buffer.configure_media(
            enabled=settings.cache_images,
            max_edge=settings.image_max_edge,
            quality=settings.image_quality,
            audio_enabled=settings.cache_audio,
        )
        for bid in list(self._cleanup):
            if not self._plugin.execution.is_idle(bid):
                continue
            for umo in self._owned.pop(bid, ()):
                self._buffer.clear(umo)
            self._cleanup.discard(bid)
        restores = []
        if self._enabled:
            for binding in self._store.active_bindings():
                bid = binding.id
                if (
                    bid not in self._ready
                    and bid not in self._pending
                    and bid not in self._prepared
                    and bid not in self._cleanup
                    and bid not in self._stopping
                ):
                    restores.append(self._restore(self._new(binding)))
        if restores:
            # 只等待本次发起的预热，不替另一条正在启动的绑定排队。
            results = await asyncio.gather(*restores, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        if self._closed:
            return
        await self._plugin.scheduler.rebuild()
        # 最后一个 await 之后复查、发布和开放接管，失败应用不会提前放行。
        self._refresh()
        for bid, prepared in list(self._prepared.items()):
            if bid in self._cleanup or bid in self._stopping:
                continue
            self._buffer.publish_history(prepared.messages)
            self._owned[bid] = tuple(prepared.binding.targets)
            self._ready[bid] = tuple(prepared.binding.targets)
            self._prepared.pop(bid)
        for bid in list(self._stopping):
            if bid not in self._cleanup and self._plugin.execution.is_idle(bid):
                self._stopping.discard(bid)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        self._refresh()
        result = {}
        for binding in self._store.bindings.values():
            bid = binding.id
            phase = (
                "stopping"
                if bid in self._stopping or bid in self._cleanup
                else "starting"
                if bid in self._pending or bid in self._prepared
                else "ready"
                if bid in self._ready
                else "paused"
                if binding.enabled and not self._enabled
                else "pending"
                if binding.enabled
                else "stopped"
            )
            result[bid] = {"lifecycle": phase, "ready": phase == "ready"}
        return result

    async def close(self) -> None:
        self._closed = True
        self._refresh()
        await self._plugin.execution.close()
        await self._plugin.scheduler.stop()
        self._buffer.clear_all()
        self._owned.clear()
        self._cleanup.clear()
        self._stopping.clear()
