"""用真实配置提交、历史读取和执行入口验证绑定生命周期的并发保证。"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.media_store import MediaStore
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin, AmbientSessionFilter
from astrbot_plugin_ambient_chat.rules.models import RuleError
from astrbot_plugin_ambient_chat.rules.store import RuleStore
from astrbot_plugin_ambient_chat.runtime.binding_lifecycle import BindingLifecycle
from astrbot_plugin_ambient_chat.runtime.configuration import ConfigurationApplication
from astrbot_plugin_ambient_chat.runtime.execution import SpeechExecution
from astrbot_plugin_ambient_chat.runtime.scheduler import AmbientScheduler
from astrbot_plugin_ambient_chat.speech.blockers import BlockerChain
from astrbot_plugin_ambient_chat.speech.delivery import PreparedSpeech
from astrbot_plugin_ambient_chat.speech.intent import IntentResult

TZ = ZoneInfo("UTC")


def umo(index=0):
    return f"test:GroupMessage:{index}"


def rule(rule_id="r0", **changes):
    return {
        "id": rule_id,
        "name": rule_id,
        "triggers": [{"type": "interval", "every_minutes": 60}],
        "inject_persona": False,
        **changes,
    }


def binding(binding_id="b0", index=0):
    return {
        "id": binding_id,
        "name": binding_id,
        "platform_id": "test",
        "rule_id": "r0",
        "targets": [umo(index)],
    }


class History:
    def __init__(self):
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.calls = 0
        self.paused = True
        self.fail = False

    async def get(self, **kwargs):
        self.calls += 1
        paused = self.paused
        if paused:
            self.waiting.set()
            # 故意模拟不服从取消的外部读取：生命周期不能依赖 task.cancel 保证安全。
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    pass
        if self.fail:
            raise OSError("历史暂不可读")
        result = [
            SimpleNamespace(
                content={
                    "type": "user",
                    "message": [{"type": "text", "text": "旧历史"}],
                },
                sender_name="小明",
                created_at=datetime(2026, 11, 5, tzinfo=TZ),
            )
        ]
        if paused:
            self.finished.set()
        return result


class Runtime:
    def __init__(self, path):
        self.store = RuleStore(path)
        self.store.upsert_rule(rule(), TZ)
        self.store.upsert_binding(binding())
        self.enabled = True
        self.timezone = TZ
        self.history = History()
        self.context = SimpleNamespace(message_history_manager=self.history)
        self.buffer = ContextBuffer(media=MediaStore(path / "media"))
        self.blockers = BlockerChain(self.buffer)
        self.router = SimpleNamespace(reload=AsyncMock())
        self.judge = SimpleNamespace(
            judge=AsyncMock(return_value=IntentResult(True)),
            persona_prompt=AsyncMock(return_value=""),
        )
        self.delivery = SimpleNamespace(
            prepare=AsyncMock(return_value=PreparedSpeech(umo(), "cid", "新发言")),
            send=AsyncMock(return_value=True),
            record=self.record,
        )
        self.lifecycle = BindingLifecycle(self)
        self.execution = SpeechExecution(
            self,
            accepts=lambda bid, target: self.lifecycle.accepts(target, binding_id=bid),
        )
        self.scheduler = AmbientScheduler(self)
        self.configuration = ConfigurationApplication(
            self.store, self.lifecycle.reconcile
        )
        # record_bot_message 依赖的配置与账号名服务：测试里固定回退「我」。
        self.config = {}
        self.account = SimpleNamespace(
            cached=lambda _umo: "", name_for=AsyncMock(return_value="")
        )
        self._self_name = lambda _umo: "我"

    @property
    def settings(self):
        return self.store.settings

    def is_enabled(self):
        return self.enabled

    async def record(self, speech):
        self.buffer.append(speech.umo, ChatMessage("我", speech.text, 1, is_bot=True))
        return True

    async def begin(self):
        task = asyncio.create_task(self.lifecycle.start("b0"))
        async with asyncio.timeout(2):
            await self.history.waiting.wait()
        return task


@pytest_asyncio.fixture
async def runtime(tmp_path):
    result = Runtime(tmp_path)
    yield result
    result.history.release.set()
    await result.lifecycle.close()
    await result.buffer.close()
    # 允许刻意忽略取消的历史替身收尾，避免把它遗留给下一测试。
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stop_invalidates_image_download_without_waiting_for_it(
    runtime, tmp_path
):
    h = runtime
    h.history.paused = False
    h.store.update_settings({"cache_images": True})
    await h.lifecycle.start("b0")
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    source = tmp_path / "source.img"
    source.write_bytes(b"image")

    async def download():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()  # 外部下载即使迟到也不能污染下一次启动。
        finally:
            finished.set()
        return str(source)

    entry = ChatMessage("小明", "看图", 1, attachments=("image",))
    assert h.lifecycle.record(
        umo(), entry, image_components=[SimpleNamespace(convert_to_file_path=download)]
    )
    await entered.wait()
    try:
        async with asyncio.timeout(1):
            await h.lifecycle.stop("b0")
        assert not finished.is_set()
        assert not h.buffer.has(umo())
        await h.lifecycle.start("b0")
    finally:
        release.set()
        await finished.wait()
    await h.buffer.close()
    assert not entry.images
    assert not h.buffer.media_root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["edit", "delete", "rule", "revert", "stop", "global", "close"]
)
async def test_invalidated_preheat_cannot_enable_or_publish_late_result(
    runtime, change
):
    h = runtime
    start = await h.begin()
    assert h.lifecycle.snapshot()["b0"]["lifecycle"] == "starting"
    if change == "edit":
        h.lifecycle.save_binding(binding(index=1))
    elif change == "delete":
        h.lifecycle.delete_binding("b0")
    elif change in {"rule", "revert"}:
        h.store.upsert_rule(rule(context_turns=50), TZ)
        if change == "revert":
            h.store.upsert_rule(rule(), TZ)
    elif change == "stop":
        assert not (await h.lifecycle.stop("b0")).enabled
    elif change == "global":
        h.enabled = False
        await h.configuration.apply()
    else:
        await h.lifecycle.close()
    async with asyncio.timeout(1):
        with pytest.raises(RuleError, match="取消"):
            await start
    h.history.release.set()
    await h.history.finished.wait()
    assert h.buffer.tracked == 0
    current = h.store.get_binding("b0")
    assert current is None or not current.enabled
    assert not h.lifecycle.accepts(umo())
    assert not h.scheduler.status() or not any(
        x["running"] for x in h.scheduler.status()
    )


@pytest.mark.asyncio
async def test_old_preheat_cannot_overwrite_target_transferred_to_another_binding(
    runtime,
):
    h = runtime
    old = await h.begin()
    h.lifecycle.save_binding(binding(index=1))
    h.lifecycle.save_binding(binding("b1"))
    h.history.paused = False
    await h.lifecycle.start("b1")
    h.lifecycle.record(umo(), ChatMessage("小红", "新的接管", 2))
    before = h.buffer.recent(umo(), 100)
    h.history.release.set()
    with pytest.raises(RuleError):
        await old
    await h.history.finished.wait()
    assert h.buffer.recent(umo(), 100) == before
    assert h.lifecycle.accepts(umo(), binding_id="b1")
    assert not h.lifecycle.accepts(umo(), binding_id="b0")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["name", "unrelated", "interval", "failed_commit"])
async def test_non_effective_or_failed_changes_do_not_cancel_start(
    runtime, monkeypatch, change
):
    h = runtime
    task = await h.begin()
    if change == "name":
        h.store.upsert_rule(rule(name="只改展示名"), TZ)
    elif change == "unrelated":
        h.store.upsert_rule(rule("other"), TZ)
    elif change == "interval":
        h.store.update_settings({"send_interval_seconds": 2})
    else:

        def fail(*args):
            raise OSError("写入失败")

        with monkeypatch.context() as patch:
            patch.setattr(
                "astrbot_plugin_ambient_chat.rules.store.os.replace",
                fail,
            )
            with pytest.raises(OSError):
                h.lifecycle.save_binding(binding(index=1))
    h.history.release.set()
    assert (await task).enabled
    assert h.lifecycle.accepts(umo())
    assert h.buffer.recent(umo(), 10)[0].text == "旧历史"


@pytest.mark.asyncio
async def test_duplicate_start_rejected_and_cancel_does_not_wait_for_uncooperative_history(
    runtime,
):
    h = runtime
    old = await h.begin()
    with pytest.raises(RuleError, match="正在启动"):
        await h.lifecycle.start("b0")
    async with asyncio.timeout(1):
        await h.lifecycle.stop("b0")
        with pytest.raises(RuleError, match="取消"):
            await old
    # 同绑定可立即发起新启动，旧读取仍未结束。
    h.history.paused = False
    assert (await h.lifecycle.start("b0")).enabled
    h.history.release.set()
    await h.history.finished.wait()
    assert h.lifecycle.accepts(umo())


@pytest.mark.asyncio
async def test_stopping_guards_binding_but_does_not_block_unrelated_binding(runtime):
    h = runtime
    h.history.paused = False
    await h.lifecycle.start("b0")
    sending, release = asyncio.Event(), asyncio.Event()

    async def send(speech):
        sending.set()
        await release.wait()
        return True

    h.delivery.send.side_effect = send
    execution = asyncio.create_task(h.scheduler.trigger_target("b0", umo()))
    await sending.wait()
    stop = asyncio.create_task(h.lifecycle.stop("b0"))
    await asyncio.sleep(0)
    assert h.lifecycle.snapshot()["b0"]["lifecycle"] == "stopping"
    with pytest.raises(RuleError, match="停止收尾"):
        await h.lifecycle.start("b0")
    with pytest.raises(RuleError, match="停止收尾"):
        h.store.upsert_binding(binding(index=1))
    with pytest.raises(RuleError, match="停止收尾"):
        h.store.delete_binding("b0")
    h.lifecycle.save_binding(binding("b1", 1))
    async with asyncio.timeout(1):
        await h.lifecycle.start("b1")
    assert not stop.done()
    release.set()
    await stop
    assert (await execution)["spoke"]
    assert not h.buffer.has(umo())
    assert h.lifecycle.snapshot()["b0"]["lifecycle"] == "stopped"
    assert h.lifecycle.accepts(umo(1))


@pytest.mark.asyncio
async def test_failed_enable_does_not_publish_preheat(runtime, monkeypatch):
    h = runtime
    task = await h.begin()

    def fail(*args):
        raise OSError("启用写入失败")

    with monkeypatch.context() as patch:
        patch.setattr("astrbot_plugin_ambient_chat.rules.store.os.replace", fail)
        h.history.release.set()
        with pytest.raises(OSError):
            await task
    assert not h.store.get_binding("b0").enabled
    assert h.buffer.tracked == 0


@pytest.mark.asyncio
async def test_saved_enable_and_stop_apply_failures_keep_gate_until_retry(runtime):
    h = runtime
    h.history.paused = False
    rebuild = h.scheduler.rebuild
    h.scheduler.rebuild = AsyncMock(side_effect=RuntimeError("调度应用失败"))
    assert (await h.lifecycle.start("b0")).enabled
    assert not h.lifecycle.accepts(umo())
    assert h.configuration.snapshot()["state"] == "failed"
    revision = h.store.revision
    h.scheduler.rebuild = rebuild
    assert (await h.configuration.apply())["state"] == "applied"
    assert h.store.revision == revision
    assert h.lifecycle.accepts(umo())
    h.scheduler.rebuild = AsyncMock(side_effect=RuntimeError("调度应用失败"))
    await h.lifecycle.stop("b0")
    with pytest.raises(RuleError, match="停止收尾"):
        h.lifecycle.save_binding(binding(index=1))
    assert not h.lifecycle.accepts(umo())
    h.scheduler.rebuild = rebuild
    await h.configuration.apply()
    h.lifecycle.save_binding(binding(index=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["initialize", "reopen"])
async def test_bulk_preheat_blocks_messages_and_both_execution_entries(
    runtime, monkeypatch, mode
):
    h = runtime
    h.store.set_binding_enabled("b0", True)
    if mode == "reopen":
        h.enabled = False
        await h.configuration.apply()
        h.enabled = True
    apply = asyncio.create_task(h.configuration.apply())
    await h.history.waiting.wait()
    monkeypatch.setattr("astrbot_plugin_ambient_chat.main._plugin_ref", h)
    event = SimpleNamespace(unified_msg_origin=umo())
    assert not AmbientSessionFilter().filter(event, {})
    await AmbientChatPlugin.record_bot_message(h, event)
    assert not h.lifecycle.record(umo(), ChatMessage("小明", "预热期间", 2))
    assert not (await h.scheduler.trigger_target("b0", umo()))["spoke"]
    await h.scheduler._fire(h.store.get_binding("b0"))
    h.judge.judge.assert_not_awaited()
    assert h.buffer.tracked == 0
    h.history.release.set()
    await apply
    assert AmbientSessionFilter().filter(event, {})
    assert (await h.scheduler.trigger_target("b0", umo()))["spoke"]


@pytest.mark.asyncio
async def test_global_off_start_only_saves_enabled_and_reopen_preheats(runtime):
    h = runtime
    h.enabled = False
    assert (await h.lifecycle.start("b0")).enabled
    assert h.history.calls == 0
    assert not h.lifecycle.accepts(umo())
    h.history.paused = False
    h.enabled = True
    await h.configuration.apply()
    assert h.history.calls == 1
    assert h.lifecycle.accepts(umo())
    h.enabled = False
    await h.configuration.apply()
    assert h.store.get_binding("b0").enabled
    assert h.buffer.tracked == 0


@pytest.mark.asyncio
async def test_normal_bot_reply_uses_current_ready_ownership_and_session_cap(
    runtime, monkeypatch
):
    h = runtime
    result = SimpleNamespace(get_plain_text=lambda **kw: "普通回复")
    event = SimpleNamespace(unified_msg_origin=umo(), get_result=lambda: result)
    await AmbientChatPlugin.record_bot_message(h, event)
    assert h.buffer.tracked == 0
    h.history.paused = False
    await h.lifecycle.start("b0")
    await AmbientChatPlugin.record_bot_message(h, event)
    assert h.buffer.recent(umo(), 1)[0].text == "普通回复"
    await h.lifecycle.stop("b0")
    await AmbientChatPlugin.record_bot_message(h, event)
    assert h.buffer.tracked == 0
    await h.lifecycle.start("b0")
    # 回复实际发送时该目标已重新就绪，按当前归属记录。
    await AmbientChatPlugin.record_bot_message(h, event)
    assert h.buffer.recent(umo(), 1)[0].text == "普通回复"
    h.buffer.clear(umo())
    monkeypatch.setattr(
        "astrbot_plugin_ambient_chat.runtime.binding_lifecycle.MAX_TRACKED_SESSIONS", 0
    )
    await AmbientChatPlugin.record_bot_message(h, event)
    assert h.buffer.tracked == 0
    event.unified_msg_origin = umo(99)
    await AmbientChatPlugin.record_bot_message(h, event)
    assert h.buffer.tracked == 0


@pytest.mark.asyncio
async def test_missing_history_is_not_a_start_failure(runtime):
    h = runtime
    h.history.paused = False
    h.history.fail = True
    assert (await h.lifecycle.start("b0")).enabled
    assert h.lifecycle.accepts(umo())
    assert h.buffer.tracked == 0


@pytest.mark.asyncio
async def test_request_cancel_before_enable_cannot_publish_late_history(runtime):
    h = runtime
    task = await h.begin()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    h.history.release.set()
    await h.history.finished.wait()
    assert not h.store.get_binding("b0").enabled
    assert h.buffer.tracked == 0
    assert h.lifecycle.snapshot()["b0"]["lifecycle"] == "stopped"


@pytest.mark.asyncio
async def test_request_cancel_after_commit_keeps_saved_enable_closed_until_reapply(
    runtime,
):
    h = runtime
    h.history.paused = False
    entered = asyncio.Event()

    async def rebuild():
        entered.set()
        await asyncio.Event().wait()

    original = h.scheduler.rebuild
    h.scheduler.rebuild = rebuild
    task = asyncio.create_task(h.lifecycle.start("b0"))
    await entered.wait()
    assert h.store.get_binding("b0").enabled
    assert h.buffer.tracked == 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.configuration.snapshot()["state"] == "failed"
    assert not h.lifecycle.accepts(umo())
    h.scheduler.rebuild = original
    revision = h.store.revision
    await h.configuration.apply()
    assert h.store.revision == revision
    assert h.lifecycle.accepts(umo())


@pytest.mark.asyncio
async def test_late_initialization_cannot_restart_scheduler_after_unload(runtime):
    h = runtime
    h.store.set_binding_enabled("b0", True)
    apply = asyncio.create_task(h.configuration.apply())
    await h.history.waiting.wait()
    await h.lifecycle.close()
    async with asyncio.timeout(1):
        await apply
    h.history.release.set()
    await h.history.finished.wait()
    await h.configuration.apply()
    assert h.buffer.tracked == 0
    assert not h.lifecycle.accepts(umo())
    assert not any(item["running"] for item in h.scheduler.status())
