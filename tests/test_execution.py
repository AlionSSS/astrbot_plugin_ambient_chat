"""通过执行 interface 验证时序；替身不会调用真实模型或发送消息。"""

import asyncio
import copy
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin
from astrbot_plugin_ambient_chat.rules.models import (
    Binding,
    DecisionModel,
    Rule,
    Trigger,
)
from astrbot_plugin_ambient_chat.rules.store import RuleStore
from astrbot_plugin_ambient_chat.runtime.attention import AttentionStore
from astrbot_plugin_ambient_chat.runtime.binding_lifecycle import BindingLifecycle
from astrbot_plugin_ambient_chat.runtime.configuration import ConfigurationApplication
from astrbot_plugin_ambient_chat.runtime.execution import SpeechExecution
from astrbot_plugin_ambient_chat.runtime.scheduler import AmbientScheduler
from astrbot_plugin_ambient_chat.speech.blockers import BlockDecision, BlockerChain
from astrbot_plugin_ambient_chat.speech.decision import DecisionRouter
from astrbot_plugin_ambient_chat.speech.delivery import (
    Delivery,
    PreparedSpeech,
    SendOutcome,
)
from astrbot_plugin_ambient_chat.speech.intent import IntentResult

_ATTENTION_DIR: Path | None = None
"""当前用例专属的关注度目录：由 autouse fixture 指向 pytest 的 ``tmp_path``。"""


@pytest.fixture(autouse=True)
def attention_data_dir(tmp_path: Path, monkeypatch) -> None:
    """每个用例一个独立的关注度目录，清理由 pytest 负责，不依赖对象回收。"""
    monkeypatch.setattr(sys.modules[__name__], "_ATTENTION_DIR", tmp_path / "attention")


class Harness:
    def __init__(self, *, send_timeout=60, record_timeout=10):
        self.store = RuleStore()
        self.settings = self.store.settings
        self.settings.concurrency = 2
        self.settings.send_interval_seconds = 0
        self.enabled = True
        self.timezone = ZoneInfo("UTC")
        self.buffer = ContextBuffer()
        self.blockers = BlockerChain(self.buffer)
        # 关注度档案写在用例自己的临时目录并关掉延迟写：替身不该碰真实数据目录。
        assert _ATTENTION_DIR is not None, "attention_data_dir fixture 未生效"
        self.attention = AttentionStore(_ATTENTION_DIR, save_delay=3600)
        self.account = SimpleNamespace(name_for=AsyncMock(return_value=""))
        self.router = SimpleNamespace(reload=AsyncMock())
        self.gates = {}
        self.entered = {}
        self.judged = []
        self.sent = []
        self.recorded = []
        self.send_ok = True
        self.record_ok = True
        self.judge = SimpleNamespace(
            judge=self.judgment, persona_prompt=AsyncMock(return_value="")
        )
        self.delivery = SimpleNamespace(
            prepare=self.prepare, send=self.send, record=self.record
        )
        for i in range(3):
            self.store.rules[f"r{i}"] = Rule(
                f"r{i}",
                f"规则{i}",
                triggers=[Trigger(type="interval", every_minutes=90)],
                inject_persona=False,
            )
            self.store.bindings[f"b{i}"] = Binding(
                f"b{i}",
                "test",
                f"r{i}",
                targets=[self.umo(i)],
            )
        self.execution = SpeechExecution(
            self, send_timeout=send_timeout, record_timeout=record_timeout
        )
        self.scheduler = AmbientScheduler(self)

    @staticmethod
    def umo(i):
        return f"test:GroupMessage:{i}"

    def is_enabled(self):
        return self.enabled

    def pause(self, phase, i=0):
        self.gates[phase, self.umo(i)] = asyncio.Event()

    def release(self, phase, i=0):
        self.gates[phase, self.umo(i)].set()

    async def wait(self, phase, i=0):
        async with asyncio.timeout(2):
            await self.entered.setdefault((phase, self.umo(i)), asyncio.Event()).wait()

    async def stage(self, phase, umo):
        self.entered.setdefault((phase, umo), asyncio.Event()).set()
        if (phase, umo) in self.gates:
            await self.gates[phase, umo].wait()

    async def judgment(self, rule, umo, buffer, now):
        self.judged.append((umo, now))
        await self.stage("judging", umo)
        return IntentResult(True, reason="直接放行")

    async def prepare(self, rule, umo, hint, persona, now):
        await self.stage("generating", umo)
        return PreparedSpeech(umo, "conversation", "hello")

    async def send(self, speech, *, should_continue=None):
        self.sent.append((speech.umo, asyncio.get_running_loop().time()))
        await self.stage("sending", speech.umo)
        if not self.send_ok:
            return SendOutcome(0, 1, "")
        return SendOutcome(1, 1, speech.text)

    async def record(self, speech):
        self.buffer.append(speech.umo, ChatMessage("我", speech.text, 0, is_bot=True))
        await self.stage("recording", speech.umo)
        self.recorded.append(speech.umo)
        return self.record_ok

    async def begin(self, i=0):
        task = asyncio.create_task(self.execution.execute(f"b{i}", self.umo(i)))
        await asyncio.sleep(0)
        return task

    async def stop(self, i=0):
        self.store.bindings[f"b{i}"].enabled = False
        await self.execution.reconcile()

    async def use_configuration(self, path):
        self.store._data_dir = path
        self.context = SimpleNamespace()
        self.lifecycle = BindingLifecycle(self)
        self.configuration = ConfigurationApplication(
            self.store, self.lifecycle.reconcile
        )
        await self.lifecycle.reconcile()


async def wait_phase(h, phase, i=0):
    async with asyncio.timeout(2):
        while h.execution.snapshot().get((f"b{i}", h.umo(i)), {}).get("phase") != phase:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_busy_skip_including_wait_for_slot_and_slot_covers_recording():
    h = Harness()
    h.settings.concurrency = 1
    h.pause("recording")
    await h.execution.reconcile()
    first = await h.begin()
    await h.wait("recording")
    second = await h.begin(1)
    await wait_phase(h, "queued", 1)
    for i in (0, 1):
        duplicate = await h.execution.execute(f"b{i}", h.umo(i))
        assert duplicate.state == "busy"
    assert len(h.judged) == 1
    assert h.execution.snapshot()["b0", h.umo(0)]["phase"] == "recording"
    h.release("recording")
    assert (await first).ok and (await second).ok
    assert h.execution.snapshot()["b0", h.umo(0)]["state"] == "sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["queued", "judging", "generating", "pacing"])
async def test_stop_cancels_each_pre_send_phase_and_preserves_other_binding(phase):
    h = Harness()
    if phase == "queued":
        h.settings.concurrency = 1
        h.pause("generating", 1)
    elif phase == "pacing":
        h.settings.send_interval_seconds = 3600
    else:
        h.pause(phase)
    await h.execution.reconcile()
    other = await h.begin(1)
    if phase == "queued":
        await h.wait("generating", 1)
    elif phase == "pacing":
        await other
    first = await h.begin()
    await wait_phase(h, phase)
    await h.stop()
    assert (await first).state == "cancelled"
    assert not any(umo == h.umo(0) for umo, _ in h.sent)
    if phase == "queued":
        h.release("generating", 1)
    assert (await other).ok


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["sending", "recording"])
async def test_real_stop_binding_waits_for_finishing_then_clears_buffer(
    phase, tmp_path
):
    h = Harness()
    await h.use_configuration(tmp_path)
    h.pause(phase)
    await h.execution.reconcile()
    task = await h.begin()
    await h.wait(phase)
    stop = asyncio.create_task(AmbientChatPlugin.stop_binding(h, "b0"))
    await asyncio.sleep(0)
    assert not stop.done()
    assert (await h.execution.execute("b0", h.umo(0))).state == "busy"
    h.release(phase)
    await stop
    assert (await task).ok
    assert h.recorded == [h.umo(0)]
    assert not h.buffer.has(h.umo(0))
    assert (await h.execution.execute("b0", h.umo(0))).state == "cancelled"


@pytest.mark.asyncio
async def test_actual_sends_spaced_after_parallel_generation_and_failed_attempt():
    h = Harness()
    h.settings.send_interval_seconds = 0.025
    h.pause("generating")
    h.send_ok = False
    await h.execution.reconcile()
    first = await h.begin()
    await h.wait("generating")
    assert (await h.execution.execute("b1", h.umo(1))).state == "skipped"
    h.send_ok = True
    h.release("generating")
    assert (await first).ok
    assert [umo for umo, _ in h.sent] == [h.umo(1), h.umo(0)]
    assert h.sent[1][1] - h.sent[0][1] >= 0.024


@pytest.mark.asyncio
async def test_segment_abort_reports_partial_and_records_only_sent_text():
    h = Harness()
    seen = {}

    async def partial_send(speech, *, should_continue=None):
        h.sent.append((speech.umo, asyncio.get_running_loop().time()))
        seen["before"] = should_continue()
        # 段间发现绑定已停：后面的片段绝不能再发。
        h.store.bindings["b0"].enabled = False
        seen["after"] = should_continue()
        return SendOutcome(1, 2, "第一段。", "cancelled")

    h.delivery.send = partial_send
    await h.execution.reconcile()
    result = await h.execution.execute("b0", h.umo(0))
    assert seen == {"before": True, "after": False}
    assert result.state == "partial" and result.ok
    assert "1/2" in result.reason
    assert h.recorded == [h.umo(0)]
    assert h.buffer.has(h.umo(0))


@pytest.mark.asyncio
async def test_lower_concurrency_preserves_existing_work_and_next_waits():
    h = Harness()
    h.pause("generating", 0)
    h.pause("generating", 1)
    await h.execution.reconcile()
    first, second = await h.begin(), await h.begin(1)
    await h.wait("generating", 0)
    await h.wait("generating", 1)
    h.settings.concurrency = 1
    await h.execution.reconcile()
    third = await h.begin(2)
    await wait_phase(h, "queued", 2)
    h.release("generating", 0)
    assert (await first).ok
    assert len(h.judged) == 2
    h.release("generating", 1)
    assert (await second).ok and (await third).ok


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage,expected", [("sending", "unknown"), ("recording", "history_failed")]
)
async def test_bounded_finishing_releases_slot_and_does_not_retry(stage, expected):
    h = Harness(send_timeout=0.01, record_timeout=0.01)
    h.settings.concurrency = 1
    h.pause(stage)
    await h.execution.reconcile()
    first = await h.begin()
    await h.wait(stage)
    second = await h.begin(1)
    result = await first
    assert result.state == expected
    assert result.ok == (expected == "history_failed")
    assert (await second).ok
    assert len([umo for umo, _ in h.sent if umo == h.umo(0)]) == 1
    assert h.umo(0) not in h.recorded
    assert h.buffer.has(h.umo(0)) == (expected == "history_failed")


@pytest.mark.asyncio
async def test_send_timeout_bounds_wait_even_if_adapter_suppresses_cancellation():
    h = Harness(send_timeout=0.01)
    finished = asyncio.Event()

    async def stubborn_send(speech, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await finished.wait()
        return SendOutcome(1, 1, speech.text)

    h.delivery.send = stubborn_send
    await h.execution.reconcile()
    async with asyncio.timeout(1):
        result = await h.execution.execute("b0", h.umo(0))
        await h.execution.close()
    assert result.state == "unknown"
    assert not h.recorded
    finished.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not h.recorded


@pytest.mark.asyncio
async def test_new_interval_wakes_waiting_send_without_repeating_decision():
    h = Harness()
    h.settings.send_interval_seconds = 3600
    await h.execution.reconcile()
    await h.execution.execute("b1", h.umo(1))
    task = await h.begin()
    await wait_phase(h, "pacing")
    h.settings.send_interval_seconds = 0
    await h.execution.reconcile()
    async with asyncio.timeout(1):
        assert (await task).ok
    assert len(h.judged) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", ["queued", "pacing"])
async def test_blockers_use_fresh_buffer_after_waiting(waiting):
    h = Harness()
    h.store.rules["r0"].blockers = [{"type": "cooldown", "minutes": 10}]
    if waiting == "queued":
        h.settings.concurrency = 1
        h.pause("generating", 1)
    else:
        h.settings.send_interval_seconds = 3600
    await h.execution.reconcile()
    other = await h.begin(1)
    if waiting == "queued":
        await h.wait("generating", 1)
    else:
        await other
    task = await h.begin()
    await wait_phase(h, waiting)
    h.buffer.append(
        h.umo(0), ChatMessage("我", "普通回复", datetime.now().timestamp(), is_bot=True)
    )
    if waiting == "queued":
        h.release("generating", 1)
        await other
    else:
        h.settings.send_interval_seconds = 0
        await h.execution.reconcile()
    result = await task
    assert result.state == "skipped" and "冷却" in result.reason
    assert len([umo for umo, _ in h.judged if umo == h.umo(0)]) == (waiting == "pacing")
    assert not any(umo == h.umo(0) for umo, _ in h.sent)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["rule", "model"])
async def test_config_change_cancels_only_affected_execution(change):
    h = Harness()
    h.settings.decision_models = [
        DecisionModel("dm", "model", "https://example.invalid", "v1")
    ]
    h.store.rules["r0"].intent.decision_provider_id = "custom:dm"
    h.pause("judging")
    h.pause("generating", 1)
    await h.execution.reconcile()
    first, other = await h.begin(), await h.begin(1)
    await h.wait("judging")
    await h.wait("generating", 1)
    if change == "rule":
        h.store.rules["r0"].context_turns += 1
    else:
        h.settings.decision_models[0].model = "v2"
    await h.execution.reconcile()
    assert (await first).state == "cancelled"
    assert not other.done()
    h.release("generating", 1)
    assert (await other).ok


@pytest.mark.asyncio
async def test_presentation_changes_do_not_cancel_execution():
    h = Harness()
    h.pause("generating")
    await h.execution.reconcile()
    task = await h.begin()
    await h.wait("generating")
    h.store.rules["r0"].updated_at = "new"
    h.store.rules["r0"].name = "renamed"
    h.store.bindings["b0"].name = "renamed"
    await h.execution.reconcile()
    assert not task.done()
    h.release("generating")
    assert (await task).ok


@pytest.mark.asyncio
async def test_scheduler_keeps_unrelated_task_once_consumption_and_interval_anchor():
    h = Harness()
    await h.execution.reconcile()
    await h.scheduler.rebuild()
    await asyncio.sleep(0)
    original = dict(h.scheduler._tasks)
    anchor = copy.copy(h.scheduler._interval_anchor)
    h.scheduler._consumed_once.add(("b1", 1))
    h.store.bindings["b0"].enabled = False
    await h.execution.reconcile()
    await h.scheduler.rebuild()
    assert h.scheduler._tasks["b1"] is original["b1"]
    assert not original["b1"].cancelled()
    assert ("b1", 1) in h.scheduler._consumed_once
    assert h.scheduler._interval_anchor == anchor
    h.store.rules["r1"].context_turns += 1
    await h.execution.reconcile()
    await h.scheduler.rebuild()
    assert h.scheduler._tasks["b1"] is original["b1"]
    await h.scheduler.stop()


@pytest.mark.asyncio
async def test_forced_trigger_busy_does_not_overwrite_result_or_merge_baseline():
    h = Harness()
    h.pause("generating")
    await h.execution.reconcile()
    first = asyncio.create_task(h.scheduler.trigger_target("b0", h.umo(0)))
    await h.wait("generating")
    admitted_at = h.scheduler._binding_fired_at["b0"]
    busy = await h.scheduler.trigger_target("b0", h.umo(0))
    assert busy["state"] == "busy" and not busy["spoke"]
    assert h.scheduler._binding_fired_at["b0"] == admitted_at
    h.release("generating")
    assert (await first)["state"] == "sent"
    assert h.scheduler._binding_fired_at["b0"] == admitted_at
    target = h.scheduler.status()[0]["targets"][0]
    assert target["state"] == "sent" and target["last_fire_at"]


@pytest.mark.asyncio
async def test_scheduled_merge_baseline_is_first_admission_not_completion(monkeypatch):
    from datetime import timedelta

    from astrbot_plugin_ambient_chat.runtime import scheduler

    now = datetime(2026, 11, 6, 12, tzinfo=ZoneInfo("UTC"))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz)

    monkeypatch.setattr(scheduler, "datetime", Clock)
    h = Harness()
    h.store.bindings["b0"].targets.append(h.umo(1))
    h.pause("generating")
    await h.execution.reconcile()
    task = asyncio.create_task(h.scheduler._fire(h.store.bindings["b0"]))
    await h.wait("generating")
    admitted_at = now
    assert h.scheduler._binding_fired_at["b0"] == admitted_at
    now += timedelta(seconds=5)
    h.release("generating")
    await task
    assert h.scheduler._binding_fired_at["b0"] == admitted_at
    assert h.scheduler._should_merge(h.store.bindings["b0"], now)
    now = admitted_at + timedelta(seconds=60)
    assert not h.scheduler._should_merge(h.store.bindings["b0"], now)


@pytest.mark.asyncio
async def test_queued_admission_updates_merge_baseline_before_slot_available():
    h = Harness()
    h.settings.concurrency = 1
    h.pause("generating")
    await h.execution.reconcile()
    first = await h.begin()
    await h.wait("generating")
    queued = asyncio.create_task(h.scheduler.trigger_target("b1", h.umo(1)))
    await wait_phase(h, "queued", 1)
    admitted_at = h.scheduler._binding_fired_at["b1"]
    h.release("generating")
    await first
    await queued
    assert h.scheduler._binding_fired_at["b1"] == admitted_at


@pytest.mark.asyncio
async def test_router_reloads_only_changed_clients():
    h = Harness()
    h.settings.decision_models = [
        DecisionModel("dm0", "first", "https://example.invalid", "v1"),
        DecisionModel("dm1", "second", "https://example.invalid", "v1"),
    ]
    router = DecisionRouter(None, h.store)
    clients = [SimpleNamespace(close=AsyncMock()) for _ in range(2)]
    for model, client in zip(h.settings.decision_models, clients):
        router._clients[model.id] = (
            (model.base_url, model.api_key, model.model),
            client,
        )
    h.settings.decision_models[0].model = "v2"
    await router.reload()
    clients[0].close.assert_awaited_once()
    clients[1].close.assert_not_awaited()
    assert list(router._clients) == ["dm1"]
    await router.close()
    clients[1].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_history_retry_is_reported_as_failure_when_both_attempts_fail():
    manager = SimpleNamespace(
        get_conversation=AsyncMock(side_effect=RuntimeError("offline"))
    )
    delivery = Delivery(SimpleNamespace(conversation_manager=manager), ContextBuffer())
    assert (
        await delivery._record_in_conversation("test:GroupMessage:0", "cid", "hello")
        is False
    )
    assert manager.get_conversation.await_count == 2


@pytest.mark.asyncio
async def test_close_cancels_queued_work_and_waits_for_send_already_started():
    h = Harness()
    h.settings.concurrency = 1
    h.pause("sending")
    await h.execution.reconcile()
    sending = await h.begin()
    await h.wait("sending")
    queued = await h.begin(1)
    await wait_phase(h, "queued", 1)
    close = asyncio.create_task(h.execution.close())
    assert (await queued).state == "cancelled"
    assert not close.done()
    h.release("sending")
    await close
    assert (await sending).ok
    assert len(h.sent) == 1
    assert (await h.execution.execute("b2", h.umo(2))).state == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["generating", "sending"])
async def test_caller_cancellation_obeys_execution_finishing(phase):
    h = Harness()
    h.pause(phase)
    await h.execution.reconcile()
    caller = await h.begin()
    await h.wait(phase)
    caller.cancel()
    await asyncio.sleep(0)
    if phase == "sending":
        assert not caller.done()
        h.release(phase)
    with pytest.raises(asyncio.CancelledError):
        await caller
    result = h.execution.snapshot()["b0", h.umo(0)]
    assert result["state"] == ("sent" if phase == "sending" else "cancelled")


@pytest.mark.asyncio
async def test_rule_edit_during_send_finishes_then_uses_new_policy():
    h = Harness()
    h.pause("sending")
    await h.execution.reconcile()
    task = await h.begin()
    await h.wait("sending")
    h.store.rules["r0"].context_turns += 1
    sync = asyncio.create_task(h.execution.reconcile())
    await asyncio.sleep(0)
    assert not sync.done()
    h.release("sending")
    await sync
    assert (await task).ok
    assert (await h.execution.execute("b0", h.umo(0))).ok


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,state", [("sending", "unknown"), ("recording", "history_failed")]
)
async def test_forced_result_and_status_preserve_unknown_and_record_failure(
    phase, state
):
    h = Harness(send_timeout=0.01, record_timeout=0.01)
    h.pause(phase)
    await h.execution.reconcile()
    result = await h.scheduler.trigger_target("b0", h.umo(0))
    assert result["state"] == state
    assert result["spoke"] == (state == "history_failed")
    status = h.scheduler.status()[0]
    assert status["targets"][0]["state"] == state
    assert status["targets"][0]["last_result"] == result["reason"]
    assert result["reason"] in status["last_result"]


@pytest.mark.asyncio
async def test_delivery_stages_keep_generation_separate_from_send_and_history():
    manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="cid"),
        get_conversation=AsyncMock(
            side_effect=[
                SimpleNamespace(history="[]"),
                SimpleNamespace(history='[{"role":"assistant","content":"hello"}]'),
            ]
        ),
        update_conversation=AsyncMock(),
    )
    context = SimpleNamespace(
        conversation_manager=manager,
        get_current_chat_provider_id=AsyncMock(return_value="chat"),
        llm_generate=AsyncMock(return_value=SimpleNamespace(completion_text="hello")),
        send_message=AsyncMock(return_value=True),
    )
    buffer = ContextBuffer()
    delivery = Delivery(context, buffer)
    rule = Rule("r0", "规则")
    speech = await delivery.prepare(
        rule, "test:GroupMessage:0", "hint", now=datetime.now(ZoneInfo("UTC"))
    )
    assert speech.text == "hello"
    context.send_message.assert_not_awaited()
    manager.update_conversation.assert_not_awaited()
    assert (await delivery.send(speech)).complete
    assert not buffer.has(speech.umo)
    assert await delivery.record(speech)
    assert buffer.has(speech.umo)
    manager.update_conversation.assert_awaited_once_with(
        speech.umo, "cid", history=[{"role": "assistant", "content": "hello"}]
    )


@pytest.mark.asyncio
async def test_stopping_another_binding_does_not_wait_for_existing_stop_finishing():
    h = Harness()
    h.pause("sending", 0)
    h.pause("generating", 1)
    await h.execution.reconcile()
    first, second = await h.begin(), await h.begin(1)
    await h.wait("sending", 0)
    await h.wait("generating", 1)
    stop_first = asyncio.create_task(h.stop(0))
    await asyncio.sleep(0)
    async with asyncio.timeout(1):
        await h.stop(1)
        assert (await second).state == "cancelled"
    assert not stop_first.done()
    h.release("sending", 0)
    await stop_first
    assert (await first).ok


@pytest.mark.asyncio
async def test_overlapping_changes_cannot_release_clients_before_judgment_exits():
    h = Harness()
    h.settings.decision_models = [
        DecisionModel("dm", "model", "https://example.invalid", "v1")
    ]
    h.store.rules["r0"].intent.decision_provider_id = "custom:dm"
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def slow_cancel(rule, umo, buffer, now):
        h.entered.setdefault(("judging", umo), asyncio.Event()).set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
            raise

    h.judge.judge = slow_cancel
    await h.execution.reconcile()
    h.router.reload.reset_mock()
    task = await h.begin()
    await h.wait("judging")
    h.settings.decision_models[0].model = "v2"
    syncing = asyncio.create_task(h.execution.reconcile())
    await cancelled.wait()
    await h.execution.reconcile()
    assert all(not call.args[0] for call in h.router.reload.await_args_list)
    assert (await h.execution.execute("b0", h.umo(0))).state == "busy"
    release.set()
    await syncing
    assert (await task).state == "cancelled"
    assert h.router.reload.await_args_list[-1].args[0] == {"dm"}


@pytest.mark.asyncio
async def test_cancel_immediately_after_admission_does_not_leave_busy_target():
    h = Harness()
    await h.execution.reconcile()
    caller = asyncio.create_task(h.execution.execute("b0", h.umo(0)))
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not h.sent
    assert h.execution.snapshot()["b0", h.umo(0)]["state"] == "cancelled"
    assert (await h.execution.execute("b0", h.umo(0))).ok


@pytest.mark.asyncio
async def test_global_disable_waits_for_send_then_clears_all_buffers(tmp_path):
    h = Harness()
    await h.use_configuration(tmp_path)
    h.pause("sending", 0)
    h.pause("generating", 1)
    await h.execution.reconcile()
    sending, generating = await h.begin(), await h.begin(1)
    await h.wait("sending")
    await h.wait("generating", 1)
    h.enabled = False
    disable = asyncio.create_task(AmbientChatPlugin.apply_store_changes(h))
    assert (await generating).state == "cancelled"
    assert not disable.done()
    h.release("sending")
    await disable
    assert (await sending).ok
    assert h.buffer.tracked == 0


@pytest.mark.asyncio
async def test_stopping_b_does_not_wait_for_scheduled_send_in_a():
    h = Harness()
    h.pause("sending", 0)
    h.pause("generating", 1)
    await h.execution.reconcile()

    # 用调度循环持有实际发送，确保配置重建也不等待另一绑定的收尾。
    async def scheduled(binding, rule):
        await h.execution.execute(binding.id, binding.targets[0])

    h.scheduler._run_binding = scheduled
    await h.scheduler.rebuild()
    await h.wait("sending", 0)
    await h.wait("generating", 1)
    stop_a = asyncio.create_task(h.stop(0))
    await asyncio.sleep(0)
    async with asyncio.timeout(1):
        await h.stop(1)
        await h.scheduler.rebuild()
    assert not stop_a.done()
    h.release("sending", 0)
    await stop_a
    await h.scheduler.stop()


@pytest.mark.asyncio
async def test_configuration_stop_b_does_not_wait_for_stop_a_to_finish(tmp_path):
    h = Harness()
    await h.use_configuration(tmp_path)
    h.pause("sending", 0)
    h.pause("generating", 1)
    await h.execution.reconcile()
    sending, generating = await h.begin(), await h.begin(1)
    await h.wait("sending", 0)
    await h.wait("generating", 1)
    stop_a = asyncio.create_task(AmbientChatPlugin.stop_binding(h, "b0"))
    await asyncio.sleep(0)
    async with asyncio.timeout(1):
        stopped = await AmbientChatPlugin.stop_binding(h, "b1")
    assert not stopped.enabled
    assert not stop_a.done()
    assert (await generating).state == "cancelled"
    h.release("sending", 0)
    await stop_a
    assert (await sending).ok
    assert h.configuration.snapshot()["state"] == "applied"


@pytest.mark.asyncio
async def test_stop_write_failure_does_not_cancel_existing_execution(
    tmp_path, monkeypatch
):
    h = Harness()
    await h.use_configuration(tmp_path)
    h.pause("generating")
    await h.execution.reconcile()
    task = await h.begin()
    await h.wait("generating")
    original = h.store.get_binding("b0")

    def fail_write(*args):
        raise OSError("写入失败")

    with monkeypatch.context() as patch:
        patch.setattr(
            "astrbot_plugin_ambient_chat.rules.store.os.replace",
            fail_write,
        )
        with pytest.raises(OSError):
            await AmbientChatPlugin.stop_binding(h, "b0")
    assert h.store.get_binding("b0") is original and original.enabled
    h.release("generating")
    assert (await task).ok
    assert h.configuration.snapshot()["state"] == "applied"


@pytest.mark.asyncio
async def test_failed_runtime_rebuild_can_be_reapplied_without_resaving(tmp_path):
    h = Harness()
    await h.use_configuration(tmp_path)
    h.pause("generating")
    await h.execution.reconcile()
    task = await h.begin()
    await h.wait("generating")
    rebuild = h.scheduler.rebuild
    h.scheduler.rebuild = AsyncMock(side_effect=RuntimeError("调度更新失败"))
    binding = await AmbientChatPlugin.stop_binding(h, "b0")
    assert not binding.enabled
    assert (await task).state == "cancelled"
    assert h.configuration.snapshot()["state"] == "failed"
    revision, disk = h.store.revision, h.store.path.read_bytes()
    h.scheduler.rebuild = rebuild
    assert (await h.configuration.apply())["state"] == "applied"
    assert h.store.path.read_bytes() == disk and h.store.revision == revision
    await h.scheduler.stop()


@pytest.mark.asyncio
async def test_retry_after_preheat_failure_actually_preheats_again(tmp_path):
    h = Harness()
    h.store.bindings = {"b0": h.store.bindings["b0"]}
    h.store._data_dir = tmp_path
    h.context = SimpleNamespace()
    h.lifecycle = BindingLifecycle(h)
    h.configuration = ConfigurationApplication(h.store, h.lifecycle.reconcile)
    h.buffer.load_history = AsyncMock(side_effect=RuntimeError("预热失败"))
    assert (await h.configuration.apply())["state"] == "failed"
    assert not h.lifecycle.accepts(h.umo(0))
    h.buffer.load_history.side_effect = None
    h.buffer.load_history.return_value = {}
    assert (await h.configuration.apply())["state"] == "applied"
    assert h.lifecycle.accepts(h.umo(0))
    assert h.buffer.load_history.await_count == 2
    await h.scheduler.stop()
