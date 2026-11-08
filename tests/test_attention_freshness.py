"""关注度只按新内容记账：缓冲标记与判定侧的折算比例（ADR-0035）。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import Rule
from astrbot_plugin_ambient_chat.speech.intent import IntentJudge

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:FriendMessage:1"


class _Router:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.calls = 0

    async def probabilities(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"interested": 0.4, "want_new_topic": 0.3}


def _rule(context_turns: int = 20) -> Rule:
    # 关掉注入人格：这里只验证记账份量，不需要核心的人格接口。
    return Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "decision", "threshold": 0.7},
            "context_turns": context_turns,
            "inject_persona": False,
        },
        TZ,
    )


def _buffer(count: int, *, maxlen: int = 100) -> ContextBuffer:
    buffer = ContextBuffer(maxlen=maxlen)
    base = time.time()
    for index in range(count):
        buffer.append(UMO, ChatMessage("A", f"第 {index} 条", base + index))
    return buffer


def _say(buffer: ContextBuffer, text: str = "新消息") -> None:
    buffer.append(UMO, ChatMessage("B", text, time.time()))


def test_fresh_count_only_counts_messages_after_the_marker() -> None:
    buffer = _buffer(3)
    assert buffer.fresh_count(UMO, buffer.recent(UMO, 3)) == 3

    buffer.mark_judged(UMO, buffer.recent(UMO, 3))
    assert buffer.fresh_count(UMO, buffer.recent(UMO, 3)) == 0

    _say(buffer)
    assert buffer.fresh_count(UMO, buffer.recent(UMO, 10)) == 1


def test_narrower_window_does_not_turn_old_messages_into_new_ones() -> None:
    """字符预算或配置把窗口截短后，剩下的仍然都是已记账的旧内容。"""
    buffer = _buffer(10)
    buffer.mark_judged(UMO, buffer.recent(UMO, 10))

    assert buffer.fresh_count(UMO, buffer.recent(UMO, 2)) == 0


def test_evicted_marker_falls_back_to_all_new() -> None:
    """标记被淘汰后无法再分新旧，全部按新增算：宁可多记，也不漏记。"""
    buffer = ContextBuffer(maxlen=10)
    for index in range(10):
        buffer.append(UMO, ChatMessage("A", f"旧 {index}", time.time() + index))
    buffer.mark_judged(UMO, buffer.recent(UMO, 10))

    for index in range(10):
        buffer.append(UMO, ChatMessage("A", f"新 {index}", time.time() + 100 + index))

    assert buffer.fresh_count(UMO, buffer.recent(UMO, 10)) == 10


def test_clearing_the_buffer_drops_the_marker() -> None:
    buffer = _buffer(2)
    buffer.mark_judged(UMO, buffer.recent(UMO, 2))
    buffer.clear(UMO)

    _say(buffer)

    assert buffer.fresh_count(UMO, buffer.recent(UMO, 1)) == 1


def test_window_that_slid_past_the_marker_counts_the_whole_slice() -> None:
    """窗口整体滑过标记（切片里一条旧消息都没有）时，切片里每一条都是新增。"""
    buffer = _buffer(10)
    buffer.mark_judged(UMO, buffer.recent(UMO, 10))
    for index in range(4):
        _say(buffer, f"新 {index}")

    assert buffer.fresh_count(UMO, buffer.recent(UMO, 4)) == 4


def test_evicted_message_stops_being_the_marker() -> None:
    """标记被挤出缓冲后立刻放手，不再拖住一个媒体已释放、也再也匹配不上的对象。"""
    buffer = ContextBuffer(maxlen=10)
    for index in range(10):
        buffer.append(UMO, ChatMessage("A", f"旧 {index}", time.time() + index))
    buffer.mark_judged(UMO, buffer.recent(UMO, 1))

    for index in range(10):
        _say(buffer, f"新 {index}")

    assert buffer._judged_upto.get(UMO) is None


def test_preheated_history_is_not_new_content() -> None:
    """启动预热读进缓冲的是历史：那次判定不该把整窗历史再记一遍满额账（ADR-0035）。"""
    buffer = ContextBuffer(maxlen=100)
    history = [ChatMessage("A", f"历史 {index}", 0.0) for index in range(8)]

    buffer.publish_history({UMO: history})

    assert buffer.fresh_count(UMO, buffer.recent(UMO, 20)) == 0
    _say(buffer)
    assert buffer.fresh_count(UMO, buffer.recent(UMO, 20)) == 1


@pytest.mark.asyncio
async def test_judge_reports_the_share_of_new_content() -> None:
    buffer = _buffer(4)
    judge = IntentJudge(None, _Router())  # type: ignore[arg-type]

    first = await judge.judge(_rule(), UMO, buffer, datetime.now(TZ))
    assert first.fresh_ratio == pytest.approx(1.0)

    unchanged = await judge.judge(_rule(), UMO, buffer, datetime.now(TZ))
    assert unchanged.fresh_ratio == 0.0

    _say(buffer)
    grown = await judge.judge(_rule(), UMO, buffer, datetime.now(TZ))
    # 5 条窗口里只有 1 条是新的。
    assert grown.fresh_ratio == pytest.approx(1 / 5)


@pytest.mark.asyncio
async def test_ratio_is_measured_on_the_window_actually_sent() -> None:
    """分母是真正进请求的条数：配置只注入 2 条时，被排除的旧消息不参与折算。"""
    buffer = _buffer(5)
    judge = IntentJudge(None, _Router())  # type: ignore[arg-type]

    await judge.judge(_rule(context_turns=2), UMO, buffer, datetime.now(TZ))
    again = await judge.judge(_rule(context_turns=2), UMO, buffer, datetime.now(TZ))

    assert again.fresh_ratio == 0.0


@pytest.mark.asyncio
async def test_failed_judgment_keeps_the_content_fresh() -> None:
    """判定失败没有记账，这段内容下次重试仍算新内容，一份证据不会被吞掉。"""
    router = _Router()
    router.error = RuntimeError("连接被重置")
    judge = IntentJudge(None, router)  # type: ignore[arg-type]
    buffer = _buffer(3)

    failed = await judge.judge(_rule(), UMO, buffer, datetime.now(TZ))
    assert failed.scores is None

    router.error = None
    retried = await judge.judge(_rule(), UMO, buffer, datetime.now(TZ))
    assert retried.fresh_ratio == pytest.approx(1.0)
