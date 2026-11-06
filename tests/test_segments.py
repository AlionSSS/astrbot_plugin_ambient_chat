"""分段拟人发言：切分的保真与投递循环的时序。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.rules.models import Settings
from astrbot_plugin_ambient_chat.speech.delivery import Delivery, PreparedSpeech
from astrbot_plugin_ambient_chat.speech.segments import (
    DEFAULT_SEGMENT_PATTERN,
    MAX_SEGMENTS,
    split,
)

UMO = "test:GroupMessage:0"


def _split(text):
    return split(text, threshold=150, pattern=DEFAULT_SEGMENT_PATTERN)


def _now():
    return asyncio.get_running_loop().time()


def _delivery(send_message, settings=None):
    return Delivery(
        SimpleNamespace(send_message=send_message),
        ContextBuffer(),
        settings=(lambda: settings) if settings is not None else None,
    )


def _recorder(sent):
    async def send_message(_umo, _chain):
        sent.append(_now())
        return True

    return send_message


# ------------------------------------------------------------------ 切分
def test_split_keeps_short_text_whole():
    assert _split("你好。") == ["你好。"]
    assert _split("没有标点") == ["没有标点"]


def test_split_cuts_on_sentence_end_marks():
    assert _split("你好。再见！在吗？") == ["你好。", "再见！", "在吗？"]


def test_split_skips_text_over_threshold():
    text = "一。" * 80
    assert split(text, threshold=150, pattern=DEFAULT_SEGMENT_PATTERN) == [text]


def test_split_falls_back_when_pattern_does_not_cover_text():
    # 只匹配标点：覆盖不上正文，宁可整条直发也不丢字。
    assert split("你好。再见。", threshold=150, pattern="。") == ["你好。再见。"]


def test_split_falls_back_when_too_many_segments():
    text = "嗯。" * (MAX_SEGMENTS + 1)
    assert _split(text) == [text]


def test_split_never_adds_or_drops_characters():
    text = "第一句。第二句！第三句？尾巴没有标点"
    assert "".join(_split(text)) == text


def test_split_survives_invalid_pattern():
    assert split("你好。", threshold=150, pattern="(") == ["你好。"]


# ------------------------------------------------------------------ 投递
@pytest.mark.asyncio
async def test_send_splits_and_waits_between_segments():
    sent = []

    async def send_message(_umo, _chain):
        sent.append(_now())
        return True

    delivery = _delivery(
        send_message,
        Settings(segmented_interval_min=0.05, segmented_interval_max=0.05),
    )
    outcome = await delivery.send(PreparedSpeech(UMO, "cid", "第一段。第二段。第三段。"))
    assert outcome.complete and outcome.sent == 3
    assert len(sent) == 3
    # 留出 Windows 计时器 15.6 ms 颗粒度的余量。
    assert sent[1] - sent[0] >= 0.03 and sent[2] - sent[1] >= 0.03


@pytest.mark.asyncio
async def test_send_keeps_one_message_when_switch_is_off():
    sent = []
    delivery = _delivery(_recorder(sent), Settings(segmented_enabled=False))
    outcome = await delivery.send(PreparedSpeech(UMO, "cid", "第一段。第二段。"))
    assert outcome.complete and outcome.sent == 1 and len(sent) == 1


@pytest.mark.asyncio
async def test_send_keeps_one_message_on_blocked_platform():
    sent = []
    delivery = _delivery(_recorder(sent), Settings())
    outcome = await delivery.send(
        PreparedSpeech("dingtalk:FriendMessage:1", "cid", "第一段。第二段。")
    )
    assert outcome.complete and outcome.sent == 1 and len(sent) == 1


@pytest.mark.asyncio
async def test_send_raises_when_first_segment_fails():
    delivery = _delivery(AsyncMock(side_effect=RuntimeError("offline")), Settings())
    with pytest.raises(RuntimeError):
        await delivery.send(PreparedSpeech(UMO, "cid", "第一段。第二段。"))


@pytest.mark.asyncio
async def test_send_reports_partial_when_a_later_segment_fails():
    sent = []

    async def send_message(_umo, _chain):
        sent.append(_now())
        if len(sent) == 2:
            raise RuntimeError("offline")
        return True

    delivery = _delivery(
        send_message,
        Settings(segmented_interval_min=0.0, segmented_interval_max=0.0),
    )
    outcome = await delivery.send(PreparedSpeech(UMO, "cid", "第一段。第二段。第三段。"))
    assert outcome.sent == 1 and outcome.total == 3
    assert outcome.stopped_by == "failed"
    assert outcome.text == "第一段。"
