"""私聊预热历史：只有最后一条带真实时间，其余渲染为「时间未知」。"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage, format_stamp

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:FriendMessage:1"
UPDATED_AT = int(datetime(2026, 11, 4, 4, 29, tzinfo=TZ).timestamp())
HISTORY = [
    {"role": "user", "content": "少吃点"},
    {"role": "assistant", "content": "知道了知道了"},
    {"role": "user", "content": "我也醒了"},
]


class _ConversationManager:
    async def get_curr_conversation_id(self, umo: str) -> str:
        return "cid-1"

    async def get_conversation(self, umo: str, cid: str) -> Any:
        return SimpleNamespace(
            history=json.dumps(HISTORY, ensure_ascii=False),
            updated_at=UPDATED_AT,
        )


def test_zero_timestamp_renders_as_unknown() -> None:
    assert format_stamp(0.0, TZ) == "时间未知"


def test_render_keeps_real_stamp() -> None:
    message = ChatMessage(sender_name="小明", text="在吗", timestamp=UPDATED_AT)

    assert message.render(TZ) == "小明 (2026-11-04 周三 04:29): 在吗"


@pytest.mark.asyncio
async def test_private_preheat_stamps_only_last_message() -> None:
    buffer = ContextBuffer(maxlen=20)
    context = SimpleNamespace(conversation_manager=_ConversationManager())

    await buffer.preheat(context, {UMO: 3})  # type: ignore[arg-type]
    messages = buffer.recent(UMO, 10)

    assert [message.timestamp for message in messages] == [0.0, 0.0, float(UPDATED_AT)]
    lines = ContextBuffer.render(messages, TZ).splitlines()
    assert lines[0] == "某人 (时间未知): 少吃点"
    assert lines[1] == "我 (时间未知): 知道了知道了"
    assert lines[2] == "某人 (2026-11-04 周三 04:29): 我也醒了"
    # 时间未知的 bot 消息不参与冷却基线。
    assert buffer.last_bot_at(UMO) is None
