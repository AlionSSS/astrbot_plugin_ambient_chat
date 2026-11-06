"""私聊预热历史没有发送者昵称：拿到对方名字后回填整段缓冲。"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:FriendMessage:1"
GROUP_UMO = "aiocqhttp:GroupMessage:2"
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


@pytest.mark.asyncio
async def test_private_history_adopts_peer_name_when_it_becomes_known() -> None:
    buffer = ContextBuffer(maxlen=20)
    context = SimpleNamespace(conversation_manager=_ConversationManager())
    await buffer.preheat(context, {UMO: 3})  # type: ignore[arg-type]

    assert buffer.recent(UMO, 1)[0].structured_sender_name() == "某人"

    buffer.append(UMO, ChatMessage("Lion", "在吗", 100.0, sender_id="10001"))

    peers = [m for m in buffer.recent(UMO, 10) if not m.is_bot]
    assert [m.sender_name for m in peers] == ["Lion", "Lion", "Lion"]
    assert ContextBuffer.last_peer_name(buffer.recent(UMO, 10)) == "Lion"
    lines = ContextBuffer.render(buffer.recent(UMO, 10), TZ).splitlines()
    assert lines[0] == "Lion (时间未知): 少吃点"
    assert lines[1] == "我 (时间未知): 知道了知道了"


def test_group_unknown_sender_is_not_backfilled() -> None:
    buffer = ContextBuffer(maxlen=20)
    buffer.append(GROUP_UMO, ChatMessage("", "系统提示", 1.0, sender_id="999"))
    buffer.append(GROUP_UMO, ChatMessage("Lion", "在吗", 2.0, sender_id="10001"))

    assert buffer.recent(GROUP_UMO, 10)[0].sender_name == ""


def test_numeric_only_name_is_not_adopted() -> None:
    buffer = ContextBuffer(maxlen=20)
    buffer.append(UMO, ChatMessage("", "在吗", 1.0, sender_id="10001"))
    buffer.append(UMO, ChatMessage("10001", "嗯", 2.0, sender_id="10001"))

    assert buffer.recent(UMO, 10)[0].sender_name == ""
