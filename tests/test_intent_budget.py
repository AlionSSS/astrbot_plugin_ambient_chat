"""判定请求的字符预算裁剪：保住最新对话、压住人格长度。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.speech.intent import (
    DECISION_STATE_CHAR_BUDGET,
    MAX_PERSONA_CHARS,
    _entry_cost,
    _fit_conversation,
    _shrink_entry,
)

TZ = ZoneInfo("Asia/Shanghai")
TS = datetime(2026, 11, 3, 22, 20, tzinfo=TZ).timestamp()

QUESTION_OVERHEAD_CHARS = 400
"""判定请求里 questions / model 等固定部分的字符数（实测约 380）。"""


def _cost(conversation: list[dict], persona: str) -> int:
    return sum(_entry_cost(entry) for entry in conversation) + len(persona)


def test_budget_stays_inside_8k_even_at_one_token_per_char() -> None:
    """预算是按「1 字 ≈ 1 token」的最坏假设定的（ADR-0018 / ADR-0023）。"""
    assert DECISION_STATE_CHAR_BUDGET + QUESTION_OVERHEAD_CHARS <= 8192


def test_keeps_newest_messages_within_budget() -> None:
    messages = [
        ChatMessage(sender_name=f"用户{index:03d}", text="字" * 300, timestamp=TS)
        for index in range(60)
    ]

    conversation, persona = _fit_conversation(messages, TZ, "人设")

    assert _cost(conversation, persona) <= DECISION_STATE_CHAR_BUDGET
    senders = [entry["sender_name"] for entry in conversation]
    assert senders[-1] == "用户059"
    assert "用户000" not in senders
    # 保留的始终是「最新的一段」，顺序仍然是时间正序。
    assert senders == sorted(senders)


def test_long_persona_is_cut_and_keeps_conversation() -> None:
    messages = [ChatMessage(sender_name="A", text="最近一句话", timestamp=TS)]

    conversation, persona = _fit_conversation(messages, TZ, "人" * 6000)

    assert len(persona) <= MAX_PERSONA_CHARS
    assert persona.endswith("…")
    assert conversation[0]["content"] == "最近一句话"
    assert _cost(conversation, persona) <= DECISION_STATE_CHAR_BUDGET


def test_unknown_time_stays_none() -> None:
    conversation, _persona = _fit_conversation(
        [ChatMessage(sender_name="A", text="在吗", timestamp=0.0)], TZ, ""
    )

    assert conversation[0]["time"] is None


def test_short_conversation_untouched() -> None:
    messages = [
        ChatMessage(sender_name="A", text="在吗", timestamp=TS),
        ChatMessage(sender_name="B", text="在", timestamp=TS + 60),
    ]

    conversation, persona = _fit_conversation(messages, TZ, "你是助手")

    assert [entry["content"] for entry in conversation] == ["在吗", "在"]
    assert conversation[0]["time"] == "2026-11-03 22:20:00"
    assert conversation[0]["sender_name"] == "A"
    assert conversation[0]["type"] == "text"
    assert "self" not in conversation[0]
    assert persona == "你是助手"


def test_shrink_entry_only_cuts_text() -> None:
    entry = {"time": None, "sender_name": "A", "content": "前" * 100 + "尾部"}

    shrunk = _shrink_entry(entry, 80)

    assert shrunk is not None
    assert shrunk["sender_name"] == "A" and shrunk["time"] is None
    assert shrunk["content"].startswith("…")
    assert shrunk["content"].endswith("尾部")
    assert len(shrunk["content"]) < len(entry["content"])


def test_shrink_entry_gives_up_when_budget_too_small() -> None:
    entry = {"time": None, "sender_name": "A", "content": "在吗"}

    assert _shrink_entry(entry, 10) is None


def test_shrink_entry_never_fabricates_content_for_poke() -> None:
    """戳一戳条目没有正文，放不下就整条丢弃，不能凭空塞一个空 content。"""
    entry = {
        "time": None,
        "sender_name": "小红",
        "type": "poke",
        "target": {"id": "10000", "is_self": True},
    }

    assert _shrink_entry(entry, 400) is None
