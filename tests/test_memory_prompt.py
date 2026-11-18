"""长期记忆的落地位置：判定进 ``state.memory``，生成进提示块，且只召回一次。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.memory import MemoryItem
from astrbot_plugin_ambient_chat.rules.models import Rule, Settings
from astrbot_plugin_ambient_chat.speech.delivery import Delivery, build_prompt
from astrbot_plugin_ambient_chat.speech.intent import IntentJudge
from astrbot_plugin_ambient_chat.speech.memory import (
    DECISION_MEMORY_CHARS,
    memory_entries,
    recall_memories,
    recall_query,
)

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:GroupMessage:123456"
NOW = datetime(2026, 11, 18, 21, 30, tzinfo=TZ)


def _item(content: str, *, score: float = 0.9) -> MemoryItem:
    return MemoryItem(memory_id=1, content=content, score=score)


def _rule() -> Rule:
    return Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "decision", "threshold": 0.7},
            "inject_persona": False,
        },
        TZ,
    )


class _Router:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    async def probabilities(
        self, provider_id: str | None, prompts: Any, state: Any = None
    ) -> dict[str, float]:
        self.state = dict(state or {})
        return {"interested": 0.1, "want_new_topic": 0.2}


class _Recall:
    """只记录调用次数的召回客户端。"""

    def __init__(self, items: list[MemoryItem] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._items = items if items is not None else [_item("用户周五去上海出差")]

    async def recall(self, query: str, **kwargs: Any) -> list[MemoryItem]:
        self.calls.append({"query": query, **kwargs})
        return list(self._items)


def _buffer() -> ContextBuffer:
    buffer = ContextBuffer(maxlen=10)
    buffer.append(
        UMO,
        ChatMessage("Lion", "周五要去上海出差", time.time() - 60, sender_id="42"),
    )
    buffer.append(
        UMO,
        ChatMessage("小明", "帮我带点特产", time.time(), sender_id="43"),
    )
    buffer.append(
        UMO, ChatMessage("我", "好呀", time.time(), is_bot=True)
    )
    return buffer


def _enabled() -> Settings:
    return Settings.parse({"memory_recall_enabled": True, "memory_recall_k": 2})


def test_recall_query_uses_recent_messages_without_bot() -> None:
    # 自己的发言不是线索，否则会不断用自己说过的话检索自己。
    assert recall_query(_buffer().recent(UMO, 10)) == "周五要去上海出差 帮我带点特产"


def test_recall_query_empty_when_only_bot_spoke() -> None:
    buffer = ContextBuffer(maxlen=5)
    buffer.append(UMO, ChatMessage("我", "自言自语", time.time(), is_bot=True))

    assert recall_query(buffer.recent(UMO, 5)) == ""


@pytest.mark.asyncio
async def test_recall_memories_respects_switch() -> None:
    recall = _Recall()
    messages = _buffer().recent(UMO, 10)

    assert await recall_memories(recall, Settings(), UMO, messages) == []
    assert recall.calls == []


@pytest.mark.asyncio
async def test_recall_memories_forwards_settings() -> None:
    recall = _Recall()
    messages = _buffer().recent(UMO, 10)

    await recall_memories(recall, _enabled(), UMO, messages)

    assert recall.calls[0]["k"] == 2
    # 群聊按最近几个不同发送者并联（默认 3），这里只有两个人且都非 bot。
    assert recall.calls[0]["senders"] == [("43", "小明"), ("42", "Lion")]
    assert recall.calls[0]["timeout"] == pytest.approx(3.0)


def test_memory_entries_respects_entry_and_total_budget() -> None:
    entries = memory_entries(
        [_item("甲" * 300), _item("乙" * 300), _item("丙" * 300)],
        entry_chars=100,
    )

    assert entries[0].endswith("…")
    assert len(entries[0]) == 100
    assert sum(len(entry) for entry in entries) <= DECISION_MEMORY_CHARS


def test_memory_entries_dedupes_and_drops_blank() -> None:
    assert memory_entries([_item("同一句"), _item(" 同一句 "), _item("  ")]) == ["同一句"]


@pytest.mark.asyncio
async def test_judge_puts_memories_into_state() -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    await judge.judge(
        _rule(),
        UMO,
        _buffer(),
        NOW,
        memories=[_item("用户周五去上海出差")],
    )

    assert router.state["memory"] == ["用户周五去上海出差"]


@pytest.mark.asyncio
async def test_judge_without_memories_omits_state_key() -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    await judge.judge(_rule(), UMO, _buffer(), NOW)

    assert "memory" not in router.state


def test_build_prompt_places_memory_between_transcript_and_reminder() -> None:
    messages = _buffer().recent(UMO, 10)
    target_umo = UMO
    from astrbot_plugin_ambient_chat.common import parse_target

    target = parse_target(target_umo)
    assert target is not None
    block = "<long_term_memory>\n记忆\n</long_term_memory>"

    prompt = build_prompt("说点什么", messages, NOW, target, memory_block=block)

    assert prompt.index("<recent_conversation>") < prompt.index(block)
    assert prompt.index(block) < prompt.index("<system_reminder>")
    assert prompt.endswith("说点什么")


def test_build_prompt_without_memory_unchanged() -> None:
    from astrbot_plugin_ambient_chat.common import parse_target

    messages = _buffer().recent(UMO, 10)
    target = parse_target(UMO)
    assert target is not None

    prompt = build_prompt("说点什么", messages, NOW, target)

    assert "<long_term_memory>" not in prompt


@pytest.mark.asyncio
async def test_delivery_reuses_given_memories_without_recalling() -> None:
    recall = _Recall()
    delivery = Delivery(
        object(),  # type: ignore[arg-type]
        _buffer(),
        settings=_enabled,
        recall=recall,  # type: ignore[arg-type]
    )

    block = await delivery._memory_block(
        UMO, _buffer().recent(UMO, 10), [_item("用户周五去上海出差")]
    )

    assert "用户周五去上海出差" in block
    assert recall.calls == []


@pytest.mark.asyncio
async def test_delivery_empty_reused_memories_does_not_recall_again() -> None:
    """执行层召回没命中时传空列表，生成侧不该再打一次请求。"""
    recall = _Recall()
    delivery = Delivery(
        object(),  # type: ignore[arg-type]
        _buffer(),
        settings=_enabled,
        recall=recall,  # type: ignore[arg-type]
    )

    assert await delivery._memory_block(UMO, _buffer().recent(UMO, 10), []) == ""
    assert recall.calls == []


@pytest.mark.asyncio
async def test_delivery_recalls_when_not_given_memories() -> None:
    recall = _Recall()
    delivery = Delivery(
        object(),  # type: ignore[arg-type]
        _buffer(),
        settings=_enabled,
        recall=recall,  # type: ignore[arg-type]
    )

    block = await delivery._memory_block(UMO, _buffer().recent(UMO, 10), None)

    assert "用户周五去上海出差" in block
    assert len(recall.calls) == 1
