"""长期记忆的落地位置：判定进 ``state.memory``，生成进提示块，且只召回一次。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.memory import MemoryItem
from astrbot_plugin_ambient_chat.rules.models import Rule, Settings
from astrbot_plugin_ambient_chat.speech.delivery import Delivery, build_prompt
from astrbot_plugin_ambient_chat.speech.intent import IntentJudge
from astrbot_plugin_ambient_chat.speech.memory import (
    MemoryRecall,
    memory_entries,
    recall_memories,
    recall_query,
)

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:GroupMessage:123456"
NOW = datetime(2026, 11, 18, 21, 30, tzinfo=TZ)


def _item(
    content: str, *, score: float = 0.9, created_at: float | None = None
) -> MemoryItem:
    return MemoryItem(memory_id=1, content=content, score=score, created_at=created_at)


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
        self.debug_meta: Any = None

    async def probabilities(
        self,
        provider_id: str | None,
        prompts: Any,
        state: Any = None,
        *,
        debug_meta: Any = None,
    ) -> dict[str, float]:
        self.state = dict(state or {})
        self.debug_meta = debug_meta
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
    buffer.append(UMO, ChatMessage("我", "好呀", time.time(), is_bot=True))
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


def test_recall_query_skips_placeholder_only_messages() -> None:
    """``[Image]`` 这类占位不是说过的话：只有占位的消息不进关键词。"""
    buffer = ContextBuffer(maxlen=5)
    buffer.append(UMO, ChatMessage("J", "[Image]", time.time(), sender_id="42"))
    buffer.append(UMO, ChatMessage("小明", "这图哪来的", time.time(), sender_id="43"))

    assert recall_query(buffer.recent(UMO, 5)) == "这图哪来的"


def test_recall_query_strips_inline_placeholders() -> None:
    buffer = ContextBuffer(maxlen=5)
    buffer.append(UMO, ChatMessage("J", "看这个 [图片]", time.time(), sender_id="42"))

    assert recall_query(buffer.recent(UMO, 5)) == "看这个"


@pytest.mark.asyncio
async def test_recall_memories_respects_switch() -> None:
    recall = _Recall()
    messages = _buffer().recent(UMO, 10)

    recalled = await recall_memories(recall, Settings(), UMO, messages)

    assert recalled.items == () and recalled.query == ""
    assert recall.calls == []


@pytest.mark.asyncio
async def test_recall_memories_forwards_settings() -> None:
    recall = _Recall()
    messages = _buffer().recent(UMO, 10)

    recalled = await recall_memories(recall, _enabled(), UMO, messages)

    # 查询词随结果一起回来：调试日志靠它还原这次检索。
    assert recalled.query == recall.calls[0]["query"]
    assert [item.content for item in recalled] == ["用户周五去上海出差"]
    assert recall.calls[0]["k"] == 2
    # 群聊按最近几个不同发送者并联（默认 3），这里只有两个人且都非 bot。
    assert recall.calls[0]["senders"] == [("43", "小明"), ("42", "Lion")]
    assert recall.calls[0]["timeout"] == pytest.approx(3.0)


def test_memory_entries_fills_budget_and_truncates_tail() -> None:
    items = [
        _item(name * 300, score=0.9 - index * 0.1)
        for index, name in enumerate("甲乙丙丁戊")
    ]

    entries = memory_entries(items, entry_chars=100, max_chars=250)

    # 前两条按单条上限截断，第三条吃掉余量收尾：预算用尽而不是留空。
    assert [len(entry) for entry in entries] == [100, 100, 50]
    assert entries[-1].endswith("…")
    assert sum(len(entry) for entry in entries) <= 250


def test_memory_entries_orders_by_score_then_shorter() -> None:
    items = [
        _item("低分长句子" * 10, score=0.4),
        _item("高分" * 5, score=0.9),
        _item("同分短", score=0.4),
    ]

    entries = memory_entries(items, entry_chars=200, max_chars=600)

    assert entries[0] == "高分" * 5
    # 同分时短的在前：余量不足时先被截断的是长条目。
    assert entries[1] == "同分短"
    assert entries[2] == "低分长句子" * 10


def test_memory_entries_dedupes_and_drops_blank() -> None:
    assert memory_entries([_item("同一句"), _item(" 同一句 "), _item("  ")]) == [
        "同一句"
    ]


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
async def test_judge_hands_recall_snapshot_to_decision_debug() -> None:
    """决策调试日志多带一份查询词与完整结果，state 里只有裁好的条目。"""
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]
    recalled = MemoryRecall(
        query="周五 上海",
        items=(
            _item("用户周五去上海出差", created_at=1762000000.0),
            _item("用户下周回北京"),
        ),
    )

    await judge.judge(_rule(), UMO, _buffer(), NOW, memories=recalled)

    payload = router.debug_meta["memory_recall"]
    assert payload["query"] == "周五 上海"
    assert [item["content"] for item in payload["items"]] == [
        "用户周五去上海出差",
        "用户下周回北京",
    ]
    # 记录时间按判定用的时区渲染；缺失时留空，不编造时间。
    first, second = payload["items"]
    assert first["created_at"] == pytest.approx(1762000000.0)
    assert first["created_at_text"] == datetime.fromtimestamp(1762000000.0, TZ).strftime(
        "%Y-%m-%d %H:%M"
    )
    assert second["created_at"] is None and second["created_at_text"] is None


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
    client = _Recall()
    delivery = Delivery(
        object(),  # type: ignore[arg-type]
        _buffer(),
        settings=_enabled,
        recall=client,  # type: ignore[arg-type]
    )

    recalled = await delivery._memory_recall(
        UMO,
        _buffer().recent(UMO, 10),
        MemoryRecall(query="出差", items=(_item("用户周五去上海出差"),)),
    )
    block = delivery._memory_block(recalled, "p1")

    assert "用户周五去上海出差" in block
    assert recalled is not None and recalled.query == "出差"
    assert client.calls == []


@pytest.mark.asyncio
async def test_delivery_empty_reused_memories_does_not_recall_again() -> None:
    """执行层召回没命中时传空快照，生成侧不该再打一次请求。"""
    client = _Recall()
    delivery = Delivery(
        object(),  # type: ignore[arg-type]
        _buffer(),
        settings=_enabled,
        recall=client,  # type: ignore[arg-type]
    )

    recalled = await delivery._memory_recall(UMO, _buffer().recent(UMO, 10), [])

    assert delivery._memory_block(recalled, "p1") == ""
    assert recalled is not None and recalled.query == ""
    assert client.calls == []


@pytest.mark.asyncio
async def test_delivery_recalls_when_not_given_memories() -> None:
    client = _Recall()
    delivery = Delivery(
        object(),  # type: ignore[arg-type]
        _buffer(),
        settings=_enabled,
        recall=client,  # type: ignore[arg-type]
    )

    recalled = await delivery._memory_recall(
        UMO, _buffer().recent(UMO, 10), None
    )

    assert "用户周五去上海出差" in delivery._memory_block(recalled, "p1")
    assert len(client.calls) == 1
    assert recalled is not None and recalled.query == client.calls[0]["query"]


class _GenerationContext:
    """生成侧所需的最小上下文：会话 ID、供应商与一次成功的生成。"""

    conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="conversation")
    )

    async def get_current_chat_provider_id(self, _umo: str) -> str:
        return "p1"

    def get_provider_by_id(self, _provider_id: str) -> Any:
        return SimpleNamespace(provider_config={"modalities": []})

    async def llm_generate(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(completion_text="好呀")


@pytest.mark.asyncio
async def test_delivery_hands_recall_snapshot_to_chat_debug(tmp_path) -> None:
    """生成侧的调试日志要能看出「拿什么查的、查回了什么」，而不只是注入后的文本。"""
    settings = Settings.parse(
        {"memory_recall_enabled": True, "chat_debug_enabled": True}
    )
    delivery = Delivery(
        _GenerationContext(),  # type: ignore[arg-type]
        _buffer(),
        settings=lambda: settings,
        recall=_Recall(),  # type: ignore[arg-type]
        data_dir=lambda: tmp_path,
    )

    speech = await delivery.prepare(_rule(), UMO, "接一句", now=NOW)

    assert speech is not None
    record = json.loads(
        next((tmp_path / "chat_debug").glob("*.json")).read_text(encoding="utf-8")
    )
    recall_debug = record["meta"]["memory_recall"]
    assert recall_debug["query"] == "周五要去上海出差 帮我带点特产"
    assert recall_debug["items"][0]["content"] == "用户周五去上海出差"
