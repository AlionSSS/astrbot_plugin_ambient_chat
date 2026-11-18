"""两条判定通过线：感兴趣与新话题各自可配，默认 0.7 / 0.5。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import IntentConfig, Rule, RuleError
from astrbot_plugin_ambient_chat.speech.intent import (
    MODE_CONTINUE,
    MODE_NEW_TOPIC,
    IntentJudge,
)

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 16, 12, 0, tzinfo=TZ)
UMO = "aiocqhttp:FriendMessage:1"


class _Router:
    def __init__(self, interested: float, new_topic: float) -> None:
        self._scores = {"interested": interested, "want_new_topic": new_topic}

    async def probabilities(
        self,
        provider_id: str | None,
        prompts: Any,
        state: Any = None,
        *,
        debug_meta: Any = None,
    ) -> dict[str, float]:
        return dict(self._scores)


def _rule(intent: dict[str, Any]) -> Rule:
    return Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "decision", **intent},
            "inject_persona": False,
        },
        TZ,
    )


def _buffer() -> ContextBuffer:
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("A", "在吗", time.time()))
    return buffer


async def _judge(rule: Rule, interested: float, new_topic: float) -> Any:
    judge = IntentJudge(None, _Router(interested, new_topic))  # type: ignore[arg-type]
    return await judge.judge(rule, UMO, _buffer(), NOW)


def test_defaults_are_07_and_05() -> None:
    config = IntentConfig.parse({"mode": "decision"})
    assert (config.interest_threshold, config.new_topic_threshold) == (0.7, 0.5)
    assert IntentConfig.parse(config.to_dict()) == config


def test_legacy_threshold_reads_as_interest() -> None:
    """旧规则只有一个 threshold，升级后当作感兴趣阈值，不丢用户调过的松紧。"""
    config = IntentConfig.parse({"threshold": 0.4})
    assert config.interest_threshold == 0.4
    assert config.new_topic_threshold == 0.5


def test_threshold_out_of_range_is_rejected() -> None:
    with pytest.raises(RuleError, match="新话题阈值"):
        IntentConfig.parse({"new_topic_threshold": 1.5})


@pytest.mark.asyncio
async def test_new_topic_uses_its_own_threshold() -> None:
    # 感兴趣不到 0.7 而新话题过 0.5：走新话题。
    result = await _judge(_rule({}), interested=0.65, new_topic=0.55)
    assert result.speak
    assert result.mode == MODE_NEW_TOPIC


@pytest.mark.asyncio
async def test_interest_threshold_is_configurable() -> None:
    result = await _judge(
        _rule({"interest_threshold": 0.6}), interested=0.62, new_topic=0.1
    )
    assert result.speak
    assert result.mode == MODE_CONTINUE


@pytest.mark.asyncio
async def test_both_lines_fail_closed() -> None:
    # 新话题阈值调高后，另一条线也不能兜住它。
    result = await _judge(
        _rule({"new_topic_threshold": 0.8}), interested=0.65, new_topic=0.55
    )
    assert not result.speak
    assert result.mode == MODE_CONTINUE
