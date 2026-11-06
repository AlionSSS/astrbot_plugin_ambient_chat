"""判定失败的原因要能直接展示：内容审核拒绝与一般失败分开说。"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import Rule
from astrbot_plugin_ambient_chat.speech.decision import DecisionContentRejected
from astrbot_plugin_ambient_chat.speech.intent import (
    FAILURE_CONTENT_REJECTED,
    FAILURE_DECISION,
    REASON_CONTENT_REJECTED,
    REASON_DETAIL_LIMIT,
    IntentJudge,
)

from astrbot.api.provider import DecisionError

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:FriendMessage:1"


class _FailingRouter:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def probabilities(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        raise self._error


def _judge(error: Exception) -> IntentJudge:
    return IntentJudge(None, _FailingRouter(error))  # type: ignore[arg-type]


def _rule() -> Rule:
    # 关掉注入人格：这里只验证失败原因，不需要核心的人格接口。
    return Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "decision", "threshold": 0.7},
            "inject_persona": False,
        },
        TZ,
    )


def _buffer() -> ContextBuffer:
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("A", "在吗", time.time()))
    return buffer


@pytest.mark.asyncio
async def test_content_rejection_has_fixed_actionable_reason() -> None:
    judge = _judge(DecisionContentRejected("「端砚决策」内容审核未通过：敏感内容"))

    result = await judge.judge(_rule(), UMO, _buffer(), datetime.now(TZ))

    assert not result.speak
    assert result.failure == FAILURE_CONTENT_REJECTED
    # 平台原文（英文报错）不进页面，页面只给可执行的结论。
    assert result.reason == REASON_CONTENT_REJECTED


@pytest.mark.asyncio
async def test_other_failures_keep_brief_detail() -> None:
    judge = _judge(DecisionError("判定失败：" + "很长的原因" * 200))

    result = await judge.judge(_rule(), UMO, _buffer(), datetime.now(TZ))

    assert not result.speak
    assert result.failure == FAILURE_DECISION
    assert result.reason.startswith("决策失败：")
    assert len(result.reason) <= len("决策失败：") + REASON_DETAIL_LIMIT
    assert result.reason.endswith("…")


@pytest.mark.asyncio
async def test_unexpected_exception_is_marked_as_failure() -> None:
    judge = _judge(RuntimeError("连接被重置"))

    result = await judge.judge(_rule(), UMO, _buffer(), datetime.now(TZ))

    assert result.failure == FAILURE_DECISION
    assert result.reason == "决策异常：连接被重置"
