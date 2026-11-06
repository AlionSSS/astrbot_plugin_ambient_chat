"""「按概率放行」：概率 1.0 等价于旧版直接放行，其余按一次随机数决定。"""

import random
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from astrbot_plugin_ambient_chat.rules.models import Rule
from astrbot_plugin_ambient_chat.speech.intent import IntentJudge

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=TZ)


def _rule(probability: float) -> Rule:
    return Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "direct", "pass_probability": probability},
        },
        TZ,
    )


def _judge() -> IntentJudge:
    # 概率放行不读上下文也不调用决策模型，两个依赖都可以为空。
    return IntentJudge(None, None)  # type: ignore[arg-type]


def test_default_probability_is_half() -> None:
    """没写该字段的规则（含历史规则）按默认 50% 处理。"""
    rule = Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "direct"},
        },
        TZ,
    )
    assert rule.intent.pass_probability == 0.5
    assert rule.intent.to_dict()["pass_probability"] == 0.5


@pytest.mark.asyncio
async def test_certain_probability_always_speaks(monkeypatch: Any) -> None:
    def fail() -> float:  # pragma: no cover - 概率 1.0 时不应掷随机数
        raise AssertionError("概率为 1 时不应调用随机数")

    monkeypatch.setattr(random, "random", fail)
    result = await _judge().judge(_rule(1.0), "u", None, NOW)  # type: ignore[arg-type]

    assert result.speak
    assert result.reason == "直接放行"


@pytest.mark.asyncio
@pytest.mark.parametrize(("draw", "expected"), [(0.4, True), (0.6, False)])
async def test_draw_decides_when_probability_is_partial(
    monkeypatch: Any, draw: float, expected: bool
) -> None:
    monkeypatch.setattr(random, "random", lambda: draw)
    result = await _judge().judge(_rule(0.5), "u", None, NOW)  # type: ignore[arg-type]

    assert result.speak is expected
    assert "50%" in result.reason
