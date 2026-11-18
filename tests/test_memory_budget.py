"""预算随模型窗口缩放：判定 state、记忆条目与生成侧记忆块。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.memory.fit import ENTRY_SHARE
from astrbot_plugin_ambient_chat.rules.models import (
    DEFAULT_DECISION_CONTEXT_TOKENS,
    DecisionModel,
    RuleError,
    Settings,
)
from astrbot_plugin_ambient_chat.speech.intent import _entry_cost, _fit_conversation
from astrbot_plugin_ambient_chat.speech.memory import (
    DECISION_MEMORY_CHARS,
    DECISION_STATE_CHAR_BUDGET,
    DELIVERY_MEMORY_CHARS,
    decision_budget,
    decision_window,
    delivery_memory_chars,
)

TZ = ZoneInfo("Asia/Shanghai")
TS = datetime(2026, 11, 18, 21, 30, tzinfo=TZ).timestamp()


class _Provider:
    """只回答上下文窗口与模型名。"""

    def __init__(self, tokens: int = 0, model: str = "") -> None:
        self.provider_config = {"max_context_tokens": tokens}
        self._model = model

    def get_model(self) -> str:
        return self._model


class _Context:
    def __init__(self, provider: Any = None, *, error: bool = False) -> None:
        self._provider = provider
        self._error = error

    def get_provider_by_id(self, _provider_id: str) -> Any:
        if self._error:
            raise LookupError("provider 不存在")
        return self._provider


def test_default_window_budget() -> None:
    """默认 16K 窗口下，预算按 ADR-0009 的比例推导。"""
    assert DECISION_STATE_CHAR_BUDGET == int(16_384 * 0.85)
    assert DECISION_MEMORY_CHARS == int(DECISION_STATE_CHAR_BUDGET * 0.06)

    budget = decision_budget()

    assert budget.state_chars == DECISION_STATE_CHAR_BUDGET
    assert budget.memory_chars == DECISION_MEMORY_CHARS
    assert budget.entry_chars == int(DECISION_MEMORY_CHARS * ENTRY_SHARE)


def test_budget_scales_with_window() -> None:
    budget = decision_budget(32768)

    assert budget.state_chars == int(32768 * 0.85)
    assert budget.memory_chars > DECISION_MEMORY_CHARS
    assert budget.state_chars > DECISION_STATE_CHAR_BUDGET


def test_budget_is_capped_at_large_windows() -> None:
    budget = decision_budget(200000)

    assert budget.state_chars == 48_000
    assert budget.memory_chars == 2_000


def test_budget_has_floor_for_tiny_windows() -> None:
    budget = decision_budget(2048)

    assert budget.state_chars == 3_000
    assert budget.memory_chars == 240


def test_entry_cap_is_a_share_of_total_budget() -> None:
    """单条上限按总预算推导：窗口变大时同步放宽，不按条数摊薄。"""
    small = decision_budget()
    large = decision_budget(32768)

    assert small.entry_chars == int(small.memory_chars * ENTRY_SHARE)
    assert large.entry_chars == int(large.memory_chars * ENTRY_SHARE)
    assert large.entry_chars > small.entry_chars


def test_bigger_window_keeps_more_conversation() -> None:
    messages = [
        ChatMessage(sender_name=f"用户{index:03d}", text="字" * 300, timestamp=TS)
        for index in range(40)
    ]

    small, persona_small = _fit_conversation(messages, TZ, "人设", budget=3_000)
    large, persona_large = _fit_conversation(messages, TZ, "人设", budget=30_000)

    assert len(large) > len(small)
    assert sum(_entry_cost(entry) for entry in large) + len(persona_large) <= 30_000
    assert sum(_entry_cost(entry) for entry in small) + len(persona_small) <= 3_000


_MODEL = {"id": "dm", "base_url": "https://example.invalid", "model": "m"}


@pytest.mark.parametrize("tokens", [2047, 200001])
def test_decision_model_rejects_out_of_range_window(tokens: int) -> None:
    with pytest.raises(RuleError, match="上下文窗口"):
        DecisionModel.parse({**_MODEL, "context_tokens": tokens})


def test_decision_model_context_tokens_roundtrip() -> None:
    """窗口按模型各自保存：字符串表单值能读回整数，缺省时用默认 16K。"""
    settings = Settings.parse(
        {"decision_models": [{**_MODEL, "context_tokens": "32768"}]}
    )
    model = settings.decision_models[0]

    assert model.context_tokens == 32768
    assert settings.to_dict()["decision_models"][0]["context_tokens"] == 32768
    assert (
        Settings.parse({"decision_models": [dict(_MODEL)]})
        .decision_models[0]
        .context_tokens
        == DEFAULT_DECISION_CONTEXT_TOKENS
    )


class _DecisionProvider:
    """只回答官方判断提供商的 id 与声明的窗口。"""

    def __init__(self, provider_id: str, tokens: int) -> None:
        self.provider_config = {"max_context_tokens": tokens}
        self._id = provider_id

    def meta(self) -> Any:
        return SimpleNamespace(id=self._id)


class _ProviderManager:
    def __init__(self, *providers: Any) -> None:
        self.decision_provider_insts = list(providers)


class _DecisionContext:
    def __init__(self, *providers: Any) -> None:
        self.provider_manager = _ProviderManager(*providers)


def test_decision_window_reads_the_selected_model() -> None:
    settings = Settings.parse(
        {"decision_models": [{**_MODEL, "context_tokens": 32768}]}
    )
    context = _DecisionContext(_DecisionProvider("dp", 65536))

    assert decision_window(context, settings, "custom:dm") == 32768
    assert decision_window(context, settings, "dp") == 65536
    # 引用为空表示走核心默认判断提供商，取第一个的声明值。
    assert decision_window(context, settings, "") == 65536
    # 模型被删或提供商没声明窗口时退回默认值，判定照常进行。
    assert decision_window(None, settings, "custom:missing") == (
        DEFAULT_DECISION_CONTEXT_TOKENS
    )
    assert decision_window(None, settings, "dp") == DEFAULT_DECISION_CONTEXT_TOKENS


@pytest.mark.parametrize("window", [8192, 16384, 65536, 131072])
def test_delivery_budget_scales_with_window_without_cap(window: int) -> None:
    """生成侧只跟着窗口等比放大：条数上限已经约束了总量，封顶只会让大窗口空转。"""
    chars = delivery_memory_chars(_Context(_Provider(tokens=window)), "p1")

    assert chars == int(DELIVERY_MEMORY_CHARS * window / 8192)
    assert chars >= DELIVERY_MEMORY_CHARS


@pytest.mark.parametrize(
    "context",
    [
        _Context(None),
        _Context(_Provider(tokens=0)),
        _Context(_Provider(tokens=0), error=True),
    ],
)
def test_delivery_budget_falls_back_without_window(context: _Context) -> None:
    assert delivery_memory_chars(context, "p1") == DELIVERY_MEMORY_CHARS


def test_delivery_budget_without_provider_id_falls_back() -> None:
    assert delivery_memory_chars(_Context(_Provider(tokens=65536)), "") == (
        DELIVERY_MEMORY_CHARS
    )
