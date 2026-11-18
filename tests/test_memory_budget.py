"""判定 state 预算与对话模型剩余空间内的记忆预算。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot.core.provider.entities import ProviderType
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import (
    DEFAULT_DECISION_CONTEXT_TOKENS,
    DecisionModel,
    RuleError,
    Settings,
)
from astrbot_plugin_ambient_chat.speech.intent import _entry_cost, _fit_conversation
from astrbot_plugin_ambient_chat.speech.decision import (
    DECISION_STATE_CHAR_BUDGET,
    decision_state_chars,
    decision_window,
)
from astrbot_plugin_ambient_chat.speech.memory import (
    DELIVERY_MEMORY_CHARS,
    delivery_memory_chars,
    provider_context_tokens,
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


@pytest.mark.parametrize(
    "tokens, expected",
    [(None, 13926), (2048, 1740), (8192, 6963), (32768, 27852), (200000, 48000)],
)
def test_decision_budget_scales_and_caps(tokens: int | None, expected: int) -> None:
    assert decision_state_chars(tokens) == expected
    assert decision_state_chars() == DECISION_STATE_CHAR_BUDGET


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


@pytest.mark.parametrize("tokens", [2048, 8191, 200001])
def test_decision_model_rejects_out_of_range_window(tokens: int) -> None:
    with pytest.raises(RuleError, match="上下文窗口"):
        DecisionModel.parse({**_MODEL, "context_tokens": tokens})


def test_decision_model_accepts_8k_window() -> None:
    assert (
        DecisionModel.parse({**_MODEL, "context_tokens": 8192}).context_tokens == 8192
    )


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
    def __init__(self, *providers: _DecisionProvider) -> None:
        self.decision_provider_insts = list(providers)
        self.curr_decision_provider_inst = providers[-1]

    def get_using_provider(self, provider_type: ProviderType) -> _DecisionProvider:
        assert provider_type == ProviderType.DECISION
        return self.curr_decision_provider_inst


class _DecisionContext:
    def __init__(self, *providers: _DecisionProvider) -> None:
        self.provider_manager = _ProviderManager(*providers)

    def get_provider_by_id(self, provider_id: str) -> _DecisionProvider | None:
        return next(
            (
                p
                for p in self.provider_manager.decision_provider_insts
                if p.meta().id == provider_id
            ),
            None,
        )


def test_decision_window_reads_the_selected_model() -> None:
    settings = Settings.parse(
        {"decision_models": [{**_MODEL, "context_tokens": 32768}]}
    )
    context = _DecisionContext(
        _DecisionProvider("dp", 65536), _DecisionProvider("active", 8192)
    )

    assert decision_window(context, settings, "custom:dm") == 32768
    assert decision_window(context, settings, "dp") == 65536
    # 默认提供商不一定排第一。
    assert decision_window(context, settings, "") == 8192
    # 模型被删或提供商没声明窗口时退回默认值，判定照常进行。
    assert decision_window(None, settings, "custom:missing") == (
        DEFAULT_DECISION_CONTEXT_TOKENS
    )
    assert decision_window(None, settings, "dp") == DEFAULT_DECISION_CONTEXT_TOKENS


@pytest.mark.parametrize(
    "window, expected", [(4096, 300), (8192, 600), (16384, 1200), (65536, 1800)]
)
def test_delivery_budget_scales_with_window_up_to_cap(
    window: int, expected: int
) -> None:
    assert delivery_memory_chars(window, used_chars=1000) == expected


def test_delivery_budget_only_uses_request_headroom() -> None:
    assert delivery_memory_chars(8192, used_chars=6800) == 163
    assert delivery_memory_chars(8192, used_chars=6000, images=1) == 0
    assert delivery_memory_chars(8192, used_chars=5000, audios=1) == 0
    assert delivery_memory_chars(8192, used_chars=7000) == 0


@pytest.mark.parametrize(
    "context",
    [
        _Context(None),
        _Context(_Provider(tokens=0)),
        _Context(_Provider(tokens=0), error=True),
    ],
)
def test_delivery_budget_falls_back_without_window(context: _Context) -> None:
    window = provider_context_tokens(context, "p1")
    assert window == 0
    assert delivery_memory_chars(window, used_chars=1000) == DELIVERY_MEMORY_CHARS
    assert delivery_memory_chars(window, used_chars=6800) == 163
