"""对话模型单次生成：成功返回响应，失败返回 None。"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.rules.models import Settings
from astrbot_plugin_ambient_chat.speech.delivery import Delivery


class _Context:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def llm_generate(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(completion_text="好呀，早点睡")


class _FailingContext:
    async def llm_generate(self, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")


@pytest.mark.asyncio
async def test_generate_forwards_prompt_to_provider() -> None:
    context = _Context()
    delivery = Delivery(context, ContextBuffer())  # type: ignore[arg-type]

    response = await delivery._generate("p1", "prompt-text", "system-text", [])

    assert response.completion_text == "好呀，早点睡"
    assert context.kwargs["chat_provider_id"] == "p1"
    assert context.kwargs["prompt"] == "prompt-text"
    assert context.kwargs["system_prompt"] == "system-text"
    assert context.kwargs["image_urls"] is None


@pytest.mark.asyncio
async def test_generate_failure_returns_none() -> None:
    delivery = Delivery(_FailingContext(), ContextBuffer())  # type: ignore[arg-type]

    assert await delivery._generate("p2", "prompt", "", []) is None


class _ProviderContext:
    """只回答「这个模型支持哪些模态」。"""

    def __init__(self, modalities: list[str] | None) -> None:
        self._modalities = modalities

    def get_provider_by_id(self, _provider_id: str) -> Any:
        return SimpleNamespace(provider_config={"modalities": self._modalities})


@pytest.mark.asyncio
async def test_generate_forwards_audio_urls() -> None:
    context = _Context()
    delivery = Delivery(context, ContextBuffer())  # type: ignore[arg-type]

    await delivery._generate("p1", "prompt", "", [], ["voice.amr"])

    assert context.kwargs["audio_urls"] == ["voice.amr"]


def test_audio_capability_follows_provider_modalities() -> None:
    supports = Delivery(_ProviderContext(["audio"]), ContextBuffer())  # type: ignore[arg-type]
    unconfigured = Delivery(_ProviderContext([]), ContextBuffer())  # type: ignore[arg-type]
    text_only = Delivery(_ProviderContext(["image"]), ContextBuffer())  # type: ignore[arg-type]

    assert supports._allows_audio("p")
    assert not unconfigured._allows_audio("p")
    assert not text_only._allows_audio("p")


@pytest.mark.parametrize("modalities", [None, [], ["text"], ["tool_use"]])
def test_undeclared_modalities_mean_no_media(modalities: list[str] | None) -> None:
    """未声明就是不支持：空列表、缺失与读不到配置都不能放行媒体。"""
    delivery = Delivery(_ProviderContext(modalities), ContextBuffer())  # type: ignore[arg-type]

    assert not delivery._allows_images("p")
    assert not delivery._allows_audio("p")


@pytest.mark.asyncio
async def test_generate_writes_chat_debug_only_when_enabled(tmp_path) -> None:
    settings = Settings()
    delivery = Delivery(
        _Context(),  # type: ignore[arg-type]
        ContextBuffer(),
        settings=lambda: settings,
        data_dir=lambda: tmp_path,
    )
    directory = tmp_path / "chat_debug"

    await delivery._generate("p1", "prompt", "system", ["a.jpg"], umo="u1")
    assert not directory.exists()

    settings.chat_debug_enabled = True
    await delivery._generate("p1", "prompt", "system", ["a.jpg"], umo="u1")
    records = list(directory.glob("*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "chat"
    assert payload["meta"]["umo"] == "u1"
    assert payload["meta"]["stage"] == "initial"
    assert payload["request"]["prompt"] == "prompt"
    assert payload["request"]["system_prompt"] == "system"
    assert payload["request"]["image_urls"] == ["a.jpg"]
    assert payload["response"]["completion_text"] == "好呀，早点睡"

    settings.chat_debug_enabled = False
    await delivery._generate("p1", "prompt", "system", [], umo="u1")
    assert len(list(directory.glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_generate_failure_is_logged_with_error(tmp_path) -> None:
    settings = Settings(chat_debug_enabled=True)
    delivery = Delivery(
        _FailingContext(),  # type: ignore[arg-type]
        ContextBuffer(),
        settings=lambda: settings,
        data_dir=lambda: tmp_path,
    )

    assert await delivery._generate("p2", "prompt", "", []) is None

    records = list((tmp_path / "chat_debug").glob("*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["error"] == "RuntimeError: provider down"
    assert payload["response"] is None


def test_provider_lookup_failure_means_no_media() -> None:
    class Missing:
        def get_provider_by_id(self, _provider_id: str) -> Any:
            raise LookupError("provider 不存在")

    delivery = Delivery(Missing(), ContextBuffer())  # type: ignore[arg-type]

    assert not delivery._allows_images("p")
    assert not delivery._allows_audio("p")
