"""对话模型单次生成：成功返回响应，失败返回 None。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
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
