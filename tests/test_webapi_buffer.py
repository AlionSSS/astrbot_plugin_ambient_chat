"""缓冲区检视接口：与对话渲染同源，返回发送者名字与渲染后的文本。

这条用例是回归：`ChatMessage.structured_sender_name()` 改名为
`structured_role_name()` 时，`webapi.get_buffer` 漏改会在这个接口上炸。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from astrbot_plugin_ambient_chat import webapi
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.webapi import AmbientWebApi

UMO = "aiocqhttp:GroupMessage:1"
TZ = ZoneInfo("Asia/Shanghai")
TS = 1_760_000_000.0


def _plugin(buffer: ContextBuffer) -> SimpleNamespace:
    binding = SimpleNamespace(
        id="b1",
        name="绑定",
        platform_id="aiocqhttp",
        rule_id="r1",
        enabled=True,
        targets=[UMO],
    )
    return SimpleNamespace(
        buffer=buffer,
        timezone=TZ,
        store=SimpleNamespace(
            bindings={"b1": binding},
            get_rule=lambda _rule_id: None,
        ),
    )


@pytest.mark.asyncio
async def test_buffer_preview_uses_rendered_text_and_role_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("小明", "在吗", TS, sender_id="444066154"))
    buffer.append(
        UMO,
        ChatMessage(
            "小红",
            "哈哈",
            TS + 1,
            sender_id="123456",
            kind="face",
            attachments=("face",),
        ),
    )
    buffer.append(
        UMO,
        ChatMessage(
            "我",
            "我在",
            TS + 2,
            is_bot=True,
            sender_id="10000",
        ),
    )
    monkeypatch.setattr(webapi, "request", SimpleNamespace(query={"binding_id": "b1"}))

    response = await AmbientWebApi(_plugin(buffer)).get_buffer()  # type: ignore[arg-type]
    payload = json.loads(response.body)

    rows = payload["sessions"][0]["messages"]
    assert [row["role"] for row in rows] == ["user", "user", "assistant"]
    assert [row["role_name"] for row in rows] == ["小明", "小红", "我"]
    # 名字与 ID 分两列：机器人不暴露自身 ID。
    assert [row["role_id"] for row in rows] == ["444066154", "123456", ""]
    # 文本沿用对话侧渲染：文字与附件占位都在。
    assert rows[1]["text"] == "哈哈 [表情]"
