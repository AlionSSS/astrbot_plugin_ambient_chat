"""bot 自己的发言回填：语音/图片不能被记成正文里的 ``[Record]`` / ``[Image]``。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin

from astrbot.api.message_components import Image, Plain, Record

UMO = "aiocqhttp:GroupMessage:1"


class _Lifecycle:
    def __init__(self) -> None:
        self.recorded = []

    def accepts(self, *_args, **_kwargs) -> bool:
        return True

    def record(self, _umo, message, **_kwargs) -> bool:
        self.recorded.append(message)
        return True


async def _record(chain: list, *, plain_text: str = ""):
    lifecycle = _Lifecycle()
    plugin = SimpleNamespace(lifecycle=lifecycle, _self_name=lambda _umo: "蓝色小肥鱼")
    result = SimpleNamespace(chain=chain, get_plain_text=lambda: plain_text)
    event = SimpleNamespace(
        unified_msg_origin=UMO,
        get_result=lambda: result,
        message_obj=SimpleNamespace(self_id="2670210348"),
    )
    await AmbientChatPlugin.record_bot_message(plugin, event)
    assert lifecycle.recorded
    return lifecycle.recorded[-1]


@pytest.mark.asyncio
async def test_bot_voice_is_audio_not_literal_record_text() -> None:
    """发语音时正文里不能出现组件名 ``[Record]``，形态要落成 audio。"""
    message = await _record([Record(file="a.wav"), Record(file="b.wav")])

    assert message.kind == "audio"
    assert message.attachments == ("audio", "audio")
    assert message.text == ""
    assert "[Record]" not in message.render_text()
    assert "[语音]" in message.render_text()


@pytest.mark.asyncio
async def test_bot_voice_keeps_tts_source_text() -> None:
    """TTS 语音自带原文时保留下来：决策模型既知道是语音，也知道说了什么。"""
    message = await _record([Record(file="a.wav", text="唱完啦")])

    assert message.kind == "audio"
    assert message.text == "唱完啦"
    assert message.body_text() == "唱完啦"


@pytest.mark.asyncio
async def test_bot_tts_dual_output_keeps_text_once() -> None:
    """同时发语音与文字（TTS 双发）时，同一段文字只留一份。"""
    message = await _record([Record(file="a.wav", text="在吗"), Plain("在吗")])

    assert message.kind == "audio"
    assert message.text == "在吗"


@pytest.mark.asyncio
async def test_bot_image_is_image_type() -> None:
    message = await _record([Image(file="a.png")])

    assert message.kind == "image"
    assert message.text == ""


@pytest.mark.asyncio
async def test_result_without_chain_falls_back_to_plain_text() -> None:
    """没有消息链的结果对象仍按纯文本记录，保持旧行为。"""
    lifecycle = _Lifecycle()
    plugin = SimpleNamespace(lifecycle=lifecycle, _self_name=lambda _umo: "我")
    result = SimpleNamespace(get_plain_text=lambda: "普通回复")
    event = SimpleNamespace(
        unified_msg_origin=UMO,
        get_result=lambda: result,
        message_obj=SimpleNamespace(self_id="1"),
    )

    await AmbientChatPlugin.record_bot_message(plugin, event)

    assert lifecycle.recorded[-1].text == "普通回复"
    assert lifecycle.recorded[-1].kind == "text"
