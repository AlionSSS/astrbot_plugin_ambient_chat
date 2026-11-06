"""语音缓存：随消息淘汰、开关与借用语义与图片一致。"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.media_store import MediaStore
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.speech.delivery import Delivery

UMO = "test:GroupMessage:1"
AUDIO_BYTES = b"amr-voice-bytes"


def configure(buffer, *, images=True, audio=True):
    buffer.configure_media(
        enabled=images, max_edge=1280, quality=80, audio_enabled=audio
    )


def message(index=1):
    return ChatMessage("小明", f"语音 {index}", index, kind="audio", attachments=("audio",))


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest_asyncio.fixture
async def buffer(tmp_path):
    result = ContextBuffer(maxlen=10, media=MediaStore(tmp_path / "media"))
    yield result
    await result.close()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "voice.amr"
    path.write_bytes(AUDIO_BYTES)
    return SimpleNamespace(path=str(path))


def files(buffer):
    return list(buffer.media_root.rglob("*.*"))


@pytest.mark.asyncio
async def test_audio_cache_is_off_by_default(buffer, source):
    entry = message()

    buffer.append(UMO, entry, audio_components=[source])
    await asyncio.sleep(0.01)

    assert not entry.audio
    assert not files(buffer)


@pytest.mark.asyncio
async def test_captured_audio_is_released_with_message(buffer, source):
    configure(buffer)
    entry = message()

    buffer.append(UMO, entry, audio_components=[source])
    await until(lambda: bool(entry.audio))

    assert entry.audio_bytes == len(AUDIO_BYTES)
    assert len(files(buffer)) == 1

    for index in range(2, 12):
        buffer.append(UMO, message(index))

    assert not entry.audio
    assert not files(buffer)


@pytest.mark.asyncio
async def test_borrowed_audio_survives_eviction_until_request_returns(buffer, source):
    configure(buffer)
    entry = message()
    buffer.append(UMO, entry, audio_components=[source])
    await until(lambda: bool(entry.audio))
    paths = list(entry.audio)

    with buffer.borrow_audio(buffer.recent(UMO, 10), limit=2) as borrowed:
        assert borrowed == paths
        for index in range(2, 12):
            buffer.append(UMO, message(index))
        assert Path(paths[0]).is_file()

    assert not Path(paths[0]).is_file()


@pytest.mark.asyncio
async def test_disabling_audio_clears_cached_files_without_touching_images(
    buffer, source
):
    configure(buffer)
    entry = message()
    buffer.append(UMO, entry, audio_components=[source])
    await until(lambda: bool(entry.audio))

    configure(buffer, images=True, audio=False)

    assert not entry.audio
    assert not files(buffer)


@pytest.mark.asyncio
async def test_bot_voice_is_not_cached(buffer, source):
    """机器人自己的语音不进媒体缓存：TTS 原文已在正文里，不需要再听一遍。"""
    configure(buffer)
    entry = ChatMessage("我", "自己说的", 1, is_bot=True, kind="audio", attachments=("audio",))

    buffer.append(UMO, entry, audio_components=[source])
    await asyncio.sleep(0.01)

    assert not entry.audio
    assert not files(buffer)


class _MediaContext:
    """只回答「模型支持哪些模态」并记录生成请求。"""

    def __init__(self, modalities):
        self.kwargs = {}
        self._modalities = modalities
        self.conversation_manager = SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="c1")
        )

    def get_config(self):
        return {}

    def get_provider_by_id(self, _provider_id):
        return SimpleNamespace(provider_config={"modalities": self._modalities})

    async def llm_generate(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(completion_text="好呀")


@pytest.mark.asyncio
async def test_prepare_attaches_recent_audio_for_audio_capable_provider(buffer, source):
    configure(buffer)
    context = _MediaContext(["audio"])
    delivery = Delivery(context, buffer)
    entry = message(1)
    buffer.append(UMO, entry, audio_components=[source])
    await until(lambda: bool(entry.audio))
    rule = SimpleNamespace(chat_provider_id="p1", context_turns=20)

    speech = await delivery.prepare(
        rule,
        UMO,
        "接一句",
        "",
        datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert speech is not None
    assert context.kwargs["audio_urls"] == [entry.audio[0]]


@pytest.mark.asyncio
async def test_prepare_skips_audio_for_text_only_provider(buffer, source):
    configure(buffer)
    context = _MediaContext(["image"])
    delivery = Delivery(context, buffer)
    entry = message(1)
    buffer.append(UMO, entry, audio_components=[source])
    await until(lambda: bool(entry.audio))
    rule = SimpleNamespace(chat_provider_id="p1", context_turns=20)

    await delivery.prepare(
        rule,
        UMO,
        "接一句",
        "",
        datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert context.kwargs["audio_urls"] is None


class _FlakyContext(_MediaContext):
    """记录每次请求携带的媒体；前 ``failures`` 次直接抛错。"""

    def __init__(self, modalities, failures):
        super().__init__(modalities)
        self._failures = failures
        self.requests = []

    async def llm_generate(self, **kwargs):
        self.requests.append(
            (
                list(kwargs.get("image_urls") or []),
                list(kwargs.get("audio_urls") or []),
            )
        )
        if len(self.requests) <= self._failures:
            raise RuntimeError("模型拒绝了这次媒体请求")
        return SimpleNamespace(completion_text="好呀")


@pytest.mark.asyncio
async def test_generation_sheds_voice_then_images(buffer, source, tmp_path):
    """带图带语音失败时先只丢语音，图片也失败才退到纯文本。"""
    configure(buffer)
    context = _FlakyContext(["image", "audio"], failures=2)
    delivery = Delivery(context, buffer)
    picture = tmp_path / "photo.jpg"
    picture.write_bytes(b"local image bytes")
    entry = message(1)
    buffer.append(
        UMO,
        entry,
        image_components=[SimpleNamespace(path=str(picture))],
        audio_components=[source],
    )
    await until(lambda: bool(entry.audio) and bool(entry.images))

    speech = await delivery.prepare(
        SimpleNamespace(chat_provider_id="p1", context_turns=20),
        UMO,
        "接一句",
        "",
        datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert speech is not None
    assert [bool(images) for images, _ in context.requests] == [True, True, False]
    assert [bool(audio) for _, audio in context.requests] == [True, False, False]


@pytest.mark.asyncio
async def test_generation_with_only_voice_retries_once_without_audio(buffer, source):
    configure(buffer)
    context = _FlakyContext(["audio"], failures=1)
    delivery = Delivery(context, buffer)
    entry = message(1)
    buffer.append(UMO, entry, audio_components=[source])
    await until(lambda: bool(entry.audio))

    speech = await delivery.prepare(
        SimpleNamespace(chat_provider_id="p1", context_turns=20),
        UMO,
        "接一句",
        "",
        datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert speech is not None
    assert len(context.requests) == 2
    assert context.requests[0][1]
    assert not context.requests[1][0] and not context.requests[1][1]
