"""通过缓冲、转存与生成 interface 验证文件所有权，不连接真实平台。"""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from astrbot_plugin_ambient_chat.conversation import context_buffer, media_store
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.media_store import MediaStore
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import Rule
from astrbot_plugin_ambient_chat.speech.delivery import Delivery

UMO = "test:GroupMessage:1"


def configure(buffer, enabled=True):
    buffer.configure_media(enabled=enabled, max_edge=1280, quality=80)


def message(index=1):
    return ChatMessage("小明", f"图片 {index}", index, attachments=("image",))


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest_asyncio.fixture
async def buffer(tmp_path):
    result = ContextBuffer(maxlen=10, media=MediaStore(tmp_path / "media"))
    configure(result)
    yield result
    await result.close()


@pytest.fixture
def source(tmp_path):
    # 解码失败时原样复制也是既有的文件 adapter 路径。
    result = tmp_path / "source.img"
    result.write_bytes(b"local image bytes")
    return SimpleNamespace(path=str(result))


def files(buffer):
    return list(buffer.media_root.rglob("*.*"))


async def add_image(buffer, source, index=1):
    entry = message(index)
    buffer.append(UMO, entry, image_components=[source])
    await until(lambda: bool(entry.images))
    return entry


class SlowSource:
    def __init__(self, path, *, ignores_cancel=False):
        self.local = str(path)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.ignores_cancel = ignores_cancel

    async def convert_to_file_path(self):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            if not self.ignores_cancel:
                raise
            await self.release.wait()
        finally:
            self.finished.set()
        return self.local


@pytest.mark.asyncio
async def test_cancel_multiphoto_capture_reclaims_already_written_files(
    tmp_path, source
):
    store = MediaStore(tmp_path / "media")
    slow = SlowSource(source.path)
    capture = asyncio.create_task(store.capture(UMO, [source, slow]))
    await slow.entered.wait()
    assert list(store.root.rglob("*.*"))
    capture.cancel()
    with pytest.raises(asyncio.CancelledError):
        await capture
    await store.drain()
    assert not list(store.root.rglob("*.*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["clear", "disable", "evict", "resize", "preheat"])
async def test_invalidated_capture_cannot_attach_after_late_download(
    buffer, source, change
):
    slow = SlowSource(source.path, ignores_cancel=True)
    entry = message()
    if change == "resize":
        buffer.set_maxlen(20)
    buffer.append(UMO, entry, image_components=[slow])
    await slow.entered.wait()
    if change == "clear":
        buffer.clear(UMO)
    elif change == "disable":
        configure(buffer, False)
        configure(buffer, True)
    elif change in {"evict", "resize"}:
        for index in range(2, 12):
            buffer.append(UMO, message(index))
        if change == "resize":
            buffer.set_maxlen(10)
    else:
        buffer.publish_history({UMO: [message(2)]})
    slow.release.set()
    await slow.finished.wait()
    await buffer.close()
    assert not entry.images
    assert not files(buffer)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["clear", "disable", "evict", "quota"])
async def test_borrow_survives_retirement_until_last_request_returns(
    buffer, source, monkeypatch, change
):
    entry = await add_image(buffer, source)
    path = Path(entry.images[0])
    with buffer.borrow_images([entry], limit=8) as first:
        with buffer.borrow_images([entry], limit=8) as second:
            assert first == second == [str(path)]
            if change == "clear":
                buffer.clear(UMO)
            elif change == "disable":
                configure(buffer, False)
            elif change == "evict":
                for index in range(2, 12):
                    buffer.append(UMO, message(index))
            else:
                monkeypatch.setattr(context_buffer, "MAX_MEDIA_BYTES_PER_SESSION", 0)
                buffer.append(UMO, message(2))
            assert not entry.images
            assert path.read_bytes() == Path(source.path).read_bytes()
            with buffer.borrow_images([entry], limit=8) as new_request:
                assert new_request == []
        assert path.exists()
    assert not path.exists()


@pytest.mark.asyncio
async def test_close_waits_for_borrow_but_prevents_new_capture(buffer, source):
    entry = await add_image(buffer, source)
    with buffer.borrow_images([entry], limit=8) as paths:
        close = asyncio.create_task(buffer.close())
        await until(lambda: not buffer.has(UMO))
        assert not close.done()
        assert Path(paths[0]).exists()
        buffer.append(UMO, message(2), image_components=[source])
        assert not buffer.has(UMO)
    await close
    assert not buffer.media_root.exists()
    await buffer.close()  # 卸载收尾可以重复等待。


@pytest.mark.asyncio
async def test_close_does_not_wait_for_download_that_ignores_cancellation(
    buffer, source
):
    slow = SlowSource(source.path, ignores_cancel=True)
    entry = message()
    buffer.append(UMO, entry, image_components=[slow])
    await slow.entered.wait()
    try:
        async with asyncio.timeout(1):
            await buffer.close()
        assert not slow.finished.is_set()
        assert not buffer.media_root.exists()
    finally:
        slow.release.set()
        await slow.finished.wait()
    assert not entry.images
    assert not buffer.media_root.exists()


class Writer:
    """控制实际线程的写入，检查取消不会提前释放压缩名额。"""

    def __init__(self):
        self.release = threading.Event()
        self.entered = asyncio.Event()
        self.loop = asyncio.get_running_loop()
        self.count = 0

    def __call__(self, source, target, stem, suffix, max_edge, quality):
        self.count += 1
        self.loop.call_soon_threadsafe(self.entered.set)
        if not self.release.wait(3):
            raise TimeoutError("test writer was not released")
        destination = target / f"{stem}.jpg"
        destination.write_bytes(b"encoded image")
        return destination


@pytest.mark.asyncio
async def test_close_waits_for_actual_writer_and_reclaims_late_output(
    buffer, source, monkeypatch
):
    writer = Writer()
    monkeypatch.setattr(media_store, "_save_image", writer)
    entry = message()
    buffer.append(UMO, entry, image_components=[source])
    try:
        await writer.entered.wait()
        buffer.clear(UMO)  # 停止不等线程，但它的迟到产物必须有人回收。
        close = asyncio.create_task(buffer.close())
        await asyncio.sleep(0)
        assert not close.done()
    finally:
        writer.release.set()
    await close
    assert not entry.images
    assert not buffer.media_root.exists()


@pytest.mark.asyncio
async def test_cancelled_write_keeps_compression_slot_until_thread_finishes(
    tmp_path, source, monkeypatch
):
    monkeypatch.setattr(media_store, "MAX_CONCURRENT_COMPRESSIONS", 1)
    writer = Writer()
    monkeypatch.setattr(media_store, "_save_image", writer)
    store = MediaStore(tmp_path / "media")
    first = asyncio.create_task(store.capture(UMO, [source]))
    second = None
    try:
        await writer.entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(store.capture(UMO, [source]))
        # 足够让第二次请求经过下载进入压缩排队；不应开始第二个线程。
        for _ in range(10):
            await asyncio.sleep(0)
        assert writer.count == 1
    finally:
        writer.release.set()
        if second is not None:
            result = await second
            store.discard(path for path, _ in result)
        await store.drain()
    assert writer.count == 2
    assert not list(store.root.rglob("*.*"))


@pytest.mark.asyncio
async def test_writer_failure_reclaims_partial_outputs(buffer, source, monkeypatch):
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    def failing(source, target, stem, *args):
        (target / f"{stem}.png").write_bytes(b"partial")
        loop.call_soon_threadsafe(finished.set)
        raise OSError("disk full")

    monkeypatch.setattr(media_store, "_save_image", failing)
    entry = message()
    buffer.append(UMO, entry, image_components=[source])
    await finished.wait()
    await buffer.close()
    assert not entry.images
    assert not files(buffer)


@pytest.mark.asyncio
async def test_cancelled_close_request_does_not_abandon_writer_cleanup(
    buffer, source, monkeypatch
):
    writer = Writer()
    monkeypatch.setattr(media_store, "_save_image", writer)
    buffer.append(UMO, message(), image_components=[source])
    close = None
    try:
        await writer.entered.wait()
        close = asyncio.create_task(buffer.close())
        await until(lambda: not buffer.has(UMO))
        close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close
    finally:
        writer.release.set()
        await buffer.close()
    assert not buffer.media_root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancel", "failure"])
async def test_delivery_holds_images_only_for_the_active_model_request(
    buffer, source, outcome
):
    entry = await add_image(buffer, source)
    path = Path(entry.images[0])
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs.get("image_urls"))
        if len(calls) == 1:
            assert calls[0] == [str(path)]
            entered.set()
            await release.wait()
            assert path.exists()
            if outcome == "failure":
                raise RuntimeError("vision unavailable")
        else:
            assert not path.exists()  # 纯文本重试不再借用图片。
        return SimpleNamespace(completion_text="看到了")

    context = SimpleNamespace(
        llm_generate=generate,
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="cid")
        ),
    )
    delivery = Delivery(context, buffer)
    rule = Rule("r_test", "规则", chat_provider_id="p1")
    task = asyncio.create_task(delivery.prepare(rule, UMO, "接话", now=None))
    try:
        await entered.wait()
        configure(buffer, False)
        assert not entry.images
        assert path.exists()
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            assert (await task).text == "看到了"
        assert not path.exists()
        if outcome == "failure":
            assert calls == [[str(path)], None]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "modalities,expected",
    [(["text"], False), (["image", "text"], True), ([], True), (None, True)],
)
async def test_delivery_reads_provider_image_capability_each_request(
    buffer, source, modalities, expected
):
    entry = await add_image(buffer, source)
    config = {"modalities": modalities}
    generate = AsyncMock(return_value=SimpleNamespace(completion_text="收到"))
    context = SimpleNamespace(
        get_provider_by_id=lambda _: SimpleNamespace(provider_config=config),
        llm_generate=generate,
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="cid")
        ),
    )
    delivery = Delivery(context, buffer)
    rule = Rule("r_test", "规则", chat_provider_id="p1")
    await delivery.prepare(rule, UMO, "接话")
    assert generate.call_args.kwargs["image_urls"] == (
        list(entry.images) if expected else None
    )
    config["modalities"] = ["text"] if expected else ["text", "image"]
    await delivery.prepare(rule, UMO, "接话")
    assert generate.call_args.kwargs["image_urls"] == (
        None if expected else list(entry.images)
    )


@pytest.mark.asyncio
async def test_transient_image_failure_does_not_disable_future_images(buffer, source):
    entry = await add_image(buffer, source)
    generate = AsyncMock(
        side_effect=[
            RuntimeError("temporary outage"),
            SimpleNamespace(completion_text="重试"),
            SimpleNamespace(completion_text="看到了"),
        ]
    )
    context = SimpleNamespace(
        get_provider_by_id=lambda _: SimpleNamespace(
            provider_config={"modalities": ["text", "image"]}
        ),
        llm_generate=generate,
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="cid")
        ),
    )
    delivery = Delivery(context, buffer)
    rule = Rule("r_test", "规则", chat_provider_id="p1")
    assert (await delivery.prepare(rule, UMO, "接话")).text == "重试"
    assert (await delivery.prepare(rule, UMO, "接话")).text == "看到了"
    assert [call.kwargs["image_urls"] for call in generate.call_args_list] == [
        list(entry.images),
        None,
        list(entry.images),
    ]
