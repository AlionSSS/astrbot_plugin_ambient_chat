"""会话图片的磁盘缓存。

上下文缓冲只在内存里保存文件路径，图片字节写到
``data/plugin_data/astrbot_plugin_ambient_chat/media/<会话摘要>/`` 下，
文件由上下文缓冲持有；生成中的借用与卸载收尾见 ``docs/adr/0030``。

落盘时会按「最长边 + JPEG 质量」重编码以压掉体积（允许损失画质），
见 ``docs/adr/0010``；小图或压缩反变更大时保留原文件。
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.star import StarTools

from ..common import PLUGIN_NAME

try:  # 核心依赖 Pillow；万一缺失就退化为「原样复制」。
    from PIL import Image as PILImage
    from PIL import ImageOps
except ImportError:  # pragma: no cover - 正常安装的 AstrBot 都带 Pillow
    PILImage = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

MEDIA_DIRNAME = "media"

MAX_IMAGES_PER_MESSAGE = 4
"""单条消息最多缓存几张图片，避免图片墙把磁盘写满。"""

MAX_SOURCE_BYTES = 64 * 1024 * 1024
"""源图体积上限，超过直接丢弃：再大的图解码会吃掉大量内存。"""

MAX_SOURCE_PIXELS = 50_000_000
"""源图像素上限（约 50 MP，覆盖主流手机直出）。

按像素而不是按长边判断：解码后的位图大小只跟像素数有关。超出直接丢弃，
既不压缩也不原样复制，避免为一张图吃掉几百 MiB 内存。
只读文件头即可拿到宽高，不会触发解码。
"""

COMPRESS_THRESHOLD_BYTES = 512 * 1024
"""小于此字节数、且长边未超限的图片原样复制，不做重编码。"""

DEFAULT_IMAGE_MAX_EDGE = 1280
"""落盘图片的最长边，与核心 ``image_compress_options.max_size`` 默认值一致。"""

MIN_IMAGE_MAX_EDGE = 320
MAX_IMAGE_MAX_EDGE = 2048

DEFAULT_IMAGE_QUALITY = 80
"""JPEG 重编码质量。"""

MIN_IMAGE_QUALITY = 40
MAX_IMAGE_QUALITY = 95

ALPHA_PNG_FALLBACK_BYTES = 1024 * 1024
"""带透明通道的图片重编码成 PNG 后仍大于此值，就铺白底转 JPEG。

体积只影响磁盘；真正影响多模态 token 的是像素尺寸，所以 1 MiB 以内且长边
已压到 ``max_edge`` 的 PNG 允许保留透明通道。
"""

OPAQUE_ALPHA_LEVEL = 250
"""alpha ≥ 该值的像素视为不透明。"""

OPAQUE_RATIO_THRESHOLD = 0.995
"""不透明像素占比达到该值就当「没有实际透明」，直接按不透明图处理。"""

CAPTURE_TIMEOUT_SECONDS = 20.0
"""单张图片「取回本地」的超时。"""

MAX_CONCURRENT_COMPRESSIONS = 2
"""同时在跑的解码/重编码数量上限。

``asyncio.to_thread`` 用的是全局默认线程池（最多 32 个线程），多个会话同时收到
大图时会一起解码：12MP 解码后位图约 36 MiB、48MP 约 144 MiB，很容易把内存打满。
压缩是后台任务，排队多花一点时间用户无感，所以闸门收紧到 2。
"""

_SUFFIXES_BY_FORMAT: dict[str, tuple[str, ...]] = {
    "JPEG": (".jpg", ".jpeg"),
    "PNG": (".png",),
    "WEBP": (".webp",),
    "GIF": (".gif",),
    "BMP": (".bmp",),
}
"""后缀与真实格式一致时，原文件才能直接沿用到缓存目录。"""


class MediaStore:
    """把消息里的图片转存到插件数据目录，返回本地绝对路径与字节数。"""

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_edge: int = DEFAULT_IMAGE_MAX_EDGE,
        quality: int = DEFAULT_IMAGE_QUALITY,
    ) -> None:
        self._root = root
        self._max_edge = _clamp(max_edge, MIN_IMAGE_MAX_EDGE, MAX_IMAGE_MAX_EDGE)
        self._quality = _clamp(quality, MIN_IMAGE_QUALITY, MAX_IMAGE_QUALITY)
        self._gate: asyncio.Semaphore | None = None
        self._gate_loop: asyncio.AbstractEventLoop | None = None
        self._writes: set[asyncio.Task] = set()

    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = StarTools.get_data_dir(PLUGIN_NAME) / MEDIA_DIRNAME
        return self._root

    def configure(self, *, max_edge: int, quality: int) -> None:
        """更新压缩参数，只影响之后转存的图片。"""
        self._max_edge = _clamp(max_edge, MIN_IMAGE_MAX_EDGE, MAX_IMAGE_MAX_EDGE)
        self._quality = _clamp(quality, MIN_IMAGE_QUALITY, MAX_IMAGE_QUALITY)

    def _throttle(self) -> asyncio.Semaphore:
        """压缩并发闸门：线程池是全局共享的，避免多会话同时解码大图。

        线程按事件循环懒创建：插件热重载会换循环，旧的 ``Semaphore`` 不能复用。
        """
        loop = asyncio.get_running_loop()
        if self._gate is None or self._gate_loop is not loop:
            self._gate = asyncio.Semaphore(MAX_CONCURRENT_COMPRESSIONS)
            self._gate_loop = loop
        return self._gate

    async def capture(
        self,
        umo: str,
        components: Sequence[Any],
    ) -> list[tuple[str, int]]:
        """转存一条消息里的图片，返回 ``(路径, 字节数)``，失败的图片直接跳过。"""
        saved: list[tuple[str, int]] = []
        try:
            for component in list(components)[:MAX_IMAGES_PER_MESSAGE]:
                item = await self._capture_one(umo, component)
                if item is not None:
                    saved.append(item)
        except BaseException:
            # 尚未交给缓冲的整批产物仍由文件 adapter 持有。
            self.discard(path for path, _ in saved)
            raise
        return saved

    async def _capture_one(self, umo: str, component: Any) -> tuple[str, int] | None:
        download = asyncio.create_task(_resolve_source(component))
        download.add_done_callback(_observe_download)
        try:
            # 外部下载即使不响应取消，也不能拖住停止/卸载；迟到路径不再落盘。
            done, _ = await asyncio.wait({download}, timeout=CAPTURE_TIMEOUT_SECONDS)
            if not done:
                raise TimeoutError("图片取回超时")
            source = download.result()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 图片取回失败，跳过：%s", exc)
            return None
        finally:
            if not download.done():
                download.cancel()
        if not source:
            return None

        source_path = Path(source)
        try:
            size = source_path.stat().st_size
        except OSError:
            return None
        if size <= 0 or size > MAX_SOURCE_BYTES:
            logger.debug(
                "[Ambient] 图片过大或为空（%d 字节），跳过：%s", size, source_path.name
            )
            return None

        target = self.root / _session_key(umo)
        stem = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        suffix = source_path.suffix.lower() or ".img"
        invalid = asyncio.Event()
        write = asyncio.create_task(
            self._write_one(source_path, target, stem, suffix, invalid)
        )
        self._writes.add(write)
        write.add_done_callback(self._writes.discard)
        try:
            # 取消取图请求不能释放实际仍在工作的压缩名额。
            destination = await asyncio.shield(write)
        except asyncio.CancelledError:
            invalid.set()
            if write.done() and not write.cancelled():
                result = write.result()
                if result is not None:
                    self.discard([str(result)])
            raise
        if destination is None:
            # 像素数超限：连原图都不留，消息只保留文本占位。
            return None
        try:
            written = destination.stat().st_size
        except OSError:
            written = size
        return str(destination), written

    async def _write_one(
        self,
        source: Path,
        target: Path,
        stem: str,
        suffix: str,
        invalid: asyncio.Event,
    ) -> Path | None:
        async with self._throttle():
            if invalid.is_set():
                return None
            try:
                destination = await asyncio.to_thread(
                    _store_image,
                    source,
                    target,
                    stem,
                    suffix,
                    self._max_edge,
                    self._quality,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Ambient] 图片落盘失败，本次不缓存：%s", exc)
                return None
            if invalid.is_set() and destination is not None:
                self.discard([str(destination)])
                return None
            return destination

    async def drain(self) -> None:
        """等待已经开始的文件写入及其取消清理，不再强行取消线程。"""
        while self._writes:
            await asyncio.gather(*self._writes)

    def discard(self, paths: Iterable[str]) -> None:
        """删除已不再被缓冲引用的图片。"""
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("[Ambient] 删除图片缓存失败（%s）：%s", path, exc)

    def purge_all(self) -> None:
        """清空整个图片缓存目录（缓冲只在内存，重启后没有任何引用）。"""
        root = self.root
        if not root.exists():
            return
        shutil.rmtree(root, ignore_errors=True)
        logger.info("[Ambient] 已清理图片缓存目录：%s", root)


def _store_image(
    source: Path,
    target: Path,
    stem: str,
    suffix: str,
    max_edge: int,
    quality: int,
) -> Path | None:
    """一个写盘工作的全部产物同生共灭，失败也不留下半成品。"""
    destination = None
    try:
        target.mkdir(parents=True, exist_ok=True)
        destination = _save_image(source, target, stem, suffix, max_edge, quality)
        return destination
    finally:
        if target.exists():
            for candidate in target.glob(f"{stem}.*"):
                if candidate != destination:
                    _unlink(candidate)


def _save_image(
    source: Path,
    target_dir: Path,
    stem: str,
    suffix: str,
    max_edge: int,
    quality: int,
) -> Path | None:
    """把源图写到缓存目录：需要时缩放 + 重编码，否则原样复制。

    返回 ``None`` 表示图片超大（像素数超出 ``MAX_SOURCE_PIXELS``），应当直接丢弃。
    """
    destination = target_dir / f"{stem}{suffix}"
    size = source.stat().st_size
    if PILImage is None:
        shutil.copyfile(source, destination)
        return destination

    trustworthy = False
    try:
        with PILImage.open(source) as image:
            width, height = image.size
            if width * height > MAX_SOURCE_PIXELS:
                logger.debug(
                    "[Ambient] 图片尺寸过大（%dx%d），丢弃：%s",
                    width,
                    height,
                    source.name,
                )
                return None
            # 后缀与真实格式一致时，原文件才能直接沿用到缓存目录。
            trustworthy = suffix in _SUFFIXES_BY_FORMAT.get(image.format or "", ())
            if (
                trustworthy
                and size <= COMPRESS_THRESHOLD_BYTES
                and max(width, height) <= max_edge
            ):
                shutil.copyfile(source, destination)
                return destination
            compressed = _compress(image, target_dir, stem, max_edge, quality)
    except Exception as exc:  # noqa: BLE001
        # 格式不认识 / 解码失败：退回原样复制，至少图片还能用。
        logger.debug("[Ambient] 图片压缩失败，保留原图：%s（%s）", source.name, exc)
        shutil.copyfile(source, destination)
        return destination

    compressed_size = _file_size(compressed)
    if compressed_size > 0 and (compressed_size < size or not trustworthy):
        return compressed
    # 重编码反而更大（例如小尺寸高质量 JPEG）：留原图更划算。
    _unlink(compressed)
    shutil.copyfile(source, destination)
    return destination


def _compress(
    image: Any,
    target_dir: Path,
    stem: str,
    max_edge: int,
    quality: int,
) -> Path:
    """缩放并按模式选择输出格式，返回新文件路径（可能与 ``destination`` 同名）。"""
    if getattr(image, "is_animated", False):
        # 动图只留首帧：注入给模型的也只是静态画面。
        image.seek(0)
    if image.format == "JPEG":
        # JPEG 支持在解码阶段按 1/2、1/4、1/8 取样：先降采样再解码，省掉大部分像素运算。
        # 必须在任何 load() 之前调用，所以放在 exif_transpose 前面（后者会触发解码）。
        try:
            image.draft("RGB", (max_edge, max_edge))
        except (OSError, ValueError):
            pass
    if ImageOps is not None:
        # 先摆正方向，否则缩图后仍会按 EXIF 旋转，核心也会当成「被旋转的图」。
        image = ImageOps.exif_transpose(image) or image

    icc_profile = image.info.get("icc_profile")
    alpha = _has_alpha(image)
    image = image.convert("RGBA" if alpha else "RGB")
    image.thumbnail((max_edge, max_edge), PILImage.LANCZOS)

    if not alpha or _is_effectively_opaque(image):
        # 透明通道只是摆设（安卓截图、PNG 转码常见）：直接 JPEG，省掉一次 PNG 编码。
        return _write(
            image.convert("RGB"),
            target_dir / f"{stem}.jpg",
            "JPEG",
            quality,
            icc_profile,
        )

    png_path = _write(image, target_dir / f"{stem}.png", "PNG", quality, icc_profile)
    png_size = _file_size(png_path)
    if 0 < png_size <= ALPHA_PNG_FALLBACK_BYTES:
        return png_path

    # 透明通道的「照片型」大图：PNG 压不动，铺白底转 JPEG。
    flattened = PILImage.new("RGB", image.size, (255, 255, 255))
    flattened.paste(image, mask=image.getchannel("A"))
    jpeg_path = _write(
        flattened, target_dir / f"{stem}.jpg", "JPEG", quality, icc_profile
    )
    jpeg_size = _file_size(jpeg_path)
    if jpeg_size and (png_size <= 0 or jpeg_size < png_size):
        _unlink(png_path)
        return jpeg_path
    _unlink(jpeg_path)
    return png_path


def _write(
    image: Any,
    destination: Path,
    image_format: str,
    quality: int,
    icc_profile: Any,
) -> Path:
    params: dict[str, Any] = {"optimize": True}
    if icc_profile:
        params["icc_profile"] = icc_profile
    if image_format == "JPEG":
        params["quality"] = quality
        params["progressive"] = True
    image.save(destination, image_format, **params)
    return destination


def _has_alpha(image: Any) -> bool:
    """判断图片是否真的带透明通道（调色板图要看 transparency 信息）。"""
    if image.mode in ("RGBA", "LA", "PA"):
        return True
    return image.mode == "P" and "transparency" in image.info


def _is_effectively_opaque(image: Any) -> bool:
    """透明通道是否存在「实际透明」的像素；全不透明的 RGBA 不值得按 PNG 存。

    探针是一次 256 桶直方图，1280×1024 上约 1.9 ms，对比省下的 PNG 编码（~180 ms）很值。
    """
    try:
        alpha = image.getchannel("A")
        histogram = alpha.histogram()
    except (OSError, ValueError):
        return False
    total = image.size[0] * image.size[1]
    if total <= 0:
        return False
    opaque = sum(histogram[OPAQUE_ALPHA_LEVEL:])
    return opaque / total >= OPAQUE_RATIO_THRESHOLD


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _clamp(value: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


async def _resolve_source(component: Any) -> str:
    """把消息组件解析为本地文件路径；网络图片会先下载到核心临时目录。"""
    local = getattr(component, "path", None)
    if local and Path(local).is_file():
        return str(local)
    convert = getattr(component, "convert_to_file_path", None)
    if convert is None:
        raise ValueError("消息组件不支持转换为本地文件")
    return await convert()


def _observe_download(task: asyncio.Task) -> None:
    """读取已分离的外部下载异常；其迟到结果不拥有插件图片文件。"""
    if not task.cancelled():
        task.exception()


def _session_key(umo: str) -> str:
    return hashlib.sha1(umo.encode("utf-8")).hexdigest()[:16]
