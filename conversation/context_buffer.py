"""按目标会话维护的近期消息缓冲。

决策模型与对话模型都以它作为「最近 N 条消息」的来源，
见 ``docs/adr/0004`` 与 ``CONTEXT.md`` 中的「上下文缓冲」「启动预热」。

图片与语音不放在内存里：消息只带本地文件路径，字节落在插件数据目录，
消息被淘汰时释放媒体；生成中借用的文件延迟到请求结束回收（ADR-0030）。
语音缓存默认关闭，条数不单独设限——跟着缓冲条数一起进退（ADR-0031）。
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import tzinfo as _tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..common import parse_target
from . import history
from .media_store import MediaStore
from .messages import ChatMessage

if TYPE_CHECKING:
    from astrbot.api.star import Context

MAX_RENDER_LINES = 200

MAX_MEDIA_BYTES_PER_SESSION = 64 * 1024 * 1024
"""单个会话的图片磁盘占用上限，超出后从最旧的图片开始删。"""

PREHEAT_CONCURRENCY = 4
"""预热时并发读取的会话数：绑定可能有多个目标会话，少量并发避免拖慢启动请求。"""


@dataclass
class _Activity:
    last_bot_at: float | None = None
    streak_floor_at: float | None = None
    """「连发」计数的时间下限：不晚于它的 bot 消息已不计入连发（退避期满后重置）。"""
    unread_by_sender: Counter[str] = field(default_factory=Counter)

    def record(self, message: ChatMessage) -> None:
        if message.is_bot:
            if message.timestamp > 0:
                self.last_bot_at = message.timestamp
            self.unread_by_sender.clear()
        else:
            self.unread_by_sender[message.sender_id.strip()] += 1


class ContextBuffer:
    """按 UMO 分组的环形缓冲。"""

    def __init__(
        self,
        maxlen: int = 100,
        media: MediaStore | None = None,
        *,
        bot_name: Callable[[str], str] | None = None,
    ) -> None:
        self._maxlen = max(10, int(maxlen))
        self._bot_name = bot_name
        """预热时给机器人自己的发言补署名；核心历史把名字写成 ``bot``。"""
        self._buffers: dict[str, deque[ChatMessage]] = {}
        self._activity: dict[str, _Activity] = {}
        """冷却与未读基线独立于消息容量；按发送者计数以支持事后过滤。"""
        self._session_names: dict[str, str] = {}
        """会话的可读名称（群名），来自运行期事件，供判定请求的 scene 使用。"""
        self._ignored_senders: frozenset[str] = frozenset()
        """「更多设置」里过滤的发送者 ID：其消息既不进缓冲，也不参与预热。"""
        self._media = media or MediaStore()
        self._media_enabled = False
        self._audio_enabled = False
        self._media_tasks: set[asyncio.Task] = set()
        self._captures: dict[int, asyncio.Task] = {}
        self._audio_captures: dict[int, asyncio.Task] = {}
        self._pins: Counter[str] = Counter()
        self._retired: set[str] = set()
        self._pins_idle = asyncio.Event()
        self._pins_idle.set()
        self._closed = False
        self._close_task: asyncio.Task | None = None

    @property
    def media_root(self) -> Path:
        return self._media.root

    def configure_media(
        self,
        *,
        enabled: bool,
        max_edge: int,
        quality: int,
        audio_enabled: bool = False,
    ) -> None:
        """关闭时使对应媒体的旧转存失效；重新打开不复活旧消息的转存任务。

        图片与语音各有一个开关：关掉哪一个就只清哪一类，互不牵连。
        """
        self._media.configure(max_edge=max_edge, quality=quality)
        self._media_enabled = enabled and not self._closed
        self._audio_enabled = audio_enabled and not self._closed
        if not self._media_enabled:
            self._clear_images()
        if not self._audio_enabled:
            self._clear_audio()

    @property
    def ignored_senders(self) -> frozenset[str]:
        return self._ignored_senders

    def set_ignored_senders(self, sender_ids: Sequence[str] | frozenset[str]) -> None:
        """设置要过滤的发送者 ID；变更时清掉缓冲里已有的这些人的消息。

        只过滤用户消息：机器人自己的发言仍留在缓冲，冷却与未读基线保持完整。
        """
        ignored = frozenset(
            str(item).strip() for item in sender_ids if str(item).strip()
        )
        if ignored == self._ignored_senders:
            return
        self._ignored_senders = ignored
        for activity in self._activity.values():
            for sender_id in ignored:
                activity.unread_by_sender.pop(sender_id, None)
        removed = 0
        for umo, buffer in list(self._buffers.items()):
            kept = [message for message in buffer if not self._is_ignored(message)]
            if len(kept) == len(buffer):
                continue
            for message in buffer:
                if self._is_ignored(message):
                    self._discard(message)
            self._buffers[umo] = deque(kept, maxlen=self._maxlen)
            removed += len(buffer) - len(kept)
        logger.info(
            "[Ambient] 上下文缓冲过滤 %d 个用户 ID，已清理 %d 条旧消息。",
            len(ignored),
            removed,
        )

    def _is_ignored(self, message: ChatMessage) -> bool:
        return (
            bool(self._ignored_senders)
            and not message.is_bot
            and message.sender_id.strip() in self._ignored_senders
        )

    def set_session_name(self, umo: str, name: str) -> None:
        """记录会话名称；空串忽略，绝不覆盖已有名称成空。"""
        name = name.strip()
        if name:
            self._session_names[umo] = name

    def session_name(self, umo: str) -> str:
        return self._session_names.get(umo, "")

    def _buffer(self, umo: str) -> deque[ChatMessage]:
        buffer = self._buffers.get(umo)
        if buffer is None:
            buffer = deque(maxlen=self._maxlen)
            self._buffers[umo] = buffer
        return buffer

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def set_maxlen(self, maxlen: int) -> None:
        """调整容量（「更多设置」里的每会话保留条数），保留已有消息。"""
        size = max(10, int(maxlen))
        if size == self._maxlen:
            return
        self._maxlen = size
        for umo, buffer in list(self._buffers.items()):
            messages = list(buffer)
            for message in messages[:-size]:
                self._discard(message)
            self._buffers[umo] = deque(messages[-size:], maxlen=size)
        logger.info("[Ambient] 上下文缓冲容量已调整为 %d 条/会话。", size)

    def append(
        self,
        umo: str,
        message: ChatMessage,
        *,
        image_components: Sequence[Any] = (),
        audio_components: Sequence[Any] = (),
    ) -> None:
        if self._closed or self._is_ignored(message):
            return
        buffer = self._buffer(umo)
        last = buffer[-1] if buffer else None
        if (
            last is not None
            and last.is_bot == message.is_bot
            and last.sender_name == message.sender_name
            and abs(last.timestamp - message.timestamp) < 1e-6
            and last.text == message.text
        ):
            return
        self._push(buffer, message)
        self._adopt_private_name(umo, buffer, message)
        self._activity.setdefault(umo, _Activity()).record(message)
        self._trim_media(buffer)
        if image_components and self._media_enabled and not message.is_bot:
            task = asyncio.create_task(
                self._capture_media(umo, message, tuple(image_components))
            )
            self._captures[id(message)] = task
            self._media_tasks.add(task)
            task.add_done_callback(self._media_tasks.discard)
        if audio_components and self._audio_enabled and not message.is_bot:
            task = asyncio.create_task(
                self._capture_audio(umo, message, tuple(audio_components))
            )
            self._audio_captures[id(message)] = task
            self._media_tasks.add(task)
            task.add_done_callback(self._media_tasks.discard)

    def _adopt_private_name(
        self, umo: str, buffer: deque[ChatMessage], message: ChatMessage
    ) -> None:
        """私聊只有一个对方：学到昵称后回填该会话里还没名字的旧消息。

        私聊预热读的是核心 LLM 会话历史，本身不带发送者，条目只能渲染成「某人」
        （见 ADR-0020）。等对方发来下一条消息才第一次拿到昵称，此时把旧条目一起
        补上；否则同一段对话里前面几条是「某人」、后面几条是真名，模型会当成两个人。
        """
        if message.is_bot:
            return
        name = message.sender_name.strip()
        if not name or name == message.sender_id.strip():
            return
        target = parse_target(umo)
        if target is None or target.is_group:
            # 群聊里的「没名字」不能归给刚发言的这个人。
            return
        for item in buffer:
            if not item.is_bot and not item.sender_name.strip():
                item.sender_name = name

    async def _capture_media(
        self, umo: str, message: ChatMessage, components: Sequence[Any]
    ) -> None:
        captured: list[tuple[str, int]] = []
        task = asyncio.current_task()
        try:
            captured = await self._media.capture(umo, components)
            if (
                self._captures.get(id(message)) is not task
                or not self._media_enabled
                or not self._contains(umo, message)
            ):
                return
            message.images = tuple(path for path, _ in captured)
            message.image_bytes = sum(size for _, size in captured)
            captured = []  # 所有权已经交给消息，淘汰和配额负责释放。
            self._trim_media(self._buffers[umo])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 图片缓存失败（%s）：%s", umo, exc)
        finally:
            self._media.discard(path for path, _ in captured)
            if self._captures.get(id(message)) is task:
                self._captures.pop(id(message))

    async def _capture_audio(
        self, umo: str, message: ChatMessage, components: Sequence[Any]
    ) -> None:
        captured: list[tuple[str, int]] = []
        task = asyncio.current_task()
        try:
            captured = await self._media.capture_audio(umo, components)
            if (
                self._audio_captures.get(id(message)) is not task
                or not self._audio_enabled
                or not self._contains(umo, message)
            ):
                return
            message.audio = tuple(path for path, _ in captured)
            message.audio_bytes = sum(size for _, size in captured)
            captured = []  # 所有权已经交给消息，淘汰和配额负责释放。
            self._trim_media(self._buffers[umo])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 语音缓存失败（%s）：%s", umo, exc)
        finally:
            self._media.discard(path for path, _ in captured)
            if self._audio_captures.get(id(message)) is task:
                self._audio_captures.pop(id(message))

    def _cancel_capture(self, message: ChatMessage) -> None:
        task = self._captures.pop(id(message), None)
        if task is not None and not task.cancelling():
            task.cancel()

    def _cancel_audio_capture(self, message: ChatMessage) -> None:
        task = self._audio_captures.pop(id(message), None)
        if task is not None and not task.cancelling():
            task.cancel()

    def _release_media(self, paths: Sequence[str]) -> None:
        """删除媒体文件；正在被生成借用的先记下，等请求退出再删。"""
        for path in paths:
            if self._pins[path]:
                self._retired.add(path)
            else:
                self._media.discard([path])

    @contextmanager
    def _borrow(self, paths: list[str]) -> Iterator[list[str]]:
        """借用一批媒体路径至请求退出；淘汰不删除借用中的文件。"""
        self._pins.update(paths)
        if paths:
            self._pins_idle.clear()
        try:
            yield list(paths)
        finally:
            for path in paths:
                self._pins[path] -= 1
                if not self._pins[path]:
                    del self._pins[path]
                    if path in self._retired:
                        self._retired.remove(path)
                        self._media.discard([path])
            if not self._pins:
                self._pins_idle.set()

    @contextmanager
    def borrow_images(
        self, messages: Sequence[ChatMessage], *, limit: int
    ) -> Iterator[list[str]]:
        """快照最近图片并借用至请求退出；淘汰不删除借用中的文件。"""
        paths = (
            [path for message in messages for path in message.images][-limit:]
            if self._media_enabled and limit > 0
            else []
        )
        with self._borrow(paths) as borrowed:
            yield borrowed

    @contextmanager
    def borrow_audio(
        self, messages: Sequence[ChatMessage], *, limit: int
    ) -> Iterator[list[str]]:
        """快照最近语音并借用至请求退出；条数由调用方按模型能力决定。"""
        paths = (
            [path for message in messages for path in message.audio][-limit:]
            if self._audio_enabled and limit > 0
            else []
        )
        with self._borrow(paths) as borrowed:
            yield borrowed

    def _push(self, buffer: deque[ChatMessage], message: ChatMessage) -> None:
        """写入缓冲，被挤出的消息连同磁盘媒体一起清掉。"""
        while len(buffer) >= self._maxlen:
            self._discard(buffer.popleft())
        buffer.append(message)

    def _discard(self, message: ChatMessage) -> None:
        self._cancel_capture(message)
        self._cancel_audio_capture(message)
        self._release_media(message.images)
        message.images = ()
        message.image_bytes = 0
        self._release_media(message.audio)
        message.audio = ()
        message.audio_bytes = 0

    def _trim_media(self, buffer: deque[ChatMessage]) -> None:
        """把该会话的媒体占用压到上限内：先删最旧的语音，再删最旧的图片。

        语音体积远小于图片，通常不会触发配额；放在前面只是保证超限时
        先牺牲最不占带宽价值的那类。条数本身不设上限——消息被挤出缓冲时
        媒体随 ``_discard`` 一起释放。
        """
        total = sum(message.image_bytes + message.audio_bytes for message in buffer)
        if total <= MAX_MEDIA_BYTES_PER_SESSION:
            return
        for message in buffer:
            while message.audio and total > MAX_MEDIA_BYTES_PER_SESSION:
                path = message.audio[0]
                size = _file_size(path)
                self._release_media([path])
                message.audio = message.audio[1:]
                message.audio_bytes = max(0, message.audio_bytes - size)
                total -= size
            if total <= MAX_MEDIA_BYTES_PER_SESSION:
                return
        for message in buffer:
            while message.images and total > MAX_MEDIA_BYTES_PER_SESSION:
                path = message.images[0]
                size = _file_size(path)
                self._release_media([path])
                message.images = message.images[1:]
                message.image_bytes = max(0, message.image_bytes - size)
                total -= size
            if total <= MAX_MEDIA_BYTES_PER_SESSION:
                return

    def _contains(self, umo: str, message: ChatMessage) -> bool:
        """判断某条消息是否仍在缓冲里（后台转存媒体完成时用）。"""
        buffer = self._buffers.get(umo)
        return buffer is not None and any(item is message for item in buffer)

    def has(self, umo: str) -> bool:
        return umo in self._buffers

    @property
    def tracked(self) -> int:
        return len(self._buffers)

    def recent(self, umo: str, count: int) -> list[ChatMessage]:
        """取该会话最近 ``count`` 条消息（上下文长度按条计，不是对话轮次）。"""
        buffer = self._buffers.get(umo)
        if not buffer:
            return []
        size = max(1, int(count))
        return list(buffer)[-size:]

    def known_name(self, umo: str, sender_id: str) -> str:
        """该发送者在缓冲里最近一次的可读昵称；只记得数字 ID 时返回空串。"""
        if not sender_id:
            return ""
        for message in reversed(self._buffers.get(umo, ())):
            if message.is_bot or message.sender_id != sender_id:
                continue
            name = message.sender_name.strip()
            if name and name != sender_id:
                return name
        return ""

    def last_bot_at(self, umo: str) -> float | None:
        """最近一条 bot 消息的时间；时间未知的消息（私聊预热历史）不算数，返回 None。"""
        activity = self._activity.get(umo)
        return activity.last_bot_at if activity else None

    def bot_streak(self, umo: str) -> int:
        """自最后一条非 bot 消息（或退避基线）以来，bot 连续发言的条数。"""
        buffer = self._buffers.get(umo)
        if not buffer:
            return 0
        activity = self._activity.get(umo)
        floor = activity.streak_floor_at if activity else None
        count = 0
        for message in reversed(buffer):
            if not message.is_bot:
                break
            if floor is not None and message.timestamp <= floor:
                break
            count += 1
        return count

    def rebase_bot_streak(self, umo: str) -> None:
        """把当前最后一条 bot 消息设为新基线，让「连发」从下一条发言重新累计。

        连发无人回应的退避期满时调用：更早的 bot 消息不再计入连发，
        于是需要重新连发满 N 条才会再次进入等待，而不是每 N 分钟才被
        放行一条（见 ``speech.blockers``）。
        """
        activity = self._activity.get(umo)
        if activity is None or activity.last_bot_at is None:
            return
        activity.streak_floor_at = activity.last_bot_at

    def unread_count(self, umo: str) -> int:
        """自 bot 最后一次发言以来收到的消息条数。"""
        activity = self._activity.get(umo)
        return activity.unread_by_sender.total() if activity else 0

    def clear(self, umo: str) -> None:
        buffer = self._buffers.pop(umo, None)
        self._activity.pop(umo, None)
        self._session_names.pop(umo, None)
        for message in buffer or ():
            self._discard(message)

    def clear_all(self) -> None:
        """清空所有会话的缓冲（含磁盘媒体）：全局开关关闭、重新开启前调用。"""
        for umo in list(self._buffers):
            self.clear(umo)

    def clear_media(self) -> None:
        """撤销转存并释放全部图片与语音；借用中的文件请求结束后删除。"""
        self._clear_images()
        self._clear_audio()

    def _clear_images(self) -> None:
        for buffer in self._buffers.values():
            for message in buffer:
                self._cancel_capture(message)
                self._release_media(message.images)
                message.images = ()
                message.image_bytes = 0

    def _clear_audio(self) -> None:
        for buffer in self._buffers.values():
            for message in buffer:
                self._cancel_audio_capture(message)
                self._release_media(message.audio)
                message.audio = ()
                message.audio_bytes = 0

    def reset_media(self) -> None:
        """仅在初始化、尚未接入消息前清空上次进程遗留的媒体。"""
        self._media.purge_all()

    async def close(self) -> None:
        """禁止新任务，等待实际写盘与借用结束，最后清空整个媒体目录。"""
        if self._close_task is None:
            self._closed = True
            self._media_enabled = False
            self._audio_enabled = False
            self.clear_all()
            self._close_task = asyncio.create_task(self._close_media())
        await asyncio.shield(self._close_task)

    async def _close_media(self) -> None:
        if self._media_tasks:
            await asyncio.gather(*self._media_tasks, return_exceptions=True)
        await self._media.drain()
        await self._pins_idle.wait()
        self._media.purge_all()

    @staticmethod
    def render(
        messages: list[ChatMessage],
        tz: _tzinfo,
        *,
        with_id: bool = False,
    ) -> str:
        """渲染最近消息；群聊传 ``with_id=True`` 以带完整发送者 ID。"""
        lines = [message.render(tz, with_id=with_id) for message in messages][
            -MAX_RENDER_LINES:
        ]
        return "\n".join(lines)

    @staticmethod
    def last_peer_name(messages: list[ChatMessage]) -> str:
        """最近一条非 bot 消息的发送者昵称，用于私聊场景描述对方。"""
        for message in reversed(messages):
            if not message.is_bot and message.sender_name:
                return message.sender_name
        return ""

    async def preheat(
        self,
        context: Context,
        targets: Mapping[str, int],
        *,
        concurrency: int = PREHEAT_CONCURRENCY,
    ) -> int:
        """启动预热：把会话历史一次性读进缓冲，返回读入条数（``targets`` 的值是条数）。

        群聊取核心的平台消息历史 ``message_history_manager``（核心只在群聊持久化，
        且需要 ``provider_ltm_settings.group_message_history_enable``）；私聊取核心
        的 LLM 会话历史 ``conversation_manager``（只含走过 LLM 的消息，可能为空）。

        覆盖策略：缓冲里已有消息且**比历史多**时保留现有内容，否则用历史整块替换
        （替换前先 :meth:`clear` 释放旧图片）。正常路径由调用方保证「先预热、后放行
        监听」，缓冲是空的；这条比较只兜住「预热途中已有新消息进来」的并发边界，
        避免历史把更新的消息挤掉后留下一段残缺上下文。

        ``concurrency`` 控制同时读取的会话数，绑定内目标较多时缩短启动等待。
        """
        loaded = await self.load_history(context, targets, concurrency=concurrency)
        return self.publish_history(loaded)

    async def load_history(
        self,
        context: Context,
        targets: Mapping[str, int],
        *,
        concurrency: int = PREHEAT_CONCURRENCY,
    ) -> dict[str, list[ChatMessage]]:
        """只读取历史，不修改正式缓冲；由生命周期决定结果是否仍可发布。"""
        history_manager = getattr(context, "message_history_manager", None)
        conversation_manager = getattr(context, "conversation_manager", None)
        if not targets:
            return {}

        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def _load(umo: str, count: int) -> tuple[str, list[ChatMessage]]:
            target = parse_target(umo)
            if target is None or count <= 0:
                return umo, []
            limit = min(int(count), self._maxlen)
            bot_name = self._bot_name(umo) if self._bot_name else ""
            async with semaphore:
                if target.is_group:
                    messages = await history.load_group_history(
                        history_manager,
                        umo,
                        target.platform_id,
                        limit,
                        bot_name=bot_name,
                    )
                else:
                    messages = await history.load_private_history(
                        conversation_manager, umo, limit, bot_name=bot_name
                    )
            return umo, messages

        loaded = await asyncio.gather(
            *(_load(umo, count) for umo, count in targets.items())
        )
        return dict(loaded)

    def publish_history(self, loaded: Mapping[str, list[ChatMessage]]) -> int:
        """同步发布已确认有效的历史，保留既有覆盖与图片释放语义。"""
        filled = 0
        sessions = 0
        for umo, messages in loaded.items():
            messages = [
                message for message in messages if not self._is_ignored(message)
            ]
            if not messages:
                continue
            existing = self._buffers.get(umo)
            if existing and len(existing) > min(len(messages), self._maxlen):
                logger.info(
                    "[Ambient] 跳过预热 %s：缓冲已有 %d 条，多于历史 %d 条。",
                    umo,
                    len(existing),
                    len(messages),
                )
                continue
            if existing:
                self.clear(umo)
            activity = _Activity()
            for message in messages:
                activity.record(message)
            self._activity[umo] = activity
            self._buffers[umo] = deque(messages[-self._maxlen :], maxlen=self._maxlen)
            filled += len(self._buffers[umo])
            sessions += 1
            logger.info("[Ambient] 已预热 %s：%d 条历史消息", umo, len(messages))
        if sessions:
            logger.info(
                "[Ambient] 预热完成：%d 个会话，共 %d 条历史消息。",
                sessions,
                filled,
            )
        return filled


def _file_size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0
