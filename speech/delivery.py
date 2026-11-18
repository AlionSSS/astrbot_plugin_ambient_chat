"""发言投递：直接调用对话模型生成，再把结果发到目标会话。

生成走独立请求（上下文由插件组装），不经过核心管线，因此提示语不会写进会话历史，
也不受 ``platform_settings.ignore_bot_self_message`` 影响；发送成功后把这条发言作为
assistant 追加进当前会话历史，让用户回复时核心能感知到。

代价是表情包、装饰等依赖管线的插件不参与，见 ``docs/adr/0011``；长期记忆由插件自己
向 LivingMemory 召回后注入，见 ``docs/adr/0013``。
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import MessageChain

from ..common import (
    Target,
    format_reminder,
    format_session_line,
    parse_target,
    resolve_timezone,
)
from ..conversation.context_buffer import ContextBuffer
from ..conversation.messages import SELF_NAME, ChatMessage, to_datetime
from ..memory import render_memory_block
from ..rules.models import Settings
from .debug_log import to_jsonable, write_debug_record
from .memory import delivery_memory_chars, recall_memories
from .segments import is_blocked_platform, split

if TYPE_CHECKING:
    from astrbot.api.star import Context

    from ..memory import LivingMemoryRecall, MemoryItem
    from ..rules.models import Rule

MAX_IMAGES = 8
"""单次请求最多附带的缓冲图片数，按时间取最近的若干张。"""

MAX_AUDIOS = 2
"""单次请求最多附带的缓冲语音数：语音信息密度低，太多只是白烧 token。"""

CHAT_DEBUG_DIRNAME = "chat_debug"
"""对话模型调试日志目录，与决策日志分开，各自轮转。"""


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
    umo: str
    conversation_id: str
    text: str


@dataclass(frozen=True, slots=True)
class SendOutcome:
    """一次投递实际发出去多少：片段数 + 真正发出的文本。

    ``stopped_by`` 为空表示全部片段都发了；``failed`` 是某段发送失败后中止剩余，
    ``cancelled`` 是段间发现执行已失效（停止绑定、配置变更）而中止。
    """

    sent: int
    total: int
    text: str
    stopped_by: str = ""

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.sent == self.total


class Delivery:
    """把一次「想说的内容」变成一次真实发言。"""

    def __init__(
        self,
        context: Context,
        buffer: ContextBuffer,
        *,
        self_name: Callable[[str], str] | None = None,
        settings: Callable[[], Settings] | None = None,
        recall: LivingMemoryRecall | None = None,
        data_dir: Callable[[], Path] | None = None,
    ) -> None:
        self._context = context
        self._buffer = buffer
        self._self_name = self_name
        """机器人自身的署名来源；缺省时用「我」，测试与旧调用方不必提供。"""
        self._settings: Callable[[], Settings] = settings or Settings
        """运行参数来源；缺省时用默认值，测试与旧调用方不必提供。"""
        self._recall = recall
        """长期记忆客户端；缺省时该能力整体关闭。"""
        self._data_dir = data_dir
        """插件数据目录来源；缺省时不写对话模型调试日志。"""
        self._timezone: tzinfo | None = None

    async def prepare(
        self,
        rule: Rule,
        umo: str,
        hint: str,
        persona_prompt: str = "",
        now: datetime | None = None,
        *,
        memories: Sequence[MemoryItem] | None = None,
    ) -> PreparedSpeech | None:
        """组装并生成一次发言。

        ``memories`` 是执行层已经召回好的记忆（判定阶段取过，这里复用）；``None``
        表示调用方没有召回，本方法会自己取一次，供旧调用点与直接调用使用。
        """
        target = parse_target(umo)
        if target is None:
            logger.warning("[Ambient] 非法目标会话：%s", umo)
            return None
        return await self._prepare(
            rule,
            target,
            hint,
            persona_prompt,
            now or datetime.now(self._tz()),
            memories,
        )

    # ------------------------------------------------------------------ 工具
    def _tz(self) -> tzinfo:
        """跟随 AstrBot 全局时区，与调度器保持一致。"""
        if self._timezone is None:
            try:
                configured = self._context.get_config().get("timezone")
            except Exception:  # noqa: BLE001
                configured = None
            self._timezone = resolve_timezone(str(configured or ""))
        return self._timezone

    async def _memory_block(
        self,
        umo: str,
        messages: list[ChatMessage],
        memories: Sequence[MemoryItem] | None,
        provider_id: str,
    ) -> str:
        """渲染长期记忆块；执行层已召回时直接复用，避免重复检索。

        ``memories`` 为 ``None`` 才自己召回（旧调用点、直接调用），为空列表表示
        「召回过但没有命中」，不再打第二次请求。字符预算跟着对话模型的窗口走：
        换成大模型时自动放宽，读不到窗口才回退默认值。
        """
        if self._recall is None:
            return ""
        items = (
            memories
            if memories is not None
            else await recall_memories(self._recall, self._settings(), umo, messages)
        )
        if not items:
            return ""
        return render_memory_block(
            items,
            max_chars=delivery_memory_chars(self._context, provider_id),
            tz=self._tz(),
        )

    # ------------------------------------------------------------------ 生成
    def _allows_images(self, provider_id: str) -> bool:
        """只有供应商明确声明了图片能力才带图；每次读取以跟随配置变更。"""
        return self._supports_modality(provider_id, "image")

    def _allows_audio(self, provider_id: str) -> bool:
        """音频能力同理：未声明就不带，避免白带一份必然被拒的附件。"""
        return self._supports_modality(provider_id, "audio")

    def _supports_modality(self, provider_id: str, modality: str) -> bool:
        """供应商声明的模态能力；未声明即视为不支持。

        模态能力只有 ``provider_config["modalities"]`` 一个来源：模型真实能力
        无法在请求前探测，而未声明的附件只会换来一次失败请求（核心对音频没有
        失败回退，图片也只在特定错误码上兜底）。因此这里按「未声明 = 不支持」
        从紧处理——少带一份附件是可接受的降级，带上去白烧一次请求不是。
        """
        try:
            provider = self._context.get_provider_by_id(provider_id)
            modalities = provider.provider_config.get("modalities")
        except Exception:  # noqa: BLE001
            return False
        return isinstance(modalities, list) and modality in modalities

    async def _generate(
        self,
        provider_id: str,
        prompt: str,
        system_prompt: str,
        image_urls: list[str],
        audio_urls: list[str] | None = None,
        *,
        umo: str = "",
        stage: str = "initial",
    ) -> Any:
        """单独请求一次对话模型；失败返回 None，由调用方决定是否重试。

        只走 ``provider.text_chat``，不传会话对象、不产生会话历史写入。
        ``umo`` / ``stage`` 仅用于调试日志的定位信息。
        """
        logger.info("[Ambient] 对话模型请求（%s）", provider_id)
        started = time.perf_counter()
        request = {
            "provider_id": provider_id,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "image_urls": list(image_urls),
            "audio_urls": list(audio_urls or ()),
        }
        meta = {
            "provider_id": provider_id,
            "umo": umo,
            "stage": stage,
            "image_count": len(image_urls),
            "audio_count": len(audio_urls or ()),
        }
        try:
            response = await self._context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt or None,
                image_urls=image_urls or None,
                audio_urls=audio_urls or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 生成失败：%s", exc)
            self._debug_record(
                "chat",
                request,
                started=started,
                meta=meta,
                error=f"{type(exc).__name__}: {exc}",
            )
            return None

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "[Ambient] 对话模型响应（%s，%dms）：%d 字",
            provider_id,
            elapsed_ms,
            len(getattr(response, "completion_text", "") or ""),
        )
        self._debug_record(
            "chat",
            request,
            _response_payload(response),
            started=started,
            meta=meta,
        )
        return response

    def _debug_record(
        self,
        kind: str,
        request: Any,
        response: Any = None,
        *,
        started: float,
        meta: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """写入时读取已保存开关，关闭后在途请求也不再落盘。"""
        if self._data_dir is None or not self._settings().chat_debug_enabled:
            return
        write_debug_record(
            kind,
            request,
            response,
            directory=self._data_dir() / CHAT_DEBUG_DIRNAME,
            enabled=True,
            meta=meta,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            error=error,
        )

    async def _prepare(
        self,
        rule: Rule,
        target: Target,
        hint: str,
        persona_prompt: str,
        now: datetime,
        memories: Sequence[MemoryItem] | None = None,
    ) -> PreparedSpeech | None:
        umo = target.umo
        provider_id = rule.chat_provider_id
        if not provider_id:
            try:
                provider_id = await self._context.get_current_chat_provider_id(umo)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Ambient] 无法确定对话模型：%s", exc)
                return None

        conversation_id = await self._current_conversation_id(umo)
        if not conversation_id:
            logger.info("[Ambient] %s 还没有 LLM 会话，跳过本次发言。", umo)
            return None

        messages = self._buffer.recent(umo, rule.context_turns)
        self_name = self._self_name(umo) if self._self_name else ""
        prompt = build_prompt(
            hint,
            messages,
            now,
            target,
            self_name=self_name,
            group_name=self._buffer.session_name(umo),
            memory_block=await self._memory_block(umo, messages, memories, provider_id),
        )
        with self._buffer.borrow_images(
            messages,
            limit=MAX_IMAGES if self._allows_images(provider_id) else 0,
        ) as image_urls:
            with self._buffer.borrow_audio(
                messages,
                limit=MAX_AUDIOS if self._allows_audio(provider_id) else 0,
            ) as audio_urls:
                response = await self._generate(
                    provider_id,
                    prompt,
                    persona_prompt,
                    image_urls,
                    audio_urls,
                    umo=umo,
                    stage="initial",
                )
            if response is None and audio_urls:
                # 逐级退让：先只丢语音（音频支持最不稳定），图片往往还能用。
                # 语音借用已退出，这次不再持有音频文件。
                logger.warning(
                    "[Ambient] 带语音生成失败，去掉语音重试（%s，%d 张图）",
                    umo,
                    len(image_urls),
                )
                response = await self._generate(
                    provider_id,
                    prompt,
                    persona_prompt,
                    image_urls,
                    [],
                    umo=umo,
                    stage="without-audio",
                )
        if response is None and image_urls:
            # 网络、额度等临时错误不能作为模型不支持媒体的证据，只影响本次。
            logger.warning(
                "[Ambient] 带图生成失败，改为纯文本重试（%s，%d 张图）",
                umo,
                len(image_urls),
            )
            response = await self._generate(
                provider_id,
                prompt,
                persona_prompt,
                [],
                [],
                umo=umo,
                stage="text-only",
            )
        if response is None:
            return None

        text = (response.completion_text or "").strip()
        if not text:
            logger.warning("[Ambient] 生成为空，放弃本次发送：%s", umo)
            return None

        return PreparedSpeech(umo, conversation_id, text)

    async def send(
        self,
        speech: PreparedSpeech,
        *,
        should_continue: Callable[[], bool] | None = None,
    ) -> SendOutcome:
        """按「分段拟人发言」逐段发送；``should_continue`` 在段间被问一次。

        第一段不等待。第 1 段就失败时抛异常 / 返回 0 段，与单条发送的语义一致；
        后续某段失败或中止只记部分发送，已发出的片段照旧保留。
        """
        umo = speech.umo
        segments = self._segments(umo, speech.text)
        settings = self._settings()
        sent_text: list[str] = []
        stopped_by = ""
        for index, segment in enumerate(segments):
            if index:
                if should_continue is not None and not should_continue():
                    logger.info(
                        "[Ambient] 段间中止投递（%s）：已发 %d/%d 段。",
                        umo,
                        len(sent_text),
                        len(segments),
                    )
                    stopped_by = "cancelled"
                    break
                await asyncio.sleep(
                    random.uniform(
                        settings.segmented_interval_min,
                        settings.segmented_interval_max,
                    )
                )
            try:
                ok = await self._context.send_message(
                    umo, MessageChain().message(segment)
                )
            except Exception as exc:  # noqa: BLE001
                if not sent_text:
                    logger.warning(
                        "[Ambient] 发送失败：%s: %s（%s）%s",
                        type(exc).__name__,
                        exc,
                        umo,
                        _send_hint(exc),
                    )
                    raise
                logger.warning(
                    "[Ambient] 第 %d/%d 段发送失败，中止剩余片段（%s）：%s",
                    index + 1,
                    len(segments),
                    umo,
                    exc,
                )
                stopped_by = "failed"
                break
            if not ok:
                if not sent_text:
                    logger.warning(
                        "[Ambient] 未找到 %s 对应的平台实例，消息未发送。", umo
                    )
                    break
                logger.warning(
                    "[Ambient] 第 %d/%d 段未发送成功，中止剩余片段（%s）。",
                    index + 1,
                    len(segments),
                    umo,
                )
                stopped_by = "failed"
                break
            sent_text.append(segment)

        return SendOutcome(
            sent=len(sent_text),
            total=len(segments),
            text="".join(sent_text),
            stopped_by=stopped_by,
        )

    def _segments(self, umo: str, text: str) -> list[str]:
        """本次该发几条：关闭、平台不支持或不适合作切分时只有一条。"""
        settings = self._settings()
        if not settings.segmented_enabled:
            return [text]
        target = parse_target(umo)
        if target is None or is_blocked_platform(target.platform_id):
            return [text]
        return split(
            text,
            threshold=settings.segmented_threshold,
            pattern=settings.segmented_pattern,
        )

    async def record(self, speech: PreparedSpeech) -> bool:
        """确认发送后更新缓冲，再写入核心历史；失败不抹去已发送事实。"""
        umo, text = speech.umo, speech.text
        self._buffer.append(
            umo,
            ChatMessage(
                sender_name=self._self_name(umo) if self._self_name else SELF_NAME,
                text=text,
                timestamp=time.time(),
                is_bot=True,
            ),
        )
        recorded = await self._record_in_conversation(umo, speech.conversation_id, text)
        logger.info("[Ambient] %s 已发送，%d 字。", umo, len(text))
        return recorded

    async def _current_conversation_id(self, umo: str) -> str:
        """取当前 LLM 会话 ID；没有会话时返回空串，调用方据此跳过本次发言。"""
        manager = getattr(self._context, "conversation_manager", None)
        if manager is None:
            return ""
        try:
            return str(await manager.get_curr_conversation_id(umo) or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 读取当前会话失败（%s）：%s", umo, exc)
            return ""

    async def _record_in_conversation(
        self,
        umo: str,
        conversation_id: str,
        text: str,
    ) -> bool:
        """把这次发言作为 assistant 追加进核心的 LLM 会话历史。

        只追加 assistant，不伪造用户发言；生成阶段仍走独立请求，读不到这份历史，
        不会自激。写完回读确认没有被并发写入覆盖，写失败或被覆盖时重试一次。
        """
        manager = getattr(self._context, "conversation_manager", None)
        if manager is None:
            return False
        entry = {"role": "assistant", "content": text}
        for attempt in range(2):
            try:
                conversation = await manager.get_conversation(umo, conversation_id)
                history = _parse_history(getattr(conversation, "history", ""))
                history.append(entry)
                await manager.update_conversation(umo, conversation_id, history=history)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[Ambient] 写入会话历史失败（%s，第 %d 次）：%s",
                    umo,
                    attempt + 1,
                    exc,
                )
                continue
            try:
                if await self._conversation_has_entry(
                    manager, umo, conversation_id, entry
                ):
                    return True
            except Exception:  # noqa: BLE001
                continue
            logger.warning("[Ambient] 会话历史疑似被并发覆盖，重试一次（%s）", umo)
        logger.warning("[Ambient] 会话历史未能写入（%s）。", umo)
        return False

    @staticmethod
    async def _conversation_has_entry(
        manager: Any,
        umo: str,
        conversation_id: str,
        entry: dict,
    ) -> bool:
        """回读确认这次发言还在历史末尾附近（防止读-改-写被覆盖）。"""
        conversation = await manager.get_conversation(umo, conversation_id)
        history = _parse_history(getattr(conversation, "history", ""))
        return any(item == entry for item in history[-3:])


def build_prompt(
    hint: str,
    messages: list[ChatMessage],
    now: datetime,
    target: Target,
    *,
    self_name: str = "",
    group_name: str = "",
    memory_block: str = "",
) -> str:
    """拼用户消息：最近会话记录 → 长期记忆 → 时间与会话信息 → 发言提示。

    人格走 ``system_prompt``，这里只放「这一次请求」的内容。记录块带说明与
    ``<recent_conversation>`` 包裹，避免模型把它当成用户刚发来的消息；
    ``self_name`` 非空时点明「记录里这个署名的发言是你自己」，与决策请求的
    ``state.self`` 一致，避免模型把账号昵称当成别的群友。``group_name`` 是
    群聊的可读名称（私聊忽略），让模型知道这是个什么群。

    ``memory_block`` 是长期记忆（``<long_term_memory>`` 包裹），放在记录之后、
    时间提醒之前：先说「以前聊过什么」，再说「现在几点、冷场多久」，最后才是
    这次的任务，从背景到当下逐层收拢。
    """
    parts: list[str] = []
    transcript = ContextBuffer.render(
        messages,
        now.tzinfo or timezone.utc,
        with_id=target.is_group,
    )
    if transcript:
        note = (
            "以下是这个会话最近的聊天记录，仅供你理解语境；"
            "它不是用户刚刚发给你的消息，你也不需要逐条回应。"
        )
        name = (self_name or "").strip()
        if name:
            note += f"记录里署名「{name}」的发言都是你自己之前说的。"
        if "[图片]" in transcript or "[语音]" in transcript:
            # 图片与语音不在文本里，而是作为附件平铺附带：说明顺序对应关系，
            # 否则模型只知道「有图」，不知道哪张图属于哪条消息。
            note += (
                "若本次带有图片或语音附件，它们按记录里 [图片] / [语音] "
                "出现的先后顺序排列。"
            )
        parts.append(
            f"{note}\n<recent_conversation>\n{transcript}\n</recent_conversation>"
        )

    if memory_block:
        parts.append(memory_block)

    tag = format_reminder(
        now,
        session=format_session_line(
            target,
            ContextBuffer.last_peer_name(messages),
            group_name=group_name,
        ),
        last_message_at=_last_message_at(messages, now.tzinfo or timezone.utc),
    )
    if tag:
        parts.append(tag)

    parts.append(hint)
    return "\n\n".join(parts)


def _last_message_at(messages: list[ChatMessage], tz: tzinfo) -> datetime | None:
    """最后一条消息的时间；空列表或时间戳未知（``<=0``）时返回 None。

    时间未知时整行省略而不是猜一个：提醒块里的间隔会直接影响「现在插话是否
    合适」的判断，宁可少一句，也不给模型一个编造的「刚刚」。
    """
    if not messages:
        return None
    return to_datetime(messages[-1].timestamp, tz)


def _response_payload(response: Any) -> dict[str, Any]:
    """只挑可读字段：``raw_completion`` 体积大且与 ``completion_text`` 重复。"""
    usage = getattr(response, "usage", None)
    return {
        "role": getattr(response, "role", ""),
        "completion_text": getattr(response, "completion_text", "") or "",
        "reasoning_content": getattr(response, "reasoning_content", None),
        "usage": to_jsonable(usage) if usage is not None else None,
    }


def _parse_history(raw: Any) -> list[dict]:
    """解析 ``Conversation.history``；格式异常时当空历史处理。"""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _send_hint(exc: Exception) -> str:
    """给空异常文本的平台错误补一句可操作的提示。"""
    if type(exc).__name__ == "ApiNotAvailable":
        return (
            "；平台适配器当前没有可用的 OneBot API 连接，"
            "请检查 OneBot 端是否在线、是否有多个账号连到同一实例"
        )
    return ""
