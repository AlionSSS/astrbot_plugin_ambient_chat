"""发言投递：直接调用对话模型生成，再把结果发到目标会话。

生成走独立请求（上下文由插件组装），不经过核心管线，因此提示语不会写进会话历史，
也不受 ``platform_settings.ignore_bot_self_message`` 影响；发送成功后把这条发言作为
assistant 追加进当前会话历史，让用户回复时核心能感知到。

代价是表情包、记忆、装饰等依赖管线的插件不参与，见 ``docs/adr/0011``。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
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
from ..conversation.messages import SELF_NAME, ChatMessage

if TYPE_CHECKING:
    from astrbot.api.star import Context

    from ..rules.models import Rule

MAX_IMAGES = 8
"""单次请求最多附带的缓冲图片数，按时间取最近的若干张。"""

MAX_AUDIOS = 2
"""单次请求最多附带的缓冲语音数：语音信息密度低，太多只是白烧 token。"""


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
    umo: str
    conversation_id: str
    text: str


class Delivery:
    """把一次「想说的内容」变成一次真实发言。"""

    def __init__(
        self,
        context: Context,
        buffer: ContextBuffer,
        *,
        self_name: Callable[[str], str] | None = None,
    ) -> None:
        self._context = context
        self._buffer = buffer
        self._self_name = self_name
        """机器人自身的署名来源；缺省时用「我」，测试与旧调用方不必提供。"""
        self._timezone: tzinfo | None = None

    async def prepare(
        self,
        rule: Rule,
        umo: str,
        hint: str,
        persona_prompt: str = "",
        now: datetime | None = None,
    ) -> PreparedSpeech | None:
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

    # ------------------------------------------------------------------ 生成
    def _allows_images(self, provider_id: str) -> bool:
        """仅在供应商明确未启用图片能力时跳过图片；每次读取以跟随配置变更。"""
        return self._supports_modality(provider_id, "image")

    def _allows_audio(self, provider_id: str) -> bool:
        """音频能力同理：只有明确不含 ``audio`` 时才跳过，避免白带音频报错。"""
        return self._supports_modality(provider_id, "audio")

    def _supports_modality(self, provider_id: str, modality: str) -> bool:
        try:
            provider = self._context.get_provider_by_id(provider_id)
            modalities = provider.provider_config.get("modalities")
        except Exception:  # noqa: BLE001
            return True
        # 与核心迁移语义一致：空列表不是明确的能力限制。
        return (
            not isinstance(modalities, list) or not modalities or modality in modalities
        )

    async def _generate(
        self,
        provider_id: str,
        prompt: str,
        system_prompt: str,
        image_urls: list[str],
        audio_urls: list[str] | None = None,
    ) -> Any:
        """单独请求一次对话模型；失败返回 None，由调用方决定是否重试。

        只走 ``provider.text_chat``，不传会话对象、不产生会话历史写入。
        """
        logger.info("[Ambient] 对话模型请求（%s）", provider_id)
        started = time.perf_counter()
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
            return None

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "[Ambient] 对话模型响应（%s，%dms）：%d 字",
            provider_id,
            elapsed_ms,
            len(getattr(response, "completion_text", "") or ""),
        )
        return response

    async def _prepare(
        self,
        rule: Rule,
        target: Target,
        hint: str,
        persona_prompt: str,
        now: datetime,
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
                    provider_id, prompt, persona_prompt, image_urls, audio_urls
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
                    provider_id, prompt, persona_prompt, image_urls, []
                )
        if response is None and image_urls:
            # 网络、额度等临时错误不能作为模型不支持媒体的证据，只影响本次。
            logger.warning(
                "[Ambient] 带图生成失败，改为纯文本重试（%s，%d 张图）",
                umo,
                len(image_urls),
            )
            response = await self._generate(provider_id, prompt, persona_prompt, [], [])
        if response is None:
            return None

        text = (response.completion_text or "").strip()
        if not text:
            logger.warning("[Ambient] 生成为空，放弃本次发送：%s", umo)
            return None

        return PreparedSpeech(umo, conversation_id, text)

    async def send(self, speech: PreparedSpeech) -> bool:
        umo, text = speech.umo, speech.text
        try:
            sent = await self._context.send_message(umo, MessageChain().message(text))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Ambient] 发送失败：%s: %s（%s）%s",
                type(exc).__name__,
                exc,
                umo,
                _send_hint(exc),
            )
            raise
        if not sent:
            logger.warning("[Ambient] 未找到 %s 对应的平台实例，消息未发送。", umo)
            return False

        return True

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
) -> str:
    """拼用户消息：最近会话记录 → 时间与会话信息 → 发言提示。

    人格走 ``system_prompt``，这里只放「这一次请求」的内容。记录块带说明与
    ``<recent_conversation>`` 包裹，避免模型把它当成用户刚发来的消息；
    ``self_name`` 非空时点明「记录里这个署名的发言是你自己」，与决策请求的
    ``state.self`` 一致，避免模型把账号昵称当成别的群友。``group_name`` 是
    群聊的可读名称（私聊忽略），让模型知道这是个什么群。提示语放最后，
    让模型最后看到的是任务本身。
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
        parts.append(
            f"{note}\n<recent_conversation>\n{transcript}\n</recent_conversation>"
        )

    tag = format_reminder(
        now,
        session=format_session_line(
            target,
            ContextBuffer.last_peer_name(messages),
            group_name=group_name,
        ),
    )
    if tag:
        parts.append(tag)

    parts.append(hint)
    return "\n\n".join(parts)


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
