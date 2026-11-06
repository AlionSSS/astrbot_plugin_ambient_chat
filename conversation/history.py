"""读取 AstrBot 群聊与私聊历史，转换为尚未发布的上下文消息。"""

from __future__ import annotations

import json
from datetime import timezone
from typing import Any

from astrbot.api import logger

from ..common import AMBIENT_HINT_PREFIX
from .messages import COMPONENT_KINDS, ChatMessage


async def load_group_history(
    manager: Any,
    umo: str,
    platform_id: str,
    limit: int,
    *,
    bot_name: str = "",
) -> list[ChatMessage]:
    """读取群聊最近 ``limit`` 条核心平台历史，按时间升序返回。``limit`` 按条计。

    核心把机器人自己的发言写成 ``sender_id="bot"`` / ``sender_name="bot"``，
    这里用调用方给的 ``bot_name`` 覆盖，拿不到就留空，由渲染层回退「我」。
    """
    if manager is None:
        return []
    try:
        rows = await manager.get(
            platform_id=platform_id,
            user_id=umo,
            page=1,
            page_size=max(limit, 20),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Ambient] 读取群聊历史失败（%s）：%s", umo, exc)
        return []
    messages = [m for m in (_row_to_message(row, bot_name) for row in rows or []) if m]
    messages.sort(key=lambda item: item.timestamp)
    return messages[-max(1, limit) :]


async def load_private_history(
    manager: Any,
    umo: str,
    limit: int,
    *,
    bot_name: str = "",
) -> list[ChatMessage]:
    """读取私聊最近 ``limit`` 条核心 LLM 会话历史。

    会话历史是 OpenAI 格式且不带逐条时间：只有最后一条按会话更新时间取真实时间，
    其余保持「时间未知」，不做逐条倒推（见 ``docs/adr/0020``）。未被 LLM 处理过的
    消息不在这份历史里。
    """
    if manager is None:
        return []
    try:
        conversation_id = await manager.get_curr_conversation_id(umo)
        if not conversation_id:
            return []
        conversation = await manager.get_conversation(umo, conversation_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Ambient] 读取私聊会话失败（%s）：%s", umo, exc)
        return []
    if conversation is None:
        return []

    records = _parse_history(getattr(conversation, "history", ""))
    messages = [
        message
        for message in (_history_to_message(item, bot_name) for item in records)
        if message
    ]
    if not messages:
        return []
    messages = messages[-max(1, limit) :]
    _stamp_last_message(messages, int(getattr(conversation, "updated_at", 0) or 0))
    return messages


def _row_to_message(row: Any, bot_name: str = "") -> ChatMessage | None:
    content = getattr(row, "content", None)
    if not isinstance(content, dict):
        return None
    parts = content.get("message") or []
    texts: list[str] = []
    attachments: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type in ("plain", "text"):
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
        else:
            attachments.append(COMPONENT_KINDS.get(part_type, "event"))
    text = " ".join(texts).strip()
    if text.startswith(AMBIENT_HINT_PREFIX):
        # 历史版本用合成事件投递时，提示语被写成了「收到的消息」，不属于对话内容。
        return None
    if not text and not attachments:
        return None

    role = str(content.get("type") or "user")
    # 核心写入的 role 是 "user" / "bot"；"assistant" 是旧版/其它写入方的取值。
    is_bot = role in ("bot", "assistant")
    return ChatMessage(
        sender_name=bot_name if is_bot else str(getattr(row, "sender_name", "") or ""),
        text=text,
        timestamp=_row_timestamp(row),
        is_bot=is_bot,
        # 核心对机器人写的是占位值 "bot"，不是真实账号 ID，丢掉避免冒充成员。
        sender_id="" if is_bot else str(getattr(row, "sender_id", "") or ""),
        kind=attachments[0] if attachments else "text",
        attachments=tuple(attachments),
    )


def _row_timestamp(row: Any) -> float:
    """把核心历史记录的 ``created_at`` 换成 epoch 秒。

    核心以「UTC 的 naive 值」落库（``DateTime(timezone=False)`` + ``now(utc)``），
    直接 ``timestamp()`` 会按本地时区解释，导致时间整体偏移若干小时。
    """
    created_at = getattr(row, "created_at", None)
    if created_at is None:
        return 0.0
    try:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        return created_at.timestamp()
    except (AttributeError, OSError, ValueError):
        return 0.0


def _parse_history(raw: Any) -> list[dict]:
    """把 ``Conversation.history`` 解析成消息列表；异常时当空历史。"""
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return []
        items = parsed if isinstance(parsed, list) else []
    else:
        return []
    return [item for item in items if isinstance(item, dict)]


def _history_to_message(record: dict, bot_name: str = "") -> ChatMessage | None:
    """把一条 OpenAI 格式的会话消息转成缓冲消息，跳过非对话内容。"""
    role = str(record.get("role") or "")
    if role not in ("user", "assistant"):
        # system（人设/指令）与 tool（函数结果）不属于聊天内容。
        return None
    text, attachments = _history_content(record.get("content"))
    if text.startswith(AMBIENT_HINT_PREFIX):
        # 历史版本留下的提示语是 user 消息，读历史时必须剔除。
        return None
    if not text and not attachments:
        return None
    return ChatMessage(
        sender_name=bot_name if role == "assistant" else "",
        text=text,
        timestamp=0.0,
        is_bot=role == "assistant",
        kind=attachments[0] if attachments else "text",
        attachments=attachments,
    )


def _history_content(content: Any) -> tuple[str, tuple[str, ...]]:
    """把会话消息的 ``content`` 拆成 ``(纯文本, 附件形态)``，两者都不丢。"""
    if isinstance(content, str):
        return " ".join(content.split()).strip(), ()
    if not isinstance(content, list):
        return "", ()
    texts: list[str] = []
    attachments: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type in ("text", "input_text"):
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
        elif part_type in ("image_url", "image"):
            attachments.append("image")
        elif part_type in COMPONENT_KINDS:
            attachments.append(COMPONENT_KINDS[part_type])
        else:
            attachments.append("event")
    return " ".join(texts).strip(), tuple(attachments)


def _stamp_last_message(messages: list[ChatMessage], last_at: int) -> None:
    """会话历史没有逐条时间：只给最后一条打上会话更新时间，其余留在 0（时间未知）。

    按固定间隔向前倒推会给整段历史编造时间：几十条消息全挤在同一分钟里，
    与「跨了几小时」的实际对话直接矛盾，比「不知道」更误导，见 ``docs/adr/0020``。
    ``last_at`` 未知（<=0）时全部保持 0。
    """
    if last_at <= 0 or not messages:
        return
    messages[-1].timestamp = float(last_at)
