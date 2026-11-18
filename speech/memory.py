"""放行后为生成召回一次 LivingMemory 记忆，不更新访问统计。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING, Any

from ..conversation.messages import strip_placeholders, strip_system_reminder
from ..memory import senders_for_recall
from ..memory.livingmemory import MAX_QUERY_CHARS

if TYPE_CHECKING:
    from ..conversation.messages import ChatMessage
    from ..memory import LivingMemoryRecall, MemoryItem
    from ..rules.models import Settings

RECALL_TIMEOUT_SECONDS = 3.0
"""单次召回的超时。

比 ``LivingMemoryRecall`` 的默认 8 秒更短：这里处在一次发言的 60 秒预算中间，
记忆属于锦上添花，等不起就放弃本次召回（``recall`` 内部 fail-open）。
"""

QUERY_MESSAGE_COUNT = 6
"""查询消息的条数上限，包含最新消息；实际条数由有效文本和剩余字符决定。"""

QUERY_PRIOR_MESSAGE_CHARS = 48
QUERY_MIN_PRIOR_FRAGMENT_CHARS = 12
_QUERY_ACKS = frozenset(
    {
        "嗯",
        "嗯嗯",
        "哦",
        "噢",
        "啊",
        "好",
        "好的",
        "收到",
        "哈哈",
        "哈哈哈",
        "ok",
        "okay",
        "lol",
    }
)
_QUERY_REFERENCES = frozenset(
    {"看这个", "看那个", "这个呢", "那个呢", "然后呢", "这个", "那个"}
)


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    """查询词与筛过权限的命中快照；调试日志保留候选的原始顺序。"""

    query: str = ""
    items: tuple[MemoryItem, ...] = ()

    def debug_payload(self, tz: tzinfo | None = None) -> dict[str, Any]:
        """调试日志的形状：查询词 + 每条命中的 ID / 分数 / 正文 / 记录时间。

        时间来自 LivingMemory 记忆元数据里的 ``create_time``（写入时刻，epoch 秒）。
        ``created_at`` 保留原始值，``created_at_text`` 按 ``tz`` 渲染成可读文本；
        缺失或非法时为 ``None``，不编造时间。``tz`` 缺省用本地时区。
        """
        zone = tz or datetime.now().astimezone().tzinfo
        return {
            "query": self.query,
            "count": len(self.items),
            "items": [
                {
                    "memory_id": item.memory_id,
                    "score": round(float(item.score or 0.0), 4),
                    "importance": item.importance,
                    "is_recent_filler": item.is_recent_filler,
                    "created_at": item.created_at,
                    "created_at_text": _format_created_at(item.created_at, zone),
                    "content": item.content,
                }
                for item in self.items
            ],
        }


DELIVERY_MEMORY_CHARS = 600
DELIVERY_MEMORY_BASE_WINDOW = 8_192
DELIVERY_MEMORY_MAX_CHARS = 1_800
DELIVERY_INPUT_RATIO = 0.85
# 供应商没有统一的媒体 token 计费接口；带附件时预留近似空间，不声称是精确计数。
IMAGE_TOKEN_ALLOWANCE = 1_024
AUDIO_TOKEN_ALLOWANCE = 2_048


def delivery_memory_chars(
    window: int, *, used_chars: int, images: int = 0, audios: int = 0
) -> int:
    """按整份请求剩余空间估算记忆预算，不为记忆裁剪近期聊天。"""
    window = window if window > 0 else DELIVERY_MEMORY_BASE_WINDOW
    cap = min(
        DELIVERY_MEMORY_MAX_CHARS,
        window * DELIVERY_MEMORY_CHARS // DELIVERY_MEMORY_BASE_WINDOW,
    )
    available = (
        int(window * DELIVERY_INPUT_RATIO)
        - used_chars
        - images * IMAGE_TOKEN_ALLOWANCE
        - audios * AUDIO_TOKEN_ALLOWANCE
    )
    return max(0, min(cap, available))


def provider_context_tokens(context: Any, provider_id: str) -> int:
    """对话模型的上下文窗口；读不到返回 0。

    先看 AstrBot 供应商配置里的 ``max_context_tokens``（agent 路径会自动回填），
    再查内置模型元数据；两者都没有时由调用方决定兜底值。
    """
    if not provider_id:
        return 0
    try:
        provider = context.get_provider_by_id(provider_id)
    except Exception:  # noqa: BLE001
        return 0
    if provider is None:
        return 0

    config = getattr(provider, "provider_config", None) or {}
    declared = _int_or_zero(config.get("max_context_tokens"))
    if declared > 0:
        return declared

    getter = getattr(provider, "get_model", None)
    if not callable(getter):
        return 0
    try:
        model = str(getter() or "").strip()
    except Exception:  # noqa: BLE001
        return 0
    if not model:
        return 0

    try:
        from astrbot.core.utils.llm_metadata import LLM_METADATAS
    except Exception:  # noqa: BLE001
        return 0
    info = LLM_METADATAS.get(model)
    if not isinstance(info, dict):
        return 0
    limit = info.get("limit")
    if not isinstance(limit, dict):
        return 0
    return _int_or_zero(limit.get("context"))


def recall_query(messages: Sequence[ChatMessage]) -> str:
    """先装最新的有效人类文本，再用剩余预算补前文，最后按时间顺序拼接。"""
    parts: list[str] = []
    seen: set[str] = set()
    remaining = MAX_QUERY_CHARS
    anchor_needs_context = False
    for message in reversed(messages):
        if message.is_bot:
            continue
        text = " ".join(strip_placeholders(strip_system_reminder(message.text)).split())
        key = "".join(char for char in text.casefold() if char.isalnum())
        if not key or key in _QUERY_ACKS or key in seen:
            continue
        if parts and key in _QUERY_REFERENCES:
            continue
        if not parts:
            anchor_needs_context = key in _QUERY_REFERENCES
            # 长消息常把具体问题放在末尾；超长时保留首尾，且不让前文顶掉锚点。
            if len(text) > remaining:
                head = remaining // 2
                tail = remaining - head - 1
                text = f"{text[:head].rstrip()} {text[-tail:].lstrip()}"
        else:
            room = min(QUERY_PRIOR_MESSAGE_CHARS, remaining)
            if len(text) > room and room < QUERY_MIN_PRIOR_FRAGMENT_CHARS:
                continue
            text = text[:room].rstrip()
        parts.append(text)
        seen.add(key)
        remaining -= len(text) + 1  # 分隔用的空格
        if remaining <= 0 or len(parts) >= QUERY_MESSAGE_COUNT:
            break
    if anchor_needs_context and len(parts) == 1:
        return ""
    return " ".join(reversed(parts))


async def recall_memories(
    recall: LivingMemoryRecall | None,
    settings: Settings,
    umo: str,
    messages: Sequence[ChatMessage],
    *,
    timeout: float = RECALL_TIMEOUT_SECONDS,
) -> MemoryRecall:
    """按「更多设置」的开关与参数召回一次；任何不可用都返回空快照。

    Args:
        recall: 召回客户端；``None`` 表示插件没构造（旧调用方或测试）。
        settings: 运行参数，读 ``memory_recall_*`` 三项。
        umo: 目标会话，决定记忆作用域与人格。
        messages: 该会话上下文缓冲，按时间正序。
        timeout: 覆盖召回超时。

    Returns:
        查询词与召回结果；关闭、无关键词、未安装或失败时 ``items`` 为空。
        查询词原样保留（空串表示没有发起召回），调试日志靠它还原当时的检索。
    """
    if recall is None or not settings.memory_recall_enabled:
        return MemoryRecall()
    query = recall_query(messages)
    if not query:
        return MemoryRecall()
    items = await recall.recall(
        query,
        umo=umo,
        senders=senders_for_recall(umo, messages, limit=settings.memory_recall_fanout),
        k=settings.memory_recall_k,
        timeout=timeout,
    )
    return MemoryRecall(query=query, items=tuple(items))


def _format_created_at(value: float | None, tz: tzinfo) -> str | None:
    """epoch 秒 → ``YYYY-MM-DD HH:MM``；未知时间返回 ``None``，不写 1970 年。

    与提示块里的时间戳（``memory.render``）同一口径，日志与提示对得上。
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return f"{datetime.fromtimestamp(seconds, tz):%Y-%m-%d %H:%M}"
    except (OverflowError, OSError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)
