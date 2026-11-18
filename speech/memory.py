"""把 LivingMemory 的长期记忆接进判定与生成。

记忆只服务两个消费者，取数时机跟着意向模式走（``docs/adr/0013``）：

- 决策模型：通过拦截后、判定前取一次，判定读 ``state.memory``，生成复用同一份；
- 概率放行：判定不读语境，放行后由生成侧取，被拒的那次完全不检索。

无论哪条路径，一次发言最多召回一次。召回不更新 LivingMemory 的访问统计：
取数发生在「还不知道会不会发言」的时刻，把这次检索记成「记忆被使用」会把被
拦截、被放弃的检索也算成强化；而想等发送前再补记，又要依赖对方没有公开的
批量接口。所以本插件的记忆只充当语境，不参与记忆自身的衰减排序。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..memory import MemoryItem, senders_for_recall

if TYPE_CHECKING:
    from ..conversation.messages import ChatMessage
    from ..memory import LivingMemoryRecall
    from ..rules.models import Settings

RECALL_TIMEOUT_SECONDS = 3.0
"""单次召回的超时。

比 ``LivingMemoryRecall`` 的默认 8 秒更短：这里处在一次发言的 60 秒预算中间，
记忆属于锦上添花，等不起就放弃本次召回（``recall`` 内部 fail-open）。
"""

QUERY_MESSAGE_COUNT = 3
"""召回关键词用最近几条非 bot 消息拼成。"""

QUERY_MAX_CHARS = 120

DELIVERY_MEMORY_CHARS = 600
"""注入对话模型提示的字符预算。"""

DECISION_ENTRY_CHARS = 120
"""决策 state 里单条记忆的上限。"""

DECISION_MEMORY_CHARS = 360
"""决策 state 里记忆条目的总预算：判定本来就只有 8K 上下文。"""


def recall_query(
    messages: Sequence[ChatMessage],
    *,
    limit: int = QUERY_MESSAGE_COUNT,
    max_chars: int = QUERY_MAX_CHARS,
) -> str:
    """最近几条非 bot 消息拼成召回关键词。

    自主发言没有「当前用户消息」，最近的对话本身就是主题线索；机器人自己的发言
    不算线索，否则会不断用自己说过的话去检索自己。
    """
    parts: list[str] = []
    for message in reversed(messages):
        if message.is_bot:
            continue
        text = message.body_text()
        if text:
            parts.append(text)
        if len(parts) >= limit:
            break
    return " ".join(reversed(parts))[:max_chars]


async def recall_memories(
    recall: LivingMemoryRecall | None,
    settings: Settings,
    umo: str,
    messages: Sequence[ChatMessage],
    *,
    timeout: float = RECALL_TIMEOUT_SECONDS,
) -> list[MemoryItem]:
    """按「更多设置」的开关与参数召回一次；任何不可用都返回空列表。

    Args:
        recall: 召回客户端；``None`` 表示插件没构造（旧调用方或测试）。
        settings: 运行参数，读 ``memory_recall_*`` 三项。
        umo: 目标会话，决定记忆作用域与人格。
        messages: 该会话上下文缓冲，按时间正序。
        timeout: 覆盖召回超时。

    Returns:
        召回结果；关闭、无关键词、未安装或失败时为空列表。
    """
    if recall is None or not settings.memory_recall_enabled:
        return []
    query = recall_query(messages)
    if not query:
        return []
    return await recall.recall(
        query,
        umo=umo,
        senders=senders_for_recall(umo, messages, limit=settings.memory_recall_fanout),
        k=settings.memory_recall_k,
        timeout=timeout,
    )


def memory_entries(
    memories: Sequence[MemoryItem],
    *,
    entry_chars: int = DECISION_ENTRY_CHARS,
    max_chars: int = DECISION_MEMORY_CHARS,
) -> list[str]:
    """把记忆压成决策 state 用的短条目：单行、限长、按预算截断。

    决策模型拿到的 ``state`` 是结构化内容，这里保持「字符串列表」的形状，与
    ``conversation`` 的条目风格一致；超出预算的旧条目直接丢弃，不做半条截断。
    """
    entries: list[str] = []
    used = 0
    seen: set[str] = set()
    for item in memories:
        text = " ".join(str(item.content or "").split())
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        if len(text) > entry_chars:
            text = f"{text[: max(0, entry_chars - 1)].rstrip()}…"
        if used + len(text) > max_chars:
            break
        entries.append(text)
        used += len(text)
    return entries


__all__ = [
    "DECISION_ENTRY_CHARS",
    "DECISION_MEMORY_CHARS",
    "DELIVERY_MEMORY_CHARS",
    "QUERY_MAX_CHARS",
    "QUERY_MESSAGE_COUNT",
    "RECALL_TIMEOUT_SECONDS",
    "memory_entries",
    "recall_memories",
    "recall_query",
]
