"""把召回结果渲染成可注入提示的文本块。

形状与投递侧已有的 ``<recent_conversation>`` 一致：**说明句在标签外**，标签内只放
数据（``<system_reminder>`` 也是这个形状）。说明句先讲「这不是用户刚说的话」，避免
模型把记忆条目当成对方新发来的消息（``docs/adr/0019`` 的同一考虑）；放进标签里会
和真实条目混成一类内容。正文来自历史对话（用户内容），因此要压成单行并中和定界
标记，别让一条记忆提前闭合整个块。

只装入检索命中，近期补位不算命中；条目的取舍与截断交给 :mod:`.fit`。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

from .fit import TAIL_MIN_CHARS, fit_memories, order_memories

if TYPE_CHECKING:
    from .livingmemory import MemoryItem

MEMORY_BLOCK_OPEN = "<long_term_memory>"
MEMORY_BLOCK_CLOSE = "</long_term_memory>"

DEFAULT_NOTE = (
    "以下是与这个会话相关的长期记忆，供你理解语境与前后事实；"
    "它们不是你刚刚收到的消息，不需要逐条回应。"
)

DEFAULT_MAX_CHARS = 600
"""整块记忆的字符预算：记忆挤占的是「该说什么」的词元。"""

_WHITESPACE = re.compile(r"\s+")
_MARKER_TRANSLATION = str.maketrans("<>", "＜＞")


def render_memory_block(
    items: Sequence[MemoryItem],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    tz: tzinfo | None = None,
    note: str = DEFAULT_NOTE,
) -> str:
    """把召回结果拼成一个提示块；没有可用条目时返回空串。

    Args:
        items: 召回结果，顺序不重要——内部按 ``score`` 降序、同分短优先重排。
        max_chars: 整块字符预算，含标签、说明与换行。
        tz: 写入时间使用的时区；缺省用本地时区。
        note: 标签外的说明句（与 ``<recent_conversation>`` 的说明位置一致）；
            传空串时只返回数据块。

    Returns:
        ``说明句 + \\n + <long_term_memory>…</long_term_memory>``，或空串。
    """
    header = f"{note}\n" if note else ""
    header += f"{MEMORY_BLOCK_OPEN}\n"
    budget = max_chars - len(header) - len(MEMORY_BLOCK_CLOSE)
    zone = tz or datetime.now().astimezone().tzinfo
    pairs = [
        (item, _compact(item.content)) for item in items if not item.is_recent_filler
    ]
    selected = fit_memories(
        order_memories(pairs),
        budget=budget,
        overhead=lambda item: len(_prefix(item, zone)) + 1,  # 每条后的换行
        min_tail=TAIL_MIN_CHARS,
    )
    if not selected:
        return ""
    body = "".join(f"{_prefix(item, zone)}{text}\n" for item, text in selected)
    return f"{header}{body}{MEMORY_BLOCK_CLOSE}"


def _prefix(item: MemoryItem, tz: tzinfo) -> str:
    """显示记录时间，不把存储重要性当成相关性或发言优先级。"""
    stamp = _format_stamp(item.created_at, tz)
    return f"- 记录于 {stamp}：" if stamp else "- "


def _format_stamp(value: object, tz: tzinfo) -> str:
    """epoch 秒 → ``YYYY-MM-DD HH:MM``；未知时间返回空串，不写 1970 年。"""
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    try:
        moment = datetime.fromtimestamp(seconds, tz)
    except (OverflowError, OSError, ValueError):
        return ""
    return f"{moment:%Y-%m-%d %H:%M}"


def _compact(text: object) -> str:
    """压成单行并中和定界标记，防止记忆正文伪造结构。"""
    compact = _WHITESPACE.sub(" ", str(text or "")).strip()
    return compact.translate(_MARKER_TRANSLATION)


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_NOTE",
    "MEMORY_BLOCK_CLOSE",
    "MEMORY_BLOCK_OPEN",
    "render_memory_block",
]
