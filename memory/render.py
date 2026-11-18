"""把召回结果渲染成可注入提示的文本块。

形状与投递侧已有的 ``<recent_conversation>`` 一致：**说明句在标签外**，标签内只放
数据（``<system_reminder>`` 也是这个形状）。说明句先讲「这不是用户刚说的话」，避免
模型把记忆条目当成对方新发来的消息（``docs/adr/0019`` 的同一考虑）；放进标签里会
和真实条目混成一类内容。正文来自历史对话（用户内容），因此要压成单行并中和定界
标记，别让一条记忆提前闭合整个块。

条目的取舍与截断交给 :mod:`.fit`，与判定侧共用同一套规则。
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
_SENSITIVE_MARKERS = (
    MEMORY_BLOCK_OPEN,
    MEMORY_BLOCK_CLOSE,
    "<system_reminder>",
    "</system_reminder>",
)


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
        max_chars: 条目部分（不含定界与说明）的字符预算；按余量装填，超出即
            截断本条收尾。
        tz: 写入时间使用的时区；缺省用本地时区。
        note: 标签外的说明句（与 ``<recent_conversation>`` 的说明位置一致）；
            传空串时只返回数据块。

    Returns:
        ``说明句 + \\n + <long_term_memory>…</long_term_memory>``，或空串。
    """
    budget = max(0, int(max_chars))
    zone = tz or datetime.now().astimezone().tzinfo
    pairs = [(item, _compact(getattr(item, "content", ""))) for item in items]
    selected = fit_memories(
        order_memories(pairs),
        budget=budget,
        overhead=lambda item: len(_prefix(item, zone)),
        min_tail=TAIL_MIN_CHARS,
    )
    if not selected:
        return ""
    body = "\n".join(f"{_prefix(item, zone)}{text}" for item, text in selected)
    block = "\n".join([MEMORY_BLOCK_OPEN, body, MEMORY_BLOCK_CLOSE])
    # 说明句留在标签外：标签内只放可被当成数据的条目，和会话记录保持同一形状。
    return f"{note}\n{block}" if note else block


def _prefix(item: MemoryItem, tz: tzinfo) -> str:
    """``- 2026-11-02 14:30 · 重要性 0.80：``；缺失的字段整段省略。"""
    head: list[str] = []
    stamp = _format_stamp(getattr(item, "created_at", None), tz)
    if stamp:
        head.append(stamp)
    importance = getattr(item, "importance", None)
    if isinstance(importance, (int, float)) and not isinstance(importance, bool):
        head.append(f"重要性 {float(importance):.2f}")
    return f"- {' · '.join(head)}：" if head else "- "


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
    for marker in _SENSITIVE_MARKERS:
        compact = compact.replace(marker, marker.replace("<", "＜").replace(">", "＞"))
    return compact


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_NOTE",
    "MEMORY_BLOCK_CLOSE",
    "MEMORY_BLOCK_OPEN",
    "render_memory_block",
]
