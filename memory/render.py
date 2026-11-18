"""把召回结果渲染成可注入提示的文本块。

形状对齐投递侧已有的 ``<system_reminder>``：用一对定界标签包住，并先说明「这不是
用户刚说的话」，避免模型把记忆条目当成对方新发来的消息（``docs/adr/0019`` 的同一
考虑）。正文来自历史对话（用户内容），因此要压成单行并中和定界标记，别让一条记忆
提前闭合整个块。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, tzinfo
from typing import TYPE_CHECKING

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

ENTRY_MAX_CHARS = 160
"""单条记忆的渲染上限：一次导入的长记忆不该独占整块预算。"""

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
        items: 召回结果，按相关度从高到低；正文完全相同的条目只保留第一条。
        max_chars: 条目部分（不含定界与说明）的字符预算，超出即停止追加。
        tz: 写入时间使用的时区；缺省用本地时区。
        note: 块首的说明句，调用方可按提示风格替换。

    Returns:
        ``<long_term_memory>`` 包裹的文本块，或空串。
    """
    budget = max(0, int(max_chars))
    zone = tz or datetime.now().astimezone().tzinfo
    lines: list[str] = []
    used = 0
    seen: set[str] = set()

    for item in items:
        content = _compact(getattr(item, "content", ""))
        if not content:
            continue
        key = content.casefold()
        if key in seen:
            continue
        seen.add(key)

        line = _format_entry(item, content, zone)
        if used + len(line) > budget:
            remaining = budget - used
            if not lines or remaining <= 1:
                break
            lines.append(_truncate(line, remaining))
            break
        lines.append(line)
        used += len(line)

    if not lines:
        return ""
    body = "\n".join(lines)
    return "\n".join([MEMORY_BLOCK_OPEN, note, body, MEMORY_BLOCK_CLOSE])


def _format_entry(item: MemoryItem, content: str, tz: tzinfo) -> str:
    """``- 2026-11-02 14:30 · 重要性 0.80：正文``；缺失的字段整段省略。"""
    head: list[str] = []
    stamp = _format_stamp(getattr(item, "created_at", None), tz)
    if stamp:
        head.append(stamp)
    importance = getattr(item, "importance", None)
    if isinstance(importance, (int, float)) and not isinstance(importance, bool):
        head.append(f"重要性 {float(importance):.2f}")
    prefix = f"- {' · '.join(head)}：" if head else "- "
    return f"{prefix}{_truncate(content, ENTRY_MAX_CHARS)}"


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


def _truncate(text: str, limit: int) -> str:
    limit = max(0, int(limit))
    return text if len(text) <= limit else f"{text[: max(0, limit - 1)].rstrip()}…"


def _compact(text: object) -> str:
    """压成单行并中和定界标记，防止记忆正文伪造结构。"""
    compact = _WHITESPACE.sub(" ", str(text or "")).strip()
    for marker in _SENSITIVE_MARKERS:
        compact = compact.replace(marker, marker.replace("<", "＜").replace(">", "＞"))
    return compact


__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_NOTE",
    "ENTRY_MAX_CHARS",
    "MEMORY_BLOCK_CLOSE",
    "MEMORY_BLOCK_OPEN",
    "render_memory_block",
]
