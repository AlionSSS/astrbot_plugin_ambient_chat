"""召回结果的装填规则：相关度优先，在字符预算内尽量装完整的记忆。

判定侧（``state.memory``）与生成侧（``<long_term_memory>`` 块）共用这一套规则：
同一次召回在两个消费者里必须看到同一批条目、同一段截断位置，否则会出现「判定
说有话可说、生成时那条理由已经被砍掉」的漂移（ADR-0013）。

三条规则：

1. 按 ``score`` 降序，同分时短条目优先——余量不足时先装短的，截断损失最小。
2. 单条最多占总预算的 :data:`ENTRY_SHARE`：防止一条超长记忆把其余条目挤光。
3. 余量装不下整条就截断本条收尾，而不是丢弃；只有余量低于 :data:`TAIL_MIN_CHARS`
   才放弃——那种长度只剩半句话，接不上也说不出什么。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .livingmemory import MemoryItem

ENTRY_SHARE = 0.6
"""单条记忆最多占总预算的比例。"""

TAIL_MIN_CHARS = 40
"""收尾截断的最小长度：再短就只剩半句话，不如把余量空着。"""

ELLIPSIS = "…"

Pair = tuple["MemoryItem", str]
"""``(记忆, 已压成单行的正文)``。"""


def order_memories(pairs: Iterable[Pair]) -> list[Pair]:
    """去重后按相关度排序：``score`` 降序，同分时短条目优先。

    同一条正文只留一份，保留 ``score`` 更高的那次出现；正文为空或纯空白的条目
    直接丢弃。``score`` 缺失或为 ``NaN`` 时按 0 处理，排序退化为「短优先」，
    不会抛错。排序是稳定的：完全同分的等长条目保持召回时的先后。
    """
    best: dict[str, Pair] = {}
    for item, text in pairs:
        if not text:
            continue
        key = text.casefold()
        current = best.get(key)
        if current is None or _priority(item, text) < _priority(*current):
            best[key] = (item, text)
    return sorted(best.values(), key=lambda pair: _priority(*pair))


def fit_memories(
    pairs: Sequence[Pair],
    *,
    budget: int,
    entry_cap: int | None = None,
    overhead: Callable[[MemoryItem], int] | None = None,
    min_tail: int = TAIL_MIN_CHARS,
) -> list[Pair]:
    """按顺序在 ``budget`` 内装填，返回 ``(item, 截断后的正文)``。

    Args:
        pairs: 已排好序的条目。
        budget: 正文与 ``overhead`` 合计可用的字符总数。
        entry_cap: 单条正文上限；``None`` 表示按 :data:`ENTRY_SHARE` 推导。
        overhead: 每条除正文外的固定开销（时间戳、前缀等）。
        min_tail: 收尾截断的最小正文长度。

    Returns:
        装进预算的条目；正文可能被截断并带省略号，装不下的部分整体丢弃。
    """
    total = max(0, int(budget))
    cost = overhead or _no_overhead
    tail = max(1, int(min_tail))
    cap = int(total * ENTRY_SHARE) if entry_cap is None else int(entry_cap)
    cap = max(tail, min(cap, total))
    fitted: list[Pair] = []
    left = total
    for item, text in pairs:
        extra = max(0, int(cost(item)))
        room = min(cap, left - extra)
        if room <= 0:
            break
        if len(text) > room:
            if room < tail:
                break
            text = truncate(text, room)
        fitted.append((item, text))
        left -= extra + len(text)
        if left <= 0:
            break
    return fitted


def truncate(text: str, limit: int) -> str:
    """截到 ``limit`` 字符以内，截断时以省略号收尾。"""
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    if limit <= 1:
        return ELLIPSIS[:limit]
    return f"{text[: limit - 1].rstrip()}{ELLIPSIS}"


def _priority(item: MemoryItem, text: str) -> tuple[float, int]:
    """排序键：分数越高越前，同分时越短越前。"""
    return (-_score(item), len(text))


def _score(item: MemoryItem) -> float:
    try:
        value = float(getattr(item, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(value) else value


def _no_overhead(_item: MemoryItem) -> int:
    return 0


__all__ = [
    "ELLIPSIS",
    "ENTRY_SHARE",
    "Pair",
    "TAIL_MIN_CHARS",
    "fit_memories",
    "order_memories",
    "truncate",
]
