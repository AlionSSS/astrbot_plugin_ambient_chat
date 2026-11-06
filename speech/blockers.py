"""拦截机制：静默时段、自身活跃度冷却、群未读条数、连发无人回应。

四条为「或」关系，任一命中即放弃本次发言。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo

from ..common import (
    BLOCKER_COOLDOWN,
    BLOCKER_SILENCE,
    BLOCKER_UNANSWERED,
    BLOCKER_UNREAD,
    parse_hhmm,
)
from ..conversation.context_buffer import ContextBuffer
from ..rules.models import DEFAULT_UNREAD_MIN_COUNT, Rule


@dataclass(slots=True)
class BlockDecision:
    blocked: bool
    reason: str = ""


class BlockerChain:
    """按规则配置依次执行拦截判断。"""

    def __init__(self, buffer: ContextBuffer) -> None:
        self._buffer = buffer

    def check(self, rule: Rule, umo: str, now: datetime) -> BlockDecision:
        silence = rule.blocker(BLOCKER_SILENCE)
        if silence is not None and self._in_silence(silence, now):
            return BlockDecision(True, "静默时段")

        cooldown = rule.blocker(BLOCKER_COOLDOWN)
        if cooldown is not None:
            minutes = int(cooldown.get("minutes", 0) or 0)
            last_at = self._buffer.last_bot_at(umo)
            if last_at is not None and minutes > 0:
                if now.timestamp() - last_at < minutes * 60:
                    return BlockDecision(True, f"冷却中（{minutes} 分钟）")

        unread = rule.blocker(BLOCKER_UNREAD)
        if unread is not None:
            raw_min = unread.get("min_count", DEFAULT_UNREAD_MIN_COUNT)
            min_count = (
                int(raw_min) if raw_min is not None else DEFAULT_UNREAD_MIN_COUNT
            )
            if ":GroupMessage:" in umo and self._buffer.unread_count(umo) < min_count:
                return BlockDecision(
                    True,
                    f"群未读消息不足（{self._buffer.unread_count(umo)}/{min_count}）",
                )

        unanswered = rule.blocker(BLOCKER_UNANSWERED)
        if unanswered is not None:
            reason = self._unanswered(unanswered, umo, now)
            if reason:
                return BlockDecision(True, reason)

        return BlockDecision(False)

    def _unanswered(self, blocker: dict, umo: str, now: datetime) -> str:
        """连发 N 条后一直没人回复：自最后一次发言起再等待 N 分钟。

        判定依赖上下文缓冲：机器人连续发言、中间没有其他人的消息即算「连发」，
        一旦有人发言，连发计数清零，拦截自然解除。

        等待期满按「退避一次」处理：把当前最后一条发言设为连发新基线，
        此后要重新连发满 count 条才会再次进入等待。否则基线一直停在
        那串没人理的发言上，等待期满也只会被放行一条、随即重新计时，
        实际效果就成了永久限速（每 N 分钟一条）。
        """
        count = int(blocker.get("count", 0) or 0)
        minutes = int(blocker.get("minutes", 0) or 0)
        if count <= 0 or minutes <= 0:
            return ""
        streak = self._buffer.bot_streak(umo)
        if streak < count:
            return ""
        last_at = self._buffer.last_bot_at(umo)
        if last_at is None:
            return ""
        deadline = last_at + minutes * 60
        if now.timestamp() >= deadline:
            self._buffer.rebase_bot_streak(umo)
            return ""
        until = _format_until(deadline, now.tzinfo)
        if not until:
            return f"连发 {streak} 条无人回应，已进入永久等待"
        return f"连发 {streak} 条无人回应，等待至 {until}"

    @staticmethod
    def _in_silence(blocker: dict, now: datetime) -> bool:
        minute = now.hour * 60 + now.minute
        for window in blocker.get("windows") or []:
            start = parse_hhmm(str(window.get("start") or ""))
            end = parse_hhmm(str(window.get("end") or ""))
            if not start or not end:
                continue
            start_minute = start[0] * 60 + start[1]
            end_minute = end[0] * 60 + end[1]
            if start_minute < end_minute:
                if start_minute <= minute < end_minute:
                    return True
            elif minute >= start_minute or minute < end_minute:
                # 跨午夜，如 23:00 ~ 次日 09:00
                return True
        return False


def _format_until(timestamp: float, tz: tzinfo | None) -> str:
    """把拦截截止时刻格式化成 ``MM-DD HH:MM``；大到无法表示（等同永久）时返回空串。"""
    try:
        return datetime.fromtimestamp(timestamp, tz).strftime("%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return ""
