"""从上下文缓冲里挑出「最近若干不同的发送者」。

``user``（按人共享）作用域下，群聊里每条记忆挂在具体某个人的身份上，群号查不到；
要覆盖多人只能并联多份作用域。这里只负责「查谁」，取数与并联交给
:class:`~.livingmemory.LivingMemoryRecall`。

按最近优先而不是按发言频次取人：上下文窗口本来就是最近若干条，频次在这么短的窗口
里没有额外信息，反而会把「刚说话的人」排到后面。

私聊不需要这个方法：对端只有一个，直接用缓冲里的 ``sender_id``。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..common import parse_target
from .livingmemory import SenderRef

if TYPE_CHECKING:
    from ..conversation.messages import ChatMessage

DEFAULT_FANOUT = 3
"""群聊默认查最近几个人：人越多召回面越广，也越容易把别人的话题带进当前语境。"""


def senders_for_recall(
    umo: str,
    messages: Sequence[ChatMessage],
    *,
    limit: int = DEFAULT_FANOUT,
) -> list[SenderRef]:
    """按会话类型决定「查谁」：群聊并联最近 ``limit`` 人，私聊只查对端。

    这是本插件对 ``user``（按人共享）作用域的既定策略，调用方直接透传给
    ``LivingMemoryRecall.recall(senders=...)`` 即可，不必自己判断会话类型。

    Args:
        umo: 目标会话。
        messages: 该会话的上下文缓冲，按时间正序。
        limit: 群聊最多并联几个人。

    Returns:
        ``[(sender_id, sender_name), ...]``；会话类型非法或缓冲里没有可用发送者时为空。
    """
    target = parse_target(umo)
    if target is None:
        return []
    if target.is_group:
        return collect_recent_senders(messages, limit=limit)
    return _last_peer(messages)


def _last_peer(messages: Sequence[ChatMessage]) -> list[SenderRef]:
    """私聊只有一个对端：取最近一条非 bot 消息的发送者。"""
    for message in reversed(messages):
        if bool(getattr(message, "is_bot", False)):
            continue
        sender_id = str(getattr(message, "sender_id", "") or "").strip()
        if sender_id:
            return [(sender_id, _name_of(message))]
    return []


def collect_recent_senders(
    messages: Sequence[ChatMessage],
    *,
    limit: int = DEFAULT_FANOUT,
    exclude_bot: bool = True,
) -> list[SenderRef]:
    """按「最近优先」取 ``limit`` 个不同发送者。

    Args:
        messages: 一个会话的上下文缓冲，按时间正序（与 ``ContextBuffer.recent`` 一致）。
        limit: 最多取几个不同的人；``<= 0`` 时返回空列表（等价于不查询）。
        exclude_bot: 跳过机器人自己的消息。召回是「别人说过什么」，自己的发言不算。

    Returns:
        ``[(sender_id, sender_name), ...]``，最近说话的人在前；缺少 ``sender_id``
        的消息无法映射到记忆作用域，直接跳过。
    """
    if limit <= 0:
        return []
    picked: list[SenderRef] = []
    seen: set[str] = set()
    for message in reversed(messages):
        if exclude_bot and bool(getattr(message, "is_bot", False)):
            continue
        sender_id = str(getattr(message, "sender_id", "") or "").strip()
        if not sender_id or sender_id in seen:
            continue
        seen.add(sender_id)
        picked.append((sender_id, _name_of(message)))
        if len(picked) >= limit:
            break
    return picked


def _name_of(message: Any) -> str:
    """昵称只在 ``identity_aliases`` 里参与身份映射，取不到就给空串。"""
    return str(getattr(message, "sender_name", "") or "").strip()


__all__ = ["DEFAULT_FANOUT", "collect_recent_senders", "senders_for_recall"]
