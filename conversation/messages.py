"""上下文消息的数据表示、时间格式与对话文本渲染。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from datetime import tzinfo as _tzinfo

MAX_TEXT_CHARS = 300

MESSAGE_KINDS = (
    "text",
    "poke",
    "image",
    "audio",
    "video",
    "file",
    "face",
    "dice",
    "rps",
    "shake",
    "location",
    "contact",
    "share",
    "music",
    "forward",
    "reply",
    "at",
    "event",
)
"""条目 ``type`` 的全部取值；``event`` 是未知组件的兜底。"""

COMPONENT_KINDS = {
    "poke": "poke",
    "image": "image",
    "record": "audio",
    "video": "video",
    "file": "file",
    "face": "face",
    "dice": "dice",
    "rps": "rps",
    "shake": "shake",
    "location": "location",
    "contact": "contact",
    "share": "share",
    "music": "music",
    "forward": "forward",
    "reply": "reply",
    "at": "at",
}
"""平台组件类名（小写）→ 条目 ``type``；没登记的落到 ``event``。"""

PLACEHOLDERS = {
    "image": "图片",
    "audio": "语音",
    "video": "视频",
    "file": "文件",
    "face": "表情",
    "dice": "骰子",
    "rps": "猜拳",
    "shake": "窗口抖动",
    "location": "位置",
    "contact": "名片",
    "share": "分享",
    "music": "音乐",
    "forward": "合并转发",
    "reply": "引用",
    "at": "提及",
    "event": "事件",
}
"""对话模型的文本占位，统一中文短词：读者是中文对话模型，英文词只会增加噪音。"""

SELF_NAME = "我"
"""机器人自己的默认署名：拿不到账号昵称时统一用它，模型据此认出自己的发言。"""

WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

SYSTEM_REMINDER_PATTERN = re.compile(
    r"<system_reminder>.*?</system_reminder>",
    re.DOTALL,
)
"""平台注入的提醒块（User ID / 时间 / 星期）。渲染时整块剔除，
时间改为紧凑时间戳，见 ``docs/adr/0019``。"""


def strip_system_reminder(text: str) -> str:
    """去掉 ``<system_reminder>…</system_reminder>``，可处理一条消息里的多个块。"""
    if not text:
        return ""
    return SYSTEM_REMINDER_PATTERN.sub("", text)


def format_stamp(timestamp: float, tz: _tzinfo) -> str:
    """把 epoch 秒渲染成 ``2026-11-03 周二 22:20``；时间缺失（<=0）时给出可读占位。"""
    moment = to_datetime(timestamp, tz)
    if moment is None:
        return "时间未知"
    return f"{moment:%Y-%m-%d} {WEEKDAY_ZH[moment.weekday()]} {moment:%H:%M}"


def format_time(timestamp: float, tz: _tzinfo) -> str | None:
    """把 epoch 秒渲染成 ``2026-11-03 22:20:45``；时间缺失（<=0）时返回 None。

    结构化决策请求用这个格式：周几已经由 ``state.now`` 给出，逐条只保留时刻本身。
    """
    moment = to_datetime(timestamp, tz)
    if moment is None:
        return None
    return f"{moment:%Y-%m-%d %H:%M:%S}"


def to_datetime(timestamp: float, tz: _tzinfo) -> datetime | None:
    """epoch 秒 → 带时区的 ``datetime``；0 / 非法值返回 None（表示「时间未知」）。"""
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        # 0 是「未知」的哨兵值，不能让 fromtimestamp 把它渲染成 1970 年。
        return None
    try:
        return datetime.fromtimestamp(value, tz)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass(slots=True)
class ChatMessage:
    """缓冲中的一条消息。"""

    sender_name: str
    text: str
    timestamp: float
    is_bot: bool = False
    images: tuple[str, ...] = ()
    """图片在磁盘上的绝对路径，字节数见 ``image_bytes``。"""
    image_bytes: int = 0
    sender_id: str = ""
    """发送者在平台的原始 ID；群聊渲染时附在昵称后用于区分同名用户。"""
    kind: str = "text"
    """消息形态（见 :data:`MESSAGE_KINDS`）；结构化决策请求据此输出 ``type``。"""
    attachments: tuple[str, ...] = ()
    """除正文以外的组件形态，按出现顺序排列（图片/语音/文件/表情…）。

    与 :attr:`text` 解耦：一条消息可以既有文字又有附件，两者都不能丢。
    """
    file_name: str = ""
    """``kind == "file"`` 时的文件名：文件消息除了类型，只剩这条可读信息。"""
    mentions_me: bool = False
    replies_me: bool = False
    """这条消息 @ 了机器人 / 引用了机器人的消息。

    只记这两个事实：其他人的 @ 与引用不进缓冲（记全量要占预算，而判定只关心
    「有没有人在叫我」）。@机器人 与 @全体都被核心适配器从 ``message_str`` 里
    剔除了，所以只能从消息链的组件读。
    """
    target_id: str = ""
    target_name: str = ""
    target_is_self: bool = False
    """``kind == "poke"`` 时戳的对象：ID、展示名、是否指向机器人自己。"""

    def render(self, tz: _tzinfo, *, with_id: bool = False) -> str:
        return (
            f"{self.display_name(with_id)} ({format_stamp(self.timestamp, tz)}): "
            f"{self.render_text()}"
        )

    def body_text(self) -> str:
        """纯正文：剔除 reminder、压空白、超长截断；不含任何附件占位。"""
        text = " ".join(strip_system_reminder(self.text).split())
        if len(text) > MAX_TEXT_CHARS:
            text = f"{text[:MAX_TEXT_CHARS]}…"
        return text

    def placeholder_text(self) -> str:
        """附件/动作的文本占位，供对话模型的会话记录使用。"""
        parts: list[str] = []
        if self.kind == "poke":
            parts.append(self._poke_text())
        images = sum(1 for kind in self.attachments if kind == "image")
        if images:
            count = f"×{images}" if images > 1 else ""
            parts.append(f"[图片{count}]")
        for kind in self.attachments:
            if kind == "image":
                continue
            parts.append(f"[{PLACEHOLDERS.get(kind, PLACEHOLDERS['event'])}]")
        return " ".join(parts)

    def render_text(self) -> str:
        """对话模型看到的正文：@/引用标记 + 纯文字 + 附件占位。"""
        chunks = [self._addressing_text(), self.body_text(), self.placeholder_text()]
        return " ".join(chunk for chunk in chunks if chunk)

    def _addressing_text(self) -> str:
        """「在叫机器人」的标记：@ 了我 / 引用了我。"""
        parts: list[str] = []
        if self.mentions_me:
            parts.append("[提及 我]")
        if self.replies_me:
            parts.append("[引用 我]")
        return " ".join(parts)

    def _poke_text(self) -> str:
        """戳一戳的文本占位：指向机器人自己时写「我」，否则带上目标 ID。"""
        if not self.target_id:
            return "[戳一戳]"
        if self.target_is_self:
            return "[戳一戳 我]"
        return f"[戳一戳 用户#{self.target_id}]"

    def display_name(self, with_id: bool = False) -> str:
        """渲染用名字：私聊只给昵称，群聊额外带完整 ID 以区分同名。"""
        if self.is_bot:
            # 署名可由账号昵称覆盖；拿不到时退回「我」，保持可读。
            return self.sender_name.strip() or SELF_NAME
        name = self.sender_name.strip()
        uid = self.sender_id.strip()
        if not name:
            return f"用户#{uid}" if uid else "某人"
        return f"{name}#{uid}" if with_id and uid else name

    def structured_sender_name(self) -> str:
        """结构化条目里的 ``sender_name``：bot 用署名，其余给昵称（缺昵称时退化）。"""
        if self.is_bot:
            return self.sender_name.strip() or SELF_NAME
        name = self.sender_name.strip()
        if name:
            return name
        uid = self.sender_id.strip()
        return f"用户#{uid}" if uid else "某人"
