"""上下文消息的数据表示、时间格式与对话文本渲染。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import tzinfo as _tzinfo
from typing import Any

MAX_TEXT_CHARS = 300
MAX_FILE_NAME_CHARS = 40
"""对话侧占位里文件名的上限：文件名来自平台，过长会挤占生成提示的词元。"""

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
    "event",
)
"""条目 ``type`` 与 ``parts`` 各段 ``type`` 的全部取值；``event`` 是未知组件的兜底。

``at`` / ``reply`` 不在此列：它们进 ``parts`` 表达位置与目标（见 :data:`DECORATION_KINDS`），
但不算「消息形态」，也不进 ``attachments``。``at`` 与 ``reply`` 单独列出，是因为
「这条消息是什么」由文字/媒体决定，@ 谁、引用谁只是行内装饰。
"""

DECORATION_KINDS = ("at", "reply")
"""行内装饰段：``at`` / ``reply`` 各带一个 ``target_role_id``，以及仅供渲染的 ``is_self``。"""

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
    "event": "事件",
}
"""对话模型的文本占位，统一中文短词：读者是中文对话模型，英文词只会增加噪音。"""

TEXT_SEGMENT = "text"
"""有序单元里代表正文的那一段；它只标记位置，正文始终由 :attr:`ChatMessage.text` 提供。"""


def shape_from_parts(
    parts: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...], str]:
    """从有序单元派生 ``(kind, attachments, file_name)``。

    ``kind`` 取第一个非文字单元，只为兼容旧字段（渲染与判定都以 ``parts`` 为准）；
    ``file_name`` 不再要求文件排在第一——那正是旧实现丢文件名的地方（ADR-0009）。
    """
    kinds = tuple(
        str(part.get("type") or "")
        for part in parts
        if part.get("type") != TEXT_SEGMENT and part.get("type") not in DECORATION_KINDS
    )
    file_name = next(
        (
            str(part.get("filename") or "").strip()
            for part in parts
            if part.get("type") == "file"
        ),
        "",
    )
    return (kinds[0] if kinds else TEXT_SEGMENT), kinds, file_name


def _shorten(name: str, limit: int = MAX_FILE_NAME_CHARS) -> str:
    return name if len(name) <= limit else f"{name[: limit - 1]}…"


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
    audio: tuple[str, ...] = ()
    """语音在磁盘上的绝对路径，字节数见 ``audio_bytes``；缓存关闭时为空。"""
    audio_bytes: int = 0
    sender_id: str = ""
    """发送者在平台的原始 ID；群聊渲染时附在昵称后用于区分同名用户。"""
    kind: str = "text"
    """消息形态（见 :data:`MESSAGE_KINDS`）。兼容字段：渲染与判定都改由 ``parts`` 决定。"""
    parts: tuple[dict[str, Any], ...] = ()
    """有序组成单元（ADR-0009）：``{"type": "text"}`` 标记文字的位置，其余段是组件。

    文字段不带正文——正文始终由 :attr:`text` 提供，避免同一段文字存两份，也避免
    截断规则在两处漂移。空元组表示「未采集顺序」（旧构造点与测试），此时按
    :attr:`kind` + :attr:`attachments` 的旧约定兜底，文字排在附件之前。

    ``at`` / ``reply`` 也作为段出现（ADR-0009）：各带 ``target_role_id``（目标账号）
    与 ``is_self``（是否指向机器人自己）；``is_self`` 只供渲染，进判定请求前剥掉。
    """
    attachments: tuple[str, ...] = ()
    """除正文以外的组件形态，按出现顺序排列（图片/语音/文件/表情…），由 ``parts`` 派生。"""
    file_name: str = ""
    """消息里第一个文件的文件名；与它在链里的位置无关（ADR-0009）。"""
    target_id: str = ""
    target_is_self: bool = False
    """消息里有戳一戳时戳的对象 ID，以及它是否指向机器人自己。

    不记目标昵称（ADR-0009 修订）：戳别人时昵称可能和机器人重名，模型做等值比较后
    会误判成「戳的是自己」；``is_self`` 才是可靠依据。
    """

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

    def ordered_parts(self) -> tuple[dict[str, Any], ...]:
        """有序组成单元：优先用采集到的顺序，否则按旧约定兜底（正文在最前）。

        采集侧给出真实顺序（文字段只标记位置）；没有 ``parts`` 时按 ``kind`` +
        ``attachments`` 推导，让旧构造点与测试不必逐个补齐——但那种情况拿不到
        「文字在图片前还是后」，只能按旧约定把正文排在前面。
        """
        if self.parts:
            return self.parts
        segments: list[dict[str, Any]] = []
        if self.body_text():
            segments.append({"type": TEXT_SEGMENT})
        kinds = list(self.attachments)
        if self.kind != TEXT_SEGMENT and self.kind not in kinds:
            kinds.insert(0, self.kind)
        for kind in kinds:
            segment: dict[str, Any] = {"type": kind}
            if kind == "file" and self.file_name:
                segment["filename"] = self.file_name
            segments.append(segment)
        return tuple(segments)

    def render_text(self) -> str:
        """对话模型看到的正文：@/引用标记 + 按真实顺序排列的文字与占位。"""
        chunks = [self._addressing_text()]
        for segment in self.ordered_parts():
            if segment.get("type") == TEXT_SEGMENT:
                chunks.append(self.body_text())
            else:
                chunks.append(self._segment_text(segment))
        return " ".join(chunk for chunk in chunks if chunk)

    def _segment_text(self, segment: Mapping[str, Any]) -> str:
        """一个非文字单元的占位：戳一戳带目标，文件带文件名，其余用中文短词。"""
        kind = str(segment.get("type") or "")
        if kind == "poke":
            return self._poke_text()
        if kind == "file":
            name = str(segment.get("filename") or self.file_name or "").strip()
            if name:
                return f"[文件 {_shorten(name)}]"
        if kind in DECORATION_KINDS:
            # 行内装饰不占占位：标记由 _addressing_text 统一放在正文前。
            return ""
        return f"[{PLACEHOLDERS.get(kind, PLACEHOLDERS['event'])}]"

    def _addressing_text(self) -> str:
        """「在叫机器人」的标记：@ 了我 / 引用了我。

        事实读自 ``parts`` 的 ``at`` / ``reply`` 段，不再单存两个布尔：同一条消息里
        这两件事各有位置，判定请求也直接用这些段（ADR-0009）。
        """
        marks: list[str] = []
        for segment in self.ordered_parts():
            kind = str(segment.get("type") or "")
            if not segment.get("is_self"):
                continue
            if kind == "at":
                marks.append("[提及 我]")
            elif kind == "reply":
                marks.append("[引用 我]")
        return " ".join(marks)

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

    def structured_role_name(self) -> str:
        """结构化条目里的 ``role_name``：与渲染用名字共用一套规则（只是不带 ID）。

        单单元条目、多单元条目的 ``parts``、WebUI 缓冲区都从这里取名字；再写一份
        同样的兜底判断，改一处忘一处就会出现「对话侧叫匿名成员、决策侧叫用户#10001」。
        """
        return self.display_name()
