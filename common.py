"""插件共享常量与轻量工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PLUGIN_NAME = "astrbot_plugin_ambient_chat"
"""插件唯一识别名，同时用于 Web API 路由前缀与数据目录名。"""

AMBIENT_HINT_PREFIX = "（此刻没有人点你"
"""自主发言提示语的开头。

历史版本用合成事件投递时，提示语会被核心当成一条收到的消息写进会话历史；
现在改为直接调用模型（见 ``docs/adr/0011``），但旧记录可能仍留在历史里，
预热读取时据此剔除，避免模型把它当成对方说的话。
"""

GROUP_MESSAGE = "GroupMessage"
FRIEND_MESSAGE = "FriendMessage"

DEFAULT_TIMEZONE = "Asia/Shanghai"

TRIGGER_CRON = "cron"
TRIGGER_ONCE = "once"
TRIGGER_WINDOW = "window"
TRIGGER_INTERVAL = "interval"
TRIGGER_TYPES = (TRIGGER_CRON, TRIGGER_ONCE, TRIGGER_WINDOW, TRIGGER_INTERVAL)

TRIGGER_WEEKDAYS = (1, 2, 3, 4, 5, 6, 7)
"""时段随机可选的星期（ISO：周一为 1）；缺省为空列表，表示每天。"""

BLOCKER_SILENCE = "silence"
BLOCKER_COOLDOWN = "cooldown"
BLOCKER_UNREAD = "unread"
BLOCKER_UNANSWERED = "unanswered"
BLOCKER_TYPES = (
    BLOCKER_SILENCE,
    BLOCKER_COOLDOWN,
    BLOCKER_UNREAD,
    BLOCKER_UNANSWERED,
)

INTENT_DIRECT = "direct"
INTENT_DECISION = "decision"

POKE_SELF_NAME = "机器人（你）"
"""戳一戳指向机器人自己时，结构化决策请求里 ``target.name`` 的固定写法。

不加修饰的「机器人」会和群里昵称就叫「机器人」的成员撞车，实测会让决策模型判错方向，
见 ``scripts/poke_probe.py`` 的对照实验。
"""


@dataclass(frozen=True, slots=True)
class Target:
    """一个目标会话（UMO）。"""

    umo: str
    platform_id: str
    message_type: str
    session_id: str

    @property
    def is_group(self) -> bool:
        return self.message_type == GROUP_MESSAGE


def parse_target(umo: str) -> Target | None:
    """把 `platform_id:message_type:session_id` 解析为 Target。"""
    if not isinstance(umo, str):
        return None
    parts = umo.strip().split(":", 2)
    if len(parts) != 3:
        return None
    platform_id, message_type, session_id = (part.strip() for part in parts)
    if not platform_id or not message_type or not session_id:
        return None
    if message_type not in (GROUP_MESSAGE, FRIEND_MESSAGE, "OtherMessage"):
        return None
    return Target(umo.strip(), platform_id, message_type, session_id)


def resolve_timezone(name: str | None) -> ZoneInfo:
    """解析时区名，失败时回退到本地时区。"""
    candidate = (name or "").strip()
    if candidate:
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return datetime.now().astimezone().tzinfo or ZoneInfo(DEFAULT_TIMEZONE)


WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def format_session_line(
    target: Target,
    peer_name: str = "",
    *,
    group_name: str = "",
) -> str:
    """一句话描述会话类型与对象，避免模型把私聊当成群聊。

    群名取运行期缓存（``ContextBuffer.set_session_name``），拿不到时只写群号。
    """
    if target.is_group:
        name = group_name.strip()
        if name:
            return f"会话类型：群聊（群名：{name}，群号：{target.session_id}）"
        return f"会话类型：群聊（群号：{target.session_id}）"
    return f"会话类型：私聊（对方：{peer_name or target.session_id}）"


def format_reminder(
    now: datetime,
    *,
    include_datetime: bool = True,
    session: str = "",
) -> str:
    """构造注入给对话模型的 ``<system_reminder>`` 内容块。

    只包含当前日期时间与会话描述；每条聊天记录已经带了自己的时间，
    不再重复「距最后一条消息多久」。
    ``include_datetime=False`` 用于核心已经注入了绝对时间、插件不重复注入的场景。
    """
    lines: list[str] = []
    if include_datetime:
        zone = now.strftime("%Z") or now.strftime("%z")
        weekday = WEEKDAY_NAMES[now.weekday()]
        lines.append(
            f"Current datetime: {now.strftime('%Y-%m-%d %H:%M')} ({zone}), "
            f"Weekday: {weekday}"
        )
    if session:
        lines.append(session)
    if not lines:
        return ""
    return "<system_reminder>\n" + "\n".join(lines) + "\n</system_reminder>"


def parse_hhmm(value: str) -> tuple[int, int] | None:
    """解析 `HH:MM` 形式的时刻。"""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) != 5 or text[2] != ":":
        return None
    hour_text, minute_text = text[:2], text[3:]
    if not (hour_text.isdigit() and minute_text.isdigit()):
        return None
    hour, minute = int(hour_text), int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def minutes_of_day(hour: int, minute: int) -> int:
    return hour * 60 + minute
