"""插件共享常量与轻量工具。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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
"""按概率放行。键名沿用历史的 ``direct``，已保存的规则无需迁移。"""

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
    "周一",
    "周二",
    "周三",
    "周四",
    "周五",
    "周六",
    "周日",
)
"""提醒块里的星期写法，与聊天记录里 ``WEEKDAY_ZH`` 的输出保持一致。"""


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


def format_elapsed(delta: timedelta) -> str:
    """把时间差写成口语化间隔；时钟漂移导致的负数按「刚刚」处理。"""
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "刚刚"
    minutes = seconds // 60
    if minutes < 60:
        return f"约 {minutes} 分钟前"
    hours = minutes // 60
    if hours < 24:
        return f"约 {hours} 小时前"
    return f"约 {hours // 24} 天前"


def format_reminder(
    now: datetime,
    *,
    include_datetime: bool = True,
    session: str = "",
    last_message_at: datetime | None = None,
) -> str:
    """构造注入给对话模型的 ``<system_reminder>`` 内容块。

    包含当前日期时间、可选「距最后一条消息多久」与会话描述。间隔这一行是给
    模型判断「现在接话是否合适、该说什么」用的：绝对时间只能说明现在几点，
    说明不了对话冷了多久；两者都不让它自己相减（``docs/adr/0019``）。
    ``last_message_at`` 为 ``None`` 表示最后一条消息的时间未知，整行省略，不编造。
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
    if last_message_at is not None:
        lines.append(
            f"最后一条消息：{last_message_at:%Y-%m-%d %H:%M}"
            f"（{format_elapsed(now - last_message_at)}）"
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
