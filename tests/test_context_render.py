"""会话渲染格式：完整日期/周几/时间，群聊带完整 ID，剔除 system_reminder。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot_plugin_ambient_chat.common import parse_target
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import (
    ChatMessage,
    strip_system_reminder,
)
from astrbot_plugin_ambient_chat.speech.delivery import build_prompt

TZ = ZoneInfo("Asia/Shanghai")
TS = datetime(2026, 11, 3, 22, 20, tzinfo=TZ).timestamp()


def _message(**overrides: str) -> ChatMessage:
    fields = {
        "sender_name": "小明",
        "text": "在吗",
        "timestamp": TS,
        "sender_id": "444066154",
    }
    fields.update(overrides)
    return ChatMessage(**fields)  # type: ignore[arg-type]


def test_group_render_has_full_date_weekday_time_and_id() -> None:
    line = _message().render(TZ, with_id=True)

    assert line == "小明#444066154 (2026-11-03 周二 22:20): 在吗"


def test_private_render_has_no_id() -> None:
    line = _message().render(TZ)

    assert line == "小明 (2026-11-03 周二 22:20): 在吗"


def test_bot_message_renders_as_me() -> None:
    line = _message(sender_name="我", is_bot=True).render(TZ, with_id=True)  # type: ignore[arg-type]

    assert line.startswith("我 (2026-11-03 周二 22:20): ")


def test_missing_name_falls_back_to_full_id() -> None:
    line = _message(sender_name="").render(TZ, with_id=True)

    assert line.startswith("用户#444066154 (2026-11-03 周二 22:20): ")


def test_system_reminder_is_stripped() -> None:
    text = (
        "少吃点 <system_reminder>User ID: 444066154, Nickname: A Lion~ "
        "Current datetime: 2026-11-03 22:20 (CST), Weekday: Tuesday</system_reminder>"
    )
    line = _message(text=text).render(TZ)

    assert "system_reminder" not in line
    assert line.endswith(": 少吃点")


def test_strip_multiple_reminder_blocks() -> None:
    text = (
        "a <system_reminder>x</system_reminder> b <system_reminder>y</system_reminder>"
    )
    stripped = strip_system_reminder(text)

    assert "system_reminder" not in stripped
    assert "a" in stripped and "b" in stripped


def test_buffer_render_switches_id_by_target() -> None:
    messages = [_message(), _message(sender_name="小红", sender_id="123")]

    group = ContextBuffer.render(messages, TZ, with_id=True)
    private = ContextBuffer.render(messages, TZ)

    assert "小明#444066154" in group and "小红#123" in group
    assert "小明 (" in private and "#" not in private


def test_build_prompt_uses_same_render_for_chat_model() -> None:
    target = parse_target("aiocqhttp:GroupMessage:1")
    assert target is not None
    prompt = build_prompt(
        "（此刻没有人点你，但你正好想接一句。）",
        [_message()],
        datetime(2026, 11, 3, 22, 21, tzinfo=TZ),
        target,
    )

    assert "小明#444066154 (2026-11-03 周二 22:20): 在吗" in prompt
    # 每条消息已有时间，提醒块不再重复「最后一条消息：约 N 分钟前」。
    assert "最后一条消息" not in prompt
    assert "会话类型：群聊" in prompt
    transcript = prompt.split("<recent_conversation>")[1]
    transcript = transcript.split("</recent_conversation>")[0]
    assert "system_reminder" not in transcript
    # 不传昵称时不额外解释署名，保持旧行为。
    assert "署名" not in prompt


def test_build_prompt_names_own_messages() -> None:
    """署名改成账号昵称后，对话模型也要知道记录里哪个名字是自己。"""
    target = parse_target("aiocqhttp:GroupMessage:1")
    assert target is not None
    messages = [
        ChatMessage("小明", "在吗", TS, sender_id="444066154"),
        ChatMessage("蓝色小肥鱼", "在的", TS + 10, is_bot=True, sender_id="2670210348"),
    ]

    prompt = build_prompt(
        "（此刻没有人点你，但你正好想接一句。）",
        messages,
        datetime(2026, 11, 3, 22, 21, tzinfo=TZ),
        target,
        self_name="蓝色小肥鱼",
    )

    assert "记录里署名「蓝色小肥鱼」的发言都是你自己之前说的。" in prompt
    assert "蓝色小肥鱼 (2026-11-03 周二 22:20): 在的" in prompt


def test_build_prompt_carries_group_name() -> None:
    """群名跟着会话描述走，让模型知道这是个什么群；私聊不受影响。"""
    now = datetime(2026, 11, 3, 22, 21, tzinfo=TZ)
    group = parse_target("aiocqhttp:GroupMessage:1")
    private = parse_target("aiocqhttp:FriendMessage:444066154")
    assert group is not None and private is not None
    messages = [_message()]

    named = build_prompt("接一句", messages, now, group, group_name="复古游戏开发群")
    fallback = build_prompt("接一句", messages, now, group)
    private_prompt = build_prompt("接一句", messages, now, private, group_name="不该出现")

    assert "会话类型：群聊（群名：复古游戏开发群，群号：1）" in named
    assert "会话类型：群聊（群号：1）" in fallback
    assert "不该出现" not in private_prompt
