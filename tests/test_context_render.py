"""会话渲染格式：完整日期/周几/时间，群聊带完整 ID，剔除 system_reminder。"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from astrbot_plugin_ambient_chat.common import format_elapsed, parse_target
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
    # 提醒块补上距最后一条消息的间隔，模型不用自己把两个时间相减。
    assert "最后一条消息：2026-11-03 22:20（约 1 分钟前）" in prompt
    # 星期与记录里的写法一致，不再英文/中文混排。
    assert "Weekday: 周二" in prompt
    assert "会话类型：群聊" in prompt
    transcript = prompt.split("<recent_conversation>")[1]
    transcript = transcript.split("</recent_conversation>")[0]
    assert "system_reminder" not in transcript
    # 不传昵称时不额外解释署名，保持旧行为。
    assert "署名" not in prompt


def test_media_labels_number_only_attached_media() -> None:
    """预热历史里的占位没有文件、拿不到编号；被条数上限挡掉的旧图也一样。"""
    buffer = ContextBuffer()
    buffer.configure_media(enabled=True, max_edge=1280, quality=80)
    history = ChatMessage(
        "小明", "", TS, sender_id="1", kind="image", parts=({"type": "image"},)
    )
    old = ChatMessage(
        "小红",
        "",
        TS + 1,
        sender_id="2",
        kind="image",
        parts=({"type": "image"},),
        images=("C:/tmp/a.png",),
    )
    fresh = ChatMessage(
        "小红",
        "",
        TS + 2,
        sender_id="2",
        kind="image",
        parts=({"type": "image"},),
        images=("C:/tmp/b.png",),
    )

    labels = buffer.media_labels([history, old, fresh], image_limit=1, audio_limit=0)

    assert id(history) not in labels
    assert labels[id(old)] == {"image": [None]}
    assert labels[id(fresh)] == {"image": ["图片 1"]}


def test_build_prompt_numbers_only_attached_media() -> None:
    """带编号的占位才是附件：预热历史的 [图片] 不能把新图错位到旧消息上。"""
    target = parse_target("aiocqhttp:GroupMessage:1")
    assert target is not None
    buffer = ContextBuffer()
    buffer.configure_media(enabled=True, max_edge=1280, quality=80)
    history = ChatMessage(
        "小明", "", TS, sender_id="444066154", kind="image", parts=({"type": "image"},)
    )
    fresh = ChatMessage(
        "小明",
        "看这个",
        TS + 60,
        sender_id="444066154",
        kind="image",
        parts=({"type": "text"}, {"type": "image"}),
        images=("C:/tmp/a.png",),
    )
    messages = [history, fresh]

    prompt = build_prompt(
        "接一句",
        messages,
        datetime(2026, 11, 3, 22, 21, tzinfo=TZ),
        target,
        media_labels=buffer.media_labels(messages, image_limit=8, audio_limit=0),
    )

    assert "看这个 [图片 1]" in prompt
    assert "带编号的 [图片 n] / [语音 n] 是本条请求随附的附件" in prompt
    assert "本次没有附带对应内容" in prompt


def test_build_prompt_explains_placeholders_without_attachment() -> None:
    """只有占位、没有附件（预热历史）时说明看不到内容，不按顺序暗示有附件。"""
    target = parse_target("aiocqhttp:GroupMessage:1")
    assert target is not None
    messages = [
        ChatMessage(
            "小明",
            "看这个",
            TS,
            sender_id="444066154",
            kind="image",
            parts=({"type": "text"}, {"type": "image"}),
        )
    ]

    prompt = build_prompt(
        "接一句",
        messages,
        datetime(2026, 11, 3, 22, 21, tzinfo=TZ),
        target,
    )

    assert "本次请求没有附带任何图片或语音" in prompt
    assert "先后顺序" not in prompt


def test_build_prompt_omits_attachment_note_for_pure_text() -> None:
    """纯文本会话不加附件说明，避免模型以为这次带了图。"""
    target = parse_target("aiocqhttp:GroupMessage:1")
    assert target is not None

    prompt = build_prompt(
        "接一句",
        [_message()],
        datetime(2026, 11, 3, 22, 21, tzinfo=TZ),
        target,
    )

    assert "先后顺序" not in prompt


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
    private_prompt = build_prompt(
        "接一句", messages, now, private, group_name="不该出现"
    )

    assert "会话类型：群聊（群名：复古游戏开发群，群号：1）" in named
    assert "会话类型：群聊（群号：1）" in fallback
    assert "不该出现" not in private_prompt


def test_build_prompt_omits_elapsed_when_time_unknown() -> None:
    """最后一条消息时间未知时省略整行，不编造「刚刚」。"""
    target = parse_target("aiocqhttp:GroupMessage:1")
    assert target is not None

    prompt = build_prompt(
        "接一句",
        [_message(timestamp=0)],
        datetime(2026, 11, 3, 22, 21, tzinfo=TZ),
        target,
    )

    reminder = prompt.split("<system_reminder>")[1].split("</system_reminder>")[0]
    assert "最后一条消息" not in reminder


def test_format_elapsed_buckets() -> None:
    """间隔按分钟 / 小时 / 天分档；时钟漂移的负数按「刚刚」。"""
    assert format_elapsed(timedelta(seconds=-5)) == "刚刚"
    assert format_elapsed(timedelta(seconds=59)) == "刚刚"
    assert format_elapsed(timedelta(seconds=60)) == "约 1 分钟前"
    assert format_elapsed(timedelta(minutes=35)) == "约 35 分钟前"
    assert format_elapsed(timedelta(hours=3)) == "约 3 小时前"
    assert format_elapsed(timedelta(days=2, hours=5)) == "约 2 天前"
