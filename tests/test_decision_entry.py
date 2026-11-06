"""判定请求的条目结构：type 标记、sender 拆分、戳一戳的 target。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.speech.intent import _message_entry

TZ = ZoneInfo("Asia/Shanghai")
TS = datetime(2026, 11, 5, 21, 29, tzinfo=TZ).timestamp()


def _entry(message: ChatMessage, *, with_id: bool = True) -> dict:
    return _message_entry(message, TZ, with_id=with_id)


def test_text_entry_splits_sender_name_and_id() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小明", text="在吗", timestamp=TS, sender_id="444066154"
        )
    )

    assert entry == {
        "time": "2026-11-05 21:29:00",
        "sender_name": "小明",
        "sender_id": "444066154",
        "type": "text",
        "content": "在吗",
    }


def test_private_chat_keeps_name_only() -> None:
    """私聊渲染本来就不带 ID，条目里也不给 sender_id。"""
    entry = _entry(
        ChatMessage(
            sender_name="小明", text="在吗", timestamp=TS, sender_id="444066154"
        ),
        with_id=False,
    )

    assert entry["sender_name"] == "小明"
    assert "sender_id" not in entry


def test_bot_entry_carries_self_flag_and_id() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="我",
            text="我在",
            timestamp=TS,
            is_bot=True,
            sender_id="10000",
        )
    )

    assert entry["sender_name"] == "我"
    assert entry["sender_id"] == "10000"
    assert entry["self"] is True


def test_poke_entry_has_type_and_target_but_no_content() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="poke",
            target_id="10000",
            target_name="机器人（你）",
            target_is_self=True,
        )
    )

    assert entry["type"] == "poke"
    assert entry["target"] == {
        "id": "10000",
        "name": "机器人（你）",
        "is_self": True,
    }
    # 戳一戳不再带可读正文，避免和 target 重复。
    assert "content" not in entry


def test_poke_without_known_name_keeps_id_and_flag() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="poke",
            target_id="555001",
        )
    )

    assert entry["target"] == {"id": "555001", "is_self": False}


def test_image_entry_uses_image_type_without_redundant_content() -> None:
    """type 已经说明是图片，不再重复塞一个图片占位正文。"""
    entry = _entry(
        ChatMessage(
            sender_name="小明",
            text="",
            timestamp=TS,
            sender_id="444066154",
            kind="image",
            attachments=("image",),
            images=("C:/tmp/a.png",),
            image_bytes=10,
        )
    )

    assert entry["type"] == "image"
    assert "content" not in entry


def test_audio_entry_only_carries_type() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="audio",
            attachments=("audio",),
        )
    )

    assert entry == {
        "time": "2026-11-05 21:29:00",
        "sender_name": "小红",
        "sender_id": "123456",
        "type": "audio",
    }


def test_file_entry_keeps_filename() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="file",
            attachments=("file",),
            file_name="报告.pdf",
        )
    )

    assert entry["type"] == "file"
    assert entry["filename"] == "报告.pdf"


def test_text_plus_attachment_keeps_text_and_lists_attachments() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="看这个",
            timestamp=TS,
            sender_id="123456",
            kind="file",
            attachments=("file", "image"),
            file_name="报告.pdf",
        )
    )

    assert entry["type"] == "file"
    assert entry["content"] == "看这个"
    assert entry["attachments"] == ["file", "image"]
    assert entry["filename"] == "报告.pdf"


def test_entry_marks_mention_and_reply_to_self() -> None:
    """@ 了机器人 / 引用了机器人，只在条目上带事实标记。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="这游戏不错",
            timestamp=TS,
            sender_id="123456",
            mentions_me=True,
            replies_me=True,
        )
    )

    assert entry["mentions_me"] is True
    assert entry["replies_me"] is True
    assert entry["content"] == "这游戏不错"


def test_entry_has_no_addressing_flags_by_default() -> None:
    entry = _entry(
        ChatMessage(sender_name="小红", text="在吗", timestamp=TS, sender_id="123456")
    )

    assert "mentions_me" not in entry
    assert "replies_me" not in entry


def test_event_entry_without_text_has_no_content() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="face",
            attachments=("face",),
        )
    )

    assert entry["type"] == "face"
    assert "content" not in entry
