"""判定请求的条目结构：role 标记、type 标记、戳一戳的 target。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.speech.intent import _message_entry

TZ = ZoneInfo("Asia/Shanghai")
TS = datetime(2026, 11, 5, 21, 29, tzinfo=TZ).timestamp()


def _entry(message: ChatMessage, *, with_id: bool = True) -> dict:
    return _message_entry(message, TZ, with_id=with_id)


def test_text_entry_splits_role_name_and_id() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小明", text="在吗", timestamp=TS, sender_id="444066154"
        )
    )

    assert entry == {
        "time": "2026-11-05 21:29:00",
        "role": "user",
        "role_name": "小明",
        "role_id": "444066154",
        "type": "text",
        "content": "在吗",
    }


def test_private_chat_keeps_name_only() -> None:
    """私聊渲染本来就不带 ID，条目里也不给 role_id。"""
    entry = _entry(
        ChatMessage(
            sender_name="小明", text="在吗", timestamp=TS, sender_id="444066154"
        ),
        with_id=False,
    )

    assert entry["role_name"] == "小明"
    assert "role_id" not in entry


def test_bot_entry_is_assistant_and_carries_self_flag() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="我",
            text="我在",
            timestamp=TS,
            is_bot=True,
            sender_id="10000",
        )
    )

    assert entry["role"] == "assistant"
    assert entry["role_name"] == "我"
    assert entry["role_id"] == "10000"
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
            target_is_self=True,
        )
    )

    assert entry["type"] == "poke"
    assert entry["target"] == {"id": "10000", "is_self": True}
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
        "role": "user",
        "role_name": "小红",
        "role_id": "123456",
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


def test_text_plus_attachments_use_ordered_parts() -> None:
    """文字 + 多个附件：顺序与组成都进 parts，不再靠 type/attachments 二选一。"""
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

    assert entry["parts"] == [
        {"type": "text", "content": "看这个"},
        {"type": "file", "filename": "报告.pdf"},
        {"type": "image"},
    ]
    assert "type" not in entry and "attachments" not in entry


def test_text_plus_single_attachment_keeps_both_facts() -> None:
    """文字 + 单附件：此前只有 type（且报成附件形态），文字只能靠 content 反推。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="哈哈",
            timestamp=TS,
            sender_id="123456",
            kind="face",
            attachments=("face",),
        )
    )

    assert entry["parts"] == [
        {"type": "text", "content": "哈哈"},
        {"type": "face"},
    ]


def test_attachment_order_is_preserved() -> None:
    """类型不再由「第一个组件」决定：表情在前还是图片在前，如实给出。"""
    face_first = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="face",
            attachments=("face", "image"),
        )
    )
    image_first = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="image",
            attachments=("image", "face"),
        )
    )

    assert face_first["parts"] == [{"type": "face"}, {"type": "image"}]
    assert image_first["parts"] == [{"type": "image"}, {"type": "face"}]


def test_repeated_attachment_keeps_count_by_repetition() -> None:
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="image",
            attachments=("image", "image"),
        )
    )

    assert entry["parts"] == [{"type": "image"}, {"type": "image"}]


def test_filename_survives_when_file_is_not_first() -> None:
    """文件名不再要求文件排在第一位：旧实现在这里把「报告.pdf」丢了。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="看这个",
            timestamp=TS,
            sender_id="123456",
            kind="image",
            attachments=("image", "file"),
            file_name="报告.pdf",
        )
    )

    assert entry["parts"] == [
        {"type": "text", "content": "看这个"},
        {"type": "image"},
        {"type": "file", "filename": "报告.pdf"},
    ]


def test_text_plus_poke_keeps_target() -> None:
    """文字 + 戳一戳：此前整条 poke 连目标一起消失（kind 被降级成 text）。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="戳你一下",
            timestamp=TS,
            sender_id="123456",
            kind="text",
            attachments=("poke",),
            target_id="10000",
            target_is_self=True,
        )
    )

    assert entry["parts"] == [
        {"type": "text", "content": "戳你一下"},
        {"type": "poke", "target": {"id": "10000", "is_self": True}},
    ]


def test_poke_plus_attachment_keeps_both() -> None:
    """戳 + 图：不再因为 poke 提前返回而丢掉附件。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="",
            timestamp=TS,
            sender_id="123456",
            kind="poke",
            attachments=("image",),
            target_id="10000",
            target_is_self=True,
        )
    )

    assert entry["parts"] == [
        {"type": "poke", "target": {"id": "10000", "is_self": True}},
        {"type": "image"},
    ]


def test_at_and_reply_parts_carry_target_role_id() -> None:
    """@ 与引用作为段下发目标 ID，且不带仅供渲染的 ``is_self``（ADR-0009）。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="这游戏不错",
            timestamp=TS,
            sender_id="123456",
            parts=(
                {"type": "at", "target_role_id": "10000", "is_self": True},
                {"type": "reply", "target_role_id": "10000", "is_self": True},
                {"type": "text"},
            ),
        )
    )

    assert entry["parts"] == [
        {"type": "at", "target_role_id": "10000"},
        {"type": "reply", "target_role_id": "10000"},
        {"type": "text", "content": "这游戏不错"},
    ]
    assert "mentions_me" not in entry
    assert "replies_me" not in entry


def test_at_without_target_is_dropped() -> None:
    """历史里凑不出目标 ID 时宁可少一段，也不能下发一个空 @。"""
    entry = _entry(
        ChatMessage(
            sender_name="小红",
            text="这游戏不错",
            timestamp=TS,
            sender_id="123456",
            parts=({"type": "at"}, {"type": "text"}),
        )
    )

    assert entry["type"] == "text"
    assert entry["content"] == "这游戏不错"


def test_plain_entry_has_no_decoration_segments() -> None:
    entry = _entry(
        ChatMessage(sender_name="小红", text="在吗", timestamp=TS, sender_id="123456")
    )

    assert entry["type"] == "text"
    assert "parts" not in entry
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
