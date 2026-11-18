"""核心平台历史里的 ``[Image]`` 占位：必须还原成组件单元，不能当正文读。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from astrbot_plugin_ambient_chat.conversation.history import _row_to_message


def _row(*parts: dict, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(
        content={"type": role, "message": list(parts)},
        sender_name="小明",
        sender_id="444066154",
        created_at=datetime(2026, 11, 18, 16, 17),
    )


def test_group_history_image_placeholder_becomes_media_part() -> None:
    """核心把图片写成 plain "[Image]"：读历史时还原成图片段，正文为空。"""
    message = _row_to_message(_row({"type": "plain", "text": "[Image]"}))

    assert message is not None
    assert message.text == ""
    assert message.kind == "image"
    assert message.attachments == ("image",)
    assert message.render_text() == "[图片]"
    # 只有占位、没有正文的消息不该给出任何召回关键词。
    assert message.body_text() == ""


def test_group_history_keeps_text_around_placeholder() -> None:
    """文字与占位都要留下，并且按原顺序排列。"""
    message = _row_to_message(
        _row(
            {"type": "plain", "text": "看这个"},
            {"type": "plain", "text": "[Image]"},
        )
    )

    assert message is not None
    assert message.text == "看这个"
    assert [part["type"] for part in message.parts] == ["text", "image"]
    assert message.render_text() == "看这个 [图片]"


def test_group_history_media_placeholders_map_to_known_kinds() -> None:
    message = _row_to_message(
        _row(
            {"type": "plain", "text": "[Record]"},
            {"type": "plain", "text": "[Video]"},
            {"type": "plain", "text": "[File]"},
        )
    )

    assert message is not None
    assert message.text == ""
    assert [part["type"] for part in message.parts] == ["audio", "video", "file"]
    assert message.render_text() == "[语音] [视频] [文件]"


def test_group_history_at_stays_before_placeholder() -> None:
    """@ 与占位的相对顺序是事实：还原后不能挪位置。"""
    message = _row_to_message(
        _row(
            {"type": "at", "user_id": "2670210348", "name": "蓝色小肥鱼"},
            {"type": "plain", "text": "[Image]"},
        )
    )

    assert message is not None
    assert [part["type"] for part in message.parts] == ["at", "image"]
    assert message.text == ""


def test_group_history_unknown_brackets_stay_text() -> None:
    """只认组件名：用户自己打的方括号内容仍算正文。"""
    message = _row_to_message(
        _row(
            {"type": "plain", "text": "[重要] 明天见"},
            {"type": "plain", "text": "[abc]"},
        )
    )

    assert message is not None
    assert message.text == "[重要] 明天见 [abc]"
    assert [part["type"] for part in message.parts] == ["text"]
