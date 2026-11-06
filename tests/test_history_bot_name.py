"""预热历史里的机器人署名：核心写的是占位值 ``bot``，不能原样进缓冲。"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from astrbot_plugin_ambient_chat.conversation.history import (
    _history_to_message,
    _row_to_message,
)


def _bot_row() -> SimpleNamespace:
    return SimpleNamespace(
        content={"type": "bot", "message": [{"type": "plain", "text": "在"}]},
        sender_name="bot",
        sender_id="bot",
        created_at=datetime(2026, 11, 5, 20, 0),
    )


def test_group_history_replaces_placeholder_bot_name() -> None:
    message = _row_to_message(_bot_row(), "蓝色小肥鱼")

    assert message is not None
    assert message.is_bot
    assert message.sender_name == "蓝色小肥鱼"
    # 核心的 "bot" 不是真实账号 ID，不能当作成员 ID 带进判定 state。
    assert message.sender_id == ""
    assert message.display_name() == "蓝色小肥鱼"


def test_group_history_bot_name_falls_back_to_self_name() -> None:
    message = _row_to_message(_bot_row())

    assert message is not None
    assert message.sender_name == ""
    assert message.display_name() == "我"


def test_group_history_keeps_other_members_untouched() -> None:
    row = SimpleNamespace(
        content={"type": "user", "message": [{"type": "plain", "text": "在吗"}]},
        sender_name="小明",
        sender_id="444066154",
        created_at=datetime(2026, 11, 5, 20, 0),
    )

    message = _row_to_message(row, "蓝色小肥鱼")

    assert message is not None
    assert message.sender_name == "小明"
    assert message.sender_id == "444066154"
    assert not message.is_bot


def test_private_history_bot_gets_name() -> None:
    message = _history_to_message(
        {"role": "assistant", "content": "在的"}, "蓝色小肥鱼"
    )

    assert message is not None
    assert message.is_bot
    assert message.sender_name == "蓝色小肥鱼"
    assert message.display_name() == "蓝色小肥鱼"


def test_private_history_user_keeps_empty_name() -> None:
    message = _history_to_message({"role": "user", "content": "在吗"})

    assert message is not None
    assert not message.is_bot
    assert message.sender_name == ""
