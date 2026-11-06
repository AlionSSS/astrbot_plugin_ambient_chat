"""「更多设置」里按用户 ID 过滤上下文缓冲的行为。"""

import pytest

from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.media_store import MediaStore
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import Settings

UMO = "test:GroupMessage:1"


def message(sender_id, text, index=1, *, is_bot=False):
    return ChatMessage(
        sender_id if is_bot else f"用户{sender_id}",
        text,
        index,
        is_bot=is_bot,
        sender_id=sender_id,
    )


def test_settings_accepts_string_and_dedupes():
    settings = Settings.parse({"buffer_ignored_sender_ids": "10001, 10002 10001\n10003"})
    assert settings.buffer_ignored_sender_ids == ["10001", "10002", "10003"]
    assert settings.to_dict()["buffer_ignored_sender_ids"] == ["10001", "10002", "10003"]


@pytest.mark.asyncio
async def test_buffer_skips_ignored_senders_in_append_and_preheat(tmp_path):
    buffer = ContextBuffer(maxlen=10, media=MediaStore(tmp_path / "media"))
    buffer.set_ignored_senders(["10001"])
    try:
        buffer.append(UMO, message("10001", "刷屏", 1))
        buffer.append(UMO, message("10002", "正常", 2))
        buffer.append(UMO, message("999", "我说的", 3, is_bot=True))
        assert [item.sender_id for item in buffer.recent(UMO, 10)] == ["10002", "999"]

        # 预热走同样的过滤：历史里的被过滤用户不会借预热回到缓冲。
        buffer.clear(UMO)
        buffer.publish_history(
            {
                UMO: [
                    message("10001", "历史刷屏", 4),
                    message("10002", "历史对话", 5),
                ]
            }
        )
        assert [item.sender_id for item in buffer.recent(UMO, 10)] == ["10002"]
    finally:
        await buffer.close()


@pytest.mark.asyncio
async def test_set_ignored_senders_purges_existing_messages(tmp_path):
    buffer = ContextBuffer(maxlen=10, media=MediaStore(tmp_path / "media"))
    try:
        buffer.append(UMO, message("10001", "刷屏", 1))
        buffer.append(UMO, message("10002", "正常", 2))
        buffer.set_ignored_senders(["10001"])
        assert [item.sender_id for item in buffer.recent(UMO, 10)] == ["10002"]

        buffer.set_ignored_senders([])
        buffer.append(UMO, message("10001", "恢复", 3))
        assert [item.sender_id for item in buffer.recent(UMO, 10)] == [
            "10002",
            "10001",
        ]
    finally:
        await buffer.close()
