"""冷却和未读统计不能随着上下文消息淘汰或缩容丢失。"""

from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage

UMO = "test:GroupMessage:1"


def user(index, sender="a"):
    return ChatMessage(sender, str(index), index + 100, sender_id=sender)


def test_activity_survives_eviction_resize_and_resets_on_reply_and_clear():
    buffer = ContextBuffer(maxlen=20)
    buffer.append(UMO, ChatMessage("我", "开场", 100, is_bot=True))
    for index in range(150):
        buffer.append(UMO, user(index))
    buffer.set_maxlen(10)
    assert len(buffer.recent(UMO, 1000)) == 10
    assert buffer.last_bot_at(UMO) == 100
    assert buffer.unread_count(UMO) == 150
    buffer.append(UMO, ChatMessage("我", "回应", 300, is_bot=True))
    assert buffer.last_bot_at(UMO) == 300
    assert buffer.unread_count(UMO) == 0
    buffer.append(UMO, user(201))
    buffer.clear(UMO)
    assert buffer.last_bot_at(UMO) is None
    assert buffer.unread_count(UMO) == 0


def test_ignored_senders_remove_unread_even_after_their_messages_are_evicted():
    buffer = ContextBuffer(maxlen=10)
    for index in range(30):
        buffer.append(UMO, user(index, "a" if index < 20 else "b"))
    buffer.set_ignored_senders(["a"])
    assert buffer.unread_count(UMO) == 10
    buffer.append(UMO, user(30, "a"))
    assert buffer.unread_count(UMO) == 10
    buffer.set_ignored_senders([])
    buffer.append(UMO, user(31, "a"))
    assert buffer.unread_count(UMO) == 11


def test_history_initializes_activity_without_capacity_limit_and_skipped_history_keeps_it():
    buffer = ContextBuffer(maxlen=10)
    history = [ChatMessage("我", "开场", 99, is_bot=True)]
    history.extend(user(index) for index in range(30))
    buffer.publish_history({UMO: history})
    assert buffer.last_bot_at(UMO) == 99
    assert buffer.unread_count(UMO) == 30
    buffer.publish_history({UMO: [user(40)]})
    assert buffer.last_bot_at(UMO) == 99
    assert buffer.unread_count(UMO) == 30
