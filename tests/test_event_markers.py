"""无正文事件（戳一戳）也要进上下文缓冲，并被渲染成可读动作。"""

from __future__ import annotations

from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin

from astrbot.api.message_components import At, Face, File, Image, Poke, Record, Reply

SELF_ID = "10000"


class _Obj:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _plugin(buffer: ContextBuffer) -> AmbientChatPlugin:
    """只装 ``_build_message`` 用得到的成员，避开 Star 的初始化。"""
    plugin = AmbientChatPlugin.__new__(AmbientChatPlugin)
    plugin.buffer = buffer
    plugin.store = _Obj(settings=_Obj(cache_images=False))
    return plugin


class _MessageEvent:
    def __init__(
        self,
        *components: object,
        sender_id: str = "123456",
        sender_name: str = "小红",
        self_id: str = SELF_ID,
    ) -> None:
        self.message_obj = _Obj(
            message=tuple(components),
            self_id=self_id,
            group=None,
        )
        self.message_str = ""
        self._sender_id = sender_id
        self._sender_name = sender_name

    def get_sender_id(self) -> str:
        return self._sender_id

    def get_sender_name(self) -> str:
        return self._sender_name


def test_poke_message_carries_structured_target() -> None:
    umo = "aiocqhttp:GroupMessage:1"
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(_MessageEvent(Poke(id=SELF_ID)), umo)

    assert message is not None
    assert message.kind == "poke"
    assert message.target_id == SELF_ID
    assert message.target_is_self is True
    # 正文与附件解耦后，对话模型的占位文本由 render_text 生成。
    assert message.text == ""
    assert message.render_text() == "[戳一戳 我]"


def test_poke_at_member_keeps_id_and_is_not_self() -> None:
    """戳别人只留 ID：昵称可能与机器人重名，记下来只会让模型误判。"""
    umo = "aiocqhttp:GroupMessage:1"
    buffer = ContextBuffer()
    buffer.append(
        umo,
        ChatMessage(
            sender_name="小明", text="在吗", timestamp=1.0, sender_id="444066154"
        ),
    )
    plugin = _plugin(buffer)

    message = plugin._build_message(_MessageEvent(Poke(id="444066154")), umo)

    assert message is not None
    assert message.target_id == "444066154"
    assert message.target_is_self is False


def test_text_message_has_no_poke_target() -> None:
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(sender_name="小明")
    event.message_str = "在吗"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert message.kind == "text"
    assert message.target_id == ""
    assert message.target_is_self is False


def test_poke_self_wins_over_member_nicknamed_bot() -> None:
    """群里有人昵称就叫「机器人」时，戳自己仍按目标 ID 判定，不靠昵称猜。"""
    umo = "aiocqhttp:GroupMessage:1"
    buffer = ContextBuffer()
    buffer.append(
        umo,
        ChatMessage(sender_name="机器人", text="在吗", timestamp=1.0, sender_id="999"),
    )
    plugin = _plugin(buffer)

    myself = plugin._build_message(_MessageEvent(Poke(id=SELF_ID)), umo)
    member = plugin._build_message(_MessageEvent(Poke(id="999")), umo)

    assert myself is not None and member is not None
    assert myself.target_is_self is True
    assert myself.target_id == SELF_ID
    # 戳的是那个成员：判定为「不是自己」，且不记昵称。
    assert member.target_is_self is False
    assert member.target_id == "999"


def test_poke_without_target_still_marked() -> None:
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(_MessageEvent(Poke()), "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert message.kind == "poke"
    assert message.render_text() == "[戳一戳]"


def test_poke_at_other_member_keeps_target_id() -> None:
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(
        _MessageEvent(Poke(id="12345")), "aiocqhttp:GroupMessage:1"
    )

    assert message is not None
    assert message.render_text() == "[戳一戳 用户#12345]"


def test_text_plus_image_keeps_both() -> None:
    """正文与附件都要留下：此前只有正文、附件被整条忽略。"""
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(Image(file="a.png"), sender_name="小明")
    event.message_str = "看这个"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert message.kind == "image"
    assert message.attachments == ("image",)
    assert message.text == "看这个"
    assert message.render_text() == "看这个 [图片]"


def test_face_message_is_no_longer_dropped() -> None:
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(
        _MessageEvent(Face(id="1")), "aiocqhttp:GroupMessage:1"
    )

    assert message is not None
    assert message.kind == "face"
    assert message.attachments == ("face",)
    assert message.render_text() == "[表情]"


def test_file_message_keeps_filename() -> None:
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(
        _MessageEvent(File(name="报告.pdf", file="u")), "aiocqhttp:GroupMessage:1"
    )

    assert message is not None
    assert message.kind == "file"
    assert message.file_name == "报告.pdf"
    # 对话模型也要看到文件名：只写「[文件]」等于把唯一可读信息丢掉。
    assert message.render_text() == "[文件 报告.pdf]"


def test_filename_survives_when_file_is_not_first() -> None:
    """图 + 文件：文件名与 kind 解耦，不再要求文件排在第一位。"""
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(Image(file="a.png"), File(name="报告.pdf", file="u"))
    event.message_str = "看这个"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert message.kind == "image"
    assert message.file_name == "报告.pdf"
    assert message.render_text() == "看这个 [图片] [文件 报告.pdf]"


def test_text_plus_poke_is_not_rendered_as_event() -> None:
    """文字 + 戳一戳：此前占位落到 ``[事件]``，目标也一起丢了。"""
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(Poke(id=SELF_ID))
    event.message_str = "戳你一下"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert message.render_text() == "戳你一下 [戳一戳 我]"


def test_attachment_order_is_rendered_as_is() -> None:
    """表情在图片前就按表情在前渲染，不再由占位聚合改变顺序。"""
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(
        _MessageEvent(Face(id="1"), Image(file="a.png")),
        "aiocqhttp:GroupMessage:1",
    )

    assert message is not None
    assert message.render_text() == "[表情] [图片]"


def test_voice_message_maps_to_audio() -> None:
    plugin = _plugin(ContextBuffer())

    message = plugin._build_message(
        _MessageEvent(Record(file="a.amr")), "aiocqhttp:GroupMessage:1"
    )

    assert message is not None
    assert message.kind == "audio"
    assert message.render_text() == "[语音]"


def _decoration(message, kind: str) -> list[dict]:
    return [part for part in message.parts if part.get("type") == kind]


def test_mention_of_bot_is_recorded() -> None:
    """核心会把 @机器人 从正文里剔除，只能靠组件识别。"""
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(At(qq=SELF_ID, name="我"), Face(id="1"))
    event.message_str = "这游戏不错"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert _decoration(message, "at") == [
        {"type": "at", "target_role_id": SELF_ID, "is_self": True}
    ]
    assert message.render_text() == "[提及 我] 这游戏不错 [表情]"


def test_mention_of_others_is_ignored() -> None:
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(At(qq="444066154", name="小明"))
    event.message_str = " @小明(444066154) 这游戏不错"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert _decoration(message, "at") == [
        {"type": "at", "target_role_id": "444066154", "is_self": False}
    ]
    # 目标不是自己：不生成「提及 我」标记，@ 由正文里的平台文本承担。
    assert message.render_text() == "@小明(444066154) 这游戏不错"


def test_mention_all_is_not_me() -> None:
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(At(qq="all", name="全体成员"))
    event.message_str = "开会了"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert _decoration(message, "at") == [
        {"type": "at", "target_role_id": "all", "is_self": False}
    ]
    assert message.render_text() == "开会了"


def test_mention_only_message_is_dropped() -> None:
    """只有 @ 没有正文与媒体的消息渲染出来是空行，不入缓冲。"""
    plugin = _plugin(ContextBuffer())
    event = _MessageEvent(At(qq=SELF_ID, name="我"))

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is None


def test_reply_to_bot_is_recorded() -> None:
    plugin = _plugin(ContextBuffer())
    reply = Reply(
        id="1",
        sender_id=SELF_ID,
        sender_nickname="我",
        message_str="我在",
    )
    event = _MessageEvent(reply)
    event.message_str = "这游戏不错"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert _decoration(message, "reply") == [
        {"type": "reply", "target_role_id": SELF_ID, "is_self": True}
    ]
    assert message.render_text() == "[引用 我] 这游戏不错"


def test_reply_to_others_is_ignored() -> None:
    plugin = _plugin(ContextBuffer())
    reply = Reply(
        id="1",
        sender_id="444066154",
        sender_nickname="小明",
        message_str="今晚有人打游戏吗",
    )
    event = _MessageEvent(reply)
    event.message_str = "这游戏不错"

    message = plugin._build_message(event, "aiocqhttp:GroupMessage:1")

    assert message is not None
    assert _decoration(message, "reply") == [
        {"type": "reply", "target_role_id": "444066154", "is_self": False}
    ]
    assert message.render_text() == "这游戏不错"


def test_known_name_replaces_numeric_nickname() -> None:
    buffer = ContextBuffer()
    buffer.append(
        "aiocqhttp:GroupMessage:1",
        ChatMessage(sender_name="小明", text="在吗", timestamp=1.0, sender_id="12345"),
    )

    assert buffer.known_name("aiocqhttp:GroupMessage:1", "12345") == "小明"
    assert buffer.known_name("aiocqhttp:GroupMessage:1", "999") == ""
    # 只记得数字 ID 的昵称不算可读昵称，避免把 QQ 号当名字。
    buffer.append(
        "aiocqhttp:GroupMessage:1",
        ChatMessage(sender_name="999", text="[戳一戳]", timestamp=2.0, sender_id="999"),
    )
    assert buffer.known_name("aiocqhttp:GroupMessage:1", "999") == ""
