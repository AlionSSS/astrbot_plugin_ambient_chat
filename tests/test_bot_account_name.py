"""机器人账号名：只支持 aiocqhttp，拿不到就返回空串，由调用方回退「我」。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin
from astrbot_plugin_ambient_chat.runtime.account import BotAccountName

PLATFORM_ID = "default"
SELF_ID = "10000"
UMO_FRIEND = f"{PLATFORM_ID}:FriendMessage:1"
UMO_GROUP = f"{PLATFORM_ID}:GroupMessage:960524757"


class _Bot:
    def __init__(
        self,
        nickname: str = "小助手",
        user_id: str = SELF_ID,
        card: str = "",
        fail: tuple[str, ...] = (),
    ) -> None:
        self.nickname = nickname
        self.user_id = user_id
        self.card = card
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_action(self, action: str, **params: Any) -> Any:
        self.calls.append((action, params))
        if action in self.fail:
            raise RuntimeError("api down")
        if action == "get_login_info":
            return {"user_id": int(self.user_id), "nickname": self.nickname}
        if action == "get_group_member_info":
            return {"card": self.card, "nickname": self.nickname}
        raise AssertionError(f"unexpected action: {action}")


class _Adapter:
    def __init__(self, bot: _Bot, name: str = "aiocqhttp") -> None:
        self.bot = bot
        self._meta = SimpleNamespace(name=name, id=PLATFORM_ID)

    def meta(self) -> SimpleNamespace:
        return self._meta


class _Context:
    def __init__(self, *adapters: _Adapter) -> None:
        self.platform_manager = SimpleNamespace(get_insts=lambda: list(adapters))


def _account(bot: _Bot, **kwargs: Any) -> BotAccountName:
    return BotAccountName(_Context(_Adapter(bot)), **kwargs)


@pytest.mark.asyncio
async def test_nickname_is_cached_across_calls() -> None:
    bot = _Bot("小助手")
    account = _account(bot)

    assert await account.name_for(UMO_FRIEND) == "小助手"
    assert await account.name_for(UMO_FRIEND) == "小助手"

    assert [action for action, _ in bot.calls] == ["get_login_info"]
    assert account.cached(UMO_FRIEND) == "小助手"


@pytest.mark.asyncio
async def test_self_id_is_forwarded_to_onebot() -> None:
    """多账号时 OneBot 要靠 self_id 路由到对应连接。"""
    bot = _Bot("小助手")
    account = _account(bot)

    await account.name_for(UMO_FRIEND, self_id=SELF_ID)

    assert bot.calls[0][1] == {"self_id": int(SELF_ID)}


@pytest.mark.asyncio
async def test_empty_nickname_falls_back_and_cools_down() -> None:
    bot = _Bot("")
    account = _account(bot, retry_seconds=600.0)

    assert await account.name_for(UMO_FRIEND) == ""
    # 冷却期内不重试，避免每条消息都打一次接口。
    assert await account.name_for(UMO_FRIEND) == ""
    assert [action for action, _ in bot.calls] == ["get_login_info"]


@pytest.mark.asyncio
async def test_nickname_equal_to_account_id_is_rejected() -> None:
    """部分实现把未设置昵称回填成 QQ 号，不能当可读名字用。"""
    bot = _Bot(SELF_ID)
    account = _account(bot)

    assert await account.name_for(UMO_FRIEND) == ""
    assert account.cached(UMO_FRIEND) == ""


@pytest.mark.asyncio
async def test_failed_query_returns_empty() -> None:
    bot = _Bot(fail=("get_login_info",))
    account = _account(bot)

    assert await account.name_for(UMO_FRIEND) == ""


@pytest.mark.asyncio
async def test_other_platform_is_ignored() -> None:
    bot = _Bot("小助手")
    account = BotAccountName(_Context(_Adapter(bot, name="telegram")))

    assert await account.name_for(UMO_FRIEND) == ""
    assert bot.calls == []


@pytest.mark.asyncio
async def test_group_card_wins_over_nickname() -> None:
    bot = _Bot("小助手", card="群里的我")
    account = _account(bot)

    assert await account.name_for(UMO_GROUP) == "群里的我"
    assert account.cached(UMO_GROUP) == "群里的我"
    # 群名片按群缓存，私聊仍用账号昵称。
    assert await account.name_for(UMO_FRIEND) == "小助手"
    assert [action for action, _ in bot.calls] == [
        "get_login_info",
        "get_group_member_info",
    ]


@pytest.mark.asyncio
async def test_group_card_failure_keeps_nickname() -> None:
    bot = _Bot("小助手", fail=("get_group_member_info",))
    account = _account(bot)

    assert await account.name_for(UMO_GROUP) == "小助手"
    assert account.cached(UMO_GROUP) == "小助手"


@pytest.mark.asyncio
async def test_prime_resolves_in_background() -> None:
    bot = _Bot("小助手")
    account = _account(bot)

    account.prime(UMO_FRIEND, self_id=SELF_ID)
    for _ in range(10):
        await asyncio.sleep(0)
        if account.cached(UMO_FRIEND):
            break

    assert account.cached(UMO_FRIEND) == "小助手"


@pytest.mark.asyncio
async def test_identity_carries_name_and_self_id() -> None:
    bot = _Bot("蓝色小肥鱼")
    account = _account(bot)

    await account.name_for(UMO_GROUP, self_id=SELF_ID)

    assert account.identity(UMO_GROUP) == {
        "name": "蓝色小肥鱼",
        "id": SELF_ID,
    }


def test_identity_falls_back_to_self_name() -> None:
    account = _account(_Bot("小助手"))

    assert account.identity(UMO_FRIEND) == {"name": "我"}


@pytest.mark.asyncio
async def test_identity_can_ignore_account_name() -> None:
    bot = _Bot("蓝色小肥鱼")
    account = _account(bot)

    await account.name_for(UMO_FRIEND, self_id=SELF_ID)

    assert account.identity(UMO_FRIEND)["name"] == "蓝色小肥鱼"
    assert account.identity(UMO_FRIEND, use_name=False) == {
        "name": "我",
        "id": SELF_ID,
    }


def test_bot_message_uses_signature_when_present() -> None:
    message = ChatMessage(sender_name="小助手", text="我在", timestamp=0.0, is_bot=True)

    assert message.display_name() == "小助手"
    assert message.structured_sender_name() == "小助手"
    assert message.render(None) == "小助手 (时间未知): 我在"  # type: ignore[arg-type]


def test_bot_message_falls_back_to_self_name() -> None:
    message = ChatMessage(sender_name="", text="我在", timestamp=0.0, is_bot=True)

    assert message.display_name() == "我"
    assert message.structured_sender_name() == "我"


def test_self_name_uses_cached_account_name() -> None:
    plugin = SimpleNamespace(account=SimpleNamespace(cached=lambda _umo: "小助手"))

    # 解析到名字就用，拿不到再回退。
    assert AmbientChatPlugin._self_name(plugin, UMO_FRIEND) == "小助手"

    plugin.account = SimpleNamespace(cached=lambda _umo: "")
    assert AmbientChatPlugin._self_name(plugin, UMO_FRIEND) == "我"
