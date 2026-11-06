"""人格提示词的落点：``state.persona``，不混进 noul 的 instructions。"""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import Rule
from astrbot_plugin_ambient_chat.speech.intent import (
    NEUTRAL_PERSONA_PROMPT,
    IntentJudge,
)

TZ = ZoneInfo("Asia/Shanghai")
UMO = "aiocqhttp:FriendMessage:1"
PERSONA = "你是猫娘，说话带喵。"


class _Router:
    def __init__(self) -> None:
        self.prompts: dict[str, Any] = {}
        self.state: dict[str, Any] = {}

    async def probabilities(
        self,
        provider_id: str | None,
        prompts: Any,
        state: Any = None,
    ) -> dict[str, float]:
        self.prompts = dict(prompts)
        self.state = dict(state or {})
        return {"interested": 0.1, "want_new_topic": 0.2}


def _rule(inject_persona: bool) -> Rule:
    return Rule.parse(
        {
            "name": "测试规则",
            "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
            "intent": {"mode": "decision", "threshold": 0.7},
            "inject_persona": inject_persona,
        },
        TZ,
    )


@pytest.mark.asyncio
async def test_persona_goes_to_state_not_instructions(monkeypatch: Any) -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    async def fake_persona(_umo: str) -> str:
        return PERSONA

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("A", "在吗", time.time()))

    await judge.judge(_rule(True), UMO, buffer, datetime.now(TZ))

    assert router.state["persona"] == PERSONA
    for prompt in router.prompts.values():
        assert PERSONA not in prompt.instructions
        assert "猫娘" not in prompt.instructions
        # 只指路，不搬运原文：让模型知道去 state.persona 里找人设。
        assert "基于你的人设" in prompt.instructions


@pytest.mark.asyncio
async def test_state_is_structured(monkeypatch: Any) -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    async def fake_persona(_umo: str) -> str:
        return PERSONA

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    now = datetime(2026, 11, 4, 5, 40, tzinfo=TZ)
    buffer = ContextBuffer(maxlen=10)
    buffer.append(
        UMO,
        ChatMessage(
            "Lion",
            "在吗",
            datetime(2026, 11, 4, 5, 39, 12, tzinfo=TZ).timestamp(),
            sender_id="444066154",
        ),
    )

    await judge.judge(_rule(True), UMO, buffer, now)

    state = router.state
    assert "target" not in state
    # 没有身份回调时退化成「我」，state 始终有 self 键。
    assert state["self"] == {"name": "我"}
    assert state["now"] == {"date_time": "2026-11-04 05:40:00", "weekday": "周三"}
    # 私聊不带发送者 ID，对方昵称放进 scene。
    assert state["scene"] == {"type": "private", "peer_name": "Lion"}
    message = state["conversation"][0]
    assert message["sender_name"] == "Lion"
    assert "sender_id" not in message
    assert message["content"] == "在吗"
    assert message["time"] == "2026-11-04 05:39:12"


@pytest.mark.asyncio
async def test_state_carries_own_name(monkeypatch: Any) -> None:
    """state.self 告诉模型「你自己叫什么」，避免把账号昵称当成别的群友。"""
    router = _Router()
    judge = IntentJudge(
        None,  # type: ignore[arg-type]
        router,
        identity=lambda _umo: {"name": "蓝色小肥鱼", "id": "2670210348"},
    )

    async def fake_persona(_umo: str) -> str:
        return PERSONA

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("A", "在吗", time.time()))

    await judge.judge(_rule(True), UMO, buffer, datetime.now(TZ))

    assert router.state["self"] == {"name": "蓝色小肥鱼", "id": "2670210348"}


@pytest.mark.asyncio
async def test_state_self_falls_back_when_name_empty(monkeypatch: Any) -> None:
    router = _Router()
    judge = IntentJudge(
        None,  # type: ignore[arg-type]
        router,
        identity=lambda _umo: {"name": "  "},
    )

    async def fake_persona(_umo: str) -> str:
        return ""

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("A", "在吗", time.time()))

    await judge.judge(_rule(False), UMO, buffer, datetime.now(TZ))

    assert router.state["self"] == {"name": "我"}


@pytest.mark.asyncio
async def test_group_sender_carries_id(monkeypatch: Any) -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    async def fake_persona(_umo: str) -> str:
        return PERSONA

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    umo = "aiocqhttp:GroupMessage:123456"
    buffer = ContextBuffer(maxlen=10)
    buffer.append(
        umo,
        ChatMessage(
            "Lion",
            "在吗",
            datetime(2026, 11, 4, 5, 39, tzinfo=TZ).timestamp(),
            sender_id="444066154",
        ),
    )

    await judge.judge(_rule(True), umo, buffer, datetime(2026, 11, 4, 5, 40, tzinfo=TZ))

    # 不知道群名时不编造，只保留场景类型。
    assert router.state["scene"] == {"type": "group"}
    message = router.state["conversation"][0]
    # 群聊把名字与 ID 拆成两个字段，模型据此区分同名成员。
    assert message["sender_name"] == "Lion"
    assert message["sender_id"] == "444066154"


@pytest.mark.asyncio
async def test_group_name_comes_from_runtime_cache(monkeypatch: Any) -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    async def fake_persona(_umo: str) -> str:
        return PERSONA

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    umo = "aiocqhttp:GroupMessage:123456"
    buffer = ContextBuffer(maxlen=10)
    buffer.set_session_name(umo, "复古游戏开发群")
    buffer.append(umo, ChatMessage("A", "在吗", time.time()))

    await judge.judge(_rule(True), umo, buffer, datetime(2026, 11, 4, 5, 40, tzinfo=TZ))

    assert router.state["scene"] == {"type": "group", "name": "复古游戏开发群"}


class _AliasDb:
    async def get_umo_alias(self, umo: str) -> Any:
        return SimpleNamespace(user_alias="", auto_name="猫咪爱好者群")


@pytest.mark.asyncio
async def test_group_name_falls_back_to_core_alias(monkeypatch: Any) -> None:
    router = _Router()
    context = SimpleNamespace(get_db=lambda: _AliasDb())
    judge = IntentJudge(context, router)  # type: ignore[arg-type]

    async def fake_persona(_umo: str) -> str:
        return PERSONA

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    umo = "aiocqhttp:GroupMessage:123456"
    buffer = ContextBuffer(maxlen=10)
    buffer.append(umo, ChatMessage("A", "在吗", time.time()))

    await judge.judge(_rule(True), umo, buffer, datetime(2026, 11, 4, 5, 40, tzinfo=TZ))

    assert router.state["scene"]["name"] == "猫咪爱好者群"
    # 结果回填到缓冲，下次不再查库。
    assert buffer.session_name(umo) == "猫咪爱好者群"


@pytest.mark.asyncio
async def test_neutral_persona_when_disabled(monkeypatch: Any) -> None:
    router = _Router()
    judge = IntentJudge(None, router)  # type: ignore[arg-type]

    async def fake_persona(_umo: str) -> str:  # pragma: no cover - 不应被调用
        raise AssertionError("inject_persona=False 时不应读取人格")

    monkeypatch.setattr(judge, "persona_prompt", fake_persona)
    buffer = ContextBuffer(maxlen=10)
    buffer.append(UMO, ChatMessage("A", "在吗", time.time()))

    await judge.judge(_rule(False), UMO, buffer, datetime.now(TZ))

    assert router.state["persona"] == NEUTRAL_PERSONA_PROMPT
    for prompt in router.prompts.values():
        assert "若 state.persona 非空" not in prompt.instructions
