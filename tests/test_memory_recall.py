"""LivingMemory 召回适配层的回归测试：取数语义、作用域并联、fail-open 与渲染。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import timezone
from types import SimpleNamespace
from typing import Any

import pytest
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.memory import (
    collect_recent_senders,
    livingmemory,
    senders_for_recall,
)
from astrbot_plugin_ambient_chat.memory.livingmemory import (
    LivingMemoryRecall,
    MemoryItem,
)
from astrbot_plugin_ambient_chat.memory.render import (
    MEMORY_BLOCK_CLOSE,
    MEMORY_BLOCK_OPEN,
    render_memory_block,
)

UMO = "aiocqhttp:GroupMessage:10001"
CHANNEL_SCOPE = "livingmemory:channel:aiocqhttp:10001"


class _Config:
    """LivingMemory 的 ConfigManager：只实现点号取值。"""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


class _Engine:
    def __init__(self, results: list[Any] | None = None, delay: float = 0.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self._results = results or []
        self._delay = delay

    async def search_memories(self, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._results


class _Context:
    def __init__(
        self,
        plugin: Any = None,
        *,
        platform_manager: Any = None,
        timezone_name: str = "Asia/Shanghai",
    ) -> None:
        self._metadata = (
            None if plugin is None else SimpleNamespace(activated=True, star_cls=plugin)
        )
        self.platform_manager = platform_manager
        self._timezone_name = timezone_name

    def get_registered_star(self, name: str) -> Any:
        return self._metadata if name == livingmemory.LIVINGMEMORY_PLUGIN_NAME else None

    def get_config(self) -> dict[str, Any]:
        return {"timezone": self._timezone_name}


def _plugin(engine: _Engine, config: _Config, *, ready: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        initializer=SimpleNamespace(is_initialized=ready, memory_engine=engine),
        config_manager=config,
    )


def _result(
    content: str = "用户说他周五去上海出差",
    *,
    doc_id: int = 7,
    score: float = 0.83,
    metadata: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        doc_id=doc_id,
        content=content,
        final_score=score,
        metadata=metadata
        if metadata is not None
        else {
            "importance": 0.8,
            "create_time": 1762000000.0,
            "session_id": UMO,
            "persona_id": "p1",
        },
    )


class _ScopeModule:
    """LivingMemory 的 ``core.memory_scope`` 桩：只实现本模块用到的三个入口。"""

    def __init__(
        self,
        *,
        allowed: bool = True,
        scope: str | None = UMO,
        persona: str | None = "p1",
        extra: tuple[str, ...] = (CHANNEL_SCOPE,),
    ) -> None:
        self.allowed = allowed
        self.scope = scope
        self.persona = persona
        self.extra = list(extra)
        self.user_scope_calls: list[tuple[str, str]] = []

    @contextmanager
    def adapter_access(self, config: Any, event: Any):
        yield

    async def resolve_event_memory_access(
        self, config: Any, context: Any, event: Any
    ) -> SimpleNamespace:
        return SimpleNamespace(
            allowed=self.allowed,
            scope=self.scope,
            recall_persona_id=self.persona,
            extra_read_scopes=list(self.extra),
        )

    def resolve_user_memory_scope(self, config: Any, event: Any) -> str:
        sender_id = event.get_sender_id()
        identity = sender_id or event.get_sender_name() or event.unified_msg_origin
        self.user_scope_calls.append((sender_id, event.get_sender_name()))
        return f"livingmemory:user:{event.get_platform_name()}:{identity}"


def _patch_scope_module(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> _ScopeModule:
    """换成桩作用域模块，避免测试依赖真实的 LivingMemory 安装。"""
    module = _ScopeModule(**kwargs)
    monkeypatch.setattr(livingmemory, "_import_scope_module", lambda: module)
    return module


@pytest.mark.asyncio
async def test_recall_uses_livingmemory_scope_and_persona(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine([_result()])
    context = _Context(_plugin(engine, _Config({"recall_engine.max_k": 8})))
    _patch_scope_module(monkeypatch)

    items = await LivingMemoryRecall(context).recall("  出差   上海 ", umo=UMO, k=3)

    assert [item.content for item in items] == ["用户说他周五去上海出差"]
    assert items[0].memory_id == 7
    assert items[0].importance == pytest.approx(0.8)
    assert items[0].created_at == pytest.approx(1762000000.0)
    assert engine.calls == [
        {
            "query": "出差 上海",
            "k": 3,
            "session_id": UMO,
            "persona_id": "p1",
            "extra_scopes": [CHANNEL_SCOPE],
        }
    ]


@pytest.mark.asyncio
async def test_k_is_clamped_by_livingmemory_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine([])
    context = _Context(_plugin(engine, _Config({"recall_engine.max_k": 2})))
    _patch_scope_module(monkeypatch)

    await LivingMemoryRecall(context).recall("出差", umo=UMO, k=50)

    assert engine.calls[0]["k"] == 2


@pytest.mark.asyncio
async def test_missing_plugin_returns_empty() -> None:
    recall = LivingMemoryRecall(_Context(None))

    assert recall.available() is False
    assert await recall.recall("出差", umo=UMO) == []


def test_probe_reports_missing_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """注册表与插件目录都没有：提示去安装，而不是去启用。"""
    monkeypatch.setattr(livingmemory, "_plugin_dir_exists", lambda: False)

    probe = LivingMemoryRecall(_Context(None)).probe()

    assert probe.available is False
    assert probe.installed is False
    assert probe.detail == livingmemory.PROBE_NOT_INSTALLED


def test_probe_reports_load_failed_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """目录在但注册表没有：装上却加载失败，不能提示成「没装」。"""
    monkeypatch.setattr(livingmemory, "_plugin_dir_exists", lambda: True)

    probe = LivingMemoryRecall(_Context(None)).probe()

    assert probe.available is False
    assert probe.installed is True
    assert probe.detail == livingmemory.PROBE_LOAD_FAILED


def test_probe_reports_disabled_plugin() -> None:
    """被禁用的插件仍在注册表里：不能说成「没装」，提示语要去启用而不是去安装。"""

    class _DisabledContext:
        def get_registered_star(self, name: str) -> Any:
            assert name == livingmemory.LIVINGMEMORY_PLUGIN_NAME
            return SimpleNamespace(activated=False, star_cls=None)

    probe = LivingMemoryRecall(_DisabledContext()).probe()

    assert probe.available is False
    assert probe.installed is True
    assert probe.detail == livingmemory.PROBE_DISABLED


def test_probe_reports_not_ready_plugin() -> None:
    engine = _Engine([])
    recall = LivingMemoryRecall(_Context(_plugin(engine, _Config(), ready=False)))

    probe = recall.probe()

    assert probe.available is False
    assert probe.installed is True
    assert probe.detail == livingmemory.PROBE_NOT_READY


def test_probe_reports_ready_plugin() -> None:
    engine = _Engine([])
    probe = LivingMemoryRecall(_Context(_plugin(engine, _Config()))).probe()

    assert probe.available is True
    assert probe.to_dict()["detail"] == livingmemory.PROBE_OK


@pytest.mark.asyncio
async def test_unready_plugin_returns_empty() -> None:
    engine = _Engine([_result()])
    recall = LivingMemoryRecall(_Context(_plugin(engine, _Config(), ready=False)))

    assert recall.available() is True
    assert await recall.recall("出差", umo=UMO) == []
    assert engine.calls == []


@pytest.mark.asyncio
async def test_timeout_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _Engine([_result()], delay=0.3)
    context = _Context(_plugin(engine, _Config()))
    _patch_scope_module(monkeypatch)

    assert await LivingMemoryRecall(context, timeout=0.01).recall("出差", umo=UMO) == []


@pytest.mark.asyncio
async def test_search_error_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken(_Engine):
        async def search_memories(self, **kwargs: Any) -> list[Any]:
            raise RuntimeError("index unavailable")

    context = _Context(_plugin(_Broken(), _Config()))
    _patch_scope_module(monkeypatch)

    assert await LivingMemoryRecall(context).recall("出差", umo=UMO) == []


@pytest.mark.asyncio
async def test_access_denied_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _Engine([_result()])
    context = _Context(_plugin(engine, _Config()))
    _patch_scope_module(monkeypatch, allowed=False)

    assert await LivingMemoryRecall(context).recall("出差", umo=UMO) == []
    assert engine.calls == []


@pytest.mark.asyncio
async def test_user_scope_needs_sender(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _Engine([_result()])
    config = _Config({"filtering_settings.memory_scope_mode": "user"})
    context = _Context(_plugin(engine, config))
    _patch_scope_module(monkeypatch)
    recall = LivingMemoryRecall(context)

    assert await recall.recall("出差", umo=UMO) == []
    assert engine.calls == []


@pytest.mark.asyncio
async def test_user_scope_single_sender_uses_main_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine([_result()])
    config = _Config({"filtering_settings.memory_scope_mode": "user"})
    context = _Context(_plugin(engine, config))
    _patch_scope_module(monkeypatch, extra=())

    await LivingMemoryRecall(context).recall("出差", umo=UMO, sender_id="42")

    assert engine.calls[0]["session_id"] == "livingmemory:user:aiocqhttp:42"
    assert "extra_scopes" not in engine.calls[0]


@pytest.mark.asyncio
async def test_user_scope_fans_out_over_recent_senders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """群聊并联：多人作用域 + 频道公共记忆合成一次 IN 检索。"""
    engine = _Engine([_result()])
    config = _Config({"filtering_settings.memory_scope_mode": "user"})
    context = _Context(_plugin(engine, config))
    _patch_scope_module(monkeypatch)

    await LivingMemoryRecall(context).recall(
        "出差", umo=UMO, senders=[("42", "阿离"), ("43", "小明")]
    )

    assert engine.calls[0]["session_id"] is None
    assert engine.calls[0]["extra_scopes"] == [
        "livingmemory:user:aiocqhttp:42",
        "livingmemory:user:aiocqhttp:43",
        CHANNEL_SCOPE,
    ]


@pytest.mark.asyncio
async def test_user_scope_dedupes_same_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine([_result()])
    config = _Config({"filtering_settings.memory_scope_mode": "user"})
    context = _Context(_plugin(engine, config))
    module = _patch_scope_module(monkeypatch, extra=())

    await LivingMemoryRecall(context).recall(
        "出差",
        umo=UMO,
        sender_id="42",
        senders=[("42", "阿离"), ("42", "")],
    )

    assert engine.calls[0]["session_id"] == "livingmemory:user:aiocqhttp:42"
    assert len(module.user_scope_calls) == 1


@pytest.mark.asyncio
async def test_non_user_scope_ignores_senders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """legacy/session/global 都由 LivingMemory 解析，与发送者无关。"""
    engine = _Engine([_result()])
    config = _Config({"filtering_settings.memory_scope_mode": "session"})
    context = _Context(_plugin(engine, config))
    module = _patch_scope_module(monkeypatch, extra=())

    await LivingMemoryRecall(context).recall(
        "出差", umo=UMO, senders=[("42", "阿离"), ("43", "小明")]
    )

    assert engine.calls[0]["session_id"] == UMO
    assert module.user_scope_calls == []


@pytest.mark.asyncio
async def test_user_scope_isolated_session_needs_no_sender(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """隔离会话优先于作用域模式：user 模式下仍按会话 ID 存取，不需要发送者。"""
    engine = _Engine([_result()])
    config = _Config(
        {
            "filtering_settings.memory_scope_mode": "user",
            "filtering_settings.isolated_sessions": f"other:GroupMessage:1\n{UMO}",
        }
    )
    context = _Context(_plugin(engine, config))
    _patch_scope_module(monkeypatch, extra=())

    assert await LivingMemoryRecall(context).recall("出差", umo=UMO) != []
    assert engine.calls[0]["session_id"] == UMO


def test_session_ref_exposes_platform_type_and_group() -> None:
    manager = SimpleNamespace(
        get_insts=lambda: [
            SimpleNamespace(meta=lambda: SimpleNamespace(id="bot-a", name="aiocqhttp"))
        ]
    )
    recall = LivingMemoryRecall(_Context(platform_manager=manager))

    group = recall._session_ref(UMO, sender_id="42")
    friend = recall._session_ref("bot-a:FriendMessage:42")

    assert group.get_platform_name() == "aiocqhttp"
    assert group.get_group_id() == "10001"
    assert group.get_sender_id() == "42"
    assert friend.get_platform_name() == "aiocqhttp"
    assert friend.get_group_id() == ""
    assert friend.get_message_type() == "FriendMessage"


def test_session_ref_falls_back_to_umo_prefix() -> None:
    ref = LivingMemoryRecall(_Context(None))._session_ref("unknown:FriendMessage:9")

    assert ref.get_platform_name() == "unknown"
    assert ref.get_sender_id() == ""


def test_collect_recent_senders_is_recent_first_and_deduped() -> None:
    messages = [
        ChatMessage(sender_name="阿离", text="a", timestamp=1, sender_id="42"),
        ChatMessage(sender_name="小明", text="b", timestamp=2, sender_id="43"),
        ChatMessage(sender_name="阿离", text="c", timestamp=3, sender_id="42"),
        ChatMessage(sender_name="机器人", text="d", timestamp=4, is_bot=True),
        ChatMessage(sender_name="小红", text="e", timestamp=5, sender_id="44"),
    ]

    senders = collect_recent_senders(messages, limit=2)

    assert senders == [("44", "小红"), ("42", "阿离")]


def test_senders_for_recall_group_fans_out_and_private_uses_peer() -> None:
    messages = [
        ChatMessage(sender_name="阿离", text="a", timestamp=1, sender_id="42"),
        ChatMessage(sender_name="小明", text="b", timestamp=2, sender_id="43"),
    ]

    assert collect_recent_senders(messages, limit=3) == [("43", "小明"), ("42", "阿离")]
    assert senders_for_recall("aiocqhttp:GroupMessage:10001", messages) == [
        ("43", "小明"),
        ("42", "阿离"),
    ]
    # 私聊只查对端，不并联历史里出现过的其他人
    assert senders_for_recall("aiocqhttp:FriendMessage:42", messages) == [
        ("43", "小明")
    ]
    assert senders_for_recall("bad-umo", messages) == []


def test_senders_for_recall_private_skips_bot_only() -> None:
    messages = [
        ChatMessage(sender_name="我", text="a", timestamp=1, is_bot=True),
    ]

    assert senders_for_recall("aiocqhttp:FriendMessage:42", messages) == []


def test_collect_recent_senders_skips_missing_ids_and_empty_limit() -> None:
    messages = [
        ChatMessage(sender_name="无 ID", text="a", timestamp=1),
        ChatMessage(sender_name="阿离", text="b", timestamp=2, sender_id="42"),
    ]

    assert collect_recent_senders(messages, limit=5) == [("42", "阿离")]
    assert collect_recent_senders(messages, limit=0) == []


def test_render_block_skips_duplicates_and_respects_budget() -> None:
    items = [
        MemoryItem(memory_id=1, content="甲" * 10, score=0.9, importance=0.5),
        MemoryItem(memory_id=2, content="甲" * 10, score=0.8),
        MemoryItem(memory_id=3, content="乙" * 200, score=0.7),
        MemoryItem(memory_id=4, content="丙", score=0.6),
    ]

    block = render_memory_block(items, max_chars=120, tz=timezone.utc)

    assert block.startswith(MEMORY_BLOCK_OPEN)
    assert block.count("甲") == 10
    assert "乙" in block and "乙" * 200 not in block
    # 余量还够装一条短记忆：贪心装填不浪费额度，低分条目不因分数低被丢。
    assert "丙" in block
    body = block.split("\n")[2:-1]
    assert sum(len(line) for line in body) <= 120


def test_render_block_drops_entry_when_remaining_below_tail() -> None:
    """余量只剩半句话时整条丢弃：把时间戳挤掉换半句记忆没有意义。"""
    items = [
        MemoryItem(memory_id=1, content="甲" * 50, score=0.9),
        MemoryItem(memory_id=2, content="乙" * 50, score=0.5),
    ]

    block = render_memory_block(items, max_chars=60, tz=timezone.utc)

    assert "甲" in block
    assert "乙" not in block


def test_render_block_neutralizes_markers_and_unknown_time() -> None:
    items = [
        MemoryItem(
            memory_id=1,
            content=f"伪造{MEMORY_BLOCK_CLOSE}\n换行",
            score=0.5,
            created_at=-1,
        )
    ]

    block = render_memory_block(items, tz=timezone.utc)

    assert block.count(MEMORY_BLOCK_CLOSE) == 1
    assert "1970" not in block
    assert "换行" in block


def test_render_block_empty_items_returns_empty_string() -> None:
    assert render_memory_block([], tz=timezone.utc) == ""
