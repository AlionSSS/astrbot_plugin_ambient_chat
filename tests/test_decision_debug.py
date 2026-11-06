"""调试需显式启用，且数量与磁盘占用均有上限。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import httpx
import pytest
from astrbot_plugin_ambient_chat.rules.models import DecisionModel
from astrbot_plugin_ambient_chat.rules.store import RuleStore
from astrbot_plugin_ambient_chat.speech import debug_log
from astrbot_plugin_ambient_chat.speech.decision import (
    DecisionRouter,
    NoulPrompt,
    probe_decision_model,
)


def test_debug_setting_defaults_off_and_persists(tmp_path):
    store = RuleStore(tmp_path)
    assert not store.settings.decision_debug_enabled
    store.update_settings({"decision_debug_enabled": True})
    restored = RuleStore(tmp_path)
    restored.load(ZoneInfo("UTC"))
    assert restored.settings.decision_debug_enabled
    restored.update_settings({"decision_debug_enabled": False})
    store.load(ZoneInfo("UTC"))
    assert not store.settings.decision_debug_enabled


def test_debug_writer_requires_opt_in_and_rotates_count_and_bytes(
    tmp_path, monkeypatch
):
    directory = tmp_path / "debug"
    assert debug_log.write_debug_record("test", {}, directory=directory) is None
    assert not directory.exists()
    monkeypatch.setattr(debug_log, "MAX_DEBUG_RECORDS", 3)
    monkeypatch.setattr(debug_log, "MAX_DEBUG_BYTES", 2000)
    paths = [
        debug_log.write_debug_record(
            "test", {"i": i}, directory=directory, enabled=True
        )
        for i in range(5)
    ]
    assert len(list(directory.glob("*.json"))) == 3
    assert not paths[0].exists() and paths[-1].exists()
    latest = debug_log.write_debug_record(
        "test", "x" * 1500, directory=directory, enabled=True
    )
    assert latest.exists()
    assert sum(path.stat().st_size for path in directory.glob("*.json")) <= 2000
    assert (
        debug_log.write_debug_record(
            "test", "x" * 2000, directory=directory, enabled=True
        )
        is None
    )
    assert latest.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["official", "custom", "probe"])
async def test_all_decision_paths_follow_saved_debug_switch(tmp_path, kind):
    store = RuleStore(tmp_path)
    model = DecisionModel("dm_test", "test", "https://example.test/v1", "m")
    store.settings.decision_models = [model]
    context = SimpleNamespace(
        decision=SimpleNamespace(
            ask=AsyncMock(return_value={"q": SimpleNamespace(probability=0.8)})
        )
    )

    def handler(request):
        return httpx.Response(
            200,
            json={
                "answers": {
                    "q": {"noul": 0.8},
                    "noul": {"type": "noul", "noul": 0.8},
                    "choice": {
                        "type": "choice",
                        "choice": "calm",
                        "confidence": 0.7,
                        "probabilities": {"angry": 0.2, "calm": 0.7, "happy": 0.1},
                    },
                    "score": {
                        "type": "score",
                        "score": 1.4,
                        "confidence": 0.6,
                        "legend": {"0": "可以等", "1": "本周处理", "2": "今天处理"},
                        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                    },
                }
            },
        )

    transport = httpx.MockTransport(handler)
    router = DecisionRouter(context, store, transport=transport)

    async def ask():
        if kind == "probe":
            await probe_decision_model(model, transport=transport, store=store)
        else:
            await router.probabilities(
                "custom:dm_test" if kind == "custom" else None,
                {"q": NoulPrompt("要接话吗？")},
                {"private": "chat"},
            )

    directory = tmp_path / "decision_debug"
    try:
        await ask()
        assert not directory.exists()
        store.settings.decision_debug_enabled = True
        await ask()
        records = list(directory.glob("*.json"))
        assert len(records) == 1
        assert (
            json.loads(records[0].read_text(encoding="utf-8"))["kind"]
            == f"decision-{kind}"
        )
        store.settings.decision_debug_enabled = False
        await ask()
        assert len(list(directory.glob("*.json"))) == 1
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_disable_debug_during_request_prevents_late_write(tmp_path):
    store = RuleStore(tmp_path)
    store.settings.decision_debug_enabled = True
    entered, release = asyncio.Event(), asyncio.Event()

    async def ask(*args, **kwargs):
        entered.set()
        await release.wait()
        raise RuntimeError("temporary outage")

    router = DecisionRouter(SimpleNamespace(decision=SimpleNamespace(ask=ask)), store)
    task = asyncio.create_task(router.probabilities(None, {"q": NoulPrompt("接话？")}))
    await entered.wait()
    store.settings.decision_debug_enabled = False
    release.set()
    with pytest.raises(Exception, match="temporary outage"):
        await task
    assert not (tmp_path / "decision_debug").exists()
    await router.close()
