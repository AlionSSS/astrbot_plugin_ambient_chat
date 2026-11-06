"""配置提交及运行应用的关键失败语义，通过公开 interface 验证。"""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin
from astrbot_plugin_ambient_chat.rules.models import RuleError
from astrbot_plugin_ambient_chat.rules.store import RuleStore
from astrbot_plugin_ambient_chat.runtime.binding_lifecycle import BindingLifecycle
from astrbot_plugin_ambient_chat.runtime.configuration import ConfigurationApplication
from astrbot_plugin_ambient_chat.webapi import AmbientWebApi

TZ = ZoneInfo("UTC")


def rule(rule_id="r0", name="原规则"):
    return {
        "id": rule_id,
        "name": name,
        "triggers": [{"type": "interval", "every_minutes": 60}],
    }


def binding(binding_id="b0", session="0"):
    return {
        "id": binding_id,
        "platform_id": "test",
        "rule_id": "r0",
        "targets": [f"test:GroupMessage:{session}"],
    }


@pytest.fixture
def store(tmp_path):
    result = RuleStore(tmp_path)
    result.upsert_rule(rule(), TZ)
    result.upsert_rule(rule("unused"), TZ)
    result.upsert_binding(binding())
    result.upsert_binding(binding("running", "1"))
    result.set_binding_enabled("running", True)
    return result


CHANGES = [
    lambda s: s.upsert_rule(rule(name="改名"), TZ),
    lambda s: s.upsert_rule(rule("new"), TZ),
    lambda s: s.delete_rule("unused"),
    lambda s: s.upsert_binding({**binding(), "name": "改名"}),
    lambda s: s.upsert_binding(binding("new", "2")),
    lambda s: s.delete_binding("b0"),
    lambda s: s.set_binding_enabled("b0", True),
    lambda s: s.set_binding_enabled("running", False),
    lambda s: s.update_settings({"buffer_size": 321}),
]


@pytest.mark.parametrize("change", CHANGES)
@pytest.mark.parametrize("failure", ["fsync", "replace", "load_failed"])
def test_failed_commit_is_invisible_and_cannot_leak_into_later_save(
    store, monkeypatch, change, failure
):
    # 预先构建索引，同时保留旧对象，验证启停也没有原地改旧引用。
    old_binding = store.binding_for_umo("test:GroupMessage:1")
    before = copy.deepcopy(store.snapshot())
    disk = store.path.read_bytes()
    revision = store.revision
    with monkeypatch.context() as patch:
        if failure == "load_failed":
            patch.setattr(store, "load_failed", True)
        else:

            def fail(*args):
                raise OSError("模拟写入失败")

            patch.setattr(f"astrbot_plugin_ambient_chat.rules.store.os.{failure}", fail)
        with pytest.raises((OSError, RuleError)):
            change(store)
    assert store.snapshot() == before
    assert store.revision == revision
    assert store.path.read_bytes() == disk
    assert store.binding_for_umo("test:GroupMessage:1") is old_binding
    assert old_binding.enabled
    assert not store.path.with_suffix(".json.tmp").exists()

    store.update_settings({"concurrency": 2})
    reloaded = RuleStore(store.data_dir)
    reloaded.load(TZ)
    after = reloaded.snapshot()
    assert after["rules"] == before["rules"]
    assert after["bindings"] == before["bindings"]
    assert after["settings"]["buffer_size"] == before["settings"]["buffer_size"]


def test_committed_binding_replaces_old_object_and_rebuilds_index(store):
    old = store.get_binding("b0")
    assert store.binding_for_umo(old.targets[0]) is None
    enabled = store.set_binding_enabled("b0", True)
    assert enabled.enabled and not old.enabled
    assert store.binding_for_umo(enabled.targets[0]) is enabled
    reloaded = RuleStore(store.data_dir)
    reloaded.load(TZ)
    assert reloaded.get_binding("b0").enabled


ENDPOINTS = [
    ("save_rule", {"rule": rule(name="新名称")}),
    ("delete_rule", {"rule_id": "unused"}),
    ("save_binding", {"binding": {**binding(), "name": "新名称"}}),
    ("delete_binding", {"binding_id": "b0"}),
    ("save_settings", {"settings": {"buffer_size": 321}}),
    ("toggle_binding", {"binding_id": "b0", "enabled": True}),
    ("toggle_binding", {"binding_id": "running", "enabled": False}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint,payload", ENDPOINTS)
async def test_web_reports_saved_on_apply_failure_and_retry_does_not_write_again(
    store, monkeypatch, endpoint, payload
):
    apply = AsyncMock(side_effect=RuntimeError("模拟运行同步失败"))
    application = ConfigurationApplication(store, apply)
    plugin = SimpleNamespace(
        store=store,
        timezone=TZ,
        configuration=application,
        is_enabled=lambda: False,
    )
    plugin.buffer = ContextBuffer()
    plugin.lifecycle = BindingLifecycle(plugin)
    plugin.start_binding = lambda key: AmbientChatPlugin.start_binding(plugin, key)
    plugin.stop_binding = lambda key: AmbientChatPlugin.stop_binding(plugin, key)
    monkeypatch.setattr(
        "astrbot_plugin_ambient_chat.webapi.request",
        SimpleNamespace(json=AsyncMock(return_value=payload)),
    )
    monkeypatch.setattr(
        "astrbot_plugin_ambient_chat.webapi.json_response", lambda data: data
    )
    web = AmbientWebApi(plugin)
    response = await getattr(web, endpoint)()
    assert response["saved"] is True
    assert response["application"]["state"] == "failed"
    assert store.path.exists()
    reloaded = RuleStore(store.data_dir)
    reloaded.load(TZ)
    assert reloaded.snapshot() == store.snapshot()

    disk, revision = store.path.read_bytes(), store.revision
    apply.side_effect = None
    retried = await web.apply_configuration()
    assert retried["application"]["state"] == "applied"
    assert store.path.read_bytes() == disk
    assert store.revision == revision
    assert apply.await_count == 2


@pytest.mark.asyncio
async def test_late_failure_cannot_be_hidden_by_overlapping_success(store):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def apply():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
            raise RuntimeError("较早请求稍后失败")

    application = ConfigurationApplication(store, apply)
    first = asyncio.create_task(application.apply())
    await entered.wait()
    # 无关应用不等先前请求，且不会把仍在进行的应用显示成全部完成。
    async with asyncio.timeout(1):
        second = await application.apply()
    assert second["state"] == "applying"
    release.set()
    assert (await first)["state"] == "failed"
    assert (await application.apply())["state"] == "applied"


@pytest.mark.asyncio
async def test_older_success_does_not_erase_later_failure(store):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def apply():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        else:
            raise RuntimeError("较新请求失败")

    application = ConfigurationApplication(store, apply)
    first = asyncio.create_task(application.apply())
    await entered.wait()
    assert (await application.apply())["state"] == "failed"
    release.set()
    assert (await first)["state"] == "failed"


@pytest.mark.asyncio
async def test_cancelled_application_keeps_saved_configuration_for_retry(store):
    entered = asyncio.Event()

    async def apply():
        entered.set()
        await asyncio.Event().wait()

    store.update_settings({"buffer_size": 321})
    application = ConfigurationApplication(store, apply)
    task = asyncio.create_task(application.apply())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert application.snapshot()["state"] == "failed"
    reloaded = RuleStore(store.data_dir)
    reloaded.load(TZ)
    assert reloaded.settings.buffer_size == 321
