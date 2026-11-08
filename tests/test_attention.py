"""关注度：带回退先验的均值、永不为零的权重、加权抽取与档案读写（ADR-0034）。"""

import asyncio
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot_plugin_ambient_chat import webapi
from astrbot_plugin_ambient_chat.main import AmbientChatPlugin
from astrbot_plugin_ambient_chat.rules.models import Binding, Selection
from astrbot_plugin_ambient_chat.runtime.attention import (
    MIN_WEIGHT,
    PRIOR_STRENGTH,
    PRIOR_WEIGHT,
    Attention,
    AttentionStore,
    weighted_sample,
)
from astrbot_plugin_ambient_chat.runtime.scheduler import AmbientScheduler
from astrbot_plugin_ambient_chat.webapi import AmbientWebApi


def test_unobserved_session_sits_at_prior() -> None:
    """冷启动：没被判定过就是先验值，不会是 0，也不会抽不到。"""
    assert Attention().weight == PRIOR_WEIGHT


def test_weight_is_smoothed_mean_of_observations() -> None:
    """累计均值：10 次 1.0 之后是 (5*0.5 + 10) / (5 + 10)。"""
    record = Attention()
    for _ in range(10):
        record.observe(1.0, 1.0)

    assert record.weight == pytest.approx((PRIOR_WEIGHT * PRIOR_STRENGTH + 10) / 15)


def test_weight_takes_the_higher_ledger() -> None:
    """两个判定是「或」关系：新话题 0.9 / 感兴趣 0.1 不能平均成 0.5。"""
    record = Attention()
    for _ in range(10):
        record.observe(0.1, 0.9)

    assert record.weight == pytest.approx((PRIOR_WEIGHT * PRIOR_STRENGTH + 9.0) / 15)


def test_weight_never_drops_below_floor() -> None:
    """防饿死：一直被判定为不感兴趣，也只是逼近下限，不会归零。"""
    record = Attention()
    for _ in range(500):
        record.observe(0.0, 0.0)

    assert record.weight == MIN_WEIGHT


def test_partial_weight_only_adds_its_share() -> None:
    """重复判定的同一段内容只按新增占比入账：均值不变，置信度打折（ADR-0035）。"""
    record = Attention()
    record.observe(0.9, 0.1, weight=0.25)

    assert record.observations == pytest.approx(0.25)
    assert record.weight == pytest.approx(
        (PRIOR_WEIGHT * PRIOR_STRENGTH + 0.9 * 0.25) / (PRIOR_STRENGTH + 0.25)
    )


def test_repeated_observations_at_full_weight_match_one_observation() -> None:
    """20 次各 0.05 份，与一次满额观测等价——折算的是证据份量，不是分数。"""
    split = Attention()
    for _ in range(20):
        split.observe(0.8, 0.6, weight=0.05)
    whole = Attention()
    whole.observe(0.8, 0.6)

    assert split.observations == pytest.approx(whole.observations)
    assert split.weight == pytest.approx(whole.weight)


def test_weighted_sample_returns_requested_count_without_duplicates() -> None:
    items = [f"u{index}" for index in range(6)]
    picked = weighted_sample(items, lambda umo: 0.5, 3, random.Random(7))

    assert len(picked) == 3
    assert len(set(picked)) == 3
    assert set(picked) <= set(items)


def test_weighted_sample_selects_everything_in_weight_order() -> None:
    """N 不小于目标数时仍走加权路径：全选，且顺序只由关注度决定，与抽样随机无关。"""
    weights = {"low": 0.2, "mid": 0.5, "high": 0.9}
    for seed in range(20):
        picked = weighted_sample(list(weights), weights.get, 3, random.Random(seed))
        assert picked == ["high", "mid", "low"]


def test_equal_weights_fall_back_to_binding_list_order() -> None:
    """关注度全等时，入选子集是随机的，但执行顺序退回列表顺序（可预期）。"""
    items = ["first", "second", "third"]
    for seed in range(20):
        picked = weighted_sample(items, lambda _: PRIOR_WEIGHT, 3, random.Random(seed))
        assert picked == items


def test_weighted_sample_still_picks_the_floor_session() -> None:
    """下限保证弱会话保有非零入选概率，只是频率更低。"""
    rng = random.Random(2026)
    weights = {"hot": 0.9, "cold": MIN_WEIGHT}
    picks = [weighted_sample(list(weights), weights.get, 1, rng)[0] for _ in range(500)]

    assert picks.count("cold") > 0
    assert picks.count("hot") > picks.count("cold")


def test_store_round_trips_through_disk(tmp_path: Path) -> None:
    store = AttentionStore(tmp_path)
    store.observe("aiocqhttp:GroupMessage:1", interested=0.8, new_topic=0.2)
    store.observe("aiocqhttp:GroupMessage:1", interested=0.6, new_topic=0.1)
    store.flush()

    reloaded = AttentionStore(tmp_path)
    reloaded.load()

    assert reloaded.weight("aiocqhttp:GroupMessage:1") == pytest.approx(
        store.weight("aiocqhttp:GroupMessage:1")
    )
    assert reloaded.detail("aiocqhttp:GroupMessage:1")["observations"] == 2


def test_zero_weight_observation_does_not_create_record(tmp_path: Path) -> None:
    """份量为 0 表示这段内容已经记过账：既不累加，也不留空条目。"""
    store = AttentionStore(tmp_path)
    umo = "aiocqhttp:GroupMessage:1"

    store.observe(umo, interested=1.0, new_topic=1.0, weight=0.0)

    assert store.detail(umo) == {"weight": PRIOR_WEIGHT, "observations": 0}
    assert store.snapshot() == {}


def test_repeated_decisions_on_unchanged_content_leave_attention_alone(
    tmp_path: Path,
) -> None:
    """静默会话每 5 分钟被再判一次：内容没变，关注度必须一点都不动。"""
    store = AttentionStore(tmp_path)
    umo = "aiocqhttp:GroupMessage:1"
    store.observe(umo, interested=0.9, new_topic=0.9)
    before = store.detail(umo)

    for _ in range(50):
        store.observe(umo, interested=0.9, new_topic=0.9, weight=0.0)

    assert store.detail(umo) == before


def test_fractional_observations_survive_disk_round_trip(tmp_path: Path) -> None:
    """观测次数是折算后的浮点量，落盘再读出不能退化成整数。"""
    store = AttentionStore(tmp_path)
    umo = "aiocqhttp:GroupMessage:1"
    store.observe(umo, interested=0.8, new_topic=0.4, weight=0.25)
    store.flush()

    reloaded = AttentionStore(tmp_path)
    reloaded.load()

    assert reloaded.detail(umo)["observations"] == pytest.approx(0.25)
    assert reloaded.weight(umo) == pytest.approx(store.weight(umo))


def test_missing_file_is_an_empty_archive(tmp_path: Path) -> None:
    store = AttentionStore(tmp_path)
    store.load()

    assert store.weight("aiocqhttp:GroupMessage:1") == PRIOR_WEIGHT


@pytest.mark.parametrize("content", [b"{ not json", b"\xff\xfe"])
def test_corrupted_file_falls_back_to_empty_archive(
    tmp_path: Path, content: bytes
) -> None:
    """档案损坏不能阻断绑定启动：读不到就当没观测过。"""
    (tmp_path / "attention.json").write_bytes(content)
    store = AttentionStore(tmp_path)
    store.load()

    assert store.weight("aiocqhttp:GroupMessage:1") == PRIOR_WEIGHT


def test_overflowing_record_does_not_discard_valid_records(tmp_path: Path) -> None:
    """坏条目的数字溢出只跳过该条，其余会话仍可正常载入。"""
    valid, invalid = "aiocqhttp:GroupMessage:1", "aiocqhttp:GroupMessage:2"
    (tmp_path / "attention.json").write_text(
        json.dumps(
            {
                "scores": {
                    valid: {"interested_sum": 0.9, "observations": 1},
                    invalid: {"observations": float("inf")},
                }
            }
        ),
        encoding="utf-8",
    )
    store = AttentionStore(tmp_path)

    store.load()

    assert store.detail(valid)["observations"] == 1
    assert store.detail(invalid) == {"weight": PRIOR_WEIGHT, "observations": 0}


def test_reset_returns_to_prior_and_prune_drops_orphans(tmp_path: Path) -> None:
    store = AttentionStore(tmp_path)
    store.observe("aiocqhttp:GroupMessage:1", interested=1.0, new_topic=1.0)
    store.observe("aiocqhttp:GroupMessage:2", interested=1.0, new_topic=1.0)

    store.reset(["aiocqhttp:GroupMessage:1"])
    assert store.detail("aiocqhttp:GroupMessage:1")["observations"] == 0

    store.prune({"aiocqhttp:GroupMessage:1"})
    assert store.detail("aiocqhttp:GroupMessage:2")["observations"] == 0
    assert store.snapshot() == {}


def test_flush_writes_expected_shape(tmp_path: Path) -> None:
    store = AttentionStore(tmp_path)
    store.observe("aiocqhttp:GroupMessage:1", interested=0.5, new_topic=0.5)
    store.flush()

    payload = json.loads((tmp_path / "attention.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["scores"]["aiocqhttp:GroupMessage:1"]["observations"] == 1


def test_scheduler_weighted_selection_uses_attention(tmp_path: Path) -> None:
    """调度器把 weighted 接到关注度上：高关注度先执行，全选边界也不退回列表顺序。"""
    store = AttentionStore(tmp_path)
    hot, cold = "aiocqhttp:GroupMessage:1", "aiocqhttp:GroupMessage:2"
    for _ in range(10):
        store.observe(hot, interested=0.95, new_topic=0.9)
        store.observe(cold, interested=0.0, new_topic=0.0)
    scheduler = AmbientScheduler(SimpleNamespace(attention=store))
    binding = Binding(
        id="b0",
        platform_id="aiocqhttp",
        rule_id="r0",
        targets=[cold, hot],
        selection=Selection(mode="weighted", count=2),
    )

    assert scheduler._select_targets(binding) == [hot, cold]

    binding.selection = Selection(mode="all", count=2)
    assert scheduler._select_targets(binding) == [cold, hot]


@pytest.mark.asyncio
async def test_close_flushes_and_reload_reopens(tmp_path: Path) -> None:
    """卸载时不等延迟写；热重载复用同一个实例时延迟写必须重新可用。"""
    store = AttentionStore(tmp_path, save_delay=3600)
    store.observe("aiocqhttp:GroupMessage:1", interested=1.0, new_topic=1.0)
    await store.close()
    assert (tmp_path / "attention.json").exists()

    store.load()
    assert store._closed is False
    assert store.detail("aiocqhttp:GroupMessage:1")["observations"] == 1


def test_selection_parses_weighted_mode() -> None:
    selection = Selection.parse({"mode": "weighted", "count": 3})
    assert selection.mode == "weighted"
    assert (
        Binding.parse(
            {
                "platform_id": "aiocqhttp",
                "rule_id": "r_1",
                "targets": ["aiocqhttp:GroupMessage:1"],
                "selection": {"mode": "weighted", "count": 2},
            }
        ).selection.mode
        == "weighted"
    )


@pytest.mark.asyncio
async def test_observations_after_close_are_ignored(tmp_path: Path) -> None:
    """卸载后不再接受观测，避免「改了内存但永不落盘」的静默丢失。"""
    store = AttentionStore(tmp_path, save_delay=3600)
    store.observe("aiocqhttp:GroupMessage:1", interested=1.0, new_topic=1.0)
    await store.close()

    store.observe("aiocqhttp:GroupMessage:1", interested=1.0, new_topic=1.0)

    assert store.detail("aiocqhttp:GroupMessage:1")["observations"] == 1


def test_load_flushes_pending_observations_before_reloading(tmp_path: Path) -> None:
    """重载前先落盘：initialize 不一定跟在 terminate 后面，待写的判定不能凭空消失。"""
    store = AttentionStore(tmp_path, save_delay=3600)
    store.observe("aiocqhttp:GroupMessage:1", interested=0.9, new_topic=0.9)

    store.load()

    assert store.detail("aiocqhttp:GroupMessage:1")["observations"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("change", ["observe", "reset"])
async def test_failed_reload_write_preserves_changes_and_retries(
    tmp_path: Path, monkeypatch, closed: bool, change: str
) -> None:
    """重载写失败时保留观测或重置结果，恢复后无需新观测也能延迟落盘。"""
    store = AttentionStore(tmp_path, save_delay=0)
    umo = "aiocqhttp:GroupMessage:1"
    store.observe(umo, interested=0.9, new_topic=0.2)
    store.flush()
    if change == "observe":
        store.observe(umo, interested=0.8, new_topic=0.1)
    else:
        store.reset([umo])
    expected = store.detail(umo)
    saved = asyncio.Event()
    replace = os.replace
    retry_failed = False

    def fail_replace(*args):
        raise OSError("模拟文件被占用")

    def record_replace(*args):
        nonlocal retry_failed
        # 第一次延迟重试仍失败，之后也必须继续重试，无需新观测唤醒。
        if not retry_failed:
            retry_failed = True
            raise OSError("模拟文件仍被占用")
        replace(*args)
        saved.set()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                "astrbot_plugin_ambient_chat.runtime.attention.os.replace", fail_replace
            )
            if closed:
                await store.close()
            store.load()
            assert store.detail(umo) == expected

        monkeypatch.setattr(
            "astrbot_plugin_ambient_chat.runtime.attention.os.replace", record_replace
        )
        await asyncio.wait_for(saved.wait(), timeout=1)
        reloaded = AttentionStore(tmp_path)
        reloaded.load()
        assert reloaded.detail(umo) == expected
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_apply_store_changes_skips_prune_when_store_failed(
    tmp_path: Path,
) -> None:
    """settings.json 读盘失败时 store 是空的：不能把关注度当孤儿删掉（ADR-0034）。"""
    store = AttentionStore(tmp_path)
    umo = "aiocqhttp:GroupMessage:1"
    store.observe(umo, interested=0.9, new_topic=0.9)
    plugin = SimpleNamespace(
        store=SimpleNamespace(bindings={}, load_failed=True),
        attention=store,
        lifecycle=SimpleNamespace(reconcile=AsyncMock()),
    )

    await AmbientChatPlugin.apply_store_changes(plugin)

    assert store.detail(umo)["observations"] == 1


@pytest.mark.asyncio
async def test_apply_store_changes_prunes_orphans(tmp_path: Path) -> None:
    """正常读盘时，目标从所有绑定里消失才删除对应条目。"""
    store = AttentionStore(tmp_path)
    kept, orphan = "aiocqhttp:GroupMessage:1", "aiocqhttp:GroupMessage:2"
    for umo in (kept, orphan):
        store.observe(umo, interested=0.9, new_topic=0.9)
    plugin = SimpleNamespace(
        store=SimpleNamespace(
            bindings={"b0": SimpleNamespace(targets=[kept])}, load_failed=False
        ),
        attention=store,
        lifecycle=SimpleNamespace(reconcile=AsyncMock()),
    )

    await AmbientChatPlugin.apply_store_changes(plugin)

    assert set(store.snapshot()) == {kept}


@pytest.mark.asyncio
async def test_reset_binding_attention_endpoint_returns_to_prior(
    tmp_path: Path, monkeypatch
) -> None:
    """重置是运行期操作：不走绑定启停的只读校验，重置后页面读到的是先验值。"""
    store = AttentionStore(tmp_path)
    hot = "aiocqhttp:GroupMessage:1"
    store.observe(hot, interested=0.95, new_topic=0.9)
    plugin = SimpleNamespace(
        store=SimpleNamespace(bindings={"b0": SimpleNamespace(id="b0", targets=[hot])}),
        attention=store,
    )
    monkeypatch.setattr(
        webapi,
        "request",
        SimpleNamespace(json=AsyncMock(return_value={"binding_id": "b0"})),
    )

    response = await AmbientWebApi(plugin).reset_binding_attention()

    payload = json.loads(response.body)
    assert payload["reset"] is True
    assert payload["attention"][hot]["observations"] == 0
    assert store.detail(hot)["weight"] == PRIOR_WEIGHT
