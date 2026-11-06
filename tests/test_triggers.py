"""入口触发：窗口最小间隔 / 星期 / 固定间隔 / 一次性过期的解析与采样。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from astrbot_plugin_ambient_chat.rules.models import Binding, Rule, RuleError, Trigger
from astrbot_plugin_ambient_chat.rules.store import RuleStore
from astrbot_plugin_ambient_chat.runtime import scheduler as scheduler_module
from astrbot_plugin_ambient_chat.runtime.scheduler import AmbientScheduler

TZ = ZoneInfo("Asia/Shanghai")
BINDING = Binding(
    id="b_test",
    platform_id="aiocqhttp",
    rule_id="r_test",
    targets=["aiocqhttp:GroupMessage:10001"],
)


def _rule(trigger: dict) -> dict:
    return {"name": "测试规则", "triggers": [trigger]}


def _window(**overrides) -> dict:
    payload = {"type": "window", "start": "14:00", "end": "18:00", "count": 3}
    payload.update(overrides)
    return payload


def test_window_accepts_gap_and_normalizes_days() -> None:
    rule = Rule.parse(
        _rule(_window(gap_minutes=30, days=[5, 1, 5])),
        TZ,
    )
    trigger = rule.triggers[0]
    assert trigger.gap_minutes == 30
    assert trigger.days == [1, 5]


def test_window_all_days_is_stored_as_daily() -> None:
    rule = Rule.parse(_rule(_window(days=[1, 2, 3, 4, 5, 6, 7])), TZ)
    assert rule.triggers[0].days == []
    assert rule.triggers[0].to_dict()["days"] == []


def test_window_rejects_days_out_of_range() -> None:
    with pytest.raises(RuleError, match="1~7"):
        Rule.parse(_rule(_window(days=[0])), TZ)


def test_window_rejects_gap_that_does_not_fit() -> None:
    with pytest.raises(RuleError, match="放不下"):
        Rule.parse(
            _rule({"type": "window", "start": "14:00", "end": "14:05", "count": 20}),
            TZ,
        )


def test_legacy_window_degrades_instead_of_dropping_the_rule() -> None:
    """读盘时塞不下最小间隔的历史窗口要降级，而不是让整条规则被跳过。"""
    rule = Rule.parse(
        _rule({"type": "window", "start": "14:00", "end": "14:05", "count": 20}),
        TZ,
        strict=False,
    )
    assert rule.triggers[0].count == 20


def test_sampling_is_deterministic_and_keeps_the_gap() -> None:
    trigger = Trigger.parse(_window(count=4, gap_minutes=30), TZ)
    day = date(2026, 11, 5)

    first = AmbientScheduler(None)._window_samples(BINDING, 0, trigger, day, TZ)
    second = AmbientScheduler(None)._window_samples(BINDING, 0, trigger, day, TZ)

    assert first == second
    assert len(first) == 4
    gaps = [
        (later - earlier).total_seconds() for earlier, later in zip(first, first[1:])
    ]
    assert min(gaps) >= 30 * 60
    assert datetime(2026, 11, 5, 14, 0, tzinfo=TZ) <= first[0]
    assert first[-1] <= datetime(2026, 11, 5, 18, 0, tzinfo=TZ)


def test_different_bindings_get_different_samples() -> None:
    trigger = Trigger.parse(_window(count=4), TZ)
    day = date(2026, 11, 5)
    scheduler = AmbientScheduler(None)
    other = Binding(
        id="b_other",
        platform_id="aiocqhttp",
        rule_id="r_test",
        targets=["aiocqhttp:GroupMessage:10002"],
    )

    assert scheduler._window_samples(BINDING, 0, trigger, day, TZ) != (
        scheduler._window_samples(other, 0, trigger, day, TZ)
    )


def test_window_skips_weekdays_not_selected() -> None:
    # 2026-11-05 是周四，只选周五时当天不该有采样点。
    trigger = Trigger.parse(_window(days=[5]), TZ)
    scheduler = AmbientScheduler(None)

    assert scheduler._window_samples(BINDING, 0, trigger, date(2026, 11, 5), TZ) == []

    now = datetime(2026, 11, 5, 12, 0, tzinfo=TZ)
    fire_at = scheduler._window_next_fire(BINDING, 0, trigger, now, TZ)
    assert fire_at is not None and fire_at.date() == date(2026, 11, 6)


def test_interval_is_periodic_from_the_anchor() -> None:
    trigger = Trigger.parse(
        {"type": "interval", "every_minutes": 60, "jitter_percent": 0},
        TZ,
    )
    scheduler = AmbientScheduler(None)
    now = datetime(2026, 11, 5, 10, 17, tzinfo=TZ)

    first = scheduler._interval_next_fire(BINDING, 0, trigger, now)
    assert first == now + timedelta(hours=1)
    second = scheduler._interval_next_fire(BINDING, 0, trigger, first)
    assert second - first == timedelta(hours=1)


def test_interval_jitter_is_stable_and_bounded() -> None:
    trigger = Trigger.parse(
        {"type": "interval", "every_minutes": 60, "jitter_percent": 50},
        TZ,
    )
    scheduler = AmbientScheduler(None)
    now = datetime(2026, 11, 5, 10, 0, tzinfo=TZ)
    anchor = scheduler._interval_next_fire(BINDING, 0, trigger, now)

    assert scheduler._interval_next_fire(BINDING, 0, trigger, now) == anchor
    assert abs((anchor - now) - timedelta(hours=1)) <= timedelta(minutes=31)


def test_interval_rejects_out_of_range_values() -> None:
    with pytest.raises(RuleError, match="周期"):
        Trigger.parse({"type": "interval", "every_minutes": 0}, TZ)
    with pytest.raises(RuleError, match="抖动"):
        Trigger.parse({"type": "interval", "jitter_percent": 90}, TZ)


def test_other_trigger_does_not_skip_delayed_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "_interval_jitter",
        lambda seed, occurrence, span: 18 * 60 if occurrence == 1 else 0,
    )
    anchor = datetime(2026, 11, 6, 10, tzinfo=TZ)
    interval = Trigger.parse(
        {"type": "interval", "every_minutes": 60, "jitter_percent": 50}, TZ
    )
    once = Trigger.parse(
        {"type": "once", "at": (anchor + timedelta(minutes=61)).isoformat()}, TZ
    )
    rule = Rule("r_test", "测试规则", triggers=[interval, once])
    scheduler = AmbientScheduler(None)
    first_at, first_trigger = scheduler._next_fire(BINDING, rule, anchor, TZ)
    assert first_trigger is once
    scheduler._consume_once(BINDING, rule, once)
    next_at, next_trigger = scheduler._next_fire(BINDING, rule, first_at, TZ)
    assert next_trigger is interval
    assert next_at == anchor + timedelta(minutes=78)
    # 恰好到期的触发不应再次返回，重复重算则保持同一未来时刻。
    assert scheduler._interval_next_fire(
        BINDING, 0, interval, next_at
    ) == anchor + timedelta(hours=2)
    assert scheduler._interval_next_fire(BINDING, 0, interval, first_at) == next_at


@pytest.mark.parametrize("now_minutes", [0, 61, 85, 96, 97, 150, 156, 157, 3000])
def test_interval_selects_earliest_actual_time_when_cycles_cross(
    monkeypatch, now_minutes
) -> None:
    # 60% 抖动下，第 2 次 (11:24) 比第 1 次 (11:36) 更早；
    # 第 1 次过后，第 4 次 (13:24) 又比第 3 次 (13:36) 更早。
    def jitter(seed, occurrence, span):
        return span if occurrence % 2 else -span

    monkeypatch.setattr(scheduler_module, "_interval_jitter", jitter)
    anchor = datetime(2026, 11, 6, 10, tzinfo=TZ)
    trigger = Trigger.parse(
        {"type": "interval", "every_minutes": 60, "jitter_percent": 60}, TZ
    )
    scheduler = AmbientScheduler(None)
    scheduler._interval_anchor[BINDING.id, 0] = anchor
    now = anchor + timedelta(minutes=now_minutes)
    # 用足够长的完整时间表验证时间排序、跳过过期与不重复触发。
    timeline = [
        anchor + timedelta(minutes=60 * occurrence + (36 if occurrence % 2 else -36))
        for occurrence in range(1, 60)
    ]
    expected = min(at for at in timeline if at > now)
    assert scheduler._interval_next_fire(BINDING, 0, trigger, now) == expected


def test_past_once_is_kept_and_marked_expired() -> None:
    rule = Rule.parse(
        _rule({"type": "once", "at": "2020-01-01T10:00:00+08:00"}),
        TZ,
    )
    trigger = rule.triggers[0]
    assert trigger.to_dict()["at"].startswith("2020-01-01")
    assert trigger.is_expired()

    future = Trigger.parse({"type": "once", "at": "2999-01-01T10:00:00+08:00"}, TZ)
    assert not future.is_expired()


def test_merge_window_swallows_repeat_triggers() -> None:
    scheduler = AmbientScheduler(None)
    now = datetime(2026, 11, 5, 12, 0, tzinfo=TZ)

    assert scheduler._should_merge(BINDING, now) is False
    scheduler._binding_fired_at[BINDING.id] = now
    assert scheduler._should_merge(BINDING, now + timedelta(seconds=30)) is True
    assert scheduler._should_merge(BINDING, now + timedelta(seconds=90)) is False


class _PluginStub:
    """手动触发只校验前置条件，构造到「拦在真正执行之前」即可。"""

    def __init__(self) -> None:
        self.store = RuleStore()
        self.store.rules["r_test"] = Rule.parse(_rule(_window()), TZ)
        self.store.bindings[BINDING.id] = BINDING
        self.timezone = TZ

    def is_enabled(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_trigger_target_guards_binding_and_session() -> None:
    plugin = _PluginStub()
    stopped = Binding(
        id="b_stopped",
        platform_id="aiocqhttp",
        rule_id="r_test",
        targets=["aiocqhttp:GroupMessage:10002"],
        enabled=False,
    )
    plugin.store.bindings[stopped.id] = stopped
    scheduler = AmbientScheduler(plugin)

    with pytest.raises(RuleError, match="不存在"):
        await scheduler.trigger_target("b_missing", BINDING.targets[0])
    with pytest.raises(RuleError, match="启动"):
        await scheduler.trigger_target(stopped.id, stopped.targets[0])
    with pytest.raises(RuleError, match="目标"):
        await scheduler.trigger_target(BINDING.id, "aiocqhttp:GroupMessage:99999")
