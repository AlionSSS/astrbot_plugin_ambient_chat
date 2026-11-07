"""关注度：由决策模型历次判定累积出的会话级长期权重（见 ADR-0034）。

关注度按目标会话（UMO）记账，与规则、绑定无关：换规则、停止绑定、插件重启都不清零，
目标从所有绑定里消失时才删除。先验负责冷启动、下限负责防饿死，两者保证它永不为零，
因此任何目标会话在任何时刻都有非零的入选概率。
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.star import StarTools

from ..common import PLUGIN_NAME

ATTENTION_FILENAME = "attention.json"
ATTENTION_SCHEMA_VERSION = 1

PRIOR_WEIGHT = 0.5
"""新会话的先验关注度：中性值，既不偏爱也不打压。"""

PRIOR_STRENGTH = 5.0
"""先验的等价观测次数：观测 5 次后先验只占一半，10 次后占三分之一。"""

MIN_WEIGHT = 0.1
"""关注度下限：防止一旦落到低位就再也抽不到，观测永久冻结。"""

SAVE_DELAY_SECONDS = 5.0
"""标脏后合并写盘的延迟：每次判定都落盘会变成高频小写入。"""

_EPSILON = 1e-12
"""采样时把 0 概率抬离奇点：``-ln(0)`` 是无穷大。"""


@dataclass(slots=True)
class Attention:
    """一个目标会话的账本：两笔原始概率之和 + 共同观测次数。"""

    interested_sum: float = 0.0
    new_topic_sum: float = 0.0
    observations: int = 0

    @property
    def weight(self) -> float:
        """带回退先验的均值，取两笔账里更高的那笔。

        两个判定问题是「或」关系（判定顺序：先感兴趣，否则新话题），取平均会把
        「新话题 0.9 / 感兴趣 0.1」错算成 0.5，所以取较大值而不是均值。
        """
        if self.observations <= 0:
            return PRIOR_WEIGHT
        numerator = PRIOR_WEIGHT * PRIOR_STRENGTH + max(
            self.interested_sum, self.new_topic_sum
        )
        value = numerator / (PRIOR_STRENGTH + self.observations)
        return max(MIN_WEIGHT, min(1.0, value))

    def observe(self, interested: float, new_topic: float) -> None:
        self.interested_sum += _clamp(interested)
        self.new_topic_sum += _clamp(new_topic)
        self.observations += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "interested_sum": round(self.interested_sum, 4),
            "new_topic_sum": round(self.new_topic_sum, 4),
            "observations": self.observations,
        }

    @classmethod
    def parse(cls, raw: Any) -> Attention | None:
        """从磁盘还原；字段不合法时返回 ``None``，由调用方跳过这一条。"""
        if not isinstance(raw, dict):
            return None
        try:
            interested = float(raw.get("interested_sum", 0.0) or 0.0)
            new_topic = float(raw.get("new_topic_sum", 0.0) or 0.0)
            observations = int(raw.get("observations", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(map(math.isfinite, (interested, new_topic))):
            return None
        if interested < 0 or new_topic < 0 or observations < 0:
            return None
        return cls(interested, new_topic, observations)


class AttentionStore:
    """关注度档案：内存读写 + 合并写盘。

    读失败或文件损坏按空档案从头累积，绝不阻断绑定启动；写失败只记日志，
    不影响本次发言，脏标记保留给下一次写盘重试。
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        save_delay: float = SAVE_DELAY_SECONDS,
    ) -> None:
        self._data_dir = data_dir
        self._save_delay = save_delay
        self._records: dict[str, Attention] = {}
        self._dirty = False
        self._timer: asyncio.TimerHandle | None = None
        self._closed = False

    # ------------------------------------------------------------------ 路径
    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:
            self._data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        return self._data_dir

    @property
    def path(self) -> Path:
        return self.data_dir / ATTENTION_FILENAME

    # ------------------------------------------------------------------ 读写
    def load(self) -> None:
        # 重载前先把待写的观测落盘：initialize 未必总跟在 terminate 后面，
        # 直接重置内存记录会让这些判定凭空消失。
        saved = self.flush()
        # 热重载可能复用同一个实例：卸载时关掉的延迟写要在这里重新打开。
        self._closed = False
        if not saved:
            # 写失败时内存仍是最新账本，不能用旧档案覆盖；恢复延迟写继续重试。
            self._schedule()
            return
        self._records = {}
        self._dirty = False
        self._cancel_timer()
        path = self.path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("[Ambient] 读取关注度档案失败，按空档案继续：%s", exc)
            return
        if not isinstance(payload, dict):
            logger.warning("[Ambient] 关注度档案根节点不是对象，按空档案继续")
            return
        raw = payload.get("scores")
        if not isinstance(raw, dict):
            return
        for umo, item in raw.items():
            record = Attention.parse(item)
            if record is None or not str(umo).strip():
                continue
            self._records[str(umo)] = record

    def weight(self, umo: str) -> float:
        record = self._records.get(umo)
        return record.weight if record is not None else PRIOR_WEIGHT

    def detail(self, umo: str) -> dict[str, Any]:
        """页面展示用：关注度（两位小数）与观测次数。"""
        record = self._records.get(umo)
        if record is None:
            return {"weight": PRIOR_WEIGHT, "observations": 0}
        return {"weight": round(record.weight, 2), "observations": record.observations}

    def observe(self, umo: str, *, interested: float, new_topic: float) -> None:
        """记一次判定：两笔账同时累加（一次请求同时给出两个概率）。"""
        if self._closed:
            return
        if not math.isfinite(interested) or not math.isfinite(new_topic):
            return
        record = self._records.get(umo)
        if record is None:
            record = Attention()
            self._records[umo] = record
        record.observe(interested, new_topic)
        self._mark_dirty()

    def reset(self, umos: Iterable[str]) -> None:
        """清空这些会话的关注度，回到先验值（页面上的「重置关注度」）。"""
        if self._closed:
            return
        removed = [umo for umo in umos if self._records.pop(umo, None) is not None]
        if removed:
            self._mark_dirty()

    def prune(self, known: set[str]) -> None:
        """删除已不属于任何绑定目标的条目；其余画像保留。"""
        if self._closed:
            return
        stale = [umo for umo in self._records if umo not in known]
        for umo in stale:
            self._records.pop(umo, None)
        if stale:
            self._mark_dirty()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {umo: self.detail(umo) for umo in self._records}

    # ------------------------------------------------------------------ 写盘
    def flush(self) -> bool:
        """把脏数据同步写盘；返回是否已保存，失败时保留脏标记并延迟重试。"""
        self._cancel_timer()
        if not self._dirty:
            return True
        payload = {
            "schema_version": ATTENTION_SCHEMA_VERSION,
            "scores": {umo: record.to_dict() for umo, record in self._records.items()},
        }
        path = self.path
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.warning("[Ambient] 关注度档案写入失败，稍后重试：%s", exc)
            self._schedule()
            return False
        finally:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
        self._dirty = False
        return True

    async def close(self) -> None:
        """插件卸载：停掉延迟写并立刻落盘（之后不再记账）。"""
        self._closed = True
        self.flush()

    # ------------------------------------------------------------------ 内部
    def _mark_dirty(self) -> None:
        self._dirty = True
        self._schedule()

    def _schedule(self) -> None:
        if self._closed or self._timer is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有事件循环（同步调用方）：留给下一次 flush。
            return
        self._timer = loop.call_later(self._save_delay, self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        self.flush()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def weighted_sample(
    items: Sequence[str],
    weight_of: Callable[[str], float],
    count: int,
    rng: random.Random | None = None,
) -> list[str]:
    """不放回加权抽样，结果按权重从高到低。

    用 Efraimidis–Spirakis：每个候选取 ``-ln(u)/w``（指数分布的分位数），取最小的
    ``count`` 个。它等价于逐轮按剩余权重抽取，但只需要一次排序，也不必反复累加。
    权重相同时按原有顺序稳定排列，因此「关注度全等」时退回绑定里的列表顺序。
    """
    if count <= 0 or not items:
        return []
    source = rng or random
    weights = {umo: max(float(weight_of(umo)), MIN_WEIGHT) for umo in items}
    positions = {umo: order for order, umo in enumerate(items)}
    keyed: list[tuple[float, int, str]] = []
    for order, umo in enumerate(items):
        key = -math.log(max(source.random(), _EPSILON)) / weights[umo]
        keyed.append((key, order, umo))
    keyed.sort(key=lambda item: (item[0], item[1]))
    chosen = [umo for _, _, umo in keyed[:count]]
    # 抽样 key 是随机的，不能直接当执行顺序：入选之后按关注度降序排，
    # 关注度相同时退回绑定里的列表位置，保证「关注度全等」时顺序可预期。
    return sorted(chosen, key=lambda umo: (-weights[umo], positions[umo]))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
