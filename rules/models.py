"""Ambient 规则、绑定与更多设置的数据模型及单份配置校验。"""

from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from ..common import (
    BLOCKER_COOLDOWN,
    BLOCKER_SILENCE,
    BLOCKER_TYPES,
    BLOCKER_UNANSWERED,
    INTENT_DECISION,
    INTENT_DIRECT,
    TRIGGER_CRON,
    TRIGGER_INTERVAL,
    TRIGGER_ONCE,
    TRIGGER_TYPES,
    TRIGGER_WEEKDAYS,
    TRIGGER_WINDOW,
    parse_hhmm,
    parse_target,
)
from ..conversation.media_store import (
    DEFAULT_IMAGE_MAX_EDGE,
    DEFAULT_IMAGE_QUALITY,
    MAX_IMAGE_MAX_EDGE,
    MAX_IMAGE_QUALITY,
    MIN_IMAGE_MAX_EDGE,
    MIN_IMAGE_QUALITY,
)
from ..speech.segments import (
    DEFAULT_SEGMENT_ENABLED,
    DEFAULT_SEGMENT_INTERVAL_MAX,
    DEFAULT_SEGMENT_INTERVAL_MIN,
    DEFAULT_SEGMENT_PATTERN,
    DEFAULT_SEGMENT_THRESHOLD,
    MAX_SEGMENT_INTERVAL,
    MAX_SEGMENT_THRESHOLD,
    MIN_SEGMENT_INTERVAL,
    MIN_SEGMENT_THRESHOLD,
    validate_pattern,
)

DEFAULT_CONTEXT_TURNS = 20
DEFAULT_THRESHOLD = 0.7
DEFAULT_PASS_PROBABILITY = 1.0
"""「按概率放行」的默认概率：默认必定发言（同旧版「直接放行」），调低才按概率放弃。"""
DEFAULT_COOLDOWN_MINUTES = 5
DEFAULT_UNREAD_MIN_COUNT = 10
DEFAULT_UNANSWERED_COUNT = 3
"""机器人连续发言多少条后开始判断「没人理」。"""
DEFAULT_UNANSWERED_MINUTES = 180
"""连发无人回应后，自最后一次发言起等待多少分钟。"""
MAX_UNANSWERED_COUNT = 100
MAX_UNANSWERED_MINUTES = sys.maxsize
"""分钟上限取 int 最大值，设成极大值即等同永久等待。"""
MAX_CONTEXT_TURNS = 200
MAX_WINDOW_COUNT = 20

DEFAULT_WINDOW_GAP_MINUTES = 1
"""时段随机里任意两次触发的最小间隔（分钟）。"""
MAX_WINDOW_GAP_MINUTES = 720

DEFAULT_INTERVAL_MINUTES = 30
"""固定间隔触发的默认周期（分钟）；与「新建规则」弹窗的默认值保持一致。"""
MAX_INTERVAL_MINUTES = 24 * 60
DEFAULT_INTERVAL_JITTER = 15
"""固定间隔触发的默认抖动百分比（±周期 * 比例），让节奏不像闹钟。"""
MAX_INTERVAL_JITTER = 60

SELECTION_ALL = "all"
"""每次触发都覆盖绑定的全部目标会话。"""

SELECTION_RANDOM = "random"
"""每次触发等概率随机挑选 N 个目标会话，按挑选顺序依次触发。"""

SELECTION_WEIGHTED = "weighted"
"""按关注度加权随机挑选 N 个目标会话，按关注度从高到低依次触发（ADR-0034）。"""

SELECTION_MODES = (SELECTION_ALL, SELECTION_RANDOM, SELECTION_WEIGHTED)

MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 5
MIN_SEND_INTERVAL_SECONDS = 0.0
MAX_SEND_INTERVAL_SECONDS = 30.0
MIN_BUFFER_SIZE = 10
MAX_BUFFER_SIZE = 1000

DEFAULT_CONCURRENCY = 1
DEFAULT_SEND_INTERVAL_SECONDS = 2.0
DEFAULT_BUFFER_SIZE = 100
DEFAULT_CACHE_IMAGES = True
"""默认缓存图片：开启后图片按「图片最长边 / 压缩质量」转存，只有多模态模型能用上。"""

DEFAULT_CACHE_AUDIO = False
"""默认不缓存语音：音频体积不小，且只有支持音频输入的模型能用上（ADR-0031）。"""

DEFAULT_CACHE_IMAGES_ADVICE_SEEN = False
"""是否已提示过「建议开启图片缓存」：只在 WebUI 首次进入设置页时提示一次。

每次在设置页勾选开启缓存都会另行弹窗确认，不读取该标记。
"""

DEFAULT_ASTRBOT_CONFIG_ADVICE_SEEN = False
"""是否已提示过「建议调整 AstrBot 侧配置」：只在 WebUI 首次打开控制台时提示一次。

提示内容涉及核心的群消息持久化与主动回复，用户改没改本插件无从判断，
因此只按「提示过没有」记录，不跟随核心配置状态反复弹。
"""

MAX_BUFFER_IGNORED_SENDERS = 200
"""上下文缓冲最多忽略多少个发送者 ID。"""
MAX_SENDER_ID_LENGTH = 128
"""单个发送者 ID 的最大字符数：QQ 号、OpenID 都远小于这个长度。"""

MAX_SELECTION_COUNT = 50

CUSTOM_DECISION_PREFIX = "custom:"
"""规则里引用「更多设置」中自定义决策模型的前缀，后接模型 ID。"""

MAX_DECISION_MODELS = 10
MAX_DECISION_NAME_LENGTH = 40
MAX_DECISION_URL_LENGTH = 500
MAX_DECISION_MODEL_LENGTH = 100


class RuleError(ValueError):
    """规则或绑定数据不合法。"""


# ---------------------------------------------------------------------- 全局设置
@dataclass(slots=True)
class DecisionModel:
    """「更多设置」里自定义的 SystemOne 决策模型。

    规则用 ``custom:<id>`` 引用它（见 :data:`CUSTOM_DECISION_PREFIX`）；
    官方 AstrBot 判断提供商仍按 provider id 引用。
    """

    id: str
    name: str
    base_url: str
    model: str
    api_key: str = ""

    @property
    def ref(self) -> str:
        """规则里保存的引用值。"""
        return f"{CUSTOM_DECISION_PREFIX}{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
        }

    @classmethod
    def parse(cls, raw: Any) -> DecisionModel:
        if not isinstance(raw, dict):
            raise RuleError("自定义决策模型必须是对象")
        base_url = str(raw.get("base_url") or "").strip()
        if not base_url:
            raise RuleError("自定义决策模型的接口地址不能为空")
        if not base_url.startswith(("http://", "https://")):
            raise RuleError("自定义决策模型的接口地址需以 http:// 或 https:// 开头")
        if len(base_url) > MAX_DECISION_URL_LENGTH:
            raise RuleError("自定义决策模型的接口地址过长")
        model = str(raw.get("model") or "").strip()
        if not model:
            raise RuleError("自定义决策模型的模型 ID 不能为空")
        if len(model) > MAX_DECISION_MODEL_LENGTH:
            raise RuleError("自定义决策模型的模型 ID 过长")
        model_id = str(raw.get("id") or "").strip() or _new_decision_model_id()
        if not _is_valid_decision_model_id(model_id):
            raise RuleError("自定义决策模型 ID 只能包含字母、数字、下划线与短横线")
        name = str(raw.get("name") or "").strip() or model
        if len(name) > MAX_DECISION_NAME_LENGTH:
            raise RuleError(
                f"自定义决策模型名称不能超过 {MAX_DECISION_NAME_LENGTH} 个字符"
            )
        return cls(
            id=model_id,
            name=name,
            base_url=base_url,
            model=model,
            api_key=str(raw.get("api_key") or "").strip(),
        )


@dataclass(slots=True)
class Settings:
    """从插件配置页迁移到 WebUI「更多设置」的运行参数。"""

    concurrency: int = DEFAULT_CONCURRENCY
    send_interval_seconds: float = DEFAULT_SEND_INTERVAL_SECONDS
    segmented_enabled: bool = DEFAULT_SEGMENT_ENABLED
    """分段拟人发言：把正文切成片段逐条发送，模拟打字间隔。"""
    segmented_interval_min: float = DEFAULT_SEGMENT_INTERVAL_MIN
    segmented_interval_max: float = DEFAULT_SEGMENT_INTERVAL_MAX
    """片段之间的随机等待区间（秒）；第一段不等。"""
    segmented_threshold: int = DEFAULT_SEGMENT_THRESHOLD
    """原文长度超过它就不分段，整条直发（口径与核心一致：原文 len）。"""
    segmented_pattern: str = DEFAULT_SEGMENT_PATTERN
    """切分正则；匹配必须覆盖全文，覆盖不上则该次整条直发。"""
    buffer_size: int = DEFAULT_BUFFER_SIZE
    cache_images: bool = DEFAULT_CACHE_IMAGES
    cache_audio: bool = DEFAULT_CACHE_AUDIO
    """是否缓存语音：开启后语音原样转存，条数跟着缓冲条数走，随消息淘汰删除。"""
    image_max_edge: int = DEFAULT_IMAGE_MAX_EDGE
    """图片落盘时的最长边（像素），超出则等比缩小。"""
    image_quality: int = DEFAULT_IMAGE_QUALITY
    """JPEG 重编码质量，越低越省磁盘。"""
    buffer_ignored_sender_ids: list[str] = field(default_factory=list)
    """这些用户的消息不进入上下文缓冲（不影响机器人自己的发言）。"""
    decision_models: list[DecisionModel] = field(default_factory=list)
    """自定义决策模型（SystemOne 兼容端点），供规则按 `custom:<id>` 引用。"""
    cache_images_advice_seen: bool = DEFAULT_CACHE_IMAGES_ADVICE_SEEN
    """「建议开启图片缓存」提示是否已展示过：跨会话只提示一次。"""
    astrbot_config_advice_seen: bool = DEFAULT_ASTRBOT_CONFIG_ADVICE_SEEN
    """「建议调整 AstrBot 侧配置」提示是否已展示过：跨会话只提示一次。"""
    decision_debug_enabled: bool = False
    """是否记录完整决策请求与响应；默认关闭，文件按数量与总字节数轮转。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "send_interval_seconds": self.send_interval_seconds,
            "segmented_enabled": self.segmented_enabled,
            "segmented_interval_min": self.segmented_interval_min,
            "segmented_interval_max": self.segmented_interval_max,
            "segmented_threshold": self.segmented_threshold,
            "segmented_pattern": self.segmented_pattern,
            "buffer_size": self.buffer_size,
            "cache_images": self.cache_images,
            "cache_audio": self.cache_audio,
            "image_max_edge": self.image_max_edge,
            "image_quality": self.image_quality,
            "buffer_ignored_sender_ids": list(self.buffer_ignored_sender_ids),
            "decision_models": [model.to_dict() for model in self.decision_models],
            "cache_images_advice_seen": self.cache_images_advice_seen,
            "astrbot_config_advice_seen": self.astrbot_config_advice_seen,
            "decision_debug_enabled": self.decision_debug_enabled,
        }

    @classmethod
    def parse(cls, raw: Any) -> Settings:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise RuleError("设置必须是对象")
        try:
            concurrency = int(raw.get("concurrency", DEFAULT_CONCURRENCY))
        except (TypeError, ValueError) as exc:
            raise RuleError("同时进行的发言数上限必须是整数") from exc
        if not MIN_CONCURRENCY <= concurrency <= MAX_CONCURRENCY:
            raise RuleError(
                f"同时进行的发言数上限需在 {MIN_CONCURRENCY}~{MAX_CONCURRENCY} 之间"
            )

        try:
            interval = float(
                raw.get("send_interval_seconds", DEFAULT_SEND_INTERVAL_SECONDS)
            )
        except (TypeError, ValueError) as exc:
            raise RuleError("两次发言之间的最小间隔必须是数字") from exc
        if not MIN_SEND_INTERVAL_SECONDS <= interval <= MAX_SEND_INTERVAL_SECONDS:
            raise RuleError(
                "两次发言之间的最小间隔需在 "
                f"{MIN_SEND_INTERVAL_SECONDS:g}~{MAX_SEND_INTERVAL_SECONDS:g} 秒之间"
            )

        segmented_enabled = _parse_bool(
            raw.get("segmented_enabled"), DEFAULT_SEGMENT_ENABLED
        )
        try:
            segment_min = float(
                raw.get("segmented_interval_min", DEFAULT_SEGMENT_INTERVAL_MIN)
            )
            segment_max = float(
                raw.get("segmented_interval_max", DEFAULT_SEGMENT_INTERVAL_MAX)
            )
        except (TypeError, ValueError) as exc:
            raise RuleError("片段之间的等待区间必须是数字") from exc
        if not (
            MIN_SEGMENT_INTERVAL <= segment_min <= MAX_SEGMENT_INTERVAL
            and MIN_SEGMENT_INTERVAL <= segment_max <= MAX_SEGMENT_INTERVAL
        ):
            raise RuleError(
                "片段之间的等待区间需在 "
                f"{MIN_SEGMENT_INTERVAL:g}~{MAX_SEGMENT_INTERVAL:g} 秒之间"
            )
        if segment_max < segment_min:
            raise RuleError("片段等待区间的最大值不能小于最小值")

        try:
            segment_threshold = int(
                raw.get("segmented_threshold", DEFAULT_SEGMENT_THRESHOLD)
            )
        except (TypeError, ValueError) as exc:
            raise RuleError("分段字数阈值必须是整数") from exc
        if not MIN_SEGMENT_THRESHOLD <= segment_threshold <= MAX_SEGMENT_THRESHOLD:
            raise RuleError(
                f"分段字数阈值需在 {MIN_SEGMENT_THRESHOLD}~{MAX_SEGMENT_THRESHOLD} 之间"
            )

        raw_pattern = raw.get("segmented_pattern")
        segment_pattern = (
            DEFAULT_SEGMENT_PATTERN if raw_pattern is None else str(raw_pattern).strip()
        )
        if not segment_pattern:
            raise RuleError("分段正则不能为空")
        try:
            validate_pattern(segment_pattern)
        except (re.error, ValueError) as exc:
            raise RuleError(f"分段正则无效：{exc}") from exc

        try:
            buffer_size = int(raw.get("buffer_size", DEFAULT_BUFFER_SIZE))
        except (TypeError, ValueError) as exc:
            raise RuleError("每个会话保留的近期消息条数必须是整数") from exc
        if not MIN_BUFFER_SIZE <= buffer_size <= MAX_BUFFER_SIZE:
            raise RuleError(
                f"每个会话保留的近期消息条数需在 {MIN_BUFFER_SIZE}~{MAX_BUFFER_SIZE} 之间"
            )

        cache_images = _parse_bool(raw.get("cache_images"), DEFAULT_CACHE_IMAGES)
        cache_audio = _parse_bool(raw.get("cache_audio"), DEFAULT_CACHE_AUDIO)

        try:
            image_max_edge = int(raw.get("image_max_edge", DEFAULT_IMAGE_MAX_EDGE))
        except (TypeError, ValueError) as exc:
            raise RuleError("图片最长边必须是整数") from exc
        if not MIN_IMAGE_MAX_EDGE <= image_max_edge <= MAX_IMAGE_MAX_EDGE:
            raise RuleError(
                f"图片最长边需在 {MIN_IMAGE_MAX_EDGE}~{MAX_IMAGE_MAX_EDGE} 像素之间"
            )

        try:
            image_quality = int(raw.get("image_quality", DEFAULT_IMAGE_QUALITY))
        except (TypeError, ValueError) as exc:
            raise RuleError("图片压缩质量必须是整数") from exc
        if not MIN_IMAGE_QUALITY <= image_quality <= MAX_IMAGE_QUALITY:
            raise RuleError(
                f"图片压缩质量需在 {MIN_IMAGE_QUALITY}~{MAX_IMAGE_QUALITY} 之间"
            )

        advice_seen = _parse_bool(
            raw.get("cache_images_advice_seen"),
            DEFAULT_CACHE_IMAGES_ADVICE_SEEN,
        )

        astrbot_advice_seen = _parse_bool(
            raw.get("astrbot_config_advice_seen"),
            DEFAULT_ASTRBOT_CONFIG_ADVICE_SEEN,
        )

        debug_enabled = raw.get("decision_debug_enabled", False)
        if not isinstance(debug_enabled, bool):
            raise RuleError("决策调试日志开关必须是布尔值")

        return cls(
            concurrency=concurrency,
            send_interval_seconds=interval,
            segmented_enabled=segmented_enabled,
            segmented_interval_min=segment_min,
            segmented_interval_max=segment_max,
            segmented_threshold=segment_threshold,
            segmented_pattern=segment_pattern,
            buffer_size=buffer_size,
            cache_images=cache_images,
            cache_audio=cache_audio,
            image_max_edge=image_max_edge,
            image_quality=image_quality,
            buffer_ignored_sender_ids=_parse_sender_ids(
                raw.get("buffer_ignored_sender_ids")
            ),
            decision_models=_parse_decision_models(raw.get("decision_models")),
            cache_images_advice_seen=advice_seen,
            astrbot_config_advice_seen=astrbot_advice_seen,
            decision_debug_enabled=debug_enabled,
        )


# ---------------------------------------------------------------------- 规则
@dataclass(slots=True)
class Trigger:
    """一个触发入口条件。

    - ``cron``：crontab 表达式，按分钟粒度重复；
    - ``once``：一次性时刻，过去时刻允许保存，但会被标成「已过期」且永不触发；
    - ``window``：每日重复的时段窗口内随机 N 次，可限定周几与最小间隔；
    - ``interval``：从绑定启动时刻起算的固定间隔（可带抖动），不按日历对齐。
    """

    type: str
    expr: str = ""
    at: datetime | None = None
    start: str = ""
    end: str = ""
    count: int = 1
    gap_minutes: int = DEFAULT_WINDOW_GAP_MINUTES
    """时段随机里任意两次触发的最小间隔（分钟）。"""
    days: list[int] = field(default_factory=list)
    """时段随机生效的星期（ISO 1~7）；空列表表示每天。"""
    every_minutes: int = DEFAULT_INTERVAL_MINUTES
    """固定间隔触发的周期（分钟）。"""
    jitter_percent: int = DEFAULT_INTERVAL_JITTER
    """固定间隔触发的抖动比例（0~60，周期 * 比例）。"""

    def is_expired(self, now: datetime | None = None) -> bool:
        """一次性触发是否已过期（其它类型恒为 False）。"""
        if self.type != TRIGGER_ONCE or self.at is None:
            return False
        moment = now or datetime.now(dt_timezone.utc)
        return self.at <= moment

    def to_dict(self) -> dict[str, Any]:
        if self.type == TRIGGER_CRON:
            return {"type": self.type, "expr": self.expr}
        if self.type == TRIGGER_ONCE:
            return {"type": self.type, "at": self.at.isoformat() if self.at else ""}
        if self.type == TRIGGER_INTERVAL:
            return {
                "type": self.type,
                "every_minutes": self.every_minutes,
                "jitter_percent": self.jitter_percent,
            }
        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "count": self.count,
            "gap_minutes": self.gap_minutes,
            "days": list(self.days),
        }

    @classmethod
    def parse(cls, raw: Any, tz: ZoneInfo, *, strict: bool = True) -> Trigger:
        """``strict=False`` 用于读盘：历史数据里的窗口可能塞不下最小间隔，
        此时把间隔降到可行值，而不是让整条规则被跳过。"""
        if not isinstance(raw, dict):
            raise RuleError("触发入口必须是对象")
        trigger_type = str(raw.get("type") or "").strip()
        if trigger_type not in TRIGGER_TYPES:
            raise RuleError(f"未知的触发入口类型：{trigger_type}")

        if trigger_type == TRIGGER_CRON:
            expr = str(raw.get("expr") or "").strip()
            if not expr:
                raise RuleError("crontab 表达式不能为空")
            _validate_cron(expr, tz)
            return cls(type=trigger_type, expr=expr)

        if trigger_type == TRIGGER_ONCE:
            # 过去的时刻照常保存：规则页会标成「已过期」，用户改个时间即可复用，
            # 不必删掉触发器重建。
            return cls(type=trigger_type, at=_parse_datetime(raw.get("at"), tz))

        if trigger_type == TRIGGER_INTERVAL:
            try:
                every = int(raw.get("every_minutes", DEFAULT_INTERVAL_MINUTES))
            except (TypeError, ValueError) as exc:
                raise RuleError("固定间隔的周期必须是整数分钟") from exc
            if not 1 <= every <= MAX_INTERVAL_MINUTES:
                raise RuleError(f"固定间隔的周期需在 1~{MAX_INTERVAL_MINUTES} 分钟之间")
            try:
                jitter = int(raw.get("jitter_percent", DEFAULT_INTERVAL_JITTER))
            except (TypeError, ValueError) as exc:
                raise RuleError("固定间隔的抖动比例必须是整数") from exc
            if not 0 <= jitter <= MAX_INTERVAL_JITTER:
                raise RuleError(f"固定间隔的抖动比例需在 0~{MAX_INTERVAL_JITTER} 之间")
            return cls(
                type=trigger_type,
                every_minutes=every,
                jitter_percent=jitter,
            )

        return _parse_window_trigger(raw, strict=strict)


def _parse_window_trigger(raw: dict[str, Any], *, strict: bool) -> Trigger:
    start = str(raw.get("start") or "").strip()
    end = str(raw.get("end") or "").strip()
    start_hm, end_hm = parse_hhmm(start), parse_hhmm(end)
    if not start_hm or not end_hm:
        raise RuleError("时段随机的开始/结束时间必须是 HH:MM")
    if _minute_index(start_hm) >= _minute_index(end_hm):
        raise RuleError("时段随机的结束时间必须晚于开始时间（不支持跨天）")
    try:
        count = int(raw.get("count", 1))
    except (TypeError, ValueError) as exc:
        raise RuleError("时段随机的触发次数必须是整数") from exc
    if not 1 <= count <= MAX_WINDOW_COUNT:
        raise RuleError(f"时段随机次数需在 1~{MAX_WINDOW_COUNT} 之间")

    try:
        gap = int(raw.get("gap_minutes", DEFAULT_WINDOW_GAP_MINUTES))
    except (TypeError, ValueError) as exc:
        raise RuleError("时段随机的最小间隔必须是整数分钟") from exc
    if not 1 <= gap <= MAX_WINDOW_GAP_MINUTES:
        raise RuleError(f"时段随机的最小间隔需在 1~{MAX_WINDOW_GAP_MINUTES} 分钟之间")

    span = _minute_index(end_hm) - _minute_index(start_hm)
    if count > 1 and (count - 1) * gap >= span:
        if strict:
            raise RuleError(
                f"窗口 {start}-{end} 只有 {span} 分钟，"
                f"放不下 {count} 次触发（每次至少间隔 {gap} 分钟）："
                "请扩大窗口、减少次数或调小最小间隔"
            )
        # 读盘时降级：窗口本来就塞不下，退回可行间隔（采样时再兜底成均匀分布）。
        gap = max(1, span // (count - 1))

    return Trigger(
        type=TRIGGER_WINDOW,
        start=start,
        end=end,
        count=count,
        gap_minutes=gap,
        days=_parse_trigger_days(raw.get("days")),
    )


def _parse_trigger_days(raw: Any) -> list[int]:
    """星期列表：缺省或全选都归一成空列表（每天），保证落盘形态唯一。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuleError("时段随机的星期必须是列表")
    days: set[int] = set()
    for item in raw:
        try:
            day = int(item)
        except (TypeError, ValueError) as exc:
            raise RuleError("时段随机的星期必须是 1~7 的整数") from exc
        if day not in TRIGGER_WEEKDAYS:
            raise RuleError("时段随机的星期必须是 1~7 的整数（周一为 1）")
        days.add(day)
    return [] if len(days) == len(TRIGGER_WEEKDAYS) else sorted(days)


@dataclass(slots=True)
class IntentConfig:
    """发言意向机制配置。"""

    mode: str = INTENT_DIRECT
    decision_provider_id: str = ""
    threshold: float = DEFAULT_THRESHOLD
    pass_probability: float = DEFAULT_PASS_PROBABILITY
    """「按概率放行」模式下每次发言的概率，0~1；1.0 即必定发言。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "decision_provider_id": self.decision_provider_id,
            "threshold": self.threshold,
            "pass_probability": self.pass_probability,
        }

    @classmethod
    def parse(cls, raw: Any) -> IntentConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise RuleError("发言意向必须是对象")
        mode = str(raw.get("mode") or INTENT_DIRECT).strip()
        if mode not in (INTENT_DIRECT, INTENT_DECISION):
            raise RuleError(f"未知的发言意向机制：{mode}")
        try:
            threshold = float(raw.get("threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError) as exc:
            raise RuleError("判定阈值必须是数字") from exc
        if not 0.0 <= threshold <= 1.0:
            raise RuleError("判定阈值需在 0~1 之间")
        try:
            probability = float(raw.get("pass_probability", DEFAULT_PASS_PROBABILITY))
        except (TypeError, ValueError) as exc:
            raise RuleError("放行概率必须是数字") from exc
        if not 0.0 <= probability <= 1.0:
            raise RuleError("放行概率需在 0~1 之间")
        return cls(
            mode=mode,
            decision_provider_id=str(raw.get("decision_provider_id") or "").strip(),
            threshold=threshold,
            pass_probability=probability,
        )


@dataclass(slots=True)
class Rule:
    """一条 Ambient 规则：只是一份配置，启停由绑定决定。"""

    id: str
    name: str
    triggers: list[Trigger] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    intent: IntentConfig = field(default_factory=IntentConfig)
    chat_provider_id: str = ""
    context_turns: int = DEFAULT_CONTEXT_TURNS
    """注入给模型的最近消息**条数**（字段名为历史兼容保留，语义不是对话轮次）。"""
    inject_persona: bool = True
    """是否把会话当前人格注入给决策模型与对话模型，两者同开同关（规则级配置）。"""

    updated_at: str = ""
    """最近一次保存内容的时刻（ISO 字符串），只用于 WebUI 排序，见 ADR-0014。

    历史数据没有该字段，读成空串，不会伪造时间。
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "triggers": [trigger.to_dict() for trigger in self.triggers],
            "blockers": list(self.blockers),
            "intent": self.intent.to_dict(),
            "chat_provider_id": self.chat_provider_id,
            "context_turns": self.context_turns,
            "inject_persona": self.inject_persona,
            "updated_at": self.updated_at,
        }

    def blocker(self, blocker_type: str) -> dict[str, Any] | None:
        for item in self.blockers:
            if item.get("type") == blocker_type:
                return item
        return None

    @classmethod
    def parse(cls, raw: Any, tz: ZoneInfo, *, strict: bool = True) -> Rule:
        if not isinstance(raw, dict):
            raise RuleError("规则必须是对象")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise RuleError("规则名称不能为空")

        triggers_raw = raw.get("triggers") or []
        if not isinstance(triggers_raw, list) or not triggers_raw:
            raise RuleError("至少需要一个触发入口")
        triggers = [Trigger.parse(item, tz, strict=strict) for item in triggers_raw]

        try:
            context_turns = int(raw.get("context_turns", DEFAULT_CONTEXT_TURNS))
        except (TypeError, ValueError) as exc:
            raise RuleError("对话条数必须是整数") from exc
        if not 1 <= context_turns <= MAX_CONTEXT_TURNS:
            raise RuleError(f"对话条数需在 1~{MAX_CONTEXT_TURNS} 之间")

        rule_id = str(raw.get("id") or "").strip() or _new_rule_id()
        return cls(
            id=rule_id,
            name=name,
            triggers=triggers,
            blockers=_parse_blockers(raw.get("blockers")),
            intent=IntentConfig.parse(raw.get("intent")),
            chat_provider_id=str(raw.get("chat_provider_id") or "").strip(),
            context_turns=context_turns,
            inject_persona=bool(raw.get("inject_persona", True)),
            updated_at=str(raw.get("updated_at") or "").strip(),
        )


# ---------------------------------------------------------------------- 绑定
@dataclass(slots=True)
class Selection:
    """一对多时的目标会话挑选策略。"""

    mode: str = SELECTION_ALL
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "count": self.count}

    @classmethod
    def parse(cls, raw: Any) -> Selection:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise RuleError("会话挑选策略必须是对象")
        mode = str(raw.get("mode") or SELECTION_ALL).strip()
        if mode not in SELECTION_MODES:
            raise RuleError(f"未知的会话挑选策略：{mode}")
        try:
            count = int(raw.get("count", 1))
        except (TypeError, ValueError) as exc:
            raise RuleError("随机挑选个数必须是整数") from exc
        if not 1 <= count <= MAX_SELECTION_COUNT:
            raise RuleError(f"随机挑选个数需在 1~{MAX_SELECTION_COUNT} 之间")
        return cls(mode=mode, count=count)


@dataclass(slots=True)
class Binding:
    """一条 Ambient 绑定：机器人应用 + 规则 + 目标会话集合。

    - 一个机器人应用（平台适配器实例，即 UMO 前缀）可以有多条绑定，每条各带一条规则；
      「一条绑定多条规则」不提供，要给同一个号再加规则就再建一条绑定（见 ADR-0012）；
    - 目标会话在机器人应用内唯一：同一个 UMO 最多出现在该机器人的一条绑定里；
    - 一条规则可以被多个机器人应用绑定，也可以绑定多个私聊/群聊；
    - 绑定自身可启动/停止，不影响规则定义。
    """

    id: str
    platform_id: str
    rule_id: str
    name: str = ""
    """绑定名称（人类可读标签）。留空时由 :meth:`RuleStore.upsert_binding` 兜底生成。"""
    targets: list[str] = field(default_factory=list)
    selection: Selection = field(default_factory=Selection)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "platform_id": self.platform_id,
            "rule_id": self.rule_id,
            "name": self.name,
            "targets": list(self.targets),
            "selection": self.selection.to_dict(),
            "enabled": self.enabled,
        }

    @classmethod
    def parse(cls, raw: Any) -> Binding:
        if not isinstance(raw, dict):
            raise RuleError("Ambient 绑定必须是对象")

        platform_id = str(raw.get("platform_id") or "").strip()
        if not platform_id:
            raise RuleError("请选择或填写机器人应用")

        rule_id = str(raw.get("rule_id") or "").strip()
        if not rule_id:
            raise RuleError("请选择要绑定的规则")

        targets_raw = raw.get("targets") or []
        if not isinstance(targets_raw, list):
            raise RuleError("目标会话必须是列表")
        targets: list[str] = []
        for item in targets_raw:
            target = parse_target(str(item))
            if target is None:
                raise RuleError(f"目标会话格式无效：{item}")
            if target.platform_id != platform_id:
                raise RuleError(f"目标会话 {target.umo} 不属于机器人应用 {platform_id}")
            if target.umo not in targets:
                targets.append(target.umo)
        if not targets:
            raise RuleError("至少需要一个目标会话（私聊或群聊）")

        binding_id = str(raw.get("id") or "").strip() or _new_binding_id()
        return cls(
            id=binding_id,
            platform_id=platform_id,
            rule_id=rule_id,
            name=str(raw.get("name") or "").strip(),
            targets=targets,
            selection=Selection.parse(raw.get("selection")),
            enabled=bool(raw.get("enabled", True)),
        )


# ---------------------------------------------------------------------- 校验
def _new_rule_id() -> str:
    return f"r_{uuid.uuid4().hex[:8]}"


def _new_binding_id() -> str:
    return f"b_{uuid.uuid4().hex[:8]}"


def _new_decision_model_id() -> str:
    return f"dm_{uuid.uuid4().hex[:8]}"


def _is_valid_decision_model_id(value: str) -> bool:
    if not value or len(value) > 64:
        return False
    return all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in value)


def _parse_bool(raw: Any, default: bool) -> bool:
    """设置里的开关：缺字段用默认值，兼容 JSON 布尔与表单传上来的字符串。"""
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def _parse_sender_ids(raw: Any) -> list[str]:
    """解析「不进上下文缓冲的用户 ID」：接受列表或逗号/空白分隔的字符串。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        items: list[Any] = raw.replace("，", ",").replace(",", " ").split()
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        raise RuleError("过滤的用户 ID 必须是 ID 列表")
    sender_ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        if len(value) > MAX_SENDER_ID_LENGTH:
            raise RuleError(f"用户 ID 不能超过 {MAX_SENDER_ID_LENGTH} 个字符：{value}")
        if any(ch.isspace() for ch in value):
            raise RuleError(f"用户 ID 不能包含空白字符：{value}")
        seen.add(value)
        sender_ids.append(value)
    if len(sender_ids) > MAX_BUFFER_IGNORED_SENDERS:
        raise RuleError(f"过滤的用户 ID 最多 {MAX_BUFFER_IGNORED_SENDERS} 个")
    return sender_ids


def _parse_decision_models(raw: Any) -> list[DecisionModel]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuleError("自定义决策模型必须是列表")
    if len(raw) > MAX_DECISION_MODELS:
        raise RuleError(f"自定义决策模型最多 {MAX_DECISION_MODELS} 个")
    models: list[DecisionModel] = []
    seen: set[str] = set()
    for item in raw:
        model = DecisionModel.parse(item)
        if model.id in seen:
            raise RuleError(f"自定义决策模型 ID 重复：{model.id}")
        seen.add(model.id)
        models.append(model)
    return models


def _minute_index(value: tuple[int, int]) -> int:
    return value[0] * 60 + value[1]


def _validate_cron(expr: str, tz: ZoneInfo) -> None:
    try:
        CronTrigger.from_crontab(expr, timezone=tz)
    except (TypeError, ValueError) as exc:
        raise RuleError(f"crontab 表达式无法解析：{expr}") from exc


def _parse_datetime(value: Any, tz: ZoneInfo) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise RuleError("一次性触发必须指定时间")
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RuleError(f"无法解析时间：{text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return parsed


def _parse_blockers(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuleError("拦截机制必须是列表")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise RuleError("拦截机制条目必须是对象")
        blocker_type = str(item.get("type") or "").strip()
        if blocker_type not in BLOCKER_TYPES:
            raise RuleError(f"未知的拦截类型：{blocker_type}")
        if blocker_type in seen:
            raise RuleError(f"拦截类型重复：{blocker_type}")
        seen.add(blocker_type)

        if blocker_type == BLOCKER_SILENCE:
            result.append({"type": blocker_type, "windows": _parse_silence(item)})
        elif blocker_type == BLOCKER_COOLDOWN:
            try:
                minutes = int(item.get("minutes", DEFAULT_COOLDOWN_MINUTES))
            except (TypeError, ValueError) as exc:
                raise RuleError("冷却分钟数必须是整数") from exc
            if not 1 <= minutes <= 24 * 60:
                raise RuleError("冷却分钟需在 1~1440 之间")
            result.append({"type": blocker_type, "minutes": minutes})
        elif blocker_type == BLOCKER_UNANSWERED:
            try:
                count = int(item.get("count", DEFAULT_UNANSWERED_COUNT))
            except (TypeError, ValueError) as exc:
                raise RuleError("连续发送条数必须是整数") from exc
            if not 1 <= count <= MAX_UNANSWERED_COUNT:
                raise RuleError(f"连续发送条数需在 1~{MAX_UNANSWERED_COUNT} 之间")
            try:
                minutes = int(item.get("minutes", DEFAULT_UNANSWERED_MINUTES))
            except (TypeError, ValueError) as exc:
                raise RuleError("等待分钟数必须是整数") from exc
            if not 1 <= minutes <= MAX_UNANSWERED_MINUTES:
                raise RuleError(
                    f"等待分钟数需在 1~{MAX_UNANSWERED_MINUTES} 之间"
                    "（设为极大值即视为永久等待）"
                )
            result.append({"type": blocker_type, "count": count, "minutes": minutes})
        else:
            try:
                min_count = int(item.get("min_count", DEFAULT_UNREAD_MIN_COUNT))
            except (TypeError, ValueError) as exc:
                raise RuleError("未读条数必须是整数") from exc
            if not 0 <= min_count <= 10000:
                raise RuleError("未读条数需在 0~10000 之间")
            result.append({"type": blocker_type, "min_count": min_count})
    return result


def _parse_silence(item: dict[str, Any]) -> list[dict[str, str]]:
    windows_raw = item.get("windows") or []
    if not isinstance(windows_raw, list):
        raise RuleError("静默时段必须是列表")
    windows: list[dict[str, str]] = []
    for window in windows_raw:
        if not isinstance(window, dict):
            raise RuleError("静默时段条目必须是对象")
        start = str(window.get("start") or "").strip()
        end = str(window.get("end") or "").strip()
        if not parse_hhmm(start) or not parse_hhmm(end):
            raise RuleError("静默时段的时间必须是 HH:MM")
        if start == end:
            raise RuleError("静默时段的开始与结束不能相同")
        windows.append({"start": start, "end": end})
    if not windows:
        raise RuleError("静默时段至少需要一个时间窗口")
    return windows
