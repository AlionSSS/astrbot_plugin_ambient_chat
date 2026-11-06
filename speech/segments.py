"""分段拟人发言：把一次发言的正文切成若干「片段」。

核心 AstrBot 也有一个分段回复（``platform_settings.segmented_reply``），但它挂在
管线的 ``result_decorate`` / ``respond`` 两个 stage 上；插件走「直接生成」，不经过
管线（ADR-0011），所以只能在这里自建一份（ADR-0032）。

本模块只负责纯文本切分，不发送、不等待；投递循环在 :mod:`speech.delivery`。
"""

from __future__ import annotations

import re

from astrbot.api import logger

DEFAULT_SEGMENT_ENABLED = True
"""默认开启：插件尚无存量用户，不存在「悄悄改变既有发言形态」的顾虑。"""

DEFAULT_SEGMENT_INTERVAL_MIN = 1.5
DEFAULT_SEGMENT_INTERVAL_MAX = 3.5
"""片段之间的随机等待区间（秒），与核心默认值一致。"""

MIN_SEGMENT_INTERVAL = 0.0
MAX_SEGMENT_INTERVAL = 30.0

DEFAULT_SEGMENT_THRESHOLD = 150
"""原文长度超过它就不分段；计量口径是原文 ``len``，与核心判定一致。"""

MIN_SEGMENT_THRESHOLD = 1
MAX_SEGMENT_THRESHOLD = 2000

DEFAULT_SEGMENT_PATTERN = r".*?[。？！~…]+|.+$"
"""与核心默认正则一致：按句末标点切，最后没有标点的残余整句收尾。"""

MAX_SEGMENT_PATTERN_LENGTH = 500
"""分段正则的最大长度：够表达，也不至于让病态正则拖死切分。"""

MAX_SEGMENTS = 8
"""一条发言最多切成几段；超限说明正则配得不对，退回整条直发。"""

SEGMENT_BLOCKED_PLATFORMS = frozenset(
    {"qq_official_webhook", "weixin_official_account", "dingtalk"}
)
"""被动回复 / 条数受限的平台不分段：多条会被拒绝或快速烧掉主动消息额度。"""


def validate_pattern(pattern: str) -> None:
    """保存前校验分段正则，非法时抛 ``re.error`` 或 ``ValueError``。"""
    if len(pattern) > MAX_SEGMENT_PATTERN_LENGTH:
        raise ValueError(f"分段正则不能超过 {MAX_SEGMENT_PATTERN_LENGTH} 个字符")
    re.compile(pattern)


def split(text: str, *, threshold: int, pattern: str) -> list[str]:
    """把 ``text`` 切成片段；不适合切分时原样返回 ``[text]``。

    保真不变量：片段按序拼接等于原文，只允许去掉片段首尾空白、丢弃纯空白片段。
    为此要求正则的匹配区间首尾相接且覆盖全文——覆盖不上就退回整条直发，
    宁可少分段也不丢字。
    """
    if not text or len(text) > threshold:
        return [text]
    try:
        matches = list(re.finditer(pattern, text, re.DOTALL | re.MULTILINE))
    except re.error:
        logger.warning("[Ambient] 分段正则无效，本次整条发送：%r", pattern)
        return [text]

    pieces: list[str] = []
    cursor = 0
    for match in matches:
        if match.start() != cursor:
            return [text]
        cursor = match.end()
        piece = match.group(0).strip()
        if piece:
            pieces.append(piece)
    if cursor != len(text) or len(pieces) <= 1 or len(pieces) > MAX_SEGMENTS:
        return [text]
    return pieces


def is_blocked_platform(platform_id: str) -> bool:
    """该平台是否不适合分段（见 :data:`SEGMENT_BLOCKED_PLATFORMS`）。"""
    return platform_id in SEGMENT_BLOCKED_PLATFORMS
