"""把 LivingMemory 的长期记忆接进判定与生成。

记忆只服务两个消费者，取数时机跟着意向模式走（``docs/adr/0013``）：

- 决策模型：通过拦截后、判定前取一次，判定读 ``state.memory``，生成复用同一份；
- 概率放行：判定不读语境，放行后由生成侧取，被拒的那次完全不检索。

无论哪条路径，一次发言最多召回一次。召回不更新 LivingMemory 的访问统计：
取数发生在「还不知道会不会发言」的时刻，把这次检索记成「记忆被使用」会把被
拦截、被放弃的检索也算成强化；而想等发送前再补记，又要依赖对方没有公开的
批量接口。所以本插件的记忆只充当语境，不参与记忆自身的衰减排序。

字符预算不写死，而是按模型能力推导（ADR-0009）：判定侧读该决策模型自己声明的
窗口（自定义模型在「更多设置」里填，官方判断提供商读 ``max_context_tokens``），
生成侧自动读对话模型的 ``max_context_tokens``；换算沿用「1 字 ≈ 1 token」的保守
假设，不引入 tokenizer——不同模型的 tokenizer 不同，中英混排时误差反而更大。

预算之内怎么装填由 ``memory.fit`` 统一决定：按 ``score`` 降序、同分短优先，整条
装得下就整条放，余量不足时截断本条收尾。判定与生成共用这套规则，避免两处看到
不同的记忆。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..conversation.messages import strip_placeholders
from ..memory import MemoryItem, senders_for_recall
from ..memory.fit import (
    ENTRY_SHARE,
    TAIL_MIN_CHARS,
    fit_memories,
    order_memories,
)
from ..rules.models import CUSTOM_DECISION_PREFIX, DEFAULT_DECISION_CONTEXT_TOKENS

if TYPE_CHECKING:
    from ..conversation.messages import ChatMessage
    from ..memory import LivingMemoryRecall
    from ..rules.models import Settings

RECALL_TIMEOUT_SECONDS = 3.0
"""单次召回的超时。

比 ``LivingMemoryRecall`` 的默认 8 秒更短：这里处在一次发言的 60 秒预算中间，
记忆属于锦上添花，等不起就放弃本次召回（``recall`` 内部 fail-open）。
"""

QUERY_MESSAGE_COUNT = 3
"""召回关键词用最近几条非 bot 消息拼成。"""

QUERY_MAX_CHARS = 120

# ---------------------------------------------------------------- 判定侧预算
DECISION_STATE_RATIO = 0.85
"""上下文窗口里留给判定 state 的比例：剩下的给 questions 与输出。"""

DECISION_STATE_MIN_CHARS = 3_000
"""再小的窗口也要留出人格与最近对话：低于这个值判定会退化成「只看一句话」。"""

DECISION_STATE_HARD_MAX_CHARS = 48_000
"""state 上限：判定模型每次都要读完整份，再大只是拖慢判定与烧钱。"""

DECISION_MEMORY_RATIO = 0.06
"""记忆在 state 预算里的占比：语境是配角，旧对话才是判定依据。"""

DECISION_MEMORY_MIN_CHARS = 240
DECISION_MEMORY_MAX_CHARS = 2_000

DECISION_ENTRY_SHARE = ENTRY_SHARE
"""单条记忆最多占记忆总预算的比例：防止一条长记忆把其余条目挤光。

沿用 ``memory.fit`` 的同一比例，判定侧与生成侧的取舍口径保持一致。
"""

DEFAULT_DECISION_STATE_CHARS = int(
    DEFAULT_DECISION_CONTEXT_TOKENS * DECISION_STATE_RATIO
)
"""默认窗口（8K）推导出的 state 预算，约 6963 字。"""

DECISION_STATE_CHAR_BUDGET = DEFAULT_DECISION_STATE_CHARS
"""``_fit_conversation`` 的默认预算；实际值由 :func:`decision_budget` 按窗口给出。"""

DECISION_MEMORY_CHARS = max(
    DECISION_MEMORY_MIN_CHARS,
    min(
        int(DEFAULT_DECISION_STATE_CHARS * DECISION_MEMORY_RATIO),
        DECISION_MEMORY_MAX_CHARS,
    ),
)
"""默认窗口下的记忆总预算，约 417 字。"""

# ---------------------------------------------------------------- 生成侧预算
DELIVERY_MEMORY_CHARS = 600
"""默认窗口（8K）下注入对话提示的字符预算。"""

DELIVERY_MEMORY_BASE_WINDOW = 8_192
"""基线窗口：模型窗口翻倍，记忆预算同比例放大。"""


@dataclass(frozen=True, slots=True)
class DecisionBudget:
    """一次判定请求的字符预算分配。"""

    state_chars: int
    """``state`` 里人格 + 对话的总预算。"""

    memory_chars: int
    """其中记忆条目的总预算。"""

    entry_chars: int
    """单条记忆的正文上限，按总预算的 :data:`DECISION_ENTRY_SHARE` 推导。"""


def decision_budget(tokens: int | None = None) -> DecisionBudget:
    """按决策模型的上下文窗口推导判定预算。

    Args:
        tokens: 该决策模型声明的窗口（token）；``None`` / 非法值时用默认窗口。

    Returns:
        ``state`` / 记忆总量 / 单条上限三档预算；窗口越大，三档同步变大。
    """
    context_tokens = _positive_int(tokens, DEFAULT_DECISION_CONTEXT_TOKENS)
    state = _clamp(
        int(context_tokens * DECISION_STATE_RATIO),
        DECISION_STATE_MIN_CHARS,
        DECISION_STATE_HARD_MAX_CHARS,
    )
    memory = _clamp(
        int(state * DECISION_MEMORY_RATIO),
        DECISION_MEMORY_MIN_CHARS,
        DECISION_MEMORY_MAX_CHARS,
    )
    # 单条上限不再按条数摊薄：条数只决定「装几条」，长度由预算与相关度决定。
    entry = max(TAIL_MIN_CHARS, int(memory * DECISION_ENTRY_SHARE))
    return DecisionBudget(state, memory, entry)


def decision_window(context: Any, settings: Settings, ref: str) -> int:
    """规则引用的决策模型声明的上下文窗口。

    窗口按模型各自配置，没有全局开关：

    - ``custom:<id>``：读「更多设置」里该模型自己填的 ``context_tokens``；
    - 官方判断提供商：读 AstrBot 供应商配置里的 ``max_context_tokens``，与对话侧
      同一个字段；
    - 引用为空（走核心默认判断提供商）时取第一个判断提供商的声明值；
    - 都读不到时用 :data:`DEFAULT_DECISION_CONTEXT_TOKENS`，判定照常进行。
    """
    cleaned = (ref or "").strip()
    if cleaned.startswith(CUSTOM_DECISION_PREFIX):
        model_id = cleaned[len(CUSTOM_DECISION_PREFIX) :]
        for model in getattr(settings, "decision_models", ()) or ():
            if getattr(model, "id", "") == model_id:
                return _positive_int(
                    getattr(model, "context_tokens", None),
                    DEFAULT_DECISION_CONTEXT_TOKENS,
                )
        return DEFAULT_DECISION_CONTEXT_TOKENS
    return (
        _official_decision_window(context, cleaned) or DEFAULT_DECISION_CONTEXT_TOKENS
    )


def _official_decision_window(context: Any, provider_id: str) -> int:
    """官方判断提供商声明的窗口；未声明或读不到时返回 0。"""
    manager = getattr(context, "provider_manager", None)
    for provider in getattr(manager, "decision_provider_insts", None) or ():
        if provider_id:
            meta = getattr(provider, "meta", None)
            meta = meta() if callable(meta) else None
            if str(getattr(meta, "id", "")) != provider_id:
                continue
        config = getattr(provider, "provider_config", None) or {}
        return _int_or_zero(config.get("max_context_tokens"))
    return 0


def delivery_memory_chars(context: Any, provider_id: str) -> int:
    """按对话模型的上下文窗口推导记忆块预算。

    只随窗口等比放大，不设上限：生成侧一次最多带 ``memory_recall_k`` 条（至多
    5 条）记忆，总量本就有限，封顶只会让大窗口空转。读不到窗口（自建端点没填
    ``max_context_tokens``、模型不在元数据里）时退回默认 600 字：宁可少给一点，
    也不猜一个可能超窗的数字。
    """
    window = provider_context_tokens(context, provider_id)
    if window <= 0:
        return DELIVERY_MEMORY_CHARS
    scaled = DELIVERY_MEMORY_CHARS * (window / DELIVERY_MEMORY_BASE_WINDOW)
    return max(DELIVERY_MEMORY_CHARS, int(scaled))


def provider_context_tokens(context: Any, provider_id: str) -> int:
    """对话模型的上下文窗口；读不到返回 0。

    先看 AstrBot 供应商配置里的 ``max_context_tokens``（agent 路径会自动回填），
    再查内置模型元数据；两者都没有时由调用方决定兜底值。
    """
    if not provider_id:
        return 0
    try:
        provider = context.get_provider_by_id(provider_id)
    except Exception:  # noqa: BLE001
        return 0
    if provider is None:
        return 0

    config = getattr(provider, "provider_config", None) or {}
    declared = _int_or_zero(config.get("max_context_tokens"))
    if declared > 0:
        return declared

    getter = getattr(provider, "get_model", None)
    if not callable(getter):
        return 0
    try:
        model = str(getter() or "").strip()
    except Exception:  # noqa: BLE001
        return 0
    if not model:
        return 0

    try:
        from astrbot.core.utils.llm_metadata import LLM_METADATAS
    except Exception:  # noqa: BLE001
        return 0
    info = LLM_METADATAS.get(model)
    if not isinstance(info, dict):
        return 0
    limit = info.get("limit")
    if not isinstance(limit, dict):
        return 0
    return _int_or_zero(limit.get("context"))


def recall_query(
    messages: Sequence[ChatMessage],
    *,
    limit: int = QUERY_MESSAGE_COUNT,
    max_chars: int = QUERY_MAX_CHARS,
) -> str:
    """最近几条非 bot 消息拼成召回关键词。

    自主发言没有「当前用户消息」，最近的对话本身就是主题线索；机器人自己的发言
    不算线索，否则会不断用自己说过的话去检索自己。组件占位（``[Image]`` 这类）
    一律剔除：那是「这里有个附件」，不是说过的话；只剩占位的消息不作数。
    """
    parts: list[str] = []
    for message in reversed(messages):
        if message.is_bot:
            continue
        text = strip_placeholders(message.body_text())
        if not text:
            continue
        parts.append(text)
        if len(parts) >= limit:
            break
    return " ".join(reversed(parts))[:max_chars]


async def recall_memories(
    recall: LivingMemoryRecall | None,
    settings: Settings,
    umo: str,
    messages: Sequence[ChatMessage],
    *,
    timeout: float = RECALL_TIMEOUT_SECONDS,
) -> list[MemoryItem]:
    """按「更多设置」的开关与参数召回一次；任何不可用都返回空列表。

    Args:
        recall: 召回客户端；``None`` 表示插件没构造（旧调用方或测试）。
        settings: 运行参数，读 ``memory_recall_*`` 三项。
        umo: 目标会话，决定记忆作用域与人格。
        messages: 该会话上下文缓冲，按时间正序。
        timeout: 覆盖召回超时。

    Returns:
        召回结果；关闭、无关键词、未安装或失败时为空列表。
    """
    if recall is None or not settings.memory_recall_enabled:
        return []
    query = recall_query(messages)
    if not query:
        return []
    return await recall.recall(
        query,
        umo=umo,
        senders=senders_for_recall(umo, messages, limit=settings.memory_recall_fanout),
        k=settings.memory_recall_k,
        timeout=timeout,
    )


def memory_entries(
    memories: Sequence[MemoryItem],
    *,
    entry_chars: int | None = None,
    max_chars: int = DECISION_MEMORY_CHARS,
    min_tail: int = TAIL_MIN_CHARS,
) -> list[str]:
    """把记忆压成决策 state 用的短条目：单行、按预算装填。

    顺序按 ``score`` 降序、同分短优先；整条装得下就整条放，余量不足时截断本条
    收尾（``memory.fit``）。决策模型拿到的 ``state`` 是结构化内容，这里保持
    「字符串列表」的形状，与 ``conversation`` 的条目风格一致。
    """
    pairs = [(item, " ".join(str(item.content or "").split())) for item in memories]
    fitted = fit_memories(
        order_memories(pairs),
        budget=max_chars,
        entry_cap=entry_chars,
        min_tail=min_tail,
    )
    return [text for _item, text in fitted]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _int_or_zero(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


__all__ = [
    "DECISION_ENTRY_SHARE",
    "DECISION_MEMORY_CHARS",
    "DECISION_STATE_CHAR_BUDGET",
    "DELIVERY_MEMORY_CHARS",
    "DecisionBudget",
    "QUERY_MAX_CHARS",
    "QUERY_MESSAGE_COUNT",
    "RECALL_TIMEOUT_SECONDS",
    "decision_budget",
    "decision_window",
    "delivery_memory_chars",
    "memory_entries",
    "provider_context_tokens",
    "recall_memories",
    "recall_query",
]
