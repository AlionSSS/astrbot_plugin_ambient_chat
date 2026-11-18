"""决策模型分派：官方判断提供商与「更多设置」里的自定义 SystemOne 配置。

规则里的 ``intent.decision_provider_id`` 有三种取值（见 ``docs/adr/0017``）：

- 空串：交给核心的默认判断提供商；
- ``custom:<id>``：用本插件的 SystemOne SDK 直连「更多设置」里的自定义端点，
  这类端点（如书生端砚）要求 ``noul`` 必须带 ``criteria.true`` / ``criteria.false``；
- 其他：按 provider id 交给核心。

任一分支失败都转成 :class:`DecisionError`，由调用方 fail-closed（ADR-0008）。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.provider import DecisionBooleanQuestion, DecisionError
from astrbot.core.provider.entities import ProviderType

from ..rules.models import (
    CUSTOM_DECISION_PREFIX,
    DEFAULT_DECISION_CONTEXT_TOKENS,
    DecisionModel,
    Settings,
)
from ..rules.store import RuleStore
from .debug_log import write_debug_record
from .systemone_sdk import (
    CallTrace,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    State,
    SystemOneClient,
    SystemOneConfig,
    SystemOneContentRejected,
    choice,
    noul,
    score,
)

DECISION_STATE_RATIO = 0.85
DECISION_STATE_HARD_MAX_CHARS = 48_000
DECISION_STATE_CHAR_BUDGET = int(DEFAULT_DECISION_CONTEXT_TOKENS * DECISION_STATE_RATIO)


def decision_state_chars(tokens: int | None = None) -> int:
    """判定 state 的字符预算；不抬高宿主声明的较小窗口。"""
    context_tokens = _positive_int(tokens, DEFAULT_DECISION_CONTEXT_TOKENS)
    return min(
        int(context_tokens * DECISION_STATE_RATIO), DECISION_STATE_HARD_MAX_CHARS
    )


def decision_window(context: Any, settings: Settings, ref: str) -> int:
    """规则引用的决策模型声明的上下文窗口。

    窗口按模型各自配置，没有全局开关：

    - ``custom:<id>``：读「更多设置」里该模型自己填的 ``context_tokens``；
    - 官方判断提供商：读 AstrBot 供应商配置里的 ``max_context_tokens``，与对话侧
      同一个字段；
    - 引用为空时读取核心当前选择的默认判断提供商；
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
    try:
        if provider_id:
            provider = context.get_provider_by_id(provider_id)
        else:
            # 与核心 DecisionService 使用同一套默认选择（无 umo 时同步和异步入口等价）。
            provider = manager.get_using_provider(ProviderType.DECISION)
        config = getattr(provider, "provider_config", None) or {}
        return _positive_int(config.get("max_context_tokens"), 0)
    except Exception:  # noqa: BLE001 - 读不到窗口不影响判定
        return 0


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


if TYPE_CHECKING:
    import httpx

    from astrbot.api.star import Context

DECISION_TIMEOUT = 10.0
"""自定义决策模型单次 HTTP 尝试的超时秒数。"""

DECISION_TOTAL_TIMEOUT = 15.0
"""整次判断（含重试预算）的超时秒数：主动发言宁可放弃，也不拖住调度。"""

DECISION_MAX_RETRIES = 0
"""默认不重试：一次判断失败就放弃本次发言，避免连续失败时排队等待。"""

DECISION_DEBUG_DIRNAME = "decision_debug"
"""决策模型调试日志目录；控制台「运行信息」直接展示这个路径。"""

NOUL_TRUE_FALLBACK = "该说法成立"
NOUL_FALSE_FALLBACK = "该说法不成立"
"""noul 判定说明的兜底文案。书生端砚等平台要求 ``criteria`` 必带
``true`` / ``false`` 两个键，因此在插件层无条件补齐，不依赖 SDK 的默认值。"""


@dataclass(frozen=True, slots=True)
class NoulPrompt:
    """一个 noul 判断：要判断的陈述，以及什么算「是」/「否」。"""

    instructions: str
    when_true: str = ""
    when_false: str = ""


class DecisionContentRejected(DecisionError):
    """决策平台因内容审核拒绝了请求。

    请求格式合法，是会话内容被判为敏感；单独一类，好让页面给出「改内容」而不是
    「改配置」的提示（见 ADR-0008）。
    """


PROBE_STATE = "我本月被重复扣费了，请把多扣的钱退回来。"
"""测试按钮用的固定输入，便于横向比较不同端点。"""


async def probe_decision_model(
    model: DecisionModel,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    store: RuleStore | None = None,
) -> dict[str, Any]:
    """用**一次**请求同时验证 noul / choice / score 三种题型。

    供「更多设置」的测试按钮使用：不保存模型配置，调试日志遵循开关，返回可直接展示的
    结果（端点、耗时、用量与三种答案）。

    Raises:
        DecisionError: 响应缺少某种题型或结果不符合契约。
        SystemOneError: 网络失败、鉴权失败或响应不合法。
    """
    started = time.perf_counter()
    async with SystemOneClient(
        _config_for(model),
        transport=transport,
    ) as client:
        questions = {
            "noul": noul(
                "这条消息是在要求退款吗？",
                when_true="用户明确要求退款，或指出被重复扣费",
                when_false="与退款、账单无关",
            ),
            "choice": choice(
                "这条消息的语气是？",
                {"angry": "不满或敌意", "calm": "中性或礼貌", "happy": "积极愉快"},
            ),
            "score": score(
                "这条消息有多紧急？",
                ["可以等", "本周处理", "今天处理"],
            ),
        }
        trace = CallTrace()
        try:
            answers = await client.ask(questions, state=PROBE_STATE, trace=trace)
        except Exception as exc:
            _debug_record(
                "decision-probe",
                trace.request_body,
                trace.response_json(),
                started=started,
                store=store,
                meta=_custom_meta(model, client, trace),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _debug_record(
            "decision-probe",
            trace.request_body,
            trace.response_json(),
            started=started,
            store=store,
            meta=_custom_meta(model, client, trace),
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    noul_answer = answers["noul"]
    choice_answer = answers["choice"]
    score_answer = answers["score"]
    if not isinstance(noul_answer, NoulAnswer):
        raise DecisionError("测试响应没有返回 noul 答案")
    if not isinstance(choice_answer, ChoiceAnswer):
        raise DecisionError("测试响应没有返回 choice 答案")
    if not isinstance(score_answer, ScoreAnswer):
        raise DecisionError("测试响应没有返回 score 答案")

    return {
        "endpoint": client.endpoint,
        "model": trace.model or model.model,
        "elapsed_ms": elapsed_ms,
        "usage": trace.usage.model_dump() if trace.usage is not None else None,
        "answers": {
            "noul": {
                "type": noul_answer.type,
                "probability": noul_answer.noul,
            },
            "choice": {
                "type": choice_answer.type,
                "choice": choice_answer.choice,
                "confidence": choice_answer.confidence,
                "probabilities": choice_answer.probabilities,
            },
            "score": {
                "type": score_answer.type,
                "score": score_answer.score,
                "confidence": score_answer.confidence,
                "probabilities": score_answer.probabilities,
                "legend": score_answer.legend,
            },
        },
    }


def _config_for(model: DecisionModel) -> SystemOneConfig:
    return SystemOneConfig(
        base_url=model.base_url,
        api_key=model.api_key,
        model=model.model,
        timeout=DECISION_TIMEOUT,
        total_timeout=DECISION_TOTAL_TIMEOUT,
        max_retries=DECISION_MAX_RETRIES,
    )


def _debug_record(
    kind: str,
    request: Any,
    response: Any = None,
    *,
    started: float,
    store: RuleStore | None = None,
    meta: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """在写入时读取已保存开关，关闭后在途请求也不再落盘。"""
    if store is None or not store.settings.decision_debug_enabled:
        return
    write_debug_record(
        kind,
        request,
        response,
        directory=store.data_dir / DECISION_DEBUG_DIRNAME,
        enabled=True,
        meta=meta,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        error=error,
    )


def _custom_meta(
    model: DecisionModel, client: SystemOneClient, trace: CallTrace
) -> dict[str, Any]:
    """自定义决策模型的端点与响应状态，便于对照落盘文件。"""
    return {
        "model_id": model.id,
        "model_name": model.name,
        "endpoint": client.endpoint,
        "status_code": trace.status_code,
    }


def _merge_meta(
    meta: Mapping[str, Any], debug_meta: Mapping[str, Any] | None
) -> dict[str, Any]:
    """在定位字段之上叠加调用方给的调试字段；同键以调用方为准。"""
    return {**meta, **(debug_meta or {})}


class DecisionRouter:
    """按规则的配置选择决策模型，并统一返回 ``{问题键: 概率}``。"""

    def __init__(
        self,
        context: Context,
        store: RuleStore,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._context = context
        self._store = store
        self._transport = transport
        """测试注入用；生产环境交给 httpx 自己建连接池。"""
        self._clients: dict[str, tuple[tuple, SystemOneClient]] = {}
        """自定义模型 ID → (配置指纹, 客户端)，配置一变就重建。"""
        self._clients_lock = asyncio.Lock()
        """保护 ``_clients`` 的检查-替换：配置变更时的并发判断不能各建一个客户端。"""

    async def probabilities(
        self,
        provider_id: str | None,
        prompts: Mapping[str, NoulPrompt],
        state: State | None = None,
        *,
        debug_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        """一次判断多个 noul 问题。

        Args:
            provider_id: 规则里的决策模型引用，``None`` / 空串表示核心默认。
            prompts: 问题键 → 判断内容与判定标准。
            state: 原样发给决策模型的上下文。
            debug_meta: 额外写进调试日志 ``meta`` 的字段（不含在请求体里），
                例如本次召回记忆的查询词与结果。

        Returns:
            问题键 → 「陈述为真」的概率。

        Raises:
            DecisionError: 模型未配置、调用失败或响应不符合契约。
        """
        if not prompts:
            raise DecisionError("没有需要判断的问题")
        ref = (provider_id or "").strip()
        if ref.startswith(CUSTOM_DECISION_PREFIX):
            model_id = ref[len(CUSTOM_DECISION_PREFIX) :]
            return await self._ask_custom(model_id, prompts, state, debug_meta)
        return await self._ask_official(ref or None, prompts, state, debug_meta)

    async def reload(self, model_ids: set[str] | None = None) -> None:
        """受影响判断退出后，仅释放已变更或删除的客户端。

        调用方必须已取消并等待相关判断收尾（见 ``SpeechExecution.reconcile``）：
        这里会关闭客户端，在途请求会以「客户端被关闭」失败。
        """
        current = {
            model.id: (model.base_url, model.api_key, model.model)
            for model in self._store.settings.decision_models
        }
        async with self._clients_lock:
            clients = []
            for model_id, (fingerprint, client) in list(self._clients.items()):
                if model_ids is not None and model_id not in model_ids:
                    continue
                if current.get(model_id) != fingerprint:
                    self._clients.pop(model_id)
                    clients.append(client)
        for client in clients:
            await client.close()

    async def close(self) -> None:
        async with self._clients_lock:
            clients = [client for _, client in self._clients.values()]
            self._clients.clear()
        for client in clients:
            await client.close()

    # ------------------------------------------------------------------ 内部
    async def _ask_official(
        self,
        provider_id: str | None,
        prompts: Mapping[str, NoulPrompt],
        state: State | None,
        debug_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        service = getattr(self._context, "decision", None)
        if service is None:
            raise DecisionError("核心未提供决策模型服务")
        questions = {
            head: DecisionBooleanQuestion(prompt.instructions)
            for head, prompt in prompts.items()
        }
        # 核心按 provider 组装 HTTP body，调试日志记录等价内容。
        request = {
            "state": state,
            "questions": {
                head: {"type": "noul", "instructions": question.instructions}
                for head, question in questions.items()
            },
        }
        meta = _merge_meta({"provider_id": provider_id or "(核心默认)"}, debug_meta)
        started = time.perf_counter()
        try:
            answers = await service.ask(
                questions,
                state=state,
                provider_id=provider_id,
            )
        except DecisionError as exc:
            _debug_record(
                "decision-official",
                request,
                store=self._store,
                started=started,
                meta=meta,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        except Exception as exc:  # noqa: BLE001
            _debug_record(
                "decision-official",
                request,
                store=self._store,
                started=started,
                meta=meta,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise DecisionError(f"判断提供商调用失败：{exc}") from exc
        _debug_record(
            "decision-official",
            request,
            answers,
            store=self._store,
            started=started,
            meta=meta,
        )

        result: dict[str, float] = {}
        for head in prompts:
            answer = answers.get(head) if isinstance(answers, Mapping) else None
            probability = getattr(answer, "probability", None)
            if isinstance(probability, bool) or not isinstance(
                probability, int | float
            ):
                raise DecisionError(f"判断提供商没有返回 {head} 的判定")
            result[head] = float(probability)
        return result

    async def _ask_custom(
        self,
        model_id: str,
        prompts: Mapping[str, NoulPrompt],
        state: State | None,
        debug_meta: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        model = self._store.find_decision_model(model_id)
        if model is None:
            raise DecisionError(f"自定义决策模型不存在：{model_id}")
        client = await self._client_for(model)
        questions = {
            head: noul(
                prompt.instructions,
                when_true=prompt.when_true or NOUL_TRUE_FALLBACK,
                when_false=prompt.when_false or NOUL_FALSE_FALLBACK,
            )
            for head, prompt in prompts.items()
        }
        # 每次调用各持一份 trace：客户端是共享的长连接池，落到它身上的
        # 「最近一次」状态在并发下会把别人的请求/响应记进本次日志。
        trace = CallTrace()
        started = time.perf_counter()
        try:
            answers = await client.ask(questions, state=state, trace=trace)
        except Exception as exc:  # noqa: BLE001
            # 含 SDK 的协议错误、网络错误，以及客户端被并发关闭等意外。
            _debug_record(
                "decision-custom",
                trace.request_body,
                trace.response_json(),
                store=self._store,
                started=started,
                meta=_merge_meta(_custom_meta(model, client, trace), debug_meta),
                error=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, SystemOneContentRejected):
                raise DecisionContentRejected(
                    f"自定义决策模型「{model.name}」内容审核未通过：{exc}"
                ) from exc
            raise DecisionError(
                f"自定义决策模型「{model.name}」判定失败：{exc}"
            ) from exc
        _debug_record(
            "decision-custom",
            trace.request_body,
            trace.response_json(answers),
            store=self._store,
            started=started,
            meta=_merge_meta(_custom_meta(model, client, trace), debug_meta),
        )
        if trace.usage is not None:
            logger.debug(
                "[Ambient] 决策模型「%s」用量：%s",
                model.name,
                trace.usage,
            )

        result: dict[str, float] = {}
        for head in prompts:
            answer = answers.get(head)
            if not isinstance(answer, NoulAnswer):
                raise DecisionError(
                    f"自定义决策模型「{model.name}」没有返回 {head} 的判定"
                )
            result[head] = answer.noul
        return result

    async def _client_for(self, model: DecisionModel) -> SystemOneClient:
        """按配置指纹取客户端，并在锁内完成「检查-替换」。

        临界区里没有 ``await``：两个判断同时发现配置过期时，不会各建一个客户端
        而漏掉前一个的 ``close()``。旧客户端在锁外关闭，不拖住其它模型的判断。
        """
        fingerprint = (
            model.base_url,
            model.api_key,
            model.model,
        )
        async with self._clients_lock:
            cached = self._clients.get(model.id)
            if cached is not None and cached[0] == fingerprint:
                return cached[1]
            client = SystemOneClient(
                _config_for(model),
                transport=self._transport,
            )
            self._clients[model.id] = (fingerprint, client)
        if cached is not None:
            await cached[1].close()
        return client
