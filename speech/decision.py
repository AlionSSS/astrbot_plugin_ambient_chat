"""决策模型分派：官方判断提供商与「更多设置」里的自定义 SystemOne 配置。

规则里的 ``intent.decision_provider_id`` 有三种取值（见 ``docs/adr/0017``）：

- 空串：交给核心的默认判断提供商；
- ``custom:<id>``：用本插件的 SystemOne SDK 直连「更多设置」里的自定义端点，
  这类端点（如书生端砚）要求 ``noul`` 必须带 ``criteria.true`` / ``criteria.false``；
- 其他：按 provider id 交给核心。

任一分支失败都转成 :class:`DecisionError`，由调用方 fail-closed（ADR-0005）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.provider import DecisionBooleanQuestion, DecisionError

from ..rules.models import CUSTOM_DECISION_PREFIX, DecisionModel
from ..rules.store import RuleStore
from .debug_log import write_debug_record
from .systemone_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    State,
    SystemOneClient,
    SystemOneConfig,
    choice,
    noul,
    score,
)

if TYPE_CHECKING:
    import httpx

    from astrbot.api.star import Context

DECISION_TIMEOUT = 10.0
"""自定义决策模型单次 HTTP 尝试的超时秒数。"""

DECISION_TOTAL_TIMEOUT = 15.0
"""整次判断（含重试预算）的超时秒数：主动发言宁可放弃，也不拖住调度。"""

DECISION_MAX_RETRIES = 0
"""默认不重试：一次判断失败就放弃本次发言，避免连续失败时排队等待。"""

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
        try:
            answers = await client.ask(questions, state=PROBE_STATE)
        except Exception as exc:
            _debug_record(
                "decision-probe",
                client.last_request_body,
                _raw_response_body(client),
                started=started,
                store=store,
                meta=_custom_meta(model, client),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _debug_record(
            "decision-probe",
            client.last_request_body,
            _raw_response_body(client),
            started=started,
            store=store,
            meta=_custom_meta(model, client),
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
        "model": client.last_model or model.model,
        "elapsed_ms": elapsed_ms,
        "usage": (
            client.last_usage.model_dump() if client.last_usage is not None else None
        ),
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
        directory=store.data_dir / "decision_debug",
        enabled=True,
        meta=meta,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        error=error,
    )


def _custom_meta(model: DecisionModel, client: SystemOneClient) -> dict[str, Any]:
    """自定义决策模型的端点与响应状态，便于对照落盘文件。"""
    return {
        "model_id": model.id,
        "model_name": model.name,
        "endpoint": client.endpoint,
        "status_code": client.last_status_code,
    }


def _raw_response_body(client: SystemOneClient, answers: Any = None) -> Any:
    """优先返回服务端原始响应 body；解析失败时退回已解析的答案。"""
    raw = client.last_response_text
    if raw:
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return answers


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

    async def probabilities(
        self,
        provider_id: str | None,
        prompts: Mapping[str, NoulPrompt],
        state: State | None = None,
    ) -> dict[str, float]:
        """一次判断多个 noul 问题。

        Args:
            provider_id: 规则里的决策模型引用，``None`` / 空串表示核心默认。
            prompts: 问题键 → 判断内容与判定标准。
            state: 原样发给决策模型的上下文。

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
            return await self._ask_custom(model_id, prompts, state)
        return await self._ask_official(ref or None, prompts, state)

    async def reload(self, model_ids: set[str] | None = None) -> None:
        """受影响判断退出后，仅释放已变更或删除的客户端。"""
        current = {
            model.id: (model.base_url, model.api_key, model.model)
            for model in self._store.settings.decision_models
        }
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
        meta = {"provider_id": provider_id or "(核心默认)"}
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
        started = time.perf_counter()
        try:
            answers = await client.ask(questions, state=state)
        except Exception as exc:  # noqa: BLE001
            # 含 SDK 的协议错误、网络错误，以及客户端被并发关闭等意外。
            _debug_record(
                "decision-custom",
                client.last_request_body,
                _raw_response_body(client),
                store=self._store,
                started=started,
                meta=_custom_meta(model, client),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise DecisionError(
                f"自定义决策模型「{model.name}」判定失败：{exc}"
            ) from exc
        _debug_record(
            "decision-custom",
            client.last_request_body,
            _raw_response_body(client, answers),
            store=self._store,
            started=started,
            meta=_custom_meta(model, client),
        )
        if client.last_usage is not None:
            logger.debug(
                "[Ambient] 决策模型「%s」用量：%s",
                model.name,
                client.last_usage,
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
        fingerprint = (
            model.base_url,
            model.api_key,
            model.model,
        )
        cached = self._clients.get(model.id)
        if cached is not None:
            if cached[0] == fingerprint:
                return cached[1]
            await cached[1].close()
        client = SystemOneClient(
            _config_for(model),
            transport=self._transport,
        )
        self._clients[model.id] = (fingerprint, client)
        return client
