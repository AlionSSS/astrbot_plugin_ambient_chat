"""SystemOne 决策模型的最小单文件 SDK（不依赖 AstrBot）。

为什么要自己实现：
    AstrBot 核心的 Decision Provider 在序列化 ``noul`` 问题时只发送
    ``type`` 与 ``instructions``，不发送 ``criteria``。而部分 SystemOne
    兼容平台（如书生端砚）要求 ``noul`` 必须携带
    ``criteria: {"true": ..., "false": ...}``，否则请求直接被拒。

    本 SDK 按官方协议 https://docs.typesafe.ai/api/system-one 实现，
    支持 ``noul`` / ``choice`` / ``score`` 三种问题，``noul`` 可按需带上
    ``criteria``；请求模型与响应模型都由 pydantic 约束。

用法::

    from systemone_sdk import SystemOneClient, SystemOneConfig

    config = SystemOneConfig(
        base_url="https://api.typesafe.ai/v1",
        api_key="sk-...",
        model="jev-latest",
    )
    async with SystemOneClient(config) as client:
        ok = await client.yes_no(
            "这条消息值得主动接一句吗？",
            state={"conversation": "..."},
            threshold=0.7,
            default=False,
        )

    # noul 的 criteria 完全由调用方决定：官方协议里可以省略；
    # 对要求必填的平台（如书生端砚）在调用处显式传入 true/false。
    question = noul(
        "这条消息值得主动接一句吗？",
        when_true="有话可说，且此刻接话不突兀",
        when_false="无话可说，或接话会显得突兀",
    )
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneClient",
    "SystemOneConfig",
    "SystemOneConfigError",
    "SystemOneContentRejected",
    "SystemOneError",
    "SystemOneRequestError",
    "SystemOneResponseError",
    "SystemOneTimeout",
    "SystemOneUsage",
    "choice",
    "noul",
    "resolve_endpoint",
    "score",
]

Criterion = str | dict[str, Any] | list[Any]
"""``instructions`` / ``criteria`` 里允许的值：字符串或任意 JSON 结构。"""

State = str | list[Any] | dict[str, Any]
"""``state`` 允许的值，原样发给决策模型。"""

_SUM_TOLERANCE = 1e-3
"""概率之和的容差，部分实现会有浮点误差。"""

_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 4.0
_MAX_SUMMARY_CHARS = 300


# --------------------------------------------------------------------- 异常
class SystemOneError(RuntimeError):
    """SDK 抛出的所有异常的基类。"""


class SystemOneConfigError(SystemOneError, ValueError):
    """请求构造不合法：问题为空、判定标准不足、配置取值越界等。"""


class SystemOneRequestError(SystemOneError):
    """网络层失败或服务端返回错误状态码。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class SystemOneTimeout(SystemOneRequestError):
    """整次请求（含重试）超出 ``total_timeout``。"""


class SystemOneResponseError(SystemOneError):
    """响应不符合 SystemOne 协议：非 JSON、结构不对、概率分布自相矛盾等。"""


class SystemOneContentRejected(SystemOneResponseError):
    """平台内容审核拒绝了请求（如 ``content_check_rejected``）。

    与普通协议错误区分开：请求格式、字段、鉴权都没问题，是**内容**被判为
    敏感，用户该改的是会话内容或人格提示，不是端点配置。
    """

    def __init__(self, message: str, *, code: str = "", detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail
        """平台返回的原始说明，页面与日志可直接展示。"""


# --------------------------------------------------------------------- 请求模型
class NoulCriteria(BaseModel):
    """``noul`` 问题的判定标准。

    官方协议中该字段可选；两个键都不给时不会序列化 ``criteria``。
    """

    model_config = ConfigDict(extra="forbid")

    true: Criterion | None = None
    false: Criterion | None = None


class NoulQuestion(BaseModel):
    """是 / 否（真 / 假）判断。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: Criterion | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """在 ``criteria`` 的多选一中挑一个。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: Criterion | None = None
    criteria: dict[str, Criterion] = Field(min_length=1, max_length=255)


class ScoreQuestion(BaseModel):
    """按 ``criteria`` 的等级打分，等级从 0 开始。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: Criterion | None = None
    criteria: list[Criterion] = Field(min_length=1, max_length=10)


Question = NoulQuestion | ChoiceQuestion | ScoreQuestion

_QUESTION_ADAPTER = TypeAdapter(
    Annotated[Question, Field(discriminator="type")],
)


def noul(
    instructions: str,
    *,
    when_true: Criterion | None = None,
    when_false: Criterion | None = None,
) -> NoulQuestion:
    """构造 ``noul`` 问题；``when_true`` / ``when_false`` 对应协议里的
    ``criteria.true`` / ``criteria.false``。

    官方协议中 ``criteria`` 是**可选**的：两者都不传就不会发送该字段。
    SDK 不会补默认文案——需要必填的平台由调用方显式传入。
    """
    criteria = (
        NoulCriteria(true=when_true, false=when_false)
        if when_true is not None or when_false is not None
        else None
    )
    return NoulQuestion(instructions=instructions, criteria=criteria)


def choice(
    instructions: str,
    criteria: Mapping[str, str] | Sequence[str],
) -> ChoiceQuestion:
    """构造 ``choice`` 问题。

    ``criteria`` 可以是「选项 ID → 选项描述」的映射，也可以是选项列表
    （此时描述与 ID 相同）。
    """
    normalized = _normalize_choice_criteria(criteria)
    return ChoiceQuestion(instructions=instructions, criteria=normalized)


def score(instructions: str, criteria: Sequence[str]) -> ScoreQuestion:
    """构造 ``score`` 问题；``criteria`` 由低到高排列，等级从 0 开始。"""
    levels = [str(level) for level in criteria]
    if not levels:
        raise SystemOneConfigError("score 的 criteria 不能为空")
    return ScoreQuestion(instructions=instructions, criteria=levels)


# --------------------------------------------------------------------- 响应模型
class NoulAnswer(BaseModel):
    """``noul`` 的回答：陈述为真的概率。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["noul"] | None = None
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    """``choice`` 的回答：选中项、置信度与完整概率分布。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"] | None = None
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    """``score`` 的回答：期望分、置信度、等级说明与概率分布。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["score"] | None = None
    score: float
    confidence: float = Field(ge=0.0, le=1.0)
    legend: dict[str, Criterion]
    probabilities: dict[str, float]


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class SystemOneUsage(BaseModel):
    """token 用量。"""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0


class _SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = ""
    answers: dict[str, Any]
    usage: SystemOneUsage | None = None


# --------------------------------------------------------------------- 配置
@dataclass(frozen=True, slots=True)
class SystemOneConfig:
    """客户端配置。

    Attributes:
        base_url: 服务的 Base URL，如 ``https://host/v1``；SDK 会自动补
            ``/systemone``（已带则不再补），规则见 :func:`resolve_endpoint`。
        api_key: Bearer 令牌，留空则不发送 ``Authorization``。
        model: 模型名或别名，如 ``jev-latest``。
        endpoint: 完整端点，非空时忽略 ``base_url`` 的补全逻辑。
        timeout: 单次 HTTP 尝试的超时秒数。
        total_timeout: 含重试在内的整次请求预算秒数。
        max_concurrency: 并发请求上限。
        max_retries: 网络错误 / 5xx / 429 的重试次数。
        proxy: 代理地址。
        headers: 追加的请求头，可覆盖内置头。
        strict: 为真时校验概率之和、键集合与选中项是否为最大概率，
            为假时只校验类型与取值范围。
        danger_zone_images: 发送 ``TypeSafe-Danger-Zone-Images: true``。
        danger_zone_long_context: 发送 ``TypeSafe-Danger-Zone-Long-Context: true``。
    """

    base_url: str = ""
    api_key: str = ""
    model: str = "jev-latest"
    endpoint: str = ""
    timeout: float = 10.0
    total_timeout: float = 20.0
    max_concurrency: int = 4
    max_retries: int = 1
    proxy: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    strict: bool = True
    danger_zone_images: bool = False
    danger_zone_long_context: bool = False

    def __post_init__(self) -> None:
        if not (self.base_url or "").strip() and not (self.endpoint or "").strip():
            raise SystemOneConfigError("base_url 与 endpoint 不能同时为空")
        if not (self.model or "").strip():
            raise SystemOneConfigError("model 不能为空")
        if self.timeout <= 0:
            raise SystemOneConfigError("timeout 必须大于 0")
        if self.total_timeout <= 0:
            raise SystemOneConfigError("total_timeout 必须大于 0")
        if self.max_concurrency <= 0:
            raise SystemOneConfigError("max_concurrency 必须大于 0")
        if self.max_retries < 0:
            raise SystemOneConfigError("max_retries 不能小于 0")


def resolve_endpoint(base_url: str) -> str:
    """把配置里的 Base URL 补成完整的 SystemOne 端点。

    规则：先去掉结尾的 ``/``，已以 ``systemone`` 结尾就原样使用，
    否则在末尾补 ``/systemone``。例如：

    - ``https://discovery-api.intern-ai.org.cn/v1`` → ``.../v1/systemone``
    - ``https://discovery-api.intern-ai.org.cn/v1/systemone`` → 原样
    - ``https://api.typesafe.ai/v1`` → ``.../v1/systemone``

    需要完全自定义的路径时，用 :attr:`SystemOneConfig.endpoint` 显式指定。
    """
    raw = (base_url or "").strip()
    if not raw:
        raise SystemOneConfigError("base_url 不能为空")
    url = raw.rstrip("/")
    if url.endswith("/systemone"):
        return url
    return f"{url}/systemone"


# --------------------------------------------------------------------- 客户端
class SystemOneClient:
    """SystemOne HTTP 客户端。

    一个实例持有一个 ``httpx.AsyncClient``；用 ``async with`` 或显式
    ``await client.close()`` 释放。
    """

    def __init__(
        self,
        config: SystemOneConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.last_usage: SystemOneUsage | None = None
        """最近一次成功响应的 token 用量。"""
        self.last_model: str = ""
        """最近一次成功响应实际使用的模型名（可能被服务端解析别名）。"""
        self.last_request_body: dict[str, Any] | None = None
        """调试用：最近一次请求发送的 JSON body，原样保留。"""
        self.last_response_text: str | None = None
        """调试用：最近一次响应（含失败尝试）的原始文本 body。"""
        self.last_status_code: int | None = None
        """调试用：最近一次响应的 HTTP 状态码；网络失败时为 ``None``。"""
        self._endpoint = (config.endpoint or "").strip() or resolve_endpoint(
            config.base_url
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout),
            proxy=config.proxy or None,
            transport=transport,
            headers=self._headers(),
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.danger_zone_images:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        if self.config.danger_zone_long_context:
            headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        headers.update({str(k): str(v) for k, v in self.config.headers.items()})
        return headers

    @property
    def endpoint(self) -> str:
        """实际请求的完整端点，便于排错。"""
        return self._endpoint

    async def __aenter__(self) -> SystemOneClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ 调用
    async def ask(
        self,
        questions: Mapping[str, Question | Mapping[str, Any]],
        state: State | None = None,
        *,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Answer]:
        """一次请求回答多个问题，按键名返回带类型的答案。

        Args:
            questions: 问题键 → 问题对象（也接受等价的原始 dict）。
            state: 所有问题共同参照的内容，``None`` 时发送 ``{}``。
            model: 覆盖配置里的模型名。
            max_retries: 覆盖配置里的重试次数。

        Raises:
            SystemOneConfigError: 问题定义不合法。
            SystemOneRequestError: 网络失败或服务端返回错误状态码。
            SystemOneTimeout: 整次请求超出 ``total_timeout``。
            SystemOneResponseError: 响应不符合协议契约。
        """
        prepared = self._prepare(questions)
        payload = self._payload(prepared, state=state, model=model)
        response = await self._request(payload, max_retries=max_retries)
        return self._build_answers(prepared, self._parse_response(response))

    def build_payload(
        self,
        questions: Mapping[str, Question | Mapping[str, Any]],
        state: State | None = None,
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """构造请求体，便于日志与离线排查（不做任何网络请求）。"""
        return self._payload(self._prepare(questions), state=state, model=model)

    async def yes_no(
        self,
        question: str,
        *,
        state: State | None = None,
        threshold: float = 0.5,
        when_true: Criterion | None = None,
        when_false: Criterion | None = None,
        default: bool | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> bool:
        """布尔判断：概率达到 ``threshold`` 即为真。

        Args:
            question: 待判断的陈述或问题。
            state: 参照内容。
            threshold: 判定阈值，0~1。
            when_true: ``criteria.true``，说明什么情况算「是」。
            when_false: ``criteria.false``，说明什么情况算「否」。
            default: 调用失败时的返回值；``None`` 表示把异常抛给调用方。
        """
        if not question.strip():
            raise SystemOneConfigError("question 不能为空")
        if not 0.0 <= threshold <= 1.0:
            raise SystemOneConfigError("threshold 需在 0~1 之间")
        try:
            answers = await self.ask(
                {
                    "answer": noul(
                        question,
                        when_true=when_true,
                        when_false=when_false,
                    )
                },
                state,
                model=model,
                max_retries=max_retries,
            )
        except (SystemOneRequestError, SystemOneResponseError):
            if default is None:
                raise
            return default
        answer = answers["answer"]
        if not isinstance(answer, NoulAnswer):
            raise SystemOneResponseError("决策服务没有返回 noul 答案")
        return answer.noul >= threshold

    async def choose(
        self,
        question: str,
        criteria: Mapping[str, str] | Sequence[str],
        *,
        state: State | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> ChoiceAnswer:
        """多选一。返回的 ``choice`` / ``probabilities`` 已换回调用方的选项值。"""
        normalized = _normalize_choice_criteria(criteria)
        option_ids = {f"o{index}": value for index, value in enumerate(normalized, 1)}
        answers = await self.ask(
            {
                "answer": ChoiceQuestion(
                    instructions=question,
                    criteria={
                        option_id: normalized[value]
                        for option_id, value in option_ids.items()
                    },
                )
            },
            state,
            model=model,
            max_retries=max_retries,
        )
        answer = answers["answer"]
        if not isinstance(answer, ChoiceAnswer):
            raise SystemOneResponseError("决策服务没有返回 choice 答案")
        if answer.choice not in option_ids:
            raise SystemOneResponseError(f"决策服务返回了未知选项：{answer.choice!r}")
        return ChoiceAnswer(
            type=answer.type,
            choice=option_ids[answer.choice],
            confidence=answer.confidence,
            probabilities={
                option_ids[option_id]: probability
                for option_id, probability in answer.probabilities.items()
                if option_id in option_ids
            },
        )

    async def score(
        self,
        question: str,
        criteria: Sequence[str],
        *,
        state: State | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> ScoreAnswer:
        """按等级打分，返回期望分（可为小数）、置信度与概率分布。"""
        answer = await self.ask(
            {"answer": score(question, criteria)},
            state,
            model=model,
            max_retries=max_retries,
        )
        result = answer["answer"]
        if not isinstance(result, ScoreAnswer):
            raise SystemOneResponseError("决策服务没有返回 score 答案")
        return result

    # ------------------------------------------------------------------ 内部
    def _prepare(
        self,
        questions: Mapping[str, Question | Mapping[str, Any]],
    ) -> dict[str, Question]:
        if not questions:
            raise SystemOneConfigError("questions 不能为空")
        prepared: dict[str, Question] = {}
        for head, question in questions.items():
            name = str(head).strip()
            if not name:
                raise SystemOneConfigError("问题键名不能为空")
            prepared[name] = _coerce_question(question)
        return prepared

    def _payload(
        self,
        prepared: Mapping[str, Question],
        *,
        state: State | None,
        model: str | None,
    ) -> dict[str, Any]:
        name = (model or self.config.model).strip()
        if not name:
            raise SystemOneConfigError("model 不能为空")
        return {
            "model": name,
            "state": {} if state is None else state,
            "questions": {
                head: question.model_dump(exclude_none=True)
                for head, question in prepared.items()
            },
        }

    async def _request(
        self,
        payload: dict[str, Any],
        *,
        max_retries: int | None,
    ) -> httpx.Response:
        retries = self.config.max_retries if max_retries is None else max_retries
        try:
            async with self._semaphore:
                return await asyncio.wait_for(
                    self._post_with_retry(payload, retries),
                    timeout=self.config.total_timeout,
                )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise SystemOneTimeout(
                f"SystemOne 请求超过 {self.config.total_timeout:g}s"
            ) from exc

    async def _post_with_retry(
        self,
        payload: dict[str, Any],
        retries: int,
    ) -> httpx.Response:
        attempt = 0
        self.last_request_body = payload
        self.last_response_text = None
        self.last_status_code = None
        while True:
            attempt += 1
            try:
                response = await self._client.post(self._endpoint, json=payload)
            except httpx.RequestError as exc:
                if attempt <= retries:
                    await _backoff(attempt)
                    continue
                raise SystemOneRequestError(
                    f"SystemOne 请求失败（{type(exc).__name__}）：{exc}"
                ) from exc
            self.last_status_code = response.status_code
            self.last_response_text = response.text
            if response.status_code < 400:
                return response
            detail = _error_detail(response)
            rejected = _content_check_rejection(response.text)
            if rejected is not None:
                # 内容审核不会因为重试而通过，直接给出可读原因。
                raise rejected
            if response.status_code in _RETRY_STATUS and attempt <= retries:
                await _backoff(attempt)
                continue
            raise SystemOneRequestError(
                f"SystemOne 返回 HTTP {response.status_code}：{detail}",
                status_code=response.status_code,
                detail=detail,
            )

    def _parse_response(self, response: httpx.Response) -> _SystemOneResponse:
        try:
            data = response.json()
        except ValueError as exc:
            # 部分平台（如书生端砚）用 200 + ``data: {"error": ...}`` 表示内容被拦，
            # 这不是协议错误，必须让用户看到「审核未通过」而不是「不是合法 JSON」。
            rejected = _content_check_rejection(response.text)
            if rejected is not None:
                raise rejected from exc
            raise SystemOneResponseError(
                f"SystemOne 响应不是合法 JSON：{_truncate(response.text)}"
            ) from exc
        if not isinstance(data, Mapping):
            raise SystemOneResponseError(
                f"SystemOne 响应根节点必须是对象：{_summarize(data)}"
            )
        rejected = _content_check_from_payload(data)
        if rejected is not None:
            raise rejected
        code, message = _error_envelope(data)
        if code or message:
            raise SystemOneResponseError(
                f"SystemOne 返回错误（{code or 'unknown'}）：{_truncate(message)}"
            )
        try:
            parsed = _SystemOneResponse.model_validate(data)
        except ValidationError as exc:
            raise SystemOneResponseError(
                f"SystemOne 响应结构不符合协议：{_validation_message(exc)}"
            ) from exc
        self.last_model = parsed.model
        self.last_usage = parsed.usage
        return parsed

    def _build_answers(
        self,
        prepared: Mapping[str, Question],
        parsed: _SystemOneResponse,
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        for head, question in prepared.items():
            raw = parsed.answers.get(head)
            if not isinstance(raw, Mapping):
                raise SystemOneResponseError(
                    f"缺少问题 {head} 的答案：{_summarize(parsed.answers)}"
                )
            answers[head] = _parse_answer(
                head,
                question,
                raw,
                strict=self.config.strict,
            )
        return answers


# --------------------------------------------------------------------- 工具函数
def _normalize_choice_criteria(
    criteria: Mapping[str, str] | Sequence[str],
) -> dict[str, str]:
    if isinstance(criteria, Mapping):
        normalized = {str(key): str(value) for key, value in criteria.items()}
    elif isinstance(criteria, Sequence) and not isinstance(criteria, str | bytes):
        normalized = {str(value): str(value) for value in criteria}
    else:
        raise SystemOneConfigError("choice 的 criteria 必须是映射或字符串序列")
    if not normalized:
        raise SystemOneConfigError("choice 的 criteria 不能为空")
    if any(not key.strip() for key in normalized):
        raise SystemOneConfigError("choice 的选项值不能为空")
    return normalized


def _coerce_question(question: Question | Mapping[str, Any]) -> Question:
    if isinstance(question, NoulQuestion | ChoiceQuestion | ScoreQuestion):
        return question
    try:
        return _QUESTION_ADAPTER.validate_python(question)
    except ValidationError as exc:
        raise SystemOneConfigError(
            f"问题定义不合法：{_validation_message(exc)}"
        ) from exc


def _parse_answer(
    head: str,
    question: Question,
    raw: Mapping[str, Any],
    *,
    strict: bool,
) -> Answer:
    raw_type = raw.get("type")
    if raw_type is not None and raw_type != question.type:
        raise SystemOneResponseError(
            f"问题 {head} 的答案类型是 {raw_type!r}，与问题类型 "
            f"{question.type!r} 不一致"
        )
    if isinstance(question, NoulQuestion):
        return _parse_noul(head, raw)
    if isinstance(question, ChoiceQuestion):
        return _parse_choice(head, question, raw, strict=strict)
    return _parse_score(head, question, raw, strict=strict)


def _parse_noul(head: str, raw: Mapping[str, Any]) -> NoulAnswer:
    value = raw.get("noul", raw.get("probability"))
    try:
        return NoulAnswer(type=raw.get("type"), noul=value)
    except ValidationError as exc:
        raise SystemOneResponseError(
            f"问题 {head} 的 noul 概率不合法：{_summarize(raw)}"
        ) from exc


def _parse_choice(
    head: str,
    question: ChoiceQuestion,
    raw: Mapping[str, Any],
    *,
    strict: bool,
) -> ChoiceAnswer:
    selected = raw.get("choice")
    if not isinstance(selected, str) or selected not in question.criteria:
        raise SystemOneResponseError(
            f"问题 {head} 的 choice={selected!r} 不在候选中"
            f"（{_summarize(list(question.criteria))}）"
        )
    probabilities = _number_mapping(head, "probabilities", raw.get("probabilities"))
    if strict:
        if set(probabilities) != set(question.criteria):
            raise SystemOneResponseError(
                f"问题 {head} 的概率键与候选不一致："
                f"{_summarize(sorted(probabilities))} != "
                f"{_summarize(sorted(question.criteria))}"
            )
        _ensure_distribution(head, probabilities)
        _ensure_argmax(head, selected, probabilities)
    confidence = raw.get("confidence")
    if confidence is None:
        confidence = probabilities.get(selected, 0.0)
    try:
        return ChoiceAnswer(
            type=raw.get("type"),
            choice=selected,
            confidence=confidence,
            probabilities=probabilities,
        )
    except ValidationError as exc:
        raise SystemOneResponseError(
            f"问题 {head} 的 choice 答案不合法：{_summarize(raw)}"
        ) from exc


def _parse_score(
    head: str,
    question: ScoreQuestion,
    raw: Mapping[str, Any],
    *,
    strict: bool,
) -> ScoreAnswer:
    probabilities = _number_mapping(head, "probabilities", raw.get("probabilities"))
    legend_raw = raw.get("legend")
    if not isinstance(legend_raw, Mapping):
        raise SystemOneResponseError(f"问题 {head} 缺少 legend：{_summarize(raw)}")
    legend = {str(key): value for key, value in legend_raw.items()}
    expected_keys = {str(index) for index in range(len(question.criteria))}
    if strict and set(legend) != expected_keys:
        raise SystemOneResponseError(
            f"问题 {head} 的 legend 等级与 criteria 不一致："
            f"{_summarize(sorted(legend))} != {_summarize(sorted(expected_keys))}"
        )
    if strict and set(probabilities) != set(legend):
        raise SystemOneResponseError(
            f"问题 {head} 的概率键与 legend 不一致："
            f"{_summarize(sorted(probabilities))} != {_summarize(sorted(legend))}"
        )
    if strict:
        _ensure_distribution(head, probabilities)
    value = raw.get("score")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemOneResponseError(
            f"问题 {head} 的 score 不是数字：{_summarize(raw)}"
        )
    upper = len(question.criteria) - 1
    if not math.isfinite(value) or not (
        -_SUM_TOLERANCE <= value <= upper + _SUM_TOLERANCE
    ):
        raise SystemOneResponseError(
            f"问题 {head} 的 score={value!r} 超出 0~{upper} 的范围"
        )
    confidence = raw.get("confidence")
    try:
        return ScoreAnswer(
            type=raw.get("type"),
            score=float(value),
            confidence=0.0 if confidence is None else confidence,
            legend=legend,
            probabilities=probabilities,
        )
    except ValidationError as exc:
        raise SystemOneResponseError(
            f"问题 {head} 的 score 答案不合法：{_summarize(raw)}"
        ) from exc


def _number_mapping(
    head: str,
    field_name: str,
    value: Any,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise SystemOneResponseError(
            f"问题 {head} 的 {field_name} 必须是对象：{_summarize(value)}"
        )
    result: dict[str, float] = {}
    for key, item in value.items():
        if (
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(item)
            or not 0.0 <= item <= 1.0
        ):
            raise SystemOneResponseError(
                f"问题 {head} 的 {field_name}[{key!r}]={item!r} 不是 0~1 的概率"
            )
        result[str(key)] = float(item)
    return result


def _ensure_distribution(head: str, probabilities: Mapping[str, float]) -> None:
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, abs_tol=_SUM_TOLERANCE):
        raise SystemOneResponseError(f"问题 {head} 的概率之和为 {total:.6f}，不等于 1")


def _ensure_argmax(
    head: str,
    selected: str,
    probabilities: Mapping[str, float],
) -> None:
    best = max(probabilities.values())
    if not math.isclose(probabilities[selected], best, abs_tol=_SUM_TOLERANCE):
        raise SystemOneResponseError(
            f"问题 {head} 的 choice={selected!r} 不是概率最高的选项"
        )


async def _backoff(attempt: int) -> None:  # pragma: no cover - 仅在重试时触发
    delay = min(_MAX_BACKOFF_SECONDS, 0.2 * (2 ** (attempt - 1)))
    await asyncio.sleep(delay)


_CONTENT_CHECK_TAG = "content_check"
"""平台内容审核错误的 code / type 片段（如 ``content_check_rejected``）。"""


def _error_envelope(payload: Any) -> tuple[str, str]:
    """取出「错误信封」里的 ``(code, message)``；不是错误响应时返回两个空串。

    兼容 ``{"error": {...}}``、``{"detail": {...}}``，以及 ``detail`` 里再包
    一层 ``error`` 的 OpenAI 风格响应。
    """
    if not isinstance(payload, Mapping):
        return "", ""
    for key in ("error", "detail"):
        item = payload.get(key)
        if not isinstance(item, Mapping):
            continue
        nested = item.get("error")
        error = nested if isinstance(nested, Mapping) else item
        code = str(error.get("code") or error.get("type") or "")
        message = str(error.get("message") or "")
        if code or message:
            return code, message
    return "", ""


def _error_payloads(text: str) -> list[Any]:
    """响应体可能是纯 JSON，也可能是 ``data: {...}`` 的流式帧，两种都解一遍。"""
    candidates = [(text or "").strip()]
    candidates.extend(
        line[len("data:") :].strip()
        for line in (text or "").splitlines()
        if line.startswith("data:")
    )
    payloads: list[Any] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payloads.append(json.loads(candidate))
        except ValueError:
            continue
    return payloads


def _content_check_rejection(text: str) -> SystemOneContentRejected | None:
    """响应文本是内容审核拒绝时返回专用异常，否则返回 ``None``。"""
    for payload in _error_payloads(text):
        rejected = _content_check_from_payload(payload)
        if rejected is not None:
            return rejected
    return None


def _content_check_from_payload(payload: Any) -> SystemOneContentRejected | None:
    code, message = _error_envelope(payload)
    if _CONTENT_CHECK_TAG not in code.lower():
        return None
    detail = _truncate(message) or _truncate(code)
    return SystemOneContentRejected(
        f"内容审核未通过（{code}）：{detail}",
        code=code,
        detail=detail,
    )


def _error_detail(response: httpx.Response) -> str:
    """从 400 / 422 响应里提取可读的错误说明。"""
    try:
        payload = response.json()
    except ValueError:
        return _truncate(response.text) or "(空响应体)"
    if not isinstance(payload, Mapping):
        return _summarize(payload)
    detail = payload.get("detail", payload)
    if isinstance(detail, str):
        return _truncate(detail)
    if isinstance(detail, Mapping):
        # OpenAI 风格信封：{"error": {"message": "...", "code": "..."}}
        error = detail.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return _truncate(error["message"])
        if isinstance(detail.get("message"), str):
            return _truncate(detail["message"])
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, Mapping):
                location = ".".join(str(part) for part in item.get("loc", ()))
                message = item.get("msg", "invalid")
                parts.append(f"{location}: {message}" if location else str(message))
            else:
                parts.append(str(item))
        if parts:
            return _truncate("; ".join(parts))
    return _summarize(detail)


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid")
        parts.append(f"{location}: {message}" if location else str(message))
    return _truncate("; ".join(parts)) or str(exc)


def _truncate(text: str, limit: int = _MAX_SUMMARY_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _summarize(value: Any, limit: int = _MAX_SUMMARY_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - 极端兜底
        text = repr(value)
    return _truncate(text, limit)
