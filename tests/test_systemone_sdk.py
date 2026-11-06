"""SystemOne SDK 的关键用例：noul criteria 兼容性与响应契约。"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from astrbot_plugin_ambient_chat.speech.systemone_sdk import (
    SystemOneClient,
    SystemOneConfig,
    SystemOneContentRejected,
    SystemOneRequestError,
    SystemOneResponseError,
    noul,
    resolve_endpoint,
)

CONTENT_REJECTED_BODY = (
    'data: {"error":{"message":"The request contains sensitive content. '
    'Please modify your input and try again.","type":"content_check_error",'
    '"code":"content_check_rejected"}}\n\n'
)
"""书生端砚的真实拒绝形态：HTTP 200 + 流式帧包裹的错误信封。"""


def _config(**overrides: Any) -> SystemOneConfig:
    params: dict[str, Any] = {
        "base_url": "https://example.test/v1",
        "api_key": "sk-test",
        "max_retries": 0,
    }
    params.update(overrides)
    return SystemOneConfig(**params)


def _client(handler: Any, **overrides: Any) -> SystemOneClient:
    return SystemOneClient(
        _config(**overrides),
        transport=httpx.MockTransport(handler),
    )


def _noul_body(probability: float = 0.9) -> dict[str, Any]:
    return {
        "model": "jev-latest",
        "answers": {"answer": {"type": "noul", "noul": probability}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


@pytest.mark.asyncio
async def test_noul_sends_criteria_true_false() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_noul_body())

    async with _client(handler) as client:
        result = await client.yes_no(
            "这条消息值得接一句吗？",
            state={"conversation": "..."},
            when_true="有话可说且不突兀",
            when_false="接话会显得突兀",
        )

    assert result is True
    assert captured["url"] == "https://example.test/v1/systemone"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    question = captured["payload"]["questions"]["answer"]
    assert question == {
        "type": "noul",
        "instructions": "这条消息值得接一句吗？",
        "criteria": {"true": "有话可说且不突兀", "false": "接话会显得突兀"},
    }


def test_resolve_endpoint_appends_systemone_once() -> None:
    assert (
        resolve_endpoint("https://discovery-api.intern-ai.org.cn/v1")
        == "https://discovery-api.intern-ai.org.cn/v1/systemone"
    )
    assert (
        resolve_endpoint("https://discovery-api.intern-ai.org.cn/v1/systemone")
        == "https://discovery-api.intern-ai.org.cn/v1/systemone"
    )
    # 结尾的 / 只做归一化，不改变「补 /systemone」的语义。
    assert resolve_endpoint("https://host/api/") == "https://host/api/systemone"
    assert (
        resolve_endpoint("https://host/api/systemone/") == "https://host/api/systemone"
    )


@pytest.mark.asyncio
async def test_noul_without_criteria_stays_optional() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_noul_body())

    async with _client(handler) as client:
        await client.ask({"answer": noul("是吗？")})

    assert "criteria" not in captured["payload"]["questions"]["answer"]


@pytest.mark.asyncio
async def test_choose_maps_option_ids_back_to_values() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["questions"]["answer"]["criteria"] == {
            "o1": "简短回复",
            "o2": "详细回复",
        }
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": "o2",
                        "confidence": 0.8,
                        "probabilities": {"o1": 0.2, "o2": 0.8},
                    }
                },
            },
        )

    async with _client(handler) as client:
        answer = await client.choose(
            "用哪种策略？", {"short": "简短回复", "long": "详细回复"}
        )

    assert answer.choice == "long"
    assert answer.type == "choice"
    assert answer.probabilities == {"short": 0.2, "long": 0.8}


@pytest.mark.asyncio
async def test_inconsistent_probabilities_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": "o1",
                        "confidence": 0.6,
                        "probabilities": {"o1": 0.6, "o2": 0.6},
                    }
                },
            },
        )

    async with _client(handler) as client:
        with pytest.raises(SystemOneResponseError, match="概率之和"):
            await client.choose("挑一个", ["a", "b"])


@pytest.mark.asyncio
async def test_http_error_detail_is_surfaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": "questions.answer.noul.criteria: Field required"},
        )

    async with _client(handler) as client:
        with pytest.raises(SystemOneRequestError) as info:
            await client.ask({"answer": noul("是吗？")})

    assert info.value.status_code == 400
    assert "criteria" in info.value.detail


@pytest.mark.asyncio
async def test_yes_no_uses_default_on_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    async with _client(handler) as client:
        assert await client.yes_no("是吗？", default=False) is False
        with pytest.raises(SystemOneRequestError):
            await client.yes_no("是吗？")


@pytest.mark.asyncio
async def test_raw_dict_question_with_noul_criteria_is_accepted() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {"q": {"type": "noul", "noul": 0.5}},
            },
        )

    async with _client(handler) as client:
        answers = await client.ask(
            {
                "q": {
                    "type": "noul",
                    "instructions": "是吗？",
                    "criteria": {"true": "是", "false": "否"},
                }
            }
        )

    assert captured["payload"]["questions"]["q"]["criteria"] == {
        "true": "是",
        "false": "否",
    }
    assert getattr(answers["q"], "noul") == 0.5
    assert answers["q"].type == "noul"  # 响应里的 type 原样保留


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 400])
async def test_content_check_rejection_is_not_a_protocol_error(status: int) -> None:
    """消息格式合法、只是内容被判敏感时，要说「审核未通过」而不是「不是合法 JSON」。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=CONTENT_REJECTED_BODY)

    async with _client(handler) as client:
        with pytest.raises(SystemOneContentRejected) as info:
            await client.ask({"answer": noul("是吗？")})

    assert info.value.code == "content_check_rejected"
    assert "sensitive content" in info.value.detail
    assert "内容审核未通过" in str(info.value)


@pytest.mark.asyncio
async def test_score_parses_expectation_and_legend() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "jev-latest",
                "answers": {
                    "answer": {
                        "type": "score",
                        "score": 1.7,
                        "confidence": 0.55,
                        "legend": {"0": "可以等", "1": "本周处理", "2": "今天处理"},
                        "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                    }
                },
            },
        )

    async with _client(handler) as client:
        answer = await client.score("有多急？", ["可以等", "本周处理", "今天处理"])

    assert answer.score == 1.7
    assert answer.type == "score"
    assert answer.legend["2"] == "今天处理"
    assert answer.probabilities == {"0": 0.1, "1": 0.2, "2": 0.7}
