"""自定义决策模型：设置持久化校验与分派路径。"""

from __future__ import annotations

import json
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from astrbot_plugin_ambient_chat.rules.models import DecisionModel, RuleError, Settings
from astrbot_plugin_ambient_chat.rules.store import RuleStore
from astrbot_plugin_ambient_chat.speech.decision import (
    DecisionContentRejected,
    DecisionRouter,
    NoulPrompt,
    probe_decision_model,
)

from astrbot.api.provider import DecisionError

TZ = ZoneInfo("Asia/Shanghai")
BASE_URL = "https://example.test/v1"
MODEL_PAYLOAD: dict[str, Any] = {
    "id": "dm_test",
    "base_url": BASE_URL,
    "model": "intern-decision-4b",
    "api_key": "sk-test",
}


def _rule(ref: str) -> dict[str, Any]:
    return {
        "name": "测试规则",
        "triggers": [{"type": "cron", "expr": "*/30 * * * *"}],
        "intent": {
            "mode": "decision",
            "decision_provider_id": ref,
            "threshold": 0.7,
        },
    }


def test_settings_roundtrip_decision_models() -> None:
    # noul_true / noul_false 是已移除的旧字段：读到时忽略，不再落盘。
    settings = Settings.parse(
        {
            "decision_models": [
                {**MODEL_PAYLOAD, "noul_true": "旧字段", "noul_false": "旧字段"}
            ]
        }
    )
    model = settings.decision_models[0]
    assert model.id == "dm_test"
    assert model.name == "intern-decision-4b"
    payload = settings.to_dict()["decision_models"][0]
    assert payload["base_url"] == BASE_URL
    assert "noul_true" not in payload
    assert "noul_false" not in payload
    assert "require_noul_criteria" not in payload


def test_invalid_decision_model_is_rejected() -> None:
    with pytest.raises(RuleError, match="接口地址"):
        Settings.parse({"decision_models": [{"base_url": "ftp://x", "model": "m"}]})
    with pytest.raises(RuleError, match="模型 ID"):
        Settings.parse({"decision_models": [{"base_url": BASE_URL}]})


def test_rule_must_reference_existing_custom_model(tmp_path) -> None:
    store = RuleStore(tmp_path)
    with pytest.raises(RuleError, match="自定义决策模型"):
        store.upsert_rule(_rule("custom:dm_missing"), TZ)

    store.update_settings({"decision_models": [MODEL_PAYLOAD]})
    rule = store.upsert_rule(_rule("custom:dm_test"), TZ)
    assert rule.intent.decision_provider_id == "custom:dm_test"


def test_referenced_decision_model_cannot_be_deleted(tmp_path) -> None:
    store = RuleStore(tmp_path)
    store.update_settings({"decision_models": [MODEL_PAYLOAD]})
    store.upsert_rule(_rule("custom:dm_test"), TZ)

    with pytest.raises(RuleError, match="正被规则"):
        store.update_settings({"decision_models": []})


@pytest.mark.asyncio
async def test_router_custom_model_sends_noul_criteria(tmp_path) -> None:
    store = RuleStore(tmp_path)
    store.update_settings({"decision_models": [MODEL_PAYLOAD]})
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "intern-decision-4b",
                "answers": {"interested": {"noul": 0.82}},
                "usage": {"input_tokens": 9, "output_tokens": 1},
            },
        )

    router = DecisionRouter(
        None,  # type: ignore[arg-type]
        store,
        transport=httpx.MockTransport(handler),
    )
    try:
        probabilities = await router.probabilities(
            "custom:dm_test",
            {"interested": NoulPrompt("想接一句吗？", "有话可说", "无话可说")},
            {"conversation": "..."},
        )
    finally:
        await router.close()

    assert probabilities == {"interested": 0.82}
    # 配置里只填到 /v1，由 SDK 自动补 /systemone。
    assert captured["url"] == f"{BASE_URL}/systemone"
    question = captured["payload"]["questions"]["interested"]
    assert question["criteria"] == {"true": "有话可说", "false": "无话可说"}


@pytest.mark.asyncio
async def test_probe_covers_three_question_types_in_one_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "intern-decision-4b",
                "answers": {
                    "noul": {"type": "noul", "noul": 0.93},
                    "choice": {
                        "type": "choice",
                        "choice": "calm",
                        "confidence": 0.7,
                        "probabilities": {"angry": 0.2, "calm": 0.7, "happy": 0.1},
                    },
                    "score": {
                        "type": "score",
                        "score": 1.4,
                        "confidence": 0.6,
                        "legend": {"0": "可以等", "1": "本周处理", "2": "今天处理"},
                        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
                    },
                },
                "usage": {"input_tokens": 30, "output_tokens": 3},
            },
        )

    result = await probe_decision_model(
        DecisionModel.parse(MODEL_PAYLOAD),
        transport=httpx.MockTransport(handler),
    )

    questions = captured["payload"]["questions"]
    assert {question["type"] for question in questions.values()} == {
        "noul",
        "choice",
        "score",
    }
    # 默认开启强制 criteria，noul 一定带 true/false。
    assert set(questions["noul"]["criteria"]) == {"true", "false"}
    assert result["answers"]["noul"]["probability"] == 0.93
    assert result["answers"]["choice"]["choice"] == "calm"
    assert result["answers"]["score"]["score"] == 1.4
    assert result["usage"] == {"input_tokens": 30, "output_tokens": 3}
    assert result["endpoint"] == f"{BASE_URL}/systemone"


@pytest.mark.asyncio
async def test_router_always_fills_noul_criteria(tmp_path) -> None:
    """即使调用方没给判定说明，插件层也要补上 true/false（不依赖 SDK 默认值）。"""
    store = RuleStore(tmp_path)
    store.update_settings({"decision_models": [MODEL_PAYLOAD]})
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "intern-decision-4b", "answers": {"q": {"noul": 0.4}}},
        )

    router = DecisionRouter(
        None,  # type: ignore[arg-type]
        store,
        transport=httpx.MockTransport(handler),
    )
    try:
        await router.probabilities("custom:dm_test", {"q": NoulPrompt("是吗？")})
    finally:
        await router.close()

    assert captured["payload"]["questions"]["q"]["criteria"] == {
        "true": "该说法成立",
        "false": "该说法不成立",
    }


@pytest.mark.asyncio
async def test_router_missing_custom_model_fails_closed() -> None:
    router = DecisionRouter(None, RuleStore())  # type: ignore[arg-type]
    with pytest.raises(DecisionError, match="不存在"):
        await router.probabilities("custom:dm_missing", {"q": NoulPrompt("是吗？")})


@pytest.mark.asyncio
async def test_router_marks_content_rejection_apart_from_other_failures(
    tmp_path,
) -> None:
    """内容审核拒绝要单独一类：页面得提示「改内容」，不是「改配置」。"""
    store = RuleStore(tmp_path)
    store.update_settings({"decision_models": [MODEL_PAYLOAD]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                'data: {"error":{"message":"The request contains sensitive content.",'
                '"type":"content_check_error","code":"content_check_rejected"}}'
            ),
        )

    router = DecisionRouter(
        None,  # type: ignore[arg-type]
        store,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(DecisionContentRejected, match="内容审核"):
            await router.probabilities("custom:dm_test", {"q": NoulPrompt("是吗？")})
    finally:
        await router.close()
