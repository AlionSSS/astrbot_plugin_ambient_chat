"""规则弹窗的「默认判断模型不可用」判定依赖官方/自定义来源标记。

前端用 `source == "official"` 判断 AstrBot 是否配置了判断提供商；自定义决策模型
存在「更多设置」里，不算 AstrBot 的提供商，必须在下拉里显式选中才生效。
"""

from types import SimpleNamespace

from astrbot_plugin_ambient_chat.webapi import AmbientWebApi


def _web(instances: list, custom_models: list) -> AmbientWebApi:
    plugin = SimpleNamespace(
        context=SimpleNamespace(
            provider_manager=SimpleNamespace(decision_provider_insts=instances)
        ),
        store=SimpleNamespace(settings=SimpleNamespace(decision_models=custom_models)),
    )
    return AmbientWebApi(plugin)


def _official(provider_id: str = "official-1") -> SimpleNamespace:
    return SimpleNamespace(
        meta=lambda: SimpleNamespace(id=provider_id, model="gpt-4o-mini")
    )


def test_custom_model_alone_does_not_count_as_astrbot_provider() -> None:
    custom = SimpleNamespace(ref="custom:1", model="local", name="我的模型")
    providers = _web([], [custom])._decision_providers()

    assert [item["source"] for item in providers] == ["custom"]
    assert not any(item["source"] == "official" for item in providers)


def test_official_provider_is_marked_as_official() -> None:
    providers = _web([_official()], [])._decision_providers()

    assert providers[0]["source"] == "official"
    assert providers[0]["id"] == "official-1"
