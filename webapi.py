"""WebUI 桥接 API：规则 CRUD、Ambient 绑定 CRUD/启停、运行状态、更多设置。

路由前缀固定为插件名，视图侧用相对 endpoint 调用（见
``docs/adr/0002``）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.provider import DecisionError
from astrbot.api.web import error_response, json_response, request

from .common import PLUGIN_NAME, parse_target
from .conversation.messages import format_stamp
from .rules.models import DecisionModel, RuleError
from .speech.decision import probe_decision_model
from .speech.systemone_sdk import SystemOneError

if TYPE_CHECKING:
    from .main import AmbientChatPlugin

ROUTE_PREFIX = f"/{PLUGIN_NAME}"

MAX_SESSION_SUGGESTIONS = 200

MAX_BUFFER_PREVIEW = 200
"""「运行管理 → 缓冲区」每个会话最多返回的消息条数。

缓冲容量可由「更多设置」调到 1000，绑定又可能有多个目标会话，
一次性全量返回会让弹窗和接口都很重；超出部分只在汇总里体现条数。
"""

NON_BINDABLE_PLATFORMS = frozenset({"webchat"})
"""不能作为 Ambient 绑定目标的平台：WebChat 是 WebUI 内置聊天入口，无法主动发消息。"""


class AmbientWebApi:
    """把插件能力暴露给 ``views/console`` 页面。"""

    def __init__(self, plugin: AmbientChatPlugin) -> None:
        self._plugin = plugin

    def register(self) -> None:
        routes = [
            (
                f"{ROUTE_PREFIX}/overview",
                self.get_overview,
                ["GET"],
                "控制台初始化数据（规则 / 绑定 / 设置 / 可选模型）",
            ),
            (
                f"{ROUTE_PREFIX}/status",
                self.get_status,
                ["GET"],
                "已启动绑定的运行状态",
            ),
            (
                f"{ROUTE_PREFIX}/sessions",
                self.list_sessions,
                ["GET"],
                "按机器人应用列出最近会话，供绑定选择",
            ),
            (
                f"{ROUTE_PREFIX}/buffer",
                self.get_buffer,
                ["GET"],
                "只读查看绑定的上下文缓冲（可按单个会话过滤）",
            ),
            (
                f"{ROUTE_PREFIX}/rules/save",
                self.save_rule,
                ["POST"],
                "新建或更新规则",
            ),
            (
                f"{ROUTE_PREFIX}/rules/delete",
                self.delete_rule,
                ["POST"],
                "删除规则（被绑定引用时拒绝）",
            ),
            (
                f"{ROUTE_PREFIX}/bindings/save",
                self.save_binding,
                ["POST"],
                "新建或更新 Ambient 绑定",
            ),
            (
                f"{ROUTE_PREFIX}/bindings/delete",
                self.delete_binding,
                ["POST"],
                "删除 Ambient 绑定",
            ),
            (
                f"{ROUTE_PREFIX}/bindings/toggle",
                self.toggle_binding,
                ["POST"],
                "启动 / 停止 Ambient 绑定",
            ),
            (
                f"{ROUTE_PREFIX}/bindings/trigger",
                self.trigger_binding_target,
                ["POST"],
                "对一个目标会话立即手动触发一次（跳过入口触发与挑选策略）",
            ),
            (
                f"{ROUTE_PREFIX}/settings/save",
                self.save_settings,
                ["POST"],
                "保存更多设置",
            ),
            (
                f"{ROUTE_PREFIX}/settings/advice_seen",
                self.mark_cache_images_advice_seen,
                ["POST"],
                "记录「建议开启图片缓存」提示已展示",
            ),
            (
                f"{ROUTE_PREFIX}/settings/astrbot_advice_seen",
                self.mark_astrbot_config_advice_seen,
                ["POST"],
                "记录「建议调整 AstrBot 侧配置」提示已展示",
            ),
            (
                f"{ROUTE_PREFIX}/configuration/apply",
                self.apply_configuration,
                ["POST"],
                "重新应用当前已保存配置（不重复写盘）",
            ),
            (
                f"{ROUTE_PREFIX}/decision_models/test",
                self.test_decision_model,
                ["POST"],
                "测试自定义决策模型（一次请求覆盖三种题型）",
            ),
        ]
        for route, handler, methods, desc in routes:
            self._plugin.context.register_web_api(route, handler, methods, desc)

    # ------------------------------------------------------------------ 读
    async def get_overview(self) -> Any:
        store = self._plugin.store
        snapshot = store.snapshot()
        platforms = await self._platforms()
        labels = {item["id"]: item["label"] for item in platforms}
        rule_names = {rule.id: rule.name for rule in store.rules.values()}
        bindings = []
        lifecycle = self._plugin.lifecycle.snapshot()
        for binding in store.bindings.values():
            item = binding.to_dict()
            item.update(lifecycle.get(binding.id, {}))
            item["rule_name"] = rule_names.get(binding.rule_id, binding.rule_id)
            item["platform_label"] = labels.get(
                binding.platform_id, binding.platform_id
            )
            bindings.append(item)

        return json_response(
            {
                "plugin_name": PLUGIN_NAME,
                "enabled": self._plugin.is_enabled(),
                "timezone": str(self._plugin.timezone),
                "config_path": snapshot["path"],
                "media_path": str(self._plugin.buffer.media_root),
                "schema_version": snapshot["schema_version"],
                "load_error": snapshot["load_error"],
                "application": self._plugin.configuration.snapshot(),
                "settings": snapshot["settings"],
                "rules": snapshot["rules"],
                "bindings": bindings,
                "platforms": platforms,
                "chat_providers": self._chat_providers(),
                "decision_providers": self._decision_providers(),
            }
        )

    async def get_status(self) -> Any:
        return json_response(
            {
                "enabled": self._plugin.is_enabled(),
                "timezone": str(self._plugin.timezone),
                "instances": self._plugin.scheduler.status(),
                "load_error": self._plugin.store.last_error,
                "application": self._plugin.configuration.snapshot(),
            }
        )

    async def get_buffer(self) -> Any:
        """只读查看某条绑定的上下文缓冲，按会话分组。

        传 ``umo`` 时只返回该目标会话；不传则覆盖绑定的全部目标会话。
        缓冲只在内存里，停止的绑定会被清空，所以这里给的只是「此刻」的快照；
        单条消息沿用决策 / 对话请求的渲染（剔除 ``system_reminder``、截断正文、
        图片转成占位符），保证「弹窗里看到的」与「模型看到的」是同一份文本。
        """
        binding_id = str(request.query.get("binding_id", "") or "").strip()
        if not binding_id:
            return error_response("缺少 binding_id", status_code=400)
        binding = self._plugin.store.bindings.get(binding_id)
        if binding is None:
            return error_response(f"绑定不存在：{binding_id}", status_code=404)

        targets = list(binding.targets)
        only_umo = str(request.query.get("umo", "") or "").strip()
        if only_umo:
            # 只允许查该绑定的目标会话，避免这个只读接口变成任意 UMO 的窥视口。
            if only_umo not in targets:
                return error_response(
                    f"该会话不属于此绑定：{only_umo}", status_code=400
                )
            targets = [only_umo]

        buffer = self._plugin.buffer
        tz = self._plugin.timezone
        sessions: list[dict[str, Any]] = []
        total = 0
        for umo in targets:
            target = parse_target(umo)
            is_group = bool(target and target.is_group)
            # recent(…, maxlen) 即整条环形缓冲（容量上限就是 maxlen）。
            messages = buffer.recent(umo, buffer.maxlen)
            shown = messages[-MAX_BUFFER_PREVIEW:]
            total += len(messages)
            sessions.append(
                {
                    "umo": umo,
                    "session_name": buffer.session_name(umo),
                    "message_type": target.message_type if target else "",
                    "is_group": is_group,
                    "tracked": buffer.has(umo),
                    "count": len(messages),
                    "truncated": len(messages) > len(shown),
                    "media_count": sum(len(item.images) for item in messages),
                    "media_bytes": sum(item.image_bytes for item in messages),
                    "audio_count": sum(len(item.audio) for item in messages),
                    "audio_bytes": sum(item.audio_bytes for item in messages),
                    "messages": [
                        {
                            "at": format_stamp(item.timestamp, tz),
                            # 名字与 ID 分开给：前端按两列展示，同名用户必须能区分开。
                            "sender_name": item.structured_sender_name(),
                            "sender_id": "" if item.is_bot else item.sender_id.strip(),
                            "self": item.is_bot,
                            "text": item.render_text(),
                            "images": len(item.images),
                            "audio": len(item.audio),
                        }
                        for item in shown
                    ],
                }
            )

        rule = self._plugin.store.get_rule(binding.rule_id)
        return json_response(
            {
                "binding_id": binding.id,
                "binding_name": binding.name,
                "platform_id": binding.platform_id,
                "rule_name": rule.name if rule else binding.rule_id,
                "enabled": binding.enabled,
                "buffer_maxlen": buffer.maxlen,
                "preview_limit": MAX_BUFFER_PREVIEW,
                "total": total,
                "sessions": sessions,
            }
        )

    async def list_sessions(self) -> Any:
        platform_id = str(request.query.get("platform_id", "") or "").strip()
        try:
            limit = int(request.query.get("limit", 50, type=int))
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(limit, MAX_SESSION_SUGGESTIONS))

        sessions: dict[str, dict[str, Any]] = {}
        manager = getattr(self._plugin.context, "conversation_manager", None)
        if manager is None:
            return json_response({"sessions": [], "notice": "核心未提供会话管理器"})

        try:
            conversations, _ = await manager.get_filtered_conversations(
                page=1,
                page_size=limit,
                platform_ids=[platform_id] if platform_id else None,
                include_history=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 读取最近会话失败：%s", exc)
            return json_response({"sessions": [], "notice": f"读取会话失败：{exc}"})

        for conversation in conversations or []:
            umo = str(getattr(conversation, "user_id", "") or "")
            target = parse_target(umo)
            if target is None or umo in sessions:
                continue
            sessions[umo] = {
                "umo": umo,
                "platform_id": target.platform_id,
                "message_type": target.message_type,
                "session_id": target.session_id,
                "title": str(getattr(conversation, "title", "") or ""),
                "updated_at": int(getattr(conversation, "updated_at", 0) or 0),
            }
        return json_response({"sessions": list(sessions.values()), "notice": ""})

    # ------------------------------------------------------------------ 写
    async def save_rule(self) -> Any:
        payload = await request.json(default={})
        raw_rule = payload.get("rule") if isinstance(payload, dict) else None
        if not isinstance(raw_rule, dict):
            return error_response("缺少 rule 字段", status_code=400)
        try:
            rule = self._plugin.store.upsert_rule(raw_rule, self._plugin.timezone)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            logger.error("[Ambient] 保存规则失败：%s", exc)
            return error_response(f"写入失败：{exc}", status_code=500)
        application = await self._plugin.configuration.apply()
        return json_response(
            {"rule": rule.to_dict(), "saved": True, "application": application}
        )

    async def delete_rule(self) -> Any:
        payload = await request.json(default={})
        rule_id = str((payload or {}).get("rule_id") or "").strip()
        if not rule_id:
            return error_response("缺少 rule_id", status_code=400)
        if rule_id not in self._plugin.store.rules:
            return error_response(f"规则不存在：{rule_id}", status_code=404)
        try:
            self._plugin.store.delete_rule(rule_id)
        except RuleError as exc:
            # 被绑定引用时拒绝删除，报文里点名占用方。
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            return error_response(str(exc), status_code=500)
        application = await self._plugin.configuration.apply()
        return json_response(
            {"deleted": rule_id, "saved": True, "application": application}
        )

    async def save_binding(self) -> Any:
        payload = await request.json(default={})
        raw_binding = payload.get("binding") if isinstance(payload, dict) else None
        if not isinstance(raw_binding, dict):
            return error_response("缺少 binding 字段", status_code=400)
        try:
            binding = self._plugin.lifecycle.save_binding(raw_binding)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            logger.error("[Ambient] 保存绑定失败：%s", exc)
            return error_response(f"写入失败：{exc}", status_code=500)
        application = await self._plugin.configuration.apply()
        return json_response(
            {"binding": binding.to_dict(), "saved": True, "application": application}
        )

    async def delete_binding(self) -> Any:
        payload = await request.json(default={})
        binding_id = str((payload or {}).get("binding_id") or "").strip()
        if not binding_id:
            return error_response("缺少 binding_id", status_code=400)
        try:
            removed = self._plugin.lifecycle.delete_binding(binding_id)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            return error_response(f"写入失败：{exc}", status_code=500)
        if not removed:
            return error_response(f"绑定不存在：{binding_id}", status_code=404)
        application = await self._plugin.configuration.apply()
        return json_response(
            {"deleted": binding_id, "saved": True, "application": application}
        )

    async def toggle_binding(self) -> Any:
        """启动 / 停止绑定：启动会先预热，请求返回即代表预热完成（ADR-0016）。"""
        payload = await request.json(default={})
        payload = payload if isinstance(payload, dict) else {}
        binding_id = str(payload.get("binding_id") or "").strip()
        if not binding_id:
            return error_response("缺少 binding_id", status_code=400)
        if "enabled" not in payload:
            return error_response("缺少 enabled", status_code=400)
        try:
            if bool(payload.get("enabled")):
                binding = await self._plugin.start_binding(binding_id)
            else:
                binding = await self._plugin.stop_binding(binding_id)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            return error_response(f"写入失败：{exc}", status_code=500)
        return json_response(
            {
                "binding": binding.to_dict(),
                "saved": True,
                "application": self._plugin.configuration.snapshot(),
            }
        )

    async def trigger_binding_target(self) -> Any:
        """手动触发一个目标会话：只跳过入口触发与挑选，判断链路与定时触发一致。"""
        payload = await request.json(default={})
        payload = payload if isinstance(payload, dict) else {}
        binding_id = str(payload.get("binding_id") or "").strip()
        umo = str(payload.get("umo") or "").strip()
        if not binding_id or not umo:
            return error_response("缺少 binding_id 或 umo", status_code=400)
        try:
            result = await self._plugin.scheduler.trigger_target(binding_id, umo)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        return json_response(result)

    async def save_settings(self) -> Any:
        payload = await request.json(default={})
        payload = payload if isinstance(payload, dict) else {}
        raw = payload.get("settings")
        if not isinstance(raw, dict):
            raw = payload
        try:
            settings = self._plugin.store.update_settings(raw)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            return error_response(f"写入失败：{exc}", status_code=500)
        application = await self._plugin.configuration.apply()
        return json_response(
            {"settings": settings.to_dict(), "saved": True, "application": application}
        )

    async def mark_cache_images_advice_seen(self) -> Any:
        """只写提示标记：不影响运行中的调度，因此不重新应用配置。"""
        try:
            settings = self._plugin.store.mark_cache_images_advice_seen()
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            return error_response(f"写入失败：{exc}", status_code=500)
        return json_response({"settings": settings.to_dict(), "saved": True})

    async def mark_astrbot_config_advice_seen(self) -> Any:
        """只写提示标记：不改变任何运行时行为，因此不重新应用配置。"""
        try:
            settings = self._plugin.store.mark_astrbot_config_advice_seen()
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        except OSError as exc:
            return error_response(f"写入失败：{exc}", status_code=500)
        return json_response({"settings": settings.to_dict(), "saved": True})

    async def apply_configuration(self) -> Any:
        application = await self._plugin.configuration.apply()
        return json_response({"application": application})

    async def test_decision_model(self) -> Any:
        """测试自定义决策模型：只读，不落盘。

        用**一次**请求同时问 noul / choice / score 三题，验证端点、鉴权、
        `noul.criteria` 兼容性与三种题型的响应格式。
        """
        payload = await request.json(default={})
        payload = payload if isinstance(payload, dict) else {}
        raw = payload.get("model")
        if not isinstance(raw, dict):
            return error_response("缺少 model 字段", status_code=400)
        try:
            model = DecisionModel.parse(raw)
        except RuleError as exc:
            return error_response(str(exc), status_code=400)
        try:
            result = await probe_decision_model(model, store=self._plugin.store)
        except (SystemOneError, DecisionError) as exc:
            logger.warning("[Ambient] 决策模型测试失败（%s）：%s", model.name, exc)
            return error_response(f"测试失败：{exc}", status_code=400)
        except Exception as exc:  # noqa: BLE001
            logger.error("[Ambient] 决策模型测试异常（%s）：%s", model.name, exc)
            return error_response(f"测试异常：{exc}", status_code=500)
        return json_response({"result": result})

    # ------------------------------------------------------------------ 辅助
    async def _platforms(self) -> list[dict[str, Any]]:
        online = await self._plugin.live_platform_status()
        result: list[dict[str, Any]] = []
        for adapter in self._plugin.context.platform_manager.get_insts():
            meta = adapter.meta()
            platform_id = str(getattr(meta, "id", "") or "")
            name = str(getattr(meta, "name", "") or "")
            display = str(
                getattr(meta, "adapter_display_name", "")
                or getattr(meta, "description", "")
                or name
            )
            if not platform_id:
                continue
            result.append(
                {
                    "id": platform_id,
                    "name": name,
                    # 前端把 id 作为主标识、display 作为次要说明分别排版，
                    # label 只留作纯文本兜底（如 title 属性）。
                    "display": display,
                    "label": f"{platform_id}（{display}）" if display else platform_id,
                    "bindable": platform_id not in NON_BINDABLE_PLATFORMS
                    and name not in NON_BINDABLE_PLATFORMS,
                    "online": online.get(platform_id),
                }
            )
        return result

    def _chat_providers(self) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for provider in self._plugin.context.get_all_providers():
            meta = provider.meta()
            result.append(
                {
                    "id": str(getattr(meta, "id", "")),
                    "model": str(getattr(meta, "model", "") or ""),
                    "label": f"{getattr(meta, 'id', '')}"
                    f" / {getattr(meta, 'model', '') or '默认模型'}",
                }
            )
        return result

    def _decision_providers(self) -> list[dict[str, str]]:
        """规则「发言意向」下拉选项：官方判断提供商 + 「更多设置」里的自定义模型。"""
        manager = getattr(self._plugin.context, "provider_manager", None)
        instances = getattr(manager, "decision_provider_insts", None) or []
        result: list[dict[str, str]] = []
        for provider in instances:
            meta = provider.meta()
            result.append(
                {
                    "id": str(getattr(meta, "id", "")),
                    "model": str(getattr(meta, "model", "") or ""),
                    "label": f"{getattr(meta, 'id', '')}"
                    f" / {getattr(meta, 'model', '') or '默认模型'}",
                    "source": "official",
                }
            )
        for model in self._plugin.store.settings.decision_models:
            result.append(
                {
                    "id": model.ref,
                    "model": model.model,
                    "label": f"{model.name} / {model.model}（自定义）",
                    "source": "custom",
                }
            )
        return result
