"""规则、绑定与更多设置的读取、整体约束校验和原子提交。

配置文件仍位于插件数据目录的 settings.json；提交约定见 ADR-0028。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from astrbot.api import logger
from astrbot.api.star import StarTools

from ..common import PLUGIN_NAME
from .models import (
    CUSTOM_DECISION_PREFIX,
    Binding,
    DecisionModel,
    Rule,
    RuleError,
    Settings,
)

SCHEMA_VERSION = 5
SETTINGS_FILENAME = "settings.json"
DEFAULT_BINDING_NAME_PREFIX = "未命名"
"""绑定名称为空时的占位名前缀，拼成 ``未命名-N``（N 取该机器人下的第一个空位）。"""


class RuleStore:
    """规则、绑定与更多设置的提交 module。

    写入方法返回即代表落盘并发布成功，不再要求调用方另行 save。
    读取到的对象只供查看；更改必须穿过写入 interface。
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir
        self.rules: dict[str, Rule] = {}
        self.bindings: dict[str, Binding] = {}
        self.settings = Settings()
        self.load_failed = False
        self._last_error = ""
        self._umo_index: dict[str, str] | None = None
        self._revision = 0
        self._binding_guard: Callable[[str], None] | None = None
        self._on_commit: Callable[[], None] | None = None

    @property
    def revision(self) -> int:
        """进程内已发布配置的版本；不写入配置文件。"""
        return self._revision

    # ------------------------------------------------------------------ 路径
    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:
            self._data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        return self._data_dir

    @property
    def path(self) -> Path:
        return self.data_dir / SETTINGS_FILENAME

    @property
    def last_error(self) -> str:
        return self._last_error

    # ------------------------------------------------------------------ 读写
    def load(self, tz: ZoneInfo) -> None:
        """从磁盘载入规则；文件缺失视为空配置。"""
        self._revision += 1
        self.load_failed = False
        self._last_error = ""
        self.rules = {}
        self.bindings = {}
        self.settings = Settings()
        self._invalidate()

        path = self.path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._fail(f"读取 {path} 失败：{exc}")
            return
        if not isinstance(payload, dict):
            self._fail("settings.json 根节点必须是对象")
            return

        for raw in payload.get("rules") or []:
            try:
                rule = Rule.parse(raw, tz, strict=False)
            except RuleError as exc:
                logger.error("[Ambient] 跳过非法规则：%s", exc)
                continue
            self.rules[rule.id] = rule

        try:
            self.settings = Settings.parse(payload.get("settings"))
        except RuleError as exc:
            logger.warning("[Ambient] 全局设置不合法，回退默认值：%s", exc)
            self.settings = Settings()

        self._load_bindings(payload.get("bindings"))

    def _load_bindings(self, raw: Any) -> None:
        if not isinstance(raw, list):
            return
        for item in raw:
            try:
                binding = Binding.parse(item)
            except RuleError as exc:
                logger.error("[Ambient] 跳过非法绑定：%s", exc)
                continue
            if binding.rule_id not in self.rules:
                logger.warning(
                    "[Ambient] 清理指向不存在规则的绑定：%s -> %s",
                    binding.id,
                    binding.rule_id,
                )
                continue
            overlap = next(
                (
                    other
                    for other in self.bindings.values()
                    if other.platform_id == binding.platform_id
                    and any(umo in other.targets for umo in binding.targets)
                ),
                None,
            )
            if overlap is not None:
                logger.warning(
                    "[Ambient] 绑定 %s 的目标会话已被绑定 %s 占用，跳过这条。",
                    binding.id,
                    overlap.id,
                )
                continue
            self.bindings[binding.id] = binding
        self._invalidate()

    def _fail(self, message: str) -> None:
        self.load_failed = True
        self._last_error = message
        logger.error("[Ambient] %s", message)

    def _commit(
        self,
        *,
        rules: dict[str, Rule] | None = None,
        bindings: dict[str, Binding] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """同步完成校验后的写盘及发布；失败不改变任何已发布对象。

        调用方准备新的字典或对象，不能原地修改已发布内容。本方法与各个写入
        interface 之间没有 await，在插件的单事件循环中提交不会交错。
        """
        if self.load_failed:
            raise RuleError(
                "上次读取配置失败，已阻止写入以免覆盖原文件，请先修复 settings.json"
            )
        rules = self.rules if rules is None else rules
        bindings = self.bindings if bindings is None else bindings
        settings = self.settings if settings is None else settings
        payload = {
            "schema_version": SCHEMA_VERSION,
            "settings": settings.to_dict(),
            "rules": [rule.to_dict() for rule in rules.values()],
            "bindings": [binding.to_dict() for binding in bindings.values()],
        }
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            # 清理失败不改变提交结果，尤其不能在 replace 成功后假报写入失败。
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
        self.rules, self.bindings, self.settings = rules, bindings, settings
        self._revision += 1
        self._invalidate()
        if self._on_commit is not None:
            self._on_commit()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "path": str(self.path),
            "settings": self.settings.to_dict(),
            "rules": [rule.to_dict() for rule in self.ordered_rules()],
            "bindings": [binding.to_dict() for binding in self.bindings.values()],
            "load_error": self.last_error,
        }

    # ------------------------------------------------------------------ 规则
    def ordered_rules(self) -> list[Rule]:
        """规则的展示顺序：最近修改的在前，见 ADR-0014。

        同一秒保存的规则按插入顺序倒序，保证"后写入的在上面"。
        历史数据没有 ``updated_at``，这些规则统一沉到最后；它们之间同样按插入顺序倒序
        （老数据只会追加，倒序近似「最近新建的在前」）。
        """
        entries = list(enumerate(self.rules.values()))
        stamped = sorted(
            (entry for entry in entries if entry[1].updated_at),
            key=lambda entry: (entry[1].updated_at, entry[0]),
            reverse=True,
        )
        unstamped = [rule for _, rule in entries if not rule.updated_at]
        unstamped.reverse()
        return [rule for _, rule in stamped] + unstamped

    def get_rule(self, rule_id: str) -> Rule | None:
        return self.rules.get(rule_id)

    def upsert_rule(self, raw: Any, tz: ZoneInfo) -> Rule:
        rule = Rule.parse(raw, tz)
        self._ensure_decision_model_exists(rule.intent.decision_provider_id)
        # 时间是「这份配置最后一次被改」，写入点只有这里：绑定引用/解绑不刷新它。
        rule.updated_at = datetime.now(tz).isoformat(timespec="seconds")
        self._commit(rules={**self.rules, rule.id: rule})
        return rule

    def delete_rule(self, rule_id: str) -> int:
        """删除未被任何绑定引用的规则。

        被绑定引用时拒绝删除（不再级联，见 ADR-0014），返回被删除的绑定数（恒为 0，
        保留返回值以兼容既有调用方）。
        """
        rule = self.rules.get(rule_id)
        if rule is None:
            return 0
        used = self.bindings_of_rule(rule_id)
        if used:
            labels = "、".join(self.binding_label(binding) for binding in used[:3])
            if len(used) > 3:
                labels += f" 等 {len(used)} 条"
            raise RuleError(
                f"规则「{rule.name}」正被绑定使用（{labels}），请先解除绑定再删除"
            )
        self._commit(
            rules={key: value for key, value in self.rules.items() if key != rule_id}
        )
        return 0

    # ------------------------------------------------------------------ 绑定
    def get_binding(self, binding_id: str) -> Binding | None:
        return self.bindings.get(binding_id)

    def bindings_of_rule(self, rule_id: str) -> list[Binding]:
        return [
            binding for binding in self.bindings.values() if binding.rule_id == rule_id
        ]

    def upsert_binding(self, raw: Any) -> Binding:
        """新建或更新一条绑定。

        ``enabled`` 不由调用方决定（见 ADR-0015）：新建的绑定落在停止态，
        编辑已有绑定则沿用磁盘上的启停状态，启停只能通过 :meth:`set_binding_enabled` 改。
        运行中的绑定不允许改配置（ADR-0016）：先停止再编辑，避免目标会话、
        规则与已预热缓冲不一致。
        """
        binding = Binding.parse(raw)
        if self._binding_guard is not None:
            self._binding_guard(binding.id)
        if binding.rule_id not in self.rules:
            raise RuleError(f"规则不存在：{binding.rule_id}")
        running = self.bindings.get(binding.id)
        if running is not None and running.enabled:
            raise RuleError(
                f"绑定「{self.binding_label(running)}」运行中，请先停止后再修改"
            )
        # 目标会话在机器人应用内唯一（含已停止的绑定）：启停是暂停绑定，不是释放会话。
        for other in self.bindings.values():
            if other.id == binding.id or other.platform_id != binding.platform_id:
                continue
            occupied = set(other.targets)
            conflict = next(
                (umo for umo in binding.targets if umo in occupied),
                None,
            )
            if conflict is None:
                continue
            raise RuleError(
                f"会话 {conflict} 已被绑定「{self.binding_label(other)}」占用"
            )
        if not binding.name:
            binding.name = self.next_binding_name(binding.platform_id, binding.id)
        # 启停只由「运行管理」页的 toggle 决定（ADR-0015）：新建的绑定一律落在停止态，
        # 编辑保存只改配置、沿用磁盘上的原有启停状态，不因保存而启动。
        existing = self.bindings.get(binding.id)
        binding.enabled = existing.enabled if existing is not None else False
        self._commit(bindings={**self.bindings, binding.id: binding})
        return binding

    def binding_label(self, binding: Binding) -> str:
        """绑定的展示名：优先自带的名称，缺名时回退到规则名。"""
        if binding.name:
            return binding.name
        rule = self.rules.get(binding.rule_id)
        return rule.name if rule is not None else binding.rule_id

    def next_binding_name(self, platform_id: str, exclude_id: str = "") -> str:
        """在指定机器人下取第一个未被占用的 ``未命名-N``。"""
        used = {
            binding.name
            for binding in self.bindings.values()
            if binding.platform_id == platform_id
            and binding.id != exclude_id
            and binding.name
        }
        index = 1
        while f"{DEFAULT_BINDING_NAME_PREFIX}-{index}" in used:
            index += 1
        return f"{DEFAULT_BINDING_NAME_PREFIX}-{index}"

    def delete_binding(self, binding_id: str) -> bool:
        """删除绑定；运行中的绑定先停止才能删（ADR-0016）。"""
        if self._binding_guard is not None:
            self._binding_guard(binding_id)
        binding = self.bindings.get(binding_id)
        if binding is not None and binding.enabled:
            raise RuleError(
                f"绑定「{self.binding_label(binding)}」运行中，请先停止后再删除"
            )
        removed = binding is not None
        if removed:
            self._commit(
                bindings={
                    key: value
                    for key, value in self.bindings.items()
                    if key != binding_id
                }
            )
        return removed

    def set_binding_enabled(self, binding_id: str, enabled: bool) -> Binding:
        binding = self.bindings.get(binding_id)
        if binding is None:
            raise RuleError(f"绑定不存在：{binding_id}")
        updated = replace(binding, enabled=bool(enabled))
        self._commit(bindings={**self.bindings, binding_id: updated})
        return updated

    # ------------------------------------------------------------------ 设置
    def find_decision_model(self, model_id: str) -> DecisionModel | None:
        for model in self.settings.decision_models:
            if model.id == model_id:
                return model
        return None

    def update_settings(self, raw: Any) -> Settings:
        settings = Settings.parse(raw)
        self._ensure_decision_models_kept(settings)
        self._commit(settings=settings)
        return settings

    def mark_cache_images_advice_seen(self) -> Settings:
        """记录「建议开启图片缓存」提示已展示，之后不再重复提示。"""
        if self.settings.cache_images_advice_seen:
            return self.settings
        settings = replace(self.settings, cache_images_advice_seen=True)
        self._commit(settings=settings)
        return settings

    def _ensure_decision_model_exists(self, ref: str) -> None:
        """规则只能引用「更多设置」里真实存在的自定义决策模型。"""
        if not ref.startswith(CUSTOM_DECISION_PREFIX):
            return
        model_id = ref[len(CUSTOM_DECISION_PREFIX) :]
        if self.find_decision_model(model_id) is None:
            raise RuleError("所选自定义决策模型已不存在，请先在「更多设置」中添加")

    def _ensure_decision_models_kept(self, settings: Settings) -> None:
        """被规则引用的自定义决策模型不允许删除。"""
        kept = {model.id for model in settings.decision_models}
        removed = {model.id for model in self.settings.decision_models} - kept
        if not removed:
            return
        for rule in self.rules.values():
            ref = rule.intent.decision_provider_id
            if not ref.startswith(CUSTOM_DECISION_PREFIX):
                continue
            if ref[len(CUSTOM_DECISION_PREFIX) :] in removed:
                raise RuleError(
                    f"决策模型正被规则「{rule.name}」使用，请先修改该规则再删除"
                )

    # ------------------------------------------------------------------ 运行期
    def active_bindings(self) -> list[Binding]:
        """已启动且存在目标会话的绑定，供调度使用。"""
        result: list[Binding] = []
        for binding in self.bindings.values():
            if not binding.enabled or not binding.targets:
                continue
            if self.rules.get(binding.rule_id) is None:
                continue
            result.append(binding)
        return result

    def binding_for_umo(self, umo: str) -> Binding | None:
        """该会话当前由哪条已启动的绑定接管。

        目标会话在机器人应用内唯一（见 :meth:`upsert_binding`），所以这里必然单值命中。
        """
        if self._umo_index is None:
            self._umo_index = {
                target: binding.id
                for binding in self.active_bindings()
                for target in binding.targets
            }
        binding_id = self._umo_index.get(umo)
        return self.bindings.get(binding_id) if binding_id else None

    def rule_for(self, umo: str) -> Rule | None:
        binding = self.binding_for_umo(umo)
        return self.rules.get(binding.rule_id) if binding else None

    def _invalidate(self) -> None:
        self._umo_index = None
