"""跨插件召回长期记忆：读取 LivingMemory 的记忆库。

自主发言走「直接生成」（``docs/adr/0011``），不经过核心管线，LivingMemory 挂在
``on_llm_request`` 上的自动注入因此不会生效；本模块补上这一段：决策模型与对话
模型需要「这个人/这个群以前说过什么」时，从这里拿。

调用手段不是 HTTP：LivingMemory 没有面向其他插件的稳定 Python API，外部记忆 API
还需要 API Key 与独立监听。这里直接拿 AstrBot 插件注册表里的插件实例
（``StarMetadata.star_cls``），复用：

- 它自己的 ``MemoryEngine.search_memories``（向量 + 关键词 + 图记忆的混合检索，
  与聊天内自动注入是同一条路径，去重与访问统计行为一致）；
- 它自己的 ``core.memory_scope``（作用域模式、频道公共记忆、人格过滤、适配器边界、
  黑白名单），避免在这些语义上另写一套而与主流程漂移。

前提是两者同进程：任何一环不可用（插件未安装、未就绪、模块导入失败、超时、抛错）
都返回空列表，调用方按「没有记忆」继续。主动发言宁可少一条上下文，也不能因为
记忆不可用而放弃整次发言（fail-open）。

``user``（按人共享）作用域下没有群级记忆：每条记忆挂在具体发送者的身份上，群号
查不到。因此群聊按「最近若干不同发送者」并联召回（:func:`collect_recent_senders`），
私聊直接传对端 ID。

内部字段（``initializer`` / ``memory_engine``）没有稳定性承诺，升级 LivingMemory
后需要回归；读取一律用 ``getattr`` 防御，不用直接属性访问。
"""

from __future__ import annotations

import asyncio
import importlib
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger

from ..common import FRIEND_MESSAGE, GROUP_MESSAGE, parse_target

if TYPE_CHECKING:
    from astrbot.api.star import Context

LIVINGMEMORY_PLUGIN_NAME = "astrbot_plugin_livingmemory"
"""插件名（``metadata.yaml`` 的 ``name``），用 ``Context.get_registered_star`` 查。"""

LIVINGMEMORY_MODULE_PATH = "data.plugins.astrbot_plugin_livingmemory.main"
"""插件模块路径，用于在 ``star_map`` 里兜底查找（插件名与目录名不一致时）。"""

LIVINGMEMORY_SCOPE_MODULE = "data.plugins.astrbot_plugin_livingmemory.core.memory_scope"
"""作用域/人格解析模块；与插件在同一进程内共享同一个已导入实例。"""

RECALL_TIMEOUT_SECONDS = 8.0
"""单次召回的超时。

检索要走 embedding，网络慢时可能卡住；一次发言总共只有 60 秒预算，记忆属于
锦上添花，等不起就必须放弃本次召回。
"""

DEFAULT_K = 5
DEFAULT_MAX_K = 10
"""``k`` 的默认值与兜底上限；上限还会被 LivingMemory 的 ``recall_engine.max_k`` 收紧。"""

MAX_QUERY_CHARS = 200
"""召回关键词上限。检索用的是 embedding，过长只会稀释语义。"""

_SCOPE_MODE_KEY = "filtering_settings.memory_scope_mode"
_ISOLATED_KEY = "filtering_settings.isolated_sessions"

PROBE_OK = "ok"
PROBE_NOT_INSTALLED = "not_installed"
PROBE_LOAD_FAILED = "load_failed"
PROBE_DISABLED = "disabled"
PROBE_NOT_READY = "not_ready"
"""``MemoryProbe.detail`` 的取值。

依次是：可用 / 没装 / 装了但没加载成功 / 装了没启用 / 启用了还没就绪。
"""

SenderRef = tuple[str, str]
"""``(sender_id, sender_name)``；昵称只用于 ``identity_aliases`` 映射，可为空串。"""


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """一条召回结果，已脱离 LivingMemory 的内部类型。"""

    memory_id: int
    content: str
    score: float
    importance: float | None = None
    created_at: float | None = None
    """写入时间（epoch 秒）；解析不出来时为 ``None``，渲染时不编造时间。"""
    session_id: str | None = None
    persona_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryProbe:
    """LivingMemory 可调用性探测结果。

    只读插件注册表与实例状态，不碰数据库、不发请求，因此可以在 WebUI 的开关
    事件里同步调用。
    """

    available: bool
    """现在能否正常召回：插件已启用、初始化完成、引擎与配置都在。"""

    installed: bool
    """AstrBot 插件注册表里是否有这个插件（装了但被禁用也算 ``True``）。"""

    detail: str
    """不可用的原因，取值见 :data:`PROBE_OK` 等常量。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "installed": self.installed,
            "detail": self.detail,
            "plugin_name": LIVINGMEMORY_PLUGIN_NAME,
        }


class LivingMemoryRecall:
    """LivingMemory 召回客户端。

    一个实例可以被整个插件共用：它不缓存检索结果，也不持有连接，每次调用都重新
    确认插件实例，因此 LivingMemory 热重载后无需重建本对象。
    """

    def __init__(
        self,
        context: Context,
        *,
        timeout: float = RECALL_TIMEOUT_SECONDS,
        default_k: int = DEFAULT_K,
    ) -> None:
        self._context = context
        self._timeout = float(timeout)
        self._default_k = int(default_k)

    # ------------------------------------------------------------------ 对外接口

    def available(self) -> bool:
        """LivingMemory 是否已加载（同步、只看注册表，不碰数据库）。"""
        return _find_plugin(self._context) is not None

    def probe(self) -> MemoryProbe:
        """检查 LivingMemory 现在能否正常召回。

        开启长期记忆前先问一句：不只是「装没装」，还要「启用没启用」「初始化完没完」。
        LivingMemory 的初始化是异步的，刚启动时可能还没就绪，此时开启只会静默跳过
        召回，用户会以为配置没生效。
        """
        metadata = _find_metadata(self._context)
        if metadata is None:
            # 注册表里没有：装了但导入失败（依赖缺失、报错）时也是这样，靠目录区分。
            if _plugin_dir_exists():
                return MemoryProbe(False, True, PROBE_LOAD_FAILED)
            return MemoryProbe(False, False, PROBE_NOT_INSTALLED)
        if not bool(getattr(metadata, "activated", True)):
            return MemoryProbe(False, True, PROBE_DISABLED)
        # 插件被禁用时 star_cls 为 None，此时仍算「装了没启用」。
        plugin = getattr(metadata, "star_cls", None)
        if plugin is None or not _engine_ready(plugin):
            return MemoryProbe(False, True, PROBE_NOT_READY)
        return MemoryProbe(True, True, PROBE_OK)

    async def recall(
        self,
        query: str,
        *,
        umo: str,
        event: Any = None,
        k: int | None = None,
        persona_id: str | None = None,
        sender_id: str = "",
        sender_name: str = "",
        senders: Sequence[SenderRef] | None = None,
        track_access: bool = False,
        timeout: float | None = None,
    ) -> list[MemoryItem]:
        """召回与 ``query`` 相关的长期记忆。

        Args:
            query: 召回关键词。用主题词、人名、承诺这类短语，不要整条消息原文。
            umo: 目标会话；决定记忆作用域与默认人格。
            event: 真实消息事件。只有 ``unified_msg_origin`` 可用时传 ``None``，
                本模块会用 :class:`_SessionRef` 替身补齐平台/群号/发送者。
            k: 期望条数；缺省用构造参数，并按 LivingMemory 的 ``recall_engine.max_k``
                收紧。
            persona_id: 强制指定人格过滤；缺省沿用 LivingMemory 按会话解析的结果。
            sender_id: 单个发送者 ID（私聊即对端）。``user``（按人共享）作用域下
                不提供、``senders`` 也为空时跳过召回。
            sender_name: 单个发送者昵称，仅在 ``identity_aliases`` 配置里用于映射身份。
            senders: 多个发送者 ``[(sender_id, sender_name), ...]``，最近优先。
                ``user`` 作用域下并联成一次 ``IN`` 检索（群聊按最近若干人召回）。
            track_access: 是否记为「记忆被使用」并更新访问统计。注入进提示才算使用，
                由调用方决定；默认不计，避免把召回当成强化。
            timeout: 覆盖本次的超时秒数。

        Returns:
            召回结果列表；任何不可用或失败情况都返回空列表，不抛异常。
        """
        cleaned = _compact_query(query)
        if not cleaned:
            return []

        ready = self._ready_engine(umo)
        if ready is None:
            return []
        engine, config = ready

        limit = self._resolve_k(config, k)
        pairs = _merge_senders(sender_id, sender_name, senders)
        platform_name = _platform_name(self._context, _platform_id(umo))
        if not self._scope_usable(config, umo, pairs):
            return []
        ref = event
        if ref is None:
            # 私聊可像真实事件一样按对端身份过白名单；群聊仍只按会话授权。
            target = parse_target(umo)
            sender = ("", "")
            if pairs and target and target.message_type == FRIEND_MESSAGE:
                sender = pairs[0]
            ref = _SessionRef(
                umo,
                platform_name=platform_name,
                sender_id=sender[0],
                sender_name=sender[1],
            )

        budget = self._timeout if timeout is None else float(timeout)
        try:
            return await asyncio.wait_for(
                self._search(
                    engine,
                    config,
                    ref,
                    umo=umo,
                    platform_name=platform_name,
                    pairs=pairs,
                    query=cleaned,
                    k=limit,
                    persona_id=persona_id,
                    track_access=track_access,
                ),
                budget,
            )
        except TimeoutError:
            logger.warning(
                "[Ambient] 召回长期记忆超时（%.1fs），本次按无记忆继续：%s", budget, umo
            )
            return []
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # 记忆是可选上下文：检索失败不能升级成发言失败，也不该刷屏。
            logger.debug(
                "[Ambient] 召回长期记忆失败（%s）：%s: %s", umo, type(exc).__name__, exc
            )
            return []

    # ------------------------------------------------------------------ 内部

    def _ready_engine(self, umo: str) -> tuple[Any, Any] | None:
        """取 ``(memory_engine, config_manager)``，顺带做就绪判定。"""
        plugin = _find_plugin(self._context)
        if plugin is None:
            logger.debug("[Ambient] 未加载 LivingMemory，跳过长期记忆召回")
            return None
        initializer = getattr(plugin, "initializer", None)
        if initializer is None:
            return None
        # 不等待初始化：等待可能长达数十秒，主动发言等不起，也等不出更好的结果。
        if not bool(getattr(initializer, "is_initialized", False)):
            logger.debug("[Ambient] LivingMemory 尚未就绪，跳过长期记忆召回：%s", umo)
            return None
        engine = getattr(initializer, "memory_engine", None)
        config = getattr(plugin, "config_manager", None)
        if engine is None or config is None:
            return None
        return engine, config

    def _session_ref(
        self,
        umo: str,
        *,
        sender_id: str = "",
        sender_name: str = "",
        platform_name: str = "",
    ) -> _SessionRef:
        """构造事件替身；``platform_name`` 已算出时直接复用，避免重复查平台实例。"""
        return _SessionRef(
            umo,
            platform_name=platform_name
            or _platform_name(self._context, _platform_id(umo)),
            sender_id=sender_id,
            sender_name=sender_name,
        )

    @staticmethod
    def _scope_usable(config: Any, umo: str, pairs: Sequence[SenderRef]) -> bool:
        """``user`` 作用域必须知道查谁，否则算出来的 scope 永远匹配不到写入值。

        隔离会话（``filtering_settings.isolated_sessions``）在 LivingMemory 里优先于
        作用域模式，按会话 ID 存取，因此不需要发送者身份。
        """
        if not _is_user_scope_mode(config):
            return True
        if umo in _isolated_sessions(config):
            return True
        if pairs:
            return True
        logger.debug(
            "[Ambient] LivingMemory 使用按人作用域但未提供发送者身份，跳过召回：%s", umo
        )
        return False

    @staticmethod
    def _resolve_k(config: Any, k: int | None) -> int:
        requested = DEFAULT_K if k is None else k
        try:
            requested_int = int(requested)
        except (TypeError, ValueError):
            requested_int = DEFAULT_K
        max_k = _config_get(config, "recall_engine.max_k", DEFAULT_MAX_K)
        try:
            max_k_int = int(max_k)
        except (TypeError, ValueError):
            max_k_int = DEFAULT_MAX_K
        return max(1, min(requested_int, max(1, max_k_int)))

    async def _search(
        self,
        engine: Any,
        config: Any,
        ref: Any,
        *,
        umo: str,
        platform_name: str,
        pairs: Sequence[SenderRef],
        query: str,
        k: int,
        persona_id: str | None,
        track_access: bool,
    ) -> list[MemoryItem]:
        async with _open_access(config, self._context, ref) as access:
            if access is None:
                return []
            if not bool(getattr(access, "allowed", False)):
                logger.debug("[Ambient] LivingMemory 判定本会话不可记忆，跳过召回")
                return []
            scopes = _scope_filter(
                config, access, umo=umo, platform_name=platform_name, pairs=pairs
            )
            if scopes is None:
                return []
            scope, extra = scopes
            kwargs: dict[str, Any] = {
                "query": query,
                "k": k,
                "session_id": scope,
                "persona_id": persona_id or getattr(access, "recall_persona_id", None),
            }
            if extra:
                kwargs["extra_scopes"] = extra
            if track_access:
                kwargs["track_access"] = True
            results = await engine.search_memories(**kwargs)
            return [
                item
                for item in (_to_item(result) for result in results or [])
                if item is not None
            ]


class _SessionRef:
    """把 UMO 伪装成一个消息事件，复用 LivingMemory 的作用域解析。

    ``memory_scope`` 只用 duck typing 读几个访问器（``unified_msg_origin`` /
    ``get_platform_name`` / ``get_group_id`` / ``get_sender_id`` /
    ``get_sender_name``），因此自主发言这种「只有目标会话、没有真实事件」的路径
    也能算出与聊天内一致的 scope 与人格，不必在插件层重写一套作用域语义。

    ``get_platform_name`` 必须是适配器**类型**（如 ``aiocqhttp``）而不是配置 ID：
    频道公共记忆的作用域里写的就是类型名，返回 ID 会查不到。
    """

    def __init__(
        self,
        umo: str,
        *,
        platform_name: str = "",
        sender_id: str = "",
        sender_name: str = "",
    ) -> None:
        target = parse_target(umo)
        self.unified_msg_origin = umo
        self._platform_id = target.platform_id if target else umo.partition(":")[0]
        self._platform_name = platform_name or self._platform_id
        self._group_id = target.session_id if target and target.is_group else ""
        self._is_group = bool(target and target.is_group)
        self._sender_id = sender_id
        self._sender_name = sender_name

    def get_platform_id(self) -> str:
        return self._platform_id

    def get_platform_name(self) -> str:
        return self._platform_name

    def get_group_id(self) -> str:
        return self._group_id

    def get_sender_id(self) -> str:
        return self._sender_id

    def get_sender_name(self) -> str:
        return self._sender_name

    def get_message_type(self) -> str:
        return GROUP_MESSAGE if self._is_group else FRIEND_MESSAGE


@asynccontextmanager
async def _open_access(config: Any, context: Any, event: Any) -> AsyncIterator[Any]:
    """在 LivingMemory 的适配器边界内解析本次召回的作用域与人格。

    适配器边界用它的 ``adapter_access`` 上下文管理器压栈，检索期间生效，退出即
    复位；解析出的 ``scope`` / ``recall_persona_id`` / ``extra_read_scopes`` 与
    聊天内召回完全相同。
    """
    module = _import_scope_module()
    if module is None:
        yield None
        return
    with module.adapter_access(config, event):
        yield await module.resolve_event_memory_access(config, context, event)


def _import_scope_module() -> Any | None:
    """导入 LivingMemory 的作用域模块；未安装时返回 ``None``。"""
    try:
        return importlib.import_module(LIVINGMEMORY_SCOPE_MODULE)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[Ambient] 未能导入 LivingMemory 作用域模块：%s", exc)
        return None


def _find_plugin(context: Any) -> Any | None:
    """按插件名取已加载的 LivingMemory 插件实例。

    插件可能被禁用或热重载：每次调用都重新查注册表，不缓存实例（旧实例的数据库
    已经关闭）。
    """
    return _instance_from(_find_metadata(context))


def _find_metadata(context: Any) -> Any | None:
    """取 LivingMemory 的 ``StarMetadata``；注册表里没有时返回 ``None``。

    被禁用的插件仍然留在注册表里（``activated=False``、``star_cls=None``），
    :meth:`LivingMemoryRecall.probe` 靠这一点区分「没装」与「装了没启用」。
    """
    getter = getattr(context, "get_registered_star", None)
    if callable(getter):
        try:
            metadata = getter(LIVINGMEMORY_PLUGIN_NAME)
        except Exception:  # noqa: BLE001
            metadata = None
        if metadata is not None:
            return metadata
    try:
        from astrbot.core.star.star import star_map
    except Exception:  # noqa: BLE001
        return None
    return star_map.get(LIVINGMEMORY_MODULE_PATH)


def _engine_ready(plugin: Any) -> bool:
    """``initializer`` 已初始化，且 ``memory_engine`` 与 ``config_manager`` 都在。"""
    initializer = getattr(plugin, "initializer", None)
    if initializer is None:
        return False
    if not bool(getattr(initializer, "is_initialized", False)):
        return False
    if getattr(initializer, "memory_engine", None) is None:
        return False
    return getattr(plugin, "config_manager", None) is not None


def _plugin_dir_exists() -> bool:
    """插件目录是否存在。

    注册表里查不到插件时用来区分「没装」与「装了但没加载成功」（依赖缺失、导入报错、
    目录名与插件名不一致）。取不到 AstrBot 路径时按没装处理。
    """
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_plugin_path

        plugin_dir = Path(get_astrbot_plugin_path()) / LIVINGMEMORY_PLUGIN_NAME
        return plugin_dir.is_dir()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[Ambient] 未能判断 LivingMemory 插件目录：%s", exc)
        return False


def _instance_from(metadata: Any) -> Any | None:
    if metadata is None or not bool(getattr(metadata, "activated", True)):
        return None
    return getattr(metadata, "star_cls", None)


def _platform_name(context: Any, platform_id: str) -> str:
    """UMO 前缀是适配器配置 ID，这里换成适配器类型名（如 ``aiocqhttp``）。"""
    manager = getattr(context, "platform_manager", None)
    get_insts = getattr(manager, "get_insts", None)
    if callable(get_insts):
        try:
            instances = list(get_insts() or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 读取平台实例失败：%s", exc)
            instances = []
        for adapter in instances:
            meta = getattr(adapter, "meta", None)
            if not callable(meta):
                continue
            try:
                info = meta()
            except Exception:  # noqa: BLE001
                continue
            if str(getattr(info, "id", "") or "") == platform_id:
                return str(getattr(info, "name", "") or "") or platform_id
    return platform_id


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    """读 LivingMemory 的 ``ConfigManager``；缺失或异常时回退默认值。"""
    getter = getattr(config, "get", None)
    if not callable(getter):
        return default
    try:
        value = getter(key, default)
    except Exception:  # noqa: BLE001
        return default
    return default if value is None else value


def _value_list(value: Any) -> set[str]:
    """解析逗号/分号/换行分隔的配置列表，与 LivingMemory 的 ``parse_value_list`` 一致。"""
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = re.split(r"[,;\n]", str(value or ""))
    return {item.strip() for item in items if item.strip()}


def _event_value(event: Any, method_name: str) -> str:
    method = getattr(event, method_name, None)
    if not callable(method):
        return ""
    try:
        return str(method() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _platform_id(umo: str) -> str:
    target = parse_target(umo)
    return target.platform_id if target else umo.partition(":")[0]


def _merge_senders(
    sender_id: str,
    sender_name: str,
    senders: Sequence[SenderRef] | None,
) -> list[SenderRef]:
    """把单个发送者快捷参数与 ``senders`` 合并成去重后的列表（最近优先保持原序）。

    昵称取先出现的非空值：同一个人可能在缓冲里带着空昵称出现多次。
    """
    ordered: list[SenderRef] = []
    given: list[SenderRef] = [(sender_id, sender_name)]
    if senders:
        given.extend(senders)
    names: dict[str, str] = {}
    for raw_id, raw_name in given:
        cleaned_id = str(raw_id or "").strip()
        if not cleaned_id:
            continue
        cleaned_name = str(raw_name or "").strip()
        if cleaned_id in names:
            if cleaned_name and not names[cleaned_id]:
                names[cleaned_id] = cleaned_name
            continue
        names[cleaned_id] = cleaned_name
        ordered.append((cleaned_id, cleaned_name))
    return [(sender, names[sender]) for sender, _ in ordered]


def _is_user_scope_mode(config: Any) -> bool:
    return str(_config_get(config, _SCOPE_MODE_KEY, "legacy")).casefold() == "user"


def _isolated_sessions(config: Any) -> set[str]:
    return _value_list(_config_get(config, _ISOLATED_KEY, ""))


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _scope_filter(
    config: Any,
    access: Any,
    *,
    umo: str,
    platform_name: str,
    pairs: Sequence[SenderRef],
) -> tuple[str | None, list[str]] | None:
    """算出本次检索的 ``(主作用域, 并联作用域)``。

    返回 ``None`` 表示必要作用域解析失败，必须跳过召回；不能把它当成无过滤。

    - 非 ``user`` 模式：主作用域由 LivingMemory 自己解析（umo / global / None）；
    - ``user`` 模式：记忆按人存，主作用域必须由发送者身份算，多个发送者并联成一次
      ``IN``；``access.scope`` 在这种模式下是「身份退化成 umo」的无效值，必须丢弃；
    - 隔离会话优先：LivingMemory 对 ``isolated_sessions`` 按会话 ID 存取，沿用
      ``access.scope``；
    - 频道公共记忆等附加作用域（``extra_read_scopes``）始终并联。
    """
    extra = [str(scope) for scope in getattr(access, "extra_read_scopes", ()) or ()]
    raw_scope = getattr(access, "scope", None)
    main_scope = (raw_scope.strip() or None) if isinstance(raw_scope, str) else None
    isolated = _isolated_sessions(config)
    if umo in isolated:
        return (main_scope, _dedupe(extra)) if main_scope == umo else None
    mode = str(_config_get(config, _SCOPE_MODE_KEY, "legacy")).casefold()
    if mode != "user":
        # 只有明确关闭 legacy 会话过滤才允许空主作用域；存在隔离会话时也不能查全库。
        unfiltered = (
            mode == "legacy"
            and not isolated
            and not _config_get(
                config, "filtering_settings.use_session_filtering", True
            )
        )
        return (main_scope, _dedupe(extra)) if main_scope or unfiltered else None
    scopes = _resolve_user_scopes(config, umo, platform_name, pairs)
    if not scopes:
        return None
    merged = _dedupe([*scopes, *extra])
    if len(merged) == 1:
        return merged[0], []
    return None, merged


def _resolve_user_scopes(
    config: Any,
    umo: str,
    platform_name: str,
    pairs: Sequence[SenderRef],
) -> list[str]:
    """按发送者身份算出 ``livingmemory:user:<平台>:<身份>`` 列表。

    交给 LivingMemory 自己的 ``resolve_user_memory_scope`` 计算，别在本地拼字符串：
    身份可能被 ``access_control.identity_aliases`` 映射成规范名。
    """
    module = _import_scope_module()
    if module is None:
        return []
    resolver = getattr(module, "resolve_user_memory_scope", None)
    if not callable(resolver):
        return []
    scopes: list[str] = []
    for sender_id, sender_name in pairs:
        ref = _SessionRef(
            umo,
            platform_name=platform_name,
            sender_id=sender_id,
            sender_name=sender_name,
        )
        try:
            scope = str(resolver(config, ref) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 解析用户记忆作用域失败（%s）：%s", sender_id, exc)
            continue
        if scope:
            scopes.append(scope)
    return _dedupe(scopes)


def _compact_query(query: Any) -> str:
    """召回关键词压成单行并限长：换行只会让 embedding 抓到多余语义。"""
    text = " ".join(str(query or "").split())
    return text if len(text) <= MAX_QUERY_CHARS else text[:MAX_QUERY_CHARS]


def _to_item(result: Any) -> MemoryItem | None:
    """转成内部结果；过滤空正文和未披露的私密记忆。"""
    content = str(getattr(result, "content", "") or "").strip()
    if not content:
        return None
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    if str(metadata.get("visibility") or "").strip().casefold() == "private":
        # 自主发言无法确认听众是否知情；只接受明确标记已披露的私密条目。
        disclosed = str(metadata.get("disclosed") or "").strip().casefold()
        if disclosed not in {"true", "1", "yes", "on"}:
            return None
    importance = _as_float(metadata.get("importance"))
    return MemoryItem(
        memory_id=_as_int(getattr(result, "doc_id", None)),
        content=content,
        score=_as_float(getattr(result, "final_score", None)) or 0.0,
        importance=importance,
        created_at=_as_timestamp(metadata.get("create_time")),
        session_id=_as_text(metadata.get("session_id")),
        persona_id=_as_text(metadata.get("persona_id")),
    )


def _as_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_timestamp(value: Any) -> float | None:
    """把 ``create_time`` 解析成 epoch 秒；``0`` 与无法解析的值一律视为「时间未知」。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if seconds > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = 0.0
    if seconds > 0:
        return seconds
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


__all__ = [
    "DEFAULT_K",
    "DEFAULT_MAX_K",
    "LIVINGMEMORY_PLUGIN_NAME",
    "LivingMemoryRecall",
    "MemoryItem",
    "RECALL_TIMEOUT_SECONDS",
    "SenderRef",
]
