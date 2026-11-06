"""机器人自身在平台上的显示名（仅 aiocqhttp）。

取名字走 OneBot 的 ``get_login_info``：响应里的 ``nickname`` 是账号昵称。
群聊另试 ``get_group_member_info`` 的 ``card``（群名片），为空再退回昵称。

协议保证 ``nickname`` 字段存在，但运行时不保证可用：调用可能失败，值可能为空
字符串，也可能被实现回填成 QQ 号。这几种情况都视为「没有名字」，由调用方回退
到「我」，并在一段时间后重试，见 :data:`ACCOUNT_NAME_RETRY_SECONDS`。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from astrbot.api import logger

from ..common import GROUP_MESSAGE, Target, parse_target
from ..conversation.messages import SELF_NAME

ACCOUNT_NAME_TIMEOUT = 5.0
"""单次 OneBot 查询的超时；超时就回退「我」，不拖住消息记录。"""

ACCOUNT_NAME_RETRY_SECONDS = 600.0
"""取不到名字（异常 / 空昵称 / 昵称等于 QQ 号）后的重试间隔。"""

GROUP_CARD_TTL_SECONDS = 1800.0
"""群名片有效期：群名片随群变化，过期后重新查询。"""

AIOCQHTTP_PLATFORM = "aiocqhttp"


def as_self_id(value: Any) -> str:
    """只接受字符串 / 整数形态的 ID。

    aiocqhttp 等适配器的 ``__getattr__`` 会把**任何未知属性**变成 API 调用的偏函数
    （``partial(call_action, 'self_id')``），``getattr(bot, "self_id")`` 拿到的就是它。
    这种值 truthy 且能 str() 成一段无意义的 repr，必须过滤掉，否则会被当成 ID 缓存。
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return ""
    text = str(value).strip()
    return text if text else ""


def attr_self_id(adapter: Any) -> str:
    """从适配器属性里取机器人自身 ID；只读属性，不发网络请求。"""
    bot = getattr(adapter, "bot", None)
    for attr in ("self_id", "bot_self_id"):
        value = as_self_id(getattr(adapter, attr, None)) or as_self_id(
            getattr(bot, attr, None)
        )
        if value:
            return value
    client = getattr(adapter, "client", None)
    return as_self_id(getattr(getattr(client, "user", None), "id", None)) or as_self_id(
        getattr(client, "bot_id", None)
    )


async def adapter_self_id(
    adapter: Any,
    *,
    timeout: float = ACCOUNT_NAME_TIMEOUT,
) -> tuple[str, bool]:
    """尽力取平台适配器的机器人自身 ID。

    返回 ``(自身 ID, 能否据此判定连接状态)``：取到 ID 视为在线；
    适配器没有可查询的接口时第二个值为 ``False``，表示「无法判定」。
    """
    value = attr_self_id(adapter)
    if value:
        return value, True

    bot = getattr(adapter, "bot", None)
    call = getattr(bot, "call_action", None)
    if call is None:
        return "", False
    meta = adapter.meta()
    try:
        info = await asyncio.wait_for(call("get_login_info"), timeout)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "[Ambient] 适配器查询自身 ID 失败（%s）：%s: %s",
            getattr(meta, "id", "") or getattr(meta, "name", ""),
            type(exc).__name__,
            exc,
        )
        return "", True
    user_id = info.get("user_id") if isinstance(info, dict) else None
    return (str(user_id), True) if user_id else ("", True)


class BotAccountName:
    """按会话解析机器人自己的显示名，进程内缓存。

    只支持 aiocqhttp：别的平台一律返回空串，调用方回退「我」。缓存分两层：
    账号昵称按平台实例永久缓存，群名片按 ``(平台, 群)`` 带 TTL 缓存。
    """

    def __init__(
        self,
        context: Any,
        *,
        timeout: float = ACCOUNT_NAME_TIMEOUT,
        retry_seconds: float = ACCOUNT_NAME_RETRY_SECONDS,
        group_card_ttl: float = GROUP_CARD_TTL_SECONDS,
    ) -> None:
        self._context = context
        self._timeout = timeout
        self._retry_seconds = retry_seconds
        self._group_card_ttl = group_card_ttl
        self._names: dict[str, str] = {}
        self._ids: dict[str, str] = {}
        self._retry_at: dict[str, float] = {}
        self._cards: dict[tuple[str, str], tuple[float, str]] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed = False

    # ------------------------------------------------------------------ 读取
    def cached(self, umo: str) -> str:
        """同步读已解析的名字；没有时返回空串。"""
        target = parse_target(umo)
        if target is None:
            return ""
        if target.is_group:
            card = self._fresh_card(target)
            if card:
                return card
        return self._names.get(target.platform_id, "")

    def identity(self, umo: str, *, use_name: bool = True) -> dict[str, str]:
        """机器人自己的署名信息：``{"name": ..., "id": ...}``。

        名字取已解析结果，没有或 ``use_name`` 为假时用「我」；ID 取消息事件里
        见过的自身 ID，没有就省略，不编造。
        """
        target = parse_target(umo)
        name = (self.cached(umo) if use_name else "") or SELF_NAME
        identity = {"name": name}
        if target is not None:
            self_id = self._ids.get(target.platform_id)
            if self_id:
                identity["id"] = self_id
        return identity

    def prime(self, umo: str, self_id: str = "") -> None:
        """后台解析名字；已缓存 / 在途 / 冷却中时什么都不做。

        ``self_id`` 来自消息事件：aiocqhttp 的 bot 对象上没有可靠的账号 ID 属性，
        多账号时查询必须指定 ``self_id`` 才能路由到对应连接。
        """
        target = parse_target(umo)
        if target is None:
            return
        if self_id:
            self._ids[target.platform_id] = self_id
        if self._closed or self.cached(umo):
            return
        if self._cooling(target.platform_id):
            return
        if umo in self._pending:
            return
        task = asyncio.create_task(self._prime(umo, self_id))
        self._pending[umo] = task
        task.add_done_callback(lambda _task, key=umo: self._pending.pop(key, None))

    async def name_for(
        self, umo: str, *, refresh: bool = False, self_id: str = ""
    ) -> str:
        """解析并返回名字；拿不到时返回空串，不抛异常。"""
        target = parse_target(umo)
        if target is None or self._closed:
            return ""
        if self_id:
            self._ids[target.platform_id] = self_id
        if not refresh:
            cached = self.cached(umo)
            if cached or self._cooling(target.platform_id):
                return cached
        lock = self._locks.setdefault(target.platform_id, asyncio.Lock())
        async with lock:
            if not refresh:
                cached = self.cached(umo)
                if cached or self._cooling(target.platform_id):
                    return cached
            return await self._resolve(target)

    async def close(self) -> None:
        """停用并取消在途查询（插件重载 / 卸载）。"""
        self._closed = True
        tasks = list(self._pending.values())
        self._pending.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._locks.clear()

    def forget(self) -> None:
        """清空缓存，供插件重新初始化时使用。"""
        self._closed = False
        self._names.clear()
        self._ids.clear()
        self._retry_at.clear()
        self._cards.clear()
        self._locks.clear()

    # ------------------------------------------------------------------ 内部
    async def _prime(self, umo: str, self_id: str) -> None:
        try:
            await self.name_for(umo, self_id=self_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 解析机器人昵称失败（%s）：%s", umo, exc)

    async def _resolve(self, target: Target) -> str:
        adapter = self._find_adapter(target.platform_id)
        if adapter is None:
            return ""
        name = await self._fetch(adapter, target.platform_id)
        if not name:
            self._retry_at[target.platform_id] = time.monotonic() + self._retry_seconds
            return ""
        self._names[target.platform_id] = name
        self._retry_at.pop(target.platform_id, None)
        if target.is_group:
            card = await self._fetch_card(adapter, target)
            if card:
                self._cards[(target.platform_id, target.session_id)] = (
                    time.monotonic() + self._group_card_ttl,
                    card,
                )
        return self.cached(target.umo) or name

    async def _fetch(self, adapter: Any, platform_id: str) -> str:
        bot = getattr(adapter, "bot", None)
        call = getattr(bot, "call_action", None)
        if call is None:
            return ""
        self_id = self._ids.get(platform_id) or attr_self_id(adapter)
        params = {"self_id": int(self_id)} if self_id.isdigit() else {}
        try:
            info = await asyncio.wait_for(
                call("get_login_info", **params), self._timeout
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[Ambient] 查询机器人昵称失败（%s）：%s: %s",
                platform_id,
                type(exc).__name__,
                exc,
            )
            return ""
        if not isinstance(info, dict):
            return ""
        user_id = str(info.get("user_id") or "").strip()
        name = str(info.get("nickname") or "").strip()
        if user_id:
            self._ids[platform_id] = user_id
        if not name or name in {self_id, user_id}:
            # 空昵称或被实现回填成 QQ 号，都不能当可读名字用。
            return ""
        return name

    async def _fetch_card(self, adapter: Any, target: Target) -> str:
        if target.message_type != GROUP_MESSAGE:
            return ""
        self_id = self._ids.get(target.platform_id) or attr_self_id(adapter)
        if not (target.session_id.isdigit() and self_id.isdigit()):
            return ""
        bot = getattr(adapter, "bot", None)
        call = getattr(bot, "call_action", None)
        if call is None:
            return ""
        params = {"self_id": int(self_id)}
        try:
            info = await asyncio.wait_for(
                call(
                    "get_group_member_info",
                    group_id=int(target.session_id),
                    user_id=int(self_id),
                    **params,
                ),
                self._timeout,
            )
        except Exception as exc:  # noqa: BLE001
            # 群名片只是锦上添花：拿不到就继续用账号昵称。
            logger.debug(
                "[Ambient] 查询机器人群名片失败（%s）：%s: %s",
                target.session_id,
                type(exc).__name__,
                exc,
            )
            return ""
        if not isinstance(info, dict):
            return ""
        return str(info.get("card") or "").strip()

    def _fresh_card(self, target: Target) -> str:
        entry = self._cards.get((target.platform_id, target.session_id))
        if entry is None:
            return ""
        expires_at, card = entry
        if expires_at <= time.monotonic():
            self._cards.pop((target.platform_id, target.session_id), None)
            return ""
        return card

    def _cooling(self, platform_id: str) -> bool:
        retry_at = self._retry_at.get(platform_id)
        return retry_at is not None and retry_at > time.monotonic()

    def _find_adapter(self, platform_id: str) -> Any:
        manager = getattr(self._context, "platform_manager", None)
        if manager is None:
            return None
        try:
            insts = manager.get_insts()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 读取平台实例失败：%s", exc)
            return None
        candidates = []
        for adapter in insts:
            meta = getattr(adapter, "meta", None)
            if meta is None:
                continue
            try:
                info = meta()
            except Exception:  # noqa: BLE001
                continue
            if str(getattr(info, "name", "")) != AIOCQHTTP_PLATFORM:
                continue
            if str(getattr(info, "id", "")) == platform_id:
                return adapter
            candidates.append(adapter)
        # 配置 id 与 umo 前缀偶尔不一致（如 old 版本迁移），单实例时直接用它。
        return candidates[0] if len(candidates) == 1 else None
