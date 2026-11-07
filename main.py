"""聊天氛围组：让 AI 像真人一样在群聊/私聊中自主发言。"""

from __future__ import annotations

import time
from typing import Any
from zoneinfo import ZoneInfo

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, File, Image, Plain, Poke, Record, Reply
from astrbot.api.star import Context, Star, register

from .common import PLUGIN_NAME, POKE_SELF_NAME, resolve_timezone
from .conversation.context_buffer import ContextBuffer
from .conversation.messages import (
    COMPONENT_KINDS,
    SELF_NAME,
    ChatMessage,
    strip_system_reminder,
)
from .rules.models import Binding
from .rules.store import RuleStore
from .runtime.account import BotAccountName, adapter_self_id
from .runtime.attention import AttentionStore
from .runtime.binding_lifecycle import BindingLifecycle
from .runtime.configuration import ConfigurationApplication
from .runtime.execution import SpeechExecution
from .runtime.scheduler import AmbientScheduler
from .speech.blockers import BlockerChain
from .speech.decision import DecisionRouter
from .speech.delivery import Delivery
from .speech.intent import IntentJudge
from .webapi import AmbientWebApi

LIVE_PROBE_TIMEOUT_SECONDS = 2.0
LIVE_PROBE_TTL_SECONDS = 30.0
"""WebUI 平台在线状态探测：单实例超时与结果缓存时长。"""

_plugin_ref: AmbientChatPlugin | None = None
"""供唤醒阶段过滤器检查目标接管（装饰器在类定义期生效）。"""


class AmbientSessionFilter(filter.CustomFilter):
    """只让「已启动绑定」覆盖的会话唤醒本插件，缩小副作用范围。"""

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        plugin = _plugin_ref
        return plugin is not None and plugin.lifecycle.accepts(
            event.unified_msg_origin, for_message=True
        )


@register(
    PLUGIN_NAME,
    "AlionSSS",
    "让 AI 在群聊与私聊中像真人一样「在场」：按时机、按兴趣决定要不要接话，偶尔主动开个新话题，而不是只能等被点名。",
    "0.4.5",
)
class AmbientChatPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.store = RuleStore()
        self.attention = AttentionStore()
        global _plugin_ref
        _plugin_ref = self
        self.buffer = ContextBuffer(
            maxlen=self.store.settings.buffer_size, bot_name=self._self_name
        )
        self.blockers = BlockerChain(self.buffer)
        self.router = DecisionRouter(context, self.store)
        self.account = BotAccountName(context)
        self.judge = IntentJudge(context, self.router, identity=self._identity)
        self.delivery = Delivery(
            context,
            self.buffer,
            self_name=self._self_name,
            settings=lambda: self.settings,
        )
        self.lifecycle = BindingLifecycle(self)
        self.execution = SpeechExecution(
            self, accepts=lambda bid, umo: self.lifecycle.accepts(umo, binding_id=bid)
        )
        self.scheduler = AmbientScheduler(self)
        self.configuration = ConfigurationApplication(
            self.store, self.apply_store_changes
        )
        self._live_cache: dict[str, bool | None] | None = None
        self._live_cache_at = 0.0
        self._timezone: ZoneInfo | None = None
        AmbientWebApi(self).register()

    # ------------------------------------------------------------------ 生命周期
    async def initialize(self) -> None:
        """插件激活 / 热重载后调用。"""
        self.store.load(self.timezone)
        self.attention.load()
        self.buffer.reset_media()
        self.account.forget()
        await self.configuration.apply()

    async def terminate(self) -> None:
        await self.lifecycle.close()
        await self.buffer.close()
        await self.router.close()
        await self.account.close()
        await self.attention.close()
        logger.info("[Ambient] 调度已停止。")

    async def apply_store_changes(self) -> None:
        """同步绑定接管、执行、缓冲及调度；应用失败由 configuration 记录。"""
        await self.lifecycle.reconcile()
        # 读盘失败时 store 是空的：照常清理会把全部关注度当成孤儿删掉并落盘，
        # 那是「settings.json 坏了」连累画像一起丢。宁可留着脏数据等修好再清。
        if self.store.load_failed:
            return
        # 目标从所有绑定里消失后，它的关注度也就没有归属了（ADR-0034）。
        self.attention.prune(
            {umo for binding in self.store.bindings.values() for umo in binding.targets}
        )

    async def start_binding(self, binding_id: str) -> Binding:
        return await self.lifecycle.start(binding_id)

    async def stop_binding(self, binding_id: str) -> Binding:
        return await self.lifecycle.stop(binding_id)

    # ------------------------------------------------------------------ 消息入口
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.custom_filter(AmbientSessionFilter)
    async def on_any_message(self, event: AstrMessageEvent):
        """把目标会话的消息记进上下文缓冲。"""
        umo = event.unified_msg_origin
        if not self.lifecycle.accepts(umo, for_message=True):
            return
        self._remember_session_name(event, umo)
        # 后台解析账号名：解析要发 OneBot 请求，不能挡住消息记录。
        self.account.prime(umo, self_id=_event_self_id(event))
        message = self._build_message(event, umo)
        if message is None:
            return
        chain = getattr(event.message_obj, "message", None) or []
        self.lifecycle.record(
            umo,
            message,
            image_components=[item for item in chain if isinstance(item, Image)],
            # 语音原样转存，由缓冲决定是否启用（默认关闭，见 ADR-0031）。
            audio_components=[item for item in chain if isinstance(item, Record)],
        )

    def _remember_session_name(self, event: AstrMessageEvent, umo: str) -> None:
        """记下平台给的群名（私聊没有），供判定请求的 ``scene`` 使用。"""
        group = getattr(getattr(event, "message_obj", None), "group", None)
        name = str(getattr(group, "group_name", "") or "").strip()
        if name:
            self.buffer.set_session_name(umo, name)

    def _build_message(self, event: AstrMessageEvent, umo: str) -> ChatMessage | None:
        """组装待入缓冲的消息，图片任务由上下文缓冲在接纳后管理。"""
        text = strip_system_reminder(event.message_str or "").strip()
        # 正文与附件分开记：一条消息可以既有文字又有图片/语音/文件，两者都不能丢。
        kind, attachments = _message_shape(event, has_text=bool(text))
        if not text and kind == "text" and not attachments:
            return None
        poke = _first_poke(event) if kind == "poke" else None
        sender_id = str(event.get_sender_id() or "")
        self_id = _event_self_id(event)
        chain = getattr(event.message_obj, "message", None) or []
        target_id = str(poke.target_id() or "") if poke is not None else ""
        target_is_self = bool(target_id) and target_id == self_id
        message = ChatMessage(
            sender_name=self._sender_name(event, umo, sender_id),
            text=text,
            timestamp=time.time(),
            is_bot=False,
            sender_id=sender_id,
            kind=kind,
            attachments=attachments,
            file_name=_first_file_name(event) if kind == "file" else "",
            mentions_me=_mentions_self(chain, self_id),
            replies_me=_replies_to_self(chain, self_id),
            target_id=target_id,
            target_name=(
                POKE_SELF_NAME
                if target_is_self
                else self.buffer.known_name(umo, target_id)
            ),
            target_is_self=target_is_self,
        )
        return message

    def _sender_name(self, event: AstrMessageEvent, umo: str, sender_id: str) -> str:
        """发送者昵称；通知类事件（戳一戳等）平台只给数字 ID，用缓冲里的已知昵称补上。"""
        name = (event.get_sender_name() or "").strip()
        if sender_id and (not name or name == sender_id):
            known = self.buffer.known_name(umo, sender_id)
            if known:
                return known
        return name

    # ------------------------------------------------------------------ 发送后钩子
    @filter.after_message_sent()
    async def record_bot_message(self, event: AstrMessageEvent) -> None:
        """记录 bot 自己的发言，用于冷却与未读基线。

        结果链里的语音/图片不能退化成 ``[Record]`` / ``[Image]`` 这类英文占位塞进
        正文：那样决策模型会把「我发了一条语音」读成「我打了一行字」。形态交给
        ``kind`` / ``attachments``，正文只留真正的文字；TTS 语音自带原文
        （``Record.text``）时一并保留，模型至少知道这条语音说了什么。
        """
        if not self.lifecycle.accepts(event.unified_msg_origin, for_message=True):
            return
        result = event.get_result()
        if result is None:
            return
        chain = tuple(getattr(result, "chain", None) or ())
        text = _chain_text(chain)
        if not text:
            # 兜底：没有消息链的结果对象（测试桩、第三方发送路径）只取纯文本。
            try:
                text = result.get_plain_text().strip()
            except Exception:  # noqa: BLE001
                text = ""
        kind, attachments = _chain_shape(chain, has_text=bool(text))
        if not text and kind == "text" and not attachments:
            text = "[非文本消息]"
        self.lifecycle.record(
            event.unified_msg_origin,
            ChatMessage(
                sender_name=self._self_name(event.unified_msg_origin),
                text=text,
                timestamp=time.time(),
                is_bot=True,
                sender_id=_event_self_id(event),
                kind=kind,
                attachments=attachments,
                file_name=_first_file_name_in(chain) if kind == "file" else "",
            ),
        )

    def _self_name(self, umo: str) -> str:
        """机器人发言的署名：解析到账号名时用它，否则统一「我」。"""
        return self.account.cached(umo) or SELF_NAME

    def _identity(self, umo: str) -> dict[str, str]:
        """决策请求里的 ``state.self``：与发言署名共用同一个账号名。"""
        return self.account.identity(umo)

    # ------------------------------------------------------------------ 工具
    @property
    def timezone(self) -> ZoneInfo:
        if self._timezone is None:
            try:
                configured = self.context.get_config().get("timezone")
            except Exception:  # noqa: BLE001
                configured = None
            self._timezone = resolve_timezone(str(configured or ""))
        return self._timezone

    def is_enabled(self) -> bool:
        """「启用自主发言」开关：关闭后仅暂停调度，规则与页面保留。"""
        return bool(self.config.get("enabled", True))

    @property
    def settings(self):
        """运行参数（并发上限 / 间隔 / 缓冲条数），由 WebUI「更多设置」维护。"""
        return self.store.settings

    async def live_platform_status(self) -> dict[str, bool | None]:
        """各平台实例的连接情况：True 可用 / False 不可用 / None 无法判定。

        WebUI 用来提示「绑定的机器人应用选错了」，30 秒内复用上一次探测结果。
        """
        now = time.monotonic()
        if self._live_cache is not None and (
            now - self._live_cache_at < LIVE_PROBE_TTL_SECONDS
        ):
            return self._live_cache
        status: dict[str, bool | None] = {}
        for adapter in self.context.platform_manager.get_insts():
            meta = adapter.meta()
            platform_id = str(getattr(meta, "id", "") or "")
            if not platform_id:
                continue
            self_id, probeable = await adapter_self_id(
                adapter, timeout=LIVE_PROBE_TIMEOUT_SECONDS
            )
            status[platform_id] = True if self_id else (False if probeable else None)
        self._live_cache = status
        self._live_cache_at = now
        return status

    def __repr__(self) -> str:
        return f"<AmbientChatPlugin rules={len(self.store.rules)}>"


def _message_shape(
    event: AstrMessageEvent,
    *,
    has_text: bool,
) -> tuple[str, tuple[str, ...]]:
    """把收到的消息链拆成 ``(主形态, 其余附件)``。"""
    return _chain_shape(getattr(event.message_obj, "message", None), has_text=has_text)


def _chain_shape(
    chain: Any,
    *,
    has_text: bool,
) -> tuple[str, tuple[str, ...]]:
    """把消息链拆成 ``(主形态, 其余附件)``。

    戳一戳单独占主形态（纯动作消息），其余组件按出现顺序记成附件；这样
    「文字 + 图片/语音/文件」不会只剩文字，「表情/骰子」这类也终于能进缓冲。
    收到的消息与 bot 的发送结果共用这套判定，语音在两边都是 ``audio``。
    """
    kinds: list[str] = []
    for component in chain or ():
        name = type(component).__name__.lower()
        # plain 是正文本身；at / reply 属于正文里的行内装饰（提及、引用），
        # 不改变这条消息的形态，否则一条「@某人 你好」会被记成 at 类型。
        if name in ("plain", "at", "reply"):
            continue
        kinds.append(COMPONENT_KINDS.get(name, "event"))
    if not kinds:
        return "text", ()
    if "poke" in kinds:
        others = tuple(kind for kind in kinds if kind != "poke")
        # 带正文时不把整条当成纯戳一戳，否则正文会被丢掉。
        return ("text" if has_text else "poke"), others
    return kinds[0], tuple(kinds)


def _chain_text(chain: Any) -> str:
    """结果链里的正文：``Plain`` 的文字 + 语音组件自带的 TTS 原文。

    ``get_plain_text(with_other_comps_mark=True)`` 会把语音、图片渲染成
    ``[Record]`` / ``[Image]``，那是调试用的英文组件名，当正文存下来只会让决策
    模型以为机器人真打了这几个字。这里只取真文字；同一段文字既发语音又发文字
    （TTS 双发）时只留一份。
    """
    texts: list[str] = []
    for component in chain or ():
        if not isinstance(component, (Plain, Record)):
            continue
        value = strip_system_reminder(getattr(component, "text", "") or "").strip()
        if value and value not in texts:
            texts.append(value)
    return " ".join(texts)


def _mentions_self(chain: tuple[Any, ...] | list[Any], self_id: str) -> bool:
    """消息链里有没有 @ 机器人。核心会把 @机器人 从正文里剔除，只能看组件。"""
    if not self_id:
        return False
    return any(
        isinstance(component, At) and str(getattr(component, "qq", "")) == self_id
        for component in chain
    )


def _replies_to_self(chain: tuple[Any, ...] | list[Any], self_id: str) -> bool:
    """引用的那条消息是不是机器人自己发的。"""
    if not self_id:
        return False
    return any(
        isinstance(component, Reply)
        and str(getattr(component, "sender_id", "") or "") == self_id
        for component in chain
    )


def _first_file_name(event: AstrMessageEvent) -> str:
    """文件消息的文件名；没有则空串。"""
    return _first_file_name_in(getattr(event.message_obj, "message", None))


def _first_file_name_in(chain: Any) -> str:
    """消息链里第一个文件组件的文件名；没有则空串。"""
    for component in chain or ():
        if isinstance(component, File):
            return str(getattr(component, "name", "") or "").strip()
    return ""


def _first_poke(event: AstrMessageEvent) -> Poke | None:
    """消息链里的第一个戳一戳组件；没有则 None。"""
    for component in getattr(event.message_obj, "message", None) or []:
        if isinstance(component, Poke):
            return component
    return None


def _event_self_id(event: AstrMessageEvent) -> str:
    """机器人在该平台的账号 ID；拿不到时返回空串。"""
    return str(getattr(getattr(event, "message_obj", None), "self_id", "") or "")
