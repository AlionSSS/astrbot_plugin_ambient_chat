"""发言意向机制：按概率放行 / 决策模型判定。

判定失败一律 fail-closed（``docs/adr/0005``）。
"""

from __future__ import annotations

import inspect
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import tzinfo as _tzinfo
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.provider import DecisionError

from ..common import AMBIENT_HINT_PREFIX, INTENT_DECISION, parse_target
from ..conversation.context_buffer import ContextBuffer
from ..conversation.messages import (
    SELF_NAME,
    WEEKDAY_ZH,
    ChatMessage,
    format_time,
)
from ..rules.models import Rule
from .decision import DecisionContentRejected, DecisionRouter, NoulPrompt

if TYPE_CHECKING:
    from astrbot.api.star import Context

MODE_CONTINUE = "continue"
MODE_NEW_TOPIC = "new_topic"
MODE_DIRECT = "direct"

FAILURE_CONTENT_REJECTED = "content_rejected"
FAILURE_DECISION = "decision_failed"
"""判定失败分类：页面按它标红并给出提示，不把「失败」混进普通「跳过」。"""

REASON_CONTENT_REJECTED = (
    "决策失败：决策平台内容审核未通过（会话含敏感内容），本次已跳过"
)

REASON_DETAIL_LIMIT = 160
"""失败原因在页面上最多显示的字符数，完整信息仍写进 WARNING 与调试日志。"""

_TIME_FIT_TIP = (
    "请让这句话贴合当下的时间与对话间隔：问候、吃饭、睡觉、上下班这类内容要和现在"
    "对得上，也不要接一个已经过去很久的话头。"
)
"""三条提示共用的时间要求。

具体的时间事实由 ``<system_reminder>`` 给出（当前时间 + 距最后一条消息多久），
这里只说「拿它做什么」：模型往往照抄「早上好」这类模板问候，不写这句会把
凌晨的群也当成上午。
"""

HINT_CONTINUE = (
    f"{AMBIENT_HINT_PREFIX}，但你正好想接一句。请顺着最近的对话，"
    "用你自己的口吻自然地接一句。"
    f"{_TIME_FIT_TIP}"
    "保持简短，不要复述别人说过的话，也不要提到这条提示。）"
)
HINT_NEW_TOPIC = (
    f"{AMBIENT_HINT_PREFIX}，但你有点想开个新话题。请用你自己的口吻自然地起个头。"
    f"{_TIME_FIT_TIP}"
    "保持简短，不要说教，也不要提到这条提示。）"
)
HINT_DIRECT = (
    f"{AMBIENT_HINT_PREFIX}，但你正好想说点什么。你可以顺着最近的对话，用你自己的口吻自然地"
    "接一句；也可以另起一个你觉得有意思的新话题。"
    f"{_TIME_FIT_TIP}"
    "保持简短，不要复述别人说过的话，不要说教，也不要提到这条提示。）"
)

_HINT_TEXT = {
    MODE_CONTINUE: HINT_CONTINUE,
    MODE_NEW_TOPIC: HINT_NEW_TOPIC,
    MODE_DIRECT: HINT_DIRECT,
}

DECISION_STATE_CHAR_BUDGET = 6500
"""判定请求里 ``state``（conversation 消息条目 + persona）的字符预算。

``intern-decision-4b`` 的上下文是 8K。实测（书生端砚，中文群聊）真实请求的
token/字符比约 **0.72**：3703 字符的请求体对应 2686 input tokens。按 ADR-0018
最初的「1 字 ≈ 1 token」最坏假设，6500 字符 + questions（约 380 字符）仍低于
8192；按实测比例则只用掉约 61% 的窗口（ADR-0023）。

成本按条目序列化后的 JSON 文本计，key 与引号的开销已算在内。
"""

MAX_PERSONA_CHARS = 2000
"""人格最多保留的字符数，超出截断尾部（保留开头的身份描述）。"""

MIN_CONVERSATION_CHARS = 800
"""即使人格很长，也给最近对话留出的最小字符数。"""

NEUTRAL_PERSONA_PROMPT = (
    "你是一个普通的对话参与者，没有特别的人设或身份；以自然、克制的口吻参与对话。"
)
"""人格为空时的中性占位，让指令里不必再写「若为空则……」的分支。"""


@dataclass(slots=True)
class IntentResult:
    """发言意向判定结果。"""

    speak: bool
    mode: str = MODE_CONTINUE
    reason: str = ""
    failure: str = ""
    """判定失败分类（见 :data:`FAILURE_DECISION` 等）；未失败时为空串。"""
    scores: tuple[float, float] | None = None
    """判定给出的原始概率 ``(感兴趣, 新话题)``，用于累积关注度（ADR-0034）。

    只有决策模型判定成功时才有；按概率放行与判定失败都为空，后者不能当作
    「不感兴趣」记账，否则一次网络抖动就会拉低这个会话的关注度。
    """
    fresh_ratio: float = 0.0
    """本次判定里「没记过账」的消息占比，用来折算关注度的观测权重（ADR-0035）。

    比例按**真正进入请求**的条数算：被字符预算裁掉的旧消息没有参与判定，不能进
    分母。为 0 表示这段内容已经记过账，重复判定不应再改变关注度。缺省给 0 而不是
    1：漏传这份份量时宁可不记账，也不要按整份证据多记一次。
    """

    @property
    def hint(self) -> str:
        return _HINT_TEXT.get(self.mode, HINT_CONTINUE)


class IntentJudge:
    """「思考要不要发言」：给出这一次是否发言的结论。"""

    def __init__(
        self,
        context: Context,
        router: DecisionRouter,
        *,
        identity: Callable[[str], Mapping[str, str]] | None = None,
    ) -> None:
        self._context = context
        self._router = router
        self._identity = identity
        """机器人自身的署名信息；缺省时只发「我」，测试与旧调用方不必提供。"""

    async def judge(
        self,
        rule: Rule,
        umo: str,
        buffer: ContextBuffer,
        now: datetime,
    ) -> IntentResult:
        if rule.intent.mode != INTENT_DECISION:
            return self._by_probability(rule.intent.pass_probability)

        target = parse_target(umo)
        messages = buffer.recent(umo, rule.context_turns)
        if not messages:
            return IntentResult(False, MODE_CONTINUE, "上下文为空")

        persona_prompt = await self.persona_prompt(umo) if rule.inject_persona else ""
        # 人格为空（未配置，或规则关闭了注入）时用中性占位，指令保持无条件。
        persona_prompt = persona_prompt or NEUTRAL_PERSONA_PROMPT
        # 决策模型上下文有限（如 intern-decision-4b 只有 8K），超了会直接报错，
        # 因此在插件侧按字符预算裁剪：先砍最旧的对话，再压人格尾部。
        conversation, persona_prompt = _fit_conversation(
            messages,
            now.tzinfo or timezone.utc,
            persona_prompt,
            with_id=bool(target and target.is_group),
        )
        if not conversation:
            return IntentResult(False, MODE_CONTINUE, "上下文为空")
        # 真正送进请求的是裁剪后的尾部：比例必须在同一段上算，否则被裁掉的旧消息
        # 会稀释「新增内容」，让重复判定仍然按大半份证据入账（ADR-0035）。
        used = messages[-len(conversation) :]
        fresh_ratio = _fresh_ratio(buffer, umo, used)
        prompts = {
            "interested": NoulPrompt(
                instructions=_build_instructions("interested"),
                when_true=(
                    "你对当前会话有话可说，想自然地接一句，而且此刻接话不突兀。"
                ),
                when_false="你无话可说、不想接话。",
            ),
            "want_new_topic": NoulPrompt(
                instructions=_build_instructions("new_topic"),
                when_true=("当前会话已经冷场，或当前适合开启的新话题，或你有新想法"),
                when_false=("没必要开新话题。"),
            ),
        }
        state: dict[str, Any] = {}
        # 人格是「我是谁」，属于 state 而不是问题本身；不给时不下发该键。
        if persona_prompt:
            state["persona"] = persona_prompt
        # 署名同理：明确告诉模型「你自己叫什么」，避免把账号昵称当成别的群友。
        state["self"] = _identity_entry(self._identity, umo)
        state["now"] = _format_now(now)
        state["scene"] = await self._scene(umo, messages, buffer)
        state["conversation"] = conversation

        try:
            probabilities = await self._router.probabilities(
                rule.intent.decision_provider_id or None,
                prompts,
                state,
            )
        except DecisionContentRejected as exc:
            # 请求格式没问题，是内容被平台判为敏感：页面要提示改内容，而不是改配置。
            logger.warning("[Ambient] 决策模型内容审核未通过，跳过本次发言：%s", exc)
            return IntentResult(
                False,
                MODE_CONTINUE,
                REASON_CONTENT_REJECTED,
                FAILURE_CONTENT_REJECTED,
            )
        except DecisionError as exc:
            logger.warning("[Ambient] 决策模型不可用，跳过本次发言：%s", exc)
            return IntentResult(
                False,
                MODE_CONTINUE,
                f"决策失败：{_brief(exc)}",
                FAILURE_DECISION,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Ambient] 决策调用异常，跳过本次发言：%s", exc)
            return IntentResult(
                False,
                MODE_CONTINUE,
                f"决策异常：{_brief(exc)}",
                FAILURE_DECISION,
            )

        # 判定成功才把这段内容记为已记账：失败时下次重试仍算新内容，
        # 一份证据不会因为一次抖动被永久吞掉（ADR-0035）。
        buffer.mark_judged(umo, used)
        threshold = rule.intent.threshold
        interested = probabilities.get("interested", 0.0)
        new_topic = probabilities.get("want_new_topic", 0.0)
        # 运行管理页的「上次结果」只展示这一个字符串：结论在前，括号里固定
        # 带上两个判定值，落选的那项也能看出打了多少分。
        scores = f"感兴趣 {interested:.2f} / 新话题 {new_topic:.2f}"
        metric = (interested, new_topic)
        if interested >= threshold:
            return IntentResult(
                True,
                MODE_CONTINUE,
                f"感兴趣（{scores}）",
                scores=metric,
                fresh_ratio=fresh_ratio,
            )
        if new_topic >= threshold:
            return IntentResult(
                True,
                MODE_NEW_TOPIC,
                f"新话题（{scores}）",
                scores=metric,
                fresh_ratio=fresh_ratio,
            )
        return IntentResult(
            False,
            MODE_CONTINUE,
            f"不感兴趣（{scores}）",
            scores=metric,
            fresh_ratio=fresh_ratio,
        )

    @staticmethod
    def _by_probability(probability: float) -> IntentResult:
        """按概率决定是否放行：``1.0`` 时等价于旧版「直接放行」，不做随机。"""
        if probability >= 1.0:
            return IntentResult(True, MODE_DIRECT, "直接放行")
        percent = f"{probability * 100:.0f}%"
        if probability > 0.0 and random.random() < probability:
            return IntentResult(True, MODE_DIRECT, f"按概率放行（{percent}）")
        return IntentResult(False, MODE_CONTINUE, f"按概率未放行（{percent}）")

    async def _scene(
        self,
        umo: str,
        messages: list[ChatMessage],
        buffer: ContextBuffer,
    ) -> dict[str, Any]:
        """场景：私聊给对方昵称，群聊给群名；两者都只描述判定用得上的信息。"""
        target = parse_target(umo)
        if target is None:
            return {"type": "unknown"}
        if not target.is_group:
            scene: dict[str, Any] = {"type": "private"}
            peer = ContextBuffer.last_peer_name(messages)
            if peer:
                scene["peer_name"] = peer
            return scene
        scene: dict[str, Any] = {"type": "group"}
        name = await self._group_name(umo, buffer)
        if name:
            scene["name"] = name
        return scene

    async def _group_name(self, umo: str, buffer: ContextBuffer) -> str:
        """群名称：优先运行期缓存，其次核心的会话别名表（用户别名 / 自动识别名）。"""
        cached = buffer.session_name(umo)
        if cached:
            return cached
        getter = getattr(self._context, "get_db", None)
        if getter is None:
            return ""
        try:
            alias = await getter().get_umo_alias(umo)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 读取群名称失败：%s", exc)
            return ""
        name = (
            str(getattr(alias, "user_alias", "") or "").strip()
            or str(getattr(alias, "auto_name", "") or "").strip()
        )
        if name:
            buffer.set_session_name(umo, name)
        return name

    async def persona_prompt(self, umo: str) -> str:
        """读取该会话**当前生效**的人格提示词。

        优先走核心的 ``resolve_selected_persona``：它会叠加会话自身的
        ``persona_id``（用户用 /persona 切换过）、session 级强制人格与平台覆盖，
        与正常对话用的人格一致。旧版核心没有该方法时退回默认人格。
        读不到、或会话明确指定「不使用人格」时返回空串。
        """
        manager = getattr(self._context, "persona_manager", None)
        if manager is None:
            return ""
        resolved, persona = await self._selected_persona(manager, umo)
        if not resolved:
            persona = await self._default_persona(manager, umo)
        return _persona_text(persona)

    async def _selected_persona(self, manager: Any, umo: str) -> tuple[bool, Any]:
        """按会话解析人格，返回 ``(是否解析成功, 人格对象)``。"""
        resolver = getattr(manager, "resolve_selected_persona", None)
        if resolver is None:
            return False, None
        conversation = await self._current_conversation(umo)
        try:
            _, persona, _, _ = await resolver(
                umo=umo,
                conversation_persona_id=getattr(conversation, "persona_id", None),
                platform_name=self._platform_name(umo),
                provider_settings=self._provider_settings(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 解析会话人格失败，回退默认人格：%s", exc)
            return False, None
        return True, persona

    async def _default_persona(self, manager: Any, umo: str) -> Any:
        """兼容旧版核心的默认人格接口（可能是同步或异步）。"""
        getter = getattr(manager, "get_default_persona_v3", None)
        if getter is None:
            return None
        try:
            persona = getter(umo)
            if inspect.isawaitable(persona):
                persona = await persona
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 读取默认人格失败：%s", exc)
            return None
        return persona

    async def _current_conversation(self, umo: str) -> Any:
        """取该会话当前对话对象，用来读取它的 ``persona_id``。"""
        manager = getattr(self._context, "conversation_manager", None)
        if manager is None:
            return None
        try:
            conversation_id = await manager.get_curr_conversation_id(umo)
            if not conversation_id:
                return None
            return await manager.get_conversation(umo, conversation_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[Ambient] 读取当前会话失败：%s", exc)
            return None

    def _platform_name(self, umo: str) -> str:
        """平台类型名（核心算 webchat 特殊默认人格时用）。"""
        target = parse_target(umo)
        if target is None:
            return ""
        try:
            adapters = self._context.platform_manager.get_insts()
        except Exception:  # noqa: BLE001
            return ""
        for adapter in adapters:
            meta = adapter.meta()
            if meta.id == target.platform_id or meta.name == target.platform_id:
                return str(getattr(meta, "name", "") or "")
        return ""

    def _provider_settings(self) -> dict[str, Any]:
        try:
            settings = self._context.get_config().get("provider_settings", {})
        except Exception:  # noqa: BLE001
            return {}
        return settings if isinstance(settings, dict) else {}


def _persona_text(persona: Any) -> str:
    """从人格对象里取提示词；类型异常时记警告并忽略，避免静默丢人格。"""
    if persona is None:
        return ""
    if isinstance(persona, dict):
        text = persona.get("prompt")
    else:
        text = getattr(persona, "prompt", None)
    if text is None:
        return ""
    if not isinstance(text, str):
        logger.warning(
            "[Ambient] 人格提示词类型异常（%s），本次按「无人格」处理。",
            type(text).__name__,
        )
        return ""
    return text.strip()


def _identity_entry(
    provider: Callable[[str], Mapping[str, str]] | None,
    umo: str,
) -> dict[str, str]:
    """``state.self``：机器人自己的名字与可选 ID；取不到名字时用「我」。"""
    if provider is None:
        return {"name": SELF_NAME}
    try:
        data = provider(umo)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[Ambient] 读取机器人署名失败：%s", exc)
        return {"name": SELF_NAME}
    name = str(data.get("name") or "").strip() or SELF_NAME
    entry = {"name": name}
    self_id = str(data.get("id") or "").strip()
    if self_id:
        entry["id"] = self_id
    return entry


def _format_now(now: datetime) -> dict[str, str]:
    """当前时间：``{"date_time": "2026-11-04 05:40:00", "weekday": "周三"}``，不带时区。"""
    return {
        "date_time": f"{now:%Y-%m-%d %H:%M:%S}",
        "weekday": WEEKDAY_ZH[now.weekday()],
    }


def _brief(exc: BaseException) -> str:
    """异常原文压成一行：页面那格只放得下短句，完整信息仍进日志。"""
    text = " ".join(str(exc).split())
    if len(text) <= REASON_DETAIL_LIMIT:
        return text
    return f"{text[: REASON_DETAIL_LIMIT - 1]}…"


def _fresh_ratio(buffer: ContextBuffer, umo: str, used: Sequence[ChatMessage]) -> float:
    """本次请求的消息里没记过账的占比（ADR-0035）。

    静默会话每次触发都会把同一段内容再判一遍，若按一次完整观测记账，关注度就变成
    「决策次数」的函数而不是「内容」的函数。折算后等价于「每攒够约一个窗口的新
    消息才记一次账」，关注度的变化速度与触发间隔解耦。
    """
    if not used:
        return 0.0
    return buffer.fresh_count(umo, used) / len(used)


def _message_entry(
    message: ChatMessage,
    tz: _tzinfo,
    *,
    with_id: bool,
) -> dict[str, Any]:
    """一条消息的结构化形态；时间未知时 ``time`` 为 None，不编造。

    发送者拆成 ``sender_name`` / ``sender_id``（``with_id`` 为真才带 ID），
    正文按形态给出 ``type``（``text`` / ``image`` / ``audio`` / ``video`` /
    ``file`` / ``face`` 等，见 :data:`MESSAGE_KINDS`）。``content`` 只放真正的
    文字，没有文字就整个省略；多附件时另给 ``attachments``。@ 了机器人 / 引用了
    机器人的消息会在条目上带 ``mentions_me`` / ``replies_me``（只在这两种关系成立
    时出现）。戳一戳不带可读正文，改为 ``target`` 对象（``name`` / ``id`` / ``is_self``）。

    为什么这么拆：判定模型是 4B 小模型，`昵称#ID` 挤在一个字符串里时它既不做
    字段对齐，也会把「戳一戳」整条当噪声；给 ``type`` + ``target.name`` 之后
    方向判定才稳定，见 ``scripts/poke_probe.py`` 的对照实验。
    """
    entry: dict[str, Any] = {
        "time": format_time(message.timestamp, tz),
        "sender_name": message.structured_sender_name(),
    }
    if with_id and message.sender_id:
        entry["sender_id"] = message.sender_id
    if message.is_bot:
        entry["self"] = True
    if message.kind == "poke":
        entry["type"] = "poke"
        target: dict[str, Any] = {}
        if message.target_id:
            target["id"] = message.target_id
        if message.target_name:
            target["name"] = message.target_name
        target["is_self"] = message.target_is_self
        entry["target"] = target
        return entry
    entry["type"] = message.kind
    # 只记「在叫机器人」的两个事实，@别人/引用别人不进条目。
    if message.mentions_me:
        entry["mentions_me"] = True
    if message.replies_me:
        entry["replies_me"] = True
    body = message.body_text()
    if body:
        entry["content"] = body
    if len(message.attachments) > 1:
        # 一条消息可以有多个附件（文字 + 图片 + 文件…），单附件时 type 已说明。
        entry["attachments"] = list(message.attachments)
    if message.kind == "file" and message.file_name:
        entry["filename"] = message.file_name
    return entry


def _entry_cost(entry: dict[str, Any]) -> int:
    """条目在请求体里占的字符数：序列化后的 JSON 文本 + 分隔逗号。"""
    return len(json.dumps(entry, ensure_ascii=False)) + 1


def _fit_conversation(
    messages: list[ChatMessage],
    tz: _tzinfo,
    persona: str,
    *,
    with_id: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """按字符预算裁剪对话与人格，返回 ``(消息条目, 人格)``。

    顺序：先给对话留够最小额度、限制人格长度；再用剩余预算从**最新**往旧保留
    条目（超出时丢最旧的整条）；单条消息仍放不下就只截正文，结构字段不动。
    成本按序列化后的 JSON 长度估算，避免 key 与引号等语法开销被漏算。
    """
    persona = persona.strip()
    persona_limit = min(
        MAX_PERSONA_CHARS,
        DECISION_STATE_CHAR_BUDGET - MIN_CONVERSATION_CHARS,
    )
    if len(persona) > persona_limit:
        persona = _cut(persona, persona_limit)
    remaining = DECISION_STATE_CHAR_BUDGET - len(persona)

    entries = [_message_entry(message, tz, with_id=with_id) for message in messages]
    kept: list[dict[str, Any]] = []
    used = 0
    for entry in reversed(entries):
        cost = _entry_cost(entry)
        if used + cost > remaining:
            break
        kept.append(entry)
        used += cost
    if not kept and entries:
        shrunk = _shrink_entry(entries[-1], remaining)
        if shrunk is not None:
            kept = [shrunk]
    if len(kept) != len(entries):
        logger.debug(
            "[Ambient] 判定请求超出 %d 字符预算，对话 %d→%d 条，人格 %d 字符。",
            DECISION_STATE_CHAR_BUDGET,
            len(entries),
            len(kept),
            len(persona),
        )
    return list(reversed(kept)), persona


def _shrink_entry(entry: dict[str, Any], remaining: int) -> dict[str, Any] | None:
    """单条也放不下时保留正文尾部；连结构字段都放不下则放弃这条。

    戳一戳条目没有正文可截，放不下就整条丢弃，绝不硬塞一个空 ``content``。
    """
    if "content" not in entry:
        return None
    limit = remaining - _entry_cost({**entry, "content": ""})
    if limit < 2:
        return None
    shrunk = dict(entry)
    shrunk["content"] = _cut(str(entry.get("content") or ""), limit, keep="tail")
    return shrunk


def _cut(text: str, limit: int, *, keep: str = "head") -> str:
    """把文本裁到 ``limit`` 字符以内，并加省略号标记。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    if keep == "tail":
        return "…" + text[-(limit - 1) :]
    return text[: limit - 1] + "…"


def _build_instructions(question: str) -> str:
    """构造判断题的指令；人格设定走 ``state.persona``，不混进指令。

    指令里只点了 ``now`` 与 ``time`` 两个字段名的单词，不写 ``state.now``
    这样的路径：判定模型是小模型，靠字段位置和值理解，路径字符串对它没有指代
    作用（同一现象见 ``_message_entry`` 的说明）。点名的目的是减少歧义——会话
    正文里也可能出现「现在」「今天」等词，不指名它会把自己提到的时间当成当前
    时间，也容易只看 ``now`` 而不读逐条消息的 ``time``。
    """
    if question == "interested":
        return (
            "基于你的人设，结合 now 与每条消息的 time，判断："
            "你是否对当前会话内容感兴趣？或是否想自然地接一句话？"
        )
    return (
        "基于你的人设，结合 now 与每条消息的 time，判断："
        "针对当前会话内容，你是否应主动开启一个新话题？"
    )
