"""连发无人回应：等待期满只结束这一次退避，而不是永久限速。"""

from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot_plugin_ambient_chat.common import BLOCKER_UNANSWERED
from astrbot_plugin_ambient_chat.conversation.context_buffer import ContextBuffer
from astrbot_plugin_ambient_chat.conversation.messages import ChatMessage
from astrbot_plugin_ambient_chat.rules.models import Rule
from astrbot_plugin_ambient_chat.speech.blockers import BlockerChain

UMO = "test:GroupMessage:1"
TZ = ZoneInfo("UTC")
START = 1_700_000_000.0
WAIT = 30 * 60


def rule(count: int = 3, minutes: int = 30) -> Rule:
    return Rule(
        id="r_test",
        name="连发测试",
        blockers=[{"type": BLOCKER_UNANSWERED, "count": count, "minutes": minutes}],
    )


def blocked(buffer: ContextBuffer, moment: float) -> bool:
    return BlockerChain(buffer).check(
        rule(), UMO, datetime.fromtimestamp(moment, TZ)
    ).blocked


def speak(buffer: ContextBuffer, moment: float, text: str) -> None:
    buffer.append(UMO, ChatMessage("我", text, moment, is_bot=True))


def reply(buffer: ContextBuffer, moment: float) -> None:
    buffer.append(UMO, ChatMessage("某人", "在的", moment, sender_id="10001"))


def test_waiting_expiry_ends_the_backoff_instead_of_throttling_forever():
    buffer = ContextBuffer(maxlen=20)
    for index in range(3):
        speak(buffer, START + index, f"第 {index + 1} 条")
    last = START + 2

    assert blocked(buffer, last + 60)
    assert blocked(buffer, last + WAIT - 1)

    # 等待期满：本次放行，并把连发基线重置到那串没人理的发言。
    assert not blocked(buffer, last + WAIT)
    assert buffer.bot_streak(UMO) == 0

    # 之后要重新连发满 3 条才会再次进入等待，而不是发一条又被拦。
    for index in range(2):
        speak(buffer, last + WAIT + index * 2 + 1, f"新第 {index + 1} 条")
        assert not blocked(buffer, last + WAIT + index * 2 + 2)
    speak(buffer, last + WAIT + 5, "新第 3 条")
    assert blocked(buffer, last + WAIT + 6)


def test_a_reply_lifts_the_block_immediately():
    buffer = ContextBuffer(maxlen=20)
    for index in range(3):
        speak(buffer, START + index, f"第 {index + 1} 条")

    assert blocked(buffer, START + 2 + 60)
    reply(buffer, START + 2 + 120)
    assert not blocked(buffer, START + 2 + 121)
