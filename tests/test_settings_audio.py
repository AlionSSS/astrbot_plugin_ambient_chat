"""「更多设置」里的语音缓存开关：默认关闭，兼容表单字符串。"""

from astrbot_plugin_ambient_chat.rules.models import Settings


def test_cache_audio_defaults_off() -> None:
    assert Settings().cache_audio is False
    assert Settings.parse({}).cache_audio is False


def test_cache_audio_accepts_bool_and_string_and_round_trips() -> None:
    assert Settings.parse({"cache_audio": True}).cache_audio is True
    assert Settings.parse({"cache_audio": "on"}).cache_audio is True
    assert Settings.parse({"cache_audio": "off"}).cache_audio is False
    assert Settings.parse({"cache_audio": True}).to_dict()["cache_audio"] is True
