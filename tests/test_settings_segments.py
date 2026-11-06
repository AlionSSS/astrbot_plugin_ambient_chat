"""更多设置里的分段拟人发言参数：默认值、往返与保存校验。"""

import pytest
from astrbot_plugin_ambient_chat.rules.models import RuleError, Settings
from astrbot_plugin_ambient_chat.speech.segments import DEFAULT_SEGMENT_PATTERN


def test_segment_defaults_match_core():
    settings = Settings()
    assert settings.segmented_enabled is True
    assert (settings.segmented_interval_min, settings.segmented_interval_max) == (1.5, 3.5)
    assert settings.segmented_threshold == 150
    assert settings.segmented_pattern == DEFAULT_SEGMENT_PATTERN


def test_segment_settings_round_trip():
    saved = Settings.parse(
        {
            "segmented_interval_min": 0.5,
            "segmented_interval_max": 2,
            "segmented_threshold": 80,
            "segmented_pattern": r"[^。]+。?",
        }
    )
    assert Settings.parse(saved.to_dict()) == saved


def test_empty_pattern_is_rejected():
    with pytest.raises(RuleError, match="不能为空"):
        Settings.parse({"segmented_pattern": "   "})


def test_invalid_pattern_is_rejected():
    with pytest.raises(RuleError, match="正则无效"):
        Settings.parse({"segmented_pattern": "("})


def test_interval_order_is_enforced():
    with pytest.raises(RuleError, match="最大值不能小于最小值"):
        Settings.parse({"segmented_interval_min": 3, "segmented_interval_max": 1})


def test_threshold_out_of_range_is_rejected():
    with pytest.raises(RuleError, match="分段字数阈值"):
        Settings.parse({"segmented_threshold": 0})
