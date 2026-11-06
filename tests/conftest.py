import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[1]
# 插件模块与单文件 SDK 统一按插件包路径导入。
sys.path.insert(0, str(_PLUGIN_DIR.parent))
