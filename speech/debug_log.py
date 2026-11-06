"""默认关闭的决策调试日志；按文件数量与总字节数保留最近记录。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from astrbot.api import logger

MAX_DEBUG_RECORDS = 100
MAX_DEBUG_BYTES = 10 * 1024 * 1024


def to_jsonable(value: Any) -> Any:
    """把 pydantic 模型、dataclass、映射等转成可 json 序列化的结构。"""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(item) for item in value]
    return repr(value)


def write_debug_record(
    kind: str,
    request: Any,
    response: Any = None,
    *,
    directory: Path,
    enabled: bool = False,
    meta: Mapping[str, Any] | None = None,
    elapsed_ms: int | None = None,
    error: str | None = None,
) -> Path | None:
    """启用时记录请求与响应，超过总容量的单条记录不写入。

    写入失败只记一条警告，不影响判断流程。
    """
    if not enabled:
        return None
    record = {
        "time": datetime.now().isoformat(timespec="milliseconds"),
        "kind": kind,
        "meta": to_jsonable(dict(meta or {})),
        "elapsed_ms": elapsed_ms,
        "error": error,
        "request": to_jsonable(request),
        "response": to_jsonable(response),
    }
    path = directory / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex}.json"
    try:
        content = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
        if len(content) > MAX_DEBUG_BYTES:
            logger.warning("[Ambient] 单条决策调试日志超过容量上限，跳过记录。")
            _prune(directory)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        _prune(directory)
    except (OSError, TypeError, ValueError) as exc:  # pragma: no cover - 磁盘异常
        logger.warning("[Ambient] 写入调试文件失败：%s", exc)
        return None
    return path


def _prune(directory: Path) -> None:
    """仅轮转专属目录内的 JSON 记录；保留最新文件。"""
    records = []
    for path in directory.glob("*.json"):
        stat = path.stat()
        records.append((stat.st_mtime_ns, path.name, path, stat.st_size))
    records.sort(reverse=True)
    count = 0
    total = 0
    for _, _, path, size in records:
        if count >= MAX_DEBUG_RECORDS or total + size > MAX_DEBUG_BYTES:
            path.unlink(missing_ok=True)
        else:
            count += 1
            total += size
