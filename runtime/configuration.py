"""已保存配置的运行应用：失败保留配置，重试只重新应用当前配置。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from astrbot.api import logger

from ..rules.store import RuleStore


class ConfigurationApplication:
    """在运行应用的 seam 记录结果，不串行等待无关绑定的收尾。

    RuleStore 负责同步提交；本 module 只处理提交之后的异步应用。
    重叠应用各自完成，成功只清除本次开始前已知的错误，不能覆盖后来发生的失败。
    """

    def __init__(self, store: RuleStore, apply: Callable[[], Awaitable[None]]) -> None:
        self._store = store
        self._apply = apply
        self._sequence = 0
        self._pending: set[int] = set()
        self._errors: dict[int, str] = {}

    def snapshot(self) -> dict[str, Any]:
        errors = list(self._errors.values())
        return {
            "revision": self._store.revision,
            "state": "failed" if errors else "applying" if self._pending else "applied",
            "pending": len(self._pending),
            "error": errors[-1] if errors else "",
        }

    async def apply(self) -> dict[str, Any]:
        self._sequence += 1
        ticket = self._sequence
        known_errors = set(self._errors)
        self._pending.add(ticket)
        try:
            await self._apply()
        except asyncio.CancelledError:
            self._errors[ticket] = "运行应用被中断，请重新应用已保存配置"
            raise
        except Exception as exc:
            logger.exception("[Ambient] 配置已保存，运行应用失败")
            self._errors[ticket] = str(exc) or type(exc).__name__
        else:
            for previous in known_errors:
                self._errors.pop(previous, None)
        finally:
            self._pending.discard(ticket)
        return self.snapshot()
