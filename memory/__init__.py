"""跨插件召回长期记忆（当前实现：LivingMemory）。

``LivingMemoryRecall`` 负责取数与语义（作用域、人格、适配器边界），
``render_memory_block`` 负责把结果拼成提示块，``collect_recent_senders`` 负责在
``user``（按人共享）作用域下决定「群聊该查谁」，三者可以分别使用。
"""

from .livingmemory import (
    LIVINGMEMORY_PLUGIN_NAME,
    RECALL_TIMEOUT_SECONDS,
    LivingMemoryRecall,
    MemoryItem,
    SenderRef,
)
from .render import MEMORY_BLOCK_CLOSE, MEMORY_BLOCK_OPEN, render_memory_block
from .senders import DEFAULT_FANOUT, collect_recent_senders, senders_for_recall

__all__ = [
    "DEFAULT_FANOUT",
    "LIVINGMEMORY_PLUGIN_NAME",
    "LivingMemoryRecall",
    "MEMORY_BLOCK_CLOSE",
    "MEMORY_BLOCK_OPEN",
    "MemoryItem",
    "RECALL_TIMEOUT_SECONDS",
    "SenderRef",
    "collect_recent_senders",
    "render_memory_block",
    "senders_for_recall",
]
