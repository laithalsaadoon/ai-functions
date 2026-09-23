"""Per-cycle runtime context the runtime passes to ``Thread.execute``."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..protocols import Coordinator, OnEventCallback, OnInterruptCallback

from .ids import ThreadId


@dataclass
class ThreadContext:
    """Runtime context the worker builds fresh for every cycle.

    Invariants:
        - Every field is populated by the worker; executors may rely on
          non-None values.
        - A ``ThreadContext`` instance is never reused across cycles.
    """

    thread_id: ThreadId
    """Id of the thread this cycle runs on."""

    coordinator: Coordinator
    """The coordinator this thread is registered with. Threads use it to
    spawn children, inject messages into peers, read events, and
    introspect the thread registry."""

    on_event: OnEventCallback
    """Sink bound to ``Coordinator.append_event`` for content, tool, and usage events."""

    on_interrupt: OnInterruptCallback
    """Handler for a batch of executor-raised interrupts; awaited inline."""

    pause_signal: asyncio.Event
    """Rate-limit / manual-pause signal the executor MUST await at every model-call boundary."""

    cancel_signal: asyncio.Event
    """Cooperative-cancel signal the executor MUST check at every model-call boundary."""

    parent_id: ThreadId | None = None
    """Id of the parent thread, if this thread was spawned with a ``parent_id``."""

    metadata: dict[str, object] = field(default_factory=lambda: dict[str, object]())
    """Application metadata (priority, etc.)."""


@dataclass(frozen=True)
class ThreadScope:
    """The ambient ``(coordinator, thread_id)`` of the running thread.

    ``thread_id`` is ``None`` in a coordinator-only scope
    (:func:`ai_functions.scope`): the coordinator to run on is bound, with no
    thread to attribute against or parent to.
    """

    coordinator: Coordinator
    thread_id: ThreadId | None = None


@contextmanager
def thread_scope(coordinator: Coordinator, thread_id: ThreadId | None = None) -> Iterator[ThreadScope]:
    """Bind the ambient thread to ``(coordinator, thread_id)`` for the duration of a block.

    Omitting ``thread_id`` binds the coordinator alone: bare AI-function calls
    in the block run on it, with no thread to parent them to.
    """


def current_thread_scope() -> ThreadScope | None:
    """Return the ambient :class:`ThreadScope`, or ``None`` outside any scope."""

@contextmanager
def no_thread_scope() -> Iterator[None]:
    """Clear the ambient thread scope for the duration of a block.

    Library-internal AI-function calls (memory query/consolidate helpers, the
    optimizer's backward function) run inside this so they never attribute to
    the user's running thread or pollute its event log.
    """
