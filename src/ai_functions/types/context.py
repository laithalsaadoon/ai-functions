"""Per-cycle runtime context the runtime passes to ``Thread.execute``."""

from __future__ import annotations

import asyncio
import contextvars
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


# ── Ambient thread scope ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThreadScope:
    """The ambient ``(coordinator, thread_id)`` of the running thread.

    A minimal projection of a :class:`ThreadContext`'s routing fields, set by
    the runtime for the duration of each cycle so code the runtime can't hand a
    ``ThreadContext`` — a ``MemoryBackend`` recall, say — can still tell which
    thread it runs in and emit against it. Flat by design: a nested scope
    replaces and restores the ambient value but records no link to the one it
    shadowed; thread parent/child structure is recovered from the event log by
    ``build_graph``, not from a scope chain.

    ``thread_id`` is ``None`` in a coordinator-only scope
    (:func:`ai_functions.scope`): the coordinator to run on is bound, with no
    thread to attribute against or parent to. Readers that need a thread behave
    as they do outside any scope.
    """

    coordinator: Coordinator
    thread_id: ThreadId | None = None


_thread_scope: contextvars.ContextVar[ThreadScope | None] = contextvars.ContextVar(
    "ai_functions_thread_scope",
    default=None,
)


@contextmanager
def thread_scope(coordinator: Coordinator, thread_id: ThreadId | None = None) -> Iterator[ThreadScope]:
    """Bind the ambient thread to ``(coordinator, thread_id)`` for the duration of a block.

    Within the block, code that reads :func:`current_thread_scope` — e.g. a
    ``MemoryBackend`` recall/query/search with no explicit ``coordinator`` /
    ``thread_id`` — is attributed to this thread. The runtime opens one per
    cycle from the cycle's ``ThreadContext``, so in-cycle code needs no wiring;
    orchestration code opens its own::

        with thread_scope(coord, handle.id):
            guidelines = await memory.recall("joke_guidelines")

    Omitting ``thread_id`` binds the coordinator alone: bare AI-function calls
    in the block run on it, with no thread to parent them to.
    :func:`ai_functions.scope` wraps this form with the coordinator and worker
    lifecycle.

    Explicit ``recall(coordinator=..., thread_id=...)`` arguments override the
    ambient scope.

    Concurrency:
        Each :class:`asyncio.Task` gets an isolated copy, so concurrent cycles
        never see each other's scope. Code that fans out with ``asyncio.gather``
        in one task should open a scope per branch rather than share one.
    """
    scope = ThreadScope(coordinator=coordinator, thread_id=thread_id)
    token = _thread_scope.set(scope)
    try:
        yield scope
    finally:
        _thread_scope.reset(token)


def current_thread_scope() -> ThreadScope | None:
    """Return the ambient :class:`ThreadScope`, or ``None`` outside any scope."""
    return _thread_scope.get()


@contextmanager
def no_thread_scope() -> Iterator[None]:
    """Clear the ambient thread scope for the duration of a block.

    Library-internal AI-function calls — memory ``query`` / ``consolidate``
    helpers, the optimizer's backward function — must not attribute to the
    user's running thread: with the scope left in place they would spawn as
    children on the user's coordinator and pollute the very event log
    ``build_graph`` later reads. Wrapping them in ``no_thread_scope`` makes
    them run on private throwaway coordinators, exactly as if no thread were
    active. The prior scope is restored on exit.
    """
    token = _thread_scope.set(None)
    try:
        yield
    finally:
        _thread_scope.reset(token)
