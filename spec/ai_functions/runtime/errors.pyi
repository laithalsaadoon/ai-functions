"""Errors specific to runtime execution: dispatcher and distributed transport.

Each error declares an ``error_kind`` class attribute — the
:data:`~ai_functions.network.wire.ErrorKind` value a peer reads when the error
crosses the wire (see :func:`ai_functions.network.error_kinds.classify`).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from ..types import EventKind, ThreadId, WorkerId


class ThreadIdMismatchError(ValueError):
    """An event's ``thread_id`` disagrees with the routing context.

    ``worker._route_event`` stamps each event with the thread id of
    the cycle it's being emitted from. If the caller also set
    ``event.thread_id`` and the two disagree, the call is almost
    certainly a bug — refusing it surfaces the mismatch at the offending
    frame.

    Args:
        event_thread_id: ``event.thread_id`` as the caller set it.
        routing_thread_id: ``thread_id`` ``_route_event`` was called with.

    Invariants:
        Events stored via a ``Coordinator`` always carry a ``thread_id``
        that matches the thread whose log they live in.
    """

    error_kind: ClassVar[str]
    event_thread_id: ThreadId
    routing_thread_id: ThreadId

    def __init__(self, event_thread_id: ThreadId, routing_thread_id: ThreadId) -> None: ...


class EventEmissionError(RuntimeError):
    """An event of the wrong kind was routed through ``worker._route_event``.

    The runtime routing gates all ``append_event`` calls through a single routing
    function that knows who the emitter is:

    - ``source="thread"`` — calls from inside a cycle, delivered via
      ``ThreadContext.on_event``. These may NOT emit lifecycle events
      (``STARTED``, ``COMPLETED``, ``FAILED``, ``CANCELLED``, ``RESULT``);
      lifecycle emission is the dispatcher's job (I5).
    - ``source="runtime"`` — calls from the runtime itself (dispatcher,
      message router, approval handling, session management). These may
      NOT emit ``MESSAGE_USER`` events; user-message emission is the
      bridge hook's job (I7).

    Args:
        kind: The ``EventKind`` the caller tried to emit.
        thread_id: Id of the thread whose log the event targeted.
        source: Where the call came from (``"runtime"`` or ``"thread"``).

    Invariants:
        I5, I7.
    """

    error_kind: ClassVar[str]
    kind: EventKind
    thread_id: ThreadId
    source: Literal["runtime", "thread"]

    def __init__(
        self,
        kind: EventKind,
        thread_id: ThreadId,
        source: Literal["runtime", "thread"],
    ) -> None: ...


class ThreadNotFoundError(KeyError):
    """The coordinator has no thread registered under the given id.

    Raised by every ``Coordinator`` method that takes a ``thread_id``
    and must route to a worker. Callers that want an Option-style
    lookup should catch this explicitly.

    Args:
        thread_id: The id that was not found.
    """

    error_kind: ClassVar[str]
    thread_id: ThreadId

    def __init__(self, thread_id: ThreadId) -> None: ...


class DistributedError(Exception):
    """Base error for distributed execution failures."""


class WorkerLostError(DistributedError):
    """Worker process crashed or disconnected.

    Raised by every ``Coordinator`` method that must route ``thread_id`` to a
    worker when the hosting worker is gone. The thread's ``ThreadInfo`` survives
    the loss with status ``FAILED``, so callers can still read its status and
    replay its log; only routing fails. Over the wire it reaches the caller as a
    ``RemoteError`` classified ``"worker_lost"``, because ``worker_id`` and
    ``thread_ids`` cannot be rebuilt from an ``ErrorFrame``'s message.

    Args:
        worker_id: Identifier of the worker that was lost.
        thread_ids: Ids of threads previously hosted on this worker.
    """

    error_kind: ClassVar[str]
    worker_id: WorkerId
    thread_ids: list[ThreadId]

    def __init__(self, worker_id: WorkerId, thread_ids: list[ThreadId]) -> None: ...


class SerializationError(DistributedError):
    """Spawnable could not be cloudpickled for remote execution.

    Args:
        function_name: Name of the spawnable that failed to serialise.
        reason: Cloudpickle error text.
    """

    error_kind: ClassVar[str]
    function_name: str

    def __init__(self, function_name: str, reason: str) -> None: ...


class ConnectionLostError(DistributedError):
    """WebSocket connection to the server was lost and reconnection failed.

    Args:
        url: WebSocket URL that could not be reached.
        retries: Number of failed reconnection attempts.
    """

    error_kind: ClassVar[str]
    url: str
    retries: int

    def __init__(self, url: str, retries: int) -> None: ...
