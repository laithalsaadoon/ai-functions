"""Errors specific to runtime execution: dispatcher and distributed transport.

Each error declares an ``error_kind`` class attribute — the
:data:`~ai_functions.network.wire.ErrorKind` value a peer reads when the error
crosses the wire (see :func:`ai_functions.network.error_kinds.classify`).
"""

# ``error_kind`` is typed ``str`` rather than ``ErrorKind`` so this module does
# not import the network layer.

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

    error_kind: ClassVar[str] = "invalid_input"

    def __init__(self, event_thread_id: ThreadId, routing_thread_id: ThreadId) -> None:
        self.event_thread_id: ThreadId = event_thread_id
        self.routing_thread_id: ThreadId = routing_thread_id
        super().__init__(
            f"Event carries thread_id={event_thread_id!r} but is being routed "
            f"to thread {routing_thread_id!r}. Leave ``thread_id`` unset and "
            "let the runtime stamp it, or pass the matching id."
        )


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

    error_kind: ClassVar[str] = "invalid_input"

    def __init__(
        self,
        kind: EventKind,
        thread_id: ThreadId,
        source: Literal["runtime", "thread"],
    ) -> None:
        self.kind: EventKind = kind
        self.thread_id: ThreadId = thread_id
        self.source: Literal["runtime", "thread"] = source
        if source == "thread":
            hint = (
                "lifecycle events (STARTED/COMPLETED/FAILED/CANCELLED/RESULT) are owned by the runtime dispatcher (I5)"
            )
        else:
            hint = (
                "MESSAGE_USER events are emitted only by the event-bridge hook "
                "draining ``message_queue`` at a model-call boundary (I7)"
            )
        super().__init__(
            f"Event of kind {kind.value!r} may not be emitted by source {source!r} for thread {thread_id!r}: {hint}."
        )


class ThreadNotFoundError(KeyError):
    """The coordinator has no thread registered under the given id.

    Raised by every ``Coordinator`` method that takes a ``thread_id``
    and must route to a worker. Callers that want an Option-style
    lookup should catch this explicitly.

    Args:
        thread_id: The id that was not found.
    """

    error_kind: ClassVar[str] = "not_found"

    def __init__(self, thread_id: ThreadId) -> None:
        self.thread_id: ThreadId = thread_id
        super().__init__(thread_id)


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

    error_kind: ClassVar[str] = "worker_lost"

    def __init__(self, worker_id: WorkerId, thread_ids: list[ThreadId]) -> None:
        self.worker_id: WorkerId = worker_id
        self.thread_ids: list[ThreadId] = thread_ids
        super().__init__(f"Worker {worker_id} lost. Affected threads: {thread_ids}")


class SerializationError(DistributedError):
    """Spawnable could not be cloudpickled for remote execution.

    Args:
        function_name: Name of the spawnable that failed to serialise.
        reason: Cloudpickle error text.
    """

    error_kind: ClassVar[str] = "invalid_input"

    def __init__(self, function_name: str, reason: str) -> None:
        self.function_name: str = function_name
        super().__init__(f"Cannot serialize '{function_name}': {reason}")


class ConnectionLostError(DistributedError):
    """WebSocket connection to the server was lost and reconnection failed.

    Args:
        url: WebSocket URL that could not be reached.
        retries: Number of failed reconnection attempts.
    """

    error_kind: ClassVar[str] = "connection_lost"

    def __init__(self, url: str, retries: int) -> None:
        self.url: str = url
        self.retries: int = retries
        super().__init__(f"Connection to {url} lost after {retries} retries")
