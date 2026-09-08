"""Worker: process-local execution engine.

A worker owns one asyncio dispatcher task per thread it hosts, plus the
queues, signals, and live ``Thread`` instances needed to drive them. It
registers itself with a :class:`Coordinator` and serves operations that
the coordinator routes to it via the :class:`WorkerAdapter` protocol.

A concrete :class:`LocalWorker` is provided. Users who want a different
execution strategy (multi-process, thread-pool-backed, etc.) can write
their own concrete class that implements ``WorkerAdapter``; there is no
``Worker`` protocol beyond the adapter.

Invariants:
    - I1 — the worker is the sole host for every thread it registers.
    - I4 — the dispatcher task for a thread is created only after all
      per-thread state (and any seeded history) is fully populated.
    - I5 — the dispatcher is the sole emitter of lifecycle events
      (``STARTED``, ``COMPLETED``, ``CANCELLED``, ``FAILED``, ``RESULT``).
    - I12 — if a dispatcher task dies with an uncaught exception,
      every pending ``PromptRequest`` future on that thread MUST be
      failed with that exception. No caller of ``handle.run(...)`` is
      allowed to hang because the dispatcher crashed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, cast, runtime_checkable

from ..handle import ThreadHandle
from ..protocols import Coordinator, OnEventCallback, Spawnable, Thread
from ..types import (
    ApprovalRequestEvent,
    CancelledEvent,
    CompletedEvent,
    Event,
    EventKind,
    FailedEvent,
    ResultEvent,
    StartedEvent,
    ThreadContext,
    ThreadId,
    ThreadInfo,
    ThreadSpawnedEvent,
    ThreadStatus,
    WorkerId,
    thread_scope,
)
from .errors import EventEmissionError, ThreadIdMismatchError, ThreadNotFoundError

# Lifecycle event kinds the worker dispatcher owns exclusively (I5).
_LIFECYCLE_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.STARTED,
        EventKind.COMPLETED,
        EventKind.FAILED,
        EventKind.CANCELLED,
        EventKind.RESULT,
    }
)

# Event kinds the event-bridge hook owns exclusively (I7).
_BRIDGE_KINDS: frozenset[EventKind] = frozenset({EventKind.MESSAGE_USER})


_logger: logging.Logger = logging.getLogger("ai_functions.runtime.worker")


# ── Work items ─────────────────────────────────────────────────────────────


@dataclass
class PromptRequest[T]:
    """A ``handle.run(*args, **kwargs)`` request carried through the work queue."""

    args: tuple[object, ...]
    kwargs: dict[str, object]
    future: asyncio.Future[T]


class TerminateAfterIdle:
    """Graceful termination marker; dispatcher runs teardown and exits on dequeue."""


WorkItem = PromptRequest[Any] | TerminateAfterIdle  # pyright: ignore[reportExplicitAny]


# ── WorkerAdapter protocol (coordinator → worker routing) ──────────────────


@runtime_checkable
class WorkerAdapter(Protocol):
    """Contract the coordinator uses to reach a worker.

    Public extension point: any class satisfying this protocol can act
    as a worker behind a :class:`Coordinator`. The library ships
    :class:`LocalWorker` as the default in-process implementation and
    builds an internal shim for remote workers served through the
    network endpoint. Most users never implement this protocol
    themselves — :class:`LocalWorker` is expected to cover in-process
    execution, and the network layer covers cross-process execution.

    Every method takes a ``thread_id`` the adapter is expected to host
    (the coordinator guarantees this via its registration table; the
    adapter may raise ``ThreadNotFoundError`` if asked to operate on a
    thread it doesn't host).
    """

    @property
    def worker_id(self) -> WorkerId:
        """Stable id assigned to this worker at registration time."""
        ...

    async def spawn[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId,
        thread_name: str | None,
        parent_id: ThreadId | None,
        metadata: dict[str, object],
    ) -> None:
        """Allocate per-thread state on this worker for ``thread_id``.

        Invoked by ``Coordinator.spawn`` after the coordinator has
        allocated the thread id and registered the ``ThreadInfo``.
        The adapter builds the live ``Thread`` from ``target``
        (in-process ``target.to_thread()`` for ``LocalWorker``;
        cloudpickle-then-``to_thread`` for remote workers), allocates
        its work queue and pause/cancel signals, and starts the
        dispatcher task.

        The coordinator, not the adapter, is responsible for calling
        ``register_thread`` and ``copy_events``; the adapter is only
        responsible for the worker-local per-thread state and the
        dispatcher lifecycle (I4).

        Args:
            target: Spawnable whose ``to_thread`` produces the live instance.
            thread_id: Id the coordinator has already allocated.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread for hierarchical rollup.
            metadata: Application metadata attached to the thread.
        """
        ...

    def submit(
        self,
        thread_id: ThreadId,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> asyncio.Future[Any]:  # pyright: ignore[reportExplicitAny]
        """Enqueue a ``PromptRequest`` on ``thread_id``'s work queue.

        Args:
            thread_id: Thread whose queue receives the request.
            args: Positional arguments forwarded to ``Thread.execute``.
            kwargs: Keyword arguments forwarded to ``Thread.execute``.

        Returns:
            A future the dispatcher resolves once the cycle completes.

        Concurrency:
            Synchronous enqueue; the cycle runs later on the dispatcher.

        Liveness (I12):
            The returned future MUST eventually resolve. If the
            dispatcher task dies with an uncaught exception, the
            adapter is responsible for failing this future (and every
            other pending future on ``thread_id``) with that exception
            rather than leaving callers awaiting forever.
            :class:`LocalWorker` attaches a done-callback on the
            dispatcher task to enforce this; remote adapters must
            surface the equivalent signal.
        """
        ...

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver a side-channel message to ``thread_id``'s live instance.

        Forwards to ``Thread.notify``. No cycle is started by
        this call.

        Args:
            thread_id: Thread receiving the message.
            text: Message body delivered to the live thread.
        """
        ...

    async def cancel(self, thread_id: ThreadId) -> None:
        """Cooperatively cancel the in-flight cycle on ``thread_id``.

        Args:
            thread_id: Thread whose in-flight cycle should be cancelled.
        """
        ...

    async def pause(self, thread_id: ThreadId) -> None:
        """Set ``thread_id``'s pause signal.

        Args:
            thread_id: Thread to pause.
        """
        ...

    async def resume(self, thread_id: ThreadId) -> None:
        """Clear ``thread_id``'s pause signal.

        Args:
            thread_id: Thread to resume.
        """
        ...

    async def terminate(self, thread_id: ThreadId) -> None:
        """Schedule graceful termination behind currently-queued work.

        Args:
            thread_id: Thread to terminate.
        """
        ...

    async def terminate_now(self, thread_id: ThreadId) -> None:
        """Tear ``thread_id`` down immediately; cancel in-flight, drop queued work.

        Args:
            thread_id: Thread to tear down.
        """
        ...

    async def get_fork_spawnable(self, thread_id: ThreadId) -> Spawnable[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Return a resumption spawnable for ``thread_id``.

        Delegates to the live ``Thread.fork()``. The returned spawnable
        is safe to pass to ``Coordinator.spawn`` with ``seed_from`` to
        materialize the forked thread.

        Args:
            thread_id: Source thread whose fork spawnable is requested.

        Returns:
            A spawnable that, when instantiated, resumes the source
            thread's non-event state (if any).

        Raises:
            NotImplementedError: The thread does not support forking.
        """
        ...

    def resolve_approval(
        self,
        thread_id: ThreadId,
        approval_id: str,
        decision: str,
    ) -> bool:
        """Resolve a pending tool-approval future inside this worker.

        Invoked by the coordinator when a user calls
        ``Coordinator.resolve_approval``. The adapter completes the
        pending future held by its ``on_interrupt`` handler; the
        coordinator then appends the ``ApprovalDecidedEvent``.

        Args:
            thread_id: Thread whose approval future is being resolved.
            approval_id: Id of the ``APPROVAL_REQUEST`` this resolves.
            decision: Policy decision to return to the executor.

        Returns:
            True iff a matching pending future was found and resolved.
        """
        ...


# ── Per-worker internal records ────────────────────────────────────────────


class _ThreadRecord:
    """Per-thread bookkeeping held by ``LocalWorker._records``.

    Coordinator owns ``status``; the worker only tracks what it uniquely
    needs: the live thread's metadata snapshot and the pending approvals
    futures waiting to be resolved.
    """

    def __init__(
        self,
        thread_name: str | None = None,
        parent_id: ThreadId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.thread_name = thread_name
        self.parent_id = parent_id
        self.metadata: dict[str, object] = metadata if metadata is not None else {}
        self.pending_approvals: dict[str, asyncio.Future[str]] = {}


# ── LocalWorker ────────────────────────────────────────────────────────────


class LocalWorker(WorkerAdapter):
    """In-process thread execution engine.

    Owns asyncio dispatcher tasks, per-thread work queues, pause/cancel
    signals, and the live ``Thread`` instances it hosts. Registers with
    the coordinator on first use (or explicitly via :meth:`register`)
    and deregisters on :meth:`close`.

    User-facing surface is narrow: construct with a coordinator, use
    :meth:`spawn_locally` for the in-process escape hatch (no
    serialization — the spawnable is passed by reference), call
    :meth:`close` to tear the worker down. Everything else flows through
    the coordinator.

    Args:
        coordinator: The coordinator this worker will register with.
        worker_id: Optional explicit id; a uuid is minted otherwise.

    Ensures:
        - Construction is pure: no I/O, no coroutine scheduling.
        - ``self`` is not yet registered with ``coordinator``.
        - No threads are hosted until :meth:`spawn_locally` or
          :meth:`spawn` is called.
    """

    __slots__ = (
        "_coordinator",
        "_worker_id",
        "_threads",
        "_queues",
        "_dispatchers",
        "_current_task",
        "_pause_signals",
        "_cancel_signals",
        "_records",
        "_registered",
        "_reaping",
    )

    _coordinator: Coordinator
    _worker_id: WorkerId
    _threads: dict[ThreadId, Thread[..., object]]
    _queues: dict[ThreadId, asyncio.Queue[WorkItem]]
    _dispatchers: dict[ThreadId, asyncio.Task[None]]
    _current_task: dict[ThreadId, asyncio.Task[object]]
    _pause_signals: dict[ThreadId, asyncio.Event]
    _cancel_signals: dict[ThreadId, asyncio.Event]
    _records: dict[ThreadId, _ThreadRecord]
    _registered: bool
    _reaping: set[asyncio.Task[None]]

    def __init__(
        self,
        coordinator: Coordinator,
        *,
        worker_id: WorkerId | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._worker_id = worker_id if worker_id is not None else WorkerId(f"worker-{uuid.uuid4().hex[:12]}")
        self._threads = {}
        self._queues = {}
        self._dispatchers = {}
        self._current_task = {}
        self._pause_signals = {}
        self._cancel_signals = {}
        self._records = {}
        self._registered = False
        self._reaping = set()

    async def register(self) -> Self:
        """Register this worker with the coordinator.

        Idempotent. Called automatically on the first
        :meth:`spawn_locally` / :meth:`spawn`; callers may invoke it
        eagerly to surface registration failures before a spawn is
        attempted.

        Returns:
            ``self``, so construction and registration can be chained
            (e.g. ``worker = await LocalWorker(coord).register()``).

        Ensures:
            After return, ``self.worker_id`` is in the coordinator's
            worker pool.
        """
        if self._registered:
            return self
        await self._coordinator.register_worker(self)
        self._registered = True
        return self

    @property
    def worker_id(self) -> WorkerId:
        """Stable id for this worker."""
        return self._worker_id

    @property
    def coordinator(self) -> Coordinator:
        """The coordinator this worker is registered with."""
        return self._coordinator

    # ── User-facing spawn (no serialization) ───────────────────────────────

    async def spawn_locally[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId | None = None,
        thread_name: str | None = None,
        parent_id: ThreadId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ThreadHandle[P, T]:
        """Host a new thread on this worker and return a handle to it.

        The escape hatch for spawnables that cannot survive
        cloudpickling: ``target`` is passed by reference, never
        serialized. The created thread is registered with
        ``self.coordinator`` before the dispatcher starts (I4); the
        returned handle is backed by that coordinator.

        Named ``spawn_locally`` (rather than ``spawn``) to avoid
        collision with the ``WorkerAdapter.spawn`` method the coordinator
        calls when routing a generic ``Coordinator.spawn`` request.

        Args:
            target: Spawnable whose ``to_thread`` produces the live instance.
            thread_id: Explicit id; one is minted if omitted.
            thread_name: Human label for telemetry.
            parent_id: Id of the parent thread for hierarchical rollup.
            metadata: Application metadata attached to the thread.

        Returns:
            A ``ThreadHandle`` whose coordinator is ``self.coordinator``.

        Ensures:
            - ``self.coordinator`` knows about the thread before the
              dispatcher starts (I4).
            - The dispatcher task is running, blocked on ``q.get``.
        """
        await self.register()
        tid = thread_id if thread_id is not None else ThreadId(f"thread-{uuid.uuid4().hex[:12]}")
        if thread_name is None:
            # Same display-name defaulting as ``Coordinator.spawn``: derive it
            # from the spawnable so event logs carry the function's name.
            name = getattr(target, "name", None)
            thread_name = name if isinstance(name, str) and name else None

        # Register the ThreadInfo with the coordinator BEFORE allocating
        # per-thread state on the worker — the coordinator is the source
        # of truth for "does this thread exist". I4: the dispatcher is
        # not started until all state is populated below.
        info = ThreadInfo(
            thread_id=tid,
            worker_id=self._worker_id,
            thread_name=thread_name,
            input_shape=target.input_shape,
            status=ThreadStatus.NOT_STARTED,
            parent_id=parent_id,
        )
        await self._coordinator.register_thread(info)
        # spawn_locally never seeds from another thread, so a parent_id here is
        # always a genuine sub-computation — record its edge for build_graph.
        if parent_id is not None:
            self._coordinator.append_event(ThreadSpawnedEvent(thread_id=parent_id, child_thread_id=tid))
        self._allocate_state(
            target,
            thread_id=tid,
            thread_name=thread_name,
            parent_id=parent_id,
            metadata=metadata if metadata is not None else {},
        )
        self._start_dispatcher(tid)
        return ThreadHandle(tid, self._coordinator)

    def _allocate_state[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId,
        thread_name: str | None,
        parent_id: ThreadId | None,
        metadata: dict[str, object],
    ) -> None:
        """Allocate per-thread queues, signals, and live ``Thread`` instance.

        Called on the hot path after the coordinator has registered the
        thread. Does NOT start the dispatcher — caller invokes
        ``_start_dispatcher`` when the (optional) event seed is in place.
        """
        thread = target.to_thread()
        self._threads[thread_id] = thread  # type: ignore[assignment]
        self._queues[thread_id] = asyncio.Queue()
        self._pause_signals[thread_id] = asyncio.Event()
        self._cancel_signals[thread_id] = asyncio.Event()
        self._records[thread_id] = _ThreadRecord(
            thread_name=thread_name,
            parent_id=parent_id,
            metadata=metadata,
        )

    def _start_dispatcher(self, thread_id: ThreadId) -> None:
        """Create and store the dispatcher task for a registered thread.

        Installs a done-callback that, if the dispatcher dies with an
        uncaught exception, fails every pending ``PromptRequest`` future
        for this thread with that exception. Otherwise callers of
        ``handle.run(...)`` would wait forever on a future nobody is
        going to resolve.
        """
        task = asyncio.get_event_loop().create_task(self._dispatch(thread_id))
        self._dispatchers[thread_id] = task
        task.add_done_callback(lambda t: self._on_dispatcher_done(thread_id, t))

    def _on_dispatcher_done(self, thread_id: ThreadId, task: asyncio.Task[None]) -> None:
        """Drain pending futures if the dispatcher died unexpectedly."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            # Clean shutdown via TerminateAfterIdle; nothing to do.
            return

        _logger.exception(
            "dispatcher for thread %s died; failing pending futures",
            thread_id,
            exc_info=exc,
        )
        queue = self._queues.get(thread_id)
        if queue is not None:
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, PromptRequest) and not item.future.done():
                    item.future.set_exception(exc)
        current = self._current_task.get(thread_id)
        if current is not None and not current.done():
            _ = current.cancel()

    # ── Dispatcher ──────────────────────────────────────────────────────────

    async def _dispatch(self, thread_id: ThreadId) -> None:
        """Per-thread dispatcher coroutine; one task per registered thread."""
        q = self._queues[thread_id]
        thread: Thread[..., object] = self._threads[thread_id]
        cancel_signal = self._cancel_signals[thread_id]
        record = self._records[thread_id]
        thread_name = record.thread_name

        while True:
            work = await q.get()

            if isinstance(work, TerminateAfterIdle):
                self._teardown(thread_id)
                return

            ctx = self._build_ctx(thread_id)
            cancel_signal.clear()

            self._route_event(
                StartedEvent(thread_name=thread_name),
                thread_id=thread_id,
                source="runtime",
            )

            coro = self._execute_in_thread_scope(thread, ctx, work.args, work.kwargs)
            task = asyncio.create_task(coro)
            self._current_task[thread_id] = task
            dispatcher_cancelling = False
            terminal_event: StartedEvent | CompletedEvent | CancelledEvent | FailedEvent
            try:
                result = await task
                self._route_event(
                    ResultEvent(
                        thread_name=thread_name,
                        payload=thread.serialize_result(result),
                    ),
                    thread_id=thread_id,
                    source="runtime",
                )
                terminal_event = CompletedEvent(thread_name=thread_name)
                work.future.set_result(result)
            except asyncio.CancelledError:
                terminal_event = CancelledEvent(thread_name=thread_name)
                if not work.future.done():
                    work.future.cancel()
                cancel_signal.clear()
                current = asyncio.current_task()
                dispatcher_cancelling = current is not None and current.cancelling() > 0
            except Exception as e:
                terminal_event = FailedEvent(thread_name=thread_name, error=repr(e))
                if not work.future.done():
                    work.future.set_exception(e)
            finally:
                self._current_task.pop(thread_id, None)

            self._route_event(terminal_event, thread_id=thread_id, source="runtime")

            if dispatcher_cancelling:
                raise asyncio.CancelledError

    async def _execute_in_thread_scope[**P, T](
        self,
        thread: Thread[P, T],
        ctx: ThreadContext,
        args: tuple[Any, ...],  # pyright: ignore[reportExplicitAny]
        kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
    ) -> T:
        """Run one cycle with the ambient thread scope set from ``ctx``.

        Opening the scope inside the cycle's task keeps the contextvar copy
        isolated per thread, so concurrent cycles never see each other's scope.
        """
        with thread_scope(ctx.coordinator, ctx.thread_id):
            return await thread.execute(ctx, *args, **kwargs)

    # ── Context building ────────────────────────────────────────────────────

    def _build_ctx(self, thread_id: ThreadId) -> ThreadContext:
        """Build a fresh ``ThreadContext`` for one cycle on ``thread_id``."""
        record = self._records[thread_id]
        return ThreadContext(
            thread_id=thread_id,
            coordinator=self._coordinator,
            on_event=self._make_thread_on_event(thread_id),
            on_interrupt=self._make_on_interrupt(thread_id),
            pause_signal=self._pause_signals[thread_id],
            cancel_signal=self._cancel_signals[thread_id],
            parent_id=record.parent_id,
            metadata=record.metadata,
        )

    def _route_event(
        self,
        event: Event,
        *,
        thread_id: ThreadId,
        source: Literal["runtime", "thread"],
    ) -> None:
        """Gate every ``append_event`` call by emitter identity (I5, I7)."""
        kind = cast(EventKind, event.kind)
        if source == "thread" and kind in _LIFECYCLE_KINDS:
            raise EventEmissionError(kind=kind, thread_id=thread_id, source=source)
        if source == "runtime" and kind in _BRIDGE_KINDS:
            raise EventEmissionError(kind=kind, thread_id=thread_id, source=source)
        event_tid = cast("ThreadId | None", getattr(event, "thread_id", None))
        if event_tid is None:
            stamped = event.model_copy(update={"thread_id": thread_id})
        elif event_tid != thread_id:
            raise ThreadIdMismatchError(event_thread_id=event_tid, routing_thread_id=thread_id)
        else:
            stamped = event
        self._coordinator.append_event(stamped)

    def _make_thread_on_event(self, thread_id: ThreadId) -> OnEventCallback:
        """Build the thread-facing ``on_event`` bound to ``source='thread'`` (I5)."""

        def _on_event(event: Event) -> None:
            self._route_event(event, thread_id=thread_id, source="thread")

        return _on_event

    def _make_on_interrupt(self, thread_id: ThreadId) -> Any:  # pyright: ignore[reportExplicitAny]
        """Build the ``on_interrupt`` callback for ``thread_id``.

        Approvals are resolved via ``Coordinator.resolve_approval``,
        which calls back into ``WorkerAdapter.resolve_approval`` to
        complete the matching future.
        """

        async def _on_interrupt(interrupts: list[Any]) -> list[Any]:  # pyright: ignore[reportExplicitAny]
            from strands.types.interrupt import InterruptResponse, InterruptResponseContent

            record = self._records[thread_id]
            futures: list[asyncio.Future[str]] = []
            for interrupt in interrupts:
                fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
                record.pending_approvals[interrupt.id] = fut
                self._route_event(
                    ApprovalRequestEvent(
                        approval_id=interrupt.id,
                        tool_name=interrupt.name,
                    ),
                    thread_id=thread_id,
                    source="runtime",
                )
                futures.append(fut)
            decisions = await asyncio.gather(*futures)
            responses: list[InterruptResponseContent] = []
            for interrupt, decision in zip(interrupts, decisions, strict=True):
                resp: InterruptResponse = {"interruptId": interrupt.id, "response": decision}
                responses.append(InterruptResponseContent(interruptResponse=resp))
            return responses

        return _on_interrupt

    # ── WorkerAdapter — called by the coordinator ──────────────────────────

    async def spawn[**P, T](
        self,
        target: Spawnable[P, T],
        *,
        thread_id: ThreadId,
        thread_name: str | None,
        parent_id: ThreadId | None,
        metadata: dict[str, object],
    ) -> None:
        """Coordinator-invoked path for :meth:`Coordinator.spawn`.

        The coordinator has already allocated the ``thread_id`` and
        registered the ``ThreadInfo``; this method allocates per-thread
        state and starts the dispatcher.
        """
        self._allocate_state(
            target,
            thread_id=thread_id,
            thread_name=thread_name,
            parent_id=parent_id,
            metadata=metadata,
        )
        self._start_dispatcher(thread_id)

    def submit(
        self,
        thread_id: ThreadId,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> asyncio.Future[Any]:  # pyright: ignore[reportExplicitAny]
        """Enqueue a ``PromptRequest`` on the thread's work queue."""
        if thread_id not in self._queues:
            raise ThreadNotFoundError(thread_id)
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()  # pyright: ignore[reportExplicitAny]

        req: PromptRequest[Any] = PromptRequest(  # pyright: ignore[reportExplicitAny]
            args=args,
            kwargs=dict(kwargs),
            future=future,
        )
        self._queues[thread_id].put_nowait(req)
        return future

    async def notify(self, thread_id: ThreadId, text: str) -> None:
        """Deliver text to the live thread's ``notify``."""
        thread = self._threads.get(thread_id)
        if thread is None:
            raise ThreadNotFoundError(thread_id)
        await thread.notify(text)

    async def cancel(self, thread_id: ThreadId) -> None:
        """Cooperatively cancel the in-flight cycle."""
        if thread_id not in self._records:
            raise ThreadNotFoundError(thread_id)
        self._cancel_signals[thread_id].set()
        task = self._current_task.get(thread_id)
        if task is not None and not task.done():
            task.cancel()

    async def pause(self, thread_id: ThreadId) -> None:
        """Set the pause signal."""
        if thread_id not in self._records:
            raise ThreadNotFoundError(thread_id)
        self._pause_signals[thread_id].set()

    async def resume(self, thread_id: ThreadId) -> None:
        """Clear the pause signal."""
        if thread_id not in self._records:
            raise ThreadNotFoundError(thread_id)
        self._pause_signals[thread_id].clear()

    async def terminate(self, thread_id: ThreadId) -> None:
        """Schedule graceful termination."""
        if thread_id not in self._queues:
            raise ThreadNotFoundError(thread_id)
        self._queues[thread_id].put_nowait(TerminateAfterIdle())

    async def terminate_now(self, thread_id: ThreadId) -> None:
        """Tear the thread down immediately."""
        if thread_id not in self._records:
            raise ThreadNotFoundError(thread_id)
        task = self._current_task.get(thread_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        dispatcher = self._dispatchers.get(thread_id)
        if dispatcher is not None and not dispatcher.done():
            dispatcher.cancel()
            try:
                await dispatcher
            except (asyncio.CancelledError, Exception):
                pass

        self._teardown(thread_id)

    async def get_fork_spawnable(self, thread_id: ThreadId) -> Spawnable[..., Any]:  # pyright: ignore[reportExplicitAny]
        """Return a resumption spawnable produced by the live thread's ``fork()``."""
        source_thread = self._threads.get(thread_id)
        if source_thread is None:
            raise ThreadNotFoundError(thread_id)
        return await source_thread.fork()

    def resolve_approval(self, thread_id: ThreadId, approval_id: str, decision: str) -> bool:
        """Resolve a pending tool-approval future inside this worker."""
        record = self._records.get(thread_id)
        if record is None:
            return False
        fut = record.pending_approvals.pop(approval_id, None)
        if fut is None or fut.done():
            return False
        fut.set_result(decision)
        return True

    def has_thread(self, thread_id: ThreadId) -> bool:
        """Return True iff this worker currently hosts ``thread_id``.

        Args:
            thread_id: Thread to probe.

        Returns:
            True if ``thread_id`` is hosted here, else False.
        """
        return thread_id in self._records

    # ── Teardown ────────────────────────────────────────────────────────────

    def _teardown(self, thread_id: ThreadId) -> None:
        """Shared teardown for graceful and hard termination paths."""
        q = self._queues.get(thread_id)
        if q is not None:
            while not q.empty():
                try:
                    item = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, PromptRequest) and not item.future.done():
                    item.future.cancel()

        thread = self._threads.get(thread_id)
        if thread is not None:
            _ = asyncio.create_task(thread.teardown())

        # Deregister from the coordinator asynchronously; the coordinator
        # drops the info and the routing entry. ``close`` awaits these tasks
        # before it deregisters the worker, so a clean shutdown never leaves a
        # torn-down thread in the registry for ``deregister_worker`` to fail.
        reap = asyncio.create_task(self._coordinator.deregister_thread(thread_id))
        self._reaping.add(reap)
        reap.add_done_callback(self._reaping.discard)

        self._threads.pop(thread_id, None)
        self._queues.pop(thread_id, None)
        self._dispatchers.pop(thread_id, None)
        self._current_task.pop(thread_id, None)
        self._pause_signals.pop(thread_id, None)
        self._cancel_signals.pop(thread_id, None)
        self._records.pop(thread_id, None)

    async def close(self) -> None:
        """Gracefully terminate every thread hosted by this worker.

        Each thread is ``terminate``-d (its ``TerminateAfterIdle`` flow
        runs, which tears down any local resources via
        ``Thread.teardown``). After every thread's dispatcher exits,
        the worker deregisters itself from the coordinator.

        Ensures:
            - No thread hosted by this worker remains registered with
              the coordinator.
            - ``self`` is deregistered from the coordinator's worker
              pool.

        Concurrency:
            Idempotent; closing an already-closed worker is a no-op.
        """
        tids = list(self._dispatchers.keys())
        for tid in tids:
            if tid in self._queues:
                self._queues[tid].put_nowait(TerminateAfterIdle())

        # Wait for every dispatcher to exit.
        dispatchers = list(self._dispatchers.values())
        for d in dispatchers:
            try:
                await d
            except (asyncio.CancelledError, Exception):
                pass

        # ``_teardown`` schedules each thread's deregistration rather than
        # awaiting it. Draining those tasks here is what makes this a clean
        # shutdown: ``deregister_worker`` fails every thread the worker still
        # hosts, so a thread whose deregistration were still pending would be
        # published as ``failed`` after an orderly close.
        for reap in list(self._reaping):
            try:
                await reap
            except (asyncio.CancelledError, Exception):
                pass

        if self._registered:
            await self._coordinator.deregister_worker(self._worker_id)
            self._registered = False
