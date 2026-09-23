"""I5/I7 enforcement and dispatcher-side lifecycle sequence guarantees.

I5: lifecycle events (``STARTED``, ``COMPLETED``, ``FAILED``, ``CANCELLED``,
``RESULT``) are emitted only by the runtime dispatcher.
I7: ``MESSAGE_USER`` is emitted only by the event-bridge hook (inside a
cycle).

Both are enforced at runtime by ``LocalWorker._route_event``, which gates
every ``append_event`` call by the caller's source (``"runtime"`` vs
``"thread"``). Violations raise ``EventEmissionError`` immediately.
``fork`` is the only runtime-side site that bypasses the gate, because it
legitimately copies both kinds in bulk.

The tests below cover:
  - A thread cannot emit lifecycle events via ``ctx.on_event`` (I5).
  - The runtime cannot emit ``MESSAGE_USER`` via ``_route_event`` (I7).
  - The dispatcher emits the correct lifecycle sequence on success/failure/cancel.
  - ``RESULT`` appears before ``COMPLETED`` on success and never appears on
    failure or cancel.
  - Losing a worker fails the threads it hosted, and a terminal status is
    never downgraded by a lifecycle event that lands afterwards.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_functions import ai_function
from ai_functions.runtime import EventEmissionError, InMemoryCoordinator, LocalWorker, WorkerLostError
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import Event, EventKind, ThreadId, ThreadStatus
from ai_functions.types.events import (
    CompletedEvent,
    FailedEvent,
    MessageUserEvent,
    StartedEvent,
)


@ai_function[str](structured_output=False)
def _simple(prompt: str) -> str:
    return prompt


# ── Runtime enforcement (I5) ──────────────────────────────────────────────


async def test_thread_cannot_emit_completed_event() -> None:
    """Emitting a lifecycle kind via ``ctx.on_event`` raises at runtime."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])
        handle = await h.spawn(_simple.replace(model=model))
        # Build a ThreadContext for the handle's thread and try to emit a
        # CompletedEvent through it.
        ctx = h.worker._build_ctx(handle.id)  # noqa: SLF001  -- testing internal contract
        with pytest.raises(EventEmissionError) as exc_info:
            ctx.on_event(CompletedEvent(thread_id=handle.id))
        assert exc_info.value.kind == EventKind.COMPLETED
        assert exc_info.value.thread_id == handle.id


@pytest.mark.parametrize(
    "kind, factory",
    [
        (EventKind.STARTED, lambda tid: StartedEvent(thread_id=tid)),
        (EventKind.COMPLETED, lambda tid: CompletedEvent(thread_id=tid)),
    ],
)
async def test_lifecycle_kinds_all_rejected(kind, factory) -> None:  # type: ignore[no-untyped-def]
    """Every lifecycle kind is rejected — not just COMPLETED."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])
        handle = await h.spawn(_simple.replace(model=model))
        ctx = h.worker._build_ctx(handle.id)  # noqa: SLF001
        with pytest.raises(EventEmissionError) as exc_info:
            ctx.on_event(factory(handle.id))
        assert exc_info.value.kind == kind


async def test_non_lifecycle_events_pass_through() -> None:
    """The filter must not interfere with normal event emission."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])
        handle = await h.spawn(_simple.replace(model=model))
        ctx = h.worker._build_ctx(handle.id)  # noqa: SLF001
        # Should not raise.
        ctx.on_event(MessageUserEvent(thread_id=handle.id, text="hello"))
        events = await h.events(handle.id)
        assert any(isinstance(e, MessageUserEvent) and e.text == "hello" for e in events)


# ── I7: runtime cannot emit MESSAGE_USER ───────────────────────────────────


async def test_runtime_cannot_emit_message_user() -> None:
    """``_route_event(source="runtime")`` rejects MESSAGE_USER (I7)."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])
        handle = await h.spawn(_simple.replace(model=model))
        with pytest.raises(EventEmissionError) as exc_info:
            h.worker._route_event(  # noqa: SLF001 -- testing the gate directly
                MessageUserEvent(thread_id=handle.id, text="forged"),
                thread_id=handle.id,
                source="runtime",
            )
        assert exc_info.value.kind == EventKind.MESSAGE_USER
        assert exc_info.value.source == "runtime"
        # The forged event must not have reached the log.
        events = await h.events(handle.id, kinds=[EventKind.MESSAGE_USER])
        assert not any(isinstance(e, MessageUserEvent) and e.text == "forged" for e in events)


async def test_runtime_source_allows_lifecycle_events() -> None:
    """Lifecycle events ARE allowed from ``source="runtime"`` — that's the dispatcher's job."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])
        handle = await h.spawn(_simple.replace(model=model))
        # Should not raise — we're pretending to be the dispatcher. We don't
        # assert anything about the log shape here; we're only verifying that
        # the filter does NOT reject lifecycle kinds from this source.
        h.worker._route_event(  # noqa: SLF001
            StartedEvent(thread_id=handle.id),
            thread_id=handle.id,
            source="runtime",
        )


# ── thread_id stamping and validation ─────────────────────────────────────


async def test_route_event_stamps_unset_thread_id() -> None:
    """A ``None`` ``thread_id`` is filled in by ``_route_event``."""
    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        ctx = h.worker._build_ctx(handle.id)  # noqa: SLF001
        # Construct without thread_id — relies on the new default.
        ctx.on_event(MessageUserEvent(text="hi"))
        events = await h.events(handle.id, kinds=[EventKind.MESSAGE_USER])
        # One user event, stamped with the routing id.
        assert len(events) == 1
        assert events[0].thread_id == handle.id


async def test_route_event_rejects_thread_id_mismatch() -> None:
    """A mismatched ``thread_id`` raises ``ThreadIdMismatchError``."""
    from ai_functions.runtime import ThreadIdMismatchError
    from ai_functions.types import ThreadId

    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        ctx = h.worker._build_ctx(handle.id)  # noqa: SLF001
        wrong = ThreadId("thr-not-this-one")
        with pytest.raises(ThreadIdMismatchError) as exc_info:
            ctx.on_event(MessageUserEvent(thread_id=wrong, text="from wrong thread"))
        assert exc_info.value.event_thread_id == wrong
        assert exc_info.value.routing_thread_id == handle.id
        # The rejected event must not have reached the log.
        events = await h.events(handle.id, kinds=[EventKind.MESSAGE_USER])
        assert all(isinstance(e, MessageUserEvent) and e.text != "from wrong thread" for e in events)


async def test_coordinator_rejects_unrouted_events() -> None:
    """``Coordinator.append_event`` raises ``ValueError`` if ``thread_id`` is unset.

    This is a safety net: nothing should reach the coordinator without a
    thread_id (the gate fills it in). If anything sidesteps the gate, the
    coordinator fails loudly rather than silently storing an unroutable event.
    """
    async with RuntimeHarness() as h:
        with pytest.raises(ValueError, match="thread_id.*unset"):
            h.coordinator.append_event(MessageUserEvent(text="orphan"))


# ── Dispatcher-side lifecycle sequence ────────────────────────────────────


async def test_successful_cycle_emits_correct_terminal_sequence() -> None:
    """On success the log ends with exactly ``RESULT`` then ``COMPLETED``."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="done")])
        handle = await h.spawn(_simple.replace(model=model))
        await handle.run("prompt")
        kinds = [e.kind for e in await h.events(handle.id)]
        # Exactly one STARTED and one COMPLETED; RESULT immediately before COMPLETED.
        assert kinds.count(EventKind.STARTED) == 1
        assert kinds.count(EventKind.COMPLETED) == 1
        assert kinds.count(EventKind.RESULT) == 1
        assert kinds.count(EventKind.FAILED) == 0
        assert kinds.count(EventKind.CANCELLED) == 0
        result_idx = kinds.index(EventKind.RESULT)
        completed_idx = kinds.index(EventKind.COMPLETED)
        assert result_idx < completed_idx


async def test_failed_cycle_emits_failed_without_result() -> None:
    """A cycle that raises an exception produces ``FAILED`` and NO ``RESULT``."""
    async with RuntimeHarness() as h:
        # Script empty → ScriptExhausted on first model call → failure.
        model = ScriptedModel([])
        handle = await h.spawn(_simple.replace(model=model))
        fut = handle.run("prompt")
        with pytest.raises(Exception):  # noqa: B017, PT011
            await fut
        kinds = [e.kind for e in await h.events(handle.id)]
        assert kinds.count(EventKind.STARTED) == 1
        assert kinds.count(EventKind.FAILED) == 1
        assert kinds.count(EventKind.COMPLETED) == 0
        assert kinds.count(EventKind.RESULT) == 0
        assert kinds.count(EventKind.CANCELLED) == 0


async def test_cancelled_cycle_emits_cancelled_without_result() -> None:
    """``cancel`` mid-flight produces ``CANCELLED`` and NO ``RESULT``.

    Pin the cycle on a barrier so we have a clear window to cancel it; the
    cancel_signal is set before the barrier releases, and the bridge's
    ``BeforeModelCallEvent`` hook re-checks the signal at the next work
    boundary — so releasing the barrier (or not) ultimately doesn't matter,
    the cycle exits on cancel.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok", await_before="hold")])
        handle = await h.spawn(_simple.replace(model=model))
        fut = handle.run("prompt")
        await h.wait_for(handle.id, EventKind.STARTED)
        # Cancel while blocked on the barrier; release the barrier so the
        # agent code can resume and hit the cancel check.
        await handle.cancel()
        h.release("hold")
        with pytest.raises(asyncio.CancelledError):
            await fut
        kinds = [e.kind for e in await h.events(handle.id)]
        assert kinds.count(EventKind.STARTED) == 1
        assert kinds.count(EventKind.CANCELLED) == 1
        assert kinds.count(EventKind.COMPLETED) == 0
        assert kinds.count(EventKind.RESULT) == 0
        assert kinds.count(EventKind.FAILED) == 0


async def test_multiple_cycles_produce_matching_lifecycle_counts() -> None:
    """N successful runs produce exactly N STARTED/RESULT/COMPLETED events."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="a"), Turn(text="b"), Turn(text="c")])
        handle = await h.spawn(_simple.replace(model=model))
        await handle.run("1")
        await handle.run("2")
        await handle.run("3")
        kinds = [e.kind for e in await h.events(handle.id)]
        assert kinds.count(EventKind.STARTED) == 3
        assert kinds.count(EventKind.RESULT) == 3
        assert kinds.count(EventKind.COMPLETED) == 3


# ── Worker loss cascades into the threads it hosted ───────────────────────


async def test_deregister_worker_fails_the_threads_it_still_hosted() -> None:
    """A thread whose host is gone reports ``failed``, not ``idle``.

    ``list_threads`` and the routing table must agree: before the cascade a
    thread orphaned by a lost worker stayed ``idle`` in the registry while
    every routed call on it raised, so discovery advertised a thread nothing
    could reach.
    """
    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        assert (await handle.run("prompt")).strip() == "ok"

        await h.coordinator.deregister_worker(h.worker.worker_id)

        infos = {info.thread_id: info for info in await h.coordinator.list_threads()}
        assert infos[handle.id].status is ThreadStatus.FAILED
        assert await h.coordinator.get_thread_status(handle.id) is ThreadStatus.FAILED

        terminal = (await h.events(handle.id))[-1]
        assert isinstance(terminal, FailedEvent)
        assert str(h.worker.worker_id) in terminal.error
        assert str(handle.id) in terminal.error


async def test_operations_on_an_orphaned_thread_name_the_lost_worker() -> None:
    """Routing to a gone worker raises ``WorkerLostError``, not ``ThreadNotFoundError``.

    The thread *is* registered; reporting "not found" sends the caller looking
    for a spawn bug instead of a dead host.
    """
    async with RuntimeHarness() as h:
        first = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="a")])))
        second = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="b")])))
        await h.coordinator.deregister_worker(h.worker.worker_id)

        with pytest.raises(WorkerLostError) as exc_info:
            _ = first.run("again")
        assert exc_info.value.worker_id == h.worker.worker_id
        assert exc_info.value.thread_ids[0] == first.id
        assert set(exc_info.value.thread_ids) == {first.id, second.id}

        with pytest.raises(WorkerLostError):
            await h.coordinator.notify(first.id, "hello")
        with pytest.raises(WorkerLostError):
            await h.coordinator.cancel(first.id)
        with pytest.raises(WorkerLostError):
            await h.coordinator.terminate(first.id)


async def test_worker_loss_is_fully_recorded_before_failure_callbacks() -> None:
    """Subscribers see every affected thread failed and the complete loss set."""
    coordinator = InMemoryCoordinator()
    async with RuntimeHarness(coordinator=coordinator) as h:
        first = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="a")])))
        second = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="b")])))
        thread_ids = {first.id, second.id}
        observed: list[tuple[ThreadId, dict[ThreadId, ThreadStatus], WorkerLostError]] = []

        def on_failure(event: Event) -> None:
            assert event.thread_id is not None
            # Synchronous callbacks cannot await discovery; snapshot the cache
            # they observe and assert outside append_event's exception isolation.
            statuses = {
                tid: coordinator._infos[tid].status  # noqa: SLF001
                for tid in thread_ids
            }
            try:
                coordinator.submit(event.thread_id, "retry")
            except WorkerLostError as exc:
                observed.append((event.thread_id, statuses, exc))

        with coordinator.on(on_failure, kinds=[EventKind.FAILED]):
            await coordinator.deregister_worker(h.worker.worker_id)

        assert len(observed) == 2
        assert {tid for tid, _, _ in observed} == thread_ids
        for tid, statuses, error in observed:
            assert statuses == dict.fromkeys(thread_ids, ThreadStatus.FAILED)
            assert error.worker_id == h.worker.worker_id
            assert error.thread_ids[0] == tid
            assert set(error.thread_ids) == thread_ids


async def test_in_flight_pause_does_not_downgrade_worker_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pause response arriving after worker loss must leave the thread failed."""
    pause_started = asyncio.Event()
    release_pause = asyncio.Event()
    original_pause = LocalWorker.pause

    async def delayed_pause(worker: LocalWorker, thread_id: ThreadId) -> None:
        await original_pause(worker, thread_id)
        pause_started.set()
        await release_pause.wait()

    monkeypatch.setattr(LocalWorker, "pause", delayed_pause)
    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        assert (await handle.run("prompt")).strip() == "ok"
        pause_task = asyncio.create_task(h.coordinator.pause(handle.id))
        try:
            await asyncio.wait_for(pause_started.wait(), timeout=2)
            await h.coordinator.deregister_worker(h.worker.worker_id)
            assert await handle.status() is ThreadStatus.FAILED

            release_pause.set()
            await asyncio.wait_for(pause_task, timeout=2)
            assert await handle.status() is ThreadStatus.FAILED
        finally:
            release_pause.set()
            await asyncio.gather(pause_task, return_exceptions=True)


async def test_deregister_worker_is_idempotent_and_fails_each_thread_once() -> None:
    """A second deregistration finds nothing non-terminal left to fail."""
    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        await h.coordinator.deregister_worker(h.worker.worker_id)
        await h.coordinator.deregister_worker(h.worker.worker_id)

        kinds = [e.kind for e in await h.events(handle.id)]
        assert kinds.count(EventKind.FAILED) == 1


async def test_a_terminal_status_is_never_downgraded_by_a_later_lifecycle_event() -> None:
    """A lifecycle event that lands after the thread died leaves the status alone.

    A cycle already in flight when its worker is lost still emits ``COMPLETED``.
    Mapping that to ``idle`` would resurrect a thread nothing can route to.
    """
    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        await h.coordinator.deregister_worker(h.worker.worker_id)
        assert await h.coordinator.get_thread_status(handle.id) is ThreadStatus.FAILED

        h.worker._route_event(  # noqa: SLF001 -- standing in for a cycle that was already in flight
            CompletedEvent(thread_id=handle.id),
            thread_id=handle.id,
            source="runtime",
        )
        assert await h.coordinator.get_thread_status(handle.id) is ThreadStatus.FAILED


async def test_a_failed_cycle_leaves_the_thread_runnable() -> None:
    """A dispatcher ``FAILED`` reports one broken cycle, not a dead thread."""
    async with RuntimeHarness() as h:
        handle = await h.spawn(_simple.replace(model=ScriptedModel([])))
        with pytest.raises(Exception):  # noqa: B017, PT011
            await handle.run("prompt")
        assert await h.coordinator.get_thread_status(handle.id) is ThreadStatus.IDLE


async def test_an_orderly_worker_close_fails_nothing() -> None:
    """``LocalWorker.close`` reaps its threads before it leaves the pool.

    ``_teardown`` only schedules each deregistration, so a close that did not
    drain those tasks would hand ``deregister_worker`` a registry still full of
    threads and publish every one of them as ``failed``.
    """
    async with RuntimeHarness() as h:
        handle = await h.coordinator.spawn(_simple.replace(model=ScriptedModel([Turn(text="ok")])))
        assert (await handle.run("prompt")).strip() == "ok"
        await h.worker.close()

        assert await h.coordinator.list_threads() == []
        kinds = [e.kind for e in await h.events(handle.id)]
        assert kinds.count(EventKind.FAILED) == 0
