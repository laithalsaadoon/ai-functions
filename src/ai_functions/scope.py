"""Bind the runtime bare AI-function calls run on.

A bare ``await my_function(...)`` runs on the coordinator bound by the
innermost open :func:`scope`, and on a private coordinator and worker — built
for that call and dropped when it returns — when no scope is open. Binding one
for a block makes the calls inside it observable: subscribe to their events
live (``on``), replay them (``get_events``), or inspect the threads they ran
on (``list_threads``)::

    import ai_functions
    from ai_functions.cli import print_event


    async def main() -> None:
        async with ai_functions.scope(on_event=print_event) as coord:
            summary = await summarize(text)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .types.context import thread_scope

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .protocols import Coordinator, OnEventCallback


@asynccontextmanager
async def scope(*, on_event: OnEventCallback | None = None) -> AsyncIterator[Coordinator]:
    """Run bare AI-function calls in the block on one shared runtime.

    Builds an :class:`~ai_functions.InMemoryCoordinator` with a registered
    :class:`~ai_functions.LocalWorker`, binds them as the ambient
    :func:`~ai_functions.types.thread_scope` for the block, and yields the
    coordinator. Every ``await my_function(...)`` in the block spawns there,
    so its events land in one log the caller can subscribe to and replay.

    Args:
        on_event: Subscriber registered on the coordinator before the block
            runs, so it sees every event from the first spawn on. Pass
            :func:`~ai_functions.cli.print_event` to print the feed to
            standard output as it happens.

    Yields:
        The :class:`~ai_functions.protocols.Coordinator` the calls run on.

    Ensures:
        - The worker is closed on exit, including when the body raises; the
          coordinator survives, so ``get_events`` still replays the log.
        - The ambient scope is restored on exit.

    Note:
        Scopes replace rather than nest, as
        :class:`~ai_functions.types.ThreadScope` is flat by design: a scope
        opened inside a running thread's cycle shadows that thread for the
        block, and calls made there are no longer attributed to it.
    """
    from .runtime import InMemoryCoordinator, LocalWorker

    coordinator = InMemoryCoordinator()
    if on_event is not None:
        _ = coordinator.on(on_event)
    worker = await LocalWorker(coordinator).register()
    try:
        with thread_scope(coordinator):
            yield coordinator
    finally:
        await worker.close()
