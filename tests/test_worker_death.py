"""A dropped worker socket fails that worker's threads across a real endpoint.

``CoordinatorEndpoint._serve_connection`` calls ``deregister_worker`` in its
``finally``: when a client hosting a worker disappears, that is the only signal
the coordinator gets. These tests run the whole path — websocket endpoint, two
clients, a ``LocalWorker`` hosted behind one of them — and pin what a surviving
peer sees after the worker's socket drops:

- the orphaned thread is ``failed`` in ``list_threads`` and its log ends with a
  ``FAILED`` event naming the lost worker;
- routing to it fails with the ``"worker_lost"`` classification, which is what a
  remote caller branches on: ``WorkerLostError`` carries a worker id and a
  thread list that an ``ErrorFrame`` cannot rebuild, so the far side sees a
  ``RemoteError`` whose ``kind`` names the failure.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from ai_functions.network import CoordinatorClient, CoordinatorEndpoint, RemoteError
from ai_functions.runtime import LocalWorker
from ai_functions.types import EventKind, InputShape, ThreadStatus
from ai_functions.types.events import FailedEvent

if TYPE_CHECKING:
    from ai_functions.protocols import Spawnable, Thread, ThreadContext

# How long to give the endpoint's 0.05s connection-close poll before asserting.
_CLOSE_GRACE_SECONDS: float = 1.0


class _EchoThread:
    """Live instance behind :class:`Echo`; one cycle returns its argument."""

    @property
    def name(self) -> str:
        """Display name carried into the thread's event log."""
        return "echo"

    async def execute(self, ctx: ThreadContext, text: str) -> str:
        """Return ``text`` unchanged; no model, no I/O."""
        del ctx
        return text

    async def notify(self, text: str) -> None:
        """Drop side-channel messages; this thread has no buffer."""
        del text

    def serialize_result(self, result: str) -> str:
        """Store the result verbatim in the ``RESULT`` event."""
        return result

    def deserialize_result(self, payload: str) -> str:
        """Recover the result stored by :meth:`serialize_result`."""
        return payload

    async def fork(self) -> Spawnable[..., str]:
        """Refuse forking; these tests only need one thread per worker."""
        raise NotImplementedError

    async def teardown(self) -> None:
        """No resources to release."""


class Echo:
    """Minimal ``Spawnable``: a chat-shaped thread that echoes its prompt."""

    def to_thread(self) -> Thread[..., str]:
        """Produce a fresh live echo thread."""
        return _EchoThread()

    @property
    def input_shape(self) -> InputShape:
        """One positional ``str``, so the thread is chat-shaped."""
        return InputShape.STR_PROMPT


async def test_a_dropped_worker_socket_fails_the_threads_it_hosted() -> None:
    """The surviving peer sees ``failed`` and a ``"worker_lost"`` failure."""
    endpoint = CoordinatorEndpoint()
    await endpoint.start(host="127.0.0.1", port=0)
    host: CoordinatorClient | None = None
    observer: CoordinatorClient | None = None
    try:
        host = await CoordinatorClient.connect(endpoint.url)
        worker = LocalWorker(host)
        handle = await worker.spawn_locally(Echo())
        assert await handle.run("hi") == "hi"

        observer = await CoordinatorClient.connect(endpoint.url)
        assert (await observer.get_thread_status(handle.id)) is ThreadStatus.IDLE

        # Drop the worker's socket. The endpoint notices on its next poll and
        # deregisters every worker that connection hosted.
        await host.close()
        host = None
        deadline = asyncio.get_event_loop().time() + _CLOSE_GRACE_SECONDS
        while asyncio.get_event_loop().time() < deadline:
            if (await observer.get_thread_status(handle.id)) is ThreadStatus.FAILED:
                break
            await asyncio.sleep(0.05)

        infos = {info.thread_id: info for info in await observer.list_threads()}
        assert infos[handle.id].status is ThreadStatus.FAILED

        terminal = (await observer.get_events(handle.id))[-1]
        assert terminal.kind is EventKind.FAILED
        assert isinstance(terminal, FailedEvent)
        assert str(worker.worker_id) in terminal.error

        with pytest.raises(RemoteError) as exc_info:
            _ = await observer.submit(handle.id, "again")
        assert exc_info.value.remote_type == "WorkerLostError"
        assert exc_info.value.kind == "worker_lost"
        with pytest.raises(RemoteError) as exc_info:
            await observer.notify(handle.id, "anyone there")
        assert exc_info.value.kind == "worker_lost"
    finally:
        for client in (host, observer):
            if client is not None:
                try:
                    await client.close()
                except Exception:  # noqa: BLE001 -- teardown is best-effort
                    pass
        await endpoint.stop()
