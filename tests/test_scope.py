"""``ai_functions.scope``: bare calls run on the scoped coordinator."""

from __future__ import annotations

import pytest

import ai_functions
from ai_functions import ai_function
from ai_functions.cli import print_event
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types import EventKind
from ai_functions.types.context import current_thread_scope


@ai_function[str](structured_output=False)
def _fn(prompt: str) -> str:
    return prompt


async def test_bare_call_runs_on_scoped_coordinator() -> None:
    """A bare call inside the block spawns on the yielded coordinator."""
    fn = _fn.replace(model=ScriptedModel([Turn(text="hello")]))
    async with ai_functions.scope() as coord:
        result = await fn("go")
        assert result.strip() == "hello"
        threads = await coord.list_threads()
        assert len(threads) == 1
        events = await coord.get_events(threads[0].thread_id)
        assert any(e.kind == EventKind.COMPLETED for e in events)


async def test_events_survive_exit_and_scope_is_restored() -> None:
    """The coordinator's log outlives the block; the ambient scope does not."""
    fn = _fn.replace(model=ScriptedModel([Turn(text="hello")]))
    async with ai_functions.scope() as coord:
        _ = await fn("go")
        thread_id = (await coord.list_threads())[0].thread_id
    assert current_thread_scope() is None
    events = await coord.get_events(thread_id)
    assert any(e.kind == EventKind.COMPLETED for e in events)


async def test_scope_restored_when_body_raises() -> None:
    """The ambient scope is cleared even when the block raises."""
    with pytest.raises(RuntimeError, match="boom"):
        async with ai_functions.scope():
            raise RuntimeError("boom")
    assert current_thread_scope() is None


async def test_on_event_print_event_prints_feed(capsys: pytest.CaptureFixture[str]) -> None:
    """``on_event=print_event`` prints lifecycle and turn events to stdout."""
    fn = _fn.replace(model=ScriptedModel([Turn(text="a scripted answer")]))
    async with ai_functions.scope(on_event=print_event):
        _ = await fn("go")
    out = capsys.readouterr().out
    assert "started" in out
    assert "a scripted answer" in out
