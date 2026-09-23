"""Opaque SDK fields stay inside application data, outside event metadata."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import SystemMessage  # noqa: E402

from ai_functions.claude_code.claude_code import ClaudeAgent, ClaudeAgentThread  # noqa: E402
from ai_functions.runtime import InMemoryCoordinator  # noqa: E402
from ai_functions.types import CustomEvent, Event, ThreadContext, ThreadId  # noqa: E402


@dataclass
class _FutureMessage:
    id: str = "source-id"
    thread_id: str = "source-thread"


@dataclass
class _FutureSystemMessage(SystemMessage):
    id: str = "source-id"
    thread_id: str = "source-thread"


@pytest.mark.parametrize("system_message", [False, True])
def test_sdk_message_fields_cannot_overwrite_event_metadata(system_message: bool) -> None:
    events: list[Event] = []
    ctx = ThreadContext(
        thread_id=ThreadId("runtime-thread"),
        coordinator=InMemoryCoordinator(),
        on_event=events.append,
        on_interrupt=None,  # type: ignore[arg-type] -- this mapping never requests approval
        pause_signal=asyncio.Event(),
        cancel_signal=asyncio.Event(),
    )
    message = _FutureSystemMessage(subtype="future", data={"kind": "sdk-data"}) if system_message else _FutureMessage()
    ClaudeAgentThread(ClaudeAgent())._emit_events_for(message, ctx)  # type: ignore[arg-type]
    event = events[0]
    assert isinstance(event, CustomEvent)
    assert event.kind == ("claude_system_future" if system_message else "claude_message")
    assert event.payload["message"]["id"] == "source-id"
    assert event.payload["message"]["thread_id"] == "source-thread"
    assert event.id != "source-id"
    assert event.thread_id is None
    assert "payload" not in event.model_dump()
