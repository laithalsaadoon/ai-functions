"""``CustomEvent`` routing survives serialization and a real WebSocket hop.

A custom event declares the routing fields every other event declares, so the
three places routing is read all agree on it: ``EVENT_ADAPTER`` keeps
``thread_id`` in the wire dict, a filtered subscription over a
``CoordinatorClient`` matches it, and ``get_events(thread_id)`` returns it under
that thread. A client-side ``append_event`` of a routed custom event is accepted
by the remote coordinator; an unrouted one is rejected and the rejection shows
up on ``client.append_errors``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field, ValidationError, create_model, field_serializer
from pydantic_core import PydanticSerializationError

from ai_functions.network import CoordinatorClient, CoordinatorEndpoint
from ai_functions.network.channel import EVENT_ADAPTER
from ai_functions.testing import RuntimeHarness
from ai_functions.types import CustomEvent, Event, ThreadId
from ai_functions.types.events import BaseEvent, StartedEvent
from ai_functions.types.ids import MessageId

_TID = ThreadId("thr-custom-wire")


class _Annotated(CustomEvent):
    """A user subclass that declares its own field next to the routing ones."""

    step: str = ""


async def _until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Poll ``predicate`` until true, or fail the test after ``timeout``."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within timeout")


# ── a: the adapter keeps routing on a stamped custom event ────────────────


def test_a1_dump_includes_thread_id() -> None:
    """The wire dict for a stamped custom event carries ``thread_id``."""
    stamped = CustomEvent(kind="my_kind", payload={"a": 1}).model_copy(update={"thread_id": _TID})
    dumped = EVENT_ADAPTER.dump_python(stamped)
    assert dumped["thread_id"] == _TID


def test_a2_round_trip_preserves_routing() -> None:
    """Every routing field survives dump → validate through the union adapter."""
    original = CustomEvent(
        kind="my_kind",
        thread_id=_TID,
        thread_name="worker-a",
        message_id=MessageId("msg-1"),
        payload={"a": 1},
    )
    reparsed = EVENT_ADAPTER.validate_python(EVENT_ADAPTER.dump_python(original))
    assert isinstance(reparsed, CustomEvent)
    assert reparsed.thread_id == _TID
    assert reparsed.thread_name == "worker-a"
    assert reparsed.message_id == MessageId("msg-1")
    assert reparsed.id == original.id
    assert reparsed.payload == {"a": 1}


def test_a3_subclass_declared_fields_round_trip() -> None:
    """A subclass's own field stays declared; routing rides alongside it."""
    original = _Annotated(kind="my_kind", thread_id=_TID, step="validate", payload={"a": 1})
    dumped = _Annotated.model_validate(original.model_dump())
    assert dumped.step == "validate"
    assert dumped.thread_id == _TID
    assert dumped.payload == {"a": 1}
    # Through the ``Event`` union the subclass degrades to ``CustomEvent``;
    # ``step`` is an undeclared key there, so it lands in the payload.
    via_union = EVENT_ADAPTER.validate_json(EVENT_ADAPTER.dump_json(original))
    assert isinstance(via_union, CustomEvent)
    assert via_union.thread_id == _TID
    assert via_union.payload == {"a": 1, "step": "validate"}
    assert _Annotated.model_validate(via_union.model_dump()) == original


@pytest.mark.parametrize("json_mode", [False, True])
def test_event_annotation_preserves_subclass_fields_in_containers(json_mode: bool) -> None:
    """Any Event-typed slot inherits the serialization policy without flags."""

    class Envelope(BaseModel):
        event: Event
        events: list[Event]
        batches: dict[str, list[Event]]

    original = _Annotated(kind="progress", thread_id=_TID, step="validate", payload={"completed": 3})
    started = StartedEvent(thread_id=_TID)
    envelope = Envelope(event=original, events=[started, original], batches={"batch": [original]})
    restored = (
        Envelope.model_validate_json(envelope.model_dump_json())
        if json_mode
        else Envelope.model_validate(envelope.model_dump())
    )
    assert restored.events[0] == started
    assert isinstance(restored.events[0], StartedEvent)
    for event in (restored.event, restored.events[1], restored.batches["batch"][0]):
        assert type(event) is CustomEvent
        assert event.id == original.id
        assert event.thread_id == _TID
        assert event.payload == {"completed": 3, "step": "validate"}
        assert _Annotated.model_validate(event.model_dump()) == original


def test_event_annotation_respects_custom_serializers_and_excluded_fields() -> None:
    class Progress(CustomEvent):
        completed: int
        local_only: str = Field(default="not-on-wire", exclude=True)

        @field_serializer("completed")
        def _format_completed(self, value: int) -> str:
            return f"{value} steps"

    original = Progress(kind="progress", thread_id=_TID, completed=3)
    dumped = EVENT_ADAPTER.dump_python(original, mode="json")
    assert dumped["completed"] == "3 steps"
    assert "local_only" not in dumped
    assert "payload" not in dumped
    restored = EVENT_ADAPTER.validate_python(dumped)
    assert type(restored) is CustomEvent
    assert restored.payload == {"completed": "3 steps"}


def test_event_annotation_still_validates_metadata() -> None:
    with pytest.raises(ValidationError, match="timestamp"):
        EVENT_ADAPTER.validate_python({"kind": "progress", "timestamp": "invalid", "completed": 3})


def test_payload_cannot_shadow_a_subclass_field() -> None:
    with pytest.raises(ValidationError, match="conflict.*step"):
        _Annotated(kind="my_kind", step="validate", payload={"step": "overwrite"})


@pytest.mark.parametrize("field", list(BaseEvent.model_fields))
def test_subclass_cannot_redefine_metadata(field: str) -> None:
    with pytest.raises(TypeError, match=f"cannot redefine BaseEvent fields: {field}"):
        create_model("InvalidEvent", __base__=CustomEvent, **{field: (str, "application-value")})


def test_subclass_cannot_inherit_conflicting_metadata_from_an_application_model() -> None:
    class ApplicationModel(BaseModel):
        id: int

    with pytest.raises(TypeError, match="cannot redefine BaseEvent fields: id"):

        class InvalidEvent(ApplicationModel, CustomEvent):
            pass


@pytest.mark.parametrize("reserved", ["id", "thread_id", "kind", "payload"])
@pytest.mark.parametrize("alias_type", ["alias", "validation_alias", "serialization_alias"])
def test_subclass_alias_cannot_use_a_framework_name(reserved: str, alias_type: str) -> None:
    with pytest.raises(TypeError, match="alias conflicting with framework fields"):
        create_model(
            "InvalidEvent",
            __base__=CustomEvent,
            application_id=(str, Field(**{alias_type: reserved})),
        )


@pytest.mark.parametrize("alias", [AliasChoices("item_id", "id"), AliasPath("thread_id", "value")])
def test_subclass_validation_alias_cannot_consume_metadata(alias: AliasChoices | AliasPath) -> None:
    with pytest.raises(TypeError, match="alias conflicting with framework fields"):
        create_model("InvalidEvent", __base__=CustomEvent, application_id=(str, Field(validation_alias=alias)))


def test_alias_generator_cannot_rename_metadata() -> None:
    with pytest.raises(TypeError, match="alias conflicting with framework fields"):

        class InvalidEvent(CustomEvent):
            model_config = {"alias_generator": str.upper}


def test_application_alias_remains_available_to_pydantic() -> None:
    class Progress(CustomEvent):
        completed: int = Field(alias="count")

    event = Progress.model_validate({"kind": "progress", "thread_id": _TID, "count": "3", "unit": "steps"})
    assert event.completed == 3
    assert event.payload == {"unit": "steps"}
    dumped = event.model_dump(by_alias=True)
    assert dumped["count"] == 3
    assert dumped["thread_id"] == _TID
    assert "payload" not in dumped
    assert Progress.model_validate(dumped) == event
    with pytest.raises(ValidationError, match="conflict.*count"):
        Progress(kind="progress", count=3, payload={"count": 4})


@pytest.mark.parametrize("copy_update", [False, True])
@pytest.mark.parametrize("use_adapter", [False, True])
def test_serialization_rejects_conflicts_introduced_after_validation(copy_update: bool, use_adapter: bool) -> None:
    event = CustomEvent(kind="my_kind", thread_id=_TID)
    if copy_update:
        event = event.model_copy(update={"payload": {"id": "application-id"}})
    else:
        event.payload["id"] = "application-id"
    with pytest.raises(PydanticSerializationError, match="conflict.*id"):
        if use_adapter:
            EVENT_ADAPTER.dump_json(event)
        else:
            event.model_dump_json()


# ── b: the runtime gate stamps the declared field ─────────────────────────


async def test_b1_route_event_stamps_custom_event() -> None:
    """An unrouted custom event leaves the runtime gate carrying the thread id."""
    async with RuntimeHarness() as h:
        h.worker._route_event(  # noqa: SLF001 -- the gate is the unit under test
            CustomEvent(kind="my_kind", payload={"a": 1}),
            thread_id=_TID,
            source="thread",
        )
        stored = await h.coordinator.get_events(_TID)
        assert [e.thread_id for e in stored] == [_TID]


# ── c: a real endpoint + client agree on the routing ─────────────────────


async def test_c1_filtered_subscription_receives_custom_event() -> None:
    """``client.on(cb, thread_id=...)`` fires for a custom event from the endpoint."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            received: list[Event] = []
            with client.on(received.append, thread_id=_TID):
                endpoint.coordinator.append_event(
                    CustomEvent(kind="my_kind", thread_id=_TID, payload={"a": 1}),
                )
                await _until(lambda: bool(received))
            event = received[0]
            assert isinstance(event, CustomEvent)
            assert event.thread_id == _TID
            assert event.payload == {"a": 1}


async def test_c2_get_events_returns_routed_custom_event() -> None:
    """``client.get_events(thread_id)`` returns the custom event with its routing."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            endpoint.coordinator.append_event(
                CustomEvent(kind="my_kind", thread_id=_TID, payload={"a": 1}),
            )
            replayed = await client.get_events(_TID)
            assert [e.thread_id for e in replayed] == [_TID]
            assert [e.kind for e in replayed] == ["my_kind"]


async def test_c3_client_append_event_lands_on_the_endpoint() -> None:
    """A routed custom event appended from the client reaches the server log."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            client.append_event(CustomEvent(kind="from_client", thread_id=_TID, payload={"a": 1}))
            await _until(lambda: bool(endpoint.coordinator._events.get(_TID)))  # noqa: SLF001
            stored = await endpoint.coordinator.get_events(_TID)
            assert [(e.kind, e.thread_id) for e in stored] == [("from_client", _TID)]
            assert client.append_errors == ()


async def test_c4_unrouted_append_is_recorded_on_append_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected append is observable without watching the log by hand."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        client = await CoordinatorClient.connect(endpoint.url)
        async with client:
            with caplog.at_level("ERROR", logger="ai_functions.network.client"):
                client.append_event(CustomEvent(kind="unrouted", payload={"a": 1}))
                await _until(lambda: bool(client.append_errors))
            assert "thread_id" in str(client.append_errors[0])
            assert "append_event RPC failed" in caplog.text
            assert await endpoint.coordinator.get_events(_TID) == []


async def test_client_append_preserves_subclass_fields_and_user_nesting() -> None:
    """A subclass keeps its fields through the RPC, broadcast and replay paths."""
    async with CoordinatorEndpoint() as endpoint:
        await endpoint.start(host="127.0.0.1", port=0)
        async with await CoordinatorClient.connect(endpoint.url) as client:
            received: list[Event] = []
            event = _Annotated(
                kind="my_kind",
                thread_id=_TID,
                step="validate",
                payload={"item_id": "item-1", "data": {"id": "source-id", "thread_id": "source-thread"}},
            )
            with client.on(received.append, thread_id=_TID):
                client.append_event(event)
                await _until(lambda: bool(received))
            expected = CustomEvent.model_validate(event.model_dump())
            assert received == [expected]
            assert await client.get_events(_TID) == [expected]
            dumped = received[0].model_dump()
            assert _Annotated.model_validate(dumped) == event
            assert dumped["step"] == "validate"
            assert dumped["item_id"] == "item-1"
            assert dumped["data"] == {"id": "source-id", "thread_id": "source-thread"}
            assert "payload" not in dumped
