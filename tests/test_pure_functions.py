"""Pure-function correctness: serialize/deserialize, CustomEvent, I2."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from ai_functions import ai_function
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import Event, ThreadId
from ai_functions.types.events import CustomEvent, MessageUserEvent, StartedEvent

# ── serialize_result / deserialize_result round-trip ──────────────────────


@pytest.mark.parametrize(
    "output_type, value",
    [
        (str, "hello"),
        (int, 42),
        (float, 3.14),
        (list[int], [1, 2, 3]),
        (dict[str, int], {"a": 1, "b": 2}),
    ],
)
async def test_serialize_deserialize_roundtrip_primitives(output_type: type, value: Any) -> None:  # type: ignore[type-arg]
    """Primitives, lists, and dicts round-trip through the serializer."""

    @ai_function[output_type](structured_output=True)
    def _fn(x: str) -> str:
        return x

    async with RuntimeHarness() as h:
        # The harness's AIThread initialization needs a live worker; the
        # thread itself exposes serialize/deserialize on the instance.
        handle = await h.spawn(_fn)
        thread = h.worker._threads[handle.id]  # noqa: SLF001 -- accessing for a pure test
        payload = thread.serialize_result(value)
        recovered = thread.deserialize_result(payload)
        assert recovered == value


class _PydModel(BaseModel):
    name: str
    count: int


async def test_serialize_deserialize_roundtrip_pydantic() -> None:
    """Pydantic models round-trip preserving field values and type."""

    @ai_function[_PydModel](structured_output=True)
    def _fn(x: str) -> str:
        return x

    async with RuntimeHarness() as h:
        handle = await h.spawn(_fn)
        thread = h.worker._threads[handle.id]  # noqa: SLF001
        value = _PydModel(name="foo", count=7)
        payload = thread.serialize_result(value)
        recovered = thread.deserialize_result(payload)
        assert isinstance(recovered, _PydModel)
        assert recovered == value


async def test_deserialize_malformed_payload_raises() -> None:
    """A malformed payload raises ``AIFunctionError``."""
    from ai_functions.ai_thread import AIFunctionError

    @ai_function[int](structured_output=True)
    def _fn(x: str) -> str:
        return x

    async with RuntimeHarness() as h:
        handle = await h.spawn(_fn)
        thread = h.worker._threads[handle.id]  # noqa: SLF001
        with pytest.raises(AIFunctionError):
            thread.deserialize_result("{not json")


# ── CustomEvent discriminated-union fallback ──────────────────────────────


def test_unknown_kind_parses_as_custom_event() -> None:
    """An unknown ``kind`` routes the discriminated union to ``CustomEvent``."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python({"kind": "my_custom_kind"})
    assert isinstance(parsed, CustomEvent)
    assert parsed.kind == "my_custom_kind"


def test_custom_event_round_trips() -> None:
    """``CustomEvent`` serializes and re-parses to an equal instance."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    original = CustomEvent(kind="my_kind", payload={"a": "hello", "b": 1})
    dumped = ta.dump_python(original)
    reparsed = ta.validate_python(dumped)
    assert isinstance(reparsed, CustomEvent)
    assert reparsed.kind == original.kind
    assert reparsed.payload == original.payload


def test_known_kind_does_not_hit_custom_event() -> None:
    """A known SystemEvent ``kind`` parses to the matching concrete class."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python({"kind": "message_user", "thread_id": "thr-xyz", "text": "hi"})
    assert isinstance(parsed, MessageUserEvent)
    assert parsed.text == "hi"


def test_fieldless_system_event_round_trips_to_its_own_class() -> None:
    """A built-in kind whose variant declares no extra field is not swallowed.

    ``StartedEvent`` sets exactly the routing fields ``CustomEvent`` also
    declares, so a union that scores members by fields-set would hand its dict
    to ``CustomEvent``; the union is left-to-right precisely to prevent that.
    """
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    original = StartedEvent(thread_id=ThreadId("thr-1"), thread_name="agent")
    reparsed = ta.validate_json(ta.dump_json(original))
    assert isinstance(reparsed, StartedEvent)
    assert reparsed == original


def test_custom_event_collects_extra_fields_into_payload() -> None:
    """Extra top-level fields are folded into ``payload`` by the before-validator."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python({"kind": "my_kind", "a": "hello", "b": 1})
    assert isinstance(parsed, CustomEvent)
    assert parsed.payload == {"a": "hello", "b": 1}


def test_custom_event_serializes_flat() -> None:
    """``model_dump`` emits the declared fields plus the payload, flat."""
    event = CustomEvent(kind="my_kind", thread_id=ThreadId("thr-1"), payload={"a": "hello", "b": 1})
    dumped = event.model_dump()
    assert dumped == {
        "id": event.id,
        "timestamp": event.timestamp,
        "thread_id": "thr-1",
        "thread_name": None,
        "message_id": None,
        "kind": "my_kind",
        "a": "hello",
        "b": 1,
    }
    assert "payload" not in dumped


def test_custom_event_flat_wire_format_round_trips() -> None:
    """Flat wire dict → validator → serializer keeps every key it was given."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    flat = {"kind": "my_kind", "thread_id": "thr-1", "a": "hello", "b": 1}
    parsed = ta.validate_python(flat)
    dumped = ta.dump_python(parsed)
    assert {k: dumped[k] for k in flat} == flat
    assert ta.validate_python(dumped) == parsed


def test_custom_event_routing_fields_are_declared_not_payload() -> None:
    """A routing key stays out of ``payload`` and survives serialization."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    event = CustomEvent(kind="my_kind", thread_id=ThreadId("thr-7"), payload={"a": 1})
    assert event.thread_id == ThreadId("thr-7")
    assert event.payload == {"a": 1}
    dumped = ta.dump_python(event)
    assert dumped["thread_id"] == "thr-7"
    assert ta.validate_python(dumped).thread_id == ThreadId("thr-7")


@pytest.mark.parametrize("key", ["id", "timestamp", "thread_id", "thread_name", "message_id", "kind", "payload"])
def test_custom_event_rejects_shadowing_payload_keys(key: str) -> None:
    """Application data cannot overwrite framework fields when flattened."""
    with pytest.raises(ValidationError, match=f"conflict.*{key}.*rename or nest"):
        CustomEvent(kind="my_kind", thread_id=ThreadId("thr-7"), payload={key: "application-value"})


def test_custom_event_application_nesting_round_trips() -> None:
    """Users can retain source field names inside their own nested object."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    item = {"id": "item-1", "thread_id": "source-thread", "kind": "source", "payload": {"text": "t"}}
    event = CustomEvent(kind="my_kind", thread_id=ThreadId("thr-7"), payload={"item": item})
    dumped = ta.dump_python(event)
    assert dumped["id"] == event.id
    assert dumped["thread_id"] == "thr-7"
    assert dumped["item"] == item
    assert "payload" not in dumped
    assert ta.validate_python(dumped) == event


def test_custom_event_is_frozen() -> None:
    """``CustomEvent`` is immutable like every other event; copies re-route it."""
    event = CustomEvent(kind="my_kind")
    with pytest.raises(Exception):  # noqa: B017, PT011 -- pydantic raises ValidationError
        event.thread_id = ThreadId("thr-injected")  # type: ignore[misc]
    assert event.model_copy(update={"thread_id": ThreadId("thr-1")}).thread_id == ThreadId("thr-1")


def test_custom_event_explicit_payload_still_collects_extras() -> None:
    """Input may mix an explicit ``payload`` with extras; extras win on key clash."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python(
        {"kind": "my_kind", "payload": {"a": "from_payload", "c": 3}, "a": "from_extra", "b": 2}
    )
    assert isinstance(parsed, CustomEvent)
    # ``a`` is specified both ways — the extra takes precedence; ``c`` only
    # exists in the explicit payload; ``b`` only as an extra.
    assert parsed.payload == {"a": "from_extra", "c": 3, "b": 2}


# ── I2: event immutability ────────────────────────────────────────────────


async def test_events_are_frozen() -> None:
    """Assigning to a field on an emitted event raises (pydantic frozen)."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])

        @ai_function[str](structured_output=False)
        def _fn(x: str) -> str:
            return x

        handle = await h.spawn(_fn.replace(model=model))
        await handle.run("x")
        events = await h.events(handle.id)
        assert events  # sanity
        sample = events[0]
        with pytest.raises(Exception):  # noqa: B017, PT011 -- pydantic raises ValidationError
            sample.thread_id = ThreadId("thr-injected")  # type: ignore[misc]
