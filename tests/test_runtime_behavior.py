"""End-to-end runtime behavior: retry loops, no-result, cancellation, messaging."""

from __future__ import annotations

import asyncio

import pytest
from strands import tool

from ai_functions import ai_function
from ai_functions.ai_thread.postcondition import PostCondition, PostConditionResult
from ai_functions.testing import AwaitBarrier, RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import EventKind
from ai_functions.types.events import MessageUserEvent


@tool
def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@ai_function[str](structured_output=False, tools=[_add])
def _with_tools(prompt: str) -> str:
    return prompt


# ── Post-condition retry loop ─────────────────────────────────────────────


def _make_counted_validator(fail_count: int) -> PostCondition:
    """Return a post-condition that fails the first ``fail_count`` times."""
    state = {"attempts": 0}

    def validate(result: str) -> PostConditionResult:
        state["attempts"] += 1
        if state["attempts"] <= fail_count:
            return PostConditionResult(passed=False, message=f"attempt {state['attempts']} too short")
        return PostConditionResult(passed=True)

    return validate


async def test_post_condition_retries_until_pass() -> None:
    """A validator that fails twice then passes yields two error MESSAGE_USER events."""

    @ai_function[str](structured_output=False)
    def _fn(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        validator = _make_counted_validator(fail_count=2)
        model = ScriptedModel([Turn(text="attempt 1"), Turn(text="attempt 2"), Turn(text="attempt 3")])
        handle = await h.spawn(
            _fn.replace(
                model=model,
                post_conditions=[validator],
                max_attempts=5,
            )
        )
        result = await handle.run("go")
        # Third attempt's text is what wins.
        assert result.strip() == "attempt 3"
        # The prompt itself is one MESSAGE_USER; two retry prompts add two more.
        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert len(user_events) == 3
        # The two retry texts carry the validation-error template and name
        # the failed post-condition.
        assert "[VALIDATION ERROR]" in user_events[1].text
        assert "attempt 1 too short" in user_events[1].text
        assert "[VALIDATION ERROR]" in user_events[2].text
        assert "Please try again" in user_events[2].text


async def test_post_condition_exhausts_max_attempts() -> None:
    """Persistent failures raise after 1 initial try + ``max_attempts`` retries."""
    from ai_functions.ai_thread import AIFunctionError

    @ai_function[str](structured_output=False)
    def _fn(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        # Always fails.
        def always_fail(result: str) -> PostConditionResult:
            del result
            return PostConditionResult(passed=False, message="never good enough")

        # max_attempts=2 -> 3 total runs (1 initial + 2 retries).
        model = ScriptedModel([Turn(text="try 1"), Turn(text="try 2"), Turn(text="try 3")])
        handle = await h.spawn(
            _fn.replace(
                model=model,
                post_conditions=[always_fail],
                max_attempts=2,
            )
        )
        with pytest.raises(AIFunctionError, match=r"not satisfied after 3 attempt\(s\) \(1 initial \+ 2 retries\)"):
            await handle.run("go")


# ── Cancel mid-cycle ──────────────────────────────────────────────────────


async def test_cancel_mid_cycle_rejects_pending_future() -> None:
    """The in-flight cycle's future rejects with ``CancelledError``."""

    @ai_function[str](structured_output=False)
    def _fn(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="won't finish", await_before="hold")])
        handle = await h.spawn(_fn.replace(model=model))
        fut = handle.run("go")
        await h.wait_for(handle.id, EventKind.STARTED)
        await handle.cancel()
        h.release("hold")
        with pytest.raises(asyncio.CancelledError):
            await fut


async def test_cancel_does_not_tear_down_thread() -> None:
    """After cancel, the thread still accepts new work.

    The cancel fires before the model call on the first cycle (the await_before
    barrier is hit first and the cancel_signal is observed at the BeforeModelCall
    boundary), so that Turn stays in the script and is consumed by the second run.
    """

    @ai_function[str](structured_output=False)
    def _fn(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="second cycle ok", await_before="hold")])
        handle = await h.spawn(_fn.replace(model=model))
        fut = handle.run("first")
        await h.wait_for(handle.id, EventKind.STARTED)
        await handle.cancel()
        h.release("hold")
        with pytest.raises(asyncio.CancelledError):
            await fut
        # Second run should succeed on the still-live thread.
        result = await handle.run("second")
        assert result.strip() == "second cycle ok"


# ── notify semantics ──────────────────────────────────────────────


async def test_notify_while_idle_does_not_start_cycle() -> None:
    """A ``notify`` to an idle thread buffers the text; no cycle runs.

    The injected text is observed on the next ``run`` — as a user turn that
    appears BEFORE the run's prompt in the event log.
    """

    @ai_function[str](structured_output=False)
    def _fn(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="first reply"), Turn(text="follow-up reply")])
        handle = await h.spawn(_fn.replace(model=model))
        await handle.run("prompt")
        # Thread is now idle. notify must NOT trigger a cycle —
        # it only appends to the thread's internal buffer.
        await handle.notify("nudge")
        # Give the scheduler a chance: no new COMPLETED should appear.
        await asyncio.sleep(0.05)
        completeds = [e for e in await h.events(handle.id) if e.kind == EventKind.COMPLETED]
        assert len(completeds) == 1, "notify must not trigger a cycle"
        # Run again: the injected "nudge" is observed before the new prompt.
        await handle.run("follow up")
        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert [e.text for e in user_events] == ["prompt", "nudge", "follow up"]


async def test_notify_during_cycle_drains_at_next_boundary() -> None:
    """A message delivered mid-cycle lands on the current cycle's next model-call boundary.

    The first cycle is multi-turn because of a tool call. We deliver an
    ``notify`` during the tool call, and the event-bridge hook
    drains it before the second model call of the same cycle, emitting
    ``MESSAGE_USER`` inline.
    """
    async with RuntimeHarness() as h:
        # First turn runs a tool; streaming is suspended on a barrier so
        # the test can deliver an injection while the agent is mid-turn.
        # The cycle continues with a second model call after the tool
        # result, where the hook drains the inject buffer.
        model = ScriptedModel(
            [
                Turn(
                    text_chunks=("thinking ", AwaitBarrier("mid")),
                    tool_calls=(("_add", {"a": 2, "b": 3}),),
                ),
                Turn(text="final"),
            ]
        )
        handle = await h.spawn(_with_tools.replace(model=model))
        fut = handle.run("first")
        await h.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_TOKEN)
        await handle.notify("by the way")
        h.release("mid")
        await fut

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        # The cycle produced two MESSAGE_USER events: the original prompt
        # plus the mid-cycle injection drained at the next model-call
        # boundary (within the same cycle, before the second turn).
        assert [e.text for e in user_events] == ["first", "by the way"]
        # Ordering: the injection lands AFTER TOOL_RESULT (i.e. at the
        # BeforeModelCallEvent of the second turn).
        kinds = [e.kind for e in await h.events(handle.id)]
        tool_result_idx = kinds.index(EventKind.TOOL_RESULT)
        user_indices = [i for i, k in enumerate(kinds) if k == EventKind.MESSAGE_USER]
        assert user_indices[-1] > tool_result_idx


async def test_post_condition_error_ends_the_cycle() -> None:
    """A ``PostConditionError`` from a post-condition reaches the caller after one attempt, without a retry."""
    from ai_functions.ai_thread import PostConditionError

    @ai_function[str](structured_output=False)
    def _fn(prompt: str) -> str:
        return prompt

    calls: list[str] = []

    def broken(result: str) -> PostConditionResult:
        calls.append(result)
        raise PostConditionError("the checker is unavailable")

    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="try 1"), Turn(text="try 2"), Turn(text="try 3")])
        handle = await h.spawn(_fn.replace(model=model, post_conditions=[broken], max_attempts=2))
        with pytest.raises(PostConditionError, match="the checker is unavailable"):
            await handle.run("go")
        assert len(calls) == 1
