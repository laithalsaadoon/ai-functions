"""Offline tests for ``methods/gated.py``: the gate as a post-condition, and the beam that filters.

Thirty tests over the gate's own surface, plus the guards this module adds: the ``MethodError`` leaves, the
gate-after-proposal ordering (G1) asserted on the wire, and a static import boundary for gated.py.

Two claims, and they are checked differently on purpose. The *sharp* properties of the
post-condition (a rejection raises with a usable message, a gate fault does not read as a verdict,
a rejection is recorded and a fault is not) are asserted by calling ``admits`` directly, because
what needs proving is what it raises. The *wiring* is then asserted through a real cycle: that the
message actually reaches the model as a retry prompt is a claim about ``ai_thread``, not about this
code, and only a real thread with a recording model can settle it.

Retirement is asserted from the coordinator's own registry rather than from ``MethodThread.live``.
A wrapper's local liveness flag can desync from the runtime, so the flag alone would prove the
object thinks it is retired, which is the weaker of the two claims and the one that survives a
broken ``finally``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from strands.models import Model

from ai_functions.experimental.methods import (
    Gate,
    GatedProposer,
    GateFaultError,
    GateRejectionError,
    MethodError,
    Verdict,
    WiringError,
    ai_method,
)
from ai_functions.experimental.methods import gated as methods_gated
from ai_functions.testing import RuntimeHarness, Turn

from .methods_wire import Capturing
from .test_methods_boundary import OWN, _imports

# ── Fixtures ──
#
# Module level, all of them: ``compile_ai_method`` resolves annotations with
# ``typing.get_type_hints`` against module globals, so a function-local output type cannot be
# resolved at compile time.


class Pick(BaseModel):
    """The proposal shape: the candidate, and the reasoning that produced it."""

    value: int = Field(description="The proposed value, which is what the gate judges.")
    evidence: str = Field(description="What led to this value. Not the subject of the gate.")


@dataclass(frozen=True)
class Threshold:
    """A trivial ``Verdict``: admitted above a floor, and a report that names the floor."""

    value: int
    floor: int

    @property
    def ok(self) -> bool:
        return self.value >= self.floor

    def report_text(self) -> str:
        return f"value {self.value} is below the floor of {self.floor}"


def rejected_values(picker: GatedProposer) -> list[int]:
    """The values ``picker`` rejected, in order; every gate in this module returns a ``Threshold``."""
    values: list[int] = []
    for verdict in picker.rejected:
        assert isinstance(verdict, Threshold)
        values.append(verdict.value)
    return values


def floor_gate(floor: int) -> Gate:
    """A gate over the whole ``Pick``. Three lines, which is the point of injecting it."""

    def gate(candidate: Pick) -> Threshold:
        return Threshold(candidate.value, floor)

    return gate


def value_gate(floor: int) -> Gate:
    """The same gate over an already-extracted ``int``, for the ``candidate_of`` override."""

    def gate(candidate: int) -> Threshold:
        return Threshold(candidate, floor)

    return gate


class Picker(GatedProposer):
    """The default shape: the response *is* the candidate, so ``candidate_of`` is untouched."""

    name = "picker"

    @ai_method(Pick, description="Propose a value, judged by the gate", max_attempts=3)
    def propose(self, hint: str) -> Pick:
        """Propose a value given {hint}."""
        ...


class ValuePicker(Picker):
    """Overrides ``candidate_of`` to hand the gate the field, not the whole response."""

    name = "value-picker"

    def candidate_of(self, response: Pick) -> int:
        return response.value


def must_be_even(response: Pick) -> None:
    """A post-condition declared on the method itself, beside the gate."""
    if response.value % 2:
        raise ValueError(f"{response.value} is odd; propose an even value")


class EvenPicker(GatedProposer):
    name = "even"

    @ai_method(Pick, description="Propose an even value", max_attempts=3, post_conditions=[must_be_even])
    def propose(self, hint: str) -> Pick:
        """Propose a value given {hint}."""
        ...


class SeedPicker(GatedProposer):
    """A propose parameter named like a ``propose_k`` keyword."""

    name = "seeded"

    @ai_method(Pick, description="Propose a value from a seed", max_attempts=1)
    def propose(self, seed: str) -> Pick:
        """Propose a value from {seed}."""
        ...


class KPicker(GatedProposer):
    """A propose parameter named like ``propose_k``'s positional-only ``k``."""

    name = "kpicker"

    @ai_method(Pick, description="Propose one of k values", max_attempts=1)
    def propose(self, k: int) -> Pick:
        """Propose one of {k} values."""
        ...


class CollidingPicker(GatedProposer):
    """A proposer whose propose parameter is named ``response``, like ``admits``'s first one.

    The collision the guard exists to catch: the runtime would fill that slot twice.
    """

    name = "colliding"

    @ai_method(Pick, description="Propose a value", max_attempts=1)
    def propose(self, response: str) -> Pick:
        """Propose a value given {response}."""
        ...


class InjectingPicker(GatedProposer):
    """A validator naming a propose parameter *second*: the injection the runtime documents.

    The guard must allow this, or it forbids the one pattern ``postcondition.py``'s own docstring
    describes ("if any argument names in the signature of the callable match keys in
    bound_args, the callable also receives those values as keyword arguments").
    """

    name = "injecting"

    def __init__(self, gate: Gate) -> None:
        super().__init__(gate)
        self.hints: list[str] = []

    # The default keeps the override compatible with ``admits(response)``; the runtime always injects ``hint``.
    def admits(self, response: Pick, hint: str = "") -> None:
        self.hints.append(hint)
        super().admits(response)

    @ai_method(Pick, description="Propose a value", max_attempts=1)
    def propose(self, hint: str) -> Pick:
        """Propose a value given {hint}."""
        ...


def picks(values: list[int]) -> Capturing:
    """A recording model that proposes ``values`` in order, one ``Pick`` per call."""
    return Capturing([Turn(tool_calls=(("Pick", {"value": v, "evidence": f"because {v}"}),)) for v in values])


async def live_threads(harness: RuntimeHarness) -> list[str]:
    """Thread ids the *coordinator* still has registered: the runtime's own answer.

    Measured: deregistration is complete by the time ``retire()`` returns, so no polling is
    needed. ``asyncio.sleep(0)`` is a yield, not a wait, and is here so this reads as a
    settled-state question rather than a race won by luck.
    """
    await asyncio.sleep(0)
    return [info.thread_id for info in await harness.coordinator.list_threads()]


def _scripting(agent: GatedProposer, model: Model) -> Any:
    """Bind a scripted model by replacing ``compiled`` on the *instance*.

    ``spawn`` compiles through ``self.compiled`` precisely so this reaches live threads; going to
    ``compile_ai_method`` directly would silently bypass the binding and reach a real model.
    """
    original = type(agent).compiled

    def compiled(name: str, **overrides: Any) -> Any:
        overrides.setdefault("model", model)
        return original(agent, name, **overrides)

    return compiled


# ── The protocols ──


def test_the_trivial_verdict_and_gate_satisfy_the_protocols() -> None:
    """The trivial verdict and gate satisfy the protocols. Guard the guard: if ``Threshold`` were not a
    ``Verdict``, every test below would be exercising the skeleton against a shape no real gate has to match."""
    assert isinstance(Threshold(1, 5), Verdict)
    assert isinstance(floor_gate(5), Gate)


# ── The post-condition path: refusal is the default, the report is the feedback ──


def test_an_admitted_candidate_passes_quietly_and_records_nothing() -> None:
    """G2: the negative half of the ledger
    claim, ``rejected`` only grows on a rejection."""
    picker = Picker(floor_gate(5))
    assert picker.admits(Pick(value=7, evidence="above the floor")) is None
    assert picker.rejected == []


def test_a_rejected_candidate_raises_with_the_report_and_the_reask() -> None:
    """G2: the gate's own report is the
    message, so the model is told the reason and the action. The refusal is the
    ``GateRejectionError`` leaf of ``MethodError``, never a bare ``AssertionError``."""
    picker = Picker(floor_gate(5))

    with pytest.raises(GateRejectionError) as raised:
        picker.admits(Pick(value=1, evidence="below the floor"))

    message = str(raised.value)
    assert "value 1 is below the floor of 5" in message
    assert "Propose a different candidate." in message
    assert rejected_values(picker) == [1]


def test_every_rejection_lands_in_the_ledger_in_order() -> None:
    """G2: the evidence the gate fired. A silently
    re-asked loop that later succeeded looks identical, from the outside, to a loop whose gate never fired."""
    picker = Picker(floor_gate(5))
    for value in (1, 3, 2):
        with pytest.raises(GateRejectionError):
            picker.admits(Pick(value=value, evidence="low"))

    assert rejected_values(picker) == [1, 3, 2]


def test_the_reask_suffix_is_overridable_without_touching_the_mechanism() -> None:
    """The reask suffix is overridable without touching the mechanism: domain wording is the subclass's;
    the placement is the base's."""

    class Loud(Picker):
        REASK = "Propose a different value. Going below the floor removes the only term."

    with pytest.raises(GateRejectionError, match=r"removes the only term"):
        Loud(floor_gate(5)).admits(Pick(value=0, evidence="x"))


async def test_the_rejection_reaches_the_model_as_the_next_attempts_prompt() -> None:
    """G1, G2: the wiring, asserted from
    what the *model* received.

    ``admits`` raising is this module's claim; the message becoming a retry prompt is ``ai_thread``'s, and only a
    real cycle settles it. The scripted model proposes a rejected value and then an admitted one, so a working
    post-condition means exactly two calls and the second one carrying the first one's report.
    """
    async with RuntimeHarness():
        picker = Picker(floor_gate(5))
        model = picks([1, 9])

        result = await picker.gated(model=model)(hint="go")

        assert result.value == 9
        assert len(model.contexts) == 2

        retry = model.prompts(1)
        assert any("[VALIDATION ERROR]" in prompt for prompt in retry)
        assert any("value 1 is below the floor of 5" in prompt for prompt in retry)
        assert any("Propose a different candidate." in prompt for prompt in retry)

        assert rejected_values(picker) == [1]


async def test_gated_keeps_the_gate_first_and_the_callers_own_conditions() -> None:
    """G1: a hand-written
    ``compiled(..., post_conditions=[mine])`` drops the gate silently. This cannot: the gate is prepended, so an
    added check is additive rather than a replacement."""

    def also(response: Pick) -> None:
        if not response.evidence:
            raise AssertionError("evidence is required")

    picker = Picker(floor_gate(5))
    conditions = picker.gated(post_conditions=[also]).config.post_conditions

    assert list(conditions) == [picker.admits, also]


async def test_the_gate_judges_each_proposal_after_the_model_produced_it_and_never_before() -> None:
    """G1, added: the gate is a *post*-condition. Interleaving on the wire is model call, then gate on exactly the
    value that call proposed, then the next model call; the gate never runs before a proposal exists and never sees
    a value the model did not produce."""
    order: list[str] = []

    class Ordered(Capturing):
        def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any) -> Any:
            order.append("model")
            return super().stream(messages, tool_specs, *args, **kwargs)

    def gate(candidate: Pick) -> Threshold:
        order.append(f"gate:{candidate.value}")
        return Threshold(candidate.value, 5)

    async with RuntimeHarness():
        picker = Picker(gate)
        model = Ordered([Turn(tool_calls=(("Pick", {"value": v, "evidence": "e"}),)) for v in (1, 9)])
        result = await picker.gated(model=model)(hint="go")

    assert result.value == 9
    assert order == ["model", "gate:1", "model", "gate:9"]


# ── The unwanted behavior: a bug must not wear a verdict's clothes ──


def test_a_fault_in_the_gate_does_not_masquerade_as_a_rejection() -> None:
    """G3: any exception inside a post-condition
    is reported to the model as a validation failure.

    So a ``KeyError`` in the gate looks identical to a refused proposal and burns every retry on a bug the model
    cannot fix. The gate call is wrapped and an internal failure re-raised as a message that says it is internal,
    and, just as important, it is not recorded as evidence against a proposal it never judged.
    """

    def broken(candidate: Pick) -> Threshold:
        raise KeyError("floor")

    picker = Picker(broken)

    with pytest.raises(GateFaultError) as raised:
        picker.admits(Pick(value=7, evidence="fine"))

    message = str(raised.value)
    assert "fault in the gate rather than a verdict" in message
    assert "KeyError" in message
    assert picker.rejected == [], "a gate fault is not evidence against the proposal"


def test_a_malformed_verdict_is_a_fault_not_a_rejection() -> None:
    """G3: a gate can return without raising and still
    hand back garbage, a verdict whose ``ok`` or ``report_text`` crashes (built around a candidate the gate should
    have refused, say). Reading it is part of judging: the crash must surface as a gate fault, not land in
    ``rejected`` and not reach the model dressed as a validation message."""

    class Broken:
        @property
        def ok(self) -> bool:
            raise TypeError("unsupported format string")

        def report_text(self) -> str:  # pragma: no cover — ok raises first
            return "never reached"

    picker = Picker(lambda candidate: Broken())  # type: ignore[arg-type, return-value]

    with pytest.raises(GateFaultError) as raised:
        picker.admits(Pick(value=7, evidence="fine"))

    message = str(raised.value)
    assert "fault in the gate rather than a verdict" in message
    assert "TypeError" in message
    assert picker.rejected == [], "a verdict nobody could read is not evidence"


async def test_a_malformed_verdict_is_a_fault_on_the_async_path_too() -> None:
    """G3: ``judge`` reads ``ok`` as well; the same
    garbage must read the same way there."""

    class Broken:
        @property
        def ok(self) -> bool:
            raise TypeError("unsupported format string")

        def report_text(self) -> str:  # pragma: no cover
            return "never reached"

    picker = Picker(lambda candidate: Broken())  # type: ignore[arg-type, return-value]

    with pytest.raises(GateFaultError, match=r"fault in the gate"):
        await picker.judge(Pick(value=7, evidence="fine"))
    assert picker.rejected == []


def test_a_broken_candidate_extractor_is_a_fault_not_a_verdict() -> None:
    """G3: ``candidate_of`` is a subclass hook, so
    a typo in an override is exactly as likely as a bug in the gate, and without the wrap it would surface as a raw
    AttributeError that the runtime reports to the model as a validation failure, burning every retry."""

    class Typoed(GatedProposer):
        name = "typoed"

        @ai_method(Pick, description="propose")
        def propose(self, hint: str) -> Pick:
            """Pick, given {hint}."""
            ...

        def candidate_of(self, response: Pick) -> int:
            return response.weight  # type: ignore[attr-defined] — the typo under test

    picker = Typoed(lambda candidate: Threshold(candidate, 5))

    with pytest.raises(GateFaultError) as raised:
        picker.admits(Pick(value=7, evidence="fine"))

    message = str(raised.value)
    assert "fault in the candidate extractor" in message
    assert "AttributeError" in message
    assert picker.rejected == []


def test_an_extra_post_condition_gets_the_collision_guard_too() -> None:
    """G5: ``gated()`` advertises extras as the safe
    wiring path; an extra whose first parameter shadows a propose parameter hits the same double-fill TypeError the
    admits guard exists for, so the guard must cover it."""

    def extra(hint: object) -> None:  # first param collides with propose's `hint`
        return None

    picker = Picker(lambda candidate: Threshold(candidate.value, 5))

    with pytest.raises(WiringError, match=r"'extra'.*'hint'|'hint'.*'extra'"):
        picker.gated("propose", post_conditions=[extra])


async def test_judge_refuses_a_rejection_whose_report_cannot_be_rendered() -> None:
    """G3: the beam path must not admit an
    unreadable rejection into the ledger; rendering it later (a training summary, say) would crash far from the gate
    that produced it."""

    class HalfBroken:
        ok = False

        def report_text(self) -> str:
            raise TypeError("unsupported format string")

    picker = Picker(lambda candidate: HalfBroken())  # type: ignore[arg-type, return-value]

    with pytest.raises(GateFaultError, match=r"fault in the gate"):
        await picker.judge(Pick(value=1, evidence="x"))
    assert picker.rejected == [], "an unreadable rejection is not evidence"


def test_an_async_gate_on_the_sync_path_is_refused_rather_than_silently_admitted() -> None:
    """G4: a coroutine is
    truthy, so ``not verdict.ok`` on one would admit everything the gate was supposed to judge, a gate that appears
    to be wired and refuses nothing."""

    async def slow(candidate: Pick) -> Threshold:
        return Threshold(candidate.value, 5)

    with pytest.raises(GateFaultError, match=r"fault in the wiring"):
        Picker(slow).admits(Pick(value=0, evidence="x"))


async def test_an_async_gate_works_on_the_async_path() -> None:
    """G4: the other half, ``judge`` is where an async gate is
    honored."""

    async def slow(candidate: Pick) -> Threshold:
        await asyncio.sleep(0)
        return Threshold(candidate.value, 5)

    picker = Picker(slow)
    assert (await picker.judge(Pick(value=9, evidence="x"))).ok is True
    assert (await picker.judge(Pick(value=1, evidence="x"))).ok is False
    assert rejected_values(picker) == [1]


async def test_a_fault_in_an_async_gate_reads_the_same_way() -> None:
    """G3: one wording for "the gate broke", whichever path
    found it."""

    async def broken(candidate: Pick) -> Threshold:
        raise KeyError("floor")

    picker = Picker(broken)
    with pytest.raises(GateFaultError, match=r"fault in the gate rather than a verdict"):
        await picker.judge(Pick(value=7, evidence="x"))
    assert picker.rejected == []


# ── The collision guard: a convention turned into a library guarantee ──


def test_a_colliding_propose_parameter_is_refused_at_wiring_time() -> None:
    """G5: the runtime would pass the result
    positionally and ``response`` by keyword, raise ``TypeError: got multiple values``, and report *that* to the
    model as a validation failure, so the gate appears to reject everything and the message names nothing a model
    can fix. Refused at wiring time instead, naming both sides, as a ``WiringError`` not a ``RuntimeError``."""
    picker = CollidingPicker(floor_gate(5))

    with pytest.raises(WiringError) as raised:
        picker.gated()

    message = str(raised.value)
    assert "'response'" in message
    assert "'propose'" in message


async def test_the_collision_guard_also_covers_the_beam_path() -> None:
    """G5: both wiring entry points, or the guarantee is
    one call away from not applying."""
    async with RuntimeHarness() as harness:
        picker = CollidingPicker(floor_gate(5))
        with pytest.raises(WiringError, match=r"'response'"):
            await picker.propose_k(2, harness.coordinator, response="go")
        assert await live_threads(harness) == []


async def test_a_later_validator_parameter_is_injected_not_forbidden() -> None:
    """G5: the guard checks the first parameter
    only, and that narrowness is deliberate.

    A validator naming a propose parameter after the result is handed that bound argument by keyword; the runtime
    documents it and it is useful. Forbidding every name would outlaw the one pattern ``postcondition.py``
    describes, so this asserts the injection really happens rather than merely that the guard stayed quiet.
    """
    async with RuntimeHarness():
        picker = InjectingPicker(floor_gate(5))
        result = await picker.gated(model=picks([9]))(hint="the injected hint")

        assert result.value == 9
        assert picker.hints == ["the injected hint"]


# ── candidate_of: the base cannot know which field is the candidate ──


def test_the_default_candidate_is_the_whole_response() -> None:
    """The default candidate is the whole response."""
    picker = Picker(floor_gate(5))
    response = Pick(value=7, evidence="x")
    assert picker.candidate_of(response) is response


def test_an_override_hands_the_gate_the_field_instead() -> None:
    """An override hands the gate the field instead: ``HarnessProposer``'s shape, the proposal carries its
    own evidence, and the evidence is the auditable artifact rather than the subject of the gate."""
    picker = ValuePicker(value_gate(5))
    assert picker.candidate_of(Pick(value=7, evidence="x")) == 7

    assert picker.admits(Pick(value=7, evidence="x")) is None
    with pytest.raises(GateRejectionError, match=r"value 1 is below the floor"):
        picker.admits(Pick(value=1, evidence="x"))
    assert rejected_values(picker) == [1]


async def test_the_override_reaches_the_beam_path_too() -> None:
    """The override reaches the beam path too: both paths extract through the same hook, so a subclass
    overrides it once."""
    async with RuntimeHarness() as harness:
        picker = ValuePicker(value_gate(5))
        picker.compiled = _scripting(picker, picks([9, 9]))  # type: ignore[method-assign]

        admitted = await picker.propose_k(2, harness.coordinator, hint="go")

        assert [candidate for candidate, _ in admitted] == [9, 9]
        assert await live_threads(harness) == []


# ── The beam path: k one-shot branches, then filter ──


async def test_propose_k_returns_every_admitted_branch_and_records_the_rest() -> None:
    """G6: k branches, one proposal each,
    filtered by the gate, and all threads retired.

    The three scripted values are one rejection and two admissions, so a working beam returns two pairs and leaves
    exactly one verdict in ``rejected``. Retirement is asserted from the coordinator's registry, which is the
    runtime's answer rather than this code's.
    """
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        model = picks([1, 7, 9])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        admitted = await picker.propose_k(3, harness.coordinator, hint="go")

        assert [candidate.value for candidate, _ in admitted] == [7, 9]
        assert all(verdict.ok for _, verdict in admitted)
        assert rejected_values(picker) == [1]
        assert len(model.contexts) == 3, "one shot per branch, no post-condition retries"
        assert await live_threads(harness) == []


async def test_propose_k_attaches_no_post_condition_so_k_stays_a_count_of_branches() -> None:
    """G6: the contrast with the
    single-thread path, asserted rather than documented.

    Every branch proposes a rejected value. A post-condition on the branches would retry each one up to
    ``max_attempts`` and the model would be called far more than k times; one shot per branch means exactly k calls
    and an empty result.
    """
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        model = picks([1, 2, 3])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        admitted = await picker.propose_k(3, harness.coordinator, hint="go")

        assert admitted == []
        assert len(model.contexts) == 3
        assert rejected_values(picker) == [1, 2, 3]
        assert await live_threads(harness) == []


async def test_every_branch_inherits_the_seed_cycle_and_only_its_own_turn() -> None:
    """G6: why the seed is a cycle and not a
    ``notify``.

    A pending ``notify`` is worker-side inject state and ``fork`` copies the log, so a branch never sees it;
    measured, and the reason this signature takes cycles. A seed *cycle* run before the fork reaches every branch's
    first model call, and each branch's own turn reaches only that branch. Four calls: one seed, three branches.
    """
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        model = picks([9, 9, 9, 9])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        admitted = await picker.propose_k(3, harness.coordinator, hint="BRANCH", seed=[{"hint": "SEEDEVIDENCE"}])

        assert len(admitted) == 3
        assert len(model.contexts) == 4

        assert not any("BRANCH" in prompt for prompt in model.prompts(0))
        for call in (1, 2, 3):
            prompts = model.prompts(call)
            assert any("SEEDEVIDENCE" in prompt for prompt in prompts), call
            assert any("BRANCH" in prompt for prompt in prompts), call

        assert await live_threads(harness) == []


async def test_propose_k_of_one_runs_the_original_thread_and_forks_nothing() -> None:
    """``propose_k`` of one runs the original thread and forks nothing: branch 0 is the spawned thread, so k=1
    is a single one-shot proposal."""
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        model = picks([9])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        admitted = await picker.propose_k(1, harness.coordinator, hint="go")

        assert [candidate.value for candidate, _ in admitted] == [9]
        assert len(model.contexts) == 1
        assert await live_threads(harness) == []


async def test_propose_k_branches_get_no_coordinator_tools_so_they_cannot_wait_on_each_other() -> None:
    """G6: the branches are independent, so no beam call is handed ``list_threads`` or ``send_message``.

    Measured on a live model with the coordinator tools on: branch 0 listed its siblings and messaged both, and
    every branch then waited on a reply from a branch that was waiting on it. A thread spawned outside the beam
    still gets the tools, which shows the absence below is the beam's doing and not the scripted model's.
    """
    coordinator_tools = {"list_threads", "send_message"}
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        model = picks([9, 9, 9, 9, 9])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        alone = await picker.spawn("propose", harness.coordinator)
        await alone.run(hint="alone")
        await alone.retire()
        assert coordinator_tools <= set(model.tool_specs[0])

        admitted = await picker.propose_k(3, harness.coordinator, hint="go", seed=[{"hint": "seed"}])

        assert len(admitted) == 3
        beam_calls = model.tool_specs[1:]
        assert len(beam_calls) == 4, "one seed cycle and three branches"
        for names in beam_calls:
            assert not coordinator_tools & set(names), names
        assert await live_threads(harness) == []


async def test_propose_k_refuses_a_beam_narrower_than_one_thread() -> None:
    """G5: a beam narrower than one thread is a ``WiringError``, refused before any thread exists."""
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        with pytest.raises(WiringError, match=r"at least one branch"):
            await picker.propose_k(0, harness.coordinator, hint="go")
        assert await live_threads(harness) == []


async def test_propose_k_retires_every_thread_when_the_gate_faults_mid_beam() -> None:
    """G3, G6: the ``finally``, and the
    reason it is one.

    The gate breaks on the second branch, after three threads exist. The documented gate-fault error propagates (a
    bug does not become a verdict just because it happened inside a beam) and every thread is still gone, asserted
    through the coordinator. A leak here would be invisible in the return value and would only surface later as
    threads nothing owns.
    """
    calls: list[int] = []

    def breaks_on_the_second(candidate: Pick) -> Threshold:
        calls.append(candidate.value)
        if len(calls) == 2:
            raise KeyError("floor")
        return Threshold(candidate.value, 5)

    async with RuntimeHarness() as harness:
        picker = Picker(breaks_on_the_second)
        picker.compiled = _scripting(picker, picks([9, 9, 9]))  # type: ignore[method-assign]

        with pytest.raises(GateFaultError, match=r"fault in the gate rather than a verdict"):
            await picker.propose_k(3, harness.coordinator, hint="go")

        assert len(calls) == 2, "the beam stopped at the fault rather than continuing"
        assert picker.rejected == []
        assert await live_threads(harness) == []


async def test_propose_k_retires_every_thread_when_a_branch_cycle_fails() -> None:
    """G6: the other unwind, the failure is in
    the runtime, not the gate.

    The model has fewer turns than there are branches, so the third branch's cycle raises. Same requirement, nothing
    left running, and it exercises the ``finally`` from a path that never reaches the gate at all.
    """
    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        picker.compiled = _scripting(picker, picks([9, 9]))  # type: ignore[method-assign]

        with pytest.raises(Exception):  # noqa: B017 — the runtime's own error, whatever it is
            await picker.propose_k(3, harness.coordinator, hint="go")

        assert await live_threads(harness) == []


async def test_the_beam_shares_a_coordinator_with_its_caller() -> None:
    """G6: ``AIFunction.spawn()`` with no coordinator
    builds a private one per call, which would put the beam outside the caller's registry, invisible to peers and
    to the retirement assertions every test above depends on. So the threads must be observable on the harness's
    coordinator *while the beam runs*, not only absent afterwards."""
    seen: list[int] = []

    async def counting(candidate: Pick) -> Threshold:
        seen.append(len(await harness.coordinator.list_threads()))
        return Threshold(candidate.value, 5)

    async with RuntimeHarness() as harness:
        picker = Picker(counting)
        picker.compiled = _scripting(picker, picks([9, 9]))  # type: ignore[method-assign]

        await picker.propose_k(2, harness.coordinator, hint="go")

        assert seen == [2, 2], "the beam's threads are not on the caller's coordinator"
        assert await live_threads(harness) == []


async def test_a_gate_fault_ends_the_gated_call_instead_of_burning_its_retries() -> None:
    """G3: a broken gate reaches the caller after the first proposal; the model never sees it as feedback."""

    def broken(candidate: Pick) -> Threshold:
        raise KeyError("floor")

    async with RuntimeHarness():
        picker = Picker(broken)
        model = picks([9, 9, 9, 9])

        with pytest.raises(GateFaultError, match=r"fault in the gate rather than a verdict"):
            await picker.gated(model=model)(hint="go")

        assert len(model.contexts) == 1
        assert picker.rejected == []


def test_gated_refuses_an_async_gate_before_any_model_call() -> None:
    """G4: ``gated()`` runs the gate as a synchronous post-condition, so an async gate is a wiring error."""

    async def slow(candidate: Pick) -> Threshold:
        return Threshold(candidate.value, 5)

    with pytest.raises(WiringError, match=r"the gate is async"):
        Picker(slow).gated()


async def test_gated_keeps_the_methods_own_post_conditions() -> None:
    """G1: the gate first, then the post-conditions declared on the method, then the caller's."""

    def extra(response: Pick) -> None:
        del response

    picker = EvenPicker(floor_gate(5))
    assert picker.gated(post_conditions=[extra]).config.post_conditions == (picker.admits, must_be_even, extra)

    async with RuntimeHarness():
        model = picks([7, 8])
        result = await picker.gated(model=model)(hint="go")

        assert result.value == 8
        assert len(model.contexts) == 2
        assert any("7 is odd; propose an even value" in prompt for prompt in model.prompts(1))
        assert picker.rejected == []


async def test_the_beam_skips_the_methods_own_post_conditions() -> None:
    """G6: one shot per branch even when the method declares post-conditions that would retry."""
    async with RuntimeHarness() as harness:
        picker = EvenPicker(floor_gate(5))
        model = picks([7, 9, 11])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        admitted = await picker.propose_k(3, harness.coordinator, hint="go")

        assert [candidate.value for candidate, _ in admitted] == [7, 9, 11]
        assert len(model.contexts) == 3
        assert await live_threads(harness) == []


async def test_propose_k_refuses_a_method_parameter_named_like_its_own_keywords() -> None:
    async with RuntimeHarness() as harness:
        with pytest.raises(WiringError, match=r"propose_k's own keywords"):
            await SeedPicker(floor_gate(5)).propose_k(2, harness.coordinator, seed="abc")
        assert await live_threads(harness) == []


async def test_a_method_parameter_named_k_reaches_the_method() -> None:
    """``k`` and ``coordinator`` are positional-only, so a propose parameter of either name passes through."""
    async with RuntimeHarness() as harness:
        picker = KPicker(floor_gate(5))
        model = picks([9])
        picker.compiled = _scripting(picker, model)  # type: ignore[method-assign]

        admitted = await picker.propose_k(1, harness.coordinator, k=7)

        assert [candidate.value for candidate, _ in admitted] == [9]
        assert any("Propose one of 7 values." in prompt for prompt in model.prompts(0))


async def test_propose_k_retires_the_other_branches_when_one_retirement_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """G6: one failed teardown neither stops the unwind nor goes unreported."""
    from ai_functions.handle import ThreadHandle

    original = ThreadHandle.terminate_now
    failed: list[str] = []

    async def flaky(self: ThreadHandle[..., Any]) -> None:
        if not failed:
            failed.append(self.id)
            raise ConnectionError("worker unreachable")
        await original(self)

    async with RuntimeHarness() as harness:
        picker = Picker(floor_gate(5))
        picker.compiled = _scripting(picker, picks([9, 9, 9]))  # type: ignore[method-assign]
        monkeypatch.setattr(ThreadHandle, "terminate_now", flaky)

        with pytest.raises(ConnectionError, match="worker unreachable"):
            await picker.propose_k(3, harness.coordinator, hint="go")

        assert await live_threads(harness) == failed


# ── The error family and the package boundary (added by this port) ──


def test_the_gated_errors_are_method_errors_naming_the_owner() -> None:
    """D1: the three leaves subclass ``MethodError`` and carry the owner label as ``function_name``, so a caller can
    catch the family and read which proposer raised."""
    assert issubclass(GateRejectionError, MethodError)
    assert issubclass(GateFaultError, MethodError)
    assert issubclass(WiringError, MethodError)

    picker = Picker(floor_gate(5))
    with pytest.raises(GateRejectionError) as rejected:
        picker.admits(Pick(value=1, evidence="x"))
    assert rejected.value.function_name == "picker"

    with pytest.raises(GateFaultError) as faulted:
        Picker(lambda candidate: (_ for _ in ()).throw(KeyError("x"))).admits(Pick(value=1, evidence="x"))  # type: ignore[arg-type, return-value]
    assert faulted.value.function_name == "picker"

    with pytest.raises(WiringError) as wired:
        CollidingPicker(floor_gate(5)).gated()
    assert wired.value.function_name == "colliding"


def test_gated_imports_nothing_from_other_experimental_packages() -> None:
    """Static boundary for gated.py alone (the package-wide guard lives in test_methods_boundary.py): parsed with
    ``ast`` so an import inside a ``TYPE_CHECKING`` block or a function body counts too. Only stdlib, third-party,
    the ai_functions core and its own package."""
    names = _imports(Path(methods_gated.__file__))
    assert names, "gated.py imports nothing? the parser found no imports"
    offenders = [
        name
        for name in names
        if name.startswith("ai_functions.experimental") and not (name == OWN or name.startswith(OWN + "."))
    ]
    assert offenders == [], offenders
    assert "ai_functions.experimental.methods.core" in names, "gated.py must build on methods.core"
