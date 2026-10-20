"""``GatedProposer``: an agent whose proposals must pass a gate, run as a post-condition.

A ``GatedProposer`` takes a :class:`Gate`, a callable that judges one candidate
and returns a :class:`Verdict`. ``gated()`` compiles the propose method with the
gate's check, ``admits``, as its first post-condition, so every proposal is
judged before the call returns and a rejection's report becomes the
``[VALIDATION ERROR]`` turn the next attempt reads. A failure inside the gate is
not a verdict: ``admits`` re-raises it as a :class:`GateFaultError`, a
``PostConditionError``, so the call ends with that error instead of retrying,
and nothing is recorded in ``rejected``.

``propose_k`` forks k branches from one seeded thread, takes one proposal from
each, and filters them with the gate directly, without post-conditions, so k
is the number of proposals drawn.

    class Picker(GatedProposer):
        def __init__(self, floor: int) -> None:
            super().__init__(lambda pick: Threshold(pick.value >= floor, floor))

        @ai_method(Pick, description="Propose a value, judged by the gate")
        def propose(self, hint: str) -> Pick:
            '''Propose a value given {hint}.'''

``Picker(5).gated()`` is an ``AIFunction`` whose post-condition is the gate, so awaiting it
either returns an admitted ``Pick`` or exhausts its retries trying; ``await
picker.propose_k(3, coord, hint="go")`` returns the admitted ``(candidate, verdict)`` pairs out
of three one-shot branches.

A refusal is a :class:`GateRejectionError`, a broken gate a :class:`GateFaultError`,
and a wiring mistake caught before any model call a :class:`WiringError`, all in
the methods package's ``MethodError`` family.

Invariants:
    G1 — ``gated()`` puts ``admits`` before the method's own post-conditions
    and any the caller passes, so the gate runs on every proposal after the
    model has produced it.

    G2 — a rejected candidate raises ``GateRejectionError`` whose text is
    ``report_text()`` followed by ``REASK``, and the verdict is appended to
    ``rejected`` in order. An admitted candidate passes quietly and records
    nothing.

    G3 — a fault is not a verdict. An exception in the gate, in
    ``candidate_of``, or in reading the verdict it returned is re-raised as
    ``GateFaultError`` naming the part that broke, and is never recorded in
    ``rejected``. One wording, whichever path found it. Raised from ``admits``,
    it ends the call and reaches the caller.

    G4 — an async gate is honored only on the async path. ``gated()`` refuses
    an async gate with ``WiringError`` before any model call, and ``admits``
    refuses an awaitable verdict with ``GateFaultError`` (closing the coroutine)
    rather than truth-testing it, because every coroutine is truthy; ``judge``
    awaits it.

    G5 — collisions are refused at wiring time. A post-condition whose first
    positional parameter shares a name with a propose parameter raises
    ``WiringError`` from ``gated()`` and from ``propose_k``, before any model
    call; later parameters are left alone so the runtime's keyword injection
    still works.

    G6 — the beam is k one-shot branches. ``propose_k`` spawns them with no
    post-conditions, judges each branch's single proposal directly, returns the
    admitted ``(candidate, verdict)`` pairs in branch order with no ordering
    imposed, and retires every thread on the caller's coordinator, even when
    one retirement fails.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ...ai_thread.errors import PostConditionError
from .core import MethodAgent, MethodError

if TYPE_CHECKING:
    from ...ai_thread import AIFunction
    from ...protocols import Coordinator

class GateRejectionError(MethodError):
    """The gate turned the candidate away (G2).

    The message is the verdict's ``report_text()`` followed by the proposer's ``REASK``, so
    when the runtime reports it to the model the model is told the reason and the action.
    Raised by ``admits`` only; ``judge`` returns the rejection instead.
    """

class GateFaultError(MethodError, PostConditionError):
    """The gate, the candidate extractor or the verdict itself broke (G3, G4).

    Not a verdict about the proposal: the message says so, names the part that failed and
    carries the original exception's type and text. Never recorded in ``rejected``. As a
    ``PostConditionError``, raised from ``admits`` it ends the call instead of becoming retry
    feedback.
    """

class WiringError(MethodError):
    """A gate or beam was wired in a way that cannot work, refused before any model call (G5).

    A post-condition's result parameter shares a name with a propose parameter, ``gated()``
    was given an async gate, a propose parameter is named like a ``propose_k`` keyword, or a
    beam was asked for fewer than one branch.
    """

@runtime_checkable
class Verdict(Protocol):
    """What a gate returns: a decision, and a report a model can act on.

    ``ok`` is the decision the caller branches on; ``report_text()`` is the reason, which
    reaches the model as the re-ask feedback. Any object with these two members is a verdict.
    """

    @property
    def ok(self) -> bool:
        """Whether the candidate may be used."""
        ...

    def report_text(self) -> str:
        """The verdict as prose the model can act on. Reaches the model verbatim."""
        ...

@runtime_checkable
class Gate(Protocol):
    """Judges one candidate. Sync or async; ``admits`` refuses a coroutine, ``judge`` awaits it.

    ``GatedProposer`` takes the gate as a constructor argument; a subclass may also assign
    ``self.gate`` in its own ``__init__``.
    """

    def __call__(self, candidate: Any) -> Verdict | Awaitable[Verdict]:
        """Judge ``candidate`` and return its verdict, or an awaitable of one."""
        ...

class GatedProposer(MethodAgent):
    """An agent whose proposal is judged by the gate it is graded against.

    The subclass supplies the propose ``@ai_method`` and the gate; this base provides the
    post-condition, the ``rejected`` ledger, the wiring guard and the beam search. The gate
    lives on ``self``, where the validator reaches it on every attempt; a value the optimizer
    should update must be a call argument of the propose method.

    Args:
        gate: The callable that judges one candidate; see :class:`Gate`.
    """

    REASK: str
    """Appended after ``report_text()`` on a rejection, so the model is told what to *do*.

    Subclasses override it with domain wording.
    """

    gate: Gate
    """The injected judge; ``admits`` and ``judge`` both call it."""

    rejected: list[Verdict]
    """Every candidate the gate turned away, in order. A gate fault is never recorded here."""

    def __init__(self, gate: Gate) -> None: ...
    def candidate_of(self, response: Any) -> Any:
        """Pull the candidate the gate judges out of the model's response.

        The default judges the whole response; a subclass whose response carries the candidate
        alongside other fields returns that field. It runs inside a validator on every attempt,
        so it must be synchronous and free of side effects.

        Args:
            response: The typed result one propose cycle produced.

        Returns:
            The candidate the gate judges; the response itself by default.
        """
        ...

    def admits(self, response: Any) -> None:
        """Post-condition: the gate must admit the proposed candidate (G1, G2, G3, G4).

        Raising is how a post-condition fails: the runtime reports the exception's text to the
        model as the next attempt's ``[VALIDATION ERROR]`` turn. A failure inside
        ``candidate_of``, the gate or the verdict is re-raised as ``GateFaultError``, which
        ends the call instead. The result parameter is named ``response``, and ``gated()``
        refuses a propose method that also has a parameter of that name (G5).

        Args:
            response: The typed result the runtime is validating.

        Raises:
            GateRejectionError: The gate refused the candidate; the message is the report and
                the re-ask, and the verdict is now the last entry of ``rejected``.
            GateFaultError: ``candidate_of``, the gate, or the verdict it returned raised, or
                the gate returned an awaitable. Not recorded in ``rejected``.
        """
        ...

    async def judge(self, candidate: Any) -> Verdict:
        """Run the gate on one candidate, awaiting it if it is async, and record a rejection.

        The direct path ``propose_k`` uses, and the only place an async gate is honored (G4).
        Unlike ``admits`` it returns a rejection rather than raising it.

        Args:
            candidate: What the gate judges, already extracted by ``candidate_of``.

        Returns:
            The gate's verdict, admitted or not; a rejection is also appended to ``rejected``.

        Raises:
            GateFaultError: The gate raised, or its verdict could not be read.
        """
        ...

    def gated(self, method: str = "propose", **overrides: Any) -> AIFunction[..., Any]:
        """Compile one propose method with the gate attached as a post-condition (G1).

        ``admits`` runs first, then the post-conditions declared on the method, then any
        ``post_conditions`` passed in, so a subclass can add checks without losing the gate.

        Args:
            method: The ``@ai_method`` to compile; ``propose`` by default.
            overrides: ``ThreadConfig`` overrides, exactly as ``compiled`` takes them;
                ``post_conditions`` is appended rather than replacing the method's own.

        Returns:
            The compiled ``AIFunction``; awaiting it returns an admitted result, exhausts
            its retries, or raises ``GateFaultError`` when the gate breaks.

        Raises:
            WiringError: The gate is async, or ``admits`` or another post-condition has a
                first parameter named like a parameter of ``method`` (G4, G5).
        """
        ...

    async def propose_k(
        self,
        k: int,
        coordinator: Coordinator,
        /,
        *run_args: Any,
        method: str = "propose",
        seed: Sequence[Mapping[str, Any]] = (),
        **run_kwargs: Any,
    ) -> list[tuple[Any, Verdict]]:
        """Take k independent proposals from one shared context and return the admitted ones (G6).

        One thread is spawned for ``method`` and run once per entry in ``seed``; then it is
        forked ``k - 1`` times, each branch runs exactly one propose cycle with the same
        arguments, and each result is judged by the gate directly, with no post-conditions,
        not even those declared on the method.
        Branches are spawned without coordinator tools, so none can message a sibling, and
        run serially, so a model with a cache point reads the shared prefix from cache.

        Args:
            k: How many branches; at least 1.
            coordinator: The caller's coordinator; every branch lands in its registry.
            run_args: Positional arguments of ``method``'s signature, the same for every branch.
            method: The ``@ai_method`` to beam over; ``propose`` by default.
            seed: Keyword-argument dicts, one cycle each, run on the root before forking.
            run_kwargs: Keyword arguments of ``method``'s signature, the same for every branch.

        Returns:
            ``(candidate, verdict)`` for every admitted branch, in branch order; the rejected
            ones are in ``rejected``. No ordering is imposed and no scalar is computed.

        Raises:
            WiringError: ``k`` is not at least 1, ``method`` has a parameter named ``method``
                or ``seed``, or ``admits`` collides with ``method`` (G5).
            GateFaultError: The gate or the candidate extractor failed. The exception
                propagates after every thread is retired.
        """
        ...
