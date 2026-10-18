"""An agent that proposes what it is then judged by: the gate as a post-condition.

A proposer judged by its own output is a shape worth keeping and worth separating from the
domain it grows in. A ``HarnessProposer`` proposes the objective weight that its own score is
then computed with, and the detectors that decide whether a weight is pathological are wired
as that method's *post-conditions* rather than as a check the caller runs afterwards. That
placement is the whole idea, and this module is that placement alone, with no domain
dependencies.

**A gate is not a convention.** A manual check after the call is a check the loop can forget,
and the loops that forget it are the ones under time pressure. A post-condition cannot be
skipped: ``ai_thread`` runs every validator against the result before the cycle returns, and
any exception a validator raises becomes the message text of a ``[VALIDATION ERROR]`` user
turn that the *next* attempt reads (``ai_thread/ai_thread.py:643-658``). So refusal is the
default and the gate's own report is the re-ask feedback: the model that has to fix the
proposal is handed the reason, in the reason's own words, without the caller writing any glue.

**A rejection and a bug are not the same event, and the runtime cannot tell them apart.**
Every exception out of a validator is reported to the model as a validation failure
(``ai_thread.py:1022-1023``), so a ``KeyError`` in the gate is indistinguishable from a
considered refusal and burns every retry on something the model cannot fix. ``admits``
therefore wraps the gate call and re-raises an internal failure as a message that says it is
internal, and it does *not* record that non-verdict in ``rejected``.

**Two paths, deliberately different.** The single-thread path retries until admitted, because
that is what a post-condition is for. The beam path (``propose_k``) forks k branches, takes one
proposal from each, and filters with the gate directly: no post-condition on the branches, or
each branch's internal retries would blur what k measures.

Errors follow the methods package's ``MethodError`` family rather than bare built-ins:
a refusal is a :class:`GateRejectionError`, a broken gate is a :class:`GateFaultError`, and a
wiring mistake caught before any model call is a :class:`WiringError`. All three reach the
model identically when raised inside a validator, because the runtime reports ``str(exc)``
whatever the type.

Invariants:
    G1 — the gate is a post-condition, never a convention. ``gated()`` prepends
    ``admits`` to whatever ``post_conditions`` the caller passes, so the gate
    runs on every proposal *after* the model has produced it and cannot be
    dropped by a hand-written override.

    G2 — refusal is the default and the report is the feedback. A rejected
    candidate raises ``GateRejectionError`` whose text is ``report_text()``
    followed by ``REASK``, and the verdict is appended to ``rejected`` in
    order. An admitted candidate passes quietly and records nothing.

    G3 — a fault is not a verdict. An exception in the gate, in
    ``candidate_of``, or in reading the verdict it returned is re-raised as
    ``GateFaultError`` naming the part that broke, and is never recorded in
    ``rejected``. One wording, whichever path found it.

    G4 — an async gate is honored only on the async path. ``admits`` refuses
    an awaitable verdict with ``GateFaultError`` (closing the coroutine) rather
    than truth-testing it, because every coroutine is truthy; ``judge`` awaits
    it.

    G5 — collisions are refused at wiring time. A post-condition whose first
    positional parameter shares a name with a propose parameter raises
    ``WiringError`` from ``gated()`` and from ``propose_k``, before any model
    call; later parameters are left alone so the runtime's keyword injection
    still works.

    G6 — the beam is k one-shot branches. ``propose_k`` attaches no
    post-condition, judges each branch's single proposal directly, returns the
    admitted ``(candidate, verdict)`` pairs in branch order with no ordering
    imposed, and retires every thread in a ``finally`` on the caller's
    coordinator.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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

class GateFaultError(MethodError):
    """The gate, the candidate extractor or the verdict itself broke (G3, G4).

    Not a verdict about the proposal: the message says so, names the part that failed and
    carries the original exception's type and text. Never recorded in ``rejected``.
    """

class WiringError(MethodError):
    """A gate or beam was wired in a way that cannot work, refused before any model call (G5).

    A post-condition's result parameter shares a name with a propose parameter, or a beam
    was asked for fewer than one branch.
    """

@runtime_checkable
class Verdict(Protocol):
    """Synchronous methods and properties; calls finish before returning.

    What a gate returns: a decision, and a report a model can act on.

    Two members because a gate needs to serve two readers. ``ok`` is for the loop, which
    branches on it. ``report_text()`` is for the model, which has to *change* something, and a
    bare ``False`` tells it nothing to change: the report is the feedback channel that makes
    the re-ask worth spending an attempt on.

    Deliberately not a base class: the verdicts this generalises over already exist and
    already differ, and a protocol lets each satisfy it without being edited.
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

    A callable rather than a method to override: the gates worth having are made of things a
    library must not import, so the gate is taken as a value. A subclass may still assign
    ``self.gate`` in its own ``__init__``; that is injection too, spelled from the inside.
    """

    def __call__(self, candidate: Any) -> Verdict | Awaitable[Verdict]:
        """Judge ``candidate`` and return its verdict, or an awaitable of one."""
        ...

class GatedProposer(MethodAgent):
    """An agent whose proposal is judged by the gate it is graded against.

    The subclass supplies the propose ``@ai_method`` and the gate; this base owns the
    skeleton: the post-condition, the ``rejected`` ledger, the wiring guard, and the beam
    search. The gate lives on ``self`` (a validator needs it on every attempt); the candidate
    is a call argument (or the optimizer cannot see it). No score, no optimizer, no memory:
    a gate answers *admissible or not*, and "best" is the caller's to define.

    Args:
        gate: The callable that judges one candidate; see :class:`Gate`.
    """

    REASK: str
    """Appended after ``report_text()`` on a rejection, so the model is told what to *do*.

    Domain wording, so a class attribute subclasses override; the mechanism does not change.
    """

    gate: Gate
    """The injected judge; ``admits`` and ``judge`` both call it."""

    rejected: list[Verdict]
    """Every candidate the gate turned away, in order. A gate fault is never recorded here."""

    def __init__(self, gate: Gate) -> None: ...
    def candidate_of(self, response: Any) -> Any:
        """Pull the thing to be judged out of the model's response. Sync, and dumb.

        The default judges the whole response; a subclass whose proposal carries its own
        evidence overrides it to return the field the gate judges. Runs inside a validator on
        every attempt, so it must stay free of side effects.

        Args:
            response: The typed result one propose cycle produced.

        Returns:
            The candidate the gate judges; the response itself by default.
        """
        ...

    def admits(self, response: Any) -> None:
        """Post-condition: the gate must admit the proposed candidate (G1, G2, G3, G4).

        Raising is how a post-condition fails; the runtime reports the text to the model as
        the next attempt's ``[VALIDATION ERROR]`` turn. The gate call is wrapped so an
        internal failure is re-raised as a message that says it is internal rather than
        masquerading as a verdict. The first parameter is named ``response`` on purpose and
        ``gated()`` enforces that it collides with no propose parameter.

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
        Returns a rejection rather than raising it, because the beam path filters where the
        single-thread path re-asks.

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

        Any ``post_conditions`` passed in are kept and ``admits`` is prepended, so a subclass
        can add checks without losing the gate.

        Args:
            method: The ``@ai_method`` to compile; ``propose`` by default.
            overrides: ``ThreadConfig`` overrides, exactly as ``compiled`` takes them;
                ``post_conditions`` is merged rather than replaced.

        Returns:
            The compiled ``AIFunction``; awaiting it returns an admitted result or exhausts
            its retries.

        Raises:
            WiringError: ``admits`` or an extra post-condition has a first parameter named
                like a parameter of ``method`` (G5).
        """
        ...

    async def propose_k(
        self,
        k: int,
        coordinator: Coordinator,
        *run_args: Any,
        method: str = "propose",
        seed: Sequence[Mapping[str, Any]] = (),
        **run_kwargs: Any,
    ) -> list[tuple[Any, Verdict]]:
        """Take k independent proposals from one shared context and return the admitted ones (G6).

        One thread is spawned for ``method`` and run once per entry in ``seed``; then it is
        forked ``k - 1`` times, each branch runs exactly one propose cycle with the same
        arguments, and each result is judged by the gate directly, with no post-condition, so
        k is what it says it is. Branches run serially so a caching model reuses the shared
        prefix.

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
            WiringError: ``k`` is not at least 1, or ``admits`` collides with ``method`` (G5).
            GateFaultError: The gate or the candidate extractor failed. Every thread is still
                retired: the unwind is in a ``finally``.
        """
        ...
