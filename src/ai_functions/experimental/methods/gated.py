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

import inspect
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from .core import MethodAgent, MethodError, MethodThread, _owner_name

if TYPE_CHECKING:
    from ai_functions.ai_thread import AIFunction
    from ai_functions.protocols import Coordinator


# ── Errors ──


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


# ── The protocols ──


@runtime_checkable
class Verdict(Protocol):
    """Synchronous methods and properties; calls finish before returning.

    What a gate returns: a decision, and a report a model can act on.

    Two members because a gate needs to serve two readers. ``ok`` is for the loop, which
    branches on it. ``report_text()`` is for the model, which has to *change* something, and a
    bare ``False`` tells it nothing to change: the report is the feedback channel that makes
    the re-ask worth spending an attempt on.

    Deliberately not a base class. The verdicts this generalises over already exist and
    already differ: an ``Admission`` dataclass derives ``ok`` from a conjunction over two
    independent detector readings and also carries ``quality``, ``threshold`` and
    ``baseline_rules``; a detection ``Probe`` derives ``ok`` from its refusals. A protocol lets
    both satisfy this without either being edited or losing what it knows.
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

    A callable rather than a method to override, which is the central choice in this module.
    The gates worth having are made of the things a library half must not import (an ``admit``
    over a real harness needs polars, a process miner and a reachability sweep), so a
    ``GatedProposer`` that reached its gate through ``self.gate(...)`` as an abstract method
    would be a library class whose only real implementation lives in the application. Taking
    the gate as a value keeps the boundary clean and makes the protocol testable with three
    lines of a trivial gate, which is why the tests here need no fixture at all.

    A subclass may still assign ``self.gate`` in its own ``__init__`` (``HarnessProposer``
    binds a closure over its event log), and that is injection too, just spelled from the
    inside.
    """

    def __call__(self, candidate: Any) -> Verdict | Awaitable[Verdict]:
        """Judge ``candidate`` and return its verdict, or an awaitable of one."""
        ...


# ── The proposer ──


class GatedProposer(MethodAgent):
    """An agent whose proposal is judged by the gate it is graded against.

    The subclass supplies the propose ``@ai_method`` and the gate; this base owns the
    skeleton: the post-condition, the ``rejected`` ledger, the wiring guard, and the beam
    search.

    **Where state goes is not a style question here.** The gate arrives on ``self`` and the
    candidate arrives as a call argument, and each has to be where it is. A validator needs
    the gate on every attempt, including attempts it has not seen yet, so the gate cannot be a
    parameter the model fills in. The learnable value must be a call argument or it is
    invisible to the optimizer: ``collect_nodes`` walks ``(args, kwargs)``, so anything on
    ``self`` can never be a gradient target (see the methods package docstring). That is the
    same split ``HarnessProposer`` documents, stated once here so subclasses inherit it rather
    than rediscover it.

    **What this class deliberately does not do.** No score, no optimizer, no memory. A gate
    answers *admissible or not*; turning admitted candidates into a scalar an optimizer climbs
    is a separate concern with its own trap (``Admission.quality`` exists because the
    objective's own peak is maximized at the pathological end), and it stays in the
    application. ``propose_k`` accordingly returns every admitted pair and imposes no
    ordering: "best" is the caller's to define, and quality shapes vary.

    Args:
        gate: The callable that judges one candidate; see :class:`Gate`.
    """

    REASK = "Propose a different candidate."
    """Appended after ``report_text()`` on a rejection, so the model is told what to *do*.

    A class attribute rather than a constant because the instruction is domain wording: the
    harness proposer says which direction on its weight axis causes the pathology it just
    refused. Subclasses override it; the mechanism does not change.
    """

    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self.rejected: list[Verdict] = []
        """Every candidate the gate turned away, in order. The evidence the gate fired.

        Kept because a loop that silently re-asked and then succeeded looks, from the outside,
        exactly like a loop whose gate never fired. A gate fault is *not* recorded here (see
        ``admits``), so this list stays readable as refusals and nothing else.
        """

    # ── The candidate ──

    def candidate_of(self, response: Any) -> Any:
        """Pull the thing to be judged out of the model's response. Sync, and dumb.

        The base cannot know which field carries the candidate: a proposal usually arrives
        alongside the reasoning that produced it (``HarnessProposal`` is ``coverage_weight``
        *and* ``evidence``, and the evidence is the auditable artifact, not the subject of the
        gate). The default judges the whole response, which is right when the response *is*
        the candidate; ``HarnessProposer`` overrides it to ``response.coverage_weight``.

        Kept synchronous and free of side effects on purpose. It runs inside a validator on
        every attempt, and an override that fetched or computed anything would put work the
        gate is supposed to own behind a hook named for an accessor.

        Args:
            response: The typed result one propose cycle produced.

        Returns:
            The candidate the gate judges; the response itself by default.
        """
        return response

    # ── The post-condition ──

    def admits(self, response: Any) -> None:
        """Post-condition: the gate must admit the proposed candidate (G1, G2, G3, G4).

        Raising is how a post-condition fails; ``ai_thread`` catches any exception from a
        validator and reports its text to the model as a validation failure. That is also the
        trap this method is written around: an unexpected exception is indistinguishable from
        a rejection and would burn every retry on a bug. So the gate call is wrapped and an
        internal failure is re-raised as a message that says it is internal, rather than being
        allowed to masquerade as a verdict about the proposal.

        The parameter is named ``response`` rather than ``proposal`` on purpose, and the name
        is not merely a convention here: ``_check_no_collision`` enforces it at wiring time. A
        post-condition whose *first* parameter shares a name with a propose parameter is
        handed the result positionally and the bound argument by keyword, which raises
        ``TypeError: got multiple values for argument`` and is then swallowed as a validation
        failure: a silent bug wearing a verdict's clothes.

        Synchronous, so it can also be called directly in a test, which is how the sharp
        properties are asserted, since what needs proving is that it raises with a usable
        message. An async gate is only reachable through the async path; a coroutine returned
        here is refused loudly rather than being truth-tested as an object, because every
        coroutine is truthy and ``not verdict.ok`` on one would silently admit everything.

        Args:
            response: The typed result the runtime is validating.

        Raises:
            GateRejectionError: The gate refused the candidate; the message is the report and
                the re-ask, and the verdict is now the last entry of ``rejected``.
            GateFaultError: ``candidate_of``, the gate, or the verdict it returned raised, or
                the gate returned an awaitable. Not recorded in ``rejected``.
        """
        try:
            candidate = self.candidate_of(response)
        except Exception as error:  # noqa: BLE001 — an extractor bug must not read as a verdict either
            raise self._fault(response, error, part="candidate extractor") from error
        try:
            verdict = self.gate(candidate)
        except Exception as error:  # noqa: BLE001 — see the docstring: a bug must not read as a verdict
            raise self._fault(candidate, error) from error
        if inspect.isawaitable(verdict):
            # Close it before raising. An abandoned coroutine is a ``RuntimeWarning`` at
            # collection time and a warning is not where this belongs; the error below is.
            close = getattr(verdict, "close", None)
            if callable(close):
                close()
            raise GateFaultError(
                f"{self._label()}: the gate returned an awaitable for candidate={candidate!r}, "
                "which is a fault in the wiring rather than a verdict about your proposal: an "
                "async gate needs the async path, because a coroutine is truthy and would be "
                "admitted without ever being judged",
                function_name=self._label(),
            )
        self._record(candidate, verdict)

    async def judge(self, candidate: Any) -> Verdict:
        """Run the gate on one candidate, awaiting it if it is async, and record a rejection.

        The direct path ``propose_k`` uses, and the only place an async gate is honored (G4).
        Raises the same ``GateFaultError`` ``admits`` does, so a bug in the gate reads the
        same way whichever path found it; unlike ``admits`` it *returns* a rejection rather
        than raising it, because the beam path filters where the single-thread path re-asks.

        Args:
            candidate: What the gate judges, already extracted by ``candidate_of``.

        Returns:
            The gate's verdict, admitted or not; a rejection is also appended to ``rejected``.

        Raises:
            GateFaultError: The gate raised, or its verdict could not be read.
        """
        try:
            returned = self.gate(candidate)
            # ``await``ing the union directly widens the result to ``object`` for a type
            # checker, because ``Awaitable``'s type argument is lost across the
            # ``isawaitable`` narrowing. Casting the awaitable rather than the result keeps
            # ``Verdict`` on the value.
            verdict = await cast("Awaitable[Verdict]", returned) if inspect.isawaitable(returned) else returned
        except Exception as error:  # noqa: BLE001 — a bug must not read as a verdict
            raise self._fault(candidate, error) from error
        try:
            ok = bool(verdict.ok)
            # Render the report now, not when the ledger is read: a rejection whose
            # ``report_text`` crashes would otherwise detonate far from the gate that
            # produced it, in whatever loop renders ``rejected`` for a summary.
            if not ok:
                verdict.report_text()
        except Exception as error:  # noqa: BLE001 — a broken verdict is a fault, not a verdict
            raise self._fault(candidate, error) from error
        if not ok:
            self.rejected.append(verdict)
        return verdict

    def _record(self, candidate: Any, verdict: Verdict) -> None:
        """Append a rejection and raise it as the re-ask, or return quietly.

        The verdict itself is untrusted here: reading ``ok`` or rendering ``report_text``
        can crash when a gate returned something malformed (a gate that swallowed a bad
        candidate and built a verdict around it, say). That crash is a fault, not a
        rejection: it must neither land in ``rejected`` nor reach the model dressed as a
        validation message it cannot act on.
        """
        try:
            ok = bool(verdict.ok)
            report = None if ok else verdict.report_text()
        except Exception as error:  # noqa: BLE001 — a broken verdict is a fault, not a verdict
            raise self._fault(candidate, error) from error
        if not ok:
            self.rejected.append(verdict)
            raise GateRejectionError(f"{report}\n\n{self.REASK}", function_name=self._label())

    def _fault(self, candidate: Any, error: Exception, *, part: str = "gate") -> GateFaultError:
        """One wording for "the gate broke", shared by both paths so both read alike."""
        return GateFaultError(
            f"{self._label()}: the {part} could not be evaluated for candidate={candidate!r}, "
            f"which is a fault in the {part} rather than a verdict about your proposal: "
            f"{type(error).__name__}: {error}",
            function_name=self._label(),
        )

    def _label(self) -> str:
        """The owner name every message here is prefixed with; the same one the tool name uses."""
        return _owner_name(self)

    # ── Wiring ──

    def gated(self, method: str = "propose", **overrides: Any) -> AIFunction[..., Any]:
        """Compile one propose method with the gate attached as a post-condition (G1).

        Any ``post_conditions`` passed in are kept and ``admits`` is prepended, so a subclass
        can add checks without losing the gate, and cannot accidentally replace it, which is
        the failure mode of a plain ``compiled(..., post_conditions=[...])`` call written by
        hand.

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
        extra = tuple(overrides.pop("post_conditions", ()))
        # Every attached condition gets the guard, not just admits: the collision trap is a
        # property of the runtime's kwarg injection, and an extra condition hits it the same
        # way the gate would.
        self._check_no_collision(method, *extra)
        return self.compiled(method, post_conditions=(self.admits, *extra), **overrides)

    def _check_no_collision(self, method: str, *conditions: Any) -> None:
        """Refuse at wiring time if a result parameter shares a propose parameter's name (G5).

        ``ai_thread`` passes the result positionally and then injects, by keyword, every bound
        argument whose name appears in the validator's signature
        (``ai_thread/ai_thread.py:1016-1019``). Those two rules are useful together (a
        validator that wants the call's ``window`` just names it) and fatal for the *first*
        parameter, which already holds the result: the same slot filled twice raises
        ``TypeError: got multiple values for argument``, which the runtime then catches and
        reports to the model as a validation failure. The gate appears to reject everything,
        the message makes no sense, and the fix is a one-word rename nothing points at.

        So this is checked here rather than left to a convention, because the convention's
        failure mode is silent and its violation is one careless rename away. Only the first
        parameter is checked, deliberately: forbidding the rest would forbid the injection the
        runtime documents and this class has no reason to prevent.
        """
        proposed = set(inspect.signature(getattr(self, method)).parameters)
        for condition in (self.admits, *conditions):
            validator = self._first_parameter(condition)
            if validator is None:
                continue
            if validator in proposed:
                label = getattr(condition, "__name__", repr(condition))
                raise WiringError(
                    f"{self._label()}: the post-condition {label!r}'s result parameter is "
                    f"named {validator!r}, which is also a parameter of {method!r}; the "
                    f"runtime would pass the result positionally and {validator!r} by "
                    f"keyword, and `TypeError: got multiple values for argument "
                    f"{validator!r}` is then reported to the model as a validation failure. "
                    f"Rename one of them.",
                    function_name=self._label(),
                )

    @staticmethod
    def _first_parameter(validator: Any) -> str | None:
        """The name of the slot the result lands in, or None if the validator takes no args."""
        positional = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        parameters = [
            name for name, parameter in inspect.signature(validator).parameters.items() if parameter.kind in positional
        ]
        return parameters[0] if parameters else None

    # ── Beam search ──

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
        forked ``k - 1`` times, so the original is branch 0 and every branch starts
        byte-identical to the seeded context and diverges from there. Each branch runs exactly
        one propose cycle with the same arguments, and each result is judged by the gate
        *directly*.

        **Why the gate is not attached as a post-condition here.** The retries a
        post-condition drives are the right behavior on the single-thread path and the wrong
        behavior on this one: a branch that quietly re-asked until it was admitted would make
        k a count of branches-that-eventually-succeeded, and a k of 3 could mean twelve model
        calls and three admitted candidates that tell you nothing about the width of the
        search. One shot per branch, then filter, so k is what it says it is and ``rejected``
        measures how much of the beam the gate turned away.

        **Why the branches get no coordinator tools.** Every branch is spawned with
        ``coordinator_tools_enabled=False``, so none is handed ``list_threads`` or
        ``send_message``. Branches are meant to be independent draws, and those tools let a
        branch find its siblings and message them. Measured on a live model, branch 0 messaged
        both forks, and each branch then waited on a reply from a branch waiting on it, so the
        beam never finished. The method's own tools still reach every branch.

        **Why the seed is a sequence of cycles rather than a ``notify``.** A pending
        ``notify`` is worker-side inject state, not log state, and ``fork`` copies the log:
        measured, a forked branch does not see it, while a seed *cycle* run before the fork
        appears in every branch's first model call. Seeding therefore means running the
        method, which is also the truthful shape: what a branch inherits is a real turn it
        could have produced. Each seed entry is one cycle's keyword arguments, because that is
        ``MethodThread.run``'s contract and a thread hosts exactly one signature.

        **Prompt-cache reuse across branches.** Every branch replays the same seeded log
        through the same model with the same arguments, so the requests are byte-identical up
        to any trailing cache point the model's ``cache_config`` has the runtime inject.
        Branches run serially below, so branch 0's response, which writes the provider cache,
        completes before branch 1's request, and from the second branch on the shared prefix
        is billed at cache-read rates rather than re-ingested. A model without a cache point
        gets correct but uncached beams.

        Args:
            k: How many branches; at least 1.
            coordinator: The caller's coordinator; every branch lands in its registry.
            run_args: Positional arguments of ``method``'s signature, the same for every branch.
            method: The ``@ai_method`` to beam over; ``propose`` by default.
            seed: Keyword-argument dicts, one cycle each, run on the root before forking.
            run_kwargs: Keyword arguments of ``method``'s signature, the same for every branch.

        Returns:
            ``(candidate, verdict)`` for every admitted branch, in branch order, and nothing
            for the rejected ones; those are in ``rejected``. No ordering is imposed and no
            scalar is computed: which admitted candidate is best depends on a quality shape
            this class has no opinion about, and inventing one here would make every subclass
            live with it.

        Raises:
            WiringError: ``k`` is not at least 1, or ``admits`` collides with ``method`` (G5).
            GateFaultError: The gate or the candidate extractor failed, with the same wording
                ``admits`` uses. The exception propagates, and every thread is still retired:
                the unwind is in a ``finally``, and ``retire`` is idempotent against the
                runtime, so unwinding threads one of which something else already tore down
                cannot crash mid-loop and leave the rest alive.
        """
        if k < 1:
            raise WiringError(f"propose_k needs at least one branch, got k={k}", function_name=self._label())
        self._check_no_collision(method)

        threads: list[MethodThread] = []
        try:
            root = await self.spawn(method, coordinator, coordinator_tools_enabled=False)
            threads.append(root)
            for cycle in seed:
                await root.run(**cycle)
            for _ in range(k - 1):
                threads.append(await root.fork())

            admitted: list[tuple[Any, Verdict]] = []
            for thread in threads:
                response = await thread.run(*run_args, **run_kwargs)
                try:
                    candidate = self.candidate_of(response)
                except Exception as error:  # noqa: BLE001 — an extractor bug must not read as a verdict
                    raise self._fault(response, error, part="candidate extractor") from error
                verdict = await self.judge(candidate)
                if verdict.ok:
                    admitted.append((candidate, verdict))
            return admitted
        finally:
            for thread in threads:
                await thread.retire()
