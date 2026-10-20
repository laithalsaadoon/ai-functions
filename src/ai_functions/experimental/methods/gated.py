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

import inspect
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from ai_functions.ai_thread.errors import PostConditionError

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


# ── The protocols ──


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


# ── The proposer ──


class GatedProposer(MethodAgent):
    """An agent whose proposal is judged by the gate it is graded against.

    The subclass supplies the propose ``@ai_method`` and the gate; this base provides the
    post-condition, the ``rejected`` ledger, the wiring guard and the beam search. The gate
    lives on ``self``, where the validator reaches it on every attempt; a value the optimizer
    should update must be a call argument of the propose method.

    Args:
        gate: The callable that judges one candidate; see :class:`Gate`.
    """

    REASK = "Propose a different candidate."
    """Appended after ``report_text()`` on a rejection, so the model is told what to *do*.

    Subclasses override it with domain wording.
    """

    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self.rejected: list[Verdict] = []
        """Every candidate the gate turned away, in order. A gate fault is never recorded here."""

    # ── The candidate ──

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
        return response

    # ── The post-condition ──

    def admits(self, response: Any) -> None:
        """Post-condition: the gate must admit the proposed candidate (G1, G2, G3, G4).

        Raising is how a post-condition fails: the runtime reports a rejection's text to the
        model as a validation failure. A failure inside ``candidate_of``, the gate or the
        verdict is re-raised as ``GateFaultError``, which ends the call instead. The result
        parameter is named ``response``, and ``gated()`` refuses a propose method that also
        has a parameter of that name (G5). A gate that returns an awaitable here is refused
        rather than truth-tested; ``judge`` is the path that awaits it (G4).

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
            # Close it so the abandoned coroutine raises no ``RuntimeWarning`` at collection.
            close = getattr(verdict, "close", None)
            if callable(close):
                close()
            raise GateFaultError(
                f"{self._label()}: the gate returned an awaitable for candidate={candidate!r}, "
                "which is a fault in the wiring rather than a verdict about the proposal: an "
                "async gate needs the async path, because a coroutine is truthy and would be "
                "admitted without ever being judged",
                function_name=self._label(),
            )
        self._record(candidate, verdict)

    async def judge(self, candidate: Any) -> Verdict:
        """Run the gate on one candidate, awaiting it if it is async, and record a rejection.

        The direct path ``propose_k`` uses, and the only place an async gate is honored (G4).
        Raises the same ``GateFaultError`` ``admits`` does; unlike ``admits`` it *returns* a
        rejection rather than raising it.

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
            # Render the report now, so a broken ``report_text`` faults here, next to the gate.
            if not ok:
                verdict.report_text()
        except Exception as error:  # noqa: BLE001 — a broken verdict is a fault, not a verdict
            raise self._fault(candidate, error) from error
        if not ok:
            self.rejected.append(verdict)
        return verdict

    def _record(self, candidate: Any, verdict: Verdict) -> None:
        """Append a rejection and raise it as the re-ask, or return quietly.

        A verdict whose ``ok`` or ``report_text`` raises is a fault: it raises
        ``GateFaultError`` and is not recorded in ``rejected``.
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
            f"which is a fault in the {part} rather than a verdict about the proposal: "
            f"{type(error).__name__}: {error}",
            function_name=self._label(),
        )

    def _label(self) -> str:
        """The owner name every message here is prefixed with; the same one the tool name uses."""
        return _owner_name(self)

    # ── Wiring ──

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
        extra = tuple(overrides.pop("post_conditions", ()))
        fn = self.compiled(method, **overrides)
        conditions = (*fn.config.post_conditions, *extra)
        if inspect.iscoroutinefunction(self.gate) or inspect.iscoroutinefunction(type(self.gate).__call__):
            raise WiringError(
                f"{self._label()}: the gate is async, and gated() runs it as a synchronous "
                "post-condition; judge and propose_k await an async gate",
                function_name=self._label(),
            )
        # Every attached condition is checked, not just admits.
        self._check_no_collision(method, *conditions)
        return fn.replace(post_conditions=(self.admits, *conditions))

    def _check_no_collision(self, method: str, *conditions: Any) -> None:
        """Refuse at wiring time if a result parameter shares a propose parameter's name (G5).

        The runtime passes the result positionally and injects, by keyword, every bound
        argument whose name appears in the validator's signature. A first parameter named
        like a bound argument is filled twice, and the resulting ``TypeError`` would reach
        the model as a validation failure. Later parameters are not checked: keyword
        injection into them is how a validator reads the call's arguments.
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
        /,
        *run_args: Any,
        method: str = "propose",
        seed: Sequence[Mapping[str, Any]] = (),
        **run_kwargs: Any,
    ) -> list[tuple[Any, Verdict]]:
        """Take k independent proposals from one shared context and return the admitted ones (G6).

        One thread is spawned for ``method`` and run once per entry in ``seed``; then it is
        forked ``k - 1`` times, so the original is branch 0 and every branch starts from the
        seeded context. Each branch runs exactly one propose cycle with the same arguments,
        and each result is judged by the gate directly. The branches carry no post-conditions,
        not even those declared on the method, so each draws one proposal and ``rejected``
        counts the ones the gate turned away.

        Every branch is spawned with ``coordinator_tools_enabled=False``, without
        ``list_threads`` or ``send_message``, so no branch can message or wait on a sibling.
        The method's own tools still reach every branch.

        Each ``seed`` entry is one cycle's keyword arguments, run on the root, so every branch
        inherits those turns. A ``notify`` still pending on the root when it forks is not
        copied into the branches.

        Branches run serially, so with a model whose ``cache_config`` places a cache point,
        every branch after the first reads the shared prefix from the provider cache.

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
            scalar is computed.

        Raises:
            WiringError: ``k`` is not at least 1, ``method`` has a parameter named ``method``
                or ``seed``, or ``admits`` collides with ``method`` (G5).
            GateFaultError: The gate or the candidate extractor failed, with the same wording
                ``admits`` uses. The exception propagates after every thread is retired.
        """
        if k < 1:
            raise WiringError(f"propose_k needs at least one branch, got k={k}", function_name=self._label())
        parameters = inspect.signature(getattr(self, method)).parameters
        shadowed = [keyword for keyword in ("method", "seed") if keyword in parameters]
        if shadowed:
            raise WiringError(
                f"{self._label()}: {method!r} has parameter(s) {shadowed!r}, which are propose_k's own "
                "keywords; rename them to beam over this method",
                function_name=self._label(),
            )
        self._check_no_collision(method)

        threads: list[MethodThread] = []
        admitted: list[tuple[Any, Verdict]] = []
        try:
            root = await self.spawn(method, coordinator, coordinator_tools_enabled=False, post_conditions=())
            threads.append(root)
            for cycle in seed:
                await root.run(**cycle)
            for _ in range(k - 1):
                threads.append(await root.fork())

            for thread in threads:
                response = await thread.run(*run_args, **run_kwargs)
                try:
                    candidate = self.candidate_of(response)
                except Exception as error:  # noqa: BLE001 — an extractor bug must not read as a verdict
                    raise self._fault(response, error, part="candidate extractor") from error
                verdict = await self.judge(candidate)
                if verdict.ok:
                    admitted.append((candidate, verdict))
        except BaseException:
            await _retire_all(threads)  # the beam's own error is the one that propagates
            raise
        failure = await _retire_all(threads)
        if failure is not None:
            raise failure
        return admitted


async def _retire_all(threads: Sequence[MethodThread]) -> Exception | None:
    """Retire every thread, returning the first failure rather than stopping at it."""
    failure: Exception | None = None
    for thread in threads:
        try:
            await thread.retire()
        except Exception as error:  # noqa: BLE001 — the remaining threads still need retiring
            failure = failure or error
    return failure
