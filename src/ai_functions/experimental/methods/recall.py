"""``Recalled`` + ``Recall``: memory as a call argument, declared on the signature.

A learning loop that retrieves guidance per decision and hands it to ``trace`` as
a handle follows four rules whose violation is silent: recall freshly per call,
pass the view whole, never interpolate, never stash it on ``self``. A loop that
breaks one reports rounds while learning nothing. This module turns all four
into a mechanism.

**Why the injection happens at the trace boundary and nowhere else.**
``collect_nodes((args, kwargs))`` runs inside ``AIFunction.trace``, and
``ThreadHandle.run`` unwraps every handle to its ``.value`` before a prompt is
built. So a ``ParameterView`` created *inside* ``prompt_fn`` is invisible to the
optimizer (the scan already happened), and a view rendered into the docstring
template is invisible twice over, because ``__str__`` returns ``str(value)`` and
drops the edge. There is exactly one place a recalled value can enter and still
be a gradient target: the argument list of the traced call. That is the whole
reason this binder exists rather than a ``self.memory.search(...)`` line inside
the method body, and it is the outer-optimizer handle: what ``Recall`` injects
is what ``TextGradOptimizer`` later routes feedback into.

**Why the marker declares and the binder performs.**
``Annotated[list[str], Recalled("guidance", k=2)]`` on the ``@ai_method`` says
where the parameter comes from; ``Recall(agent, backend).trace(...)`` is what
fetches it. Splitting them is deliberate: the parameter deliberately *stays* in
the tool schema, because a peer agent calling this capability as a typed tool
supplies the content itself and only a training loop supplies it from memory.
``load_tools`` exposes the full ``prompt_fn`` signature and no upstream
mechanism drops a parameter or auto-fills one, so a marker that hid the
parameter would be an upstream change wearing a library's clothes.

**Why nothing is cached, not the view, not the compiled function.** A
``ParameterView`` is single-use: ``recall``/``search`` emit at recall time when a
thread is resolvable, and ``trace`` flushes the rest, so one logical recall
lands in one log and a *reused* view produces a parameter node on the first
traced call and none on any later one. The binder therefore never stores a
view; reuse is unrepresentable rather than discouraged. ``compiled`` is called
per trace for the same reason ``spawn`` calls it: tests bind a scripted model by
replacing ``compiled`` on the *instance*, and a function compiled once at wiring
time would silently reach a real model.

    class Navigator(MethodAgent):
        @ai_method(Choice, description="Choose the next transition")
        def choose(self, advice: Annotated[list[str], Recalled("guidance", k=2)],
                   state: str) -> Choice:
            '''Guidance so far: {advice}

            You are in {state}. Choose.
            '''

``await Recall(navigator, store).trace("choose", state="S", queries={"advice": "S"})``
searches ``guidance`` for the two entries about this decision, injects the view
as ``advice``, and returns the ``Result`` whose ``inputs[0].meta["results"]``
names the entries the decision actually read.

Invariants:
    R1 — the marker declares, the binder performs. ``Recalled`` inside
    ``Annotated`` says which memory field a parameter comes from and how
    (``k=None`` whole, ``k=<int>`` searched); the parameter stays in the tool
    schema, so a peer can still supply it by hand. The marker is a plain
    frozen dataclass, never a ``pydantic.Field``, so the installed strands tool
    decorator accepts it.

    R2 — every recalled value enters the traced call as a keyword argument,
    so ``collect_nodes((args, kwargs))`` inside ``AIFunction.trace`` finds the
    ``ParameterView`` and it becomes a gradient target. Nothing on ``self``,
    neither the binder's nor the agent's, ever holds a ``ParameterView``.

    R3 — nothing is cached: not the view (single-use emission; a second trace
    yields a fresh node) and not the compiled function (``compiled`` is
    called through the instance per trace, so an instance-level scripted
    binding reaches every traced call).

    R4 — explicit beats implicit: a marked parameter the caller supplied by
    keyword passes through untouched and triggers no retrieval; a query for it
    is refused as dead.

    R5 — every guard fires before any retrieval and before any model call:
    a reserved-name collision or a positional-only marked parameter at
    ``bound``, a positional shadow or a query mismatch at ``trace``. A
    search-mode parameter has no default query.

    R6 — retrieval runs under ``no_thread_scope`` so emission is deferred to
    ``AIFunction.trace`` and the recall event lands on the traced thread, even
    when the binder is used from inside a running cycle.

    R7 — one distinct ``Recalled`` per annotation (looking through unions), an
    unresolvable annotation raises rather than reading as unmarked, and a
    backend exception during retrieval propagates unwrapped.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ai_functions.memory import MemoryBackend
from ai_functions.types import no_thread_scope
from ai_functions.types.graph import Result

from .core import MethodAgent, MethodError, _owner_name, _published_name

# ── Errors ──


class RecallError(MethodError):
    """Base of the recall binder's failure modes.

    Args:
        message: Human-readable explanation.
        function_name: The compiled tool name (``{owner}_{method}``) that
            raised, when known.
    """


class RecallMarkerError(RecallError):
    """A ``Recalled`` marker cannot be honored: ``k < 1``, or two distinct markers on one annotation (R7)."""


class RecallWiringError(RecallError):
    """A ``bound``/``trace`` guard refused the call before any retrieval or model call (R5)."""


# ── The marker ──


@dataclass(frozen=True)
class Recalled:
    """Annotated metadata: this parameter arrives from memory, per call.

    Two modes, and the difference is not a tuning knob. ``k=None`` recalls the
    parameter whole (``backend.recall``, derivation ``"full"``), which is right
    when the memory *is* the answer: one accumulated summary the agent reads
    every time. ``k=<int>`` searches it (``backend.search``, derivation
    ``"search"``), which is right when the memory is a collection and the agent
    should read the two or three entries about the decision in front of it.
    Search narrows the prompt and, just as importantly, narrows the gradient:
    its meta names the entries the forward pass retrieved, and consolidation
    edits only those. A full recall carries no such meta, so feedback lands on
    the whole value.

    ``source`` is the backend's parameter name (a schema field path,
    slash-separated for nesting) and is deliberately not required to match the
    Python parameter name. The signature names the argument for the model
    reading the tool schema and the template rendering the prompt; the memory
    names the field for the store. ``advice: Annotated[..., Recalled("guidance")]``
    is a legitimate and common shape.

    A plain frozen dataclass rather than a ``pydantic.Field`` (R1): the
    installed strands tool decorator refuses a ``FieldInfo`` inside
    ``Annotated`` and ignores any other metadata object, so this shape keeps
    the marked parameter offerable through ``load_tools``.

    Args:
        source: The memory field the parameter is filled from.
        k: ``None`` to recall the field whole; a positive integer to search it
            and keep the top ``k`` entries.
    """

    source: str
    k: int | None = None

    def __post_init__(self) -> None:
        """Refuse a ``k`` that cannot retrieve anything.

        ``k=0`` is accepted by ``search`` and returns an empty list with empty
        ``results`` meta, which is the fail-soft shape this module exists to
        prevent: the prompt loses its guidance section, the graph loses its
        retrieved entries, and nothing anywhere says so.

        Raises:
            RecallMarkerError: ``k`` is an integer below 1.
        """
        if self.k is not None and self.k < 1:
            raise RecallMarkerError(
                f"Recalled({self.source!r}, k={self.k!r}): k is how many entries to retrieve, so "
                f"it must be at least 1; k=None means recall the parameter whole"
            )


def _marker_of(hint: object) -> Recalled | None:
    """The ``Recalled`` in an annotation, looking through unions, or None.

    Mirrors the runtime's own marker detection (``detect_procedural_params``):
    read ``__metadata__``, then recurse into ``get_args``, so a parameter
    annotated ``Traceable[Annotated[T, Recalled(...)]]`` is found as well as a
    bare ``Annotated[T, Recalled(...)]``. Measured: ``typing.get_args`` on the
    ``Traceable`` alias yields the ``Annotated`` member with its metadata
    intact.

    Two *distinct* markers on one annotation are refused rather than resolved
    by order. Every other ambiguity in this module fails loud, and this one is
    the easiest to write by accident: a merge that keeps both sides'
    ``Annotated`` metadata, or a union whose members were marked separately.
    Which store a parameter reads from must not depend on annotation order.
    """
    found: list[Recalled] = []

    def _walk(node: object) -> None:
        for meta in getattr(node, "__metadata__", ()):
            if isinstance(meta, Recalled) and meta not in found:
                found.append(meta)
        for arg in typing.get_args(node):
            _walk(arg)

    _walk(hint)
    if len(found) > 1:
        raise RecallMarkerError(
            f"annotation carries {len(found)} distinct `Recalled` markers ({found!r}); which "
            f"retrieval runs would depend on metadata order, so declare exactly one"
        )
    return found[0] if found else None


def recalled_params(method: Callable[..., Any]) -> dict[str, Recalled]:
    """The marked parameters of a method, in signature order.

    Takes the *bound* method (``instance.choose``), whose hints already exclude
    ``self``; the unbound-function path needs a ``pop("self")`` that
    ``compile_ai_method`` has to do and this does not.

    ``get_type_hints`` is allowed to raise. The runtime's equivalent swallows
    the exception and returns an empty set, which is the correct trade there: a
    resolution failure costs a procedural parameter its sandbox definition and
    the prompt still runs. Here the same silence would mean *no retrieval*, so
    a typo in a type name would present as an agent that reads an empty
    playbook forever. An unresolvable annotation is a wiring bug and reads as
    one (R7).

    Args:
        method: A bound ``@ai_method`` (or any callable with resolvable hints).

    Returns:
        Parameter name to its ``Recalled`` marker, for the marked ones only.

    Raises:
        RecallMarkerError: An annotation carries two distinct markers.
        NameError: An annotation names a type that cannot be resolved.
    """
    hints = typing.get_type_hints(method, include_extras=True)
    hints.pop("return", None)
    found: dict[str, Recalled] = {}
    for name, hint in hints.items():
        marker = _marker_of(hint)
        if marker is not None:
            found[name] = marker
    return found


# ── The binder ──


class Recall:
    """Binds one ``MethodAgent`` to one ``MemoryBackend`` and fills its marked parameters per call.

    One binder, one backend, deliberately: ``source`` is a field path on *a*
    schema, and a binder that took several stores would have to disambiguate
    them in the marker, which would put storage topology into the agent's
    signature. Two stores means two binders over the same agent, which
    composes without either of them knowing.

    **What arrives how.** Positional ``*args`` are forwarded verbatim; injected
    parameters go in by keyword. ``queries`` maps parameter name to query
    string, so several marked parameters on one method each get their own, the
    shape a loop that recalls twice per round already has. ``overrides`` is
    ``compiled``'s override mapping, which is how a test injects a scripted
    model per trace without monkeypatching the instance.

    **Retrieval errors are not fault-wrapped (R7).** A hook that runs *inside*
    a validator must re-raise a bug as something that says it is a bug, or it
    burns every retry masquerading as a verdict. Retrieval runs before the model
    call and outside the validation path entirely: a backend that raises has
    nothing to report to a model and no attempts to burn, so the exception
    propagates to the caller unchanged. Wrapping it would only hide the
    traceback.

    **What this class deliberately does not do.** No ``run`` wrapper, no
    live-thread emission, no consolidation, no optimizer coupling. It holds the
    agent and the backend and nothing that resembles state (R2, R3).

    Args:
        agent: The ``MethodAgent`` whose ``@ai_method`` s carry ``Recalled`` markers.
        backend: The store every marker's ``source`` is a field path on.
    """

    RESERVED = ("queries", "overrides")
    """Keyword names ``trace`` owns. A bound parameter sharing one is refused at wiring time.

    A tuple rather than a set so the refusal message lists them in a stable order.
    """

    __slots__ = ("_agent", "_backend")

    def __init__(self, agent: MethodAgent, backend: MemoryBackend) -> None:
        self._agent = agent
        self._backend = backend

    @property
    def agent(self) -> MethodAgent:
        """The bound agent.

        Read-only: a binder that could be re-pointed would let a caller believe
        the marked parameters it introspected still describe the agent being
        called.
        """
        return self._agent

    @property
    def backend(self) -> MemoryBackend:
        """The bound store: what ``build_graph_from_result`` needs in its ``backends`` list."""
        return self._backend

    # ── Wiring ──

    def bound(self, method: str) -> dict[str, Recalled]:
        """The marked parameters of one method, after checking the method can be bound at all.

        Both halves of the wiring question in one call, because a caller that
        introspects the markers and a caller that traces need the same guard to
        have run. ``trace`` calls this rather than ``recalled_params`` directly,
        so the refusal cannot be reached only through the path nobody takes.

        Args:
            method: The ``@ai_method`` name on the bound agent.

        Returns:
            Parameter name to ``Recalled`` marker, in signature order.

        Raises:
            RecallWiringError: A parameter of the method is named after one of
                :attr:`RESERVED`, or a marked parameter is positional-only and
                therefore unfillable by keyword (R5).
        """
        target = getattr(self._agent, method)
        parameters = inspect.signature(target).parameters
        shadowed = [name for name in self.RESERVED if name in parameters]
        if shadowed:
            raise RecallWiringError(
                f"{self._label(method)}: parameter(s) {shadowed!r} collide with "
                f"{self.RESERVED!r}, which are `Recall.trace`'s own keywords; a call passing "
                f"them would fill trace's parameter and leave the method's at its default, "
                f"silently: no error, no retrieval, and a prompt missing whatever they carried. "
                f"Rename the method's parameter.",
                function_name=self._label(method),
            )
        marked = recalled_params(target)
        # The binder can only inject by keyword, so a positional-only marked parameter is
        # unfillable; without this refusal the failure arrives AFTER the retrieval was
        # spent, as a TypeError about Python calling mechanics rather than the wiring mistake.
        unfillable = [
            name for name in marked if name in parameters and parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
        ]
        if unfillable:
            raise RecallWiringError(
                f"{self._label(method)}: recalled parameter(s) {unfillable!r} are positional-only, "
                f"and the binder injects by keyword; drop the `/` or move the marker to a "
                f"keyword-bindable parameter.",
                function_name=self._label(method),
            )
        return marked

    # ── The traced call ──

    async def trace(
        self,
        method: str,
        /,
        *args: Any,
        queries: Mapping[str, str] | None = None,
        overrides: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Result[Any]:
        """Retrieve every marked parameter the caller did not supply, then trace the method.

        The order is the point. Every query is validated first, so a misuse
        costs neither a retrieval nor a model call; then each marked parameter
        is fetched *fresh* and injected as a keyword argument; then the method
        is compiled through the instance's own ``compiled`` and traced, which is
        where ``collect_nodes`` finds the views and turns them into graph edges
        (R2).

        A parameter the caller supplied explicitly is passed through untouched
        and no retrieval happens for it: explicit beats implicit (R4), which is
        what makes a marked capability still callable by a peer agent that has
        the content in hand.

        ``method`` is positional-only so that a bound parameter named ``method``
        forwards through ``**kwargs`` instead of colliding with this signature.
        That is why :attr:`RESERVED` has two entries and not three.

        Args:
            method: Which ``@ai_method`` to trace.
            args: Positional arguments, forwarded verbatim. They may not reach a
                marked parameter; see ``_check_no_positional_shadow``.
            queries: Parameter name to query string, one per search-mode
                parameter this call will retrieve. Required for those and
                refused for anything else.
            overrides: ``ThreadConfig`` overrides for ``compiled``, for example
                ``{"model": ScriptedModel(...)}``.
            kwargs: Keyword arguments, forwarded verbatim. A marked name here is
                an explicit supply.

        Returns:
            The ``Result`` from ``AIFunction.trace``. Its ``inputs`` are the
            handles this call carried, in discovery order, so a caller reads the
            retrieved entry ids off ``result.inputs[i].meta["results"]``, the
            mapping consolidation targets.

        Raises:
            RecallWiringError: A guard refused the call before any retrieval (R5).
        """
        marked = self.bound(method)
        self._check_no_positional_shadow(method, marked, len(args))
        asked = dict(queries or {})
        self._check_queries(method, marked, asked, kwargs)

        filled = dict(kwargs)
        # ``no_thread_scope`` is essential, not hygiene (R6). When this runs inside a live
        # cycle (a capability body, a gate, any orchestration hosted on a thread) the
        # runtime's ambient scope is active, and ``recall``/``search`` would emit the recall
        # event against THAT thread and mark the view emitted. ``trace``'s own flush then
        # no-ops ("one logical recall, one event"), the traced thread's log carries no event,
        # and the graph reconstructs zero parameters: rounds reported, nothing learned, no
        # error anywhere. Suppressing the ambient scope defers emission to ``AIFunction.trace``,
        # which lands it on the thread the gradient must come from. Measured both ways.
        with no_thread_scope():
            for name, marker in marked.items():
                if name in kwargs:
                    continue
                if marker.k is None:
                    filled[name] = await self._backend.recall(marker.source)
                else:
                    filled[name] = await self._backend.search(marker.source, asked[name], marker.k)
        return await self._agent.compiled(method, **dict(overrides or {})).trace(*args, **filled)

    # ── Guards ──

    def _check_no_positional_shadow(self, method: str, marked: Mapping[str, Recalled], positional: int) -> None:
        """Refuse positional arguments that would land on a marked parameter.

        Python binds positionals left to right, so a caller cannot skip one.
        When a marked parameter comes first, ``choose(playbook, state, options,
        facts)``, a call written as ``trace("choose", state, options, facts)``
        shifts every argument one slot left. Both outcomes of that are bad and
        one is worse: ``TypeError: got multiple values for argument 'playbook'``
        if the binder injects anyway (measured), or, if the binder had inferred
        "the caller supplied it" from a partial bind, ``state`` sitting in the
        playbook slot with no retrieval performed at all: a wrong prompt that
        raises nothing. ``inspect.signature(...).bind_partial("S", "O")`` on
        that shape really does report ``{'advice': 'S', 'state': 'O'}``, which
        is why this is a guard rather than a docstring.

        Narrow on purpose: only the slots the positionals actually reach are
        checked. A method whose marked parameters all come after its plain ones
        can be called positionally with no complaint, and supplying a marked
        parameter *by keyword* is always allowed; that is the sanctioned
        explicit override.
        """
        if not positional or not marked:
            return
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        signature = inspect.signature(getattr(self._agent, method))
        names = [name for name, parameter in signature.parameters.items() if parameter.kind in kinds]
        reached = [name for name in names[:positional] if name in marked]
        if reached:
            raise RecallWiringError(
                f"{self._label(method)}: {positional} positional argument(s) would land on "
                f"recalled parameter(s) {reached!r}, because positional arguments bind left to "
                f"right and cannot skip one. Pass the plain arguments by keyword (or supply "
                f"{reached[0]!r} by keyword to override the recall deliberately).",
                function_name=self._label(method),
            )

    def _check_queries(
        self,
        method: str,
        marked: Mapping[str, Recalled],
        asked: Mapping[str, str],
        supplied: Mapping[str, Any],
    ) -> None:
        """Every query must be used, and every search-mode retrieval must have one.

        Both directions, and each one is a silent failure in the other's
        absence. A query nobody reads is misuse the caller cannot see (they
        believe retrieval was steered and it was not), so the three ways to
        write one are refused with the reason each is dead: the name is not
        marked, the parameter was supplied explicitly, or the parameter is
        full-recall and takes no query.

        The missing direction is the sharper one: a search with a defaulted or
        derived query retrieves confidently-ranked garbage and the run
        *succeeds*, so the loop trains on advice about a decision nobody was
        making. There is no defensible default for a query, so there is no
        default.
        """
        known = sorted(marked)
        label = self._label(method)
        for name in asked:
            if name not in marked:
                raise RecallWiringError(
                    f"{label}: no parameter named {name!r} is marked `Recalled`, "
                    f"so the query for it would never be used; marked parameters are {known}",
                    function_name=label,
                )
            if name in supplied:
                raise RecallWiringError(
                    f"{label}: {name!r} was supplied explicitly, so no retrieval "
                    f"happens for it and the query for it would never be used; pass one or the "
                    f"other",
                    function_name=label,
                )
            if marked[name].k is None:
                raise RecallWiringError(
                    f"{label}: {name!r} is Recalled({marked[name].source!r}) with "
                    f"k=None, which recalls the parameter whole and takes no query; pass k to "
                    f"the marker to search instead",
                    function_name=label,
                )
        for name, marker in marked.items():
            if marker.k is None or name in supplied or name in asked:
                continue
            raise RecallWiringError(
                f"{label}: {name!r} is Recalled({marker.source!r}, k={marker.k}) "
                f"and needs a query for this call: pass queries={{{name!r}: ...}}. There is no "
                f"default: a derived or empty query retrieves entries ranked against the wrong "
                f"question and the call would succeed with irrelevant guidance in the prompt.",
                function_name=label,
            )

    def _label(self, method: str) -> str:
        """The capability's published name, so an error names what the tool schema names.

        ``_published_name`` rather than a local copy: it exists in ``core.py``
        precisely so the compiled tool name and the lifecycle's error messages
        cannot drift apart, and an error from here has to name a capability the
        caller can find in the schema it was reading.
        """
        return _published_name(self._agent, method)

    def __repr__(self) -> str:
        """The bound agent's owner name and the backend id."""
        return f"<Recall {_owner_name(self._agent)} over {self._backend.backend_id!r}>"
