"""``Recalled`` + ``Recall``: memory as a call argument, declared on the signature.

``Annotated[list[str], Recalled("guidance", k=2)]`` on an ``@ai_method``
parameter declares that the parameter is filled from memory;
``Recall(agent, backend).trace(...)`` retrieves it fresh on every call and
passes the ``ParameterView`` to ``AIFunction.trace`` as a keyword argument.
``collect_nodes((args, kwargs))`` runs inside ``AIFunction.trace``, so the view
becomes a parameter node that ``TextGradOptimizer`` can update. A view the
method creates in its body, renders into its docstring or reads from ``self``
is not a call argument and is not discovered.

The marked parameter stays in the tool schema, so a peer agent calling the
capability as a typed tool supplies the value itself.

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
    schema, so a peer can still supply it by hand. The marker is a frozen
    dataclass, which the strands tool decorator accepts inside ``Annotated``.

    R2 — every recalled value enters the traced call as a keyword argument,
    so ``collect_nodes((args, kwargs))`` inside ``AIFunction.trace`` finds the
    ``ParameterView`` and it becomes a gradient target. Nothing on ``self``,
    neither the binder's nor the agent's, ever holds a ``ParameterView``.

    R3 — nothing is cached: every ``trace`` retrieves a fresh view (a view
    emits once, so a second trace yields a fresh node) and compiles the method
    through the agent's ``compiled``.

    R4 — a marked parameter the caller supplied by keyword passes through
    untouched and triggers no retrieval; a query for it is refused as dead.

    R5 — every guard fires before any retrieval and before any model call:
    a reserved-name collision or a positional-only marked parameter at
    ``bound``, a positional shadow, a query mismatch or a method that does not
    compile at ``trace``. A search-mode parameter has no default query.

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

from .core import MethodAgent, MethodError, _owner_name, _parameter_hints, _published_name

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

    ``k=None`` recalls the parameter whole (``backend.recall``, derivation
    ``"full"``), for a memory the agent reads in full every time, such as one
    accumulated summary. ``k=<int>`` searches it (``backend.search``,
    derivation ``"search"``), for a memory that is a collection of entries.
    A search's meta names the entries the forward pass retrieved, so
    consolidation edits only those; a full recall carries no such meta, so
    feedback lands on the whole value.

    ``source`` is the backend's parameter name (a schema field path,
    slash-separated for nesting) and need not match the Python parameter name:
    ``advice: Annotated[..., Recalled("guidance")]`` reads the ``guidance``
    field into ``advice``.

    The strands tool decorator ignores this marker inside ``Annotated``, so a
    marked parameter can be offered through ``load_tools`` (R1).

    Args:
        source: The memory field the parameter is filled from.
        k: ``None`` to recall the field whole; a positive integer to search it
            and keep the top ``k`` entries.
    """

    source: str
    k: int | None = None

    def __post_init__(self) -> None:
        """Refuse a ``k`` that cannot retrieve anything.

        ``search`` with ``k=0`` returns no entries, so the prompt would lose its
        recalled section without any error.

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

    Reads ``__metadata__`` and recurses into ``get_args``, as
    ``detect_procedural_params`` does, so ``Traceable[Annotated[T, Recalled(...)]]``
    is found as well as a bare ``Annotated[T, Recalled(...)]``. Two distinct
    markers on one annotation are refused, so which field a parameter reads
    never depends on annotation order.
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

    Takes the *bound* method (``instance.choose``), whose hints exclude
    ``self``. An annotation ``get_type_hints`` cannot resolve raises rather
    than reading as unmarked, since an unmarked parameter is never retrieved
    (R7).

    Args:
        method: A bound ``@ai_method`` (or any callable with resolvable hints).

    Returns:
        Parameter name to its ``Recalled`` marker, for the marked ones only.

    Raises:
        RecallMarkerError: An annotation carries two distinct markers.
        NameError: An annotation names a type that cannot be resolved.
    """
    found: dict[str, Recalled] = {}
    for name, hint in _parameter_hints(method).items():
        marker = _marker_of(hint)
        if marker is not None:
            found[name] = marker
    return found


# ── The binder ──


class Recall:
    """Binds one ``MethodAgent`` to one ``MemoryBackend`` and fills its marked parameters per call.

    Every marker's ``source`` is a field path on this one backend; an agent
    that reads from two stores uses two binders.

    Positional ``*args`` are forwarded verbatim; injected parameters go in by
    keyword. ``queries`` maps parameter name to query string, so several marked
    parameters on one method each get their own. ``overrides`` is passed to
    ``compiled`` for the call. Retrieval runs before the model call, outside
    the validation path, and a backend exception propagates unchanged (R7).
    The binder holds the agent and the backend and nothing that resembles
    state (R2, R3).

    Args:
        agent: The ``MethodAgent`` whose ``@ai_method`` s carry ``Recalled`` markers.
        backend: The store every marker's ``source`` is a field path on.
    """

    RESERVED = ("queries", "overrides")
    """Keyword names ``trace`` owns. A bound parameter sharing one is refused at wiring time."""

    __slots__ = ("_agent", "_backend")

    def __init__(self, agent: MethodAgent, backend: MemoryBackend) -> None:
        self._agent = agent
        self._backend = backend

    @property
    def agent(self) -> MethodAgent:
        """The bound agent; read-only."""
        return self._agent

    @property
    def backend(self) -> MemoryBackend:
        """The bound store: what ``build_graph_from_result`` needs in its ``backends`` list."""
        return self._backend

    # ── Wiring ──

    def bound(self, method: str) -> dict[str, Recalled]:
        """The marked parameters of one method, after checking the method can be bound at all.

        ``trace`` calls this, so the same guards run for a caller that only
        introspects the markers and one that traces.

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
        # The binder injects by keyword, so a positional-only marked parameter cannot be filled.
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

        Every query is validated and the method compiled through the agent's
        ``compiled`` first, so a misuse costs neither a retrieval nor a model
        call; then each marked parameter is fetched fresh and injected as a
        keyword argument; then the method is traced, which is where
        ``collect_nodes`` finds the views and turns them into graph edges (R2). A parameter the caller
        supplied explicitly passes through untouched and no retrieval happens
        for it (R4).

        ``method`` is positional-only, so a method parameter named ``method``
        forwards through ``**kwargs``.

        Args:
            method: Which ``@ai_method`` to trace.
            args: Positional arguments, forwarded verbatim. They may not reach a
                marked parameter.
            queries: Parameter name to query string, one per search-mode
                parameter this call will retrieve. Required for those and
                refused for anything else.
            overrides: ``ThreadConfig`` overrides for ``compiled``, for example
                ``{"model": model}``.
            kwargs: Keyword arguments, forwarded verbatim. A marked name here is
                an explicit supply.

        Returns:
            The ``Result`` from ``AIFunction.trace``. Its ``inputs`` are the
            handles this call carried, in discovery order, so a caller reads the
            retrieved entry ids off ``result.inputs[i].meta["results"]``, the
            mapping consolidation targets.

        Raises:
            RecallWiringError: A guard refused the call before any retrieval (R5).
            CompilationError: ``method`` is not an ``@ai_method``, before any retrieval.
        """
        marked = self.bound(method)
        self._check_no_positional_shadow(method, marked, len(args))
        asked = dict(queries or {})
        self._check_queries(method, marked, asked, kwargs)
        fn = self._agent.compiled(method, **dict(overrides or {}))

        filled = dict(kwargs)
        # Inside a running cycle, ``recall``/``search`` would emit the recall event on the
        # ambient thread and mark the view emitted, leaving the traced thread's graph with no
        # parameter nodes. ``no_thread_scope`` defers emission to ``AIFunction.trace`` (R6).
        with no_thread_scope():
            for name, marker in marked.items():
                if name in kwargs:
                    continue
                if marker.k is None:
                    filled[name] = await self._backend.recall(marker.source)
                else:
                    filled[name] = await self._backend.search(marker.source, asked[name], marker.k)
        return await fn.trace(*args, **filled)

    # ── Guards ──

    def _check_no_positional_shadow(self, method: str, marked: Mapping[str, Recalled], positional: int) -> None:
        """Refuse positional arguments that would land on a marked parameter.

        Positionals bind left to right, so with ``choose(playbook, state)`` the
        call ``trace("choose", "S")`` would put ``"S"`` in ``playbook``. Only the
        slots the positionals reach are checked, and a marked parameter
        supplied by keyword is always allowed.
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

        A query is refused, with the reason it would be unused, when its name is
        not marked, the parameter was supplied explicitly, or the parameter is
        full-recall. A search-mode parameter this call retrieves must have a
        query; there is no default.
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
        """The capability's published name, so an error names what the tool schema names."""
        return _published_name(self._agent, method)

    def __repr__(self) -> str:
        """The bound agent's owner name and the backend id."""
        return f"<Recall {_owner_name(self._agent)} over {self._backend.backend_id!r}>"
