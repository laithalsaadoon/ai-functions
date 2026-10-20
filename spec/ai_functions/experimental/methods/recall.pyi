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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...memory import MemoryBackend
from ...types.graph import Result
from .core import MethodAgent, MethodError

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

@dataclass(frozen=True)
class Recalled:
    """Annotated metadata: this parameter arrives from memory, per call.

    ``k=None`` recalls the parameter whole (``backend.recall``, derivation
    ``"full"``); ``k=<int>`` searches it (``backend.search``, derivation
    ``"search"``), whose meta names the entries the forward pass retrieved so
    consolidation edits only those. ``source`` is the backend's field path and
    need not match the Python parameter name. The strands tool decorator
    ignores this marker inside ``Annotated``, so a marked parameter can be
    offered through ``load_tools`` (R1).

    Args:
        source: The memory field the parameter is filled from.
        k: ``None`` to recall the field whole; a positive integer to search it
            and keep the top ``k`` entries.
    """

    source: str
    k: int | None = None

    def __post_init__(self) -> None:
        """Refuse a ``k`` that cannot retrieve anything.

        Raises:
            RecallMarkerError: ``k`` is an integer below 1.
        """
        ...

def recalled_params(method: Callable[..., Any]) -> dict[str, Recalled]:
    """The marked parameters of a method, in signature order.

    Takes the *bound* method, whose hints exclude ``self``. Looks through
    unions, so ``Traceable[Annotated[T, Recalled(...)]]`` is found. An
    annotation ``get_type_hints`` cannot resolve raises rather than reading as
    unmarked (R7).

    Args:
        method: A bound ``@ai_method`` (or any callable with resolvable hints).

    Returns:
        Parameter name to its ``Recalled`` marker, for the marked ones only.

    Raises:
        RecallMarkerError: An annotation carries two distinct markers.
        NameError: An annotation names a type that cannot be resolved.
    """
    ...

class Recall:
    """Binds one ``MethodAgent`` to one ``MemoryBackend`` and fills its marked parameters per call.

    Every marker's ``source`` is a field path on this one backend; an agent
    that reads from two stores uses two binders. Positional ``*args`` are
    forwarded verbatim; injected parameters go in by keyword; ``queries`` maps
    parameter name to query string; ``overrides`` is passed to ``compiled``
    for the call. A backend exception during retrieval propagates unchanged
    (R7). The binder holds the agent and the backend and nothing that
    resembles state (R2, R3).

    Args:
        agent: The ``MethodAgent`` whose ``@ai_method`` s carry ``Recalled`` markers.
        backend: The store every marker's ``source`` is a field path on.
    """

    RESERVED: tuple[str, str]
    """Keyword names ``trace`` owns. A bound parameter sharing one is refused at wiring time."""

    def __init__(self, agent: MethodAgent, backend: MemoryBackend) -> None: ...
    @property
    def agent(self) -> MethodAgent:
        """The bound agent; read-only."""
        ...

    @property
    def backend(self) -> MemoryBackend:
        """The bound store: what ``build_graph_from_result`` needs in its ``backends`` list."""
        ...

    def bound(self, method: str) -> dict[str, Recalled]:
        """The marked parameters of one method, after checking the method can be bound at all.

        Args:
            method: The ``@ai_method`` name on the bound agent.

        Returns:
            Parameter name to ``Recalled`` marker, in signature order.

        Raises:
            RecallWiringError: A parameter of the method is named after one of
                :attr:`RESERVED`, or a marked parameter is positional-only and
                therefore unfillable by keyword (R5).
        """
        ...

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

        Queries are validated and the method compiled through the agent's
        ``compiled`` first, then each marked parameter is fetched fresh under
        ``no_thread_scope`` and injected as a keyword argument, then the method
        is traced, where ``collect_nodes`` finds the views (R2). A parameter the caller
        supplied explicitly passes through untouched (R4). ``method`` is
        positional-only, so a method parameter named ``method`` forwards
        through ``**kwargs``.

        Args:
            method: Which ``@ai_method`` to trace.
            args: Positional arguments, forwarded verbatim; they may not reach
                a marked parameter.
            queries: Parameter name to query string, one per search-mode
                parameter this call will retrieve. Required for those and
                refused for anything else.
            overrides: ``ThreadConfig`` overrides for ``compiled``.
            kwargs: Keyword arguments, forwarded verbatim. A marked name here is
                an explicit supply.

        Returns:
            The ``Result`` from ``AIFunction.trace``; its ``inputs`` are the
            handles this call carried, in discovery order.

        Raises:
            RecallWiringError: A guard refused the call before any retrieval (R5).
            CompilationError: ``method`` is not an ``@ai_method``, before any retrieval.
        """
        ...

    def __repr__(self) -> str:
        """The bound agent's owner name and the backend id."""
        ...
