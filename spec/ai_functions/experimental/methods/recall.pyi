"""``Recalled`` + ``Recall``: memory as a call argument, declared on the signature.

A learning loop that retrieves guidance per decision and hands it to ``trace`` as
a handle follows four rules whose violation is silent: recall freshly per call,
pass the view whole, never interpolate, never stash it on ``self``. This module
turns all four into a mechanism. The injection happens at the trace boundary
and nowhere else, because ``collect_nodes((args, kwargs))`` runs inside
``AIFunction.trace``: the argument list of the traced call is the one place a
recalled value can enter and still be a gradient target. The marker declares
and the binder performs, so the parameter stays in the tool schema for a peer
that supplies it by hand. Nothing is cached, not the view (single-use emission)
and not the compiled function (compiled per trace through the instance).

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
    ``"search"``), which narrows the prompt and the gradient because its meta
    names the entries the forward pass retrieved. ``source`` is the backend's
    field path and need not match the Python parameter name. A plain frozen
    dataclass rather than a ``pydantic.Field`` (R1), so the installed strands
    tool decorator accepts the marked parameter through ``load_tools``.

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

    Takes the *bound* method, whose hints already exclude ``self``. Looks
    through unions, so ``Traceable[Annotated[T, Recalled(...)]]`` is found.
    ``get_type_hints`` is allowed to raise: an unresolvable annotation is a
    wiring bug and reads as one rather than as "unmarked" (R7).

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

    One binder, one backend: two stores means two binders over the same agent.
    Positional ``*args`` are forwarded verbatim; injected parameters go in by
    keyword; ``queries`` maps parameter name to query string; ``overrides`` is
    ``compiled``'s override mapping. Retrieval errors are not fault-wrapped
    (R7). The binder holds the agent and the backend and nothing that resembles
    state (R2, R3).

    Args:
        agent: The ``MethodAgent`` whose ``@ai_method`` s carry ``Recalled`` markers.
        backend: The store every marker's ``source`` is a field path on.
    """

    RESERVED: tuple[str, str]
    """Keyword names ``trace`` owns. A bound parameter sharing one is refused at wiring time."""

    def __init__(self, agent: MethodAgent, backend: MemoryBackend) -> None: ...
    @property
    def agent(self) -> MethodAgent:
        """The bound agent. Read-only, so the introspected markers keep describing the agent being called."""
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

        Queries are validated first, then each marked parameter is fetched
        fresh under ``no_thread_scope`` and injected as a keyword argument, then
        the method is compiled through the instance's own ``compiled`` and
        traced, where ``collect_nodes`` finds the views (R2). A parameter the
        caller supplied explicitly passes through untouched (R4). ``method`` is
        positional-only so a bound parameter of that name forwards through
        ``**kwargs``.

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
        """
        ...

    def __repr__(self) -> str:
        """The bound agent's owner name and the backend id."""
        ...
