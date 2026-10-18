"""``@ai_method``: the library's decorator paradigm, applied to methods.

An object-oriented agent that stays addressable by peers compiles every
capability down to one ``str`` parameter, and that costs three things the
``@ai_function`` decorator gives away free. The typed contract: ``load_tools``
builds an ``AIFunction``'s schema from ``inspect.signature(prompt_fn)``, and a
single ``request: str`` erases what makes a call checkable. The docstring as
prompt template: the decorator interpolates the docstring with the call's bound
arguments, where a hand-built ``brief()`` turns declarative text into control
flow. Learnable parameters: ``TextGradOptimizer`` routes gradients into
``ParameterNode`` s that ``collect_nodes((args, kwargs))`` discovers over the
*call arguments*, so state hidden on ``self`` cannot be optimized at all.

A bound method recovers all three at once, because Python removes ``self`` from
a bound method's signature: ``inspect.signature(instance.method)`` is already
the typed contract, while ``self`` stays reachable for what varies per instance.
The decorator keeps operating on a function; the instance supplies the closure.
A gradient target must be a call argument to be discoverable, and a fixed input
a validator needs belongs on ``self``, where the optimizer cannot reach it.

    class Analyst(MethodAgent):
        def __init__(self, plane):
            self.evidence = load(plane)

        @ai_method(Finding, description="Analyze one plane over a window")
        def analyze(self, window: str, max_records: int = 20) -> Finding:
            '''Analyze the {self.plane} plane over {window}.
            Your private evidence: {self.evidence}
            '''

``Analyst("metrics").compiled("analyze")`` is an ``AIFunction`` with
``input_shape == STRUCTURED``, a tool schema carrying ``window`` and
``max_records``, and a prompt rendered from the docstring with this instance's
evidence in it. ``Analyst("metrics").spawn("analyze", coordinator)`` is the
same capability on a live thread whose successive ``run`` calls share history.

This package needs nothing from the core beyond what is already public:
``AIFunction`` (constructed, never subclassed), ``ThreadConfig``,
``Coordinator.spawn`` and ``ThreadHandle``. It imports no other experimental
package.

Invariants:
    M1 — ``self`` never reaches the tool schema. The compiled ``prompt_fn``
    carries the *bound* method's signature, so the model is asked for the
    method's parameters and nothing else.

    M2 — annotations are resolved with ``include_extras=True``, so
    ``Annotated`` markers (``Procedural``, ``Frozen``, ``Recalled``)
    survive on the compiled function for the runtime's procedural detection
    and for any schema built from its hints. Resolving without extras flattens
    a code parameter to plain ``str`` and it silently becomes an ordinary
    prompt argument. (The installed strands tool decorator refuses a
    ``pydantic.Field`` inside ``Annotated``, so a ``Procedural`` parameter
    runs on a thread but cannot yet be offered through ``load_tools``; pinned
    in tests/methods/test_methods_ai_functions_contract.py.)

    M3 — a compiled tool is named ``{owner}_{method}`` (an
    ``@ai_method(..., name=...)`` rename wins), so two capabilities on one
    agent never collide on a caller's wire, and every error and log line names
    a thread the caller can find in the tool schema it was reading.

    M4 — the docstring is the prompt template, rendered with
    ``{self: instance, **bound.arguments}``; a non-``None`` return from the
    method body is the prompt verbatim. A method with neither is refused at
    render time, never rendered blank.

    M5 — ``spawn`` requires an explicit coordinator. The thread lands in the
    caller's registry with a parent edge and a shared event log; a private
    in-memory coordinator per call is unrepresentable here.

    M6 — ``retire`` is idempotent against this object and against the
    runtime (a thread something else already tore down is not an error), and
    every operation after ``retire`` raises ``RetiredThreadError`` naming the
    capability. A blank conversation with the right signature would be a wrong
    answer that looks like a right one.

    M7 — a ``MethodThread`` holds a ``ThreadHandle`` and nothing that
    resembles state. History is the coordinator's event log, reconstructed per
    cycle; a second copy on ``self`` would drift the moment a cycle was
    summarized, forked or replayed, and would be invisible to the optimizer.

    M8 — a capability's model is resolved most-specific first: the call's
    ``model=``, the decorator's ``model=``, then the agent's ``model``
    attribute. With none of them set, the capability runs on the Strands
    default model, as a plain ``@ai_function`` does.

    M9 — a ``MethodThread`` rehydrates through the public API.
    ``MethodAgent.spawn`` takes ``seed_events`` and ``thread_id`` and hands
    them to ``Coordinator.spawn`` unchanged, so a ``SessionStore``'s saved log
    and id resume a method thread in place exactly as they resume a bare
    ``AIFunction`` thread; the coordinator's replay contract (one of
    ``seed_from`` and ``seed_events``, seed in place before registration) is
    the whole of the rule, and this package adds no store of its own.
"""

from __future__ import annotations

from .core import (
    AIMethodSpec,
    CompilationError,
    MethodAgent,
    MethodError,
    MethodThread,
    RenderError,
    RetiredThreadError,
    ai_method,
    compile_ai_method,
    is_ai_method,
)
from .gated import Gate, GatedProposer, GateFaultError, GateRejectionError, Verdict, WiringError
from .recall import Recall, Recalled, RecallError, RecallMarkerError, RecallWiringError, recalled_params

__all__ = [
    "AIMethodSpec",
    "CompilationError",
    "Gate",
    "GateFaultError",
    "GateRejectionError",
    "GatedProposer",
    "MethodAgent",
    "MethodError",
    "MethodThread",
    "Recall",
    "RecallError",
    "RecallMarkerError",
    "RecallWiringError",
    "Recalled",
    "RenderError",
    "RetiredThreadError",
    "Verdict",
    "WiringError",
    "ai_method",
    "compile_ai_method",
    "is_ai_method",
    "recalled_params",
]
