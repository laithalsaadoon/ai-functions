"""``@ai_method``: ``@ai_function`` applied to methods.

An ``@ai_method`` on a ``MethodAgent`` compiles, per instance, into an
``AIFunction`` whose ``prompt_fn`` is the bound method. Python removes ``self``
from a bound method's signature, so the tool schema carries the method's typed
parameters, while the docstring, rendered as the prompt template, can still
read the instance through ``self``. Call arguments are what
``TextGradOptimizer`` can update; state on ``self`` is fixed per instance.

    class Analyst(MethodAgent):
        def __init__(self, plane):
            self.plane = plane
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

Invariants:
    M1 — ``self`` never reaches the tool schema. The compiled ``prompt_fn``
    carries the *bound* method's signature, so the model is asked for the
    method's parameters and nothing else.

    M2 — annotations are resolved with ``include_extras=True``, so
    ``Annotated`` markers (``Procedural``, ``Frozen``, ``Recalled``) survive
    on the compiled function for the runtime's procedural detection and for
    any schema built from its hints. The strands tool decorator refuses a
    ``pydantic.Field`` inside ``Annotated``, so a ``Procedural`` parameter
    runs on a thread but cannot be offered through ``load_tools``.

    M3 — a compiled tool is named ``{owner}_{method}`` (a ``name=`` given to
    ``@ai_method`` or ``compiled`` wins), and the ``AIFunction``, the live
    thread and every error message use the same name, so two capabilities on
    one agent never collide and every message names a tool the caller can find
    in its schema.

    M4 — the docstring is the prompt template, interpolated with
    ``{self: instance, **bound.arguments}`` and rendered as an
    ``@ai_function`` docstring is; a string or ``Template`` returned by the
    method body is the prompt instead. A method with neither is refused at
    render time, never rendered blank.

    M5 — ``spawn`` requires an explicit coordinator, so the thread lands in
    the caller's registry with a parent edge and a shared event log.

    M6 — ``retire`` is idempotent against this object and against the
    runtime (a thread something else already tore down is not an error), and
    every operation after ``retire`` raises ``RetiredThreadError`` naming the
    capability. A teardown that fails leaves the thread live.

    M7 — a ``MethodThread`` holds a ``ThreadHandle`` and nothing that
    resembles state. History is the coordinator's event log, reconstructed per
    cycle.

    M8 — a capability's model is resolved most-specific first: the call's
    ``model=``, the decorator's ``model=``, then the agent's ``model``
    attribute. With none of them set, the capability runs on the Strands
    default model, as a plain ``@ai_function`` does.

    M9 — ``MethodAgent.spawn`` takes ``seed_events`` and ``thread_id`` and
    hands them to ``Coordinator.spawn`` unchanged, so a ``SessionStore``'s
    saved log and id resume a method thread in place, as they resume an
    ``AIFunction`` thread. The coordinator's replay rules apply: at most one
    of ``seed_from`` and ``seed_events``, seeded before registration.
"""

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
