"""Offline tests for ``methods/recall.py``: the marker, the injection, and the guards.

Thirty-two tests over the binder's own surface, plus the invariant guards: R2 asserted at the trace
boundary itself (``collect_nodes`` over the traced call's ``(args, kwargs)`` finds the ``ParameterView``,
and nothing on ``self`` holds one), the ``RecallError`` family, and the static import boundary.

Three kinds of claim, checked three ways on purpose. The *marker* is introspection, so
``recalled_params`` is called directly and the compiled ``prompt_fn.__annotations__`` is read (the
second is not redundant: ``include_extras=True`` is exactly the line a refactor drops silently). The
*injection* is a claim about the optimizer's graph, so it is asserted from ``build_graph_from_result``.
The *guards* are checked by counting model calls and backend calls, never by trusting a message: every
guard test scripts a model with zero turns, so a call that reached the runtime raises ``ScriptExhausted``
instead of the expected error.

``JSONMemoryBackend`` throughout: its ``_search`` ranks with BM25 and carries ``{"results": {entry_id:
value}}``, the mapping the whole narrow-gradient chain depends on, with no embedder and no model call.
"""

from __future__ import annotations

import os
import tempfile
import typing
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, MemoryBackend
from ai_functions.experimental.methods import (
    CompilationError,
    MethodAgent,
    MethodError,
    Recall,
    Recalled,
    RecallError,
    RecallMarkerError,
    RecallWiringError,
    ai_method,
    recalled_params,
)
from ai_functions.experimental.methods import recall as recall_module
from ai_functions.optimizer._graph import build_graph_from_result
from ai_functions.testing import RuntimeHarness
from ai_functions.types import thread_scope
from ai_functions.types.graph import ParameterView, Traceable, collect_nodes

from .methods_wire import Capturing
from .test_methods_boundary import _imports

# ── Fixtures ──
#
# Module level, all of them: ``compile_ai_method`` resolves annotations with ``typing.get_type_hints``
# against module globals, so a function-local output type or ``Recalled`` annotation cannot be resolved.


class Choice(BaseModel):
    """What the navigator returns. Named for the tool call the scripted turns make."""

    transition: str = Field(description="The transition to take next.")
    reason: str = Field(description="Why.")


class Playbook(BaseModel):
    """The memory schema: a searchable collection and a full-recall scalar.

    Four seeded entries, and the count is measured rather than chosen: ``BM25Okapi``'s IDF for a term
    in one of two documents is ``log(1.0) == 0``, so at N=2 every score is 0.0 and ``sorted`` falls back
    to insertion order. At N=4 ``"invoice disputed escalate finance"`` selects the entry it names.
    """

    guidance: list[str] = Field(
        default_factory=lambda: [
            "when the permit is pending, request the missing document before deciding",
            "when the invoice is disputed, escalate to finance rather than approving",
            "when the applicant is unknown, verify identity documents first",
            "when a deadline has passed, record the delay and continue",
        ],
        description="What has been learned about deciding, one entry per lesson.",
    )
    summary: str = Field(
        default="this actor tends to loop on states it has already visited",
        description="One accumulated description of the actor, read whole every time.",
    )


class Navigator(MethodAgent):
    """One search-mode capability and one full-mode capability on the same agent.

    ``choose``'s marked parameter comes *first*, the shape the positional-shadow guard exists for.
    ``advice`` is deliberately not named ``guidance``: the signature names the argument and the marker
    names the memory field, and nothing requires them to agree.
    """

    name = "navigator"

    @ai_method(Choice, description="Choose the next transition")
    def choose(
        self,
        advice: Annotated[list[str], Recalled("guidance", k=2)],
        state: str,
        options: str,
    ) -> Choice:
        """Guidance retrieved for this decision: {advice}

        You are in {state}. The options are {options}. Choose one.
        """
        ...

    @ai_method(Choice, description="Choose using everything known about the actor")
    def reflect(
        self,
        profile: Annotated[str, Recalled("summary")],
        state: str,
    ) -> Choice:
        """What is known about this actor: {profile}

        You are in {state}. Choose.
        """
        ...

    @ai_method(Choice, description="Choose using both a search and a full recall")
    def deliberate(
        self,
        advice: Annotated[list[str], Recalled("guidance", k=1)],
        profile: Annotated[str, Recalled("summary")],
        state: str,
    ) -> Choice:
        """Guidance: {advice}
        Profile: {profile}
        You are in {state}. Choose.
        """
        ...

    @ai_method(Choice, description="Choose, with the recalled parameter last")
    def weigh(
        self,
        state: str,
        advice: Annotated[list[str], Recalled("guidance", k=1)],
    ) -> Choice:
        """You are in {state}. Guidance: {advice}. Choose."""
        ...

    @ai_method(Choice, description="Choose with nothing recalled")
    def guess(self, state: str) -> Choice:
        """You are in {state}. Guess."""
        ...


class UnionNavigator(MethodAgent):
    """The marker on a ``Traceable[...]`` union, the shape ``detect_procedural_params`` handles."""

    name = "union-navigator"

    @ai_method(Choice, description="Choose, annotated through the handle union")
    def choose(
        self,
        advice: Traceable[Annotated[list[str], Recalled("guidance", k=2)]],
        state: str,
    ) -> Choice:
        """Guidance: {advice}. You are in {state}. Choose."""
        ...


class QueryNavigator(MethodAgent):
    """A capability whose own parameter is named ``queries``: the wiring collision."""

    name = "query-navigator"

    @ai_method(Choice, description="Choose, with a colliding parameter name")
    def choose(
        self,
        advice: Annotated[list[str], Recalled("guidance", k=2)],
        queries: str = "",
    ) -> Choice:
        """Guidance: {advice}. Queries: {queries}. Choose."""
        ...


class OverrideNavigator(MethodAgent):
    """The same collision through ``overrides``, which fails exactly as silently."""

    name = "override-navigator"

    @ai_method(Choice, description="Choose, with a colliding parameter name")
    def choose(
        self,
        advice: Annotated[list[str], Recalled("guidance", k=2)],
        overrides: str = "",
    ) -> Choice:
        """Guidance: {advice}. Overrides: {overrides}. Choose."""
        ...


class MethodNavigator(MethodAgent):
    """A parameter named ``method`` is allowed, because ``trace`` is positional-only in that slot."""

    name = "method-navigator"

    @ai_method(Choice, description="Choose, naming a parameter after trace's own first")
    def choose(
        self,
        advice: Annotated[list[str], Recalled("guidance", k=2)],
        method: str = "default",
    ) -> Choice:
        """Guidance: {advice}. Method: {method}. Choose."""
        ...


class PlainHelper(MethodAgent):
    """A marked parameter on a method that is not an ``@ai_method``."""

    name = "helper"

    def plain(self, advice: Annotated[list[str], Recalled("guidance", k=2)], state: str) -> str:
        return state


def store(schema: type[BaseModel] = Playbook) -> JSONMemoryBackend:
    """A fresh JSON-backed store in its own temp directory, so no two tests share entry ids."""
    return JSONMemoryBackend(schema, "navigator", path=os.path.join(tempfile.mkdtemp(), "pb.json"))


class SpyBackend(JSONMemoryBackend):
    """A real backend that also counts what was retrieved from it, and how.

    Counting has to happen on the *public* ``recall``/``search``, where the base class emits the
    ``ParameterRecalledEvent``; overriding the ``_``-hooks instead would count correctly and skip
    emission, and every optimizer assertion below would go quiet.
    """

    def __init__(self, schema: type[BaseModel] = Playbook) -> None:
        super().__init__(schema, "navigator", path=os.path.join(tempfile.mkdtemp(), "pb.json"))
        self.calls: list[tuple[str, str, str | None]] = []
        """One ``(kind, source, query)`` per retrieval, in order."""

    async def recall(self, name: str, *args: Any, **kwargs: Any) -> ParameterView[Any]:
        self.calls.append(("recall", name, None))
        return await super().recall(name, *args, **kwargs)

    async def search(self, name: str, query: str, k: int = 5, *args: Any, **kwargs: Any) -> ParameterView[Any]:
        self.calls.append(("search", name, query))
        return await super().search(name, query, k, *args, **kwargs)


def counting(turns: int) -> Capturing:
    """A recording model scripted with ``turns`` ``Choice`` tool calls.

    ``counting(0)`` is the load-bearing case: a guard that raises before the model call leaves
    ``contexts == []``, and a guard that raises *after* it raises ``ScriptExhausted`` instead of the
    error under test.
    """
    return Capturing(turns, tool="Choice", payload={"transition": "T0", "reason": "r"})


def _scripting(agent: MethodAgent, model: Capturing) -> Any:
    """Bind a scripted model by replacing ``compiled`` on the *instance*.

    The reason ``Recall.trace`` compiles through ``self._agent.compiled`` at trace time rather than
    caching an ``AIFunction`` at wiring time: a cached one would silently bypass this binding.
    """
    original = type(agent).compiled

    def compiled(name: str, **overrides: Any) -> Any:
        overrides.setdefault("model", model)
        return original(agent, name, **overrides)

    return compiled


# ── The marker: what is declared, and what survives compilation ──


def test_the_marker_is_found_in_both_modes_and_unmarked_params_are_ignored() -> None:
    """The marker is found in both modes and unmarked params are ignored: introspection
    is the binder's whole input, so it has to see exactly the marked ones."""
    navigator = Navigator()

    assert recalled_params(navigator.choose) == {"advice": Recalled("guidance", k=2)}
    assert recalled_params(navigator.reflect) == {"profile": Recalled("summary")}
    assert recalled_params(navigator.deliberate) == {
        "advice": Recalled("guidance", k=1),
        "profile": Recalled("summary"),
    }
    assert recalled_params(navigator.guess) == {}


def test_the_marker_is_found_through_the_handle_union() -> None:
    """The marker is found through the handle union: a reader that only checked
    ``__metadata__`` would see nothing and silently perform no retrieval."""
    assert recalled_params(UnionNavigator().choose) == {"advice": Recalled("guidance", k=2)}


def test_self_is_not_a_recalled_parameter() -> None:
    """``self`` is not a recalled parameter: a bound method's hints exclude ``self``."""
    assert "self" not in recalled_params(Navigator().choose)


def test_an_unresolvable_annotation_raises_rather_than_reading_as_unmarked() -> None:
    """An unresolvable annotation raises rather than reading as unmarked (R7): the cost of
    swallowing here is *no retrieval*, presenting as an agent that reads an empty playbook forever."""

    class Broken(MethodAgent):
        name = "broken"

        @ai_method(Choice, description="Choose")
        def choose(self, advice: NoSuchType, state: str) -> Choice:  # noqa: F821  # pyright: ignore[reportUndefinedVariable] (the typo under test)
            """Guidance: {advice}. State: {state}."""
            ...

    with pytest.raises(NameError):
        recalled_params(Broken().choose)


def test_the_marker_survives_compilation_into_the_prompt_functions_annotations() -> None:
    """The marker survives compilation into the prompt function's annotations: without
    ``include_extras=True`` the annotation flattens to ``list[str]`` and the parameter silently becomes
    an ordinary prompt argument."""
    annotation = Navigator().compiled("choose").prompt_fn.__annotations__["advice"]
    assert any(isinstance(m, Recalled) for m in typing.get_args(annotation))


async def test_the_marked_parameter_stays_in_the_tool_schema() -> None:
    """The marked parameter stays in the tool schema (R1), and the pin #8 check: the marker
    is a plain dataclass inside ``Annotated``, not a ``pydantic.Field``, so the installed strands
    decorator builds the schema instead of raising ``NotImplementedError``."""
    fn = Navigator().compiled("choose")
    schema = (await fn.load_tools())[0].tool_spec["inputSchema"]["json"]
    assert set(schema["properties"]) == {"advice", "state", "options"}
    assert "advice" in schema["required"]


def test_a_marker_that_could_retrieve_nothing_is_refused_at_construction() -> None:
    """A marker that could retrieve nothing is refused at construction: ``k=0`` would
    return an empty list and empty ``results`` meta without complaint."""
    with pytest.raises(RecallMarkerError, match=r"at least 1"):
        Recalled("guidance", k=0)
    with pytest.raises(RecallMarkerError, match=r"at least 1"):
        Recalled("guidance", k=-1)


# ── The injection: a recalled value that is actually a gradient target ──


async def test_search_mode_injects_a_view_the_optimizer_can_find() -> None:
    """Search mode injects a view the optimizer can find (R2): asserted from the graph,
    not from what ``trace`` was handed. One parameter named for the *memory* field, derivation
    ``search``, and non-empty ``meta["results"]``."""
    async with RuntimeHarness():
        memory = store()
        try:
            navigator = Navigator()
            model = counting(1)
            binder = Recall(navigator, memory)

            result = await binder.trace(
                "choose",
                state="PENDING",
                options="request|decide",
                queries={"advice": "the permit is pending and a document is missing"},
                overrides={"model": model},
            )

            assert result.value.transition == "T0"
            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["guidance"]
            parameter = graph.parameters[0]
            assert parameter.derivation == "search"
            assert parameter.requires_grad is True
            assert parameter.meta["results"], "no retrieved ids, so consolidation has no target"

            assert len(model.contexts) == 1
            assert any("permit is pending" in prompt for prompt in model.prompts(0))
        finally:
            memory.close()


async def test_the_query_selects_which_entries_reach_the_prompt() -> None:
    """The query selects which entries reach the prompt: ``k=1`` over disjoint entries is
    the cheapest honest check that the *query* is used."""
    async with RuntimeHarness():
        memory = store()
        try:
            result = await Recall(Navigator(), memory).trace(
                "deliberate",
                state="DISPUTED",
                queries={"advice": "invoice disputed escalate finance"},
                overrides={"model": counting(1)},
            )

            advice = next(view.value for view in result.inputs if getattr(view, "name", "") == "guidance")
            assert advice == ["when the invoice is disputed, escalate to finance rather than approving"]
        finally:
            memory.close()


async def test_a_second_trace_gets_a_fresh_node_because_nothing_is_cached() -> None:
    """A second trace gets a fresh node because nothing is cached (R3): a ``ParameterView``
    is emitted once, so a binder that stored one would produce a node on the first traced call and
    none on any later one."""
    async with RuntimeHarness():
        memory = store()
        try:
            binder = Recall(Navigator(), memory)
            model = counting(2)
            names: list[list[str]] = []

            for _ in range(2):
                result = await binder.trace(
                    "choose",
                    state="PENDING",
                    options="a|b",
                    queries={"advice": "permit pending"},
                    overrides={"model": model},
                )
                graph = await build_graph_from_result(result, [memory])
                names.append([p.name for p in graph.parameters])

            assert names == [["guidance"], ["guidance"]], "a reused view yields no second node"
            assert len(model.contexts) == 2
        finally:
            memory.close()


async def test_full_mode_injects_a_full_recall() -> None:
    """Full mode injects a full recall: ``k=None`` means the memory *is* the answer."""
    async with RuntimeHarness():
        memory = store()
        try:
            model = counting(1)
            result = await Recall(Navigator(), memory).trace("reflect", state="LOOPING", overrides={"model": model})

            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["summary"]
            assert graph.parameters[0].derivation == "full"
            assert graph.parameters[0].requires_grad is True
            assert any("loop on states" in prompt for prompt in model.prompts(0))
        finally:
            memory.close()


async def test_two_marked_parameters_on_one_method_each_get_their_own_retrieval() -> None:
    """Two marked parameters on one method each get their own retrieval: why ``queries``
    is a mapping rather than one string."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            result = await Recall(Navigator(), memory).trace(
                "deliberate",
                state="PENDING",
                queries={"advice": "permit pending document"},
                overrides={"model": counting(1)},
            )

            assert memory.calls == [
                ("search", "guidance", "permit pending document"),
                ("recall", "summary", None),
            ]
            graph = await build_graph_from_result(result, [memory])
            assert sorted(p.name for p in graph.parameters) == ["guidance", "summary"]
            assert {p.derivation for p in graph.parameters} == {"search", "full"}
        finally:
            memory.close()


async def test_an_unmarked_method_traces_with_no_retrieval_at_all() -> None:
    """An unmarked method traces with no retrieval at all: the binder is not a gate."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            result = await Recall(Navigator(), memory).trace("guess", state="START", overrides={"model": counting(1)})
            assert result.value.transition == "T0"
            assert memory.calls == []
            assert result.inputs == []
        finally:
            memory.close()


async def test_a_method_that_does_not_compile_is_refused_before_any_retrieval() -> None:
    """R5: compilation is checked before retrieval, so a non-``@ai_method`` costs no backend call."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            with pytest.raises(CompilationError, match=r"PlainHelper\.plain is not an @ai_method"):
                await Recall(PlainHelper(), memory).trace("plain", state="S", queries={"advice": "q"})
            assert memory.calls == []
        finally:
            memory.close()


async def test_a_marker_on_a_union_annotated_parameter_is_filled_end_to_end() -> None:
    """A marker on a union-annotated parameter is filled end to end: the union is also
    *filled*, and the view still becomes a node."""
    async with RuntimeHarness():
        memory = store()
        try:
            result = await Recall(UnionNavigator(), memory).trace(
                "choose",
                state="PENDING",
                queries={"advice": "permit pending"},
                overrides={"model": counting(1)},
            )
            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["guidance"]
        finally:
            memory.close()


# ── Explicit beats implicit ──


async def test_a_caller_supplied_value_passes_through_and_no_retrieval_happens() -> None:
    """A caller supplied value passes through and no retrieval happens (R4): the backend
    spy is the assertion that matters."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            model = counting(1)
            result = await Recall(Navigator(), memory).trace(
                "choose",
                advice=["the caller already knows what to do"],
                state="PENDING",
                options="a|b",
                overrides={"model": model},
            )

            assert memory.calls == [], "a supplied value must not trigger a retrieval"
            assert any("caller already knows" in prompt for prompt in model.prompts(0))

            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == [], "a plain value is not a parameter node"
            assert result.value.transition == "T0"
        finally:
            memory.close()


async def test_one_parameter_can_be_supplied_while_the_other_is_recalled() -> None:
    """One parameter can be supplied while the other is recalled: per parameter, not per call."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            result = await Recall(Navigator(), memory).trace(
                "deliberate",
                profile="the caller's own read of the actor",
                state="PENDING",
                queries={"advice": "permit pending"},
                overrides={"model": counting(1)},
            )

            assert memory.calls == [("search", "guidance", "permit pending")]
            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["guidance"]
        finally:
            memory.close()


# ── The guards: every one of them fires before the model call ──


async def test_a_search_mode_parameter_with_no_query_raises_before_any_model_call() -> None:
    """A search-mode parameter with no query raises before any model call (R5): a model
    with zero turns proves the refusal came first."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            model = counting(0)
            with pytest.raises(RecallWiringError) as raised:
                await Recall(Navigator(), memory).trace(
                    "choose", state="PENDING", options="a|b", overrides={"model": model}
                )

            message = str(raised.value)
            assert "'advice'" in message
            assert "'guidance'" in message, "the message must name the memory field too"
            assert "navigator_choose" in message
            assert memory.calls == [], "the guard must fire before any retrieval"
            assert model.contexts == [], "the guard must fire before any model call"
        finally:
            memory.close()


async def test_a_query_for_a_full_mode_parameter_is_refused() -> None:
    """A query for a full-mode parameter is refused: misuse fails loud."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            model = counting(0)
            with pytest.raises(RecallWiringError) as raised:
                await Recall(Navigator(), memory).trace(
                    "reflect",
                    state="LOOPING",
                    queries={"profile": "what is known"},
                    overrides={"model": model},
                )

            message = str(raised.value)
            assert "'profile'" in message
            assert "k=None" in message
            assert memory.calls == []
            assert model.contexts == []
        finally:
            memory.close()


async def test_a_query_for_an_unmarked_parameter_is_refused() -> None:
    """A query for an unmarked parameter is refused: the message names what *is* marked."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            with pytest.raises(RecallWiringError, match=r"no parameter named 'state'"):
                await Recall(Navigator(), memory).trace(
                    "choose",
                    state="PENDING",
                    options="a|b",
                    queries={"advice": "permit", "state": "not a marked parameter"},
                    overrides={"model": counting(0)},
                )
            assert memory.calls == []
        finally:
            memory.close()


async def test_a_query_for_an_explicitly_supplied_parameter_is_refused() -> None:
    """A query for an explicitly supplied parameter is refused: the two ways of filling a
    parameter are alternatives, not layers."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            with pytest.raises(RecallWiringError, match=r"supplied explicitly"):
                await Recall(Navigator(), memory).trace(
                    "choose",
                    advice=["already known"],
                    state="PENDING",
                    options="a|b",
                    queries={"advice": "permit pending"},
                    overrides={"model": counting(0)},
                )
            assert memory.calls == []
        finally:
            memory.close()


def test_a_parameter_named_queries_is_refused_at_wiring_time() -> None:
    """A parameter named ``queries`` is refused at wiring time: ``bound()`` refuses so the
    introspection path and the call path cannot disagree."""
    binder = Recall(QueryNavigator(), store())
    try:
        with pytest.raises(RecallWiringError) as raised:
            binder.bound("choose")
        message = str(raised.value)
        assert "'queries'" in message
        assert "query-navigator_choose" in message
        assert raised.value.function_name == "query-navigator_choose"
    finally:
        binder.backend.close()


async def test_the_wiring_guard_also_fires_on_the_traced_path() -> None:
    """The wiring guard also fires on the traced path: both entry points, or the guarantee
    is one call away from not applying."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            model = counting(0)
            with pytest.raises(RecallWiringError, match=r"'queries'"):
                await Recall(QueryNavigator(), memory).trace(
                    "choose", queries={"advice": "permit"}, overrides={"model": model}
                )
            assert memory.calls == []
            assert model.contexts == []
        finally:
            memory.close()


def test_a_parameter_named_overrides_is_refused_too() -> None:
    """A parameter named ``overrides`` is refused too."""
    binder = Recall(OverrideNavigator(), store())
    try:
        with pytest.raises(RecallWiringError, match=r"'overrides'"):
            binder.bound("choose")
    finally:
        binder.backend.close()


async def test_a_parameter_named_method_is_allowed_because_trace_takes_it_positionally() -> None:
    """A parameter named ``method`` is allowed because ``trace`` takes it positionally: the
    contrast that keeps ``RESERVED`` at two entries. Asserted end to end: the value *arrives*."""
    async with RuntimeHarness():
        memory = store()
        try:
            model = counting(1)
            assert Recall(MethodNavigator(), memory).bound("choose") == {"advice": Recalled("guidance", k=2)}
            await Recall(MethodNavigator(), memory).trace(
                "choose",
                method="THE-METHOD-ARGUMENT",
                queries={"advice": "permit pending"},
                overrides={"model": model},
            )
            assert any("THE-METHOD-ARGUMENT" in prompt for prompt in model.prompts(0))
        finally:
            memory.close()


async def test_positional_arguments_that_would_shift_onto_a_marked_parameter_are_refused() -> None:
    """Positional arguments that would shift onto a marked parameter are refused (R5):
    positionals bind left to right and cannot skip a slot."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            model = counting(0)
            with pytest.raises(RecallWiringError) as raised:
                await Recall(Navigator(), memory).trace(
                    "choose",
                    "PENDING",
                    "a|b",
                    queries={"advice": "permit pending"},
                    overrides={"model": model},
                )

            message = str(raised.value)
            assert "'advice'" in message
            assert "left to right" in message
            assert memory.calls == []
            assert model.contexts == []
        finally:
            memory.close()


async def test_positional_arguments_that_stop_short_of_a_marked_parameter_are_allowed() -> None:
    """Positional arguments that stop short of a marked parameter are allowed: the guard
    counts slots instead of banning ``*args``, and the retrieval still has to happen."""
    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            model = counting(1)
            result = await Recall(Navigator(), memory).trace(
                "weigh",
                "PENDING",
                queries={"advice": "permit pending missing document"},
                overrides={"model": model},
            )

            assert memory.calls == [("search", "guidance", "permit pending missing document")]
            assert any("PENDING" in prompt for prompt in model.prompts(0))
            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["guidance"]
        finally:
            memory.close()


# ── Scriptability: compiled at trace time, through the instance ──


async def test_trace_compiles_through_the_instances_own_compiled() -> None:
    """``trace`` compiles through the instance's own ``compiled`` (R3): a binder that cached an
    ``AIFunction`` in ``__init__`` would have captured the real model before the rebinding."""
    async with RuntimeHarness():
        memory = store()
        try:
            navigator = Navigator()
            model = counting(1)
            navigator.compiled = _scripting(navigator, model)  # type: ignore[method-assign]

            result = await Recall(navigator, memory).trace(
                "choose",
                state="PENDING",
                options="a|b",
                queries={"advice": "permit pending"},
            )

            assert result.value.transition == "T0"
            assert len(model.contexts) == 1, "the instance binding did not reach the traced call"
            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["guidance"]
        finally:
            memory.close()


async def test_retrieval_errors_propagate_unwrapped() -> None:
    """Retrieval errors propagate unwrapped (R7): the backend's own exception must reach
    the caller with its type and traceback intact."""

    class Broken(SpyBackend):
        async def search(self, name: str, query: str, k: int = 5, *a: Any, **kw: Any) -> Any:
            raise KeyError("the store is unreachable")

    async with RuntimeHarness():
        memory = Broken()
        try:
            model = counting(0)
            with pytest.raises(KeyError, match=r"unreachable"):
                await Recall(Navigator(), memory).trace(
                    "choose",
                    state="PENDING",
                    options="a|b",
                    queries={"advice": "permit pending"},
                    overrides={"model": model},
                )
            assert model.contexts == [], "retrieval failed, so no cycle should have started"
        finally:
            memory.close()


def test_the_binder_exposes_what_it_bound_without_letting_it_be_repointed() -> None:
    """The binder exposes what it bound without letting it be repointed."""
    navigator = Navigator()
    memory = store()
    try:
        binder = Recall(navigator, memory)
        assert binder.agent is navigator
        assert binder.backend is memory
        assert isinstance(binder.backend, MemoryBackend)
        assert "navigator" in repr(binder)
        with pytest.raises(AttributeError):
            binder.backend = store()  # type: ignore[misc]
    finally:
        memory.close()


# ── Ambient scope, duplicate markers, positional-only ──


async def test_recall_under_an_ambient_thread_scope_still_lands_the_node_on_the_traced_thread() -> None:
    """Recall under an ambient thread scope still lands the node on the traced thread (R6):
    without the ``no_thread_scope`` wrap the recall event sits on the ambient thread's log and the
    traced graph is empty."""
    async with RuntimeHarness() as h:
        memory = store()
        try:
            navigator = Navigator()
            host = await navigator.spawn("reflect", h.coordinator, model=counting(1))
            with thread_scope(h.coordinator, host.id):
                result = await Recall(navigator, memory).trace(
                    "choose",
                    state="PENDING",
                    options="a|b",
                    queries={"advice": "permit pending"},
                    overrides={"model": counting(1)},
                )
            graph = await build_graph_from_result(result, [memory])
            assert [p.name for p in graph.parameters] == ["guidance"], (
                "the recall event leaked onto the ambient thread instead of the traced one"
            )
            assert graph.parameters[0].meta["results"]
            await host.retire()
        finally:
            memory.close()


def test_two_distinct_markers_on_one_annotation_are_refused() -> None:
    """Two distinct markers on one annotation are refused (R7): which store a parameter
    reads from must not depend on annotation order. The identical marker twice is fine."""

    def conflicted(
        advice: Annotated[list[str], Recalled("guidance", k=2), Recalled("summary")],
    ) -> None: ...

    with pytest.raises(RecallMarkerError, match=r"2 distinct `Recalled` markers"):
        recalled_params(conflicted)

    def repeated(
        advice: Annotated[list[str], Recalled("guidance", k=2), Recalled("guidance", k=2)],
    ) -> None: ...

    assert recalled_params(repeated) == {"advice": Recalled("guidance", k=2)}


async def test_a_positional_only_marked_parameter_is_refused_before_any_retrieval() -> None:
    """A positional-only marked parameter is refused before any retrieval (R5): the binder
    injects by keyword, so a positional-only marked slot is unfillable."""

    class Cornered(MethodAgent):
        name = "cornered"

        @ai_method(Choice, description="A marked parameter locked behind the slash")
        def choose(
            self,
            advice: Annotated[list[str], Recalled("guidance", k=2)],
            /,
            state: str = "S",
        ) -> Choice:
            """Guidance: {advice}. State: {state}. Choose."""
            ...

    async with RuntimeHarness():
        memory = SpyBackend()
        try:
            with pytest.raises(RecallWiringError, match=r"positional-only"):
                await Recall(Cornered(), memory).trace(
                    "choose",
                    queries={"advice": "anything"},
                    overrides={"model": counting(0)},
                )
            assert memory.calls == [], "the refusal must precede the retrieval it protects"
        finally:
            memory.close()


# ── Added by this port: R2 at the trace boundary itself, the error family, the import boundary ──


class _Boundary:
    """Records what the traced call carried: ``collect_nodes`` over exactly ``(args, kwargs)``."""

    def __init__(self) -> None:
        self.args: tuple[Any, ...] = ()
        self.kwargs: dict[str, Any] = {}
        self.nodes: list[Any] = []


def _observing(agent: MethodAgent, model: Capturing, boundary: _Boundary) -> Any:
    """Like ``_scripting``, but the compiled function's ``trace`` records its own arguments first.

    The scan is the same one ``AIFunction.trace`` runs (``collect_nodes((args, kwargs))``), applied to
    the arguments the binder actually passed, before the runtime unwraps them.
    """
    original = type(agent).compiled

    def compiled(name: str, **overrides: Any) -> Any:
        overrides.setdefault("model", model)
        fn = original(agent, name, **overrides)
        real_trace = fn.trace

        async def trace(*args: Any, **kwargs: Any) -> Any:
            boundary.args, boundary.kwargs = args, dict(kwargs)
            boundary.nodes = collect_nodes((args, kwargs))
            return await real_trace(*args, **kwargs)

        fn.trace = trace  # type: ignore[method-assign]
        return fn

    return compiled


def _views_on(obj: object) -> list[ParameterView[Any]]:
    """Every ``ParameterView`` reachable from an object's own attributes (``__dict__`` and ``__slots__``)."""
    state: dict[str, Any] = dict(getattr(obj, "__dict__", {}))
    for slot in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, slot):
            state[slot] = getattr(obj, slot)
    return [n for n in collect_nodes(state) if isinstance(n, ParameterView)]


async def test_the_gradient_target_reaches_the_trace_as_a_call_argument() -> None:
    """R2, asserted at the boundary the invariant names: ``collect_nodes`` over the traced call's
    ``(args, kwargs)`` finds exactly the recalled ``ParameterView``, under the parameter's name, as a
    keyword argument, and that same object is what ``Result.inputs`` carries."""
    async with RuntimeHarness():
        memory = store()
        try:
            navigator = Navigator()
            boundary = _Boundary()
            navigator.compiled = _observing(navigator, counting(1), boundary)  # type: ignore[method-assign]

            result = await Recall(navigator, memory).trace(
                "choose", state="PENDING", options="a|b", queries={"advice": "permit pending"}
            )

            views = [n for n in boundary.nodes if isinstance(n, ParameterView)]
            assert len(views) == 1, "exactly one recalled view must reach the trace as an argument"
            assert views[0].name == "guidance"
            assert boundary.kwargs["advice"] is views[0], "the view is injected by keyword, whole"
            assert boundary.args == ()
            assert result.inputs == [views[0]], "the runtime saw the same object the binder passed"
        finally:
            memory.close()


async def test_nothing_on_self_is_a_parameter_view_after_a_trace() -> None:
    """R2, the other half: after a traced call neither the binder nor the agent holds a ``ParameterView``
    anywhere in its own state. State on ``self`` is invisible to ``collect_nodes((args, kwargs))``, so a
    view parked there would be a gradient target that never reaches the optimizer."""
    async with RuntimeHarness():
        memory = store()
        try:
            navigator = Navigator()
            binder = Recall(navigator, memory)
            await binder.trace(
                "deliberate", state="PENDING", queries={"advice": "permit pending"}, overrides={"model": counting(1)}
            )
            assert _views_on(binder) == []
            assert _views_on(navigator) == []
            assert Recall.__slots__ == ("_agent", "_backend"), "the binder holds nothing that resembles state"
        finally:
            memory.close()


def test_recall_errors_are_method_errors() -> None:
    """The recall binder's refusals are ``MethodError`` leaves, so a caller catching the package base sees
    them; the two leaves stay distinct so a marker bug and a wiring bug are told apart."""
    assert issubclass(RecallError, MethodError)
    assert issubclass(RecallMarkerError, RecallError)
    assert issubclass(RecallWiringError, RecallError)
    assert not issubclass(RecallMarkerError, RecallWiringError)
    with pytest.raises(MethodError):
        Recalled("guidance", k=0)


def test_recall_imports_only_methods_core_and_the_ai_functions_core() -> None:
    """Static boundary for this module alone: no sibling experimental package, and within
    ai_functions only the memory and types surfaces plus ``.core``. Parsed with ``ast`` so a
    ``TYPE_CHECKING`` or lazy import fires the guard too."""
    allowed_ai_functions = {
        "ai_functions.memory",
        "ai_functions.types",
        "ai_functions.types.graph",
        "ai_functions.experimental.methods.core",
    }
    names = _imports(Path(recall_module.__file__))
    offenders = [n for n in names if n.startswith("ai_functions") and n not in allowed_ai_functions]
    assert offenders == [], offenders
