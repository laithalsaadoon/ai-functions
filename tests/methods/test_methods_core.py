"""Offline tests for ``@ai_method``: typed contracts, templating, composition, threads.

Tests over the decorator's own surface, plus the invariant guards: M2 (markers survive
``include_extras``), the ``CompilationError`` / ``RenderError`` / ``RetiredThreadError`` leaves, and
the ``name=`` rename. Every claim the decorator paradigm makes is checked against the schema the model
would actually see, not against our own bookkeeping; thread-lifecycle claims are asserted from what
the *model* received or from the coordinator's own event log.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import re
import sys
import tempfile
import threading
import typing
from typing import Any

import cloudpickle  # pyright: ignore[reportMissingTypeStubs]
import pytest
from tstr import Interpolation, Template

from ai_functions import JSONMemoryBackend, Procedural, ai_function
from ai_functions.experimental import methods as methods_pkg
from ai_functions.experimental.methods import (
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
from ai_functions.experimental.methods import core as methods_core
from ai_functions.experimental.methods import recall as methods_recall
from ai_functions.memory.procedural import ProceduralMarker
from ai_functions.optimizer._graph import build_graph_from_result
from ai_functions.testing import RuntimeHarness
from ai_functions.types import InputShape
from ai_functions.types.graph import ParameterView, collect_nodes

from .methods_wire import Answer, Capturing, Caseworker, MixedWorker, Solver, Tips, logged, scripted

if typing.TYPE_CHECKING:
    from collections import OrderedDict

# ── Fixtures this file adds beyond the shared ones ──


class Coder(MethodAgent):
    """A capability with a ``Procedural`` parameter: the M2 fixture."""

    name = "coder"

    @ai_method(Answer, description="Run reusable code against a question", code_execution_mode="local")
    def apply(self, recipe: Procedural, question: str) -> Answer:
        """Use {recipe} to answer {question}."""
        ...


class Renamed(MethodAgent):
    name = "renamed"

    @ai_method(Answer, name="published_alias", description="A capability published under its own name")
    def internal(self, q: str) -> Answer:
        """Answer {q}."""
        ...


class Silent(MethodAgent):
    name = "silent"

    @ai_method(Answer, description="No docstring and a None body")
    def mute(self, q: str) -> Answer:
        return None  # type: ignore[return-value]


class Computing(MethodAgent):
    name = "computing"

    @ai_method(Answer, description="Computes its prompt in the body")
    def direct(self, q: str) -> Any:
        """Never rendered: {q}."""
        return f"computed for {q}"


class Noted(MethodAgent):
    """The same prompt three ways: a docstring, a returned ``Template``, and an unsupported return."""

    name = "noted"

    def __init__(self, header: str) -> None:
        self.header = header

    @ai_method(Answer, description="A docstring prompt with a multi-line value")
    def from_docstring(self, notes: str) -> Answer:
        """{self.header}:
            {notes}
        Done."""
        ...

    @ai_method(Answer, description="The same prompt, returned as a t-string")
    def from_template(self, notes: str) -> Any:
        # A t-string on Python 3.14+; spelled with the constructor so the module parses on 3.12.
        return Template(Interpolation(self.header, "self.header"), ":\n    ", Interpolation(notes, "notes"), "\nDone.")

    @ai_method(Answer, description="Returns something that is not a prompt")
    def from_number(self, notes: str) -> Any:
        return 42


class KeywordOnly(MethodAgent):
    name = "kwonly"

    @ai_method(Answer, description="Answer a question passed by keyword")
    def ask(self, *, question: str) -> Answer:
        """Answer {question}."""
        ...


class TypeCheckingReturn(MethodAgent):
    """A return annotation that names a ``TYPE_CHECKING``-only import; ``@ai_method`` names the output type."""

    name = "tcreturn"

    @ai_method(Answer, description="Answer a question")
    def ask(self, question: str) -> OrderedDict[str, str]:
        """Answer {question}."""
        ...


_UNPICKLABLE = threading.Lock()
"""A module global cloudpickle cannot serialize, beside an agent that must still ship to a remote worker."""


class Shippable(MethodAgent):
    name = "shippable"

    @ai_method(Answer, description="Answer a question")
    def ask(self, question: str) -> Answer:
        """Answer {question}."""
        ...


@ai_function[str]
def _plain(header: str, notes: str) -> None:
    """{header}:
        {notes}
    Done."""


# ══════════════════════════════════════════════════════════════════
# The typed contract (M1)
# ══════════════════════════════════════════════════════════════════


class TestTypedContract:
    def test_typed_parameters_survive_compilation(self) -> None:
        """Typed parameters survive compilation: the model sees the real signature, not ``request: str``."""
        fn = Solver("terse").compiled("solve")
        assert fn.input_shape is InputShape.STRUCTURED

    async def test_tool_schema_carries_types_defaults_and_enums(self) -> None:
        """The tool schema carries types, defaults and enums: ``AIFunction`` is a ToolProvider; this is what a
        calling agent chooses from."""
        fn = Solver("terse").compiled("solve")
        schema = (await fn.load_tools())[0].tool_spec["inputSchema"]["json"]
        assert schema["properties"]["mode"]["enum"] == ["fast", "thorough"]
        assert schema["properties"]["depth"]["default"] == 3
        assert schema["required"] == ["problem"]

    async def test_self_never_reaches_the_schema(self) -> None:
        """M1: a bound method drops ``self``, so the model is never asked
        to supply it. Asserted on the schema AND on the compiled signature: the installed strands decorator
        (strands/tools/decorator.py:440) silently skips parameters named ``self``/``cls``/``agent``, so a leaked
        ``self`` never shows in the schema, but every other consumer of ``inspect.signature(prompt_fn)`` would see
        it: ``_infer_input_shape``, and a teams ``FunctionMember`` picking the first positional parameter to brief."""
        fn = Solver("terse").compiled("solve")
        schema = (await fn.load_tools())[0].tool_spec["inputSchema"]["json"]
        assert set(schema["properties"]) == {"problem", "mode", "depth"}
        parameters = inspect.signature(fn.prompt_fn).parameters
        assert "self" not in parameters
        assert list(parameters) == ["problem", "mode", "depth"]

    async def test_each_capability_gets_its_own_tool_name(self) -> None:
        """M3: two ``@ai_method`` s on one agent must not collide
        when handed to a caller."""
        names = {(await fn.load_tools())[0].tool_name for fn in Solver("terse").agents()}
        assert names == {"solver-terse_solve", "solver-terse_learn", "solver-terse_inherited"}

    def test_annotated_markers_survive_compilation(self) -> None:
        """M2: the compiled ``prompt_fn`` keeps ``Annotated`` metadata, so the runtime's procedural detection still
        sees ``ProceduralMarker`` on ``recipe`` (a plain ``str`` would silently become a prompt argument)."""
        fn = Coder().compiled("apply")
        hints = typing.get_type_hints(fn.prompt_fn, include_extras=True)
        assert any(isinstance(m, ProceduralMarker) for m in getattr(hints["recipe"], "__metadata__", ()))
        assert hints["question"] is str
        from ai_functions.ai_thread.code_execution import detect_procedural_params

        assert detect_procedural_params(fn.prompt_fn) == {"recipe"}

    async def test_default_tool_names_satisfy_the_provider_tool_name_pattern(self) -> None:
        """M3: a default name joins owner and method with ``_``, because Bedrock's Converse API and Strands'
        tool-use check both require ``[a-zA-Z0-9_-]+`` and reject a dotted name once the tool reaches a model."""
        tool_name = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
        for agent in (Solver("terse"), Silent(), Computing()):
            for fn in agent.agents():
                name = (await fn.load_tools())[0].tool_name
                assert tool_name.match(name), name

    def test_an_unnamed_agent_publishes_under_its_lowercased_class_name(self) -> None:
        """M3: with no ``name`` set, the owner half is the lowercased class name, so two unnamed agent classes
        publish distinct tools."""

        class Navigator(MethodAgent):
            @ai_method(Answer, description="Pick a heading")
            def choose(self, problem: str) -> Answer:
                """Pick a heading for {problem}."""
                ...

        assert Navigator().compiled("choose").config.name == "navigator_choose"

    def test_decorator_rename_wins_for_the_tool_name(self) -> None:
        """M3: ``@ai_method(..., name=...)`` names the published tool, not ``{owner}_{method}``."""
        fn = Renamed().compiled("internal")
        assert fn.config.name == "published_alias"
        assert Solver("terse").compiled("solve").config.name == "solver-terse_solve"

    def test_the_spec_is_stamped_and_the_method_stays_a_function(self) -> None:
        """``ai_method`` records an ``AIMethodSpec`` and returns the method unchanged, so MRO walks see a function."""
        spec = Solver.solve.__ai_method__  # type: ignore[attr-defined]
        assert isinstance(spec, AIMethodSpec)
        assert spec.output_type is Answer
        assert spec.config == {"description": "Solve a problem in this solver's style"}
        assert is_ai_method(Solver.solve) and not is_ai_method(Solver.__init__)
        assert callable(Solver.solve)

    def test_compiling_a_non_ai_method_is_refused_by_name(self) -> None:
        """A plain method is not a capability; the refusal names the tool the caller would have looked for."""
        with pytest.raises(CompilationError, match=r"Solver\.__init__ is not an @ai_method") as info:
            compile_ai_method(Solver("terse"), "__init__")
        assert info.value.function_name == "solver-terse___init__"
        assert isinstance(info.value, MethodError)

    async def test_a_keyword_only_method_compiles_with_its_parameters(self) -> None:
        """A method whose parameters are all keyword-only compiles and keeps them in its tool schema."""
        fn = KeywordOnly().compiled("ask")
        assert list(inspect.signature(fn.prompt_fn).parameters) == ["question"]
        schema = (await fn.load_tools())[0].tool_spec["inputSchema"]["json"]
        assert schema["required"] == ["question"]
        assert await fn.render_prompt(question="why") == "Answer why."

    async def test_a_type_checking_only_return_annotation_does_not_block_compilation(self) -> None:
        """Only parameter annotations are resolved; the output type comes from ``@ai_method``."""
        fn = TypeCheckingReturn().compiled("ask")
        assert fn.output_type is Answer
        assert await fn.render_prompt(question="why") == "Answer why."


# ══════════════════════════════════════════════════════════════════
# The docstring as prompt template (M4)
# ══════════════════════════════════════════════════════════════════


class TestTemplate:
    async def test_docstring_interpolates_arguments_and_instance_state(self) -> None:
        """M4: ``{self.style}`` and ``{mode}`` in one
        template."""
        prompt = await Solver("terse").compiled("solve").render_prompt("halting", mode="thorough")
        assert prompt == "Solve halting in the terse style, thorough, to depth 3."

    async def test_two_instances_render_two_prompts_from_one_template(self) -> None:
        """Two instances render two prompts from one template."""
        a = await Solver("terse").compiled("solve").render_prompt("x")
        b = await Solver("verbose").compiled("solve").render_prompt("x")
        assert "terse" in a and "verbose" in b and a != b

    async def test_a_non_none_body_is_the_prompt_verbatim(self) -> None:
        """M4: a method body that returns a string skips templating entirely."""
        assert await Computing().compiled("direct").render_prompt("q1") == "computed for q1"

    async def test_no_docstring_and_none_is_refused_at_render(self) -> None:
        """M4: neither a body nor a docstring is never rendered blank."""
        with pytest.raises(RenderError, match=r"Silent\.mute has no docstring") as info:
            await Silent().compiled("mute").render_prompt("q")
        assert info.value.function_name == "Silent.mute"

    def test_ai_methods_are_found_across_the_mro(self) -> None:
        """``@ai_method`` s are found across the MRO."""
        assert set(Solver("terse").ai_methods()) == {"solve", "learn", "inherited"}


# ══════════════════════════════════════════════════════════════════
# Composition: an agent as a typed tool, invoked as a function
# ══════════════════════════════════════════════════════════════════


class TestComposition:
    async def test_agent_is_callable_with_typed_arguments(self) -> None:
        """An agent is callable with typed arguments."""
        async with RuntimeHarness():
            fn = Solver("terse").compiled("solve", model=scripted())
            result = await fn("halting problem", mode="thorough", depth=1)
            assert isinstance(result, Answer)

    async def test_one_agent_holds_another_as_a_tool(self) -> None:
        """One agent holds another as a tool: composition is Python typing, not a message bus."""
        async with RuntimeHarness():
            helper = Solver("terse").compiled("solve", model=scripted())
            lead = Solver("lead").compiled("learn", model=scripted(), tools=[helper])
            assert helper in lead.config.tools

    def test_compiled_routes_strands_agent_kwargs_into_agent_kwargs(self) -> None:
        """``compiled(name, hooks=[...])`` reaches ``agent_kwargs["hooks"]`` exactly as ``AIFunction.replace`` does,
        so a ``HookProvider`` (the teams approvals hook, say) attaches to a method the same way it attaches to a
        function; a ``ThreadConfig`` field beside it still lands on the config, and an explicit ``agent_kwargs=``
        merges rather than being shadowed."""
        from strands.hooks import HookProvider, HookRegistry

        class Marker(HookProvider):
            def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
                del registry, kwargs

        marker = Marker()
        fn = Solver("terse").compiled("solve", model=scripted(), hooks=[marker], agent_kwargs={"agent_id": "s1"})
        assert fn.config.agent_kwargs.get("hooks") == [marker]
        assert fn.config.agent_kwargs.get("agent_id") == "s1"
        assert "hooks" not in {f.name for f in dataclasses.fields(fn.config)}
        assert fn.replace(hooks=[marker, marker]).config.agent_kwargs.get("hooks") == [marker, marker]


# ══════════════════════════════════════════════════════════════════
# Learnable parameters: the gap the str-only shape could not close
# ══════════════════════════════════════════════════════════════════


class TestLearnable:
    async def test_recalled_memory_is_a_gradient_target(self) -> None:
        """Recalled memory is a gradient target: ``collect_nodes`` reads call arguments, so guidance must
        arrive as one."""
        path = os.path.join(tempfile.mkdtemp(), "mem.json")
        async with RuntimeHarness():
            memory = JSONMemoryBackend(Tips, "t", path=path)
            try:
                learned = await memory.recall("guidance")
                assert isinstance(learned, ParameterView)
                assert len(collect_nodes(((learned, "problem"), {}))) == 1
            finally:
                memory.close()

    async def test_traced_call_yields_an_optimizable_parameter_node(self) -> None:
        """Traced call yields an optimizable parameter node: a real ``ParameterNode`` a TextGrad step can
        update."""
        path = os.path.join(tempfile.mkdtemp(), "mem.json")
        async with RuntimeHarness():
            memory = JSONMemoryBackend(Tips, "t", path=path)
            try:
                learned = await memory.recall("guidance")
                fn = Solver("terse").compiled("learn", model=scripted())
                result = await fn.trace(learned, "solve x")
                graph = await build_graph_from_result(result, [memory])
                assert [p.name for p in graph.parameters] == ["guidance"]
                assert graph.parameters[0].requires_grad
            finally:
                memory.close()


# ══════════════════════════════════════════════════════════════════
# The thread lifecycle: a capability that keeps its context between calls
# ══════════════════════════════════════════════════════════════════


class TestLifecycle:
    async def test_spawned_method_runs_twice_and_the_second_call_sees_the_first(self) -> None:
        """A spawned method runs twice and the second call sees the first: two calls, one conversation. Both
        calls carry the method's real typed keywords, and the second call's model context contains the first call's
        prompt."""
        async with RuntimeHarness() as h:
            model = Capturing(2)
            thread = await Solver("terse").spawn("solve", h.coordinator, model=model)
            assert isinstance(thread, MethodThread)

            first = await thread.run(problem="halting", mode="thorough", depth=1)
            second = await thread.run(problem="collatz")

            assert isinstance(first, Answer) and isinstance(second, Answer)
            assert (first.text, second.text) == ("a0", "a1")

            assert len(model.contexts) == 2
            assert not any("collatz" in p for p in model.prompts(0))
            assert any("halting" in p for p in model.prompts(1))
            assert any("collatz" in p for p in model.prompts(1))
            assert len(model.contexts[1]) > len(model.contexts[0])

    async def test_the_live_thread_keeps_the_typed_signature(self) -> None:
        """The live thread keeps the typed signature: a live thread is not a chat box; ``run`` binds the
        method's own parameters."""
        async with RuntimeHarness() as h:
            thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(1))
            assert thread.qualified_name == "solver-terse_solve"
            with pytest.raises(TypeError):
                await thread.run(request="one string, the shape we gave up")

    async def test_fork_shares_history_to_the_fork_point_then_diverges(self) -> None:
        """A fork shares history to the fork point, then diverges: one context, two continuations, neither
        contaminating the other."""
        async with RuntimeHarness() as h:
            thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(3))
            await thread.run(problem="shared")

            forked = await thread.fork()
            assert forked.id != thread.id
            assert forked.qualified_name == thread.qualified_name
            assert logged(await h.events(forked.id)) == logged(await h.events(thread.id))

            await forked.run(problem="fork-only")
            await thread.run(problem="parent-only")

            fork_log = " ".join(logged(await h.events(forked.id)))
            parent_log = " ".join(logged(await h.events(thread.id)))
            assert "shared" in fork_log and "shared" in parent_log
            assert "fork-only" in fork_log and "fork-only" not in parent_log
            assert "parent-only" in parent_log and "parent-only" not in fork_log

    async def test_a_sibling_method_can_be_seeded_from_another_thread(self) -> None:
        """A sibling method can be seeded from another thread: a thread hosts one signature, so
        ``determine`` inherits ``verify``'s context by copying its log at spawn; its *first* model call already
        contains ``verify``'s turn."""
        async with RuntimeHarness() as h:
            worker = Caseworker()
            verify = await worker.spawn("verify", h.coordinator, model=Capturing(1))
            await verify.run(claim="the permit expired")

            determined = Capturing(1)
            determine = await worker.spawn("determine", h.coordinator, seed_from=verify.id, model=determined)
            assert determine.id != verify.id
            assert determine.qualified_name == "caseworker_determine"

            await determine.run(question="is the applicant eligible")

            first_context = determined.prompts(0)
            assert any("the permit expired" in p for p in first_context)
            assert any("is the applicant eligible" in p for p in first_context)

    async def test_notify_reaches_the_next_cycle_without_starting_one(self) -> None:
        """``notify`` reaches the next cycle without starting one: the inbound side channel a STRUCTURED thread
        cannot get from ``send_message``."""
        async with RuntimeHarness() as h:
            model = Capturing(1)
            thread = await Solver("terse").spawn("solve", h.coordinator, model=model)

            await thread.notify("the deadline moved to Friday")
            assert model.contexts == []  # no cycle started

            await thread.run(problem="reschedule")
            assert any("the deadline moved to Friday" in p for p in model.prompts(0))

    async def test_a_retired_thread_refuses_every_operation_by_name(self) -> None:
        """M6: no silent respawn; each op raises
        ``RetiredThreadError`` naming the capability, never a bare ``RuntimeError``."""
        async with RuntimeHarness() as h:
            thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(2))
            await thread.run(problem="first")

            await thread.retire()
            assert thread.live is False
            assert repr(thread).endswith("retired>")

            for op in (
                lambda: thread.run(problem="second"),
                lambda: thread.fork(),
                lambda: thread.notify("hello"),
            ):
                with pytest.raises(RetiredThreadError, match=r"solver-terse_solve") as info:
                    await op()
                assert info.value.function_name == "solver-terse_solve"
            with pytest.raises(RetiredThreadError, match=r"solver-terse_solve"):
                _ = thread.handle

    async def test_retire_is_idempotent(self) -> None:
        """M6: an unwind loop should not have to track which threads it released."""
        async with RuntimeHarness() as h:
            thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(1))
            await thread.retire()
            await thread.retire()
            assert thread.live is False

    async def test_retire_survives_the_runtime_tearing_the_thread_down_first(self) -> None:
        """M6: idempotence must hold against the
        runtime, not just this object. ``ThreadNotFoundError`` is a ``KeyError``; without the catch an unwind loop
        would crash mid-unwind."""
        async with RuntimeHarness() as h:
            thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(1))
            await thread.handle.terminate_now()

            await thread.retire()  # must not raise
            assert thread.live is False

    async def test_seed_from_bridges_methods_with_different_output_types(self) -> None:
        """``seed_from`` bridges methods with different output types: the handoff history carries ``toolUse``
        blocks for a tool the second method never offers; reconstruction must still deliver the first method's turns."""
        async with RuntimeHarness() as h:
            worker = MixedWorker()
            verify_model = Capturing(1)
            verify = await worker.spawn("verify", h.coordinator, model=verify_model)
            await verify.run(claim="the invoice was paid")

            judge_model = Capturing(1, tool="Judged", payload={"ok": True})
            judge = await worker.spawn("judge", h.coordinator, seed_from=verify.id, model=judge_model)
            verdict = await judge.run(question="approve the refund?")

            assert verdict.ok is True
            assert any("the invoice was paid" in p for p in judge_model.prompts(0))

    async def test_one_agent_holds_several_live_method_threads_at_once(self) -> None:
        """M7: why the lifecycle is a separate object
        and nothing caches on the agent."""
        async with RuntimeHarness() as h:
            worker = Caseworker()
            verify = await worker.spawn("verify", h.coordinator, model=Capturing(1))
            determine = await worker.spawn("determine", h.coordinator, model=Capturing(1))

            assert verify.id != determine.id
            assert verify.live and determine.live

            await verify.retire()
            assert determine.live is True
            assert await determine.run(question="still answering") is not None

    async def test_spawn_compiles_through_the_instances_own_compiled(self) -> None:
        """``spawn`` compiles through the instance's own ``compiled``: scripting a model by rebinding ``compiled`` on
        the *instance* must reach spawned threads too."""
        async with RuntimeHarness() as h:
            agent = Solver("terse")
            model = Capturing(1)
            original = type(agent).compiled
            seen: list[str] = []

            def compiled(name: str, **overrides: Any) -> Any:
                seen.append(name)
                overrides.setdefault("model", model)
                return original(agent, name, **overrides)

            agent.compiled = compiled  # type: ignore[method-assign]

            thread = await agent.spawn("solve", h.coordinator)
            result = await thread.run(problem="scripted offline")

            assert seen == ["solve"]
            assert isinstance(result, Answer)
            assert any("scripted offline" in p for p in model.prompts(0))

    async def test_the_thread_is_named_for_the_capability_not_the_agent(self) -> None:
        """M3: event logs and tool schemas use one
        vocabulary, ``{owner}_{method}``."""
        async with RuntimeHarness() as h:
            worker = Caseworker()
            thread = await worker.spawn("verify", h.coordinator, model=Capturing(1))
            info = await h.coordinator.get_thread_info(thread.id)
            assert info.thread_name == "caseworker_verify"

            named = await worker.spawn("verify", h.coordinator, thread_name="second-opinion", model=Capturing(1))
            assert (await h.coordinator.get_thread_info(named.id)).thread_name == "second-opinion"

    async def test_a_renamed_capability_is_published_and_spawned_under_its_alias(self) -> None:
        """M3: the decorator's ``name=`` reaches the thread name and ``qualified_name`` too, so errors name a thread the
        caller can find in the tool schema."""
        async with RuntimeHarness() as h:
            thread = await Renamed().spawn("internal", h.coordinator, model=Capturing(1))
            assert thread.qualified_name == "published_alias"
            assert (await h.coordinator.get_thread_info(thread.id)).thread_name == "published_alias"
            assert (await Renamed().compiled("internal").load_tools())[0].tool_name == "published_alias"

    async def test_spawn_lands_in_the_callers_registry_with_a_parent_edge(self) -> None:
        """M5: the thread is registered on the coordinator that was passed, under the given ``parent_id``."""
        async with RuntimeHarness() as h:
            parent = await Solver("lead").spawn("solve", h.coordinator, model=Capturing(1))
            child = await Solver("terse").spawn("solve", h.coordinator, parent_id=parent.id, model=Capturing(1))
            info = await h.coordinator.get_thread_info(child.id)
            assert info.parent_id == parent.id
            assert child.id in {t.thread_id for t in await h.coordinator.list_threads()}


# ══════════════════════════════════════════════════════════════════
# M8: the model a capability runs on
# ══════════════════════════════════════════════════════════════════


class Pinned(MethodAgent):
    """An agent whose class pins its whole surface to one model."""

    name = "pinned"
    model = "global.anthropic.claude-opus-5"

    @ai_method(Answer, description="Answer on the pinned model")
    def answer(self, q: str) -> Answer:  # pyright: ignore[reportReturnType]
        """Answer {q}."""

    @ai_method(Answer, description="Answer on the decorator's model", model="global.anthropic.claude-sonnet-5")
    def quick(self, q: str) -> Answer:  # pyright: ignore[reportReturnType]
        """Answer {q} quickly."""


class TestDefaultModel:
    """M8: the model is the call's, the decorator's, then the agent's; with none set, the Strands default."""

    def test_a_bare_compile_leaves_the_model_to_strands(self) -> None:
        """No call, decorator or agent model: the config carries none, so Strands picks its default."""
        assert Solver("terse").compiled("solve").config.model is None
        assert Solver("lead").compiled("learn").config.model is None
        assert all(tool.config.model is None for tool in Solver("x").agents())

    def test_the_agent_attribute_wins_over_the_strands_default(self) -> None:
        assert Pinned().compiled("answer").config.model == "global.anthropic.claude-opus-5"
        instance = Solver("terse")
        instance.model = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert instance.compiled("solve").config.model == "global.anthropic.claude-haiku-4-5-20251001-v1:0"

    def test_the_decorator_wins_over_the_agent_attribute(self) -> None:
        assert Pinned().compiled("quick").config.model == "global.anthropic.claude-sonnet-5"

    def test_the_call_wins_over_everything(self) -> None:
        model = scripted()
        assert Pinned().compiled("quick", model=model).config.model is model
        assert Solver("terse").compiled("solve", model=model).config.model is model
        assert all(tool.config.model is model for tool in Pinned().agents(model=model))

    async def test_spawn_resolves_the_model_the_same_way(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``spawn`` goes through ``compiled``, so the thread's config carries the M8 model."""
        seen: dict[str, Any] = {}
        real = methods_core.compile_ai_method

        def spy(instance: object, name: str, **overrides: Any) -> Any:
            seen.update(overrides)
            return real(instance, name, **overrides)

        monkeypatch.setattr(methods_core, "compile_ai_method", spy)
        async with RuntimeHarness() as h:
            thread = await Pinned().spawn("answer", h.coordinator)
            try:
                assert seen["model"] == "global.anthropic.claude-opus-5"
            finally:
                await thread.retire()

    def test_a_non_ai_method_still_fails_by_name(self) -> None:
        with pytest.raises(CompilationError, match="not an @ai_method"):
            Solver("terse").compiled("__init__")

    def test_importing_the_package_builds_no_bedrock_client(self) -> None:
        """The package surface stays boto3-free at import: no module-level ``strands.models`` import."""
        source = inspect.getsource(methods_core)
        module_level = [
            line for line in source.splitlines() if line.startswith(("from strands.models", "import strands"))
        ]
        assert module_level == [], module_level


# ══════════════════════════════════════════════════════════════════
# Package surface
# ══════════════════════════════════════════════════════════════════


def test_all_is_sorted_and_names_the_public_surface() -> None:
    """``__all__`` lives only in the package ``__init__`` and is sorted ASCII (uppercase first).

    Every exported name is resolved back to the submodule that defines it (``core`` here, and any
    later module of the package such as ``gated``), so the check holds as the package grows without
    naming its modules.
    """
    assert list(methods_pkg.__all__) == sorted(methods_pkg.__all__)
    assert not hasattr(methods_core, "__all__")
    assert not hasattr(methods_recall, "__all__")
    submodules = [
        module
        for module_name, module in sys.modules.items()
        if module_name.startswith(methods_pkg.__name__ + ".") and module is not None
    ]
    for name in methods_pkg.__all__:
        exported = getattr(methods_pkg, name)
        # A class or function names its module. A constant or a ``Literal`` alias does not, so its
        # defining submodule is the one that holds it by identity, and it must be exactly one.
        module_name = getattr(exported, "__module__", None) if not isinstance(exported, str) else None
        if isinstance(module_name, str) and module_name.startswith(methods_pkg.__name__ + "."):
            defining = sys.modules[module_name]
        else:
            owners = [module for module in submodules if vars(module).get(name, _ABSENT) is exported]
            assert len(owners) == 1, (name, [module.__name__ for module in owners])
            defining = owners[0]
        assert not hasattr(defining, "__all__"), defining.__name__
        assert getattr(defining, name) is exported


_ABSENT = object()


# ══════════════════════════════════════════════════════════════════
# Rendering, naming and shipping match a plain ``@ai_function``
# ══════════════════════════════════════════════════════════════════


async def test_a_docstring_and_a_returned_template_render_as_an_ai_function_docstring() -> None:
    """M4: multi-line values indent to their placeholder's column, exactly as ``@ai_function`` renders them."""
    expected = await _plain.render_prompt(header="Notes", notes="a\nb")
    assert expected == "Notes:\n    a\n    b\nDone."
    noted = Noted("Notes")
    assert await noted.compiled("from_docstring").render_prompt(notes="a\nb") == expected
    assert await noted.compiled("from_template").render_prompt(notes="a\nb") == expected


async def test_a_body_returning_neither_str_nor_template_is_refused() -> None:
    from ai_functions.ai_thread import AIFunctionError

    with pytest.raises(AIFunctionError, match="must return str, Template, or None, got int"):
        await Noted("Notes").compiled("from_number").render_prompt(notes="x")


async def test_the_compiled_function_and_its_threads_carry_the_published_name() -> None:
    """M3: the ``AIFunction``'s own name is the published one, so logs and runtime errors use it too."""
    fn = Solver("terse").compiled("solve")
    assert fn.name == fn.config.name == "solver-terse_solve"
    async with RuntimeHarness() as h:
        thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(1))
        forked = await thread.fork()
        assert (await h.coordinator.get_thread_info(forked.id)).thread_name == "solver-terse_solve"
        await forked.retire()
        await thread.retire()


def test_compiled_takes_a_name_override() -> None:
    """M3: ``name=`` renames the published tool per call, as it does on the decorator."""
    fn = Solver("terse").compiled("solve", name="solver_v2")
    assert fn.name == fn.config.name == "solver_v2"


def test_a_compiled_method_pickles_beside_an_unpicklable_module_global() -> None:
    """A compiled method ships to a remote worker without serializing its module's globals."""
    fn = cloudpickle.loads(cloudpickle.dumps(Shippable().compiled("ask")))
    assert fn.name == "shippable_ask"


async def test_a_retire_whose_teardown_fails_leaves_the_thread_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """M6: a failed teardown is not a retirement, so the caller can retry it."""
    from ai_functions.handle import ThreadHandle

    async with RuntimeHarness() as h:
        thread = await Solver("terse").spawn("solve", h.coordinator, model=Capturing(1))
        original = ThreadHandle.terminate_now

        async def unreachable(self: ThreadHandle[..., Any]) -> None:
            raise ConnectionError("worker unreachable")

        monkeypatch.setattr(ThreadHandle, "terminate_now", unreachable)
        with pytest.raises(ConnectionError, match="worker unreachable"):
            await thread.retire()
        assert thread.live is True

        monkeypatch.setattr(ThreadHandle, "terminate_now", original)
        await thread.retire()
        assert thread.live is False
