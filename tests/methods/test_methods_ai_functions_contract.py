"""Executable pins for the installed ``ai_functions`` behaviors ``experimental/methods`` relies on.

Verified against the installed sources. ``methods/core.py`` cites
load-bearing runtime behaviors in its docstrings; those pins are prose, and an upstream change that
moved the semantics would break tool composition silently. Each test here is the executable version
of one pin, asserted on the wire (what a recording model actually received) or against the runtime
object itself, so a semantic change fails THIS file loudly.

What is pinned, and which methods docstring relies on it:

1. ``ai_thread/ai_thread.py:549-555``: exactly one ``config_hook`` resolved per cycle, its patch
   applied via ``dataclasses.replace`` so a ``tools`` key REPLACES compiled tools wholesale (no
   merge), and the ``config_hook`` key itself is popped (line 553). ``ai_thread.py:583-590``: when
   ``coordinator_tools_enabled``, the runtime APPENDS ``list_threads``/``send_message`` AFTER the
   hook's replacement. Relied on by anything that passes ``config_hook`` through ``@ai_method``
   config or ``compiled(..., config_hook=...)``.
2. Same block: one hook resolution per cycle, decoy never called.
3. ``handle.py:120-132``: ``ThreadHandle.notify`` appends to the inject buffer and starts NO cycle;
   the next run observes it as context. Relied on by ``MethodThread.notify``.
4. ``ai_thread/config.py:72-93``: ``ThreadKwargs`` declares the keys ``compile_ai_method`` and
   ``MethodAgent.spawn`` pass as config (``name``, ``description``, ``model``, ``tools``,
   ``config_hook``, ``thread_name``).
5. ``memory/base.py:120-140``: ``MemoryBackend._resolve_field`` (slash-path resolution, ``KeyError``
   on a missing field) and ``_is_procedural``/``_is_frozen`` marker detection. Relied on by the
   learning hook that lands later on top of this module.
6. ``runtime/worker.py:857-896``: ``LocalWorker.close()`` is idempotent. Relied on by every teardown
   path that may close a worker it did not open.
7. ``ai_thread/ai_function.py:425-436`` + ``ai_thread/code_execution.py:288-311``: ``load_tools``
   builds the tool schema from ``inspect.signature(prompt_fn)`` and ``prompt_fn.__annotations__``, and
   ``detect_procedural_params`` reads ``Annotated`` metadata from
   ``get_type_hints(prompt_fn, include_extras=True)``. This is the seam ``compile_ai_method`` drives
   by setting ``__signature__`` and ``__annotations__`` (M1, M2).
8. strands ``tools/decorator.py``: ``pydantic.Field`` inside ``Annotated`` is refused with
   ``NotImplementedError``, so a ``Procedural`` parameter compiles and is detected but cannot be exposed
   through ``load_tools``. An installed limitation found while porting, pinned so a change is noticed.
9. ``ai_thread/ai_function.py:434``: ``load_tools`` copies string annotations verbatim, so a plain
   ``@ai_function`` under ``from __future__ import annotations`` raises ``NameError`` for any type declared
   beside it. The reason ``compile_ai_method`` resolves hints itself (M2); pinned with the compiled
   counterpart loading fine.
"""

from __future__ import annotations

import inspect
import typing
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from strands.tools.decorator import tool as strands_tool

from ai_functions import Frozen, InMemoryCoordinator, LocalWorker, Procedural, ai_function
from ai_functions.ai_thread.code_execution import detect_procedural_params
from ai_functions.ai_thread.config import ThreadKwargs
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.memory.base import MemoryBackend
from ai_functions.memory.procedural import ProceduralMarker
from ai_functions.testing import RuntimeHarness

from .methods_wire import Capturing, Mode, Scout, Sighting, sighting

# ── Probes ──


@strands_tool(name="tool_a", description="the tool the compiled config carries")
def tool_a(x: str) -> str:
    return x


@strands_tool(name="tool_b", description="the tool the hook's patch carries")
def tool_b(x: str) -> str:
    return x


# ── 1+2. config_hook: replacement semantics, one call per cycle, key popped ──


async def test_a_hooks_tools_patch_replaces_compiled_tools_coordinator_tools_append() -> None:
    """ai_thread.py:549-555: the hook's ``tools`` REPLACE the compiled tools wholesale — tool_b is on the wire and
    tool_a is NOT. And ai_thread.py:583-590: the runtime appends ``list_threads``/``send_message`` AFTER that
    replacement, so they survive it. Driven through ``compiled(...)`` so the pin is measured on the path methods
    actually uses. (Pin 1.)"""
    model = Capturing([sighting()])

    def hook(ctx: Any) -> ThreadKwargs:
        return {"tools": [tool_b]}

    async with RuntimeHarness() as h:
        handle = await h.spawn(Scout().compiled("look", model=model, tools=[tool_a], config_hook=hook))
        await handle.run(target="anything")

    specs = model.tool_specs[0]
    assert "tool_b" in specs, f"the hook's tool never reached the wire: {specs}"
    assert "tool_a" not in specs, f"REPLACE became merge — compiled tools survived: {specs}"
    for runtime_tool in ("list_threads", "send_message"):
        assert runtime_tool in specs, f"{runtime_tool} missing: coordinator tools no longer append after the hook"


async def test_the_config_hook_is_called_exactly_once_per_cycle_and_cannot_replace_itself() -> None:
    """ai_thread.py:549-555: one hook resolution per cycle — two ``run()`` cycles on one thread are exactly two
    calls. And line 553: a ``config_hook`` key in the patch is popped, so a hook smuggling in a replacement hook
    changes nothing — the decoy is never called and the original still fires on the next cycle. (Pin 2.)"""
    calls = {"hook": 0, "decoy": 0}

    def decoy(ctx: Any) -> ThreadKwargs:
        calls["decoy"] += 1
        return {}

    def hook(ctx: Any) -> ThreadKwargs:
        calls["hook"] += 1
        return {"config_hook": decoy}  # popped by the runtime, a no-op

    model = Capturing([sighting(), sighting()])

    async with RuntimeHarness() as h:
        handle = await h.spawn(Scout().compiled("look", model=model, config_hook=hook))
        await handle.run(target="first")
        await handle.run(target="second")

    assert calls["hook"] == 2, f"expected one hook call per cycle, got {calls['hook']}"
    assert calls["decoy"] == 0, "the patch's config_hook key was applied instead of popped"
    assert len(model.contexts) == 2


# ── 3. notify: buffered, never a cycle ──


async def test_notify_starts_no_cycle_and_the_text_arrives_as_the_next_runs_context() -> None:
    """handle.py:120-132: ``notify`` appends to the inject buffer and starts nothing — zero model calls after it. The
    next ``run`` carries the text in its context. (Pin 3; ``MethodThread.notify`` rests on this.)"""
    model = Capturing([sighting()])

    async with RuntimeHarness() as h:
        handle = await h.spawn(Scout().compiled("look", model=model))
        await handle.notify("the deadline moved to Friday")
        assert model.contexts == [], "notify started a cycle"

        await handle.run(target="reschedule")
        assert any("the deadline moved to Friday" in p for p in model.prompts(0)), (
            "the notified text never reached the next cycle's context"
        )


# ── 4. ThreadKwargs surface ──


def test_thread_kwargs_declares_every_key_the_methods_module_passes() -> None:
    """config.py:72-93: the keys ``compile_ai_method`` and ``MethodAgent.spawn`` hand to ``ThreadConfig`` all exist
    on ``ThreadKwargs`` — a renamed or dropped key breaks composition silently because TypedDict keys are not
    checked at runtime. (Pin 4.)"""
    declared = ThreadKwargs.__required_keys__ | ThreadKwargs.__optional_keys__
    assert {"model", "tools", "config_hook", "system_prompt", "thread_name", "name", "description"} <= declared


# ── 5. MemoryBackend schema introspection ──


class Notes(BaseModel):
    guidance: str = Field(default="", description="prose advice")
    recipe: Procedural
    identity: Frozen[str] = "fixed"


class _ProbeBackend(MemoryBackend):
    """Minimal concrete backend: the test uses only the base-class introspection."""

    def close(self) -> None:
        pass

    def _save(self, name: str, value: Any) -> None:
        pass

    def _recall(self, name: str) -> Any:
        raise NotImplementedError

    def _query(self, name: str, query: str) -> Any:
        raise NotImplementedError

    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _consolidate(self, name: str, feedback: Any, retrieved: Any = None, **kwargs: Any) -> None:
        pass

    def _delete(self, name: str) -> None:
        pass


def test_memory_backend_field_resolution_and_marker_detection_hold() -> None:
    """memory/base.py:120-140: ``_resolve_field`` returns a ``FieldInfo``, raises ``KeyError`` on a missing name, and
    ``_is_procedural``/``_is_frozen`` answer False for a plain prose field and True for marked ones. (Pin 5.)"""
    backend = _ProbeBackend(Notes, "probe")

    assert isinstance(backend._resolve_field("guidance"), FieldInfo)
    with pytest.raises(KeyError):
        backend._resolve_field("no_such_field")

    assert backend._is_procedural("guidance") is False
    assert backend._is_frozen("guidance") is False
    assert backend._is_procedural("recipe") is True
    assert backend._is_frozen("identity") is True


# ── 6. LocalWorker.close idempotence ──


async def test_local_worker_close_is_idempotent() -> None:
    """runtime/worker.py:857-896: closing an already-closed worker is a no-op, so an unwind path that closes a worker
    somebody else already closed does not crash. (Pin 6.)"""
    coordinator = InMemoryCoordinator()
    worker = LocalWorker(coordinator)
    await worker.register()
    await worker.close()
    await worker.close()  # must not raise


# ── 7. load_tools reads __signature__ and __annotations__; procedural detection reads Annotated metadata ──


async def test_load_tools_builds_the_schema_from_signature_and_annotations_with_extras() -> None:
    """ai_function.py:425-436: the tool's signature is ``inspect.signature(prompt_fn)`` and its annotations are
    ``prompt_fn.__annotations__`` (plus ``return``); code_execution.py:288-311: ``detect_procedural_params`` reads
    ``ProceduralMarker`` from ``get_type_hints(prompt_fn, include_extras=True)``. Measured on a hand-built
    ``prompt_fn`` carrying a foreign signature, the exact seam ``compile_ai_method`` sets (M1, M2). (Pin 7.)
    The marker here is a bare ``Annotated[str, ProceduralMarker()]``: see the next pin for why the
    library's own ``Procedural`` alias cannot go through ``load_tools``."""
    from ai_functions import AIFunction
    from ai_functions.ai_thread.config import ThreadConfig

    def shape(recipe: Annotated[str, ProceduralMarker(), "reusable code"], target: str = "x") -> str:
        return ""

    async def prompt_fn(*args: Any, **kwargs: Any) -> str:
        return "irrelevant"

    prompt_fn.__signature__ = inspect.signature(shape)  # type: ignore[attr-defined]
    prompt_fn.__annotations__ = typing.get_type_hints(shape, include_extras=True)

    fn = AIFunction(prompt_fn, BaseModel, ThreadConfig(name="probe.shape", description="probe"))
    tool = (await fn.load_tools())[0]
    schema = tool.tool_spec["inputSchema"]["json"]
    assert tool.tool_name == "probe.shape"
    assert set(schema["properties"]) == {"recipe", "target"}, schema
    assert schema["properties"]["target"]["default"] == "x"
    assert schema["properties"]["recipe"]["description"] == "reusable code"
    assert detect_procedural_params(prompt_fn) == {"recipe"}
    hints = typing.get_type_hints(prompt_fn, include_extras=True)
    assert any(isinstance(m, ProceduralMarker) for m in hints["recipe"].__metadata__)


# ── 8. Installed limitation: strands refuses pydantic.Field inside Annotated on the tool path ──


class Coder(MethodAgent):
    name = "coder"

    @ai_method(Sighting, description="Run reusable code", code_execution_mode="local")
    def apply(self, recipe: Procedural, target: str) -> Sighting:
        """Use {recipe} on {target}."""
        ...


async def test_strands_tool_decorator_refuses_a_field_inside_annotated_so_procedural_cannot_be_a_tool_param() -> None:
    """strands/tools/decorator.py (``_create_input_model``): a parameter whose ``Annotated`` metadata carries a
    ``pydantic.Field`` raises ``NotImplementedError``. ``ai_functions.Procedural`` is exactly that alias, so a
    capability with a ``Procedural`` parameter compiles (M2 holds: the marker survives and the runtime detects it)
    but cannot be handed to another agent through ``load_tools``. Not a methods fact: a plain ``AIFunction`` whose
    annotations are already resolved fails the same way. Pinned so that a strands release lifting the restriction is
    noticed here first."""
    from ai_functions import AIFunction
    from ai_functions.ai_thread.config import ThreadConfig

    async def plain(recipe: Procedural, target: str) -> str:
        return "irrelevant"

    plain.__annotations__ = typing.get_type_hints(plain, include_extras=True)

    for fn in (Coder().compiled("apply"), AIFunction(plain, Sighting, ThreadConfig(name="plain"))):
        assert detect_procedural_params(fn.prompt_fn) == {"recipe"}
        with pytest.raises(NotImplementedError, match="pydantic.Field within Annotated"):
            await fn.load_tools()


# ── 9. Installed limitation: load_tools copies string annotations verbatim ──


@ai_function[Sighting]
def plain_look(target: str, mode: Mode = "fast") -> str | None:
    """Look at {target}, {mode}."""


async def test_load_tools_on_a_plain_function_cannot_resolve_a_type_declared_beside_it() -> None:
    """ai_function.py:434: ``load_tools`` copies ``prompt_fn.__annotations__`` as they are; under ``from __future__
    import annotations`` they are strings, and strands evaluates them against ``_invoke.__globals__`` (ai_function.py's
    module), so any name declared beside the function (``Mode`` here) raises ``NameError``. This is the exact failure
    ``compile_ai_method`` pre-empts by storing resolved objects in ``__annotations__`` (M2); the same signature
    compiled through ``@ai_method`` loads. Pinned so an upstream fix (resolving in ``load_tools``) is noticed and the
    resolve step can then be reconsidered."""
    with pytest.raises(NameError, match="Mode"):
        await plain_look.load_tools()

    class Looker(MethodAgent):
        name = "looker"

        @ai_method(Sighting, description="The same signature, compiled through @ai_method")
        def look(self, target: str, mode: Mode = "fast") -> Sighting:
            """Look at {target}, {mode}."""
            ...

    schema = (await Looker().compiled("look").load_tools())[0].tool_spec["inputSchema"]["json"]
    assert schema["properties"]["mode"]["enum"] == ["fast", "thorough"]
