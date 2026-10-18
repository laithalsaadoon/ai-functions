"""``ai_method``, ``compile_ai_method``, ``MethodAgent`` and ``MethodThread``.

The decorator records an :class:`AIMethodSpec` on a method; an instance
compiles it into an ``AIFunction`` whose ``prompt_fn`` is the bound method's
typed signature over that instance's state; ``MethodAgent`` collects the
compiled capabilities as typed tools and puts one on a live thread as a
``MethodThread``. Nothing here subclasses a core type: ``AIFunction`` is
constructed with a generated ``prompt_fn``, and the thread is the runtime's
own ``ThreadHandle`` wrapped once.

Invariants:
    M1 to M9 as declared in the package docstring
    (:mod:`ai_functions.experimental.methods`); this module is the sole
    implementation of each. M1 and M2 live in :func:`compile_ai_method`, M3 in
    :func:`compile_ai_method` and :meth:`MethodAgent.spawn`, M4 in the
    generated ``prompt_fn``, M5 in :meth:`MethodAgent.spawn`, M6 and M7 in
    :class:`MethodThread`, M8 in :meth:`MethodAgent.compiled`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tstr

from ai_functions.ai_thread import AIFunction
from ai_functions.ai_thread.config import AgentKwargs, ThreadConfig, split_config_and_agent_kwargs
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.runtime import ThreadNotFoundError
from ai_functions.types import InputShape

if TYPE_CHECKING:
    from strands.models import Model

    from ai_functions.handle import ThreadHandle
    from ai_functions.protocols import Coordinator
    from ai_functions.types import Event, ThreadId

_MARKER = "__ai_method__"

# ── Errors ──


class MethodError(AIFunctionError):
    """Base of the methods package's failure modes.

    Args:
        message: Human-readable explanation.
        function_name: The compiled tool name (``{owner}_{method}``) that
            raised, when known.
    """


class CompilationError(MethodError):
    """A method could not be compiled: not an ``@ai_method``, or its parameters were lost."""


class RenderError(MethodError):
    """The method body returned ``None`` and the method has no docstring to render (M4)."""


class RetiredThreadError(MethodError):
    """An operation was attempted on a ``MethodThread`` after ``retire`` (M6)."""


# ── The decorator ──


def _owner_name(instance: object) -> str:
    """The owner half of an instance's published names: its ``name``, else the lowercased class name."""
    return getattr(instance, "name", None) or type(instance).__name__.lower()


def _published_name(instance: object, method: str) -> str:
    """The name a capability is published under unless ``name=`` renames it: ``{owner}_{method}``.

    One definition, three consumers: the compiled tool name, the live thread's
    name, and the lifecycle's error messages. They have to agree, or an error
    names a thread the caller cannot find in the tool schema it was reading.
    The separator is an underscore because providers restrict tool names:
    Bedrock's Converse API and Strands' tool-use check both require
    ``[a-zA-Z0-9_-]+``, so a dotted name fails as soon as the tool reaches a model.
    """
    return f"{_owner_name(instance)}_{method}"


@dataclass(frozen=True)
class AIMethodSpec:
    """What ``@ai_method`` records on a method until an instance compiles it.

    Args:
        output_type: The structured type the compiled function must produce.
        config: ``ThreadConfig`` fields given to the decorator, applied at
            compile time under any per-call overrides.
    """

    output_type: type[Any]
    config: dict[str, Any]


def ai_method(output_type: type[Any], /, **config: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method as an AI function whose prompt is its docstring.

    The method body is normally empty: returning ``None`` hands the docstring to
    the template renderer, exactly as the library's own decorator does. Return a
    string instead to compute the prompt directly and skip templating (M4).

    Args:
        output_type: The structured type the agent must produce.
        config: ``ThreadConfig`` fields: ``description``, ``name``,
            ``post_conditions``, ``max_attempts``, ``tools``, ``model`` and so on.

    Returns:
        A decorator that stamps an :class:`AIMethodSpec` on the method and
        returns the method unchanged, so the class still holds an ordinary
        function and ``super()`` calls and MRO walks keep working.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        setattr(method, _MARKER, AIMethodSpec(output_type, dict(config)))
        return method

    return decorate


def is_ai_method(value: object) -> bool:
    """Whether ``value`` carries an ``@ai_method`` spec.

    Args:
        value: Any class attribute; typically a function found while walking a
            class's ``__dict__``.

    Returns:
        ``True`` iff :func:`ai_method` stamped it.
    """
    return hasattr(value, _MARKER)


def compile_ai_method(instance: object, name: str, **overrides: Any) -> AIFunction[..., Any]:
    """Compile one ``@ai_method`` on ``instance`` into an ``AIFunction``.

    The generated ``prompt_fn`` carries the *bound* method's signature, so the
    typed parameters reach the model and ``self`` does not (M1). Rendering
    happens here rather than through the library's docstring path because that
    path builds its context from bound arguments alone, which would leave
    ``{self.plane}`` undefined (M4).

    Args:
        instance: The object whose state the method closes over.
        name: The attribute name of an ``@ai_method`` on ``type(instance)``.
        overrides: ``ThreadConfig`` fields applied over the decorator's config;
            ``model=`` is how a test scripts a capability offline.

    Returns:
        An ``AIFunction`` whose tool name is ``{owner}_{name}`` unless the
        decorator or ``overrides`` set ``name`` (M3).

    Raises:
        CompilationError: ``name`` is not an ``@ai_method`` on the instance's class.
    """
    attribute = getattr(type(instance), name, None)
    spec: AIMethodSpec | None = getattr(attribute, _MARKER, None)
    if spec is None:
        raise CompilationError(
            f"{type(instance).__name__}.{name} is not an @ai_method",
            function_name=_published_name(instance, name),
        )
    method = getattr(instance, name)
    signature = inspect.signature(method)
    doc = inspect.getdoc(method) or ""
    globalns = getattr(method.__func__, "__globals__", {})
    qualified = f"{type(instance).__name__}.{name}"

    @functools.wraps(method)
    async def prompt_fn(*args: Any, **kwargs: Any) -> str:
        computed = method(*args, **kwargs)
        if inspect.iscoroutine(computed):
            computed = await computed
        if computed is not None:
            return str(computed)
        if not doc:
            raise RenderError(f"{qualified} has no docstring and returned None", function_name=qualified)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        context: dict[str, object] = {"self": instance, **bound.arguments}
        template = tstr.generate_template(doc, context, globals=globalns, use_eval=True)  # pyright: ignore[reportUnknownMemberType]
        return tstr.render(template)

    prompt_fn.__signature__ = signature  # type: ignore[attr-defined]
    prompt_fn.__doc__ = None  # rendering is ours; keep the library off the docstring path

    # ``load_tools`` runs ``get_type_hints`` on this function to build the tool
    # schema. Under ``from __future__ import annotations`` the method's hints are
    # strings that only resolve against *its* module's globals, not this
    # module's, so leaving them as strings raises ``NameError`` on any type
    # declared beside the agent. Resolving them here stores real objects, which
    # ``get_type_hints`` then returns untouched.
    #
    # ``include_extras=True`` is what M2 rests on: markers like ``Procedural``
    # are ``Annotated[str, ProceduralMarker(), ...]``, and the runtime reads
    # that metadata (``ai_thread/code_execution.py`` ``detect_procedural_params``)
    # to decide whether a parameter is reusable code it should define in the
    # execution environment. Resolving without extras flattens it to plain
    # ``str`` and the code silently becomes an ordinary prompt argument.
    resolved = typing.get_type_hints(method.__func__, include_extras=True)
    resolved.pop("self", None)
    prompt_fn.__annotations__ = resolved

    # The name doubles as the tool name, so it has to identify the capability
    # and not just the agent: an agent with two ``@ai_method`` s would otherwise
    # hand a caller two tools with the same name, and the second would shadow
    # the first (M3).
    settings = {**spec.config, **overrides}
    display_name = settings.pop("name", None) or _published_name(instance, name)
    description = settings.pop("description", None) or doc.split("\n", 1)[0]
    # Same partition as ``AIFunction.replace``: ``ThreadConfig`` fields set the
    # config directly, any other key (``hooks=``, ``callback_handler=``, ...) is
    # a ``strands.Agent`` kwarg and merges into ``agent_kwargs``, so a compiled
    # method and a decorated function accept one vocabulary of overrides.
    thread_kwargs, agent_kwargs = split_config_and_agent_kwargs(**settings)
    thread_settings: dict[str, Any] = dict(thread_kwargs)
    declared_agent_kwargs: dict[str, Any] = dict(thread_settings.pop("agent_kwargs", None) or {})
    thread_settings["agent_kwargs"] = AgentKwargs(**{**declared_agent_kwargs, **agent_kwargs})  # type: ignore[typeddict-item]
    config = ThreadConfig(name=display_name, description=description, **thread_settings)
    return AIFunction(prompt_fn, spec.output_type, config)


# ── The live thread ──


class MethodThread:
    """One ``@ai_method`` running as a live thread: typed calls, shared history.

    ``compiled(name)`` is stateless: awaiting the ``AIFunction`` spawns a thread,
    runs one cycle, and tears it down, so two calls share nothing. A
    ``MethodThread`` keeps the thread alive between calls, which is what turns a
    capability into a *conversation*: ``run(problem="halting")`` then
    ``run(problem="collatz")`` and the second cycle's model context contains the
    first cycle's turns.

    **Why the unit is a method-thread, not an agent-thread.** A runtime thread
    wraps exactly one ``Spawnable``: one ``AIFunction``, one ``prompt_fn``, one
    typed signature. An agent with three ``@ai_method`` s therefore has three
    threads, not one thread it multiplexes: there is no signature that is
    simultaneously ``verify(claim: str)`` and ``determine(facts: list[str])``.
    Threads are cheap, and one Caseworker holding ``verify`` + ``determine`` +
    ``advise`` live at once is the normal shape. When a capability genuinely
    needs its sibling's context, ``spawn(..., seed_from=other.id)`` copies the
    sibling's log into the new thread: a deliberate handoff at a named point
    rather than ambient shared mutable state.

    **Why history is not kept on ``self`` (M7).** It is tempting to accumulate
    turns on the agent instance and re-render them into each prompt. That would
    be invisible to the optimizer: ``TextGradOptimizer`` routes gradients into
    ``ParameterNode`` s that ``collect_nodes`` discovers by walking the *call
    arguments*, so anything hidden on ``self`` cannot be a gradient target. It
    would also duplicate machinery the runtime owns (history is the
    coordinator's event log, reconstructed fresh per cycle) and the two copies
    would drift the moment a cycle was summarized, forked, or replayed. So this
    class holds a ``ThreadHandle`` and nothing else that resembles state; the
    log is the single source of truth.

    **The ``send_message`` boundary.** The runtime's ``send_message`` targets
    any thread whose input shape is ``STR_PROMPT``, and the shape follows the
    method's signature: a method with typed parameters compiles to
    ``STRUCTURED`` and no peer's ``send_message`` reaches it, but a method
    whose only parameter is a ``str`` compiles to ``STR_PROMPT`` and one can.
    The shape alone is therefore no boundary. Peers are meant to reach a
    capability as a *typed tool* (``agents()``), which is checkable, where a
    chat box is not; a team closes the other door by giving none of its
    threads ``send_message`` at all (C15 on the teams core). ``notify()`` is
    the inbound side channel for the cases that still want one: it appends to
    the thread's log without starting a cycle, so the next ``run`` sees it as
    context.

    Args:
        agent: The ``MethodAgent`` whose method this thread runs.
        method: The ``@ai_method`` name.
        handle: The live thread's handle, as ``coordinator.spawn`` returned it.
        coordinator: The coordinator the thread was spawned onto.
        qualified: The published tool name; defaults to ``{owner}_{method}``.
    """

    __slots__ = ("_agent", "_method", "_handle", "_coordinator", "_live", "_qualified")

    def __init__(
        self,
        agent: MethodAgent,
        method: str,
        handle: ThreadHandle[..., Any],
        coordinator: Coordinator,
        *,
        qualified: str | None = None,
    ) -> None:
        self._agent = agent
        self._method = method
        self._handle = handle
        self._coordinator = coordinator
        self._live = True
        self._qualified = qualified or _published_name(agent, method)

    # ── Identity ──

    @property
    def id(self) -> ThreadId:
        """The underlying thread id: what ``seed_from`` and ``parent_id`` take."""
        return self._handle.id

    @property
    def live(self) -> bool:
        """``False`` once ``retire()`` has run. No lifecycle op respawns a dead thread."""
        return self._live

    @property
    def handle(self) -> ThreadHandle[..., Any]:
        """The raw handle, for the runtime operations this class does not wrap.

        Raises:
            RetiredThreadError: The thread has been retired (M6).
        """
        self._require_live("handle")
        return self._handle

    @property
    def qualified_name(self) -> str:
        """The name the compiled tool is published under (M3).

        Usually ``{owner}_{method}``, but an ``@ai_method(..., name=...)`` rename
        wins, because an error message must name a thread the caller can
        actually find in the tool schema it was reading. ``spawn`` passes the
        compiled name in.
        """
        return self._qualified

    # ── Cycles ──

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run one cycle, with this method's real typed signature.

        Arguments are forwarded verbatim to the compiled ``prompt_fn``, so this
        is ``run(problem="halting", mode="thorough")`` and never
        ``run(some_string)``: the typed contract survives into the live thread.
        A binding mistake raises ``TypeError`` from the signature bind, which is
        the intended fail-loud path.

        Args:
            args: Positional arguments of the method's own signature.
            kwargs: Keyword arguments of the method's own signature.

        Returns:
            The method's typed result.

        Raises:
            RetiredThreadError: The thread has been retired (M6).
        """
        self._require_live("run")
        return await self._handle.run(*args, **kwargs)

    async def notify(self, text: str) -> None:
        """Push a side-channel message into the thread's history.

        No cycle starts; the text becomes context the next ``run`` sees. This is
        the inbound half of the ``STRUCTURED`` tradeoff described in the class
        docstring.

        Args:
            text: The message body.

        Raises:
            RetiredThreadError: The thread has been retired (M6).
        """
        self._require_live("notify")
        await self._handle.notify(text)

    # ── Topology ──

    async def fork(self, *, parent_id: ThreadId | None = None) -> MethodThread:
        """Branch into a second thread over a copy of this history.

        The fork starts byte-identical and diverges from the fork point on: one
        capability exploring two continuations of the same context, which is
        how a proposer tries variants without contaminating the line it came
        from.

        Args:
            parent_id: Passes through untouched, so the library's default holds:
                the fork is attributed to *this thread's* parent, making it a
                sibling rather than a child. Pass ``self.id`` to adopt it.

        Returns:
            A live ``MethodThread`` over the forked thread, carrying this
            thread's qualified name.

        Raises:
            RetiredThreadError: The thread has been retired (M6).
        """
        self._require_live("fork")
        forked = await self._handle.fork(parent_id=parent_id)
        # The library's fork() takes no thread_name, so the coordinator's log
        # names the fork after the bare prompt_fn. This wrapper keeps the
        # qualified name for errors and repr; the log-side rename needs
        # upstream support.
        return MethodThread(self._agent, self._method, forked, self._coordinator, qualified=self._qualified)

    async def retire(self) -> None:
        """Tear the thread down now, without draining queued cycles (M6).

        Idempotent against this object *and* against the runtime: retiring twice
        is not an error, and neither is retiring a thread something else already
        tore down (a supervisor via the raw handle, a coordinator-side
        terminate). A caller unwinding several threads should not crash
        mid-unwind because one was already gone; that would leave the rest
        alive, which is the exact failure an unwind loop exists to prevent.
        ``ThreadNotFoundError`` is a ``KeyError``, so without this catch it would
        sail past callers expecting a runtime error.
        """
        if not self._live:
            return
        self._live = False
        with contextlib.suppress(ThreadNotFoundError):
            await self._handle.terminate_now()

    # ── Guards ──

    def _require_live(self, op: str) -> None:
        """Refuse every operation on a retired thread, naming agent and method.

        Silently respawning would be worse than raising: the caller believes it
        is continuing a conversation, and it would get a blank one with the same
        typed signature, a wrong answer that looks like a right one.
        """
        if not self._live:
            raise RetiredThreadError(
                f"{self.qualified_name}: {op} after retire (thread {self.id})",
                function_name=self.qualified_name,
            )

    def __repr__(self) -> str:
        """The qualified name, thread id and liveness."""
        state = "live" if self._live else "retired"
        return f"<MethodThread {self.qualified_name} id={self.id!r} {state}>"


# ── The agent ──


class MethodAgent:
    """An object whose ``@ai_method`` s are typed AI functions over its own state.

    Every compiled method is an ``AIFunction``, which is a ``ToolProvider``, so
    one agent hands another a *typed* tool per capability rather than a chat
    box, and ``agents(...)`` collects them for exactly that purpose.

    Two ways to invoke a capability. ``compiled(name)`` is the stateless one:
    await it and one throwaway thread runs one cycle. ``spawn(name, coordinator)``
    is the live one: it returns a ``MethodThread`` whose successive ``run`` calls
    share history.

    Subclasses set ``name`` (as a class attribute or on the instance) to choose
    the owner half of every ``{owner}_{method}`` tool name; left unset, it is the
    lowercased class name.

    A capability's model resolves most-specific first (M8): the ``model=``
    passed to :meth:`compiled`, :meth:`agents` or :meth:`spawn`; the
    decorator's ``@ai_method(..., model=...)``; then this object's
    :attr:`model`. With none of them set, the capability runs on the Strands
    default model, as a plain ``@ai_function`` does.
    """

    name: str | None = None
    """The owner half of ``{owner}_{method}``; ``None`` uses the lowercased class name."""

    model: Model | str | None = None
    """The model every capability on this object runs on unless the decorator or the call names one.

    ``None`` leaves the choice to the decorator, then to the Strands default
    model. A subclass sets a class attribute to
    pin its whole surface to one model; an instance sets it to pin one object.
    """

    @classmethod
    def ai_methods(cls) -> list[str]:
        """Names of every ``@ai_method`` on this class, base classes included.

        Returns:
            Attribute names in MRO order, each once; a subclass override of an
            inherited ``@ai_method`` counts under its own definition.
        """
        found: list[str] = []
        seen: set[str] = set()
        for klass in cls.__mro__:
            for attr, value in vars(klass).items():
                if attr in seen:
                    continue
                seen.add(attr)
                if is_ai_method(value):
                    found.append(attr)
        return found

    def compiled(self, name: str, **overrides: Any) -> AIFunction[..., Any]:
        """Compile one ``@ai_method`` into a stateless ``AIFunction``.

        Args:
            name: Which ``@ai_method`` to compile.
            overrides: ``ThreadConfig`` overrides applied over the decorator's
                config. ``model=`` here wins over the decorator and the
                object's :attr:`model`, in that order (M8).
                Any key that is not a ``ThreadConfig`` field (``hooks=``,
                ``callback_handler=``, ...) is a ``strands.Agent`` kwarg and
                merges into ``agent_kwargs``, as :meth:`AIFunction.replace` does.

        Returns:
            The compiled function; awaiting it runs one throwaway cycle.

        Raises:
            CompilationError: ``name`` is not an ``@ai_method``, or the method
                has parameters and the compiled function lost them (M1 checked
                from the other side: a typed method must never compile to
                ``NO_ARGS``).
        """
        fn = compile_ai_method(self, name, **self._with_model(name, overrides))
        if fn.input_shape is InputShape.NO_ARGS and inspect.signature(getattr(self, name)).parameters:
            raise CompilationError(
                f"{name} lost its parameters during compilation", function_name=fn.config.name or name
            )
        return fn

    def _with_model(self, name: str, overrides: dict[str, Any]) -> dict[str, Any]:
        """``overrides`` with the object's :attr:`model` filled in when neither the call nor the decorator set one (M8).

        A method that is not an ``@ai_method`` is left alone so
        :func:`compile_ai_method` raises its own ``CompilationError``.
        """
        if "model" in overrides:
            return overrides
        spec: AIMethodSpec | None = getattr(getattr(type(self), name, None), _MARKER, None)
        if spec is None or spec.config.get("model") is not None:
            return overrides
        if self.model is None:
            return overrides
        return {**overrides, "model": self.model}

    def agents(self, **overrides: Any) -> list[AIFunction[..., Any]]:
        """This object's capabilities as typed tools another agent can call.

        Args:
            overrides: ``ThreadConfig`` overrides applied to every compiled method.

        Returns:
            One ``AIFunction`` per ``@ai_method``, in :meth:`ai_methods` order.
        """
        return [self.compiled(name, **overrides) for name in self.ai_methods()]

    async def spawn(
        self,
        name: str,
        coordinator: Coordinator,
        *,
        parent_id: ThreadId | None = None,
        seed_from: ThreadId | None = None,
        seed_events: Sequence[Event] | None = None,
        thread_id: ThreadId | None = None,
        thread_name: str | None = None,
        **overrides: Any,
    ) -> MethodThread:
        """Put one ``@ai_method`` on a live thread and return its lifecycle handle.

        Args:
            name: Which ``@ai_method`` to spawn. One agent may hold several live
                at once; each gets its own thread, because a thread wraps one
                ``Spawnable`` (see :class:`MethodThread`).
            coordinator: The coordinator to spawn onto: always the caller's,
                never an implicit one (M5). ``AIFunction.spawn()`` taking no
                coordinator builds a *private* in-memory coordinator plus worker
                per call, which would put this thread outside the caller's
                registry: invisible to peers, no parent edge, its own event log.
                Requiring the coordinator here makes that mistake unrepresentable.
            parent_id: Parent thread to attribute this one to, for the
                supervision tree the runtime maintains.
            seed_from: Another thread's id whose history this thread starts
                from. This is the sanctioned cross-method handoff: a thread
                cannot host two signatures, so ``determine`` inherits
                ``verify``'s context by copying its log at spawn time rather than
                by sharing a thread with it. When the two methods have different
                output types, the inherited history carries ``toolUse`` blocks
                for a tool absent from this thread's schema; reconstruction
                handles that offline, but whether a live provider accepts
                historical tool calls it was not offered is the provider's
                decision, not this library's.
            seed_events: A persisted event log to rehydrate this thread from,
                as a ``SessionStore`` returns it (:mod:`ai_functions.session`).
                The coordinator's own replay contract applies: at most one of
                ``seed_from`` and ``seed_events``, and the seeded log is in
                place before the thread is registered (M9).
            thread_id: The id to resume under, as the store saved it, so the
                rehydrated thread is the same thread to every cursor and
                reference that named it; ``None`` mints a fresh id (M9).
            thread_name: Display name in the event log. Defaults to the compiled
                tool name, so logs and tool schemas use one vocabulary (M3).
            overrides: ``ThreadConfig`` overrides, exactly as :meth:`compiled`
                takes them. One key is unreachable from here: ``name`` collides
                with the first positional parameter, so renaming the published
                tool is decorator-config only (``@ai_method(..., name=...)``).

        Compiles through ``self.compiled(name, **overrides)`` rather than
        :func:`compile_ai_method` directly, and that indirection is deliberate:
        tests bind a scripted model by replacing ``compiled`` on the *instance*,
        so routing spawn through it makes every live thread scriptable offline
        for free. Going to the module function would silently bypass those
        bindings and reach a real model.

        Returns:
            A :class:`MethodThread`. Not cached on ``self``: several may be live
            at once, and an agent that stashed handles would stop being safely
            reusable across coordinators.
        """
        fn = self.compiled(name, **overrides)
        # The published tool name, which an @ai_method(name=...) rename
        # controls: errors and logs must name a thread the caller can find in
        # the tool schema.
        published = fn.config.name or _published_name(self, name)
        handle = await coordinator.spawn(
            fn,
            thread_name=thread_name or published,
            parent_id=parent_id,
            seed_from=seed_from,
            seed_events=list(seed_events) if seed_events is not None else None,
            thread_id=thread_id,
        )
        return MethodThread(self, name, handle, coordinator, qualified=published)
