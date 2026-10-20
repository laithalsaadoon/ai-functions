"""``ai_method``, ``compile_ai_method``, ``MethodAgent`` and ``MethodThread``.

The decorator records an :class:`AIMethodSpec` on a method; an instance
compiles it into an ``AIFunction`` whose ``prompt_fn`` is the bound method's
typed signature over that instance's state; ``MethodAgent`` collects the
compiled capabilities as typed tools and puts one on a live thread as a
``MethodThread``, which wraps the runtime's ``ThreadHandle``.

Invariants:
    M1 to M9 as declared in the package docstring
    (:mod:`ai_functions.experimental.methods`). M1 and M2 live in
    :func:`compile_ai_method`, M3 in :func:`compile_ai_method` and
    :meth:`MethodAgent.spawn`, M4 in the generated ``prompt_fn``, M5 and M9 in
    :meth:`MethodAgent.spawn`, M6 and M7 in :class:`MethodThread`, M8 in
    :meth:`MethodAgent.compiled`.
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
from ai_functions.ai_thread.config import ThreadConfig
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.runtime import ThreadNotFoundError

if TYPE_CHECKING:
    from strands.models import Model
    from tstr import Template

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

    The compiled tool name, the live thread's name and the error messages all
    use it (M3). The separator is an underscore because Bedrock's Converse API
    and Strands' tool-use check restrict tool names to ``[a-zA-Z0-9_-]+``.
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
    the template renderer, as with ``@ai_function``. Return a string or a
    t-string ``Template`` instead to compute the prompt directly (M4).

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


def _parameter_hints(method: Callable[..., Any]) -> dict[str, Any]:
    """A bound method's parameter annotations, resolved against its module with ``Annotated`` extras kept.

    The return annotation is left unresolved: ``@ai_method`` names the output type.
    """
    func = getattr(method, "__func__", method)
    parameters = inspect.signature(method).parameters

    def annotated() -> None: ...

    annotated.__annotations__ = {k: v for k, v in func.__annotations__.items() if k in parameters}
    return typing.get_type_hints(annotated, globalns=func.__globals__, include_extras=True)


def compile_ai_method(instance: object, name: str, /, **overrides: Any) -> AIFunction[..., Any]:
    """Compile one ``@ai_method`` on ``instance`` into an ``AIFunction``.

    The generated ``prompt_fn`` carries the *bound* method's signature, so the
    typed parameters reach the model and ``self`` does not (M1). It returns the
    docstring as a template with ``self`` and the bound arguments in scope, so a
    template can read ``{self.plane}``, and ``AIFunction`` renders it as it
    renders an ``@ai_function`` docstring (M4).

    Args:
        instance: The object whose state the method closes over.
        name: The attribute name of an ``@ai_method`` on ``type(instance)``.
        overrides: ``ThreadConfig`` fields applied over the decorator's config,
            as :meth:`AIFunction.replace` applies them.

    Returns:
        An ``AIFunction`` named ``{owner}_{name}`` unless the decorator or
        ``overrides`` set ``name`` (M3).

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
    qualified = f"{type(instance).__name__}.{name}"
    # The name is the tool name, so it identifies the capability as well as the agent (M3).
    settings: dict[str, Any] = {**spec.config, **overrides}
    settings["name"] = settings.get("name") or _published_name(instance, name)
    settings["description"] = settings.get("description") or doc.split("\n", 1)[0]

    @functools.wraps(method)
    async def prompt_fn(*args: Any, **kwargs: Any) -> str | Template:
        computed = method(*args, **kwargs)
        if inspect.iscoroutine(computed):
            computed = await computed
        if computed is not None:
            return computed  # AIFunction renders a Template and refuses anything but str
        if not doc:
            raise RenderError(f"{qualified} has no docstring and returned None", function_name=qualified)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        context: dict[str, object] = {"self": instance, **bound.arguments}
        return tstr.generate_template(doc, context, globals=method.__func__.__globals__, use_eval=True)  # pyright: ignore[reportUnknownMemberType]

    prompt_fn.__name__ = settings["name"]
    prompt_fn.__signature__ = signature  # type: ignore[attr-defined]
    prompt_fn.__doc__ = None  # prompt_fn returns the template; AIFunction must not template __doc__ again

    # ``load_tools`` runs ``get_type_hints`` on this function to build the tool
    # schema. Under ``from __future__ import annotations`` the method's hints are
    # strings that only resolve against *its* module's globals, so they are
    # resolved here, against those globals, and stored as real objects (M2).
    prompt_fn.__annotations__ = _parameter_hints(method)
    return AIFunction(prompt_fn, spec.output_type, ThreadConfig()).replace(**settings)


# ── The live thread ──


class MethodThread:
    """One ``@ai_method`` running as a live thread: typed calls, shared history.

    ``compiled(name)`` is stateless: awaiting the ``AIFunction`` spawns a thread,
    runs one cycle, and tears it down, so two calls share nothing. A
    ``MethodThread`` keeps the thread alive between calls, which is what turns a
    capability into a *conversation*: ``run(problem="halting")`` then
    ``run(problem="collatz")`` and the second cycle's model context contains the
    first cycle's turns.

    A runtime thread wraps one ``Spawnable``, so each ``@ai_method`` runs on its
    own thread; ``spawn(..., seed_from=other.id)`` starts one capability's
    thread from another's history. History is the coordinator's event log
    (M7). The runtime's ``send_message`` reaches a thread whose input shape is
    ``STR_PROMPT``: a method with typed parameters compiles to ``STRUCTURED``
    and is not reachable that way, while a method whose only parameter is a
    ``str`` compiles to ``STR_PROMPT`` and is. ``notify()`` appends to the
    thread's log without starting a cycle, so the next ``run`` sees it as
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

        Usually ``{owner}_{method}``; an ``@ai_method(..., name=...)`` rename
        wins. ``spawn`` passes the compiled name in.
        """
        return self._qualified

    # ── Cycles ──

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run one cycle, with this method's real typed signature.

        Arguments are forwarded verbatim to the compiled ``prompt_fn``; a
        binding mistake raises ``TypeError`` from the signature bind.

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

        No cycle starts; the text becomes context the next ``run`` sees.

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

        The fork starts from a copy of this thread's log and diverges from
        there; this thread is unaffected.

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
        return MethodThread(self._agent, self._method, forked, self._coordinator, qualified=self._qualified)

    async def retire(self) -> None:
        """Tear the thread down now, without draining queued cycles (M6).

        Idempotent against this object and against the runtime: retiring twice
        is not an error, and neither is retiring a thread something else already
        tore down (``ThreadNotFoundError`` is suppressed). A teardown that raises
        anything else leaves the thread live, so ``retire`` can be retried.
        """
        if not self._live:
            return
        with contextlib.suppress(ThreadNotFoundError):
            await self._handle.terminate_now()
        self._live = False

    # ── Guards ──

    def _require_live(self, op: str) -> None:
        """Refuse every operation on a retired thread, naming the capability, the operation and the thread."""
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
    ``agents(...)`` hands another agent one typed tool per capability.

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
    model. A class attribute pins every instance of a subclass to one model;
    an instance attribute pins one object.
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

    def compiled(self, name: str, /, **overrides: Any) -> AIFunction[..., Any]:
        """Compile one ``@ai_method`` into a stateless ``AIFunction``.

        Args:
            name: Which ``@ai_method`` to compile.
            overrides: ``ThreadConfig`` overrides applied over the decorator's
                config. ``model=`` here wins over the decorator and the
                object's :attr:`model`, in that order (M8), and ``name=``
                renames the published tool (M3). Any key that is not a
                ``ThreadConfig`` field (``hooks=``, ``callback_handler=``, ...)
                is a ``strands.Agent`` kwarg and merges into ``agent_kwargs``,
                as :meth:`AIFunction.replace` does.

        Returns:
            The compiled function; awaiting it runs one throwaway cycle.

        Raises:
            CompilationError: ``name`` is not an ``@ai_method``.
        """
        return compile_ai_method(self, name, **self._with_model(name, overrides))

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
        /,
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
                at once, each on its own thread.
            coordinator: The coordinator to spawn onto; the thread joins its
                registry and event log (M5).
            parent_id: Parent thread to attribute this one to, for the
                supervision tree the runtime maintains.
            seed_from: Another thread's id whose history this thread starts
                from, such as another capability's thread on this agent. When
                the two methods have different output types, the copied history
                contains ``toolUse`` blocks for a tool absent from this
                thread's schema; whether a provider accepts them is
                provider-specific.
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
                takes them.

        Returns:
            A new :class:`MethodThread`; the agent keeps no reference to it.
        """
        fn = self.compiled(name, **overrides)
        handle = await coordinator.spawn(
            fn,
            thread_name=thread_name or fn.name,
            parent_id=parent_id,
            seed_from=seed_from,
            seed_events=list(seed_events) if seed_events is not None else None,
            thread_id=thread_id,
        )
        return MethodThread(self, name, handle, coordinator, qualified=fn.name)
