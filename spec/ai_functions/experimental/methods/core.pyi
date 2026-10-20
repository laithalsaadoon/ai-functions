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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...ai_thread import AIFunction
from ...ai_thread.errors import AIFunctionError

if TYPE_CHECKING:
    from strands.models import Model

    from ...handle import ThreadHandle
    from ...protocols import Coordinator
    from ...types import Event, ThreadId


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
    ...

def is_ai_method(value: object) -> bool:
    """Whether ``value`` carries an ``@ai_method`` spec.

    Args:
        value: Any class attribute; typically a function found while walking a
            class's ``__dict__``.

    Returns:
        ``True`` iff :func:`ai_method` stamped it.
    """
    ...

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
    ...

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

    def __init__(
        self,
        agent: MethodAgent,
        method: str,
        handle: ThreadHandle[..., Any],
        coordinator: Coordinator,
        *,
        qualified: str | None = None,
    ) -> None: ...
    @property
    def id(self) -> ThreadId:
        """The underlying thread id: what ``seed_from`` and ``parent_id`` take."""
        ...

    @property
    def live(self) -> bool:
        """``False`` once ``retire()`` has run. No lifecycle op respawns a dead thread."""
        ...

    @property
    def handle(self) -> ThreadHandle[..., Any]:
        """The raw handle, for the runtime operations this class does not wrap.

        Raises:
            RetiredThreadError: The thread has been retired (M6).
        """
        ...

    @property
    def qualified_name(self) -> str:
        """The name the compiled tool is published under (M3).

        Usually ``{owner}_{method}``; an ``@ai_method(..., name=...)`` rename
        wins. ``spawn`` passes the compiled name in.
        """
        ...

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
        ...

    async def notify(self, text: str) -> None:
        """Push a side-channel message into the thread's history.

        No cycle starts; the text becomes context the next ``run`` sees.

        Args:
            text: The message body.

        Raises:
            RetiredThreadError: The thread has been retired (M6).
        """
        ...

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
        ...

    async def retire(self) -> None:
        """Tear the thread down now, without draining queued cycles (M6).

        Idempotent against this object and against the runtime: retiring twice
        is not an error, and neither is retiring a thread something else already
        tore down (``ThreadNotFoundError`` is suppressed). A teardown that raises
        anything else leaves the thread live, so ``retire`` can be retried.
        """
        ...

    def __repr__(self) -> str:
        """The qualified name, thread id and liveness."""
        ...

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

    A capability's model resolves most-specific first (M8): the call's
    ``model=``, then the decorator's, then :attr:`model`; with none set, the
    Strands default model.
    """

    name: str | None
    """The owner half of ``{owner}_{method}``; ``None`` uses the lowercased class name."""

    model: Model | str | None
    """The model every capability on this object runs on unless the decorator or the call names one; ``None`` leaves it to the decorator, then the Strands default."""

    @classmethod
    def ai_methods(cls) -> list[str]:
        """Names of every ``@ai_method`` on this class, base classes included.

        Returns:
            Attribute names in MRO order, each once; a subclass override of an
            inherited ``@ai_method`` counts under its own definition.
        """
        ...

    def compiled(self, name: str, /, **overrides: Any) -> AIFunction[..., Any]:
        """Compile one ``@ai_method`` into a stateless ``AIFunction``.

        Args:
            name: Which ``@ai_method`` to compile.
            overrides: ``ThreadConfig`` overrides applied over the decorator's
                config; ``model=`` here wins over the decorator and
                :attr:`model` (M8), and ``name=`` renames the published tool
                (M3). Any key that is not a ``ThreadConfig`` field
                (``hooks=``, ``callback_handler=``, ...) is a ``strands.Agent``
                kwarg merged into ``agent_kwargs``, as
                :meth:`AIFunction.replace` does.

        Returns:
            The compiled function; awaiting it runs one throwaway cycle.

        Raises:
            CompilationError: ``name`` is not an ``@ai_method``.
        """
        ...

    def agents(self, **overrides: Any) -> list[AIFunction[..., Any]]:
        """This object's capabilities as typed tools another agent can call.

        Args:
            overrides: ``ThreadConfig`` overrides applied to every compiled method.

        Returns:
            One ``AIFunction`` per ``@ai_method``, in :meth:`ai_methods` order.
        """
        ...

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
            parent_id: Parent thread to attribute this one to.
            seed_from: Another thread's id whose history this thread starts
                from, such as another capability's thread on this agent.
            seed_events: A persisted event log to rehydrate this thread from,
                as a ``SessionStore`` returns it; at most one of ``seed_from``
                and ``seed_events`` (M9).
            thread_id: The id to resume under; ``None`` mints a fresh id (M9).
            thread_name: Display name in the event log. Defaults to the compiled
                tool name (M3).
            overrides: ``ThreadConfig`` overrides, exactly as :meth:`compiled`
                takes them.

        Returns:
            A new :class:`MethodThread`; the agent keeps no reference to it.
        """
        ...
