"""Fixtures shared by the methods tests: agents, output types and a wire-recording model.

Shared by the core, contract, gated and recall suites. Every claim about
what the model sees is asserted from what the model actually received (``Capturing.contexts``), or from
the coordinator's own event log, never from state this package keeps for itself.

Output types and agents are module level because ``compile_ai_method`` resolves annotations with
``typing.get_type_hints`` against module globals; a function-local output type cannot be resolved.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, Literal

from pydantic import BaseModel, Field
from strands.models import Model

from ai_functions.ai_thread.reconstruction import reconstruct_messages
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types import Event

Mode = Literal["fast", "thorough"]


# ── Output types ──


class Answer(BaseModel):
    text: str


class Tips(BaseModel):
    guidance: str = Field(default="", description="What has been learned so far")


class Judged(BaseModel):
    ok: bool


class Sighting(BaseModel):
    target: str = Field(description="What was looked at")


# ── Agents ──


class Base(MethodAgent):
    name = "base"

    @ai_method(Answer, description="Inherited capability")
    def inherited(self, q: str) -> Answer:
        """Answer {q}."""
        ...


class Solver(Base):
    def __init__(self, style: str) -> None:
        self.style = style
        self.name = f"solver-{style}"

    @ai_method(Answer, description="Solve a problem in this solver's style")
    def solve(self, problem: str, mode: Mode = "fast", depth: int = 3) -> Answer:
        """Solve {problem} in the {self.style} style, {mode}, to depth {depth}."""
        ...

    @ai_method(Answer, description="Learn from accumulated guidance")
    def learn(self, guidance: str, problem: str) -> Answer:
        """Guidance so far: {guidance}

        Now solve: {problem}
        """
        ...


class Caseworker(MethodAgent):
    """Two capabilities that hand work to each other: the ``seed_from`` fixture.

    Separate from ``Solver`` on purpose: ``Solver``'s tool-name and MRO assertions pin an exact set of
    ``@ai_method`` s, and a third one there would break tests that are the frozen-surface contract.
    """

    name = "caseworker"

    @ai_method(Answer, description="Check whether a claim holds")
    def verify(self, claim: str) -> Answer:
        """Verify {claim}."""
        ...

    @ai_method(Answer, description="Decide, given what verification found")
    def determine(self, question: str) -> Answer:
        """Determine {question}."""
        ...


class MixedWorker(MethodAgent):
    """Two capabilities with *different* output types: the cross-type handoff fixture."""

    name = "mixedworker"

    @ai_method(Answer, description="Check whether a claim holds")
    def verify(self, claim: str) -> Answer:
        """Verify {claim}."""
        ...

    @ai_method(Judged, description="Judge, given what verification found")
    def judge(self, question: str) -> Judged:
        """Judge {question}."""
        ...


class Scout(MethodAgent):
    name = "scout"

    @ai_method(Sighting, description="Look at one target and report it")
    def look(self, target: str) -> Sighting:
        """Look at {target}."""
        ...


# ── Turns ──


def scripted() -> ScriptedModel:
    return ScriptedModel([Turn(tool_calls=(("Answer", {"text": "ok"}),))])


def sighting() -> Turn:
    return Turn(tool_calls=(("Sighting", {"target": "seen"}),))


# ── Recording model: contexts AND tool_specs, both wire facts ──


class Capturing(Model):
    """A ``ScriptedModel`` that also records the messages and tool specs each call received.

    ``ScriptedModel`` is ``@final`` and its ``stream`` ignores ``messages`` entirely, so there is no way
    to subclass it or read history back off it. Composition instead: this records what the model was
    handed, then delegates the response. The history assertions therefore check the thing that actually
    matters, what reached the model, rather than any bookkeeping this repository keeps.
    """

    def __init__(
        self,
        turns: int | list[Turn],
        *,
        tool: str = "Answer",
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(turns, int):
            turns = [
                Turn(tool_calls=((tool, payload if payload is not None else {"text": f"a{i}"}),)) for i in range(turns)
            ]
        self._inner = ScriptedModel(turns)
        self.contexts: list[list[Any]] = []
        self.tool_specs: list[list[str]] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, object]:
        return {"calls": len(self.contexts)}

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("scripted turns only")

    def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        self.contexts.append(list(messages))
        self.tool_specs.append([spec["name"] for spec in (tool_specs or [])])
        return self._inner.stream(messages, tool_specs, *args, **kwargs)

    def prompts(self, call: int) -> list[str]:
        """Every text block the model saw on its ``call``-th invocation."""
        return texts(self.contexts[call])


def texts(messages: list[Any]) -> list[str]:
    return [b["text"] for m in messages for b in m.get("content", []) if "text" in b]


def logged(events: list[Event]) -> list[str]:
    """The thread's history as the runtime itself reconstructs it per cycle."""
    return texts(reconstruct_messages(events))
