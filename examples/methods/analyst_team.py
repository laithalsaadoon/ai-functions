"""Method agents: typed AI methods over an object's own state.

``Analyst`` keeps one service's incident notes on ``self``. Each ``@ai_method``
is a typed AI function whose docstring is its prompt, filled in with ``self``
and the call's arguments, so callers pass only a question.

The example uses the analyst three ways. ``compiled(name)`` runs one method as
a one-shot call. ``agents()`` hands every method to an orchestrating
``@ai_function`` as a typed tool. ``spawn(name, coordinator)`` keeps one method
on a live thread, so a follow-up question can refer to the previous answer.
"""

import asyncio

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

NOTES = """\
09:02 deploy 4172 rolls out to checkout: new payment client, retry limit changed from 5 to 0
09:05 p99 latency on /pay climbs from 300 ms to 2.1 s
09:07 error rate on /pay reaches 4%; the card processor reports no incidents on its side
09:12 on-call rolls back deploy 4172
09:15 latency and error rate on /pay are back to baseline
"""


class Finding(BaseModel):
    answer: str = Field(description="The answer, in one or two sentences.")
    evidence: list[str] = Field(description="The note lines the answer rests on, quoted exactly.")


class Analyst(MethodAgent):
    """The analyst for one service. Its notes live on ``self`` and reach every prompt."""

    # Every method on this agent runs on this model unless the decorator or the call names another.
    model = models.medium

    def __init__(self, service: str, notes: str) -> None:
        self.service = service
        self.notes = notes

    # Other agents see each method as a tool named `{owner}_{method}`: `analyst_timeline`
    # and `analyst_causes`. The owner is the lowercased class name unless `name` is set.
    @ai_method(Finding, description="Answer a question about when things happened")
    def timeline(self, question: str) -> Finding:
        """You're the analyst for the {self.service} service. Answer from these notes only:

        {self.notes}
        Question: {question}
        """

    @ai_method(Finding, description="Name the most likely causes of a symptom")
    def causes(self, symptom: str, max_causes: int = 2) -> Finding:
        """You're the analyst for the {self.service} service. Using only these notes:

        {self.notes}
        Name at most {max_causes} likely causes of this symptom: {symptom}
        """


analyst = Analyst("checkout", NOTES)


# The orchestrator gets one typed tool per analyst method. It calls
# `analyst_causes(symptom=..., max_causes=...)` with typed arguments, and the notes stay on the analyst.
@ai_function(model=models.medium, tools=analyst.agents())
def postmortem(incident: str) -> str:
    """Write a postmortem of at most five lines for this incident: {incident}

    Ask the analyst tools for the timeline and the likely cause. Only state what they report.
    """


async def main() -> None:
    rule("One method, one call: compiled()")
    finding = await analyst.compiled("causes")(symptom="errors on /pay", max_causes=1)
    display("analyst_causes", finding.model_dump_json(indent=2), lang="json")

    rule("Every method as a typed tool: agents()")
    report = await postmortem(incident="payment errors on checkout during the 09:00 deploy")
    display("Postmortem", report)

    rule("One method on a live thread: spawn()")
    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()
    thread = await analyst.spawn("timeline", coord)

    first = await thread.run(question="When did latency on /pay start climbing?")
    display("Turn 1", first.answer)

    # The second call sees the first call's turn, so "that" refers to the latency spike.
    second = await thread.run(question="How many minutes after that was the rollback?")
    display("Turn 2", second.answer)

    await thread.retire()
    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
