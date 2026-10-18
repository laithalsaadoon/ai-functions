"""Gated proposals: a policy check that runs as the method's post-condition.

``RetentionOffer`` proposes a discount for a customer who wants to cancel. The
discount cap lives in code, on the gate, and the prompt doesn't state it. When
a proposal goes over the cap, the gate's report becomes the retry prompt, so
the model learns the cap from the rejection and proposes again.

The example shows the gate three ways. It rejects a hand-written candidate
with no model call. ``gated()`` runs the gate on every proposal and retries
until one is admitted. ``propose_k`` takes three one-shot proposals from forked
threads and keeps the ones the gate admits.
"""

import asyncio
from dataclasses import dataclass

from example_helpers import models
from example_helpers.utils import display, rule
from pydantic import BaseModel, Field

from ai_functions.experimental.methods import GatedProposer, GateRejectionError, ai_method
from ai_functions.runtime import InMemoryCoordinator, LocalWorker

CASE = """\
Account: AnyCompany, silver tier, a customer for four years.
They had three outages last quarter and asked to cancel unless the price comes down.
"""


class Discount(BaseModel):
    percent: int = Field(description="The discount to offer, in whole percent.")
    rationale: str = Field(description="Why this discount fits the case, in one sentence.")


# A verdict is any object with an `ok` property and a `report_text()` method.
# The report is what the model reads when its proposal is rejected.
@dataclass(frozen=True)
class CapCheck:
    percent: int
    cap: int

    @property
    def ok(self) -> bool:
        return self.percent <= self.cap

    def report_text(self) -> str:
        return f"A {self.percent}% discount is over the {self.cap}% cap for this account tier."


class RetentionOffer(GatedProposer):
    name = "retention"
    model = models.medium
    # Appended to every rejection report, so the model is told what to do next.
    REASK = "Propose a smaller discount."

    def __init__(self, cap: int) -> None:
        # The gate is any callable that takes a candidate and returns a verdict.
        super().__init__(lambda percent: CapCheck(percent, cap))

    # The gate judges the percentage alone; the rationale is for the people reading the offer.
    def candidate_of(self, response: Discount) -> int:
        return response.percent

    @ai_method(Discount, description="Propose a retention discount", max_attempts=3)
    def propose(self, case: str) -> Discount:
        """A customer wants to cancel. Propose the discount to offer them.

        {case}
        """


async def main() -> None:
    offer = RetentionOffer(cap=10)

    rule("The gate on a hand-written candidate")
    try:
        offer.admits(Discount(percent=30, rationale="Three outages in one quarter."))
    except GateRejectionError as rejection:
        display("Rejected", str(rejection))
    offer.rejected.clear()

    rule("The gate as a post-condition: gated()")
    discount = await offer.gated()(case=CASE)
    display("Admitted", f"{discount.percent}%: {discount.rationale}")
    refused = [verdict.report_text() for verdict in offer.rejected]
    display("Rejected along the way", "\n".join(refused) or "None. The first proposal was under the cap.")
    offer.rejected.clear()

    rule("Three one-shot proposals: propose_k()")
    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()
    # No retries here: each branch proposes once and the gate filters the results.
    admitted = await offer.propose_k(3, coord, case=CASE)
    display(
        "Beam",
        f"{len(admitted)} of 3 admitted: {[percent for percent, _ in admitted]}\n"
        f"Rejected: {[verdict.report_text() for verdict in offer.rejected]}",
    )
    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
