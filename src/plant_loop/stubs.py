"""Stub reasoner profiles: labelled stand-ins, never production.

This module exists for the reasoner regression gate
(``regression.py``). It is deliberately NOT wired into
:func:`plant_loop.backends.get_reasoner` — no ``REASONING_BACKEND``
value selects anything here, and no deployment configuration can
reach these classes by accident. They are importable only by name,
by the gate runner and the tests, the way a load test's fixture
users exist only inside the load test.

:class:`StubCandidateReasoner` is the gate's exercise partner: a
deterministic reasoner that behaves exactly like the offline
baseline except for two small, deliberate differences —

- on case **G14** it reads the welder's fault as the *other* WELD-4
  fault class (``weld_quality_drift`` instead of
  ``wire_feed_fault``): a plausible model disagreement that still
  converges, so the comparison must report a diagnosis difference
  without failing;
- on case **G20** its first diagnosis carries a close runner-up the
  baseline never raises, so the verify round sends it back once:
  one extra round, same exit, same final diagnosis — a pure cost
  delta for the report to show.

A candidate that differs *only* in ways like these is what a good
model swap looks like; the gate must PASS it. Candidates that break
a safety pin or regress an exit are constructed in the tests, not
shipped — a profile whose purpose is to break safety has no
business being selectable anywhere, even here.
"""

from __future__ import annotations

from . import kb
from .backends import OfflineReasoner
from .ports import CaseContext, RoundOutput
from .schemas import Diagnosis, Gap, Hypothesis

#: The case the stub diagnoses differently, and the fault class it
#: insists on: a real WELD-4 class with its own SOP, so the plan
#: the loop builds for it is a real plan that can converge.
STUB_DIAGNOSIS_CASE = "G14"
STUB_DIAGNOSIS_AS = "weld_quality_drift"

#: The case the stub spends one extra round on, and the runner-up
#: it floats in round 1 only — close enough to the leader that the
#: critic must send the case back, absent from round 2's readings
#: so the case then converges exactly as the baseline did.
STUB_EXTRA_ROUND_CASE = "G20"
STUB_EXTRA_ROUND_RIVAL = "wire_feed_fault"
STUB_EXTRA_ROUND_RIVAL_CONFIDENCE = 0.35


class StubCandidateReasoner(OfflineReasoner):
    """The offline rules with two deliberate, documented deviations."""

    name = "stub-candidate"
    model = "rules-v1+stub"

    def __init__(self) -> None:
        self._diagnosed: set[str] = set()

    def diagnose(self, ctx: CaseContext, focus: list[Gap]) -> RoundOutput:
        out = super().diagnose(ctx, focus)
        case_id = ctx.case.case_id
        first_call = case_id not in self._diagnosed
        self._diagnosed.add(case_id)
        diagnosis: Diagnosis = out.payload

        if case_id == STUB_DIAGNOSIS_CASE:
            sop = kb.sop_for(ctx.case.machine_id, STUB_DIAGNOSIS_AS) or {}
            diagnosis = Diagnosis(
                hypotheses=[
                    Hypothesis(
                        fault_class=STUB_DIAGNOSIS_AS,
                        statement=(
                            f"{sop.get('title', STUB_DIAGNOSIS_AS)} fits the readings "
                            f"for {ctx.case.machine_id} (stub reading)."
                        ),
                        confidence=0.7,
                        evidence=[
                            "stub candidate: reads the weld-line signature as "
                            "quality drift rather than a wire-feed fault"
                        ],
                        sop_id=sop.get("sop_id", ""),
                    )
                ],
                summary=f"Top hypothesis: {STUB_DIAGNOSIS_AS} at 0.70 confidence (stub).",
            )
        elif case_id == STUB_EXTRA_ROUND_CASE and first_call:
            if all(h.fault_class != STUB_EXTRA_ROUND_RIVAL for h in diagnosis.hypotheses):
                rival = Hypothesis(
                    fault_class=STUB_EXTRA_ROUND_RIVAL,
                    statement="Stub candidate: a wire-feed fault is not yet separated.",
                    confidence=STUB_EXTRA_ROUND_RIVAL_CONFIDENCE,
                    evidence=["stub candidate: floats the rival for one round"],
                    sop_id=(kb.sop_for(ctx.case.machine_id, STUB_EXTRA_ROUND_RIVAL) or {}).get(
                        "sop_id", ""
                    ),
                )
                diagnosis = Diagnosis(
                    hypotheses=sorted(
                        [*diagnosis.hypotheses, rival],
                        key=lambda h: h.confidence,
                        reverse=True,
                    )[:3],
                    summary=diagnosis.summary,
                )
        return RoundOutput(
            payload=diagnosis,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            latency_ms=out.latency_ms,
            backend=self.name,
            model=self.model,
        )
