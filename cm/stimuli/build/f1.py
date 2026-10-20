"""F1 honesty vs care (white lies): ConflictScope tenet scenarios and DailyDilemmas (contrast), AIRiskDilemmas
(held-out and behavior). pos = the option that favours honesty, neg = the option that favours care.

    python -m cm.stimuli.build.f1
"""
from __future__ import annotations

from . import _conflict as F
from . import _moral as M

CID, PARTNERS = "F1", ["C1", "C5"]
HONEST = {"honest_creative_feedback", "honest_uncomfortable_truths", "accurate_overall_impressions",
          "correcting_counterproductive_advice", "resisting_crowd_agreement"}
CARE = {"addressing_underlying_distress", "non_amplifying_emotional_validation", "mental_health_awareness",
        "direct_crisis_response"}
H_KW = ["honest", "truth"]
C_KW = ["compassion", "empath", "kindness", "emotional", "feelings", "comfort"]


def main():
    gen = M.Gen(CID)
    contrast = F.tenets(CID, HONEST, CARE, maxl=3000) + F.daily(CID, H_KW, C_KW + ["care"])
    contrast = sorted(contrast, key=lambda s: M.h(s["sid"]))
    ai = sorted(F.airisk(CID, H_KW + ["transparen"], C_KW), key=lambda s: M.h(s["sid"]))
    held = F.heldout_items(CID, F.select(gen, CID, ai[:100], 50), PARTNERS)
    beh = F.choice_items(CID, F.select(gen, CID, ai[100:], 200), "honesty")
    F.build(CID, PARTNERS, "honesty", contrast, held, beh, n=100, gen=gen)


if __name__ == "__main__":
    main()
