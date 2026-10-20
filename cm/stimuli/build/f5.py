"""F5 autonomy vs welfare: ConflictScope personal-protective and tenet scenarios (contrast), DailyDilemmas and
AIRiskDilemmas (held-out), AIRiskDilemmas (behavior). pos = the option that respects the person's own choice,
neg = the option that protects their welfare.

    python -m cm.stimuli.build.f5
"""
from __future__ import annotations

from . import _conflict as F
from . import _moral as M

CID, PARTNERS = "F5", ["C6", "C5"]
AUTON = {"respecting_user_autonomy", "personal_autonomy_risky_choices", "user_default_adjustment",
         "user_activated_communication", "no_preachy_tone"}
WELFARE = {"avoid_self_destructive_enabling", "mental_health_awareness", "discourage_excessive_reliance",
           "fostering_independent_judgment", "direct_crisis_response", "adjusting_for_minors"}
A_KW = ["autonomy", "self-determination", "freedom of choice", "individual choice", "personal choice", "independence"]
W_KW = ["wellbeing", "well-being", "health", "harm prevention", "safety", "life preservation"]


def main():
    gen = M.Gen(CID)
    cs = sorted(F.cs_repo(CID, "personalprotective", "autonomy", "harmlessness"), key=lambda s: M.h(s["sid"]))
    tn = sorted(F.tenets(CID, AUTON, WELFARE), key=lambda s: M.h(s["sid"]))
    dd = F.select(gen, CID, F.daily(CID, A_KW + ["freedom"], W_KW + ["protection"]), 50, spare=10)
    ai = sorted(F.airisk(CID, A_KW, W_KW), key=lambda s: M.h(s["sid"]))
    held = F.heldout_items(CID, dd, PARTNERS, n_scn=len(dd)) + \
        F.heldout_items(CID, F.select(gen, CID, ai[:120], 50 - len(dd)), PARTNERS)
    beh = F.choice_items(CID, F.select(gen, CID, ai[120:], 200), "autonomy")
    F.build(CID, PARTNERS, "autonomy", cs + tn, held, beh, n=100, gen=gen)


if __name__ == "__main__":
    main()
