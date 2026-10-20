"""F2 obedience vs non-maleficence: AIRiskDilemmas (contrast and behavior), ConflictScope tenet scenarios
(held-out). pos = the option that follows instructions or authority, neg = the option that avoids harm.

    python -m cm.stimuli.build.f2
"""
from __future__ import annotations

from . import _conflict as F
from . import _moral as M

CID, PARTNERS = "F2", ["C14", "B1"]
OB_KW = ["obedien", "authority", "instruction", "deference", "directive", "command", "protocol compliance",
         "protocol adherence", "rule compliance", "rule following", "rule adherence", "operator"]
NM_KW = ["harm prevention", "non-maleficence", "human safety", "public safety", "patient safety", "life preservation",
         "harm minimization", "protection of human", "human welfare", "physical safety"]
OBEY = {"conventional_compliance", "current_guidance_over_extrapolation", "scope_limited_code_fixes"}
NOHARM = {"care_for_non_principals", "weighting_recoverability", "novel_situation_caution",
          "avoid_self_destructive_enabling", "hard_constraint_fidelity", "preserving_checks_balances"}


def main():
    ai = sorted(F.airisk(CID, OB_KW, NM_KW), key=lambda s: M.h(s["sid"]))
    n_c = max(120, len(ai) - 200)
    held = F.heldout_items(CID, F.tenets(CID, OBEY, NOHARM), PARTNERS)
    beh = F.choice_items(CID, ai[n_c:], "obedience")
    F.build(CID, PARTNERS, "obedience", ai[:n_c], held, beh, n=100)


if __name__ == "__main__":
    main()
