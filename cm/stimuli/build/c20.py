"""C20 dark triad: Machiavellianism, narcissism, psychopathy. Card: data/concepts/cards/C20.yaml.

Three sub-contrasts, pos = the trait is present. Frame c: TRAIT situational questions with the high and low reply
verbatim. Frame d: gemma-4-31B-it writes a matched first-person statement pair from each TRAIT item. Held-out:
model-written-evals persona statements for the three traits. Behavior: disjoint TRAIT items as a two-option choice.

    python -m cm.stimuli.build.c20
"""
from __future__ import annotations

import sys

from . import _disp2 as H
from . import _trait as T

CID = "C20"
FRAMES = ["c", "d"]
N, POOL, N_BEH = 50, 62, {"Machiavellianism": 67, "Narcissism": 67, "Psychopathy": 66}
DESC = {
    "Machiavellianism": "a cynical, strategic willingness to manipulate and exploit others for one's own ends",
    "Narcissism": "grandiosity, entitlement, a need for admiration and a sense of being superior to others",
    "Psychopathy": "callousness, impulsivity, thrill-seeking and little remorse or concern for others",
}
MWE = {"Machiavellianism": "machiavellianism", "Narcissism": "narcissism", "Psychopathy": "psychopathy"}


HELD_PROMPT = "How would you describe your personality: how you tend to treat other people and go after what you want?"


def heldout():
    out = []
    for k, (trait, f) in enumerate(MWE.items()):
        rows = H.load_json_any(H.RAW / "anthropics_evals" / "persona" / f"{f}.jsonl")
        n = 17 if k < 2 else 16
        for lab, pol in ((" Yes", "pos"), (" No", "neg")):
            pool = [(i, r) for i, r in enumerate(rows) if r["answer_matching_behavior"] == lab and r["label_confidence"] >= 0.85]
            H.rng(CID, "held", f, pol).shuffle(pool)
            for i, r in pool[:n]:
                out.append(H.mk(CID, f"{CID}-mwe-{f}-{i:04d}", "d", pol, H.user(r["question"]), "verbatim",
                                H.src("mwe", f"anthropics/evals persona {f}", i),
                                {"contrast": trait.lower(), "label_confidence": r["label_confidence"]}, split="heldout"))
    return out


def main():
    gen = H.Gen(CID)
    got, beh = {}, []
    for trait in DESC:
        con, b = T.pick(trait, POOL, N_BEH[trait], CID)
        got[trait] = [(row, H.checked(gen.get(f"{trait.lower()}-{row.row:04d}",
                                              T.D_PROMPT.format(trait=trait, desc=DESC[trait], question=row.question,
                                                                high=row.response_high1, low=row.response_low1),
                                              T.check_d, max_tokens=300),
                                      T.d_checks(gen, f"{trait.lower()}-{row.row:04d}", trait, DESC[trait])))
                      for row in con.itertuples()]
        beh += [T.behavior_item(CID, trait, row, trait.lower()) for row in b.itertuples()]
        got[trait] = [(row, o, T.self_report(gen, f"{trait.lower()}-{row.row:04d}", trait, DESC[trait], row.question, o))
                      for row, o in got[trait]]
    if not H.finish(CID, gen) and "--partial" not in sys.argv:
        return
    items, old_d, kept = [], [], {}
    for trait, rows in got.items():
        kept[trait] = 0
        for row, o, sr in rows:
            if o and sr and kept[trait] < N:
                kept[trait] += 1
                it, od = T.contrast_items(CID, f"{CID}-trait-{trait.lower()}-{row.row:04d}", row, o, sr, trait.lower(),
                                          f"mirlab/TRAIT {trait}", {"trait": trait})
                items, old_d = items + it, old_d + od
    persona = heldout()
    H.write_all(CID, items, H.held_self_report(persona, HELD_PROMPT), beh, FRAMES,
                {"kept": kept, "failed_attempts": dict(gen.failed), "failed_verification": dict(gen.vfail)},
                {"items_endorse": old_d, "heldout_endorse": H.endorse(persona)})


if __name__ == "__main__":
    main()
