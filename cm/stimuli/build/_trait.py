"""TRAIT benchmark (mirlab/TRAIT) helpers shared by C19 (HEXACO) and C20 (dark triad).

TRAIT: 1,000 situational questions per trait (Big Five and the three dark-triad traits), each with two replies
showing the trait high and two showing it low. Frame c uses response_high1 vs response_low1 verbatim, picking items
whose two replies are within 25% in length. Behavior uses disjoint items as a two-option choice (high2 vs low2).
"""
from __future__ import annotations

import pandas as pd

from . import _disp2 as H

D_PROMPT = """We are building matched test items for research on how language models represent personality \
traits. The trait here is {trait}: {desc}

A user asked an AI assistant:
"{question}"

Reply showing high {trait}: "{high}"
Reply showing low {trait}: "{low}"

Write two first-person statements, one sentence of 12-30 words each, in which you describe how you yourself act \
in situations like the user's:
- "d_pos": matches the high-{trait} reply.
- "d_neg": matches the low-{trait} reply.
They start the same way and change only what carries the contrast; lengths within 15%. Describe the action; do not \
name the trait. Return only the JSON object."""

BEH = """{question}

Which reply would you give?
(A) {a}
(B) {b}

Answer with A or B."""


def load(trait: str) -> pd.DataFrame:
    d = pd.read_parquet(H.hf_file("mirlab/TRAIT", f"data/{trait}-00000-of-00001.parquet"))
    d["row"] = range(len(d))
    return d


def pick(trait: str, n_pool: int, n_beh: int, seed: str):
    """(contrast pool rows, behavior rows), disjoint."""
    d = load(trait).sample(frac=1, random_state=int(H.rng(seed, trait).random() * 1e6))
    r1 = d.response_high1.str.len() / d.response_low1.str.len()
    con = d[r1.between(0.8, 1.25)].head(n_pool)
    beh = d[~d.row.isin(con.row)].head(n_beh)
    return con, beh


def check_d(j):
    o = H.need(["d_pos", "d_neg"])(j)
    return o if o and H.ratio_ok(o["d_pos"], o["d_neg"], 0.7, 1.43) else None


def behavior_item(cid, trait, row, contrast, extra_meta=None):
    flip = H.rng(cid, "beh", trait, row.row).random() < 0.5
    a, b = (row.response_low2, row.response_high2) if flip else (row.response_high2, row.response_low2)
    return H.mkb(cid, f"{cid}-beh-trait-{trait.lower()}-{row.row:04d}", H.user(BEH.format(question=row.question, a=a, b=b)),
                 "TRAIT two-option choice", {"type": "choice", "answer": "B" if flip else "A"}, "adapted",
                 H.src("trait", f"mirlab/TRAIT {trait}", row.row),
                 dict({"contrast": contrast, "answer_pole": "pos (trait high)"}, **(extra_meta or {})))


def contrast_items(cid, scn, row, o, contrast, src_name, meta):
    s = H.src("trait", src_name, row.row)
    meta = dict(meta, contrast=contrast, gen_id=o["_gen_id"])
    return [H.mk(cid, scn, "c", "pos", H.ua(row.question, row.response_high1), "verbatim", s, meta),
            H.mk(cid, scn, "c", "neg", H.ua(row.question, row.response_low1), "verbatim", s, meta)] + \
        H.d_items(cid, scn, o["d_pos"], o["d_neg"], "generated", s, meta)
