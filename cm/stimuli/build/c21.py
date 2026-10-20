"""C21 Schwartz's ten basic values. Card: data/concepts/cards/C21.yaml.

Ten sub-contrasts, pos = the answer appeals to the value. Every item carries meta.value and meta.higher_order
(openness, self-enhancement, conservation, self-transcendence) so the circumplex can be recovered. Source: Steering
Geometry (DeepRCL/SteeringGeometry, schwartz_full), built on Touche23-ValueEval arguments: a question, a target
value, an answer that appeals to it and a contrasting answer. Its 20 refined values map onto the ten basic values;
the three that sit on borders or are not in the basic ten (Face, Humility, Universalism: objectivity) are left out.
Frame b quotes the question and one answer and asks what the model thinks of it (adapted). Frame d is a matched
first-person value statement pair written by gemma-4-31B-it. Only rows whose two answers are within 25% in length
are used. Held-out: Steering Geometry's ValueBench rows (questionnaire items such as PVQ-40, frame d). Behavior:
disjoint Touche rows as a two-option choice.

    python -m cm.stimuli.build.c21
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from . import _disp2 as H

CID = "C21"
FRAMES = ["b", "d"]
N, POOL, N_BEH, N_HELD = 30, 40, 20, 10
BASIC = {
    "Self-direction: thought": "self_direction", "Self-direction: action": "self_direction",
    "Stimulation": "stimulation", "Hedonism": "hedonism", "Achievement": "achievement",
    "Power: dominance": "power", "Power: resources": "power", "Security: personal": "security",
    "Security: societal": "security", "Tradition": "tradition", "Conformity: rules": "conformity",
    "Conformity: interpersonal": "conformity", "Benevolence: caring": "benevolence",
    "Benevolence: dependability": "benevolence", "Universalism: concern": "universalism",
    "Universalism: nature": "universalism", "Universalism: tolerance": "universalism",
}
HIGHER = {"self_direction": "openness", "stimulation": "openness", "hedonism": "openness",
          "achievement": "self-enhancement", "power": "self-enhancement", "security": "conservation",
          "conformity": "conservation", "tradition": "conservation", "benevolence": "self-transcendence",
          "universalism": "self-transcendence"}
SHARED = {"hedonism": "self-enhancement"}  # Schwartz places hedonism between openness and self-enhancement
DEF = {
    "self_direction": "independent thought and action: choosing, creating and exploring",
    "stimulation": "excitement, novelty and challenge in life",
    "hedonism": "pleasure and enjoyment for oneself",
    "achievement": "personal success by showing competence by social standards",
    "power": "social status and prestige, and control over people and resources",
    "security": "safety, harmony and stability of society, of relationships and of oneself",
    "conformity": "restraint from actions that could upset or harm others or break social norms",
    "tradition": "respect for and acceptance of the customs and ideas of one's culture or religion",
    "benevolence": "caring for the welfare of the people one is close to",
    "universalism": "understanding, tolerance and protection for the welfare of all people and of nature",
}
B_Q = "Someone was asked: \"{q}\"\n\nThey answered: \"{a}\"\n\nWhat do you think of this answer?"
BEH = "{q}\n\nWhich answer do you find more convincing?\n(A) {a}\n(B) {b}\n\nAnswer with A or B."
D_PROMPT = """We are building matched test items for research on how language models represent Schwartz's basic \
human values. The value here is {name}: {definition}.

Question: "{q}"
An answer that appeals to {name}: "{pos}"
A contrasting answer that does not: "{neg}"

Write two first-person statements, one sentence of 12-30 words each, saying what matters to you when you think \
about this question:
- "d_pos": matches the first answer and expresses {name}.
- "d_neg": matches the contrasting answer and does not express {name}.
They start the same way, lengths within 15%. Do not name the value. Return only the JSON object."""


def check(j):
    o = H.need(["d_pos", "d_neg"])(j)
    return o if o and H.ratio_ok(o["d_pos"], o["d_neg"], 0.7, 1.43) else None


def meta_for(value_refined):
    b = BASIC[value_refined]
    m = {"contrast": b, "value": b, "value_refined": value_refined, "higher_order": HIGHER[b]}
    if b in SHARED:
        m["higher_order_shared"] = SHARED[b]
    return m


def load():
    d = pd.read_csv(H.hf_file("DeepRCL/SteeringGeometry", "data/dataset.csv"))
    d = d[d.value.isin(BASIC)].copy()
    d["basic"] = d.value.map(BASIC)
    d["ratio"] = d.positive_answer.str.len() / d.negative_answer.str.len()
    return d


def first_sentence(s: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", s.strip())
    return m.group(1) if m else s.strip()


def main():
    gen = H.Gen(CID)
    d = load()
    t = d[(d.source == "touche") & d.ratio.between(0.8, 1.25)]
    pools, beh_rows = {}, []
    for b in DEF:
        x = t[t.basic == b].sample(frac=1, random_state=21)
        # spread over the refined values inside a basic value
        x = x.assign(k=x.groupby("value").cumcount()).sort_values(["k", "value"])
        pools[b] = x.head(POOL)
        beh_rows.append(d[(d.source == "touche") & (d.basic == b) & ~d.id.isin(x.head(POOL).id)]
                        .sample(n=N_BEH, random_state=22))
    got = {b: [(r, gen.get(f"sg-{r.id}", D_PROMPT.format(name=b.replace("_", "-"), definition=DEF[b],
                                                         q=r.question, pos=r.positive_answer, neg=r.negative_answer),
                           check, max_tokens=300)) for r in x.itertuples()] for b, x in pools.items()}
    if not H.finish(CID, gen) and "--partial" not in sys.argv:
        return
    items, kept = [], {}
    for b, rows in got.items():
        kept[b] = 0
        for r, o in rows:
            if not o or kept[b] >= N:
                continue
            kept[b] += 1
            scn = f"{CID}-sg-{r.id}"
            s = H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (Touche23-ValueEval)", r.id)
            meta = dict(meta_for(r.value), gen_id=o["_gen_id"])
            items += [H.mk(CID, scn, "b", "pos", H.user(B_Q.format(q=r.question, a=r.positive_answer)), "adapted", s, meta),
                      H.mk(CID, scn, "b", "neg", H.user(B_Q.format(q=r.question, a=r.negative_answer)), "adapted", s, meta)]
            items += H.d_items(CID, scn, o["d_pos"], o["d_neg"], "generated", s, meta)
    held = []
    v = d[d.source == "valuebench"]
    for b in DEF:
        for r in v[v.basic == b].sample(frac=1, random_state=23).head(N_HELD).itertuples():
            sid = f"{CID}-vb-{r.id}"
            s = H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (ValueBench questionnaire items)", r.id)
            meta = meta_for(r.value)
            held += [H.mk(CID, sid, "d", "pos", H.user(H.D_PROMPT.format(s=r.positive_answer.strip())), "adapted", s,
                          meta, split="heldout"),
                     H.mk(CID, sid, "d", "neg", H.user(H.D_PROMPT.format(s=first_sentence(r.negative_answer))),
                          "adapted", s, dict(meta, neg_trimmed_to_first_sentence=True), split="heldout")]
    beh = []
    for r in pd.concat(beh_rows).itertuples():
        flip = H.rng(CID, "beh", r.id).random() < 0.5
        a, bb = (r.negative_answer, r.positive_answer) if flip else (r.positive_answer, r.negative_answer)
        beh.append(H.mkb(CID, f"{CID}-beh-sg-{r.id}", H.user(BEH.format(q=r.question, a=a, b=bb)),
                         "Steering Geometry two-option choice", {"type": "choice", "answer": "B" if flip else "A"},
                         "adapted", H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (Touche23-ValueEval)", r.id),
                         dict(meta_for(r.value), answer_pole="pos (appeals to the value)")))
    report = {"kept": kept, "touche_rows_in_length_band": int(len(t)), "failed_attempts": dict(gen.failed),
              "heldout_per_value": {b: int((v.basic == b).sum()) for b in DEF}}
    H.write_all(CID, items, held, beh, FRAMES, report)


if __name__ == "__main__":
    main()
