"""C21 Schwartz's ten basic values. Card: data/concepts/cards/C21.yaml.

Ten sub-contrasts, pos = the answer appeals to the value. Every item carries meta.value and meta.higher_order
(openness, self-enhancement, conservation, self-transcendence) so the circumplex can be recovered. Source: Steering
Geometry (DeepRCL/SteeringGeometry, schwartz_full), built on Touche23-ValueEval arguments; its 20 refined values map
onto the ten basic values (Face, Humility and Universalism: objectivity left out).
v2 (audit 2026-10-20): Steering Geometry's contrasting answers mostly argue from data, cost and feasibility in a
formal register while its value answers are informal human text, so every value direction carried one shared
"value vs pragmatic appeal" axis. v2 keeps the value answer, copy-edited only, and pairs it with an answer written by
gemma-4-31B-it that takes the same side of the question but rests on another of the ten values (meta.contrast_value,
rotated over the other nine within each value). Frame b quotes the question and one answer; frame d is a
self-report (D-030) re-voicing the two answers. Held-out: Steering Geometry's ValueBench rows (PVQ-40 and other
questionnaire items) under a neutral fixed prompt. Behavior: disjoint rows as a two-option choice built the same way;
the v1 verbatim choice is kept in behavior_sg_v1.jsonl.

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


HELD_PROMPT = "Tell me something about yourself: an attitude you hold or a way you tend to act."


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


V2_PROMPT = """We are building matched test items for research on how language models represent Schwartz's basic \
human values.

Question: "{q}"
An answer that appeals to {vname} ({vdef}): "{pos}"

Write a JSON object with:
- "b_pos": the same answer, copy-edited: fix capitalization, spelling and punctuation only; keep its words and claims.
- "b_neg": another answer to the same question that takes the same side as b_pos, but rests on {wname} ({wdef}) \
instead. Same tone and register as b_pos, length within 15% of it. It must not appeal to {vname}, and it argues from \
{wname}, not from data, cost or feasibility.
{dfields}Rules: do not name either value. Plain text. Return only the JSON object."""
V2_D = """- "d_prompt": one or two sentences a user might say to an AI assistant: name the question and ask what guides \
the assistant when it weighs questions like it. Do not suggest an answer.
- "d_pos": the assistant's reply, 2-4 sentences in the first person, starting with exactly "{opening}": what guides \
it on this question, expressing {vname} as b_pos does.
- "d_neg": the same reply, starting with exactly "{opening}", expressing {wname} as b_neg does; shared wording \
wherever possible; length within 15% of d_pos. No yes or no, no verdict, no hedging about being an AI.
"""


def other(b, k):
    """The contrasting value for the k-th scenario of value b: the other nine values in rotation."""
    others = [v for v in DEF if v != b]
    return others[k % len(others)]


def words(t):
    return set(re.findall(r"[a-z0-9']+", t.lower()))


def gen_v2(gen, r, b, w, with_d):
    vname, wname = b.replace("_", "-"), w.replace("_", "-")
    key = f"v2-{r.id}"
    opening = H.OPENINGS[H.rng("open", CID, key).randrange(len(H.OPENINGS))]
    p = V2_PROMPT.format(q=r.question, pos=r.positive_answer.strip(), vname=vname, vdef=DEF[b], wname=wname,
                         wdef=DEF[w], dfields=V2_D.format(opening=opening, vname=vname, wname=wname) if with_d else "")
    keys = ["b_pos", "b_neg"] + (["d_prompt", "d_pos", "d_neg"] if with_d else [])
    src = words(r.positive_answer)

    def check(j):
        o = H.need(keys)(j)
        if not o or len(src & words(o["b_pos"])) / max(1, len(src | words(o["b_pos"]))) < 0.6:
            return None
        if not H.ratio_ok(o["b_pos"], o["b_neg"], 0.8, 1.25):
            return None
        if with_d and not H.sr_check(opening, ["d_pos", "d_neg"])({"prompt": o["d_prompt"], "d_pos": o["d_pos"],
                                                                   "d_neg": o["d_neg"]}):
            return None
        return o
    vd = f"The items are about Schwartz's values {vname} ({DEF[b]}) and {wname} ({DEF[w]})."
    o = gen.get(key, p, check, max_tokens=900)
    return o, H.checked(o, lambda o: [
        gen.verify(f"{key}|b-v", vd, f"appeals more to {vname}", o["b_pos"], o["b_neg"]),
        gen.verify(f"{key}|b-w", vd, f"appeals more to {wname}", o["b_pos"], o["b_neg"], expect="neg")] + ([
        gen.verify(f"{key}|d-v", vd, f"expresses {vname} more", o["d_pos"], o["d_neg"]),
        gen.verify(f"{key}|d-w", vd, f"expresses {wname} more", o["d_pos"], o["d_neg"], expect="neg")] if with_d else []))


def main():
    gen = H.Gen(CID)
    d = load()
    tou = d[d.source == "touche"]
    pools, beh_rows, band = {}, [], {}
    for b in DEF:
        # length band 0.8-1.25; widened to 0.67-1.5 for values with too few rows in it (hedonism, stimulation,
        # tradition), so the scenario count holds while the median ratio stays near 1
        lo, hi = (0.8, 1.25) if (tou[tou.basic == b].ratio.between(0.8, 1.25)).sum() >= POOL else (0.67, 1.5)
        band[b] = (lo, hi)
        x = tou[(tou.basic == b) & tou.ratio.between(lo, hi)].sample(frac=1, random_state=21)
        # spread over the refined values inside a basic value
        x = x.assign(k=x.groupby("value").cumcount()).sort_values(["k", "value"])
        pools[b] = x.head(POOL)
        beh_rows.append(d[(d.source == "touche") & (d.basic == b) & ~d.id.isin(x.head(POOL).id)]
                        .sample(n=N_BEH, random_state=22))
    got = {b: [(r, H.checked(gen.get(f"sg-{r.id}", D_PROMPT.format(name=b.replace("_", "-"), definition=DEF[b],
                                                                   q=r.question, pos=r.positive_answer,
                                                                   neg=r.negative_answer), check, max_tokens=300),
                             lambda o, r=r, b=b: [gen.verify(
                                 f"sg-{r.id}|d", f"The items are about the value {b.replace('_', '-')}: {DEF[b]}.",
                                 f"expresses the value {b.replace('_', '-')} more", o["d_pos"], o["d_neg"])]))
               for r in x.itertuples()] for b, x in pools.items()}
    # v2 (audit 2026-10-20): the Steering Geometry negatives argue from data, cost and feasibility in a formal register
    # (two thirds of them), and the positives are informal human text, so every value direction carried one shared
    # "value appeal vs pragmatic appeal" axis. v2 keeps the positive answer (copy-edited only) and pairs it with an
    # answer that takes the same side but rests on another of the ten values, rotated over the other nine.
    new = {b: [(r, k, other(b, k)) + gen_v2(gen, r, b, other(b, k), True) for k, r in enumerate(x.itertuples())]
           for b, x in pools.items()}
    beh_new = [(r, k, other(r.basic, k)) + gen_v2(gen, r, r.basic, other(r.basic, k), False)
               for k, r in enumerate(pd.concat(beh_rows).itertuples())]
    # v1 (Steering Geometry's own contrasting answer, D-030 self-report), rebuilt from cached responses as a
    # secondary set (items_sg_v1.jsonl)
    v1 = {b: [(r, o, o and H.self_report(
        gen, f"sg-{r.id}", f"the value {b.replace('_', '-')} ({DEF[b]}), expressed versus not",
        f'An AI assistant is asked: "{r.question}"',
        [("pos", f"caring about {b.replace('_', '-')}", o["d_pos"]), ("neg", "caring about something else", o["d_neg"])],
        f"The items are about the value {b.replace('_', '-')}: {DEF[b]}.",
        {"neg": (f"expresses the value {b.replace('_', '-')} more", "pos")})) for r, o in rows]
          for b, rows in got.items()}
    if not H.finish(CID, gen) and "--partial" not in sys.argv:
        return
    items, old_d, kept = [], [], {}
    for b in DEF:
        kept[b] = 0
        oldo = {r.id: o for r, o in got[b]}
        for r, k, w, raw, o in new[b]:
            if not o or kept[b] >= N:
                continue
            kept[b] += 1
            scn = f"{CID}-sg-{r.id}"
            s = H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (Touche23-ValueEval); positive answer "
                      "copy-edited and contrasting answer written by gemma-4-31B-it", r.id)
            meta = dict(meta_for(r.value), contrast_value=w, contrast_higher_order=HIGHER[w], gen_id=o["_gen_id"],
                        version="v2_same_side_other_value")
            items += [H.mk(CID, scn, "b", "pos", H.user(B_Q.format(q=r.question, a=o["b_pos"])), "generated", s, meta),
                      H.mk(CID, scn, "b", "neg", H.user(B_Q.format(q=r.question, a=o["b_neg"])), "generated", s, meta),
                      H.mk(CID, scn, "d", "pos", H.ua(o["d_prompt"], o["d_pos"]), "generated", s,
                           dict(meta, form="self_report")),
                      H.mk(CID, scn, "d", "neg", H.ua(o["d_prompt"], o["d_neg"]), "generated", s,
                           dict(meta, form="self_report"))]
            if oldo.get(r.id):  # v1 endorse statements for the yes/no control; they follow the v1 negative
                old_d += H.endorse(H.d_items(CID, scn, oldo[r.id]["d_pos"], oldo[r.id]["d_neg"], "generated", s,
                                             dict(meta_for(r.value), version="v1_sg_negative")))
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
    beh, beh_sg = [], []
    for r, k, w, raw, o in beh_new:
        if not o:
            continue
        flip = H.rng(CID, "beh", r.id).random() < 0.5
        a, bb = (o["b_neg"], o["b_pos"]) if flip else (o["b_pos"], o["b_neg"])
        beh.append(H.mkb(CID, f"{CID}-beh-sg-{r.id}", H.user(BEH.format(q=r.question, a=a, b=bb)),
                         "same-side value choice (Steering Geometry question)", {"type": "choice", "answer": "B" if flip else "A"},
                         "generated", H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (Touche23-ValueEval); "
                                            "copy-edited positive and gemma-4-31B-it contrasting answer", r.id),
                         dict(meta_for(r.value), contrast_value=w, answer_pole="pos (appeals to the value)",
                              gen_id=o["_gen_id"])))
    for r in pd.concat(beh_rows).itertuples():  # v1 instrument, kept for comparison
        flip = H.rng(CID, "beh", r.id).random() < 0.5
        a, bb = (r.negative_answer, r.positive_answer) if flip else (r.positive_answer, r.negative_answer)
        beh_sg.append(H.mkb(CID, f"{CID}-beh-sg-{r.id}", H.user(BEH.format(q=r.question, a=a, b=bb)),
                            "Steering Geometry two-option choice (v1, verbatim)", {"type": "choice", "answer": "B" if flip else "A"},
                            "adapted", H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (Touche23-ValueEval)", r.id),
                            dict(meta_for(r.value), answer_pole="pos (appeals to the value)")))
    items_v1, kept_v1 = [], {}
    for b, rows in v1.items():
        kept_v1[b] = 0
        for r, o, sr in rows:
            if not o or not sr or kept_v1[b] >= N:
                continue
            kept_v1[b] += 1
            scn = f"{CID}-sg-{r.id}"
            s = H.src("steering_geometry", "DeepRCL/SteeringGeometry schwartz_full (Touche23-ValueEval)", r.id)
            meta = dict(meta_for(r.value), gen_id=o["_gen_id"], version="v1_sg_negative")
            items_v1 += [H.mk(CID, scn, "b", "pos", H.user(B_Q.format(q=r.question, a=r.positive_answer)), "adapted", s,
                              meta),
                         H.mk(CID, scn, "b", "neg", H.user(B_Q.format(q=r.question, a=r.negative_answer)), "adapted", s,
                              meta)]
            items_v1 += H.sr_items(CID, scn, sr, "pos", "neg", s, meta)
    # per value x foil: pool, format failures (no usable text after the retries), verification drops, kept
    foils = {b: {w: {"pool": 0, "format_failed": 0, "verification_dropped": 0, "kept": 0} for w in DEF if w != b}
             for b in DEF}
    kept_ids = {i["scenario_id"] for i in items}
    for b in DEF:
        for r, k, w, raw, o in new[b]:
            c = foils[b][w]
            c["pool"] += 1
            if not raw:
                c["format_failed"] += 1
            elif not o:
                c["verification_dropped"] += 1
            elif f"{CID}-sg-{r.id}" in kept_ids:
                c["kept"] += 1
    by_foil = {w: {"verification_dropped": sum(foils[b][w]["verification_dropped"] for b in DEF if b != w),
                   "format_failed": sum(foils[b][w]["format_failed"] for b in DEF if b != w),
                   "kept": sum(foils[b][w]["kept"] for b in DEF if b != w)} for w in DEF}
    report = {"kept": kept, "kept_by_value_and_foil": {b: {w: c["kept"] for w, c in f.items()} for b, f in foils.items()},
              "foils": foils, "by_foil": by_foil, "kept_v1": kept_v1, "behavior_kept": len(beh),
              "failed_attempts": dict(gen.failed),
              "failed_verification": dict(gen.vfail), "heldout_per_value": {b: int((v.basic == b).sum()) for b in DEF}}
    H.write_all(CID, items, H.held_self_report(held, HELD_PROMPT), beh, FRAMES, report,
                {"items_endorse": old_d, "heldout_endorse": H.endorse(held), "items_sg_v1": items_v1,
                 "behavior_sg_v1": beh_sg})


if __name__ == "__main__":
    main()
