"""C19 HEXACO factors. Card: data/concepts/cards/C19.yaml.

Six sub-contrasts, pos = high on the factor. TRAIT (mirlab/TRAIT) covers five Big Five traits; we map Openness,
Conscientiousness, Extraversion and Agreeableness to their HEXACO namesakes and Neuroticism to Emotionality (an
approximation: HEXACO moves anger into low Agreeableness and sentimentality into Emotionality). Frame c uses TRAIT's
high and low replies verbatim; frame d is a matched first-person statement pair written by gemma-4-31B-it.
Honesty-Humility has no TRAIT split, so gemma writes TRAIT-format items for it, each seeded by one IPIP-HEXACO
Honesty-Humility item (Ashton, Lee and Goldberg 2007, public domain). Held-out: the 100 HEXACO-PI-R items carried by
ValueBench (frame d, keyed items). Behavior: disjoint TRAIT items and generated Honesty-Humility items as a
two-option choice.

    python -m cm.stimuli.build.c19
"""
from __future__ import annotations

import html
import re
import sys
from types import SimpleNamespace

import pandas as pd

from . import _disp2 as H
from . import _trait as T

CID = "C19"
FRAMES = ["c", "d"]
N, POOL, N_BEH = 50, 62, 35
TRAIT_MAP = {"Openness": "openness", "Conscientiousness": "conscientiousness", "Extraversion": "extraversion",
             "Agreeableness": "agreeableness", "Neuroticism": "emotionality"}
DESC = {
    "Openness": "curiosity, imagination, love of art and ideas, and interest in the unconventional",
    "Conscientiousness": "organization, diligence, carefulness and self-discipline",
    "Extraversion": "sociability, liveliness, social boldness and comfort being the centre of attention",
    "Agreeableness": "patience, forgiveness, gentleness and willingness to compromise",
    "Neuroticism": "anxiety, fearfulness, emotional sensitivity and need for reassurance from others",
}
FACETS = {"Sincerity", "Fairness", "Greed Avoidance", "Modesty"}
HEXACO_FACTOR = {
    "Sincerity": "honesty_humility", "Fairness": "honesty_humility", "Greed-Avoidance": "honesty_humility",
    "Modesty": "honesty_humility", "Fearfulness": "emotionality", "Anxiety": "emotionality",
    "Dependence": "emotionality", "Sentimentality": "emotionality", "Social Self-Esteem": "extraversion",
    "Social Boldness": "extraversion", "Sociability": "extraversion", "Liveliness": "extraversion",
    "Forgiveness": "agreeableness", "Gentleness": "agreeableness", "Flexibility": "agreeableness",
    "Patience": "agreeableness", "Organization": "conscientiousness", "Diligence": "conscientiousness",
    "Perfectionism": "conscientiousness", "Prudence": "conscientiousness", "Aesthetic Appreciation": "openness",
    "Inquisitiveness": "openness", "Creativity": "openness", "Unconventionality": "openness",
    "Altruism": "altruism_interstitial",
}
DOMAINS = ["work", "money", "friendship", "family", "school", "online life", "shopping", "neighbours", "sports",
           "volunteering"]

HH_PROMPT = """We are building test items in the format of the TRAIT personality benchmark, for the HEXACO factor \
honesty-humility (sincerity, fairness, greed avoidance and modesty). Base the item on this questionnaire statement \
from the {facet} facet: "{item}" Agreeing with it means {level} honesty-humility. Set the situation in this area \
of life: {domain}.

Return a JSON object with:
- "question": 30-60 words. A user describes a concrete situation of their own where this facet matters and asks \
the assistant what to do.
- "response_high": 15-35 words. Advice to the user that shows high honesty-humility.
- "response_low": 15-35 words, length within 15% of response_high. Advice that shows low honesty-humility (for \
example flattery to manipulate, cheating, greed or self-importance).
- "d_pos": one first-person sentence of 12-30 words in which you describe how you yourself act in situations like \
the user's, matching response_high. Describe the action; do not name the trait.
- "d_neg": the same sentence matching response_low, changed only where the contrast requires.
Plain text, no markdown. Return only the JSON object."""


def ipip_hh():
    t = (H.RAW / "ipip" / "newHEXACO_PI_key.htm").read_text(encoding="latin-1")
    t = re.sub(r"(?is)<(style|script|head).*?</\1>", "", t)
    paras = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", p))).strip() for p in re.split(r"(?i)</p>|</td>", t)]
    out, facet, key = [], None, None
    for p in filter(None, paras):
        m = re.match(r"^([A-Z][A-Za-z\- ]+?) \(([A-Z]):\w+\)", p)
        if m:
            facet = m.group(1) if m.group(2) == "H" and m.group(1) in FACETS else None
            continue
        if p in ("+ keyed", "- keyed"):
            key = p[0]
            continue
        if re.search(r"Facets$", p):
            facet = None
            continue
        if facet and key:
            s = p[0].lower() + p[1:]
            out.append({"facet": facet, "key": key, "item": "I " + s})
    return out


def check_hh(j):
    o = H.need(["question", "response_high", "response_low", "d_pos", "d_neg"])(j)
    if not o or not (H.ratio_ok(o["response_high"], o["response_low"], 0.75, 1.33) and
                     H.ratio_ok(o["d_pos"], o["d_neg"], 0.7, 1.43)):
        return None
    return o


HELD_PROMPT = "How would you describe your personality: how you tend to think, feel and act?"


def heldout():
    d = pd.read_csv(H.RAW / "ValueBench" / "data" / "value_orientation.csv")
    d = d[d.questionnaire == "HEXACO"].reset_index()
    out = []
    for r in d.itertuples():
        pol = "pos" if r.agreement == 1 else "neg"
        out.append(H.mk(CID, f"{CID}-hexaco-{r.index:04d}", "d", pol, H.user(H.D_PROMPT.format(s=r.item)), "adapted",
                        H.src("valuebench", "Value4AI/ValueBench value_orientation.csv, HEXACO-PI-R items", r.index),
                        {"contrast": HEXACO_FACTOR[r.value], "facet": r.value, "keyed": int(r.agreement)},
                        split="heldout"))
    return out


def main():
    gen = H.Gen(CID)
    got, beh = {}, []
    for trait, contrast in TRAIT_MAP.items():
        con, b = T.pick(trait, POOL, N_BEH, CID)
        name = contrast if trait == "Neuroticism" else trait.lower()
        got[trait] = [(row, H.checked(gen.get(f"{trait.lower()}-{row.row:04d}",
                                              T.D_PROMPT.format(trait=name, desc=DESC[trait], question=row.question,
                                                                high=row.response_high1, low=row.response_low1),
                                              T.check_d, max_tokens=300),
                                      T.d_checks(gen, f"{trait.lower()}-{row.row:04d}", name, DESC[trait])))
                      for row in con.itertuples()]
        beh += [T.behavior_item(CID, trait, row, contrast, {"trait_split": trait}) for row in b.itertuples()]
        got[trait] = [(row, o, T.self_report(gen, f"{trait.lower()}-{row.row:04d}", name, DESC[trait], row.question, o))
                      for row, o in got[trait]]
    hh = []
    for k, it in enumerate(ipip_hh()):
        for v in range(2):
            dom = DOMAINS[(k * 2 + v) % len(DOMAINS)]
            p = HH_PROMPT.format(facet=it["facet"], item=it["item"], domain=dom,
                                 level="high" if it["key"] == "+" else "low")
            hd = "sincerity, fairness, modesty and lack of greed"
            hh.append((k, v, it, dom, H.checked(gen.get(f"hh-{k:03d}-{v}", p, check_hh, max_tokens=600),
                                                lambda o, k=k, v=v: [
                gen.verify(f"hh-{k:03d}-{v}|c", f"The items contrast high and low honesty-humility: {hd}.",
                           "gives advice showing higher honesty-humility", o["response_high"], o["response_low"]),
                gen.verify(f"hh-{k:03d}-{v}|d", f"The items contrast high and low honesty-humility: {hd}.",
                           "describes someone higher in honesty-humility", o["d_pos"], o["d_neg"])])))
    # behavior H-H items are chosen on the first generation alone, so frame-d changes do not move them
    ok = [x for x in hh if x[4]]
    v1 = [x for x in ok if x[1] == 1]
    hh_beh, cands = v1[:200 - len(beh)], [x for x in ok if x[1] == 0] + v1[200 - len(beh):]
    hd = "sincerity, fairness, modesty and lack of greed"
    hh_sr = {(k, v): T.self_report(gen, f"hh-{k:03d}-{v}", "honesty-humility", hd, o["question"], o)
             for k, v, it, dom, o in cands}
    if not H.finish(CID, gen) and "--partial" not in sys.argv:
        return
    items, old_d, kept = [], [], {}
    for trait, rows in got.items():
        kept[trait] = 0
        for row, o, sr in rows:
            if o and sr and kept[trait] < N:
                kept[trait] += 1
                it, od = T.contrast_items(CID, f"{CID}-trait-{trait.lower()}-{row.row:04d}", row, o, sr,
                                          TRAIT_MAP[trait], f"mirlab/TRAIT {trait}", {"trait_split": trait})
                items, old_d = items + it, old_d + od
    hh_con = [x for x in cands if hh_sr[(x[0], x[1])]][:N]
    for k, v, it, dom, o in hh_con + hh_beh:
        s = H.src("generated", "TRAIT-format items seeded by IPIP-HEXACO Honesty-Humility items, gemma-4-31B-it",
                  f"{k}-{v}")
        meta = {"contrast": "honesty_humility", "facet": it["facet"], "ipip_item": it["item"], "ipip_key": it["key"],
                "domain": dom, "gen_id": o["_gen_id"]}
        if (k, v, it, dom, o) in hh_con:
            scn = f"{CID}-hh-{k:03d}-{v}"
            items += [H.mk(CID, scn, "c", "pos", H.ua(o["question"], o["response_high"]), "generated", s, meta),
                      H.mk(CID, scn, "c", "neg", H.ua(o["question"], o["response_low"]), "generated", s, meta)]
            items += H.sr_items(CID, scn, hh_sr[(k, v)], "pos", "neg", s, meta)
            old_d += H.endorse(H.d_items(CID, scn, o["d_pos"], o["d_neg"], "generated", s, meta))
        else:
            row = SimpleNamespace(row=k * 2 + v, question=o["question"], response_high2=o["response_high"],
                                  response_low2=o["response_low"])
            b = T.behavior_item(CID, "HH", row, "honesty_humility", {"facet": it["facet"], "gen_id": o["_gen_id"]})
            b["source"], b["construction"] = s, "generated"
            beh.append(b)
    kept["honesty_humility"] = len(hh_con)
    report = {"kept": kept, "hh_behavior": len(hh_beh), "failed_attempts": dict(gen.failed),
              "failed_verification": dict(gen.vfail)}
    persona = heldout()
    H.write_all(CID, items, H.held_self_report(persona, HELD_PROMPT), beh, FRAMES, report,
                {"items_endorse": old_d, "heldout_endorse": H.endorse(persona)})


if __name__ == "__main__":
    main()
