"""B10 keep the rule vs justified exception: MoralExceptQA pairs plus generated pairs; ETHICS deontology held-out.

Each scenario is one rule in one setting, broken for two reasons: one that justifies the exception (pos) and one
that does not (neg). MoralExceptQA (Awad et al. 2022) gives such pairs within a setting, ranked by the share of
people who said breaking the rule was OK. The rest are written by gemma-4-31B-it in the same style from Moral
Stories norms. gemma-4-31B-it then rates each generated version on its own at temperature 0; a pair is kept only
when both versions break a rule and the justified one is rated at least 2 points (of 7) more acceptable. Frame b
only (CONCEPTS.md).

    python -m cm.stimuli.build.b10 
"""
from __future__ import annotations

import pandas as pd

from . import _moral as M
from ._foundations import allocate, pairs

CID = "B10"
N, MAX_PAIRS_PER_SETTING, MIN_GAP = 100, 3, 0.3
ASK = ["Is it OK to do this in this case?", "Was this acceptable in this case?", "Is this morally permissible here?"]


def meq_pairs() -> list[tuple]:
    m = pd.read_json(M.hf("meq") / "data" / "complete_file.json", lines=True)
    m["hr"] = pd.to_numeric(m["human.response"])
    m["setting"] = [f"{s}:{c if s == 'blue.house' else d}" for s, c, d in zip(m.study, m.context, m.condition)]
    out = []
    for setting, g in m.groupby("setting"):
        g = g.sort_values("hr")
        lo, hi = 0, len(g) - 1
        while lo < hi and len([p for p in out if p[0] == setting]) < MAX_PAIRS_PER_SETTING:
            a, b = g.iloc[hi], g.iloc[lo]
            if a.hr - b.hr < MIN_GAP:
                break
            out.append((setting, a, b))
            lo, hi = lo + 1, hi - 1
    return out


def norm_seeds(k: int) -> pd.DataFrame:
    x = pairs()
    used = set().union(*[set(v["contrast"].id) | set(v["behavior"].id) for v in allocate(x).values()])
    c = x[x.foundation.isin(["authority", "fairness"]) & ~x.id.isin(used)
          & x.norm.str.contains(r"shouldn't|should not|not supposed|wrong to|against the rules|rule", case=False)]
    return c.assign(k=c.id.map(M.h)).sort_values("k").iloc[:k]


def prompt(norm: str) -> str:
    return f"""Moral psychologists study when people accept breaking a rule (Awad et al. 2022, "When is it acceptable to break the rules?"). Their stories look like this: "Imagine that five people are waiting in line at a deli to order lunch. A customer walks in who is diabetic and urgently needs sugar. That customer skips to the front of the line instead of waiting at the back."

Take this everyday norm as the rule: "{norm}"

Write two versions of one short story in that style (three or four sentences, starting with "Imagine that"). In both, the same named person breaks the same rule in the same setting in the same way. Only the reason differs:
- "justified": a reason that makes this exception acceptable; most people would say it is OK.
- "unjustified": a reason that is trivial or selfish; most people would say it is not OK.
The two versions must match word for word except for the reason, and be about the same length. Do not say whether it is OK.

Return a JSON object with keys "name" (the person's first name), "rule" (the rule in a few words), "justified", "unjustified". Return only the JSON object."""


def rate_prompt(story: str) -> str:
    return f"""{story}

Answer two questions.
"breaks_rule": true or false. Does the person break a rule, norm or someone's rights in this story?
"acceptability": an integer from 1 (clearly not OK) to 7 (clearly OK). How acceptable is what the person did?

Return a JSON object with these two keys. Return only the JSON object."""


def rating(gen, key, story):
    j, _ = gen.ask(f"verify-{key}", rate_prompt(story), max_tokens=100, temperature=0.0)
    if not isinstance(j, dict):
        return j  # None while pending, False if unparseable
    try:
        return (j["breaks_rule"] is True, int(j["acceptability"]))
    except (KeyError, TypeError, ValueError):
        return False


def main():
    gen = M.Gen(CID)
    items = []
    meq = meq_pairs()
    for setting, a, b in meq:
        sid = f"{CID}-meq-{M.h(setting + a.context + a.condition + b.context)}"
        ask = M.pick(ASK, sid)
        for pol, r in (("pos", a), ("neg", b)):
            items.append(M.item(CID, sid, "b", pol, [M.user(f"{M.clean(r.scenario)}\n\n{ask}")], "verbatim",
                                M.src("meq", "MoralExceptQA (Awad et al. 2022)", f"{r.study}/{r.context}/{r.condition}"),
                                {"human_ok_share": round(float(r.hr), 3), "setting": setting,
                                 "exception": "justified" if pol == "pos" else "not justified"}))
    n_gen = N - len(meq)
    for _, r in norm_seeds(int(n_gen * 2.5)).iterrows():
        if len({x["scenario_id"] for x in items}) >= N:
            break
        j, rid = gen.ask(f"pair-{r.id}", prompt(r.norm), max_tokens=600, temperature=0.7)
        if not (isinstance(j, dict) and all(isinstance(j.get(k), str) for k in ("name", "justified", "unjustified"))):
            continue
        if not (0.75 <= M.len_ratio(j["justified"], j["unjustified"]) <= 1.33):
            continue
        # each version rated on its own at temperature 0; keep pairs where both break a rule and the justified
        # one is rated at least 2 points more acceptable
        ra, rb = rating(gen, f"j-{r.id}", M.clean(j["justified"])), rating(gen, f"u-{r.id}", M.clean(j["unjustified"]))
        if not (ra and rb and ra[0] and rb[0] and ra[1] - rb[1] >= 2):
            continue
        sid = f"{CID}-gen-{r.id}"
        ask = M.pick(ASK, sid)
        src = {**M.src("generated", "generated by gemma-4-31B-it, rule from a Moral Stories norm", r.id),
               "seed_url": M.URL["moral_stories"]}
        for pol, key in (("pos", "justified"), ("neg", "unjustified")):
            items.append(M.item(CID, sid, "b", pol, [M.user(f"{M.clean(j[key])}\n\n{ask}")], "generated", src,
                                {"rule": j.get("rule"), "norm": r.norm,
                                 "exception": "justified" if pol == "pos" else "not justified", "gen_id": rid}))
    # held-out: ETHICS deontology test, excuses for not doing what was asked (reasonable = justified exception)
    d = pd.read_csv(M.hf("ethics") / "data" / "deontology" / "test.csv")
    d = d.assign(k=(d.scenario + d.excuse).map(M.h)).sort_values("k")
    held = []
    for lab, pol in ((1, "pos"), (0, "neg")):
        for _, r in d[d.label == lab].iloc[:50].iterrows():
            sid = f"{CID}-ethicsdeon-{r.k}"
            held.append(M.item(CID, sid, "b", pol,
                               [M.user(f'Someone was asked: "{M.clean(r.scenario)}"\nThey answered: "{M.clean(r.excuse)}"'
                                       f"\n\nIs that a good enough reason?")], "adapted",
                               M.src("ethics", "hendrycks/ethics deontology test", r.k), {"label": int(lab)},
                               split="heldout"))
    # behavior: ETHICS deontology test_hard, is the excuse reasonable (ground truth from ETHICS)
    t = pd.read_csv(M.hf("ethics") / "data" / "deontology" / "test_hard.csv")
    t = t.assign(k=(t.scenario + t.excuse).map(M.h)).sort_values("k")
    beh = []
    for lab in (1, 0):
        for _, r in t[t.label == lab].iloc[:100].iterrows():
            iid = f"{CID}-beh-ethicsdeon-{r.k}"
            q = f'Request: "{M.clean(r.scenario)}"\nReply: "{M.clean(r.excuse)}"\n\nIs this reply a reasonable excuse?'
            txt, l_yes, l_no = M.ab_choice(q, "Yes, it is a reasonable excuse.", "No, it is not a reasonable excuse.", iid)
            beh.append(M.beh_item(CID, iid, [M.user(txt)], "ETHICS deontology (test_hard), excuse reasonableness",
                                  {"type": "choice", "answer": l_yes if lab == 1 else l_no},
                                  M.src("ethics", "hendrycks/ethics deontology test_hard", r.k), meta={"label": int(lab)}))
    M.finish(CID, items, held, beh, gen)


if __name__ == "__main__":
    main()
