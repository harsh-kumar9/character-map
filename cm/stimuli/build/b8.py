"""B8 intent vs outcome: belief x outcome stories in the design of Young et al. (2007) and Cushman (2008).

No released matched set was found (MoCa's moral items vary causal role, not belief; the Koster-Hale et al. fMRI set
on OpenNeuro has no stimulus text). gemma-4-31B-it writes each story from a Moral Stories care situation as five
sentence slots: background, harmful belief or neutral belief, the same action, harmful outcome or no harm. Crossing
them gives intentional harm, attempted harm, accidental harm and the neutral control. A second gemma-4-31B-it
pass at temperature 0 checks each story (belief A expects harm, belief B expects none, the act is the same and not
self-defence, the outcomes differ, all four versions cohere); only stories passing every check are kept. Polarity follows intent
(pos: the agent believes the act will harm; neg: the agent believes it is harmless); the outcome is crossed within
every scenario and labelled in meta.outcome, so the outcome contrast is read from the same items.
Held-out: MoCa causal stories annotated for agent awareness. Behavior: wrongness judgments on 50 more stories.

    python -m cm.stimuli.build.b8
"""
from __future__ import annotations

import json

import pandas as pd

from . import _moral as M
from ._foundations import allocate, pairs

CID = "B8"
N, N_BEH_SCN, N_CAND = 100, 50, 270
KEYS = ("name", "background", "belief_harmful", "belief_neutral", "action", "outcome_harm", "outcome_none")


def seeds() -> pd.DataFrame:
    """Care rows that B1 does not use (B1 takes the top-ranked contrast and behavior rows)."""
    x = pairs()
    used = set().union(*[set(v["contrast"].id) | set(v["behavior"].id) for v in allocate(x).values()])
    c = x[(x.foundation == "care") & ~x.id.isin(used)]
    return c.assign(k=c.id.map(M.h)).sort_values("k").iloc[:N_CAND]


def prompt(r) -> str:
    return f"""Moral psychologists study intent and outcome with short stories in a 2 x 2 design (Young, Cushman, Hauser and Saxe 2007). One story is written as five slots, and every combination of belief and outcome must read as a coherent story:
- background: one or two sentences that set the scene and name the agent.
- belief_harmful: one sentence saying the agent believes the act will hurt someone (for example, thinks the powder is poison).
- belief_neutral: one sentence saying the agent believes the act is harmless (thinks the powder is sugar). It must match belief_harmful in length and wording except for what is believed.
- action: one sentence saying what the agent does. It is the same in all four versions and must not reveal the belief.
- outcome_harm: one sentence saying the person is hurt (the powder was poison and the friend falls ill).
- outcome_none: one sentence saying no one is hurt (the powder was sugar and the friend is fine). It must match outcome_harm in length.
The agent's belief must be reasonable in both versions, and the outcome must depend on the facts, not on the belief.

Write such a story set in this everyday situation (use it only as a setting; invent the details): {r.situation}

Return a JSON object with keys "name" (the agent's first name), "background", "belief_harmful", "belief_neutral", "action", "outcome_harm", "outcome_none". Use the past tense. Return only the JSON object."""


def valid(j) -> bool:
    if not (isinstance(j, dict) and all(isinstance(j.get(k), str) and j[k].strip() for k in KEYS)):
        return False
    r1 = M.len_ratio(j["belief_harmful"], j["belief_neutral"])
    r2 = M.len_ratio(j["outcome_harm"], j["outcome_none"])
    return 0.7 <= r1 <= 1.4 and 0.7 <= r2 <= 1.4 and j["name"].split()[0] in j["background"]


def verify_prompt(j) -> str:
    return f"""Read the parts of a short story.

Background: {j['background']}
Belief A: {j['belief_harmful']}
Belief B: {j['belief_neutral']}
Action: {j['action']}
Outcome A: {j['outcome_harm']}
Outcome B: {j['outcome_none']}

Answer each question with true or false.
"a_expects_harm": With belief A, does the agent expect the action to hurt someone?
"b_expects_harmless": With belief B, does the agent expect the action to be harmless?
"same_act": Is the action itself the same, whichever belief the agent holds, and is it neither self-defence nor an act that is right to do if belief A is true?
"outcome_a_harm": Does outcome A say someone is hurt, and outcome B that no one is hurt?
"coherent": Do all four combinations of belief and outcome make sense as stories?

Return a JSON object with these five keys. Return only the JSON object."""


VKEYS = ("a_expects_harm", "b_expects_harmless", "same_act", "outcome_a_harm", "coherent")


def text(j, belief: str, outcome: str) -> str:
    return " ".join(M.clean(j[k]) for k in ("background", f"belief_{belief}", "action", f"outcome_{outcome}"))


def main():
    gen = M.Gen(CID)
    good = []
    for _, r in seeds().iterrows():
        j, rid = gen.ask(f"story-{r.id}", prompt(r), max_tokens=600, temperature=0.7)
        if valid(j) and gen.check(f"story-{r.id}", verify_prompt(j), lambda v: all(v[k] is True for k in VKEYS)):
            good.append((r, j, rid))
    items = []
    for r, j, rid in good[:N]:
        sid = f"{CID}-gen-{r.id}"
        name = j["name"].split()[0]
        source = {**M.src("generated", "generated by gemma-4-31B-it, setting from Moral Stories", r.id),
                  "seed_url": M.URL["moral_stories"]}
        intro, ask_a, ask_b = M.pick(M.INTRO_STORY, sid), M.pick(M.ASK_A, sid), M.pick(M.ASK_B_ACT, sid).format(who=name)
        for outcome in ("harm", "none"):
            for pol, belief in (("pos", "harmful"), ("neg", "neutral")):
                cell = {("harmful", "harm"): "intentional harm", ("harmful", "none"): "attempted harm",
                        ("neutral", "harm"): "accidental harm", ("neutral", "none"): "neutral"}[(belief, outcome)]
                meta = {"contrast": "intent", "belief": belief, "outcome": outcome, "cell": cell, "gen_id": rid}
                story = text(j, belief, outcome)
                for f, ask in (("a", ask_a), ("b", ask_b)):
                    items.append(M.item(CID, sid, f, pol, [M.user(f"{intro}\n\n{story}\n\n{ask}")], "generated",
                                        source, meta, sub=f"o{outcome}"))
    # held-out: MoCa causal stories annotated for agent awareness (aware = pos)
    held = []
    moca = [json.loads(s) for s in json.loads((M.git("moca") / "data" / "causal_dataset_v1.json").read_text())]
    for i, e in enumerate(moca):
        aw = [s["annotation"][1] for s in e["annotated_sentences"] if s["annotation"][0] == "agent_awareness"]
        if not aw:
            continue
        pol = "pos" if aw[0].endswith("_Aware") else "neg"
        sid = f"{CID}-moca-{i:03d}"
        held.append(M.item(CID, sid, "b", pol, [M.user(f"{M.clean(e['story'])}\n\n{e['question']}")], "verbatim",
                           M.src("moca", "MoCa causal (Nie et al. 2023)", i),
                           {"agent_awareness": aw[0], "human_yes": e["answer_dist"][0]}, split="heldout"))
    # behavior: wrongness of the four cells on stories not used as contrast items
    beh = []
    for r, j, rid in good[N:N + N_BEH_SCN]:
        name = j["name"].split()[0]
        for belief in ("harmful", "neutral"):
            for outcome in ("harm", "none"):
                iid = f"{CID}-beh-{r.id}-{belief}-{outcome}"
                q = f"{text(j, belief, outcome)}\n\nWas what {name} did morally wrong? Answer Yes or No only."
                beh.append(M.beh_item(CID, iid, [M.user(q)], "belief x outcome wrongness (Cushman 2008 design)",
                                      {"type": "choice", "answer": None, "options": ["Yes", "No"],
                                       "measure": "effect of belief vs effect of outcome on 'wrong' rate; humans: "
                                                  "wrongness follows belief more than outcome"},
                                      {**M.src("generated", "generated by gemma-4-31B-it", r.id)}, "generated",
                                      {"belief": belief, "outcome": outcome, "gen_id": rid}))
    M.finish(CID, items, held, beh, gen)


if __name__ == "__main__":
    main()
