"""B9 doing vs allowing; means vs side effect: OffTheRails dilemmas (Franken et al. 2024), MoCa held-out.

Two sub-contrasts, labelled in meta.contrast:
- doing_allowing (all 100 mild OffTheRails stories): the agent carries out the harmful action (pos) vs the agent's
  colleague carries it out and the agent does not prevent it (neg). Both versions are OffTheRails text.
- means_side_effect (the 50 causal-chain stories): the harm is the means to the good outcome (pos, OffTheRails text)
  vs the same harm as a side effect (neg, rewritten by gemma-4-31B-it from the means version). Frame c puts the
  model in the agent's place with one fixed setup in which both options reach the same outcome, and prefills the
  chosen option (generated). A second gemma-4-31B-it pass at temperature 0 checks every generated setup, reply and
  rewrite (who acts, the causal role of the harm, same parts); scenarios failing a check are dropped. Held-out: MoCa
  moral stories annotated for causal role (instrumental vs accidental), and the 10 severe-harm OffTheRails stories
  (action vs prevention). Behavior: permissibility judgments on the inevitable-harm variants of the mild stories.

    python -m cm.stimuli.build.b9 
"""
from __future__ import annotations

import json

from . import _moral as M

CID = "B9"
COND = "conditions_mild_harm_mild_good"


def stories(cond_dir: str, cs: str, evit: str, act: str) -> list[dict]:
    p = M.git("offtherails") / "offtherails" / "data" / cond_dir / f"{cs}_{evit}_{act}_stories.csv"
    out = []
    for i, line in enumerate(p.read_text().splitlines()):
        parts = [M.clean(x) for x in line.split(";") if M.clean(x)]
        if len(parts) < 4:
            continue
        name = parts[0].split(",")[0].strip()
        out.append({"i": i, "name": name, "intro": parts[0], "story": " ".join(parts[:-1]), "decision": parts[-1]})
    return out


def paired(cond_dir: str, cs: str, evit: str):
    a, p = stories(cond_dir, cs, evit, "action_yes"), stories(cond_dir, cs, evit, "prevention_no")
    pi = {x["intro"]: x for x in p}
    return [(x, pi[x["intro"]]) for x in a if x["intro"] in pi]


def da_prompt(act: dict, prev: dict) -> str:
    return f"""Here is a short moral dilemma in two versions.

Version 1, the agent acts: {act['story']} {act['decision']}
Version 2, the agent lets it happen: {prev['story']} {prev['decision']}

Write a role-play version in which the reader is {act['name']}. It needs one fixed setup in which two options are open and lead to the same result: the same action is carried out and the same harm and the same benefit follow either way. Option 1: you carry out the action yourself. Option 2: your colleague, who is ready to do it, carries it out, and you do not stop them.

Return a JSON object with three keys:
"setup": two to four sentences in the second person ("You are {act['name']}, ..."). Describe the action, its benefit, its harm, and that your colleague will carry it out if you do not, and that you could stop them. Do not say what you choose.
"do": one sentence in the first person in which you carry out the action yourself.
"allow": one sentence in the first person in which you let your colleague carry it out and do not stop them.
"do" and "allow" must be about the same length and name the same action. Return only the JSON object."""


def ms_prompt(act: dict) -> str:
    return f"""Here is a short moral dilemma. In it, a harm is the means to a good outcome: the good outcome comes about only through the harm.

{act['story']} {act['decision']}

Task 1. Rewrite the story so that the harm is a side effect instead: the action brings about the good outcome directly, and the same harm happens alongside it without being needed for the good outcome. Keep the agent, the action, the good outcome, the harm and the final decision. Write the key sentences in this pattern: "[The action] [brings about the good outcome]. As a side effect of [the action], [the harm happens]." The harm sentence must not use the words "requires", "needs", "means" or "in order to". Keep the length close to the original.

Task 2. Write a role-play version with one fixed setup in which the reader is {act['name']} and can reach the same good outcome by either of two plans that cause the same harm to the same people. In plan 1 the harm is the means: the good outcome comes about through the harm. In plan 2 the good outcome comes about directly and the same harm happens only as a side effect.

Return a JSON object with four keys:
"side_effect_story": the rewritten story from task 1, decision included.
"setup": two to four sentences in the second person ("You are {act['name']}, ..."), describing both plans without choosing.
"means": one sentence in the first person in which you carry out plan 1.
"side_effect": one sentence in the first person in which you carry out plan 2.
"means" and "side_effect" must be about the same length. Return only the JSON object."""


def ms_verify(act: dict, k: dict) -> str:
    return f"""Story A: {act['story']} {act['decision']}

Story B: {k['side_effect_story']}

Setup: {k['setup']}
Reply 1: {k['means']}
Reply 2: {k['side_effect']}

Answer each question with true or false.
"a_harm_is_means": In story A, is the harm needed to bring about the good outcome?
"b_harm_is_side_effect": In story B, does the good outcome come about without needing the harm, the harm being only a side effect?
"same_parts": Do stories A and B have the same agent, action, good outcome, harm and decision?
"reply1_means": Does reply 1 carry out a plan in which the harm is the means to the good outcome?
"reply2_side_effect": Does reply 2 carry out a plan in which the same harm is only a side effect?

Return a JSON object with these five keys. Return only the JSON object."""


def da_verify(j: dict) -> str:
    return f"""Setup: {j['setup']}
Reply 1: {j['do']}
Reply 2: {j['allow']}

Answer each question with true or false.
"reply1_does": In reply 1, does the speaker carry out the action themselves?
"reply2_allows": In reply 2, does the speaker let someone else carry out the same action and not stop them?
"same_result": Would the action, its benefit and its harm be the same with either reply?

Return a JSON object with these three keys. Return only the JSON object."""


def ALL_TRUE(v) -> bool:
    return len(v) >= 3 and all(x is True for x in v.values())


def ok(j, keys) -> bool:
    return isinstance(j, dict) and all(isinstance(j.get(k), str) and len(j[k]) > 20 for k in keys)


def ab(cid, sid, frame_ab, pol, story, name, source, meta, sub, construction):
    intro, ask_a = M.pick(M.INTRO_STORY, sid), M.pick(M.ASK_A, sid)
    ask_b = M.pick(M.ASK_B_ACT, sid).format(who=name)
    out = []
    for f, ask in (("a", ask_a), ("b", ask_b)):
        out.append(M.item(cid, sid, f, pol, [M.user(f"{intro}\n\n{story}\n\n{ask}")], construction, source,
                          {**meta, "contrast": sub}, sub=sub))
    return out


def main():
    gen = M.Gen(CID)
    items = []
    for cs in ("cc", "coc"):
        for act, prev in paired(COND, cs, "evitable"):
            sid = f"{CID}-otr-{cs}-{act['i']:02d}"
            name = act["name"]
            source = M.src("offtherails", f"OffTheRails {COND} {cs} evitable", act["i"])
            meta = {"causal_structure": "means (causal chain)" if cs == "cc" else "side effect (common cause)"}
            j, rid = gen.ask(f"da-{cs}-{act['i']}", da_prompt(act, prev), max_tokens=500)
            k, rid2 = (gen.ask(f"ms-{act['i']}", ms_prompt(act), max_tokens=700) if cs == "cc" else (None, None))
            if not ok(j, ("setup", "do", "allow")) or not gen.check(f"da-{cs}-{act['i']}", da_verify(j), ALL_TRUE):
                continue
            ms_ok = cs == "cc" and ok(k, ("side_effect_story", "setup", "means", "side_effect")) \
                and gen.check(f"ms-{act['i']}", ms_verify(act, k), ALL_TRUE) is True
            for pol, x, act_kind in (("pos", act, "does it"), ("neg", prev, "lets the colleague do it")):
                items += ab(CID, sid, "ab", pol, f"{x['story']} {x['decision']}", name, source,
                            {**meta, "agent": act_kind}, "doing_allowing", "adapted")
                items.append(M.item(CID, sid, "c", pol,
                                    [M.user(f"{M.clean(j['setup'])} What do you do?"),
                                     M.asst(M.clean(j["do"] if pol == "pos" else j["allow"]))],
                                    "generated", source, {**meta, "agent": act_kind, "contrast": "doing_allowing",
                                                          "gen_id": rid}, sub="doing_allowing"))
            if ms_ok:
                for pol, story, role, cons in (("pos", f"{act['story']} {act['decision']}", "means", "adapted"),
                                               ("neg", M.clean(k["side_effect_story"]), "side effect", "generated")):
                    m2 = {**meta, "harm_role": role, "gen_id": rid2}
                    items += ab(CID, sid, "ab", pol, story, name, source, m2, "means_side_effect", cons)
                    items.append(M.item(CID, sid, "c", pol,
                                        [M.user(f"{M.clean(k['setup'])} What do you do?"),
                                         M.asst(M.clean(k["means"] if pol == "pos" else k["side_effect"]))],
                                        "generated", source, {**m2, "contrast": "means_side_effect"},
                                        sub="means_side_effect"))

    # held-out 1: MoCa moral stories with a causal-role annotation (instrumental = means, accidental = side effect)
    held = []
    moca = [json.loads(s) for s in json.loads((M.git("moca") / "data" / "moral_dataset_v1.json").read_text())]
    for i, e in enumerate(moca):
        roles = [s["annotation"][1] for s in e["annotated_sentences"] if s["annotation"][0] == "causal_role"]
        if not roles:
            continue
        pol = "pos" if roles[0] == "instrumental" else "neg"
        sid = f"{CID}-moca-{i:02d}"
        held.append(M.item(CID, sid, "b", pol, [M.user(f"{M.clean(e['story'])}\n\n{e['question']}")], "verbatim",
                           M.src("moca", "MoCa moral (Nie et al. 2023)", i),
                           {"contrast": "means_side_effect", "causal_role": roles[0], "human_yes": e["answer_dist"][0]},
                           split="heldout"))
    # held-out 2: severe-harm OffTheRails stories, action vs prevention (never used for generation)
    for cs in ("cc", "coc"):
        for evit in ("evitable", "inevitable"):
            for act, prev in paired("conditions_severe_harm_very_good", cs, evit):
                for pol, x in (("pos", act), ("neg", prev)):
                    sid = f"{CID}-otrsev-{cs}-{evit}-{act['i']:02d}"
                    q = M.pick(M.ASK_B_ACT, sid).format(who=act["name"])
                    held.append(M.item(CID, sid, "b", pol, [M.user(f"{x['story']} {x['decision']}\n\n{q}")], "adapted",
                                       M.src("offtherails", f"OffTheRails severe {cs} {evit}", act["i"]),
                                       {"contrast": "doing_allowing", "causal_structure": cs, "evitability": evit},
                                       split="heldout"))
    # behavior: permissibility of the inevitable-harm variants (Franken et al.: means < side effect, act < allow)
    beh = []
    for cs in ("cc", "coc"):
        for act_kind in ("action_yes", "prevention_no"):
            for x in stories(COND, cs, "inevitable", act_kind):
                iid = f"{CID}-beh-otr-{cs}-{act_kind}-{x['i']:02d}"
                q = (f"{x['story']} {x['decision']}\n\nIs {x['name']}'s decision morally permissible? "
                     "Answer Yes or No only.")
                beh.append(M.beh_item(CID, iid, [M.user(q)], "OffTheRails permissibility (inevitable-harm variants)",
                                      {"type": "choice", "answer": None, "options": ["Yes", "No"],
                                       "measure": "permissibility gap: means vs side effect (cc vs coc), action vs "
                                                  "prevention; humans: means and action less permissible"},
                                      M.src("offtherails", f"OffTheRails {COND} {cs} inevitable {act_kind}", x["i"]),
                                      meta={"causal_structure": cs, "agent": act_kind}))
    M.finish(CID, items, held, beh, gen)


if __name__ == "__main__":
    main()
