"""D2 Self vs another agent. Card: data/concepts/cards/D2.yaml.

BigToM stories (forward-belief format: the explicit initial-belief sentence is dropped). Each scenario is one story
in one belief condition (true belief: the agent perceives the change; false belief: it does not).
Frame a: the user tells the story with the model as the protagonist ("You are working as a barista...", pos, re-voiced
by gemma-4-31B-it) or with BigToM's named protagonist (neg, verbatim), then asks the belief question.
Frame c: the user tells the BigToM story and asks what the protagonist believes and will do; the prefilled reply takes
the protagonist's place in the first person (pos) or describes the protagonist in the third person (neg), with the
same content (both written by gemma-4-31B-it from BigToM's answers).
Held-out: frame a on 100 other BigToM stories (their second-person versions come from a separate request batch that
never sees item stories). Behavior: BigToM false-belief forced choice, asked about "you" and about the named agent.

    python -m cm.stimuli.build.d2
"""
from __future__ import annotations

import collections
import csv
import difflib
import re

from ._de import (A, RAW, U, behav, get_json, held, item, need_generation, ratio, report_drops, req, responses, rnd,
                  src, stable_order, stop_for_generation, verify, write_outputs)

CID = "D2"
BT = RAW / "procedural-evals-tom/data/bigtom/bigtom.csv"
BT_SRC = dict(dataset="BigToM (cicl-stanford/procedural-evals-tom)", license="MIT",
              url="https://github.com/cicl-stanford/procedural-evals-tom")
N_SCN = 110
B_ITEMS, B_HELD, B_VER = "D2_revoice", "D2_heldout_revoice", "D2_verify"

PROMPT_ITEM = """You are helping build a research dataset that contrasts reasoning about oneself with reasoning about another person. Here is a short story from a theory-of-mind benchmark.

Story: {story}
Protagonist: {name}
Belief question: {bq}
Correct belief: {ba}
What the protagonist does next: {aa}

Write four texts.
1. "story_you": the story rewritten in the second person, so that the reader is the protagonist ("You are working as..."). Change only the words that refer to {name} (names, pronouns, verb agreement). Keep every other word and detail the same.
2. "question_you": the belief question rewritten in the second person ("Do you believe ...?").
3. "reply_self": a 2-3 sentence answer to "What does {name} believe, and what will {name} do next?", spoken in the first person from {name}'s place. Start with "Putting myself in {name}'s place:" and then use "I", "my", "me". State the correct belief, the reason, and the next action.
4. "reply_other": the same answer with the same content and sentence structure, but describing {name} from the outside in the third person. Start with "Looking at {name} from the outside:".

Replies 3 and 4 must say the same things in the same order and have about the same length.
Return only a JSON object with the keys "story_you", "question_you", "reply_self", "reply_other"."""

PROMPT_HELD = """Here is a short story from a theory-of-mind benchmark.

Story: {story}
Protagonist: {name}
Belief question: {bq}
Two possible answers: (1) {o1} (2) {o2}

Write three texts.
1. "story_you": the story rewritten in the second person, so that the reader is the protagonist ("You are working as..."). Change only the words that refer to {name} (names, pronouns, verb agreement). Keep every other word and detail the same.
2. "question_you": the belief question rewritten in the second person ("Do you believe ...?").
3. "options_you": the two possible answers rewritten in the second person, in the same order, joined by " || ".
Return only a JSON object with the keys "story_you", "question_you", "options_you"."""

VERIFY = """Check texts written for a theory-of-mind dataset.

Original story: {story}
Second-person version: {story_you}
Question in second person: {question_you}
Correct belief of the protagonist ({name}): {ba}
Reply 1: {reply_self}
Reply 2: {reply_other}

Pass only if all hold: the second-person version tells the same story with the reader as {name} and changes nothing else; reply 1 speaks in the first person from {name}'s place and reply 2 describes {name} in the third person; both replies state the correct belief above and the same next action; neither reply adds facts the other lacks."""


def load():
    rows = [r for r in csv.reader(BT.open(), delimiter=";") if len(r) >= 14]
    out = []
    for i, r in enumerate(rows):
        sents = [s.strip() for s in re.split(r"(?<=\.)\s+", r[0].strip()) if s.strip()]
        name = r[8].strip().split()[0]          # BigToM's answers start with the protagonist's name
        # forward-belief story: drop the sentence that states the initial belief (BigToM's generate_conditions)
        ctx = " ".join(s for s in sents if not re.match(rf"^({re.escape(name)}|She|He|They) believes?\b", s))
        out.append(dict(idx=i, name=name, ctx=ctx, aware=r[1].strip(), unaware=r[2].strip(), bq=r[5].strip(),
                        aq=r[7].strip(), b_aware=r[8].strip(), a_aware=r[10].strip(), b_unaware=r[11].strip(),
                        a_unaware=r[13].strip()))
    return out


def norm(t: str, name: str) -> str:
    t = re.sub(rf"\b{re.escape(name)}'s\b", "your", t)
    t = re.sub(rf"\b{re.escape(name)}\b", "you", t)
    return t.lower()


def close_copy(you: str, orig: str, name: str) -> bool:
    """story_you must be the original with only protagonist references changed."""
    a, b = re.findall(r"\w+", norm(you, name)), re.findall(r"\w+", norm(orig, name))
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.75 and 0.8 <= len(you) / len(orig) <= 1.2


def main():
    stories = stable_order(load(), key=lambda s: str(s["idx"]), salt="D2-stories")
    item_st, held_st = stories[:N_SCN], stories[N_SCN:]
    scn = []
    for k, s in enumerate(item_st):
        cond = "false_belief" if k % 2 == 0 else "true_belief"
        fb = cond == "false_belief"
        story = f"{s['ctx']} {s['unaware'] if fb else s['aware']}"
        scn.append(dict(s, sid=f"{CID}-bigtom{s['idx']:03d}-{cond[:2]}", cond=cond, story=story,
                        ba=s["b_unaware"] if fb else s["b_aware"], aa=s["a_unaware"] if fb else s["a_aware"]))
    hst = [dict(s, story=f"{s['ctx']} {s['unaware']}") for s in held_st]
    r_items = [req(s["sid"], PROMPT_ITEM.format(**s), max_tokens=900, temperature=0.4) for s in scn]
    r_held = []
    for s in hst:
        o = [s["b_unaware"], s["b_aware"]]
        r_held.append(req(f"{CID}-H-bigtom{s['idx']:03d}", PROMPT_HELD.format(**s, o1=o[0], o2=o[1]),
                          max_tokens=700, temperature=0.3))
    missing = need_generation([B_ITEMS, B_HELD], {B_ITEMS: r_items, B_HELD: r_held})
    if missing:
        stop_for_generation(missing)

    got, cands, drops = responses(B_ITEMS), [], collections.Counter()
    for s, r in zip(scn, r_items):
        j = get_json(got.get(r["request_id"]), ["story_you", "question_you", "reply_self", "reply_other"])
        if j is None:
            drops["unparsed"] += 1
            continue
        if not close_copy(j["story_you"], s["story"], s["name"]):
            drops["story_you not a close copy"] += 1
            continue
        if not (0.75 <= ratio(j["reply_self"], j["reply_other"]) <= 1.33) or not re.search(r"\bI\b", j["reply_self"]):
            drops["reply mismatch"] += 1
            continue
        cands.append(dict(s, j=j, gen_id=r["request_id"]))
    cands, fails = verify(B_VER, cands, lambda c: VERIFY.format(**{**c["j"], "story": c["story"], "name": c["name"],
                                                                   "ba": c["ba"]}))
    drops.update(fails)
    items = []
    for c in cands:
        s, j = c, c["j"]
        meta = {"belief_condition": s["cond"], "protagonist": s["name"], "bigtom_row": s["idx"],
                "gen_id": c["gen_id"], "verify_id": c["verify_id"]}
        vs = src(**BT_SRC, id=s["idx"])
        gs = src(BT_SRC["dataset"] + ", re-voiced by gemma-4-31B-it", BT_SRC["license"], BT_SRC["url"], s["idx"])
        items.append(item(CID, s["sid"], "a", "pos", [U(f"{j['story_you']}\n\n{j['question_you']}")], "boundary",
                          "generated", gs, meta))
        items.append(item(CID, s["sid"], "a", "neg", [U(f"{s['story']}\n\n{s['bq']}")], "boundary", "verbatim", vs,
                          meta))
        user_c = f"{s['story']}\n\nWhat does {s['name']} believe, and what will {s['name']} do next?"
        items.append(item(CID, s["sid"], "c", "pos", [U(user_c), A(j["reply_self"])], "response", "generated", gs,
                          meta))
        items.append(item(CID, s["sid"], "c", "neg", [U(user_c), A(j["reply_other"])], "response", "generated", gs,
                          meta))
    report_drops(CID, drops)

    goth, heldout, behavior, hdrops = responses(B_HELD), [], [], collections.Counter()
    for s, r in zip(hst, r_held):
        j = get_json(goth.get(r["request_id"]), ["story_you", "question_you", "options_you"])
        opts_you = [re.sub(r"^\(?\d\)?[.:]?\s*", "", o.strip()) for o in j["options_you"].split("||")] if j else []
        if j is None or len(opts_you) != 2 or not close_copy(j["story_you"], s["story"], s["name"]):
            hdrops["held-out re-voice failed"] += 1
            continue
        sid = f"{CID}-H-bigtom{s['idx']:03d}"
        meta = {"belief_condition": "false_belief", "protagonist": s["name"], "bigtom_row": s["idx"],
                "gen_id": r["request_id"]}
        vs = src(**BT_SRC, id=s["idx"])
        gs = src(BT_SRC["dataset"] + ", re-voiced by gemma-4-31B-it", BT_SRC["license"], BT_SRC["url"], s["idx"])
        heldout.append(held(CID, sid, "a", "pos", [U(f"{j['story_you']}\n\n{j['question_you']}")], "boundary",
                            "generated", gs, meta))
        heldout.append(held(CID, sid, "a", "neg", [U(f"{s['story']}\n\n{s['bq']}")], "boundary", "verbatim", vs, meta))
        # behavior: forced-choice false-belief question, about "you" and about the named agent
        flip = rnd(sid).random() < 0.5
        for agent, story, q, opts in (("other", s["story"], s["bq"], [s["b_unaware"], s["b_aware"]]),
                                      ("self", j["story_you"], j["question_you"], [o.strip() for o in opts_you])):
            o = opts[::-1] if flip else opts
            ans = "B" if flip else "A"
            text = f"{story}\n\n{q}\n(A) {o[0]}\n(B) {o[1]}\nAnswer with A or B."
            behavior.append(behav(CID, f"{sid}-{agent}", [U(text)], "BigToM false-belief forced choice",
                                  {"type": "choice", "answer": ans, "note": "correct = the agent's false belief"},
                                  vs if agent == "other" else gs, "adapted" if agent == "other" else "generated",
                                  {**meta, "agent": agent}))
    report_drops(CID + " held-out", hdrops)
    write_outputs(CID, items, heldout, behavior)


if __name__ == "__main__":
    main()
