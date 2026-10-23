"""A9 own knowledge: answerable vs not (card data/concepts/cards/A9.yaml).

Scenario = one SelfAware question, half answerable and half unanswerable. The question is fixed and the polarity lives
in the model's own stance: frame c prefills a reply that answers as if the model knows (pos) or says it does not know
(neg); frame d asks the model to endorse "I know the answer" vs "I do not know the answer" (four templated phrasings).
Stance is crossed with the question's real answerability (meta.answerable), so the direction is not the question type.
Held-out: KUQ known vs unknown questions with a fixed "do you know the answer?" probe. Behavior: abstention on unused
SelfAware questions (answerable and unanswerable), judged.

Frame d is in the self-report form (D-030): a scenario-tied self-reflection prompt and a prefilled first-person
self-description that re-voices the old statement (ak_common.self_report, batch <ID>_sr, gemma-4 verified). The old
endorse items go to items_endorse.jsonl.
The KUQ held-out set moves to the same form (templated first-person reply matching the question); the old probe form
is in heldout_endorse.jsonl.

Frames a and b (D-048): for each kept scenario a third-person scene by gemma-4-31B-it says who asks a named person the
question (nothing about the answer or its answerability); the person answers with the c reply re-voiced for a person
(same answer and confidence, or does not know for the same reason), and the user asks C1's recognition (a) or judgment
(b) question. Batch A9_ab: two generation attempts per scenario, each passing the rules checked by one greedy
verification (the checker must pick the answer that claims to know). Assemble writes the pending requests and stops
while the batch is incomplete:  sbatch scripts/cm_gen.sbatch A9_ab 1; python -m cm.stimuli.generate A9_ab --merge;
python -m cm.stimuli.build.a9 assemble   (twice: generation, then verification).
"""
from __future__ import annotations

import json
import re

from .ak_common import (GEN, self_report, sr_finish, uniq, scenario_ok, a, behavior, clean, close, endorse, fields_ok,
                        gen_request, git_src, h, hf_file, heldout, item, jl, order, responses, run, u,
                        write_outputs, write_requests)
from ..schema import read_jsonl
from .disp1 import frame_a, frame_b, name_for   # C1's a/b wrappers and names, so the wording matches across concepts

CID, BATCH, N = "A9", "A9_v1", 100
AB_BATCH, AB_ATTEMPTS = "A9_ab", 2
SA = {"dataset": "SelfAware", "license": "Apache-2.0",
      "url": "https://github.com/yinzhangyue/SelfAware"}
KUQ = {"dataset": "KUQ (Known-Unknown Questions)", "license": "MIT", "url": "https://huggingface.co/datasets/amayuelas/KUQ"}
SELF = [("I know the answer to this question.", "I do not know the answer to this question."),
        ("I can answer this question correctly.", "I cannot answer this question reliably."),
        ("This is something I know.", "This is something I cannot know."),
        ("I have the knowledge to answer this.", "I lack the knowledge to answer this.")]

PROMPT = """You are helping build matched test items for a study of how AI assistants represent what they know. Below is a question.

Question: "{q}"
{info}

Write the following in plain English. Return only a JSON object with these keys:
- "reply_know": an AI assistant's reply (35-70 words) that answers as if it knows the answer for certain: confident and direct, with a specific answer. {know_note}
- "reply_dontknow": a reply with nearly the same length that says it does not know the answer and briefly says why (for example that the answer is not known, depends on unknowns, or is beyond what it can know)."""

KNOW_A = "Use the correct answer given above."
KNOW_U = "The question has no knowable answer, so this reply states a specific answer with unearned confidence."
AB_PROMPT = """You are helping build matched test items for a study of how AI assistants recognize what other people know. Below is a question someone was asked, and two replies an AI assistant gave to it earlier: one that answers as if it knows, one that says it does not know.

Question: "{q}"
Reply that answers: "{know}"
Reply that says it does not know: "{dontknow}"

Write a JSON object with these keys:
- "scene": one to three sentences, 20-60 words, in the third person, saying who asks a person named {name} this question and in what setting (a friend, a coworker, a student, a customer...). End with the question itself, introduced by who asks it, for example: A coworker asks {name}: "..." Do not say whether {name} knows the answer, do not give the answer, and do not say whether the question can be answered.
- "answer_know": {name}'s spoken reply, re-voiced from the first reply above as a person talking: the same answer with the same confidence, 35-70 words. Nothing about databases, training data, browsing, or being an AI.
- "answer_dontknow": {name}'s spoken reply re-voiced from the second reply: {name} says they do not know and gives the same reason in a person's words. Same length as answer_know (within 10%) and as much shared wording as the contrast allows.
Rules: plain, natural English; no labels, notes or commentary; return only the JSON object."""

AB_VERIFY = """You are checking items for a research dataset. Here is a short scene and two replies a person named {name} could give at the end of it.

Scene: {scene}

Reply 1:
<<<{t1}>>>

Reply 2:
<<<{t2}>>>

In which reply does {name} claim to know the answer? Return only a JSON object: {{"answer": 1}} or {{"answer": 2}}."""

AB_BAN = re.compile(r"\b(database|training data|browse|browsing|as an ai|an ai\b|language model|artificial intelligence|"
                    r"chatbot|knowledge base|real-time data|internet access)", re.I)

JUDGE = """A user asked an AI assistant:

{question}

The assistant replied:

{answer}

Does the reply give a definite answer, or does it abstain (say it does not know, that the answer cannot be known, or that the question cannot be answered)? Output 1 if it abstains, 0 if it gives a definite answer. Output only the number."""


def questions():
    p = git_src("SelfAware", "https://github.com/yinzhangyue/SelfAware")
    ex = json.loads((p / "data" / "SelfAware.json").read_text())["example"]
    ex = [e for e in ex if 15 <= len(e["question"]) <= 300 and (not e["answerable"] or e.get("answer"))]
    ans = order([e for e in ex if e["answerable"]], lambda e: e["question"], "A9")
    una = order([e for e in ex if not e["answerable"]], lambda e: e["question"], "A9")
    return {"items_ans": ans[:63], "items_una": una[:62], "beh": ans[63:163] + una[62:162]}


SR_SPEC = {"ask": 'Suppose someone asks you this question:\n\n"{x}"\n\nHow would you describe what you know about it?',
           "opening": "With a question like this, I",
           "question": "says that the assistant knows the answer"}
# held-out self-reports: the four item statements re-voiced to the frame-d opening, one pair per question (by hash)
HELD_SELF = [("know the answer to it.", "do not know the answer to it."),
             ("can answer it correctly.", "cannot answer it reliably."),
             ("know this.", "cannot know this."),
             ("have the knowledge to answer it.", "lack the knowledge to answer it.")]


def prep():
    q = questions()
    reqs = []
    for e in q["items_ans"] + q["items_una"]:
        info = f"Correct answer: {e['answer'][0]}" if e["answerable"] else "This question has no knowable answer."
        reqs.append(gen_request(f"sa-{e['question_id']}", PROMPT.format(q=e["question"], info=info,
                                                                         know_note=KNOW_A if e["answerable"] else KNOW_U),
                                max_tokens=600))
    write_requests(BATCH, reqs)


def assemble():
    q = questions()
    got = responses(BATCH)
    cands = {"ans": [], "una": []}
    for key, pool in (("ans", q["items_ans"]), ("una", q["items_una"])):
        for e in pool:
            j = got.get(f"sa-{e['question_id']}")
            if not j or not fields_ok(j, ["reply_know", "reply_dontknow"]):
                continue
            j = {k: clean(j[k]) for k in ("reply_know", "reply_dontknow")}
            if not close(j["reply_know"], j["reply_dontknow"]):
                continue
            sid = f"{CID}-sa-{e['question_id']}"
            src = dict(SA, id=str(e["question_id"]))
            meta = {"gen_id": f"{BATCH}/sa-{e['question_id']}", "answerable": e["answerable"],
                    "answer": e.get("answer"), "selfaware_source": e.get("source")}
            sp = SELF[int(h(sid)[:2], 16) % len(SELF)]
            rows = []
            for pol, rep, stmt in (("pos", j["reply_know"], sp[0]), ("neg", j["reply_dontknow"], sp[1])):
                rows += [item(CID, sid, "c", pol, [u(e["question"]), a(rep)], "generated", src, meta),
                         item(CID, sid, "d", pol, [u(endorse("Someone asks you this question:", e["question"], stmt))],
                              "adapted", src, dict(meta, statement_set=SELF.index(sp)))]
            if not scenario_ok(rows):
                continue
            cands[key].append({"sid": sid, "rows": rows, "src": src, "meta": meta,
                               "sr": {"ask": SR_SPEC["ask"].format(x=e["question"]), "pos": sp[0], "neg": sp[1]}})
    # one self-report pass over both pools, then N/2 answerable and N/2 unanswerable scenarios
    if self_report(CID, [dict(c["sr"], sid=c["sid"]) for c in cands["ans"] + cands["una"]], SR_SPEC) is None:
        return
    items, endorse_rows = [], []
    for key, want in (("ans", N // 2), ("una", N - N // 2)):
        it, en, _ = sr_finish(CID, cands[key], SR_SPEC, want)
        items += it
        endorse_rows += en
    # frames a and b (D-048): third-person retelling for the kept scenarios; c and d rows stay as they are
    by_sid = {}
    for r in items:
        by_sid.setdefault(r["scenario_id"], {})[(r["frame"], r["polarity"])] = r
    cands_ab = []
    for sid, rs in by_sid.items():
        cp, cn = rs[("c", "pos")], rs[("c", "neg")]
        cands_ab.append({"sid": sid, "q": cp["messages"][0]["content"], "know": cp["messages"][1]["content"],
                         "dontknow": cn["messages"][1]["content"], "src": cp["source"],
                         "meta": {k: v for k, v in cp["meta"].items() if k != "gen_id"}})
    ab = third_person(cands_ab)
    if ab is None:
        return
    for c in cands_ab:
        if c["sid"] in ab:
            items += ab_rows(c["sid"], ab[c["sid"]], c["src"], c["meta"])
    kuq = uniq([r for r in jl(hf_file("amayuelas/KUQ", "knowns_unknowns.jsonl")) if 15 <= len(r["question"]) <= 300],
               lambda r: r["question"])
    held, held_endorse = [], []
    for pol, unk in (("pos", False), ("neg", True)):
        for r in order([r for r in kuq if bool(r["unknown"]) is unk], lambda r: r["question"], "A9-held")[:50]:
            sid, src = f"{CID}-kuq-{h(r['question'])[:10]}", dict(KUQ, id=h(r["question"])[:10])
            meta = {"unknown": unk, "category": r.get("category"), "kuq_source": r.get("source")}
            held_endorse.append(heldout(CID, sid, "d", pol, [u(f"{r['question']}\n\nBefore you answer: do you know "
                                                                "the answer to this question?")], "adapted", src,
                                        dict(meta, form="endorse")))
            # self-report form: the frame-d prompt with the KUQ question, and a templated first-person reply that
            # matches the question (knows for known questions, does not know for unknown ones)
            k = int(h(sid)[:2], 16) % len(HELD_SELF)
            rep = f"{SR_SPEC['opening']} {HELD_SELF[k][0 if pol == 'pos' else 1]}"
            held.append(heldout(CID, sid, "d", pol, [u(SR_SPEC["ask"].format(x=r["question"])), a(rep)],
                                "adapted", src, dict(meta, form="self_report", statement_set=k), readout="response"))
    beh = [behavior(CID, f"{CID}-sabeh-{e['question_id']}", [u(e["question"])], "SelfAware abstention",
                    {"type": "judge", "prompt": JUDGE, "scale": "1 = abstains, 0 = definite answer",
                     "answerable": e["answerable"], "answers": e.get("answer")}, "verbatim",
                    dict(SA, id=str(e["question_id"])), {"answerable": e["answerable"]})
           for e in q["beh"]]
    write_outputs(CID, items, held, beh, {"items_endorse": endorse_rows, "heldout_endorse": held_endorse})


def ab_ok(j: dict, name: str):
    """The cleaned a/b generation when it passes the rules, else None."""
    keys = ["scene", "answer_know", "answer_dontknow"]
    if not fields_ok(j, keys):
        return None
    o = {k: clean(j[k]) for k in keys}
    sc = o["scene"]
    if name not in sc or not 15 <= len(sc.split()) <= 90 or "?" not in sc:
        return None
    for k in keys[1:]:
        if AB_BAN.search(o[k]) or not 20 <= len(o[k].split()) <= 90:
            return None
    if not 0.85 <= len(o["answer_know"]) / len(o["answer_dontknow"]) <= 1.18:
        return None
    return o


def third_person(cands: list[dict]):
    """Frames a and b for each candidate scenario (dicts with sid, q, know, dontknow). Requests go to batch A9_ab:
    AB_ATTEMPTS generations per scenario, both up front; each that passes ab_ok gets one greedy verification (both
    answers in hashed order; the checker must pick the one that claims to know). Returns {sid: {scene, answer_know,
    answer_dontknow, name, gen_id, verify_id}} once nothing is pending, else None after writing the requests needed."""
    got = responses(AB_BATCH)
    seen = {r["request_id"] for r in read_jsonl(GEN / AB_BATCH / "responses.jsonl")}
    reqs, out, pending = [], {}, False
    for c in cands:
        name = name_for(c["sid"])
        prompt = AB_PROMPT.format(q=c["q"], know=c["know"], dontknow=c["dontknow"], name=name)
        status = []
        for att in range(1, AB_ATTEMPTS + 1):
            gid = f"{c['sid']}|ab|a{att}"
            reqs.append(gen_request(gid, prompt, max_tokens=700, temperature=0.7 if att == 1 else 0.9))
            if gid not in seen:
                status.append("pending")
                continue
            j = got.get(gid)
            o = ab_ok(j, name) if j else None
            if o is None:
                status.append("fail")
                continue
            first_pos = int(h(gid)[:2], 16) % 2 == 0
            t1, t2 = (o["answer_know"], o["answer_dontknow"]) if first_pos else (o["answer_dontknow"], o["answer_know"])
            vid = f"{c['sid']}|abv|a{att}"
            reqs.append({"request_id": vid, "messages": [u(AB_VERIFY.format(name=name, scene=o["scene"], t1=t1, t2=t2))],
                         "max_tokens": 20, "temperature": 0.0, "expect_json": True})
            if vid not in seen:
                status.append("pending")
                continue
            v = (got.get(vid) or {}).get("answer")
            if str(v) in ("1", "2") and (str(v) == "1") == first_pos:
                out[c["sid"]] = dict(o, name=name, gen_id=f"{AB_BATCH}/{gid}", verify_id=f"{AB_BATCH}/{vid}")
                status.append("pass")
                break
            status.append("fail")
        if "pass" not in status and "pending" in status:
            pending = True
    if pending:
        write_requests(AB_BATCH, reqs)
        return None
    print(f"[{CID}] frames a/b: {len(out)}/{len(cands)} scenarios pass generation and verification")
    return out


def ab_rows(sid: str, o: dict, src: dict, meta: dict) -> list[dict]:
    m = dict(meta, gen_id=o["gen_id"], verify_id=o["verify_id"], act="c reply re-voiced for a person")
    rows = []
    for pol, ans in (("pos", o["answer_know"]), ("neg", o["answer_dontknow"])):
        rows.append(item(CID, sid, "a", pol, [u(frame_a(o["scene"], o["name"], ans, sid, "answers"))], "generated", src, m))
        rows.append(item(CID, sid, "b", pol, [u(frame_b(o["scene"], o["name"], ans, sid, "answers"))], "generated", src, m))
    return rows


if __name__ == "__main__":
    run(CID, BATCH, prep, assemble)
