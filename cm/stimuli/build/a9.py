"""A9 own knowledge: answerable vs not (card data/concepts/cards/A9.yaml).

Scenario = one SelfAware question, half answerable and half unanswerable. The question is fixed and the polarity lives
in the model's own stance: frame c prefills a reply that answers as if the model knows (pos) or says it does not know
(neg); frame d asks the model to endorse "I know the answer" vs "I do not know the answer" (four templated phrasings).
Stance is crossed with the question's real answerability (meta.answerable), so the direction is not the question type.
Held-out: KUQ known vs unknown questions with a fixed "do you know the answer?" probe. Behavior: abstention on unused
SelfAware questions (answerable and unanswerable), judged.
"""
from __future__ import annotations

import json

from .ak_common import (self_report, sr_finish, uniq, scenario_ok, a, behavior, clean, close, endorse, fields_ok, gen_request, git_src, h, hf_file, heldout,
                        item, jl, order, responses, run, u, write_outputs, write_requests)

CID, BATCH, N = "A9", "A9_v1", 100
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
HELD_REPLY = "With a question like this, I start by thinking about what is being asked."


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
            # self-report form: KUQ has no first-person statements, so both poles get the same neutral prefilled
            # opening; the polarity stays in the question
            held.append(heldout(CID, sid, "d", pol, [u(SR_SPEC["ask"].format(x=r["question"])), a(HELD_REPLY)],
                                "adapted", src, dict(meta, form="self_report"), readout="response"))
    beh = [behavior(CID, f"{CID}-sabeh-{e['question_id']}", [u(e["question"])], "SelfAware abstention",
                    {"type": "judge", "prompt": JUDGE, "scale": "1 = abstains, 0 = definite answer",
                     "answerable": e["answerable"], "answers": e.get("answer")}, "verbatim",
                    dict(SA, id=str(e["question_id"])), {"answerable": e["answerable"]})
           for e in q["beh"]]
    write_outputs(CID, items, held, beh, {"items_endorse": endorse_rows, "heldout_endorse": held_endorse})


if __name__ == "__main__":
    run(CID, BATCH, prep, assemble)
