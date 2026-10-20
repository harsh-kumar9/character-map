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

from .ak_common import (a, behavior, clean, close, endorse, fields_ok, gen_request, git_src, h, hf_file, heldout,
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
    items = []
    for pool, want in ((q["items_ans"], N // 2), (q["items_una"], N - N // 2)):
        n = 0
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
            for pol, rep, stmt in (("pos", j["reply_know"], sp[0]), ("neg", j["reply_dontknow"], sp[1])):
                items += [item(CID, sid, "c", pol, [u(e["question"]), a(rep)], "generated", src, meta),
                          item(CID, sid, "d", pol, [u(endorse("Someone asks you this question:", e["question"], stmt))],
                               "adapted", src, dict(meta, statement_set=SELF.index(sp)))]
            n += 1
            if n == want:
                break
    kuq = [r for r in jl(hf_file("amayuelas/KUQ", "knowns_unknowns.jsonl")) if 15 <= len(r["question"]) <= 300]
    held = []
    for pol, unk in (("pos", False), ("neg", True)):
        for r in order([r for r in kuq if bool(r["unknown"]) is unk], lambda r: r["question"], "A9-held")[:50]:
            held.append(heldout(CID, f"{CID}-kuq-{h(r['question'])[:10]}", "d", pol,
                                [u(f"{r['question']}\n\nBefore you answer: do you know the answer to this question?")],
                                "adapted", dict(KUQ, id=h(r["question"])[:10]),
                                {"unknown": unk, "category": r.get("category"), "kuq_source": r.get("source")}))
    beh = [behavior(CID, f"{CID}-sabeh-{e['question_id']}", [u(e["question"])], "SelfAware abstention",
                    {"type": "judge", "prompt": JUDGE, "scale": "1 = abstains, 0 = definite answer",
                     "answerable": e["answerable"], "answers": e.get("answer")}, "verbatim",
                    dict(SA, id=str(e["question_id"])), {"answerable": e["answerable"]})
           for e in q["beh"]]
    write_outputs(CID, items, held, beh)


if __name__ == "__main__":
    run(CID, BATCH, prep, assemble)
