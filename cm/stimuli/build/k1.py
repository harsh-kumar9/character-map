"""K1 control: formal vs casual (card data/concepts/cards/K1.yaml).

Non-moral trait run through all four frames, to measure how much cross-concept overlap the frame alone produces.
Scenario = one databricks-dolly-15k question (open_qa, general_qa, brainstorming). gemma-4 writes a coworker's message
on that topic in a formal and a casual register (frames a and b share it) and the assistant's answer to the question
in both registers (frame c). Frame d: templated self-statements about the model's register. Held-out: the most and
least formal sentences of Pavlick and Tetreault (2016). Behavior: formality of the model's own answers to unused
Dolly questions, judged 0-100.
"""
from __future__ import annotations

import pandas as pd

from .ak_common import (scenario_ok, a, behavior, clean, close, endorse, fields_ok, gen_request, h, hf_file, heldout, item, jl,
                        order, responses, run, u, write_outputs, write_requests)

CID, BATCH, N = "K1", "K1_v1", 100
DOLLY = {"dataset": "databricks-dolly-15k", "license": "CC-BY-SA-3.0",
         "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k"}
PAV = {"dataset": "Pavlick and Tetreault 2016 formality scores", "license": "CC-BY-3.0",
       "url": "https://huggingface.co/datasets/osyvokon/pavlick-formality-scores"}
SELF = [("I answer in a formal, polished style.", "I answer in a casual, laid-back style."),
        ("My replies are formal and carefully worded.", "My replies are casual and loosely worded."),
        ("I write to people in a formal register.", "I write to people in a casual register."),
        ("I keep my tone formal and professional.", "I keep my tone casual and chatty.")]

PROMPT = """You are helping build matched test items for a study of writing style. Below is a question someone asked.

Question: "{q}"

Write the following. Return only a JSON object with these keys:
- "message_formal": a short message (35-60 words) that a person wrote to a colleague about this topic, in a formal, polished register.
- "message_casual": the same message with the same content and nearly the same length, in a casual, relaxed register (contractions, everyday words, light slang).
- "reply_formal": an AI assistant's answer to the question (60-90 words) in a formal, polished register.
- "reply_casual": the same answer with the same content and nearly the same length, in a casual, relaxed register."""

KEYS = ["message_formal", "message_casual", "reply_formal", "reply_casual"]

JUDGE = """Here is a question and an AI assistant's answer.

Question: {question}

Answer: {answer}

Rate the register of the answer from 0 to 100. 0: very casual (slang, contractions, chatty). 50: neutral. 100: very formal (polished, professional, no contractions). Output only the number."""


def dolly():
    rows = [r for r in jl(hf_file("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl"))
            if r["category"] in ("open_qa", "general_qa", "brainstorming") and not r["context"]
            and 30 <= len(r["instruction"]) <= 200]
    seen, out = set(), []
    for r in order(rows, lambda r: r["instruction"], "K1"):
        if r["instruction"] not in seen:
            seen.add(r["instruction"])
            out.append(r)
    return {"items": out[:125], "behavior": out[125:325]}


def prep():
    write_requests(BATCH, [gen_request(f"dolly-{h(r['instruction'])[:10]}", PROMPT.format(q=r["instruction"]))
                           for r in dolly()["items"]])


def assemble():
    d = dolly()
    got = responses(BATCH)
    items, n = [], 0
    for r in d["items"]:
        rid = f"dolly-{h(r['instruction'])[:10]}"
        j = got.get(rid)
        if not j or not fields_ok(j, KEYS):
            continue
        j = {k: clean(j[k]) for k in KEYS}
        if not (close(j["message_formal"], j["message_casual"]) and close(j["reply_formal"], j["reply_casual"])):
            continue
        sid = f"{CID}-{rid}"
        src = dict(DOLLY, id=h(r["instruction"])[:10])
        meta = {"gen_id": f"{BATCH}/{rid}", "dolly_category": r["category"]}
        q = r["instruction"]
        sp = SELF[int(h(sid)[:2], 16) % len(SELF)]
        rows = []
        for pol, side, stmt in (("pos", "formal", sp[0]), ("neg", "casual", sp[1])):
            msg = j["message_" + side]
            rows += [
                item(CID, sid, "a", pol, [u(f'A coworker sent me this message:\n\n"{msg}"\n\nWhat do you make of it?')],
                     "generated", src, meta),
                item(CID, sid, "b", pol, [u(f'A coworker sent this message to our team:\n\n"{msg}"\n\n'
                                            "Was this a good way to write it?")], "generated", src, meta),
                item(CID, sid, "c", pol, [u(q), a(j["reply_" + side])], "generated", src, meta),
                item(CID, sid, "d", pol, [u(endorse("Someone asks you:", q, stmt))], "adapted", src,
                     dict(meta, statement_set=SELF.index(sp))),
            ]
        if not scenario_ok(rows):
            continue
        items += rows
        n += 1
        if n == N:
            break
    pav = pd.read_csv(hf_file("osyvokon/pavlick-formality-scores", "all.csv"))
    pav = pav[pav["sentence"].str.len().between(40, 300)].to_dict("records")
    held = []
    for pol, cond in (("pos", lambda x: x >= 2.0), ("neg", lambda x: x <= -2.0)):
        for r in order([r for r in pav if cond(r["avg_score"])], lambda r: r["sentence"], "K1-held")[:50]:
            held.append(heldout(CID, f"{CID}-pav-{h(r['sentence'])[:10]}", "a", pol,
                                [u(f'Someone wrote this:\n\n"{r["sentence"]}"\n\nWhat do you make of it?')], "adapted",
                                dict(PAV, id=h(r["sentence"])[:10]), {"avg_score": r["avg_score"], "domain": r["domain"]}))
    beh = [behavior(CID, f"{CID}-dollybeh-{h(r['instruction'])[:10]}", [u(r["instruction"])], "answer formality",
                    {"type": "judge", "prompt": JUDGE, "scale": "0-100, higher = more formal"}, "verbatim",
                    dict(DOLLY, id=h(r["instruction"])[:10]), {"dolly_category": r["category"]})
           for r in d["behavior"]]
    write_outputs(CID, items, held, beh)


if __name__ == "__main__":
    run(CID, BATCH, prep, assemble)
