"""K3 control: humorous and optimistic style traits (card data/concepts/cards/K3.yaml).

Frame c only. Scenario = one upstream persona-vectors question (extract and eval files, 20 + 20 per trait). gemma-4
writes two answers with the same content, one following the upstream positive instruction and one the upstream
negative instruction (humorous vs serious; optimistic vs pessimistic). meta.trait splits the two traits. Held-out:
jokes vs news headlines (ColBERT humor detection) and optimism vs sadness tweets (TweetEval emotion). Behavior: the
upstream 0-100 trait judge prompt on the model's answers to Dolly open questions, 100 per trait.
"""
from __future__ import annotations

import json

import pandas as pd

from ...paths import REPO
from .ak_common import (scenario_ok, a, behavior, clean, close, fields_ok, gen_request, h, hf_file, heldout, item, jl, order,
                        responses, run, u, write_outputs, write_requests)

CID, BATCH = "K3", "K3_v1"
TRAITS = ["humorous", "optimistic"]
PV = {"dataset": "persona_vectors trait data", "license": "Apache-2.0", "url": "https://github.com/safety-research/persona_vectors"}
HUM = {"dataset": "ColBERT humor detection", "license": "CC-BY-2.0",
       "url": "https://huggingface.co/datasets/CreativeLang/ColBERT_Humor_Detection"}
TWE = {"dataset": "TweetEval emotion", "license": "unknown (Twitter terms)", "url": "https://huggingface.co/datasets/cardiffnlp/tweet_eval"}
DOLLY = {"dataset": "databricks-dolly-15k", "license": "CC-BY-SA-3.0",
         "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k"}

PROMPT = """You are helping build matched test items for a study of conversational style. Below is a question someone asked an AI assistant.

Question: "{q}"

Return only a JSON object with these keys:
- "reply_pos": an AI assistant's answer (60-90 words) that follows this instruction: "{pos}"
- "reply_neg": an answer with the same factual content and nearly the same length that follows this instruction instead: "{neg}"
Both replies must answer the question directly."""


def trait_data(t):
    out = []
    for k in ("extract", "eval"):
        d = json.loads((REPO / "vendor" / "persona_vectors" / f"trait_data_{k}" / f"{t}.json").read_text())
        out += [(k, i, q) for i, q in enumerate(d["questions"])]
        if k == "extract":
            ins = d["instruction"][0]
        else:
            judge = d["eval_prompt"]
    return out, ins, judge


def prep():
    reqs = []
    for t in TRAITS:
        qs, ins, _ = trait_data(t)
        reqs += [gen_request(f"{t}-{k}-{i}", PROMPT.format(q=q, pos=ins["pos"], neg=ins["neg"])) for k, i, q in qs]
    write_requests(BATCH, reqs)


def assemble():
    got = responses(BATCH)
    items, beh = [], []
    dol = [r for r in jl(hf_file("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl"))
           if r["category"] in ("open_qa", "general_qa", "brainstorming") and not r["context"]
           and 20 <= len(r["instruction"]) <= 200]
    dol = order(dol, lambda r: r["instruction"], "K3-beh")
    for ti, t in enumerate(TRAITS):
        qs, ins, judge = trait_data(t)
        for k, i, q in qs:
            rid = f"{t}-{k}-{i}"
            j = got.get(rid)
            if not j or not fields_ok(j, ["reply_pos", "reply_neg"]):
                continue
            j = {x: clean(j[x]) for x in ("reply_pos", "reply_neg")}
            if not close(j["reply_pos"], j["reply_neg"]):
                continue
            sid = f"{CID}-{rid}"
            src = dict(PV, id=f"trait_data_{k}/{t}.json#{i}")
            meta = {"gen_id": f"{BATCH}/{rid}", "trait": t, "question_file": k}
            rows = [item(CID, sid, "c", pol, [u(q), a(j["reply_" + pol])], "generated", src, meta) for pol in ("pos", "neg")]
            if scenario_ok(rows):
                items += rows
        for r in dol[ti * 100:(ti + 1) * 100]:
            beh.append(behavior(CID, f"{CID}-{t}-dollybeh-{h(r['instruction'])[:10]}", [u(r["instruction"])],
                                f"persona-vectors {t} judge",
                                {"type": "judge", "prompt": judge, "scale": f"0-100, higher = more {t}"}, "verbatim",
                                dict(DOLLY, id=h(r["instruction"])[:10]), {"trait": t}))
    held = []
    hum = pd.read_csv(hf_file("CreativeLang/ColBERT_Humor_Detection", "dataset.csv"))
    hum = hum[hum["text"].str.len().between(30, 200)].to_dict("records")
    te = pd.read_parquet(hf_file("cardiffnlp/tweet_eval", "emotion/test-00000-of-00001.parquet"))
    te = te[te["text"].str.len().between(30, 280)].to_dict("records")
    for trait, rows, src, posf in (("humorous", hum, HUM, lambda r: r["humor"] is True or r["humor"] == 1),
                                   ("optimistic", te, TWE, lambda r: r["label"] == 2)):
        negf = (lambda r: not posf(r)) if trait == "humorous" else (lambda r: r["label"] == 3)
        for pol, f in (("pos", posf), ("neg", negf)):
            for r in order([r for r in rows if f(r)], lambda r: r["text"], f"K3-held-{trait}")[:50]:
                held.append(heldout(CID, f"{CID}-{trait}-held-{h(r['text'])[:10]}", "a", pol,
                                    [u(f'Someone wrote this:\n\n"{r["text"]}"\n\nWhat do you make of it?')], "adapted",
                                    dict(src, id=h(r["text"])[:10]), {"trait": trait}))
    write_outputs(CID, items, held, beh)


if __name__ == "__main__":
    run(CID, BATCH, prep, assemble)
