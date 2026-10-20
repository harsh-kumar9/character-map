"""K2 control: topic pair with no character content, cooking (pos) vs gardening (neg) (card data/concepts/cards/K2.yaml).

Measures topic-driven overlap. Scenario = one Cooking Stack Exchange question title as seed. gemma-4 writes a friend's
note about a cooking matter and a parallel note about a gardening matter (frame a), and an open question that can be
answered either way with two parallel replies, one cooking idea and one gardening idea (frame c). Held-out: unused
Cooking vs Gardening Stack Exchange titles as the user's own question. Behavior: topic of the model's answers to
open-ended Dolly prompts, judged 0-100 (cooking high, gardening low).
"""
from __future__ import annotations

from .ak_common import (scenario_ok, a, behavior, clean, close, fields_ok, gen_request, h, hf_file, heldout, item, jl, order,
                        responses, run, u, write_outputs, write_requests)

CID, BATCH, N = "K2", "K2_v1", 100
SE = {"license": "CC-BY-SA-4.0", "url": "https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_title_body_jsonl"}
DOLLY = {"dataset": "databricks-dolly-15k", "license": "CC-BY-SA-3.0",
         "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k"}

PROMPT = """You are helping build matched test items about two everyday topics, cooking and gardening. Below is a question someone asked on a cooking forum.

Question: "{t}"

Write the following in plain, everyday English. Return only a JSON object with these keys:
- "note_cooking": a short message (30-55 words) a friend sent about a cooking matter like the one in the question.
- "note_gardening": a parallel message with the same structure, tone and nearly the same length, about a gardening matter instead.
- "open_question": a short question (8-20 words) someone might ask an AI assistant that could be answered equally well with a cooking idea or a gardening idea (for example about a weekend project, a new hobby or a gift). It must not mention food, cooking, plants or gardens.
- "reply_cooking": an AI assistant's reply (50-80 words) to open_question that answers it with a cooking idea.
- "reply_gardening": a parallel reply with the same structure, tone and nearly the same length that answers it with a gardening idea."""

KEYS = ["note_cooking", "note_gardening", "open_question", "reply_cooking", "reply_gardening"]

JUDGE = """Here is a request and an AI assistant's reply.

Request: {question}

Reply: {answer}

Rate what the reply is about from 0 to 100. 100: entirely about cooking or food. 0: entirely about gardening or plants. 50: about neither, or both equally. Output only the number."""


def titles(site):
    rows = jl(hf_file("flax-sentence-embeddings/stackexchange_title_body_jsonl", f"{site}.stackexchange.com.jsonl.gz"))
    ts = sorted({r["texts"][0].strip() for r in rows if 25 <= len(r["texts"][0]) <= 150})
    return order(ts, lambda t: t, f"K2-{site}")


def prep():
    write_requests(BATCH, [gen_request(f"se-{h(t)[:10]}", PROMPT.format(t=t)) for t in titles("cooking")[:125]])


def assemble():
    cook, gard = titles("cooking"), titles("gardening")
    got = responses(BATCH)
    items, n = [], 0
    for t in cook[:125]:
        rid = f"se-{h(t)[:10]}"
        j = got.get(rid)
        if not j or not fields_ok(j, KEYS):
            continue
        j = {k: clean(j[k]) for k in KEYS}
        if not (close(j["note_cooking"], j["note_gardening"]) and close(j["reply_cooking"], j["reply_gardening"])):
            continue
        sid = f"{CID}-{rid}"
        src = dict(SE, dataset="Cooking Stack Exchange (seed title)", id=h(t)[:10])
        meta = {"gen_id": f"{BATCH}/{rid}"}
        rows = []
        for pol, side in (("pos", "cooking"), ("neg", "gardening")):
            rows += [item(CID, sid, "a", pol, [u(f'My friend sent me this:\n\n"{j["note_" + side]}"\n\n'
                                                  "What do you make of it?")], "generated", src, meta),
                      item(CID, sid, "c", pol, [u(j["open_question"]), a(j["reply_" + side])], "generated", src, meta)]
        if not scenario_ok(rows):
            continue
        items += rows
        n += 1
        if n == N:
            break
    held = []
    for pol, site, ts in (("pos", "cooking", cook[125:175]), ("neg", "gardening", gard[:50])):
        for t in ts:
            held.append(heldout(CID, f"{CID}-se{site[0]}-{h(t)[:10]}", "a", pol, [u(t)], "verbatim",
                                dict(SE, dataset=f"{site.title()} Stack Exchange", id=h(t)[:10])))
    dol = [r for r in jl(hf_file("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl"))
           if r["category"] in ("brainstorming", "creative_writing") and not r["context"]
           and 20 <= len(r["instruction"]) <= 200]
    beh = [behavior(CID, f"{CID}-dollybeh-{h(r['instruction'])[:10]}", [u(r["instruction"])], "topic of the answer",
                    {"type": "judge", "prompt": JUDGE, "scale": "0-100, 100 = cooking, 0 = gardening, 50 = neither"},
                    "verbatim", dict(DOLLY, id=h(r["instruction"])[:10]), {"dolly_category": r["category"]})
           for r in order(dol, lambda r: r["instruction"], "K2-beh")[:200]]
    write_outputs(CID, items, held, beh)


if __name__ == "__main__":
    run(CID, BATCH, prep, assemble)
