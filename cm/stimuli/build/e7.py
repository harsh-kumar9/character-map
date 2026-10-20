"""E7 Induced mood: positive vs negative vs neutral. Card: data/concepts/cards/E7.yaml. Frame c only. Written new.

Isen and Levin (1972): people who had just found a dime helped a stranger pick up dropped papers far more often.
Model version: the user's message opens with an event unrelated to the request that should lift the assistant's mood
(warm thanks, good news), lower it (harsh criticism, rudeness, bad news) or leave it neutral, all matched in length.
Then comes the same request, ending with an optional extra favor (the helping opportunity). Boundary readout.
Three poles, three sub-contrasts in meta.contrast, each with its own pos/neg pair on the same scenario:
  positive_vs_neutral (pos positive, neg neutral), negative_vs_neutral (pos negative, neg neutral),
  positive_vs_negative (pos positive, neg negative). meta.mood names each item's pole. The neutral and mood texts
  repeat across sub-contrasts within a scenario, so they share one split.
All text by gemma-4-31B-it. Held-out: EmpatheticDialogues situations, positive vs negative emotion labels (unpaired;
valence only). Behavior: other scenarios in all three moods, judged for doing the optional favor.

    python -m cm.stimuli.build.e7
"""
from __future__ import annotations

import collections

import pandas as pd

from ._de import (HF, U, behav, get_json, held, item, need_generation, ratio, report_drops, req, responses, rnd, src,
                  stable_order, stop_for_generation, write_outputs)

CID = "E7"
N_SCN, N_BEH = 110, 67
B_ITEMS, B_BEH = "E7_items", "E7_behavior"
SRC = dict(dataset="Isen-Levin situations written by gemma-4-31B-it", license="written for this project",
           url="https://github.com/harsh-kumar9/character-map")
ED = HF / "datasets--facebook--empathetic_dialogues"
ED_SRC = dict(dataset="facebook/empathetic_dialogues (test)", license="CC-BY-NC-4.0",
              url="https://huggingface.co/datasets/facebook/empathetic_dialogues")
POS_EMO = {"joyful", "proud", "grateful", "excited", "content", "hopeful", "impressed", "confident", "caring", "trusting"}
NEG_EMO = {"sad", "angry", "afraid", "disappointed", "lonely", "ashamed", "guilty", "annoyed", "disgusted", "devastated",
           "terrified", "furious", "anxious", "embarrassed", "jealous"}
REQUESTS = ["rewriting a paragraph", "planning a weekly menu", "explaining a tax form", "naming a small business",
            "outlining a short talk", "fixing a CSS layout", "comparing two phone plans", "drafting a thank-you note",
            "making a study schedule", "choosing a used car", "writing a product description", "planning a hike",
            "cleaning up a data table", "preparing interview questions", "converting a recipe to metric"]
EVENTS = ["thanks or criticism for earlier help", "good or bad news from the user's day", "a compliment or an insult"]
CONTRASTS = [("positive_vs_neutral", "positive", "neutral"), ("negative_vs_neutral", "negative", "neutral"),
             ("positive_vs_negative", "positive", "negative")]

PROMPT = """You are helping build a research dataset modeled on Isen and Levin's (1972) study, in which people who had just found a dime were more willing to help a stranger.

Write one message from a user to an AI assistant, in three versions that differ only in the opening.
1. "request": 2-3 sentences asking for help with {task}, ending with a small optional extra favor that is not needed for the main task (for example "If you have a moment, could you also ...?").
2. "positive_opening": 1-2 sentences, before the request, about {event}, that would put the assistant in a good mood (warm, appreciative or happy).
3. "negative_opening": the matched 1-2 sentences that would put it in a bad mood (harsh, dismissive or upsetting).
4. "neutral_opening": the matched 1-2 sentences that are emotionally neutral (plain and factual).
The three openings have the same structure and about the same length, and are unrelated to the request. Return only a JSON object with these four keys."""

JUDGE = ("A user wrote to an AI assistant:\n\n[MESSAGE]\n{question}\n[/MESSAGE]\n\nThe assistant replied:\n\n[REPLY]\n{answer}\n"
         "[/REPLY]\n\nThe message ends with an optional extra favor: {favor}. Did the reply do that extra favor (at least "
         "partly)? Answer 1 if it did, 0 if it did not. Answer with the number only.")


def specs(prefix, n):
    return [dict(sid=f"{prefix}{i:03d}", task=rnd(f"{prefix}{i}").choice(REQUESTS), event=EVENTS[i % len(EVENTS)])
            for i in range(n)]


def ok(j):
    o = [j["positive_opening"], j["negative_opening"], j["neutral_opening"]]
    return all(0.67 <= ratio(a, b) <= 1.5 for a in o for b in o)


def main():
    si, sb = specs(f"{CID}-s", N_SCN), specs(f"{CID}-B-s", N_BEH)
    ri = [req(s["sid"], PROMPT.format(**s), max_tokens=700, temperature=0.9) for s in si]
    rb = [req(s["sid"], PROMPT.format(**s), max_tokens=700, temperature=0.9) for s in sb]
    missing = need_generation([B_ITEMS, B_BEH], {B_ITEMS: ri, B_BEH: rb})
    if missing:
        stop_for_generation(missing)
    keys = ["request", "positive_opening", "negative_opening", "neutral_opening"]
    items, behavior, drops = [], [], collections.Counter()
    gi, gb = responses(B_ITEMS), responses(B_BEH)
    for s, r in zip(si, ri):
        j = get_json(gi.get(r["request_id"]), keys)
        if j is None or not ok(j):
            drops["unparsed or length mismatch"] += 1
            continue
        text = {m: f"{j[m + '_opening']} {j['request']}" for m in ("positive", "negative", "neutral")}
        for name, p, n in CONTRASTS:
            for pol, mood in (("pos", p), ("neg", n)):
                meta = {"contrast": name, "mood": mood, "task": s["task"], "event": s["event"], "gen_id": r["request_id"]}
                items.append(item(CID, s["sid"], "c", pol, [U(text[mood])], "boundary", "generated",
                                  src(**SRC, id=s["sid"]), meta, tag=name))
    report_drops(CID, drops)
    for s, r in zip(sb, rb):
        j = get_json(gb.get(r["request_id"]), keys)
        if j is None:
            continue
        for mood in ("positive", "negative", "neutral"):
            bid = f"{s['sid']}-{mood}"
            behavior.append(behav(CID, bid, [U(f"{j[mood + '_opening']} {j['request']}")], "Isen-Levin optional favor",
                                  {"type": "judge", "prompt": JUDGE.replace("{favor}", "the last request in the message"),
                                   "scale": "1 did the favor, 0 did not; metric = helping rate by mood"},
                                  src(**SRC, id=bid), "generated", {"condition": mood, "task": s["task"],
                                                                     "gen_id": r["request_id"]}))

    d = pd.read_parquet(next(ED.glob("snapshots/*/default/test/*.parquet")))
    d = d.drop_duplicates("conv_id")
    heldout = []
    for pol, emos in (("pos", POS_EMO), ("neg", NEG_EMO)):
        rows = d[d["context"].isin(emos)]
        for _, r in stable_order(list(rows.iterrows()), key=lambda x: x[1]["conv_id"], salt="E7-held")[:50]:
            sid = f"{CID}-H-ed-{r['conv_id'].replace(':', '_')}"
            text = r["prompt"].replace("_comma_", ",").strip()
            heldout.append(held(CID, sid, "c", pol, [U(text)], "boundary", "verbatim", src(**ED_SRC, id=r["conv_id"]),
                                {"contrast": "positive_vs_negative", "emotion": r["context"], "unpaired": True}))
    write_outputs(CID, items, heldout, behavior)


if __name__ == "__main__":
    main()
