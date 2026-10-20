"""C5 caring vs apathetic. Card: data/concepts/cards/C5.yaml. Frames a, c, d.

Items: EmpatheticDialogues (train split) opening turns, verbatim. gemma-4-31B-it writes a caring and an apathetic reply of
matched length (both generated, so the contrast is not human vs model text). Frame c: the user's turn and the prefilled
reply. Frame a: the person says the same thing to a named friend, who gives the same reply. Frame d: endorse a
first-person statement of attending to them vs giving a detached reply.
Held-out: EPITOME emotional-reactions (Reddit, Sharma et al. 2020), frame c: seeker post and a peer response rated
strong (level 2, pos) or none (level 0, neg). Not matched pairs.
Behavior: EmpatheticDialogues test-split opening turns; judge: the upstream persona-vectors "apathetic" prompt (0-100).
"""
from __future__ import annotations

import json

import pandas as pd

from . import disp1 as D
from ...paths import REPO

CID = "C5"
N_GEN, N_KEEP = 130, 100
EMOTIONS = {"sad", "lonely", "afraid", "devastated", "anxious", "disappointed", "terrified", "ashamed", "embarrassed",
            "guilty", "jealous", "angry", "furious", "annoyed", "apprehensive", "nostalgic", "sentimental", "hopeful",
            "proud", "excited", "grateful", "joyful", "anticipating", "surprised"}


def ed(split):
    p = next((D.hf_dir("facebook/empathetic_dialogues", "refs/convert/parquet") / "default" / split).glob("*.parquet"))
    df = pd.read_parquet(p)
    df = df[(df["utterance_idx"] == 1) & df["context"].isin(EMOTIONS)]
    rows = df.to_dict("records")
    for r in rows:
        r["utterance"] = r["utterance"].replace("_comma_", ",").strip()
        r["prompt"] = r["prompt"].replace("_comma_", ",").strip()
    return [r for r in rows if 12 <= D.words(r["utterance"]) <= 60]


def src_ed(r, split):
    return {"dataset": f"facebook/empathetic_dialogues {split}", "id": r["conv_id"], "license": "CC-BY-NC-4.0",
            "url": "https://github.com/facebookresearch/EmpatheticDialogues"}


def seeds():
    rows = ed("train")
    neg = [r for r in rows if r["context"] not in {"hopeful", "proud", "excited", "grateful", "joyful", "anticipating",
                                                     "surprised"}]
    pos = [r for r in rows if r not in neg]
    return D.shuffled(neg, CID)[:100] + D.shuffled(pos, CID)[:30]   # mostly hardship, some good news to share


PROMPT = """Someone tells a listener:
"{utterance}"
(Background: they feel {emotion}. Their situation: {situation})

Write a JSON object with these keys:
- "caring": the listener's reply, 30-55 words: it notices how they feel, responds to what matters to them, and offers something that fits (a question, comfort or help).
- "apathetic": a reply of the same length that is polite but indifferent: it acknowledges the words only in passing, does not engage with their feelings or needs, and moves on or answers flatly. It is not rude or hostile.
- "self_caring": one first-person sentence, 15-30 words, saying how you would respond if someone told you this: paying attention to how they feel and what they need.
- "self_apathetic": the same sentence, but you would give a brief, detached reply and move on. Same length and wording except for that.
""" + D.RULES


def rid(r):
    return f"{CID}-ed-{r['conv_id'].replace(':', '').replace('hit', 'h').replace('_conv', 'c')}"


def requests():
    return [D.req(rid(r), PROMPT.format(utterance=r["utterance"], emotion=r["context"], situation=r["prompt"]))
            for r in seeds()]


def ok(rid_, j):
    return (D.has(j, ["caring", "apathetic", "self_caring", "self_apathetic"]) and D.ratio_ok(j["caring"], j["apathetic"])
            and D.ratio_ok(j["self_caring"], j["self_apathetic"]))


def items(gen):
    out, n = [], 0
    for r in seeds():
        scn = rid(r)
        g = gen.get(scn)
        if not g or n >= N_KEEP:
            continue
        n += 1
        j, name, u = g["json"], D.name_for(scn), r["utterance"]
        meta = dict(D.gen_meta(g), emotion=r["context"], user_turn="verbatim")
        src = src_ed(r, "train")
        for p, rep, st in (("pos", j["caring"], j["self_caring"]), ("neg", j["apathetic"], j["self_apathetic"])):
            rep, st = D.clean(rep), D.clean(st)
            out.append(D.mk(CID, scn, "a", p, [D.U(D.frame_a(f"Someone tells {name}: \"{u}\"", name, rep, scn))],
                            "generated", src, meta))
            out.append(D.mk(CID, scn, "c", p, [D.U(u), D.A(rep)], "generated", src, meta))
            out.append(D.mk(CID, scn, "d", p, [D.U(D.frame_d(f"Imagine someone tells you: \"{u}\"", st, scn))],
                            "generated", src, meta))
    return out


def heldout():
    df = pd.read_csv(D.RAW / "Empathy-Mental-Health/dataset/emotional-reactions-reddit.csv")
    df = df[df["seeker_post"].str.split().str.len().between(8, 120) & df["response_post"].str.split().str.len().between(8, 80)]
    out = []
    for lvl, p in ((2, "pos"), (0, "neg")):
        rows = D.shuffled(df[df["level"] == lvl].to_dict("records"), CID + "held")[:100]
        for r in rows:
            scn = f"{CID}-epi-{r['sp_id']}-{r['rp_id']}"
            src = {"dataset": "behavioral-data/Empathy-Mental-Health emotional-reactions (EPITOME)",
                   "id": f"{r['sp_id']}/{r['rp_id']}", "license": "Althoff academic research license",
                   "url": "https://github.com/behavioral-data/Empathy-Mental-Health"}
            out.append(D.mk(CID, scn, "c", p, [D.U(r["seeker_post"].strip()), D.A(r["response_post"].strip())],
                            "verbatim", src, {"empathy_level": lvl, "paired": False}, split="heldout"))
    return out


def behavior():
    j = json.loads((REPO / "vendor/persona_vectors/trait_data_eval/apathetic.json").read_text())
    rows = D.shuffled(ed("test"), CID + "beh")[:200]
    return [D.beh(CID, f"{CID}-beh-{rid(r)[6:]}", [D.U(r["utterance"])], "EmpatheticDialogues reply apathy",
                  {"type": "judge", "prompt": j["eval_prompt"], "scale": "0-100, higher = more apathetic"},
                  src_ed(r, "test"), {"emotion": r["context"]}, "verbatim") for r in rows]


def build(gen):
    return items(gen), heldout(), behavior()


if __name__ == "__main__":
    D.build([CID])
