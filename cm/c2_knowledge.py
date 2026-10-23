"""D-048 check for C2's frames a and b: does the a/b direction read Gemma's own knowledge of the HaluEval fact?

    python -m cm.c2_knowledge answers    GPU: the subject's greedy answer to each of the 100 bare C2 questions, scored
                                         as known / not known (outputs/acts/stimuli/c2_knowledge.jsonl)
    python -m cm.c2_knowledge strata     CPU: the S3 a/b directions (reply32 and boundary) scored on the a/b test items,
                                         AUROC within the known and the not-known questions (results/s3/c2_knowledge.json)

Known: the right answer (alphanumerics only) appears in the reply, or every distinctive word of it does, or its last
word and more than half of them do; for yes/no answers, the reply opens with that word. Equal AUROC in both strata
means the direction is not the subject's knowledge.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from . import cfg
from .paths import MODELS, snapshot
from .s3 import ACTS, OUT as S3_OUT, auroc, load
from .stimuli.schema import item_dir, read_jsonl, write_jsonl

CID = "C2"
PATH = ACTS / "c2_knowledge.jsonl"


def questions():
    """One row per scenario: the bare question (the frame-c user turn) and the right answer."""
    seen, out = set(), []
    for it in read_jsonl(item_dir(CID) / "items.jsonl"):
        if it["frame"] == "c" and it["scenario_id"] not in seen:
            seen.add(it["scenario_id"])
            out.append({"scenario_id": it["scenario_id"], "question": it["messages"][0]["content"],
                        "right_answer": it["meta"]["right_answer"]})
    return out


STOP = {"the", "and", "of", "for", "with", "from", "that", "this"}


def known(reply: str, right: str) -> bool:
    low, ra = reply.lower(), right.strip().lower().rstrip(".")
    if ra in ("yes", "no"):
        return bool(re.match(r"\W*" + ra + r"\b", low))
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t)
    if norm(ra) and norm(ra) in norm(low):
        return True
    words = [w for w in re.findall(r"[a-z0-9]{3,}", ra) if w not in STOP]
    if not words:
        return False
    hit = [w in low for w in words]
    return all(hit) or (hit[-1] and sum(hit) > len(hit) / 2)


def cmd_answers():
    from vllm import LLM, SamplingParams
    qs = questions()
    kw = dict(model=str(snapshot(MODELS[cfg.MODEL])), dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, **cfg.vllm_kwargs())
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "audio": 0})
    except TypeError:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    stops = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>")
                                   if t in tok.get_vocab()}
    sp = SamplingParams(temperature=0.0, max_tokens=64, stop_token_ids=sorted(i for i in stops if i is not None),
                        skip_special_tokens=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False,
                                       add_generation_prompt=True, **cfg.chat_kwargs()) for q in qs]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    rows = [dict(q, reply=o.outputs[0].text, known=known(o.outputs[0].text, q["right_answer"])) for q, o in zip(qs, outs)]
    write_jsonl(PATH, rows)
    print(f"[c2_knowledge] {cfg.MODEL}: {sum(r['known'] for r in rows)}/{len(rows)} questions answered right", flush=True)


def cmd_strata():
    kn = {r["scenario_id"]: r["known"] for r in read_jsonl(PATH)}
    z, meta, _ = load(CID)
    D = np.load(S3_OUT / "directions" / f"{CID}.npz")
    out = {"model": cfg.MODEL, "n_known": sum(kn.values()), "n_questions": len(kn), "cells": []}
    for frame in "ab":
        te = [i for i, m in enumerate(meta) if m["frame"] == frame and m["split"] == "test"]
        y = np.array([meta[i]["polarity"] == "pos" for i in te], int)
        g = np.array([bool(kn.get(meta[i]["scenario_id"])) for i in te])
        for readout in ("reply32", "boundary"):
            key = f"all|{frame}|{readout}"
            if key not in D.files:
                continue
            for L, idx in cfg.DEPTHS.items():
                u = D[key][idx]
                if not np.any(u):
                    continue
                s = z[readout][te, idx] @ u
                cell = {"frame": frame, "readout": readout, "layer": L, "all": auroc(s, y),
                        "n_known": [int((y[g] == 1).sum()), int((y[g] == 0).sum())],
                        "n_unknown": [int((y[~g] == 1).sum()), int((y[~g] == 0).sum())],
                        "known": auroc(s[g], y[g]) if g.sum() > 3 else None,
                        "unknown": auroc(s[~g], y[~g]) if (~g).sum() > 3 else None}
                out["cells"].append(cell)
                print(f"[c2_knowledge] {frame} {readout:8} L{L:2}: all {cell['all']:.2f}  known {cell['known'] if cell['known'] is None else round(cell['known'], 2)}"
                      f" (n {cell['n_known']})  not known {cell['unknown'] if cell['unknown'] is None else round(cell['unknown'], 2)} (n {cell['n_unknown']})", flush=True)
    S3_OUT.mkdir(parents=True, exist_ok=True)
    (S3_OUT / "c2_knowledge.json").write_text(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["answers", "strata"])
    args = ap.parse_args()
    cmd_answers() if args.stage == "answers" else cmd_strata()


if __name__ == "__main__":
    main()
