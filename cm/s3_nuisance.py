"""Nuisance-direction controls for S3: project a surface confound's direction out and rerun the S3 analysis.

Two nuisances, both ruled in (docs/DECISIONS.md, "Rulings D-033 to D-036"):
  verdict  (D-033c)  frames a and b are read over Gemma's own reply, which opens with its verdict. gemma-4-31B-it labels
                     the opening of each a/b reply: P (positive verdict: yes, agrees, approves, says it is so),
                     N (negative: no, disagrees, disapproves, says it is not so) or X (no verdict). Projected out of
                     the a/b rows' reply32 only (frames c and d carry no verdict of Gemma's).
  negation (D-036)   a negation word in the text a readout sees: the last user turn for the boundary readout, the
                     first 32 reply tokens for reply32, the whole reply for response_all. Projected out of every row.
Direction per concept, readout and hidden-state index, leave one concept out (as D-030a): for every other concept and
polarity with at least MIN rows on each side of the label, mean(label 1) - mean(label 0) over train rows; the cell
differences are averaged with equal weight, so neither a large concept nor the concept's own pole drives it.
Outputs go to results/s3_ctrl_<nuisance>/ (same layout as results/s3/) plus labels_summary.json (label counts by
concept, frame and polarity) and dirs_summary.json (how well each direction separates its own labels on the
concept's rows).

    python -m cm.s3_nuisance label-verdict [--concepts all]     GPU: gemma-4-31B-it labels, outputs/acts/stimuli/verdicts
    python -m cm.s3_nuisance stats --nuisance negation          CPU: per-concept label sums
    python -m cm.s3_nuisance run --nuisance negation            CPU: project out and rerun S3
    python -m cm.s3_nuisance strata                             CPU: S3 directions scored with and without negation
"""
from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from . import cfg
from .paths import MODELS, OUTPUTS, RESULTS, snapshot
from .s3 import ACTS, LAYERS, auroc, directions_from, load
from .stimuli.audit_lexical import NEG
from .stimuli.schema import CARDS, item_dir, read_jsonl, write_jsonl

READOUTS = ("boundary", "reply32", "response_all")
VERDICTS = ACTS / "verdicts"
JUDGE = "google/gemma-4-31B-it"
MIN = 5
VERDICT_PROMPT = """Below is a message to an AI assistant and the opening of the assistant's reply.

Message:
{question}

Opening of the reply:
{reply}

Does the opening of the reply give a verdict? A verdict is an evaluation or a yes/no answer: that something is right
or wrong, acceptable or not, good or bad, true or false, or that the answer to a yes/no question is yes or no.
P = a positive verdict: yes, right, acceptable, good, agrees, approves. A qualified verdict that leans positive
    ("good, but somewhat vague") counts as P.
N = a negative verdict: no, wrong, unacceptable, bad, disagrees, disapproves. A qualified verdict that leans negative
    ("understandable, but wrong") counts as N.
X = no verdict: the opening names, classifies or describes the situation ("This is a classic example of ..."), says
    there is no single answer, asks a question, or is balanced without leaning either way.
Answer with one letter: P, N or X."""


def out_dir(nuisance):
    return cfg.results_dir() / f"s3_ctrl_{nuisance}"


def all_concepts():
    return sorted(p.stem for p in CARDS.glob("*.yaml") if not p.stem.startswith("_") and (ACTS / f"{p.stem}.npz").exists())


def items_by_id(cid):
    d = item_dir(cid)
    return {it["item_id"]: it for it in read_jsonl(d / "items.jsonl") + read_jsonl(d / "heldout.jsonl")}


def replies(cid):
    return {r["item_id"]: r["reply"] for r in read_jsonl(ACTS / "replies" / f"{cid}.jsonl")}


# ---- verdict labels (GPU) ----

def cmd_label_verdict(cids):
    from vllm import LLM, SamplingParams
    from .pv.judge import hetero_safe_overrides
    model_id = JUDGE   # offline cache via HF_HUB_OFFLINE, as cm.pv.judge and cm.stimuli.generate
    kw = dict(model=model_id, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, hf_overrides=partial(hetero_safe_overrides, model_id=model_id))
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "audio": 0})
    except TypeError:
        llm = LLM(**kw)
    tok = llm.get_tokenizer()
    letters = {k: tok.convert_tokens_to_ids(k) for k in "PNX"}
    assert all(tok.tokenize(k) == [k] for k in "PNX"), "P/N/X are not single tokens"
    sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    VERDICTS.mkdir(parents=True, exist_ok=True)
    for c in cids:
        path = VERDICTS / f"{c}.jsonl"
        if path.exists():
            continue
        its, rep = items_by_id(c), replies(c)
        todo = [it for it in its.values() if it["frame"] in ("a", "b") and it["readout"] == "boundary"]
        if not todo:
            continue
        prompts = []
        for it in todo:
            q = [m["content"] for m in it["messages"] if m["role"] == "user"][-1]
            text = VERDICT_PROMPT.format(question=q, reply=rep.get(it["item_id"], ""))
            prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False))
        outs = llm.generate(prompts, sp, use_tqdm=False)
        rows = []
        for it, o in zip(todo, outs):
            lp = o.outputs[0].logprobs[0]
            p = {k: math.exp(lp[t].logprob) if t in lp else 0.0 for k, t in letters.items()}
            tot = sum(p.values())
            lab = max(p, key=p.get) if tot >= 0.5 else "?"
            rows.append({"item_id": it["item_id"], "label": lab, **{f"p_{k}": v for k, v in p.items()}})
        write_jsonl(path, rows)
        print(f"[verdict] {c}: {len(rows)} a/b items, " + ", ".join(f"{k} {sum(r['label'] == k for r in rows)}"
                                                                  for k in "PNX?"), flush=True)


# ---- labels per acts row ----

_TOK = None


def first_tokens(text, n=32):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(str(snapshot(MODELS[cfg.MODEL])))
    ids = _TOK(text, add_special_tokens=False)["input_ids"][:n]
    return _TOK.decode(ids)


def row_labels(cid, meta, nuisance):
    """{readout: array of 1 / 0 / -1 (unlabelled) per acts row} and the rows the projection applies to."""
    n = len(meta)
    lab = {ro: np.full(n, -1, int) for ro in READOUTS}
    if nuisance == "verdict":
        v = {r["item_id"]: r["label"] for r in read_jsonl(VERDICTS / f"{cid}.jsonl")}
        apply = np.array([m["frame"] in ("a", "b") and m["readout"] == "boundary" for m in meta])
        for i, m in enumerate(meta):
            if apply[i] and v.get(m["item_id"]) in ("P", "N"):
                lab["reply32"][i] = int(v[m["item_id"]] == "P")
        return lab, {"reply32": apply}
    its, rep = items_by_id(cid), replies(cid)
    for i, m in enumerate(meta):
        it = its.get(m["item_id"])
        if it is None:
            continue
        cue = [x["content"] for x in it["messages"] if x["role"] == "user"][-1]
        reply = it["messages"][-1]["content"] if it["readout"] == "response" else rep.get(m["item_id"], "")
        lab["boundary"][i] = int(bool(NEG.search(cue)))
        lab["reply32"][i] = int(bool(NEG.search(first_tokens(reply))))
        if it["readout"] == "response":
            lab["response_all"][i] = int(bool(NEG.search(reply)))
    allrows = np.ones(n, bool)
    return lab, {ro: allrows for ro in READOUTS}


# ---- stats and run (CPU) ----

def cmd_stats(cids, nuisance):
    """Pooled per-label sums for one nuisance, or for several at once ("verdict,negation"): each concept's activations
    are read once and every requested nuisance is accumulated from that read (the reads dominate on NFS)."""
    nuisances = nuisance.split(",")
    if len(nuisances) > 1:
        acc = {n: ({}, {}) for n in nuisances}
        for c in cids:
            z, meta, resp = load(c)
            for n in nuisances:
                _stats_one(c, z, meta, resp, n, *acc[n])
        for n in nuisances:
            _stats_write(n, *acc[n])
        return
    sums, counts = {}, {}
    for c in cids:
        z, meta, resp = load(c)
        _stats_one(c, z, meta, resp, nuisance, sums, counts)
    _stats_write(nuisance, sums, counts)


def _stats_write(nuisance, sums, counts):
    out = out_dir(nuisance)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "pooled_sums.npz", **sums)
    (out / "pooled_counts.json").write_text(json.dumps(counts))


def _stats_one(c, z, meta, resp, nuisance, sums, counts):
    out = out_dir(nuisance)
    out.mkdir(parents=True, exist_ok=True)
    if True:
        lab, _ = row_labels(c, meta, nuisance)
        for ro in READOUTS:
            for pol in ("pos", "neg"):
                for y in (0, 1):
                    rows = [i for i, m in enumerate(meta) if m["split"] == "train" and m["polarity"] == pol and lab[ro][i] == y]
                    if ro == "response_all":
                        rows = [i for i in rows if i in resp]
                        X = np.stack([z[ro][resp[i]] for i in rows]) if rows else None
                    else:
                        X = z[ro][rows] if rows else None
                    key = f"{c}|{ro}|{pol}|{y}"
                    counts[key] = len(rows)
                    if rows:
                        sums[key] = X.astype(np.float64).sum(0)
        summary = {}
        for i, m in enumerate(meta):
            for ro in READOUTS:
                if lab[ro][i] >= 0:
                    k = f"{ro}|{m['frame']}|{m['polarity']}"
                    summary.setdefault(k, [0, 0])[lab[ro][i]] += 1
        prev = json.loads((out / "labels_summary.json").read_text()) if (out / "labels_summary.json").exists() else {}
        prev[c] = {k: {"label0": v[0], "label1": v[1]} for k, v in sorted(summary.items())}
        (out / "labels_summary.json").write_text(json.dumps(prev, indent=1))
        print(f"[{nuisance}] stats {c}", flush=True)


def nuisance_dirs(cid, nuisance):
    out = out_dir(nuisance)
    S = np.load(out / "pooled_sums.npz")
    N = json.loads((out / "pooled_counts.json").read_text())
    cids = sorted({k.split("|")[0] for k in N})
    dirs, used = {}, {}
    for ro in READOUTS:
        diffs = []
        for c in cids:
            if c == cid:
                continue
            for pol in ("pos", "neg"):
                k1, k0 = f"{c}|{ro}|{pol}|1", f"{c}|{ro}|{pol}|0"
                if N.get(k1, 0) >= MIN and N.get(k0, 0) >= MIN:
                    diffs.append(S[k1] / N[k1] - S[k0] / N[k0])
        used[ro] = len(diffs)
        if diffs:
            e = np.mean(diffs, 0)
            dirs[ro] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return dirs, used


def run_one(cid, nuisance):
    z, meta, resp = load(cid)
    lab, apply = row_labels(cid, meta, nuisance)
    dirs, used = nuisance_dirs(cid, nuisance)
    sep = {}
    for ro, u in dirs.items():
        if ro not in apply:
            continue
        rows = np.where(apply[ro])[0]
        if ro == "response_all":
            rows = np.array([i for i in rows if i in resp], int)
            idx = np.array([resp[i] for i in rows], int)
        else:
            idx = rows
        if len(idx) == 0:
            continue
        X = z[ro][idx].astype(np.float64)
        proj = np.einsum("nld,ld->nl", X, u)
        y = lab[ro][rows]
        sel = y >= 0
        if sel.sum() > 3 and len(set(y[sel])) == 2:
            sep[ro] = {str(L): auroc(proj[sel, L], y[sel]) for L in cfg.DEPTHS.values()}
        z[ro][idx] = (X - proj[..., None] * u[None]).astype(z[ro].dtype)
    out = out_dir(nuisance)
    directions_from(cid, z, meta, resp, out)
    return cid, {"cells_used": used, "label_auroc_of_dir": sep}


def cmd_run(cids, nuisance, workers):
    out = out_dir(nuisance)
    (out / "concepts").mkdir(parents=True, exist_ok=True)
    (out / "directions").mkdir(parents=True, exist_ok=True)
    stats = {}
    with ProcessPoolExecutor(workers) as ex:
        for c, st in ex.map(partial(run_one, nuisance=nuisance), cids):
            stats[c] = st
            print(f"[{nuisance}] {c} done", flush=True)
    prev = json.loads((out / "dirs_summary.json").read_text()) if (out / "dirs_summary.json").exists() else {}
    prev.update(stats)
    (out / "dirs_summary.json").write_text(json.dumps(prev, indent=1, default=float))


def strata_one(cid):
    """Matched-negation check: each S3 reply32 direction (results/s3/directions) scored on the same frame's test items,
    split by whether the reply carries a negation. A direction that still separates the poles among replies with no
    negation is not reading "not" alone. Cells with fewer than MIN items of a pole in a stratum are left out."""
    from .s3 import OUT as S3_OUT
    z, meta, resp = load(cid)
    lab, _ = row_labels(cid, meta, "negation")
    D = np.load(S3_OUT / "directions" / f"{cid}.npz")
    rows_out = []
    for key in D.files:
        sub, frame, ro = key.split("|")
        if ro != "reply32":
            continue
        te = [i for i, m in enumerate(meta) if m["frame"] == frame and m["split"] == "test" and
              (sub == "all" or m["contrast"] == sub)]
        if not te:
            continue
        y = np.array([meta[i]["polarity"] == "pos" for i in te], int)
        g = lab["reply32"][te]
        for L, idx in {9: 10, 17: 18, 22: 23, 29: 30}.items():
            u = D[key][idx]
            if not np.any(u):
                continue
            s = z["reply32"][te, idx] @ u
            row = {"concept": cid, "sub": sub, "frame": frame, "layer": L, "all": auroc(s, y) if len(set(y)) == 2 else None}
            for name, gv in (("no_negation", 0), ("negation", 1)):
                m_ = g == gv
                npos, nneg = int((y[m_] == 1).sum()), int((y[m_] == 0).sum())
                row[f"n_{name}"] = [npos, nneg]
                row[name] = auroc(s[m_], y[m_]) if min(npos, nneg) >= MIN else None
            rows_out.append(row)
    return rows_out


def cmd_strata(cids, workers):
    out = out_dir("negation")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(strata_one, cids):
            rows += r
    (out / "strata.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"[strata] {len(rows)} cells", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["label-verdict", "stats", "run", "strata"])
    ap.add_argument("--nuisance", default="negation", help="verdict | negation | verdict,negation (stats only)")
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    allc = all_concepts()
    cids = allc if args.concepts == "all" else args.concepts.split(",")
    if args.stage == "label-verdict":
        cmd_label_verdict(cids)
    elif args.stage == "stats":
        cmd_stats(allc, args.nuisance)
    elif args.stage == "strata":
        cmd_strata(cids, args.workers)
    else:
        cmd_run(cids, args.nuisance, args.workers)


if __name__ == "__main__":
    main()
