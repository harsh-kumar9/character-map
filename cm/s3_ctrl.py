"""D-030(a): S3 with Gemma's own yes/no endorsement projected out of frame-d readouts.

Frame d (endorse form) asks "Is the following statement something you would say?"; Gemma's reply opens with a verdict,
and the frame-d direction largely tracks it. For each concept with frame-d items:
  1. label each frame-d item by how Gemma's greedy reply opens (yes / no / other; outputs/acts/stimuli/replies);
  2. estimate an endorsement direction per readout and hidden-state index as mean(yes) - mean(no) over the frame-d
     train items of all OTHER concepts (leave one concept out, so the control cannot absorb this concept's signal);
  3. project that direction out of this concept's frame-d readouts (all splits) and rerun the identical S3 analysis.
Outputs go to results/s3_endorse_ctrl/ (same layout as results/s3/), plus endorse_stats.json with, per concept, the
yes/no counts by polarity and how well the endorsement direction separates Gemma's own yes and no replies.
Runs on the current endorse-form items; after the D-030(b) rebuild those items live in items_endorse.jsonl.

    python -m cm.s3_ctrl stats              pooled sums and per-concept counts
    python -m cm.s3_ctrl run --concepts all rerun S3 with the control
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .paths import OUTPUTS, RESULTS
from .s3 import ACTS, LAYERS, auroc, directions_from, load
from .stimuli.schema import read_jsonl

OUT = RESULTS / "s3_endorse_ctrl"
READOUTS = ("boundary", "reply32")
YES = re.compile(r"^\W*(yes|yeah|absolutely|definitely|certainly|of course|that's (a |an )?(perfectly|very|fantastic|great|wonderful|excellent|really))", re.I)
NO = re.compile(r"^\W*(no\b|not really|i wouldn't|i would not|that's not|absolutely not|definitely not|i don't think)", re.I)


def verdict(reply: str) -> str:
    r = reply.strip()
    return "no" if NO.match(r) else "yes" if YES.match(r) else "other"   # no first: "Absolutely not."


def frame_d_rows(cid):
    z, meta, resp = load(cid)
    rep = {r["item_id"]: r["reply"] for r in read_jsonl(OUTPUTS / "acts" / "stimuli" / "replies" / f"{cid}.jsonl")}
    rows = [i for i, m in enumerate(meta) if m["frame"] == "d" and m["readout"] == "boundary"]
    v = [verdict(rep.get(meta[i]["item_id"], "")) for i in rows]
    return z, meta, resp, rows, v


def concepts_with_d():
    out = []
    for p in sorted(ACTS.glob("*.npz")):
        meta = json.loads(str(np.load(p)["meta"]))
        if any(m["frame"] == "d" and m["readout"] == "boundary" for m in meta):
            out.append(p.stem)
    return out


def cmd_stats(cids):
    OUT.mkdir(parents=True, exist_ok=True)
    sums, stats = {}, {}
    for c in cids:
        z, meta, resp, rows, v = frame_d_rows(c)
        tr = [(r, vv) for r, vv in zip(rows, v) if meta[r]["split"] == "train" and vv in ("yes", "no")]
        s = {}
        for ro in READOUTS:
            for lab in ("yes", "no"):
                idx = [r for r, vv in tr if vv == lab]
                s[f"{ro}|{lab}"] = (z[ro][idx].sum(0) if idx else np.zeros((len(LAYERS), 2560))).astype(np.float64)
                s[f"n|{lab}"] = len(idx)
        sums[c] = s
        pol = [meta[r]["polarity"] for r in rows]
        stats[c] = {f"{p}_{lab}": sum(1 for pp, vv in zip(pol, v) if pp == p and vv == lab)
                    for p in ("pos", "neg") for lab in ("yes", "no", "other")}
    np.savez(OUT / "pooled_sums.npz", **{f"{c}|{k}": (v if not np.isscalar(v) else np.array(v)) for c, s in sums.items() for k, v in s.items()})
    (OUT / "endorse_counts.json").write_text(json.dumps(stats, indent=1))
    print(f"[ctrl] stats for {len(cids)} concepts", flush=True)


def endorse_dirs(cid):
    """Leave-one-out endorsement directions {readout: [35, 2560] unit vectors}."""
    S = np.load(OUT / "pooled_sums.npz")
    keys = sorted({k.split("|", 1)[0] for k in S.files})
    dirs = {}
    for ro in READOUTS:
        sy = sum(S[f"{c}|{ro}|yes"] for c in keys if c != cid)
        sn = sum(S[f"{c}|{ro}|no"] for c in keys if c != cid)
        ny = sum(int(S[f"{c}|n|yes"]) for c in keys if c != cid)
        nn = sum(int(S[f"{c}|n|no"]) for c in keys if c != cid)
        e = sy / max(ny, 1) - sn / max(nn, 1)
        dirs[ro] = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return dirs, ny, nn


def run_one(cid):
    z, meta, resp, rows, v = frame_d_rows(cid)
    dirs, ny, nn = endorse_dirs(cid)
    rows_a = np.array(rows)
    sep = {}
    for ro in READOUTS:
        X = z[ro][rows_a].astype(np.float64)                       # [n, 35, d]
        proj = np.einsum("nld,ld->nl", X, dirs[ro])               # endorsement score per layer
        yn = np.array([vv == "yes" for vv in v]), np.array([vv == "no" for vv in v])
        sel = yn[0] | yn[1]
        sep[ro] = {str(L): auroc(proj[sel, L], yn[0][sel].astype(int)) for L in (10, 18, 23, 30)} if sel.sum() > 3 else {}
        z[ro][rows_a] = (X - proj[..., None] * dirs[ro][None]).astype(z[ro].dtype)
    directions_from(cid, z, meta, resp, OUT)
    return cid, {"n_pooled_yes": ny, "n_pooled_no": nn, "yes_vs_no_auroc_of_endorsement_dir": sep}


def cmd_run(cids, workers):
    (OUT / "concepts").mkdir(parents=True, exist_ok=True)
    (OUT / "directions").mkdir(parents=True, exist_ok=True)
    stats = {}
    with ProcessPoolExecutor(workers) as ex:
        for c, st in ex.map(run_one, cids):
            stats[c] = st
            print(f"[ctrl] {c} done", flush=True)
    prev = json.loads((OUT / "endorse_stats.json").read_text()) if (OUT / "endorse_stats.json").exists() else {}
    prev.update(stats)
    (OUT / "endorse_stats.json").write_text(json.dumps(prev, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stats", "run"])
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    allc = concepts_with_d()
    cids = allc if args.concepts == "all" else args.concepts.split(",")
    if args.stage == "stats":
        cmd_stats(allc)
    else:
        cmd_run(cids, args.workers)


if __name__ == "__main__":
    main()
