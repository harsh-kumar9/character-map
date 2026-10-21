"""S3: dense directions per concept, frame and layer, and the within-concept frame table ("knowing vs doing").

Inputs: outputs/acts/stimuli/<ID>.npz from cm.stimx acts (readouts boundary, reply32, response_all; [n, 35, 2560]).
Per concept, sub-contrast (meta.contrast, or "all"), frame, readout and hidden-state index:
  - diff-of-means direction on the train split (D-011), scored by AUROC of the projection on the test split, with a
    permutation p-value (label shuffles on test) and Benjamini-Hochberg across all tests of this run;
  - a logistic-probe baseline (standardized, L2) at the same split;
  - cross-frame transfer: the frame-i direction scored on frame-j test items (the knowing-vs-doing table), on reply32;
    for frames read at the boundary also from the boundary direction (transfer_boundary), since their reply32 is
    Gemma's own reply and opens with its verdict;
  - held-out transfer: the train direction scored on the concept's held-out items of the same frame;
  - cosines between frame directions, raw and centred by subtracting the across-concept mean direction (D-012, G6).
Readouts per frame: boundary for items read at the boundary; reply32 for every item; response_all for prefilled
replies. The common readout across all four frames is reply32.
CPU only; submit through scripts/cm_cpu.sbatch (SLURM, mira/vega).

    python -m cm.s3 directions --concepts all          per-concept JSON in results/s3/concepts/ (--force to redo)
    python -m cm.s3 table                              summary tables in results/s3/
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .paths import OUTPUTS, RESULTS
from .stimuli.schema import CARDS, item_dir

ACTS = OUTPUTS / "acts" / "stimuli"
OUT = RESULTS / "s3"
LAYERS = list(range(35))
SAE_INDEX = {9: 10, 17: 18, 22: 23, 29: 30}   # decoder layer -> hidden-state index
N_PERM = 1000


def auroc(score, y):
    """AUROC (Mann-Whitney, ties counted half); y in {0, 1}."""
    from scipy.stats import rankdata
    score, y = np.asarray(score, float), np.asarray(y, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def perm_p(score, y, a, rng, n=N_PERM):
    if np.isnan(a):
        return np.nan
    null = np.array([auroc(score, rng.permutation(y)) for _ in range(n)])
    return (1 + np.sum(np.abs(null - 0.5) >= abs(a - 0.5))) / (n + 1)


def load(cid):
    """Load each readout array once. Indexing an NpzFile re-reads and decompresses the whole array on every access,
    which made the first S3 run take hours."""
    f = np.load(ACTS / f"{cid}.npz")
    z = {k: f[k] for k in ("boundary", "reply32", "response_all")}
    meta = json.loads(str(f["meta"]))
    resp = {int(k): j for j, k in enumerate(f["response_rows"])}
    return z, meta, resp


def readout_matrix(z, resp, readout, rows, layer):
    if readout == "response_all":
        return np.stack([z["response_all"][resp[r], layer] for r in rows])
    return z[readout][rows, layer]


def readouts_for(meta_rows):
    kinds = {m["readout"] for m in meta_rows}
    return (["boundary"] if "boundary" in kinds else []) + ["reply32"] + (["response_all"] if kinds == {"response"} else [])


def probe_auc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return np.nan
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return auroc(clf.decision_function(sc.transform(Xte)), yte)


def directions(cid):
    z, meta, resp = load(cid)
    return directions_from(cid, z, meta, resp, OUT)


def directions_from(cid, z, meta, resp, out):
    rng = np.random.default_rng(0)
    subs = sorted({m["contrast"] for m in meta if m["contrast"]}) or [None]
    res = {"concept": cid, "subcontrasts": [s or "all" for s in subs], "cells": [], "transfer": [],
           "transfer_boundary": [], "cosine": []}
    dirs = {}
    for sub in subs:
        sel = [i for i, m in enumerate(meta) if (sub is None or m["contrast"] == sub)]
        frames = sorted({meta[i]["frame"] for i in sel})
        for f in frames:
            rows_f = [i for i in sel if meta[i]["frame"] == f]
            for ro in readouts_for([meta[i] for i in rows_f]):
                tr = [i for i in rows_f if meta[i]["split"] == "train"]
                te = [i for i in rows_f if meta[i]["split"] == "test"]
                ho = [i for i in rows_f if meta[i]["split"] == "heldout"]
                ytr = np.array([meta[i]["polarity"] == "pos" for i in tr], int)
                yte = np.array([meta[i]["polarity"] == "pos" for i in te], int)
                yho = np.array([meta[i]["polarity"] == "pos" for i in ho], int)
                if len(set(ytr)) < 2 or len(set(yte)) < 2:
                    continue
                for L in LAYERS:
                    Xtr, Xte = readout_matrix(z, resp, ro, tr, L), readout_matrix(z, resp, ro, te, L)
                    v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
                    nv = np.linalg.norm(v)
                    if nv == 0:
                        continue
                    u = v / nv
                    dirs[(sub, f, ro, L)] = u
                    s_te = Xte @ u
                    a = auroc(s_te, yte)
                    cell = {"sub": sub or "all", "frame": f, "readout": ro, "layer": L, "n_train": len(tr),
                            "n_test": len(te), "auroc_dm": a, "norm": float(nv)}
                    if L in SAE_INDEX.values() or L % 4 == 2:
                        cell["p_perm"] = perm_p(s_te, yte, a, rng)
                        cell["auroc_probe"] = probe_auc(Xtr, ytr, Xte, yte)
                    if len(ho) and len(set(yho)) == 2:
                        cell["auroc_heldout"] = auroc(readout_matrix(z, resp, ro, ho, L) @ u, yho)
                        cell["n_heldout"] = len(ho)
                    res["cells"].append(cell)
        # cross-frame transfer on the common readout (reply32), at every layer
        for fi in frames:
            for fj in frames:
                te = [i for i in sel if meta[i]["frame"] == fj and meta[i]["split"] == "test"]
                y = np.array([meta[i]["polarity"] == "pos" for i in te], int)
                if len(set(y)) < 2:
                    continue
                for L in LAYERS:
                    u = dirs.get((sub, fi, "reply32", L))
                    if u is None:
                        continue
                    res["transfer"].append({"sub": sub or "all", "from": fi, "to": fj, "layer": L,
                                            "auroc": auroc(readout_matrix(z, resp, "reply32", te, L) @ u, y)})
                    # frames read at the boundary: their reply32 is Gemma's own reply, which opens with its verdict;
                    # also score the boundary direction (before any verdict) on the target's reply32
                    ub = dirs.get((sub, fi, "boundary", L))
                    if ub is not None:
                        res["transfer_boundary"].append({"sub": sub or "all", "from": fi, "to": fj, "layer": L,
                                                         "auroc": auroc(readout_matrix(z, resp, "reply32", te, L) @ ub, y)})
        for L in LAYERS:
            for a_ in range(len(frames)):
                for b_ in range(a_ + 1, len(frames)):
                    ua, ub = dirs.get((sub, frames[a_], "reply32", L)), dirs.get((sub, frames[b_], "reply32", L))
                    if ua is not None and ub is not None:
                        res["cosine"].append({"sub": sub or "all", "a": frames[a_], "b": frames[b_], "layer": L,
                                              "cos": float(ua @ ub)})
    np.savez(out / "directions" / f"{cid}.npz",
             **{f"{s or 'all'}|{f}|{ro}": np.stack([dirs.get((s, f, ro, L), np.zeros(2560)) for L in LAYERS])
                for (s, f, ro) in sorted({k[:3] for k in dirs}, key=str)})
    (out / "concepts" / f"{cid}.json").write_text(json.dumps(res, default=float))
    return cid


def cmd_directions(cids, workers, force=False):
    (OUT / "concepts").mkdir(parents=True, exist_ok=True)
    (OUT / "directions").mkdir(parents=True, exist_ok=True)
    todo = [c for c in cids if (ACTS / f"{c}.npz").exists() and (force or not (OUT / "concepts" / f"{c}.json").exists())]
    with ProcessPoolExecutor(workers) as ex:
        for c in ex.map(directions, todo):
            print(f"[s3] {c} done", flush=True)


def bh(pvals):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full(len(p), np.nan)
    if ok.sum():
        idx = np.where(ok)[0][np.argsort(p[ok])]
        m = ok.sum()
        ranked = p[idx] * m / (np.arange(m) + 1)
        q[idx] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def cmd_table():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "concepts").glob("*.json"))]
    allp = [(r["concept"], k, c["p_perm"]) for r in rows for k, c in enumerate(r["cells"]) if "p_perm" in c]
    q = bh([p for _, _, p in allp])
    qmap = {(cid, k): qq for (cid, k, _), qq in zip(allp, q)}
    summary = []
    for r in rows:
        for k, c in enumerate(r["cells"]):
            if (r["concept"], k) in qmap:
                c["q_bh"] = qmap[(r["concept"], k)]
        best = {}
        for c in r["cells"]:
            key = (c["sub"], c["frame"], c["readout"])
            if key not in best or (c["auroc_dm"] or 0) > (best[key]["auroc_dm"] or 0):
                best[key] = c
        for (sub, f, ro), c in sorted(best.items()):
            at = {L: next((x for x in r["cells"] if x["sub"] == sub and x["frame"] == f and x["readout"] == ro
                           and x["layer"] == idx), None) for L, idx in SAE_INDEX.items()}
            summary.append({"concept": r["concept"], "sub": sub, "frame": f, "readout": ro,
                            "best_layer_index": c["layer"], "best_auroc": c["auroc_dm"],
                            "heldout_at_best": c.get("auroc_heldout"),
                            **{f"auroc_L{L}": (x or {}).get("auroc_dm") for L, x in at.items()},
                            **{f"q_L{L}": (x or {}).get("q_bh") for L, x in at.items()}})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    tr = []
    for r in rows:
        by = {}
        for t in r["transfer"]:
            by.setdefault((t["sub"], t["from"], t["to"]), []).append((t["layer"], t["auroc"]))
        for (sub, a, b), vals in sorted(by.items()):
            vals = dict(vals)
            tr.append({"concept": r["concept"], "sub": sub, "from": a, "to": b,
                       "best": max(v for v in vals.values() if v == v), **{f"L{L}": vals.get(i) for L, i in SAE_INDEX.items()}})
    (OUT / "transfer.json").write_text(json.dumps(tr, indent=1, default=float))
    print(f"[s3] {len(summary)} cells, {len(tr)} transfer pairs -> {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["directions", "table"])
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="recompute concepts that already have results")
    args = ap.parse_args()
    cids = (sorted(p.stem for p in CARDS.glob("*.yaml") if not p.stem.startswith("_") and item_dir(p.stem).exists())
            if args.concepts == "all" else args.concepts.split(","))
    if args.stage == "directions":
        cmd_directions(cids, args.workers, args.force)
    else:
        cmd_table()


if __name__ == "__main__":
    main()
