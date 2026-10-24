"""C15 harm x refusal 2x2 (D-054, D-055): is seeing that a request is harmful a different direction from refusing it?

C15 holds the request fixed within a scenario (refuse vs comply, prefilled) and its scenarios carry meta.request_type:
harmful 40, harmless 30, borderline 30 per pole and frame. That crossing gives, per frame (c, d), readout and depth:
  refusal|harmful   refuse vs comply on harmful requests (refusal proper)
  refusal|harmless  refuse vs comply on harmless requests (over-refusal)
  harm|refuse       harmful vs harmless request among refusing replies
  harm|comply       the same among complying replies
  harm|both         harmful vs harmless, both replies pooled
Readouts: boundary (the last token before the reply: the request only, so refuse and comply items are identical
there and only harm directions exist; Zhao et al. 2025 read harmfulness at this position) and reply32.

Every hidden dimension is standardized over the frame's C15 items at that readout and depth (as S4 does); directions
are train-split diffs of means; scores are test-split AUROCs. For each pair of directions: cosine, the 95th
percentile of |cosine| between label-shuffled directions (floor), and the cosine divided by the geometric mean of the
two Spearman-Brown split-half reliabilities (disattenuated). Cross scores: a direction trained in one cell applied to
the other cell's test items, and the boundary harm direction applied to reply32 items and back. Cell means of the
test items on the harm and refusal axes, in units of the pooled within-cell SD, including the borderline requests.

    python -m cm.hhh.c15_2x2        (CPU; sbatch -J c15_2x2 scripts/cm_cpu.sbatch cm.hhh.c15_2x2)
Writes results/hhh/c15_2x2.json and prints the summary.
"""
from __future__ import annotations

import json

import numpy as np

from .. import cfg
from ..s3 import auroc, load
from ..stimx import load_items
from .blocks import out_dir

CID = "C15"
DEPTHS = dict(cfg.DEPTHS)
R_SPLIT, R_SHUF = 50, 200
SEED = 0


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def gather():
    z, meta, _ = load(CID)
    rt = {it["item_id"]: (it.get("meta") or {}).get("request_type") for it in load_items(CID)}
    for m in meta:
        m["request_type"] = rt.get(m["item_id"])
    return z, meta


def reliability(X, meta, rows_pos, rows_neg, rng):
    """Spearman-Brown corrected split-half cosine over train scenarios."""
    scen = sorted({meta[i]["scenario_id"] for i in rows_pos + rows_neg})
    if len(scen) < 6:
        return np.nan
    r = []
    for _ in range(R_SPLIT):
        perm = rng.permutation(len(scen)); half = set(scen[k] for k in perm[: len(scen) // 2])
        d = []
        for keep in (lambda s: s in half, lambda s: s not in half):
            p = [i for i in rows_pos if keep(meta[i]["scenario_id"])]; n = [i for i in rows_neg if keep(meta[i]["scenario_id"])]
            if not p or not n:
                break
            d.append(unit(X[p].mean(0) - X[n].mean(0)))
        if len(d) == 2:
            r.append(float(d[0] @ d[1]))
    if not r:
        return np.nan
    m = float(np.mean(r))
    return 2 * m / (1 + m) if m > -1 else np.nan


def shuffled(X, meta, rows_pos, rows_neg, rng, paired):
    """A label-shuffled direction. Paired units (same scenario on both sides) flip each scenario's sign; unpaired
    units permute the labels across scenarios."""
    if paired:
        by = {}
        for i in rows_pos:
            by.setdefault(meta[i]["scenario_id"], [[], []])[0].append(i)
        for i in rows_neg:
            by.setdefault(meta[i]["scenario_id"], [[], []])[1].append(i)
        ds = [X[p].mean(0) - X[n].mean(0) for p, n in by.values() if p and n]
        s = rng.choice([-1.0, 1.0], len(ds))
        return unit(np.einsum("k,kd->d", s, np.array(ds)))
    lab = {}
    for i in rows_pos:
        lab[meta[i]["scenario_id"]] = 1
    for i in rows_neg:
        lab[meta[i]["scenario_id"]] = 0
    scen = sorted(lab); perm = dict(zip(scen, rng.permutation([lab[s] for s in scen])))
    rows = rows_pos + rows_neg
    p = [i for i in rows if perm[meta[i]["scenario_id"]] == 1]; n = [i for i in rows if perm[meta[i]["scenario_id"]] == 0]
    return unit(X[p].mean(0) - X[n].mean(0))


def analyse(z, meta, frame, readout, hidx, rng):
    rows = [i for i, m in enumerate(meta) if m["frame"] == frame and m["split"] in ("train", "test") and m["request_type"]]
    X = z[readout][rows, hidx].astype(np.float64)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
    M = [meta[i] for i in rows]
    idx = lambda **kw: [k for k, m in enumerate(M) if all(m[a] == v for a, v in kw.items())]

    cells = {}
    for rt in ("harmful", "harmless", "borderline"):
        for pol, rep in (("pos", "refuse"), ("neg", "comply")):
            cells[(rt, rep)] = {s: idx(request_type=rt, polarity=pol, split=s) for s in ("train", "test")}

    def spec(name):
        kind, c = name.split("|")
        if kind == "refusal":
            return cells[(c, "refuse")], cells[(c, "comply")], True
        reps = ["refuse", "comply"] if c == "both" else [c]
        P = {s: sum((cells[("harmful", r)][s] for r in reps), []) for s in ("train", "test")}
        N = {s: sum((cells[("harmless", r)][s] for r in reps), []) for s in ("train", "test")}
        return P, N, False

    names = (["refusal|harmful", "refusal|harmless"] if readout == "reply32" else []) + ["harm|refuse", "harm|comply", "harm|both"]
    U, out = {}, {"units": {}, "pairs": {}, "cross": {}, "cells": {}}
    for nm in names:
        P, N, paired = spec(nm)
        if min(len(P["train"]), len(N["train"])) < 5 or min(len(P["test"]), len(N["test"])) < 3:
            continue
        d = X[P["train"]].mean(0) - X[N["train"]].mean(0)
        te = P["test"] + N["test"]
        a = auroc(X[te] @ d, np.r_[np.ones(len(P["test"])), np.zeros(len(N["test"]))])
        rel = reliability(X, M, P["train"], N["train"], rng)
        sh = [shuffled(X, M, P["train"], N["train"], rng, paired) for _ in range(R_SHUF)]
        U[nm] = (unit(d), P, N, rel, sh)
        out["units"][nm] = {"auroc": a, "reliability": rel, "n_train": [len(P["train"]), len(N["train"])],
                            "n_test": [len(P["test"]), len(N["test"])]}
    for i, a in enumerate(U):
        for b in list(U)[i + 1:]:
            ua, ub = U[a][0], U[b][0]
            c = float(ua @ ub)
            floor = float(np.percentile([abs(x @ y) for x, y in zip(U[a][4], U[b][4])], 95))
            dis = c / np.sqrt(U[a][3] * U[b][3]) if U[a][3] > 0 and U[b][3] > 0 else np.nan
            out["pairs"][f"{a} ~ {b}"] = {"cos": c, "floor95": floor, "disattenuated": float(dis)}
    # cross: a refusal direction from one cell on the other cell's test pairs; a harm direction from one reply type
    # on the other reply type's test items
    for a, b in (("refusal|harmful", "refusal|harmless"), ("refusal|harmless", "refusal|harmful"),
                 ("harm|refuse", "harm|comply"), ("harm|comply", "harm|refuse")):
        if a in U and b in U:
            _, P, N, _, _ = U[b]
            te = P["test"] + N["test"]
            out["cross"][f"{a} -> {b}"] = auroc(X[te] @ U[a][0], np.r_[np.ones(len(P["test"])), np.zeros(len(N["test"]))])
    # cell means on the harm and refusal axes (test items), in pooled within-cell SDs
    axes = {k: U[k][0] for k in ("harm|both", "refusal|harmful") if k in U}
    for ax, u in axes.items():
        proj = X @ u
        sd = np.sqrt(np.mean([proj[v["test"]].var() for v in cells.values() if len(v["test"]) > 1]))
        out["cells"][ax] = {f"{rt}/{rep}": float(proj[v["test"]].mean() / sd) for (rt, rep), v in cells.items() if v["test"]}
    return out, {k: v[0] for k, v in U.items()}, X, M


def main():
    rng = np.random.default_rng(SEED)
    z, meta = gather()
    res = {"concept": CID, "depths": DEPTHS, "frames": {}}
    for f in ("c", "d"):
        res["frames"][f] = {}
        for L, h in DEPTHS.items():
            r = {}
            dirs = {}
            for ro in ("boundary", "reply32"):
                r[ro], dirs[ro], Xr, Mr = analyse(z, meta, f, ro, h, rng)
            # the boundary harm direction on reply32 items (does the request's harm survive into the reply?) and the
            # reply32 refusal direction on boundary items (does the refusal axis already read the request's harm?)
            rows = [i for i, m in enumerate(meta) if m["frame"] == f and m["split"] == "test" and m["request_type"] in ("harmful", "harmless")]
            y = np.array([meta[i]["request_type"] == "harmful" for i in rows], float)
            xb = z["boundary"][rows, h].astype(np.float64); xr = z["reply32"][rows, h].astype(np.float64)
            rb = [i for i, m in enumerate(meta) if m["frame"] == f and m["split"] in ("train", "test") and m["request_type"]]
            B, R = z["boundary"][rb, h].astype(np.float64), z["reply32"][rb, h].astype(np.float64)
            mu_b, sd_b = B.mean(0), np.maximum(B.std(0), 1e-6)
            mu_r, sd_r = R.mean(0), np.maximum(R.std(0), 1e-6)
            r["readout_cross"] = {}
            if "harm|both" in dirs["boundary"]:
                r["readout_cross"]["harm(boundary) on reply32: harmful vs harmless"] = auroc(((xr - mu_r) / sd_r) @ dirs["boundary"]["harm|both"], y)
            if "refusal|harmful" in dirs["reply32"]:
                r["readout_cross"]["refusal(reply32) on boundary: harmful vs harmless"] = auroc(((xb - mu_b) / sd_b) @ dirs["reply32"]["refusal|harmful"], y)
            res["frames"][f][str(L)] = r
            print(f"\nframe {f}, layer {L}")
            for ro in ("boundary", "reply32"):
                for nm, v in r[ro]["units"].items():
                    print(f"  {ro:8s} {nm:17s} AUROC {v['auroc']:.2f}  rel {v['reliability']:.2f}  n {v['n_train']}/{v['n_test']}")
                for nm, v in r[ro]["pairs"].items():
                    print(f"  {ro:8s} cos {nm:36s} {v['cos']:+.2f} (floor {v['floor95']:.2f}, disatt {v['disattenuated']:+.2f})")
                for nm, v in r[ro]["cross"].items():
                    print(f"  {ro:8s} cross {nm:34s} {v:.2f}")
                for ax, cm in r[ro]["cells"].items():
                    print(f"  {ro:8s} cells on {ax:15s} " + "  ".join(f"{k} {v:+.2f}" for k, v in cm.items()))
            for nm, v in r["readout_cross"].items():
                print(f"  {nm}: {v:.2f}")
    (out_dir() / "c15_2x2.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
