"""Audit of reversed held-out transfer for D/E concepts (S3 follow-up). CPU only; reads the S3 activations.

For each concept, frame and readout, at a few hidden-state indices: the train diff-of-means direction (as in cm.s3),
its test and held-out AUROC, the cosine between it and the held-out pos-minus-neg mean, the size of that held-out
difference relative to the train one, the held-out AUROC along its own mean difference (how separable held-out
pos/neg are at all), and the spread of held-out AUROC over 200 bootstrap resamples of the train items.

    python -m cm.stimuli.build._de_audit --acts /ada1/u/harsh/character-map/outputs/acts/stimuli D1 E1 E2 E4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...s3 import auroc
from ..schema import item_dir, read_jsonl

IDX = [10, 14, 18, 22, 26, 30]


def load(path: Path):
    f = np.load(path)
    z = {k: f[k] for k in ("boundary", "reply32")}
    return z, json.loads(str(f["meta"]))


def subsets(cid, frame, meta):
    """Named held-out subsets that isolate the construct (row indices into meta)."""
    if (cid, frame) != ("E1", "d"):
        return {}
    import re
    text = {r["item_id"]: r["messages"][0]["content"] for r in read_jsonl(item_dir(cid) / "heldout.jsonl")}
    obs = re.compile(r"watch|monitor|supervis|observ|surveil|oversee|undetected|hidden|\bhide\b|attention|privacy|"
                     r"know what i|knew what i|no humans|without humans|involving humans", re.I)
    rows = [i for i, m in enumerate(meta) if m["split"] == "heldout" and m["frame"] == frame]
    keep = [i for i in rows if obs.search(text[meta[i]["item_id"]].split("\n", 1)[1])]
    return {"observation statements only": keep}


def audit(cid, acts, rng):
    z, meta = load(acts / f"{cid}.npz")
    out = []
    for frame in sorted({m["frame"] for m in meta if m["split"] == "heldout"}):
        rows = [i for i, m in enumerate(meta) if m["frame"] == frame]
        tr = [i for i in rows if meta[i]["split"] == "train"]
        te = [i for i in rows if meta[i]["split"] == "test"]
        ho = [i for i in rows if meta[i]["split"] == "heldout"]
        if not tr:
            continue
        y = lambda ix: np.array([meta[i]["polarity"] == "pos" for i in ix], int)
        ytr, yte, yho = y(tr), y(te), y(ho)
        subs = subsets(cid, frame, meta)
        for ro in ("boundary", "reply32"):
            for L in IDX:
                Xtr, Xte, Xho = z[ro][tr, L], z[ro][te, L], z[ro][ho, L]
                v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
                if not np.linalg.norm(v):
                    continue
                u = v / np.linalg.norm(v)
                dho = Xho[yho == 1].mean(0) - Xho[yho == 0].mean(0)
                boot = []
                for _ in range(200):
                    b = rng.integers(0, len(tr), len(tr))
                    vb = Xtr[b][ytr[b] == 1].mean(0) - Xtr[b][ytr[b] == 0].mean(0)
                    boot.append(auroc(Xho @ vb, yho))
                rec = {"concept": cid, "frame": frame, "readout": ro, "idx": L,
                       "test": round(float(auroc(Xte @ u, yte)), 3), "heldout": round(float(auroc(Xho @ u, yho)), 3),
                       "cos_u_dho": round(float(u @ dho / np.linalg.norm(dho)), 3),
                       "dho_over_dtrain": round(float(np.linalg.norm(dho) / np.linalg.norm(v)), 2),
                       "heldout_own_axis": round(float(auroc(Xho @ dho, yho)), 3),
                       "boot_heldout_p5_p95": [round(float(np.percentile(boot, 5)), 2),
                                               round(float(np.percentile(boot, 95)), 2)]}
                for name, ix in subs.items():
                    rec[name] = round(float(auroc(z[ro][ix, L] @ u, y(ix))), 3)
                    rec[name + " (n pos, n neg)"] = [int(y(ix).sum()), int(len(ix) - y(ix).sum())]
                out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("concepts", nargs="+")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for c in a.concepts:
        for r in audit(c, a.acts, rng):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
