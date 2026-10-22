"""C21 circumplex test (pre-registered in D-035): does a value's direction separate it better from foils that are farther
from it on Schwartz's circle?

C21 v2 contrasts each value (pos) with the same side argued from another of the ten values (neg, meta.contrast_value),
rotating over all nine. Circle order (Schwartz 1992; Schwartz et al. 2012, with tradition and conformity adjacent):
    universalism, benevolence, tradition, conformity, security, power, achievement, hedonism, stimulation,
    self-direction (and back to universalism).
Distance between two values = steps around the circle, 1 (adjacent) to 5 (opposite).
Per frame (b, d), readout and SAE depth, 5-fold cross-validation by scenario within each value: the value's
diff-of-means direction from the other folds scores each held-out pair (pos and neg item of one scenario). Per pair:
correct = pos projects above neg; margin = (pos - neg) along the direction, divided by the SD of training projections.
Prediction: margin and accuracy rise with distance. Test: Spearman rho between distance and margin over pairs, with a
one-sided permutation p (foil distances shuffled within each value, 2000 shuffles).
Writes results/analysis/c21_circumplex.json and prints the per-distance table at layers 17 and 29.

    python -m cm.c21_circumplex
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from .paths import RESULTS
from .s3 import load
from .stimuli.schema import item_dir, read_jsonl

ORDER = ["universalism", "benevolence", "tradition", "conformity", "security", "power", "achievement", "hedonism",
         "stimulation", "self_direction"]
DEPTHS = {9: 10, 17: 18, 22: 23, 29: 30}
K, N_PERM = 5, 2000


def dist(a, b):
    i, j = ORDER.index(a), ORDER.index(b)
    return min(abs(i - j), 10 - abs(i - j))


def fold(sid):
    return int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % K


def main():
    z, meta, resp = load("C21")
    foil = {it["item_id"]: it["meta"]["contrast_value"] for it in read_jsonl(item_dir("C21") / "items.jsonl")}
    rows = [i for i, m in enumerate(meta) if m["split"] in ("train", "test") and m["item_id"] in foil]
    rng = np.random.default_rng(0)
    pairs_out, summary = [], []
    for f in sorted({meta[i]["frame"] for i in rows}):
        rf = [i for i in rows if meta[i]["frame"] == f]
        readouts = ["reply32"] + (["boundary"] if any(meta[i]["readout"] == "boundary" for i in rf) else [])
        # pairs: (value, scenario) -> pos row, neg row
        pr = defaultdict(dict)
        for i in rf:
            pr[(meta[i]["contrast"], meta[i]["scenario_id"])][meta[i]["polarity"]] = i
        pairs = [(v, s, d["pos"], d["neg"]) for (v, s), d in pr.items() if "pos" in d and "neg" in d]
        for ro in readouts:
            for L, idx in DEPTHS.items():
                res = []
                for v in ORDER:
                    pv = [p for p in pairs if p[0] == v]
                    for k in range(K):
                        trn = [p for p in pv if fold(p[1]) != k]
                        tst = [p for p in pv if fold(p[1]) == k]
                        if not trn or not tst:
                            continue
                        P = z[ro][[p[2] for p in trn], idx].astype(np.float64)
                        Nn = z[ro][[p[3] for p in trn], idx].astype(np.float64)
                        u = P.mean(0) - Nn.mean(0)
                        u /= np.linalg.norm(u)
                        sd = np.concatenate([P @ u, Nn @ u]).std() or 1.0
                        for (_, s, ip, ineg) in tst:
                            m = float((z[ro][ip, idx] - z[ro][ineg, idx]).astype(np.float64) @ u) / sd
                            fv = foil[meta[ip]["item_id"]]
                            res.append({"frame": f, "readout": ro, "layer": L, "value": v, "foil": fv,
                                        "distance": dist(v, fv), "margin": m, "correct": int(m > 0)})
                d = np.array([r["distance"] for r in res])
                mg = np.array([r["margin"] for r in res])
                rho = spearmanr(d, mg).correlation
                by_v = defaultdict(list)
                for n, r in enumerate(res):
                    by_v[r["value"]].append(n)
                null = []
                for _ in range(N_PERM):
                    dp = d.copy()
                    for idxs in by_v.values():
                        dp[idxs] = rng.permutation(d[idxs])
                    null.append(spearmanr(dp, mg).correlation)
                p = (1 + sum(x >= rho for x in null)) / (1 + N_PERM)
                per = {str(k): {"n": int((d == k).sum()), "accuracy": float(np.mean([r["correct"] for r in res if r["distance"] == k])),
                                "margin": float(np.mean(mg[d == k]))} for k in range(1, 6) if (d == k).any()}
                summary.append({"frame": f, "readout": ro, "layer": L, "n_pairs": len(res), "rho": float(rho),
                                "p_one_sided": p, "by_distance": per})
                pairs_out += res
    (RESULTS / "analysis").mkdir(parents=True, exist_ok=True)
    (RESULTS / "analysis" / "c21_circumplex.json").write_text(json.dumps({"summary": summary, "pairs": pairs_out}, indent=1))
    for s in summary:
        if s["layer"] in (17, 29):
            cells = "  ".join(f"d{k}: {v['accuracy']:.2f}/{v['margin']:+.2f} (n {v['n']})" for k, v in s["by_distance"].items())
            print(f"frame {s['frame']} {s['readout']:8s} L{s['layer']}  rho {s['rho']:+.3f} p {s['p_one_sided']:.4f}  n {s['n_pairs']} | {cells}")


if __name__ == "__main__":
    main()
