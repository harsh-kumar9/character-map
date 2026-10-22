"""S4 overlap across concepts (D-012), dense part: centred cosine and RSA, with split-half ceilings and random floors,
plus the taxonomy shape tests fixed in PLAN.md ("Taxonomies as structural predictions").

Units are (concept, sub-contrast, frame) with at least MIN_TRAIN train items per pole and 5 test items per pole.
Readout: reply32 at the four SAE depths. Variants:
  verdict           (headline) Gemma's verdict direction projected out of frame a/b rows (D-033c)
  verdict+negation  (robustness) a negation direction also projected out of every row (D-036)
Per unit: the diff-of-means direction on train items; R_SPLIT random split-halves by scenario; R_RAND label shuffles
(pos/neg swapped per scenario at random, so the items stay and the contrast goes).
Measures, per depth:
  cosine  directions centred by the mean unit direction across all units, then cosine;
  RSA     every unit's direction projected onto a common probe set (up to PROBE per pole from each unit's test items),
          and the Pearson correlation between two units' projection patterns; in standardized space (probe activations
          z-scored per dimension, directions divided by the same SD), since Gemma's outlier residual dimensions would
          otherwise dominate every projection.
  ceiling per unit: split-half agreement (cosine of the halves, or correlation of their patterns), Spearman-Brown
          corrected; a pair's ceiling is the geometric mean of the two units' corrected reliabilities.
  floor   95th percentile of |measure| between real units and other units' shuffled-label directions (the
          shuffled-label directions centred on their own mean for the cosine floor).
Taxonomy tests (raw and centred cosine; half-sample ranges from the split halves):
  MFT        care+fairness vs loyalty+authority+sanctity (B1-B5, frames a and b): within minus between, exact p over
             the 10 ways to pick two of five;
  Schwartz   C21 values (frames b, d), directions centred on the C21 mean: Spearman between cosine and circle distance
             (prediction negative), p from shuffling positions on the circle;
  HEXACO     C19 honesty-humility vs each C20 dark-triad trait (frames c, d): prediction negative;
  Williams   C1 sincere vs C2 accurate (frames c, d): prediction low, well under the ceiling;
  Aristotle  vice-vs-virtue pairs C4 boastful/self-deprecating, C9 obsequious/quarrelsome, C13 blame-shifting/
             over-apologizing (pos is the virtue in both): a midpoint predicts raw cosine near -1.
Writes results/s4/units.json, results/s4/overlap_<variant>.json and prints a summary.

    python -m cm.s4_overlap [--variant verdict|verdict+negation] [--workers 12]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.stats import spearmanr

from .paths import RESULTS
from .s3 import ACTS, load
from .s3_nuisance import nuisance_dirs, row_labels

DEPTHS = {9: 10, 17: 18, 22: 23, 29: 30}
LIDX = list(DEPTHS.values())
R_SPLIT, R_RAND, PROBE, MIN_TRAIN = 10, 10, 20, 10
OUT = RESULTS / "s4"
SCHWARTZ = ["universalism", "benevolence", "tradition", "conformity", "security", "power", "achievement", "hedonism",
            "stimulation", "self_direction"]


def unit_vec(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def concept_data(cid, variant):
    z, meta, resp = load(cid)
    X = z["reply32"][:, LIDX].astype(np.float64)
    del z
    if "verdict" in variant:
        _, apply = row_labels(cid, meta, "verdict")
        dirs, _ = nuisance_dirs(cid, "verdict")
        if "reply32" in dirs and apply["reply32"].any():
            u = dirs["reply32"][LIDX]
            m = apply["reply32"]
            X[m] -= np.einsum("nl,ld->nld", np.einsum("nld,ld->nl", X[m], u), u)
    if "negation" in variant:
        dirs, _ = nuisance_dirs(cid, "negation")
        if "reply32" in dirs:
            u = dirs["reply32"][LIDX]
            X -= np.einsum("nl,ld->nld", np.einsum("nld,ld->nl", X, u), u)
    return X, meta


def concept_units(cid, variant):
    """Directions, split halves, shuffled-label directions and probe rows for every unit of one concept."""
    X, meta = concept_data(cid, variant)
    rng = np.random.default_rng(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
    out = []
    keys = sorted({(m["contrast"] or "all", m["frame"]) for m in meta if m["split"] in ("train", "test")})
    for sub, f in keys:
        sel = [i for i, m in enumerate(meta) if (m["contrast"] or "all") == sub and m["frame"] == f]
        tr = [i for i in sel if meta[i]["split"] == "train"]
        te = [i for i in sel if meta[i]["split"] == "test"]
        pos = [i for i in tr if meta[i]["polarity"] == "pos"]
        neg = [i for i in tr if meta[i]["polarity"] == "neg"]
        tpos = [i for i in te if meta[i]["polarity"] == "pos"]
        tneg = [i for i in te if meta[i]["polarity"] == "neg"]
        if min(len(pos), len(neg)) < MIN_TRAIN or min(len(tpos), len(tneg)) < 5:
            continue
        d = X[pos].mean(0) - X[neg].mean(0)
        scen = sorted({meta[i]["scenario_id"] for i in tr})
        halves = []
        for _ in range(R_SPLIT):
            perm = rng.permutation(len(scen))
            hs = []
            for part in (perm[: len(scen) // 2], perm[len(scen) // 2:]):
                ss = {scen[k] for k in part}
                p = [i for i in pos if meta[i]["scenario_id"] in ss]
                n = [i for i in neg if meta[i]["scenario_id"] in ss]
                hs.append(X[p].mean(0) - X[n].mean(0) if p and n else np.zeros_like(d))
            halves.append(hs)
        rands = []
        for _ in range(R_RAND):
            flip = {s: rng.random() < 0.5 for s in scen}
            lab = [(meta[i]["polarity"] == "pos") != flip[meta[i]["scenario_id"]] for i in tr]
            p = [i for i, l in zip(tr, lab) if l]
            n = [i for i, l in zip(tr, lab) if not l]
            rands.append(X[p].mean(0) - X[n].mean(0))
        probe = list(rng.choice(tpos, min(PROBE, len(tpos)), replace=False)) + \
            list(rng.choice(tneg, min(PROBE, len(tneg)), replace=False))
        out.append({"id": f"{cid}|{sub}|{f}", "concept": cid, "sub": sub, "frame": f, "n_train": [len(pos), len(neg)],
                    "d": d.astype(np.float32), "halves": np.array(halves, np.float32),
                    "rands": np.array(rands, np.float32), "probe": X[probe].astype(np.float32)})
    return out


def sb(r):
    r = np.clip(r, -0.99, 0.999)
    return 2 * r / (1 + r)


def corr_rows(A, B):
    """Pearson correlation between rows of A [m, n] and rows of B [k, n]."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A = unit_vec(A)
    B = unit_vec(B)
    return A @ B.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="verdict", choices=["verdict", "verdict+negation"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cids = sorted(p.stem for p in ACTS.glob("*.npz"))
    units = []
    with ProcessPoolExecutor(args.workers) as ex:
        for us in ex.map(partial(concept_units, variant=args.variant), cids):
            units += us
    U = len(units)
    ids = [u["id"] for u in units]
    pos_of = {k: i for i, k in enumerate(ids)}
    D = np.stack([u["d"] for u in units]).astype(np.float64)            # [U, 4, d]
    H = np.stack([u["halves"] for u in units]).astype(np.float64)       # [U, R, 2, 4, d]
    Rn = np.stack([u["rands"] for u in units]).astype(np.float64)       # [U, S, 4, d]
    P = np.concatenate([u["probe"] for u in units]).astype(np.float64)  # [Np, 4, d]
    print(f"[s4] {U} units, probe {len(P)} items", flush=True)
    res = {"variant": args.variant, "units": ids, "layers": {}}
    unit_info = [{"id": u["id"], "concept": u["concept"], "sub": u["sub"], "frame": u["frame"],
                  "n_train": u["n_train"]} for u in units]
    for L, li in DEPTHS.items():
        l = LIDX.index(li)
        Dn = unit_vec(D[:, l])
        mean = Dn.mean(0)
        C = unit_vec(Dn - mean)
        Hc = unit_vec(unit_vec(H[:, :, :, l]) - mean)
        # the shuffled-label universe is centred on its own mean, as the real units are on theirs
        Rn_l = unit_vec(Rn[:, :, l])
        Rc = unit_vec(Rn_l - Rn_l.reshape(-1, Rn_l.shape[-1]).mean(0))
        cos = C @ C.T
        rel_cos = sb(np.einsum("urd,urd->ur", Hc[:, :, 0], Hc[:, :, 1]).mean(1))
        rc = np.abs(np.einsum("ud,vsd->uvs", C, Rc))
        mask = ~np.eye(U, dtype=bool)
        floor_cos = float(np.percentile(rc[mask], 95))
        # RSA in standardized space: Gemma's outlier residual dimensions would otherwise dominate every projection
        mu, sd = P[:, l].mean(0), P[:, l].std(0) + 1e-6
        PL = (P[:, l] - mu) / sd
        std_dir = lambda x: unit_vec(x / sd)
        pat = std_dir(D[:, l]) @ PL.T                                     # [U, Np]
        rsa = corr_rows(pat, pat)
        ph0 = np.einsum("urd,nd->urn", std_dir(H[:, :, 0, l]), PL)
        ph1 = np.einsum("urd,nd->urn", std_dir(H[:, :, 1, l]), PL)
        rel_rsa = sb(np.array([np.mean([corr_rows(ph0[u, r][None], ph1[u, r][None])[0, 0] for r in range(R_SPLIT)])
                               for u in range(U)]))
        prand = np.einsum("usd,nd->usn", std_dir(Rn[:, :, l]), PL).reshape(U * R_RAND, -1)
        rr = np.abs(corr_rows(pat, prand)).reshape(U, U, R_RAND)
        floor_rsa = float(np.percentile(rr[mask], 95))
        raw = Dn @ Dn.T
        # half-sample ranges for any pair: cosine between unit u's half k and unit v's half k (raw, uncentred)
        Hn = unit_vec(H[:, :, :, l])

        def half_range(a, b, centred=False):
            A = Hc if centred else Hn
            vals = np.einsum("rkd,rkd->rk", A[a], A[b]).ravel()
            return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

        tax = {}
        # MFT
        for f in ("a", "b"):
            ks = [f"B{i}|all|{f}" for i in range(1, 6)]
            if all(k in pos_of for k in ks):
                ix = [pos_of[k] for k in ks]
                for name, M in (("cos", cos), ("rsa", rsa)):
                    def stat(group):
                        within = [M[ix[a], ix[b]] for a, b in itertools.combinations(range(5), 2)
                                  if (a in group) == (b in group)]
                        between = [M[ix[a], ix[b]] for a, b in itertools.combinations(range(5), 2)
                                   if (a in group) != (b in group)]
                        return float(np.mean(within) - np.mean(between))
                    obs = stat({0, 1})
                    allp = [stat(set(g)) for g in itertools.combinations(range(5), 2)]
                    tax[f"mft_{name}_{f}"] = {"within_minus_between": obs,
                                              "p_exact": sum(x >= obs for x in allp) / len(allp),
                                              "pairs": {f"{ks[a].split('|')[0]}-{ks[b].split('|')[0]}": float(M[ix[a], ix[b]])
                                                        for a, b in itertools.combinations(range(5), 2)}}
        # Schwartz
        rng = np.random.default_rng(0)
        for f in ("b", "d"):
            ks = [f"C21|{v}|{f}" for v in SCHWARTZ]
            if all(k in pos_of for k in ks):
                ix = [pos_of[k] for k in ks]
                V = unit_vec(Dn[ix] - Dn[ix].mean(0))
                M = V @ V.T
                prs = list(itertools.combinations(range(10), 2))
                dist = lambda order, a, b: min(abs(order[a] - order[b]), 10 - abs(order[a] - order[b]))
                base = list(range(10))
                cs = [M[a, b] for a, b in prs]
                rho = spearmanr([dist(base, a, b) for a, b in prs], cs).correlation
                null = []
                for _ in range(5000):
                    o = list(rng.permutation(10))
                    null.append(spearmanr([dist(o, a, b) for a, b in prs], cs).correlation)
                by_d = {k: float(np.mean([M[a, b] for a, b in prs if dist(base, a, b) == k])) for k in range(1, 6)}
                tax[f"schwartz_{f}"] = {"rho": float(rho), "p_one_sided_negative": (1 + sum(x <= rho for x in null)) / 5001,
                                        "mean_cos_by_distance": by_d}
        # HEXACO honesty-humility vs dark triad
        for f in ("c", "d"):
            h = f"C19|honesty_humility|{f}"
            for t in ("machiavellianism", "narcissism", "psychopathy"):
                k = f"C20|{t}|{f}"
                if h in pos_of and k in pos_of:
                    a, b = pos_of[h], pos_of[k]
                    tax[f"hh_vs_{t}_{f}"] = {"raw": float(raw[a, b]), "centred": float(cos[a, b]),
                                             "raw_half_range": half_range(a, b)}
        # Williams
        for f in ("c", "d"):
            a_, b_ = f"C1|all|{f}", f"C2|all|{f}"
            if a_ in pos_of and b_ in pos_of:
                a, b = pos_of[a_], pos_of[b_]
                tax[f"sincere_vs_accurate_{f}"] = {"raw": float(raw[a, b]), "centred": float(cos[a, b]),
                                                   "raw_half_range": half_range(a, b),
                                                   "ceiling": float(np.sqrt(max(rel_cos[a], 0) * max(rel_cos[b], 0)))}
        # Aristotle
        for c_, s1, s2 in (("C4", "boastful", "self-deprecating"), ("C9", "obsequious", "quarrelsome"),
                           ("C13", "blame_shifting", "over_apologizing")):
            for f in "abcd":
                a_, b_ = f"{c_}|{s1}|{f}", f"{c_}|{s2}|{f}"
                if a_ in pos_of and b_ in pos_of:
                    a, b = pos_of[a_], pos_of[b_]
                    tax[f"mean_{c_}_{f}"] = {"raw": float(raw[a, b]), "raw_half_range": half_range(a, b)}
        res["layers"][str(L)] = {"cos": np.round(cos, 4).tolist(), "rsa": np.round(rsa, 4).tolist(),
                                 "rel_cos": np.round(rel_cos, 4).tolist(), "rel_rsa": np.round(rel_rsa, 4).tolist(),
                                 "floor_cos": floor_cos, "floor_rsa": floor_rsa, "taxonomy": tax}
        iu = np.triu_indices(U, 1)
        print(f"[s4] layer {L}: floors cos {floor_cos:.3f} rsa {floor_rsa:.3f}; median reliability cos "
              f"{np.median(rel_cos):.2f} rsa {np.median(rel_rsa):.2f}; median |off-diagonal| cos "
              f"{np.median(np.abs(cos[iu])):.3f} rsa {np.median(np.abs(rsa[iu])):.3f}; share above floor cos "
              f"{np.mean(np.abs(cos[iu]) > floor_cos):.2f} rsa {np.mean(np.abs(rsa[iu]) > floor_rsa):.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "units.json").write_text(json.dumps(unit_info, indent=1))
    tag = args.variant.replace("+", "_")
    (OUT / f"overlap_{tag}.json").write_text(json.dumps(res))
    for L in DEPTHS:
        print(f"\nlayer {L} taxonomy tests ({args.variant}):")
        for k, v in res["layers"][str(L)]["taxonomy"].items():
            print(f"  {k}: " + ", ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}"
                                         for kk, vv in v.items() if kk != "pairs"))


if __name__ == "__main__":
    main()
