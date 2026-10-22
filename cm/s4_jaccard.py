"""S4 overlap, SAE part (D-012): Jaccard over selected SAE features with k swept, split-half ceilings, random floors,
and the replicas (262k medium and its seed-1 copy, D-009).

Units are the same as cm.s4_overlap: (concept, sub-contrast, frame) with at least MIN_TRAIN train items and MIN_TEST
test items per pole. Codes come from cm.stimx saecodes (SAE codes averaged over the first 32 reply tokens).
Per unit, dictionary and layer, the effect of each feature is (mean code pos - mean code neg) / pooled SD on train
items. For frame a and b units the effect is averaged within Gemma's verdict strata (P, N, X items separately,
weighted by the smaller pole count), so features that track the verdict are not selected (the SAE side of D-033c).
Selected set: the top k features by |effect|, k in KS.
  Jaccard   |S_u & S_v| / |S_u | S_v|
  ceiling   per unit, mean Jaccard between the sets from two halves of its train scenarios (R_SPLIT splits); a pair's
            ceiling is the geometric mean of the two units' values
  floor     95th percentile of Jaccard between real units' sets and other units' shuffled-label sets
  replica   Spearman between the off-diagonal Jaccard entries of two dictionaries (262k seed 0 vs seed 1; 65k vs 262k),
            and between Jaccard and the dense centred cosine of results/s4/overlap_verdict.json
Taxonomy (Jaccard has no sign): MFT within minus between (as in s4_overlap), Schwartz Jaccard vs circle distance
(prediction: overlap falls with distance).
Writes results/s4/jaccard.json (full matrices for k = 64, summaries for every k) and prints a summary.

    python -m cm.s4_jaccard [--workers 12]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

from .paths import RESULTS
from .s3 import ACTS
from .s4_overlap import MIN_TEST, MIN_TRAIN, R_RAND, R_SPLIT, SCHWARTZ
from .stimuli.schema import read_jsonl

CODES = ACTS / "saecodes"
DICTS = [("65k", "medium"), ("262k", "medium"), ("262k", "medium_seed_1")]
LAYERS = [9, 17, 22, 29]
KS = [16, 32, 64, 128, 256]
KMAX = max(KS)
OUT = RESULTS / "s4"


def load_codes(cid):
    f = np.load(CODES / f"{cid}.npz")
    meta = json.loads(str(f["meta"]))
    mats = {}
    for w, l0 in DICTS:
        F = 65536 if w == "65k" else 262144
        for L in LAYERS:
            k = f"{w}_{l0}_L{L}"
            mats[(w, l0, L)] = sparse.csr_matrix((f[f"{k}_values"], f[f"{k}_indices"], f[f"{k}_indptr"]),
                                                 shape=(len(meta), F))
    return mats, meta


def effects(M, pos, neg, strata=None):
    """Per-feature effect, pooled-SD standardized; within strata when given."""
    def mv(rows):
        X = M[rows]
        mu = np.asarray(X.mean(0)).ravel()
        sq = np.asarray(X.multiply(X).mean(0)).ravel()
        return mu, np.maximum(sq - mu ** 2, 0)
    mp, vp = mv(pos)
    mn, vn = mv(neg)
    sd = np.sqrt((vp + vn) / 2) + 1e-6
    if not strata:
        return (mp - mn) / sd
    num, wsum = 0.0, 0.0
    for p_, n_ in strata:
        w = min(len(p_), len(n_))
        a, _ = mv(p_)
        b, _ = mv(n_)
        num = num + w * (a - b)
        wsum += w
    return num / wsum / sd if wsum else (mp - mn) / sd


def topk(e):
    e = np.nan_to_num(e)
    idx = np.argpartition(-np.abs(e), KMAX)[:KMAX]
    return idx[np.argsort(-np.abs(e[idx]))]


def concept_sets(cid):
    mats, meta = load_codes(cid)
    verdict = {r["item_id"]: r["label"] for r in read_jsonl(ACTS / "verdicts" / f"{cid}.jsonl")}
    rng = np.random.default_rng(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
    out = []
    keys = sorted({(m["contrast"] or "all", m["frame"]) for m in meta if m["split"] in ("train", "test")})
    for sub, f in keys:
        sel = [i for i, m in enumerate(meta) if (m["contrast"] or "all") == sub and m["frame"] == f]
        tr = [i for i in sel if meta[i]["split"] == "train"]
        te = [i for i in sel if meta[i]["split"] == "test"]
        pos = [i for i in tr if meta[i]["polarity"] == "pos"]
        neg = [i for i in tr if meta[i]["polarity"] == "neg"]
        if min(len(pos), len(neg)) < MIN_TRAIN or min(sum(meta[i]["polarity"] == "pos" for i in te),
                                                      sum(meta[i]["polarity"] == "neg" for i in te)) < MIN_TEST:
            continue
        use_strata = f in ("a", "b") and verdict

        def strata_of(p, n):
            if not use_strata:
                return None
            st = []
            for lab in ("P", "N", "X"):
                a = [i for i in p if verdict.get(meta[i]["item_id"]) == lab]
                b = [i for i in n if verdict.get(meta[i]["item_id"]) == lab]
                if len(a) >= 3 and len(b) >= 3:
                    st.append((a, b))
            return st

        scen = sorted({meta[i]["scenario_id"] for i in tr})
        splits = []
        for _ in range(R_SPLIT):
            perm = rng.permutation(len(scen))
            hs = []
            for part in (perm[: len(scen) // 2], perm[len(scen) // 2:]):
                ss = {scen[k] for k in part}
                hs.append(([i for i in pos if meta[i]["scenario_id"] in ss], [i for i in neg if meta[i]["scenario_id"] in ss]))
            splits.append(hs)
        shuffles = []
        for _ in range(R_RAND):
            flip = {s: rng.random() < 0.5 for s in scen}
            lab = [(meta[i]["polarity"] == "pos") != flip[meta[i]["scenario_id"]] for i in tr]
            shuffles.append(([i for i, l in zip(tr, lab) if l], [i for i, l in zip(tr, lab) if not l]))
        sets = {}
        for key, M in mats.items():
            full = topk(effects(M, pos, neg, strata_of(pos, neg)))
            halves = [[topk(effects(M, p, n, strata_of(p, n))) for p, n in hs] for hs in splits]
            rands = [topk(effects(M, p, n, strata_of(p, n))) for p, n in shuffles]
            sets["|".join(map(str, key))] = {"full": full, "halves": halves, "rands": rands}
        out.append({"id": f"{cid}|{sub}|{f}", "sets": sets})
    return out


def jacc(A, B, k):
    """Jaccard between every set in A and every set in B (lists of index arrays, first k entries used)."""
    F = int(max(max(a[:k].max() for a in A), max(b[:k].max() for b in B))) + 1
    def ind(S):
        r = np.repeat(np.arange(len(S)), k)
        c = np.concatenate([s[:k] for s in S])
        return sparse.csr_matrix((np.ones(len(c), np.float32), (r, c)), shape=(len(S), F))
    inter = (ind(A) @ ind(B).T).toarray()
    return inter / (2 * k - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cids = sorted(p.stem for p in CODES.glob("*.npz"))
    units = []
    with ProcessPoolExecutor(args.workers) as ex:
        for us in ex.map(concept_sets, cids):
            units += us
    ids = [u["id"] for u in units]
    U = len(ids)
    pos_of = {k: i for i, k in enumerate(ids)}
    dense = json.loads((OUT / "overlap_verdict.json").read_text()) if (OUT / "overlap_verdict.json").exists() else None
    dense_ix = {k: i for i, k in enumerate(dense["units"])} if dense else {}
    mask = ~np.eye(U, dtype=bool)
    iu = np.triu_indices(U, 1)
    res = {"units": ids, "cells": {}}
    J_at = {}
    for w, l0 in DICTS:
        for L in LAYERS:
            key = f"{w}|{l0}|{L}"
            for k in KS:
                full = [u["sets"][key]["full"] for u in units]
                J = jacc(full, full, k)
                rel = np.array([np.mean([jacc([h[0]], [h[1]], k)[0, 0] for h in u["sets"][key]["halves"]]) for u in units])
                rand = [r for u in units for r in u["sets"][key]["rands"]]
                JR = jacc(full, rand, k).reshape(U, U, R_RAND)
                floor = float(np.percentile(JR[mask], 95))
                cell = {"median_reliability": float(np.median(rel)), "floor": floor,
                        "median_offdiag": float(np.median(J[iu])), "share_above_floor": float(np.mean(J[iu] > floor)),
                        "reliability": np.round(rel, 4).tolist()}
                if k == 64:
                    cell["J"] = np.round(J, 4).tolist()   # full matrix stored for the main k only
                if dense and str(L) in dense["layers"]:
                    C = np.array(dense["layers"][str(L)]["cos"])
                    common = [u for u in ids if u in dense_ix]
                    a = np.array([pos_of[u] for u in common])
                    b = np.array([dense_ix[u] for u in common])
                    jj = J[np.ix_(a, a)][np.triu_indices(len(a), 1)]
                    cc = C[np.ix_(b, b)][np.triu_indices(len(b), 1)]
                    cell["spearman_vs_dense_abs_cos"] = float(spearmanr(jj, np.abs(cc)).correlation)
                tax = {}
                for f in ("a", "b"):
                    ks_ = [f"B{i}|all|{f}" for i in range(1, 6)]
                    if all(x in pos_of for x in ks_):
                        ix = [pos_of[x] for x in ks_]
                        def stat(group):
                            wi = [J[ix[p], ix[q]] for p, q in itertools.combinations(range(5), 2) if (p in group) == (q in group)]
                            be = [J[ix[p], ix[q]] for p, q in itertools.combinations(range(5), 2) if (p in group) != (q in group)]
                            return float(np.mean(wi) - np.mean(be))
                        obs = stat({0, 1})
                        allp = [stat(set(g)) for g in itertools.combinations(range(5), 2)]
                        tax[f"mft_{f}"] = {"within_minus_between": obs, "p_exact": sum(x >= obs for x in allp) / 10}
                for f in ("b", "d"):
                    ks_ = [f"C21|{v}|{f}" for v in SCHWARTZ]
                    if all(x in pos_of for x in ks_):
                        ix = [pos_of[x] for x in ks_]
                        prs = list(itertools.combinations(range(10), 2))
                        js = [J[ix[p], ix[q]] for p, q in prs]
                        dd = [min(abs(p - q), 10 - abs(p - q)) for p, q in prs]
                        rho = spearmanr(dd, js).correlation
                        rng = np.random.default_rng(0)
                        null = []
                        for _ in range(5000):
                            o = rng.permutation(10)
                            null.append(spearmanr([min(abs(o[p] - o[q]), 10 - abs(o[p] - o[q])) for p, q in prs], js).correlation)
                        tax[f"schwartz_{f}"] = {"rho": float(rho), "p_one_sided_negative": (1 + sum(x <= rho for x in null)) / 5001}
                cell["taxonomy"] = tax
                res["cells"][f"{key}|k{k}"] = cell
                J_at[(w, l0, L, k)] = J
                print(f"[jaccard] {w} {l0} L{L} k{k}: median rel {cell['median_reliability']:.2f} floor {floor:.3f} "
                      f"median J {cell['median_offdiag']:.3f} above floor {cell['share_above_floor']:.2f} "
                      f"vs dense {cell.get('spearman_vs_dense_abs_cos', float('nan')):.2f}", flush=True)
    rep = {}
    for L in LAYERS:
        for k in KS:
            a = J_at[("262k", "medium", L, k)][iu]
            b = J_at[("262k", "medium_seed_1", L, k)][iu]
            c = J_at[("65k", "medium", L, k)][iu]
            rep[f"L{L}|k{k}"] = {"seed0_vs_seed1": float(spearmanr(a, b).correlation),
                                 "65k_vs_262k": float(spearmanr(c, a).correlation)}
    res["replica"] = rep
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jaccard.json").write_text(json.dumps(res))
    for k_, v in rep.items():
        print(f"[replica] {k_}: seed0 vs seed1 {v['seed0_vs_seed1']:.2f}, 65k vs 262k {v['65k_vs_262k']:.2f}")


if __name__ == "__main__":
    main()
