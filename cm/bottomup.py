"""S4 bottom-up pass (D-015, settings in D-038): which character-responsive SAE features co-activate in Gemma's own
replies to real chat prompts, and does that structure match the concept geometry and the taxonomies?

Stages:
  prompts   (CPU) first user turns from WildChat-1M (allenai, ODC-BY; shards 0, 7, 13), English, deduplicated,
            10-2000 characters, natural mix including toxic-flagged prompts; N sampled with a fixed seed.
            outputs/bottomup/prompts.jsonl (WildChat text stays on ada).
  replies   (GPU) Gemma-3-4B-IT greedy replies, up to 256 tokens (as D-028).  outputs/bottomup/replies.jsonl
  codes     (GPU) per reply, SAE codes at layers 9/17/22/29 averaged over the reply tokens, for 65k medium (main),
            262k medium and 262k medium seed 1 (replicas); stored sparse.  outputs/bottomup/codes.npz
  analyze   (CPU) per layer and dictionary:
            1. unit effects: every S4 unit's per-feature effect (cm.s4_jaccard.effects; verdict-stratified for a/b);
               character-responsive features = union of units' top 64 by |effect|, kept if active in at least
               ACTIVE_MIN of the replies; a random set of the same size, matched on activity frequency, as baseline;
            2. co-activation: Pearson correlation of log1p(mean code) across replies;
            3. clusters: Louvain on the k-nearest-neighbour graph (k 10, positive correlations), modularity vs the
               random set; adjusted mutual information between clusters and each feature's home concept group (the
               group of the unit with its largest |effect|), with a shuffled-home null;
            4. feature-level RSA: Spearman between co-activation and effect-profile similarity (correlation of two
               features' effects across all units), for character and random features;
            5. unit-level: N[u, v] = mean signed co-activation between u's and v's top features (sign of effect), the
               "in the wild" counterpart of the S4 cosine; Spearman with the S4 dense cosine; the S4 taxonomy tests on N;
            6. replicas: the same summaries with the 262k dictionaries, and the agreement of their N matrices.
            results/bottomup/summary.json, results/bottomup/N_65k.json (unit x unit N for the main dictionary)

    python -m cm.bottomup prompts|replies|codes|analyze
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .paths import HF_ROOT, MODELS, OUTPUTS, RESULTS, snapshot
from .stimuli.schema import read_jsonl, write_jsonl

OUT = OUTPUTS / "bottomup"
RES = RESULTS / "bottomup"
N_PROMPTS, REPLY_MAX, ACTIVE_MIN, TOPK, KNN = 20000, 256, 0.005, 64, 10
SHARDS = ["data/train-00000-of-00014.parquet", "data/train-00007-of-00014.parquet", "data/train-00013-of-00014.parquet"]
LAYERS = [9, 17, 22, 29]
DICTS = [("65k", "medium"), ("262k", "medium"), ("262k", "medium_seed_1")]


def cmd_prompts():
    import pyarrow.parquet as pq
    root = sorted(glob.glob(str(HF_ROOT / "datasets--allenai--WildChat-1M" / "snapshots" / "*")))[-1]
    pool, seen = [], set()
    for sh in SHARDS:
        t = pq.read_table(f"{root}/{sh}", columns=["conversation_hash", "conversation", "language", "toxic"]).to_pylist()
        for r in t:
            if r["language"] != "English" or not r["conversation"]:
                continue
            first = r["conversation"][0]
            if first.get("role") != "user":
                continue
            text = (first.get("content") or "").strip()
            key = hashlib.sha256(text.lower().encode()).hexdigest()
            if not (10 <= len(text) <= 2000) or key in seen:
                continue
            seen.add(key)
            pool.append({"id": r["conversation_hash"], "prompt": text, "toxic": bool(r["toxic"]), "shard": sh[-24:-8]})
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(pool), min(N_PROMPTS, len(pool)), replace=False))
    rows = [pool[i] for i in pick]
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "prompts.jsonl", rows)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "prompts_summary.json").write_text(json.dumps(
        {"pool": len(pool), "sampled": len(rows), "toxic_share": float(np.mean([r["toxic"] for r in rows])),
         "median_chars": float(np.median([len(r["prompt"]) for r in rows])),
         "by_shard": {s: sum(r["shard"] == s for r in rows) for s in sorted({r["shard"] for r in rows})}}, indent=1))
    print(f"[bottomup] pool {len(pool)}, sampled {len(rows)}", flush=True)


def cmd_replies():
    from vllm import LLM, SamplingParams
    rows = read_jsonl(OUT / "prompts.jsonl")
    llm = LLM(model=str(snapshot(MODELS["gemma-3-4b-it"])), dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=REPLY_MAX, stop_token_ids=[1, 106], skip_special_tokens=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
               for r in rows]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    write_jsonl(OUT / "replies.jsonl", [{"id": r["id"], "reply": o.outputs[0].text} for r, o in zip(rows, outs)])
    print(f"[bottomup] {len(rows)} replies", flush=True)


def cmd_codes():
    import torch
    from . import model as cm_model
    from .paths import sae_dir
    from .sae import JumpReLUSAE
    from .stimx import forward_hidden, tokenize
    rows = read_jsonl(OUT / "prompts.jsonl")
    rep = {r["id"]: r["reply"] for r in read_jsonl(OUT / "replies.jsonl")}
    m, tok = cm_model.load("gemma-3-4b-it")
    saes = {(w, l0, L): JumpReLUSAE.load(sae_dir("gemma-3-4b-it", L, w, l0), "cuda") for w, l0 in DICTS for L in LAYERS}
    seqs = []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        seqs.append(tokenize(tok, prompt, rep.get(r["id"], "")))
    buf = {k: ([0], [], []) for k in saes}
    ntok = []
    with torch.no_grad():
        for H, (ids, p) in zip(forward_hidden(m, [s[0] for s in seqs]), seqs):
            ntok.append(len(ids) - p)
            for (w, l0, L), sae in saes.items():
                indptr, idx, val = buf[(w, l0, L)]
                h = H[L + 1, p:]
                if h.shape[0] == 0:
                    indptr.append(indptr[-1])
                    continue
                code = sae.encode(h).mean(0)
                nz = code.nonzero().squeeze(1)
                idx.append(nz.int().cpu().numpy())
                val.append(code[nz].float().cpu().numpy())
                indptr.append(indptr[-1] + len(nz))
    arrays = {}
    for (w, l0, L), (indptr, idx, val) in buf.items():
        k = f"{w}_{l0}_L{L}"
        arrays[f"{k}_indptr"] = np.array(indptr, np.int64)
        arrays[f"{k}_indices"] = np.concatenate(idx)
        arrays[f"{k}_values"] = np.concatenate(val)
    np.savez(OUT / "codes.npz", ids=np.array([r["id"] for r in rows]), n_reply_tokens=np.array(ntok), **arrays)
    print(f"[bottomup] codes for {len(rows)} replies, median reply tokens {np.median(ntok):.0f}", flush=True)


# ---- analyze ----

def unit_effects(cid, units, key):
    """Full per-feature effect vectors for this concept's S4 units under one dictionary and layer."""
    from .s4_jaccard import effects, load_codes
    from .s3 import ACTS
    mats, meta = load_codes(cid)
    M = mats[key]
    verdict = {r["item_id"]: r["label"] for r in read_jsonl(ACTS / "verdicts" / f"{cid}.jsonl")}
    out = {}
    for uid in units:
        _, sub, f = uid.split("|")
        tr = [i for i, m in enumerate(meta) if (m["contrast"] or "all") == sub and m["frame"] == f and m["split"] == "train"]
        pos = [i for i in tr if meta[i]["polarity"] == "pos"]
        neg = [i for i in tr if meta[i]["polarity"] == "neg"]
        strata = None
        if f in ("a", "b") and verdict:
            strata = []
            for lab in ("P", "N", "X"):
                a = [i for i in pos if verdict.get(meta[i]["item_id"]) == lab]
                b = [i for i in neg if verdict.get(meta[i]["item_id"]) == lab]
                if len(a) >= 3 and len(b) >= 3:
                    strata.append((a, b))
        out[uid] = np.nan_to_num(effects(M, pos, neg, strata)).astype(np.float32)
    return out


def _effects_job(args):
    cid, units, key = args
    return unit_effects(cid, units, key)


def analyze_one(key, units, E, X, rng):
    from networkx import Graph
    from networkx.algorithms.community import louvain_communities, modularity
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_mutual_info_score
    U = len(units)
    Emat = np.stack([E[u] for u in units])                                  # [U, F]
    active = np.asarray((X > 0).mean(0)).ravel()
    tops = {u: np.argsort(-np.abs(E[u]))[:TOPK] for u in units}
    char = sorted({int(i) for t in tops.values() for i in t if active[i] >= ACTIVE_MIN})
    # activity-matched random features
    bins = np.digitize(np.log10(active + 1e-9), np.linspace(np.log10(ACTIVE_MIN), 0, 11))
    others = np.setdiff1d(np.where(active >= ACTIVE_MIN)[0], char)
    rand = []
    for b in np.unique(bins[char]):
        need = int((bins[char] == b).sum())
        cand = others[bins[others] == b]
        rand += list(rng.choice(cand, min(need, len(cand)), replace=False))
    rand = sorted(int(i) for i in rand)

    def coact(feats):
        Y = np.log1p(np.asarray(X[:, feats].todense()))
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
        return (Y.T @ Y) / len(Y)

    def clusters(Cm, feats):
        G = Graph()
        G.add_nodes_from(range(len(feats)))
        for i in range(len(feats)):
            row = Cm[i].copy()
            row[i] = -np.inf
            for j in np.argsort(-row)[:KNN]:
                if row[j] > 0:
                    G.add_edge(i, int(j), weight=float(row[j]))
        comms = louvain_communities(G, weight="weight", seed=0)
        return comms, modularity(G, comms, weight="weight")

    res = {"n_char": len(char), "n_rand": len(rand)}
    Cc, Cr = coact(char), coact(rand)
    comm_c, q_c = clusters(Cc, char)
    comm_r, q_r = clusters(Cr, rand)
    res.update({"modularity_char": float(q_c), "modularity_rand": float(q_r), "n_clusters_char": len(comm_c),
                "n_clusters_rand": len(comm_r)})
    # clusters vs home group
    lab = np.zeros(len(char), int)
    for k, c in enumerate(comm_c):
        for i in c:
            lab[i] = k
    home_unit = [units[int(np.argmax(np.abs(Emat[:, f])))] for f in char]
    home_group = [h[0] for h in home_unit]
    ami = adjusted_mutual_info_score(home_group, lab)
    null = [adjusted_mutual_info_score(list(rng.permutation(home_group)), lab) for _ in range(500)]
    res.update({"ami_cluster_vs_group": float(ami), "ami_null_95": float(np.percentile(null, 95)),
                "ami_p": float((1 + sum(x >= ami for x in null)) / 501)})
    home_frame = [h.split("|")[2] for h in home_unit]
    amif = adjusted_mutual_info_score(home_frame, lab)
    nullf = [adjusted_mutual_info_score(list(rng.permutation(home_frame)), lab) for _ in range(500)]
    res.update({"ami_cluster_vs_frame": float(amif), "ami_frame_null_95": float(np.percentile(nullf, 95))})
    # feature-level RSA
    iu = np.triu_indices(len(char), 1)
    prof = np.corrcoef(Emat[:, char].T)
    res["rsa_feature_char"] = float(spearmanr(prof[iu], Cc[iu]).correlation)
    iur = np.triu_indices(len(rand), 1)
    profr = np.corrcoef(Emat[:, rand].T)
    res["rsa_feature_rand"] = float(spearmanr(np.nan_to_num(profr[iur]), Cr[iur]).correlation)
    # unit-level natural co-activation
    pos_in = {f: k for k, f in enumerate(char)}
    N = np.full((U, U), np.nan)
    sets = {u: [(pos_in[int(i)], np.sign(E[u][i])) for i in tops[u] if int(i) in pos_in] for u in units}
    for a, b in itertools.combinations_with_replacement(range(U), 2):
        sa, sb_ = sets[units[a]], sets[units[b]]
        if not sa or not sb_:
            continue
        ia = np.array([x[0] for x in sa]); ga = np.array([x[1] for x in sa])
        ib = np.array([x[0] for x in sb_]); gb = np.array([x[1] for x in sb_])
        Mab = Cc[np.ix_(ia, ib)] * ga[:, None] * gb[None, :]
        if a == b:
            Mab = Mab[~np.eye(len(ia), dtype=bool)]
        N[a, b] = N[b, a] = float(np.mean(Mab))
    res["N"] = np.round(np.nan_to_num(N), 4).tolist()
    return res, char, comm_c, home_unit


def cmd_analyze():
    from scipy import sparse
    from scipy.stats import spearmanr
    f = np.load(OUT / "codes.npz")
    units = [u["id"] for u in json.loads((RESULTS / "s4" / "units.json").read_text())]
    by_c = {}
    for u in units:
        by_c.setdefault(u.split("|")[0], []).append(u)
    dense = json.loads((RESULTS / "s4" / "overlap_verdict.json").read_text())
    assert dense["units"] == units
    rng = np.random.default_rng(0)
    summary = {"prompts": json.loads((RES / "prompts_summary.json").read_text()), "cells": {}}
    Ns = {}
    for w, l0 in DICTS:
        F = 65536 if w == "65k" else 262144
        for L in LAYERS:
            key = (w, l0, L)
            k = f"{w}_{l0}_L{L}"
            X = sparse.csr_matrix((f[f"{k}_values"], f[f"{k}_indices"], f[f"{k}_indptr"]), shape=(len(f["ids"]), F)).tocsc()
            E = {}
            with ProcessPoolExecutor(12) as ex:
                for part in ex.map(_effects_job, [(c, us, key) for c, us in by_c.items()]):
                    E.update(part)
            res, char, comm, home = analyze_one(key, units, E, X, rng)
            N = np.array(res["N"])
            Ns[key] = N
            iu = np.triu_indices(len(units), 1)
            C = np.array(dense["layers"][str(L)]["cos"])
            ok = N[iu] != 0
            res["spearman_N_vs_dense_cos"] = float(spearmanr(N[iu][ok], C[iu][ok]).correlation)
            pos = {u: i for i, u in enumerate(units)}
            tax = {}
            for fr in ("a", "b"):
                ks = [f"B{i}|all|{fr}" for i in range(1, 6)]
                if all(x in pos for x in ks):
                    ix = [pos[x] for x in ks]
                    def stat(group):
                        wi = [N[ix[p], ix[q]] for p, q in itertools.combinations(range(5), 2) if (p in group) == (q in group)]
                        be = [N[ix[p], ix[q]] for p, q in itertools.combinations(range(5), 2) if (p in group) != (q in group)]
                        return float(np.mean(wi) - np.mean(be))
                    obs = stat({0, 1})
                    tax[f"mft_{fr}"] = {"within_minus_between": obs,
                                        "p_exact": sum(stat(set(g)) >= obs for g in itertools.combinations(range(5), 2)) / 10}
            from .s4_overlap import SCHWARTZ
            for fr in ("b", "d"):
                ks = [f"C21|{v}|{fr}" for v in SCHWARTZ]
                if all(x in pos for x in ks):
                    ix = [pos[x] for x in ks]
                    prs = list(itertools.combinations(range(10), 2))
                    vals = [N[ix[p], ix[q]] for p, q in prs]
                    dd = [min(abs(p - q), 10 - abs(p - q)) for p, q in prs]
                    rho = spearmanr(dd, vals).correlation
                    null = []
                    for _ in range(5000):
                        o = rng.permutation(10)
                        null.append(spearmanr([min(abs(o[p] - o[q]), 10 - abs(o[p] - o[q])) for p, q in prs], vals).correlation)
                    tax[f"schwartz_{fr}"] = {"rho": float(rho), "p_one_sided_negative": (1 + sum(x <= rho for x in null)) / 5001}
            for fr in ("c", "d"):
                h = f"C19|honesty_humility|{fr}"
                for t in ("machiavellianism", "narcissism", "psychopathy"):
                    x = f"C20|{t}|{fr}"
                    if h in pos and x in pos:
                        tax[f"hh_vs_{t}_{fr}"] = float(N[pos[h], pos[x]])
            res["taxonomy"] = tax
            if w == "65k":
                res["clusters"] = [{"size": len(c), "top_home_units": sorted(
                    {home[i] for i in c}, key=lambda u: -sum(home[i] == u for i in c))[:8]} for c in
                    sorted(comm, key=len, reverse=True)[:15]]
            summary["cells"][f"{w}|{l0}|{L}"] = res
            print(f"[bottomup] {w} {l0} L{L}: char {res['n_char']} | modularity char {res['modularity_char']:.3f} rand "
                  f"{res['modularity_rand']:.3f} | AMI group {res['ami_cluster_vs_group']:.3f} (null95 "
                  f"{res['ami_null_95']:.3f}) frame {res['ami_cluster_vs_frame']:.3f} (null95 {res['ami_frame_null_95']:.3f}) "
                  f"| RSA char {res['rsa_feature_char']:.3f} rand {res['rsa_feature_rand']:.3f} | N vs dense "
                  f"{res['spearman_N_vs_dense_cos']:.3f}", flush=True)
            print(f"            taxonomy {json.dumps(tax)}", flush=True)
    rep = {}
    iu = np.triu_indices(len(units), 1)
    for L in LAYERS:
        a, b, c = Ns[("262k", "medium", L)][iu], Ns[("262k", "medium_seed_1", L)][iu], Ns[("65k", "medium", L)][iu]
        rep[f"L{L}"] = {"seed0_vs_seed1": float(spearmanr(a, b).correlation), "65k_vs_262k": float(spearmanr(c, a).correlation)}
        print(f"[replica] L{L}: N seed0 vs seed1 {rep[f'L{L}']['seed0_vs_seed1']:.2f}, 65k vs 262k {rep[f'L{L}']['65k_vs_262k']:.2f}")
    summary["replica"] = rep
    RES.mkdir(parents=True, exist_ok=True)
    for cell in summary["cells"].values():
        cell.pop("N", None)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    (RES / "N_65k.json").write_text(json.dumps({"units": units, **{f"L{L}": np.round(Ns[("65k", "medium", L)], 4).tolist()
                                                                    for L in LAYERS}}))
    np.savez(OUT / "N_matrices.npz", units=np.array(units), **{f"{w}_{l0}_L{L}": Ns[(w, l0, L)] for (w, l0, L) in Ns})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prompts", "replies", "codes", "analyze"])
    args = ap.parse_args()
    {"prompts": cmd_prompts, "replies": cmd_replies, "codes": cmd_codes, "analyze": cmd_analyze}[args.stage]()


if __name__ == "__main__":
    main()
