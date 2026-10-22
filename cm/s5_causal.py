"""S5a: representation-level causal matrix (D-013, settings in D-039).

Steer one concept direction at decoder layer 17 (added to the residual stream at every position) and read every S4
unit's projection at layers 22 and 29 on that unit's own test items, with the item text held fixed.
Sources: per concept and sub-contrast, the doing-side direction (frame c, else d) and, where the concept also has a
knowing frame, the knowing-side direction (b, else a). Directions are S3 reply32 diff-of-means at hidden index 18;
frames a and b use the verdict-controlled directions (D-033c).
Interventions per source: dense direction at +dose and -dose; top-16 SAE features at +dose (the decoder-weighted sum
of the source's top 65k-medium layer-17 features by |effect|, with their mean differences as weights; added, not
reconstructed, so the SAE error term is kept). Controls: 8 norm-matched random directions at +-dose; K2 (topic) is a
source like any other and serves as the unrelated-concept control.
Effect of source s on target t at layer L: mean shift of t's items' projection onto t's direction, minus the skip-path
prediction (dose times the cosine between the injected vector and t's direction: what the added vector alone carries
forward), divided by t's baseline pos-neg gap. z against the 16 random-direction effects on the same target.
Dose: one global multiple c of the median residual-stream norm at layer 17, the largest on a grid at which
gemma-4-31B-it rates steered free generations (96 tokens) at mean coherence >= 75 for every calibration source and
both signs, on the WildChat prompts (of 64) whose unsteered reply it rates at least 90.

    python -m cm.s5_causal setup              CPU   sources, vectors, targets, probe items
    python -m cm.s5_causal calib-gen          GPU   steered generations for the dose grid
    python -m cm.s5_causal calib-judge        GPU   coherence (gemma-4-31B-it) and the chosen dose
    python -m cm.s5_causal run --shard K --nshards N   GPU
    python -m cm.s5_causal matrix             CPU   results/s5/causal_matrix.json and summary
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from .paths import OUTPUTS, RESULTS

OUT = OUTPUTS / "s5"
RES = RESULTS / "s5"
L_IN, L_OUT = 17, (22, 29)
H_IN, H_OUT = L_IN + 1, tuple(L + 1 for L in L_OUT)
GRID = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16]
N_RAND, PROBE, TOPF = 8, 6, 16
CALIB = ["A1|all|c", "B1|all|b", "C1|all|c", "C6|all|c", "C9|quarrelsome|c", "C15|refuse|c", "C19|honesty_humility|c",
         "C20|psychopathy|c", "C21|power|d", "D1|all|c", "E1|all|c", "F3|all|c"]


def units():
    return [u["id"] for u in json.loads((RESULTS / "s4" / "units.json").read_text())]


def direction(uid, hidx):
    cid, sub, f = uid.split("|")
    base = RESULTS / ("s3_ctrl_verdict" if f in "ab" else "s3") / "directions" / f"{cid}.npz"
    v = np.load(base)[f"{sub}|{f}|reply32"][hidx].astype(np.float64)
    return v / (np.linalg.norm(v) or 1.0)


_WDEC = None


def w_dec():
    global _WDEC
    if _WDEC is None:
        from safetensors.numpy import load_file
        from .paths import sae_dir
        _WDEC = load_file(str(sae_dir("gemma-3-4b-it", 17) / "params.safetensors"))["w_dec"]
    return _WDEC


def sae_vector(uid):
    """Decoder-weighted sum of the unit's top-16 layer-17 features (65k medium), weights = mean code differences."""
    from .s3 import ACTS
    from .s4_jaccard import effects, load_codes
    from .stimuli.schema import read_jsonl
    cid, sub, f = uid.split("|")
    mats, meta = load_codes(cid)
    M = mats[("65k", "medium", 17)]
    tr = [i for i, m in enumerate(meta) if (m["contrast"] or "all") == sub and m["frame"] == f and m["split"] == "train"]
    pos = [i for i in tr if meta[i]["polarity"] == "pos"]
    neg = [i for i in tr if meta[i]["polarity"] == "neg"]
    strata = None
    if f in "ab":
        verdict = {r["item_id"]: r["label"] for r in read_jsonl(ACTS / "verdicts" / f"{cid}.jsonl")}
        strata = []
        for lab in "PNX":
            a = [i for i in pos if verdict.get(meta[i]["item_id"]) == lab]
            b = [i for i in neg if verdict.get(meta[i]["item_id"]) == lab]
            if len(a) >= 3 and len(b) >= 3:
                strata.append((a, b))
        strata = strata or None
    e = np.nan_to_num(effects(M, pos, neg, strata))
    top = np.argsort(-np.abs(e))[:TOPF]
    diff = np.asarray(M[pos][:, top].mean(0) - M[neg][:, top].mean(0)).ravel()
    v = (diff[:, None] * w_dec()[top]).sum(0).astype(np.float64)
    return v / (np.linalg.norm(v) or 1.0), top.tolist()


def cmd_setup():
    from .s3 import ACTS
    U = units()
    by_c = {}
    for u in U:
        cid, sub, f = u.split("|")
        by_c.setdefault((cid, sub), []).append(f)
    sources = []
    for (cid, sub), frames in sorted(by_c.items()):
        do = next((f for f in "cd" if f in frames), None)
        kn = next((f for f in "ba" if f in frames), None)
        if do:
            sources.append(f"{cid}|{sub}|{do}")
        if kn:
            sources.append(f"{cid}|{sub}|{kn}")
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((N_RAND, 2560))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    dense = np.stack([direction(s, H_IN) for s in sources])
    sae, tops = [], {}
    for s in sources:
        v, t = sae_vector(s)
        sae.append(v)
        tops[s] = t
    tgt = {h: np.stack([direction(u, h) for u in U]) for h in H_OUT}
    probe = []
    for u in U:
        cid, sub, f = u.split("|")
        meta = json.loads(str(np.load(ACTS / f"{cid}.npz")["meta"]))
        te = [m["item_id"] for m in meta if (m["contrast"] or "all") == sub and m["frame"] == f and m["split"] == "test"]
        pol = {m["item_id"]: m["polarity"] for m in meta}
        r = np.random.default_rng(int(hashlib.sha256(u.encode()).hexdigest()[:8], 16))
        for p in ("pos", "neg"):
            c = [i for i in te if pol[i] == p]
            for i in r.choice(c, min(PROBE, len(c)), replace=False):
                probe.append({"unit": u, "item_id": str(i), "polarity": p})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "setup.npz", dense=dense, sae=np.stack(sae), rand=rand, tgt23=tgt[H_OUT[0]], tgt30=tgt[H_OUT[1]])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "setup.json").write_text(json.dumps({"sources": sources, "targets": U, "probe": probe, "sae_top_features": tops,
                                                "cos_dense_sae": [float(a @ b) for a, b in zip(dense, sae)]}, indent=1))
    print(f"[s5] {len(sources)} sources, {len(U)} targets, {len(probe)} probe items; median cos(dense, sae) "
          f"{np.median([float(a @ b) for a, b in zip(dense, sae)]):.2f}", flush=True)


def load_setup():
    S = json.loads((RES / "setup.json").read_text())
    A = np.load(OUT / "setup.npz")
    return S, A


def calib_prompts():
    from .stimuli.schema import read_jsonl
    rows = [r for r in read_jsonl(OUTPUTS / "bottomup" / "prompts.jsonl") if len(r["prompt"]) <= 300]
    return [r["prompt"] for r in rows[:64]]


def cmd_calib_gen():
    import torch
    from . import model as cm_model
    from .stimuli.schema import write_jsonl
    S, A = load_setup()
    m, tok = cm_model.load("gemma-3-4b-it")
    tok.padding_side = "left"
    prompts = calib_prompts()
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad(), cm_model.capture_resid(m, [L_IN]) as st:
        m(**enc)
    h = st[L_IN].float()
    mask = enc["attention_mask"].bool()
    norm = float(h.norm(dim=-1)[mask].median())
    conds = [("baseline", None, 0.0)]
    for s in [c for c in CALIB if c in S["sources"]]:
        for c in GRID:
            for sg in (1, -1):
                conds.append((s, A["dense"][S["sources"].index(s)], sg * c))
    for k in range(2):
        for c in GRID:
            for sg in (1, -1):
                conds.append((f"random{k}", A["rand"][k], sg * c))
    rows = []
    for name, v, c in conds:
        ctx = cm_model.steer(m, L_IN, torch.tensor(v * c * norm, device="cuda", dtype=torch.float32)) if v is not None else None
        with torch.no_grad():
            if ctx:
                with ctx:
                    out = m.generate(**enc, max_new_tokens=96, do_sample=False)
            else:
                out = m.generate(**enc, max_new_tokens=96, do_sample=False)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"cond": name, "c": c, "prompt_i": i, "prompt": p, "text": t} for i, (p, t) in enumerate(zip(prompts, texts))]
    write_jsonl(OUT / "calib_gen.jsonl", rows)
    (RES / "residual_norm_L17.json").write_text(json.dumps({"median_token_norm": norm}))
    print(f"[s5] {len(conds)} calibration conditions, median residual norm at layer {L_IN}: {norm:.1f}", flush=True)


def cmd_calib_judge():
    from .pv.judge import COHERENCE, DigitTreeJudge, fill
    from .stimuli.schema import read_jsonl, write_jsonl
    rows = read_jsonl(OUT / "calib_gen.jsonl")
    judge = DigitTreeJudge("google/gemma-4-31B-it", 128)
    scores = judge.score([fill(COHERENCE, r["prompt"], r["text"]) for r in rows])
    for r, (sc, _, _) in zip(rows, scores):
        r["coherence"] = sc
    write_jsonl(OUT / "calib_scores.jsonl", rows)
    # keep prompts whose unsteered reply the judge rates at least 90: truncated or borderline prompts make the judge
    # score fine replies 0 or decline (baseline mean 51 on the first attempt), which would hide the effect of steering
    good = {r["prompt_i"] for r in rows if r["cond"] == "baseline" and (r["coherence"] or 0) >= 90}
    table = {}
    for r in rows:
        if r["coherence"] is not None and r["prompt_i"] in good:
            table.setdefault((r["cond"], r["c"]), []).append(r["coherence"])
    means = {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in table.items()}
    chosen = None
    for c in GRID:   # the largest dose at which this and every smaller dose keep coherence
        cells = [v for k, v in means.items() if k.split("|")[-1] in (str(c), str(-c)) and not k.startswith("baseline")]
        if not cells or min(cells) < 75:
            break
        chosen = c
    out = {"means": means, "chosen_c": chosen, "baseline": means.get("baseline|0.0"), "n_prompts_kept": len(good),
           "min_by_c": {str(c): min(v for k, v in means.items() if k.split("|")[-1] in (str(c), str(-c)) and not k.startswith("baseline"))
                        for c in GRID}}
    (RES / "calibration.json").write_text(json.dumps(out, indent=1))
    print(f"[s5] {len(good)} prompts kept; baseline coherence {out['baseline']:.1f}; min mean coherence by c "
          f"{out['min_by_c']}; chosen c {chosen}", flush=True)


def conditions(S):
    conds = [("baseline", "none", 0, 0)]
    for k, s in enumerate(S["sources"]):
        conds += [(s, "dense", k, 1), (s, "dense", k, -1), (s, "sae", k, 1)]
    for k in range(N_RAND):
        conds += [(f"random{k}", "rand", k, 1), (f"random{k}", "rand", k, -1)]
    return conds


def cmd_run(shard, nshards):
    import torch
    from . import model as cm_model
    from .stimx import forward_hidden, load_items, load_replies, prompt_and_reply, tokenize
    S, A = load_setup()
    cal = json.loads((RES / "calibration.json").read_text())
    c = cal["chosen_c"]
    norm = json.loads((RES / "residual_norm_L17.json").read_text())["median_token_norm"]
    m, tok = cm_model.load("gemma-3-4b-it")
    items, cache = [], {}
    for p in S["probe"]:
        cid = p["unit"].split("|")[0]
        if cid not in cache:
            cache[cid] = ({it["item_id"]: it for it in load_items(cid)}, load_replies(cid))
        its, rep = cache[cid]
        items.append(tokenize(tok, *prompt_and_reply(tok, its[p["item_id"]], rep)))
    T = [torch.tensor(A["tgt23"], device="cuda", dtype=torch.float32), torch.tensor(A["tgt30"], device="cuda", dtype=torch.float32)]
    conds = conditions(S)
    mine = [k for k in range(len(conds)) if k % nshards == shard]
    proj = np.zeros((len(mine), len(items), 2, T[0].shape[0]), np.float32)
    for j, k in enumerate(mine):
        name, kind, idx, sg = conds[k]
        vec = None if kind == "none" else A[kind][idx] * sg * c * norm
        ctx = cm_model.steer(m, L_IN, torch.tensor(vec, device="cuda", dtype=torch.float32)) if vec is not None else None
        if ctx:
            ctx.__enter__()
        try:
            with torch.no_grad():
                for n, (H, (ids, p)) in enumerate(zip(forward_hidden(m, [x[0] for x in items], batch=16), items)):
                    for li, hidx in enumerate(H_OUT):
                        r = H[hidx, p:p + 32].float().mean(0)
                        proj[j, n, li] = (T[li] @ r).cpu().numpy()
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
        print(f"[s5] shard {shard}: condition {j + 1}/{len(mine)} {name} {kind} {sg:+d}", flush=True)
    np.savez(OUT / f"run_shard{shard}.npz", cond_index=np.array(mine), proj=proj)


def cmd_matrix(nshards):
    from scipy.stats import spearmanr
    S, A = load_setup()
    cal = json.loads((RES / "calibration.json").read_text())
    c = cal["chosen_c"]
    norm = json.loads((RES / "residual_norm_L17.json").read_text())["median_token_norm"]
    conds = conditions(S)
    P = {}
    for k in range(nshards):
        f = np.load(OUT / f"run_shard{k}.npz")
        for j, ci in enumerate(f["cond_index"]):
            P[int(ci)] = f["proj"][j]
    U = S["targets"]
    rows_of = {u: [n for n, p in enumerate(S["probe"]) if p["unit"] == u] for u in U}
    pos_of = {u: [n for n, p in enumerate(S["probe"]) if p["unit"] == u and p["polarity"] == "pos"] for u in U}
    neg_of = {u: [n for n, p in enumerate(S["probe"]) if p["unit"] == u and p["polarity"] == "neg"] for u in U}
    base = P[0]
    dose = c * norm
    tg = [A["tgt23"], A["tgt30"]]
    gap = np.array([[base[pos_of[u], li, t].mean() - base[neg_of[u], li, t].mean() for t, u in enumerate(U)] for li in range(2)])
    gap[gap <= 0] = np.nan   # a target whose probe items do not separate at baseline gets no effect estimate
    print(f"[s5] targets without a positive baseline gap: layer {L_OUT[0]} {int(np.isnan(gap[0]).sum())}, "
          f"layer {L_OUT[1]} {int(np.isnan(gap[1]).sum())}", flush=True)

    def effect(k, vec):
        d = P[k] - base
        out = np.zeros((2, len(U)))
        for li in range(2):
            skip = dose * (tg[li] @ vec)
            for t, u in enumerate(U):
                out[li, t] = (d[rows_of[u], li, t].mean() - skip[t]) / gap[li, t]
        return out

    eff = {}
    for k, (name, kind, idx, sg) in enumerate(conds):
        if kind != "none":
            eff[k] = effect(k, A[kind][idx] * sg)
    rand = np.stack([eff[k] for k, cd in enumerate(conds) if cd[1] == "rand"])     # [16, 2, U]
    mu, sd = np.nanmean(rand, 0), np.nanstd(rand, 0) + 1e-9
    res = {"dose_c": c, "dose_norm": dose, "sources": S["sources"], "targets": U, "layers": list(L_OUT),
           "target_gap": gap.round(4).tolist(), "random_sd": sd.round(4).tolist(), "kinds": {}}
    for kind, sg in (("dense", 1), ("dense", -1), ("sae", 1)):
        E = np.stack([eff[k] for k, cd in enumerate(conds) if cd[1] == kind and cd[3] == sg])  # [S, 2, U]
        Z = (E - mu) / sd
        res["kinds"][f"{kind}{'+' if sg > 0 else '-'}"] = {"E": E.round(4).tolist(), "Z": Z.round(2).tolist()}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "causal_matrix.json").write_text(json.dumps(res))
    Ed = np.array(res["kinds"]["dense+"]["E"]); Zd = np.array(res["kinds"]["dense+"]["Z"])
    Em = np.array(res["kinds"]["dense-"]["E"]); Es = np.array(res["kinds"]["sae+"]["E"])
    src = S["sources"]
    tix = {u: t for t, u in enumerate(U)}
    for li, L in enumerate(L_OUT):
        selfE = [Ed[s, li, tix[u]] for s, u in enumerate(src) if u in tix]
        selfZ = [Zd[s, li, tix[u]] for s, u in enumerate(src) if u in tix]
        off = np.array([[Zd[s, li, t] for t, u in enumerate(U) if u.split("|")[0] != src[s].split("|")[0]] for s in range(len(src))])
        ok = ~np.isnan(Ed[:, li].ravel())
        print(f"[s5] layer {L}: self effect median {np.nanmedian(selfE):.2f} gap units (z {np.nanmedian(selfZ):.1f}); "
              f"off-concept cells |z|>3: {np.nanmean(np.abs(off) > 3):.3f}; sign symmetry r(E+, -E-) "
              f"{np.corrcoef(Ed[:, li].ravel()[ok], -Em[:, li].ravel()[ok])[0, 1]:.2f}; dense vs sae Spearman "
              f"{spearmanr(Ed[:, li].ravel()[ok], Es[:, li].ravel()[ok]).correlation:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["setup", "calib-gen", "calib-judge", "run", "matrix"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "setup":
        cmd_setup()
    elif args.stage == "calib-gen":
        cmd_calib_gen()
    elif args.stage == "calib-judge":
        cmd_calib_judge()
    elif args.stage == "run":
        cmd_run(args.shard, args.nshards)
    else:
        cmd_matrix(args.nshards)


if __name__ == "__main__":
    main()
