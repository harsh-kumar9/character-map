"""D-060 stage 1: why an injected honesty direction does not propagate in gemma-4-31b-it. CPU, read-only on the saved
activations, the part-1 levers and the honesty-block S5a grid; Qwen3.5-27B (and Gemma-3-4B where its inputs exist)
as the comparison columns.

anatomy   per cell at the steering depth: the injected raw diff of means (S5a setup `dense`), the standardized
          direction (pooled SD over the 22 honesty cells, as the levers), its primal raw form sd * u and its dual
          u / sd (the levers' `|resid`), the per-cell probe; norm fraction on the top-k outlier dimensions (k = 8, 32,
          128; outliers ranked by the per-dimension RMS of boundary tokens at that depth), cosines between them;
          residual norm profile at every hidden index (median token norm, top-k share of the mean square); each grid
          dose in raw units, class-gap units and standardized units; the 28 x 28 cosine matrices; the per-layer AUROC
          and direction-persistence profile of every cell (raw rows, no verdict projection); the native maintenance
          of the steering-depth direction at the later depths; gemma-4's per-layer residual scalars.
grid      per dose and readout depth of the S5a grid: the naive skip term (dose x cosine, identity residual), the
          layer-scalar pass-through, the slope and R^2 of the raw shift on the naive skip term per source, the self
          effect and the off-concept |z| > 3 share under the naive, pass-through and fitted-slope skip terms.
report    results/diag_prop/stage1.{json,md}: the stage-1 table across models with one-line readings.

    python -m cm.diag_prop anatomy [--workers 8]     results/<tag>/diag_prop/anatomy.json
    python -m cm.diag_prop grid                      results/<tag>/diag_prop/grid.json
    python -m cm.diag_prop report --tags gemma-4-31b-it,qwen3.5-27b,gemma-3-4b-it
stage 2 (D-060 item 2 as amended by the stage-1 ruling): per cell, injection depth (the four depths), direction (raw
diff of means; the probe's primal form sd * u; the diff of means with the top-8 outlier dims masked) and dose in
class-gap units (1/8, 1/4, 1/2, 1 of the median raw class gap at that depth), the teacher-forced reply32 readout at
every depth on 24 test items of the cell, 16 random directions per depth and dose as the band, and free generation
with the coherence judge. Effects in gap units beyond the layer-scalar pass-through.
    python -m cm.diag_prop stage2-setup [--workers 8]                       CPU   results/<tag>/diag_prop/stage2/setup.json, outputs/.../setup.npz
    python -m cm.diag_prop stage2-run --cells d --shard K --nshards N       GPU   outputs/<tag>/diag_prop/stage2/run_<cells>_shardK.npz
    python -m cm.diag_prop stage2-gen --cells d                             GPU   gen_<cells>.{jsonl,npz}
    python -m cm.diag_prop stage2-judge --tags a,b --cells d                GPU   gen_<cells>_coherence.json per tag
    python -m cm.diag_prop stage2-report --cells d                          CPU   results/<tag>/diag_prop/stage2/report_<cells>.json
    python -m cm.diag_prop stage2-table --tags a,b --cells d                CPU   results/diag_prop/stage2_<cells>.md
CM_MODEL selects the subject; unset, the Gemma-3-4B reference at the unprefixed paths.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import cfg
from .paths import MODELS, OUTPUTS as ROOT_OUTPUTS, RESULTS as ROOT_RESULTS, snapshot
from .s3 import ACTS, auroc
from .s3_nuisance import nuisance_dirs, row_labels
from .s4_overlap import LIDX, unit_vec
from .s5_causal import GRID, HON_SOURCES

RESULTS = cfg.results_dir()
OUT = RESULTS / "diag_prop"
CONCEPTS = ["A9", "C2", "C1", "C3", "C8", "C19", "K1", "K2"]
HON = [c for c in HON_SOURCES if not c.startswith(("K1|", "K2|"))]
L_IN, L_OUT = cfg.L_STEER, tuple(cfg.L_OUT)
H_IN = L_IN + 1
STEER_IDX = LIDX.index(H_IN)
TOPK = (8, 32, 128)
C_PROBE, MAX_ITER = 1.0, 2000


def layer_scalars():
    """gemma-4's per-layer residual multipliers from the checkpoint (None where the architecture has none)."""
    if not cfg.IS_GEMMA4:
        return None
    from safetensors import safe_open
    p = snapshot(MODELS[cfg.MODEL])
    idx = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {i: f"model.language_model.layers.{i}.layer_scalar" for i in range(cfg.SPEC["n_layers"])}
    if keys[0] not in idx:
        return None
    vals = {}
    by_file = {}
    for i, k in keys.items():
        by_file.setdefault(idx[k], []).append((i, k))
    for f, ks in by_file.items():
        with safe_open(str(p / f), "pt") as st:
            for i, k in ks:
                vals[i] = float(st.get_tensor(k).float())
    return [vals[i] for i in range(cfg.SPEC["n_layers"])]


def pass_through(scal, l_in, l_out):
    """Product of the residual multipliers a vector added to the output of layer l_in meets before the output of
    layer l_out is read (1 where there are none)."""
    if scal is None:
        return 1.0
    return float(np.prod(scal[l_in + 1:l_out + 1]))


def topk_share(u, order):
    """Fraction of the unit vector's squared norm on the first k dims of `order`, and on its own k largest entries."""
    u2 = u ** 2 / max(float(u @ u), 1e-24)
    own = np.sort(u2)[::-1]
    return {f"top{k}": float(u2[order[:k]].sum()) for k in TOPK} | {f"own{k}": float(own[:k].sum()) for k in TOPK}


def probe_dir(Xs, y):
    from sklearn.linear_model import LogisticRegression
    w = np.zeros(len(y))
    for p in (0, 1):
        w[y == p] = 1.0 / max((y == p).sum(), 1)
    clf = LogisticRegression(C=C_PROBE, max_iter=MAX_ITER).fit(Xs, y, sample_weight=w)
    return unit_vec(clf.coef_[0].astype(np.float64))


def _probe_job(args):
    cid_cell, Xs, y = args
    return cid_cell, probe_dir(Xs, y)


def cell_rows(meta, sub, f):
    rows = {s: [i for i, m in enumerate(meta) if (m["contrast"] or "all") == sub and m["frame"] == f and m["split"] == s]
            for s in ("train", "test")}
    y = {s: np.array([meta[i]["polarity"] == "pos" for i in rows[s]], int) for s in rows}
    return rows, y


def verdict_projected(cid, meta, R):
    """reply32 rows at the four depths with the verdict direction projected out of a/b rows (cm.s4_overlap.concept_data)."""
    X = R[:, LIDX].astype(np.float64)
    _, apply = row_labels(cid, meta, "verdict")
    dirs, _ = nuisance_dirs(cid, "verdict")
    if "reply32" in dirs and apply["reply32"].any():
        u = dirs["reply32"][LIDX]
        m = apply["reply32"]
        X[m] -= np.einsum("nl,ld->nld", np.einsum("nld,ld->nl", X[m], u), u)
    return X


def cmd_anatomy(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    S, d = cfg.N_STATES, cfg.D
    scal = layer_scalars()
    out = {"model": cfg.MODEL, "depth_layers": list(cfg.DEPTH_LAYERS), "hidden_index": list(LIDX), "steer_layer": L_IN,
           "layer_scalar": scal,
           "pass_through": {str(L): pass_through(scal, L_IN, L) for L in L_OUT}}
    sumsq_b = np.zeros((S, d)); sumsq_r = np.zeros((S, d)); n_b = n_r = 0
    norms_b, norms_r, bnd_in = [], [], []
    cells, prof = {}, {}
    for cid in CONCEPTS:
        f = np.load(ACTS / f"{cid}.npz")
        meta = json.loads(str(f["meta"]))
        B = f["boundary"]; R = f["reply32"]
        for i in range(0, len(B), 64):
            sumsq_b += np.einsum("nsd,nsd->sd", B[i:i + 64], B[i:i + 64], dtype=np.float64)
            sumsq_r += np.einsum("nsd,nsd->sd", R[i:i + 64], R[i:i + 64], dtype=np.float64)
        n_b += len(B); n_r += len(R)
        norms_b.append(np.linalg.norm(B, axis=2)); norms_r.append(np.linalg.norm(R, axis=2))
        bnd_in.append(B[:, H_IN].astype(np.float32))
        X4 = verdict_projected(cid, meta, R)
        for uid in HON_SOURCES:
            c, sub, fr = uid.split("|")
            if c != cid:
                continue
            rows, y = cell_rows(meta, sub, fr)
            tr, te = rows["train"], rows["test"]
            if not tr or not te or len(set(y["train"])) < 2 or len(set(y["test"])) < 2:
                raise SystemExit(f"{uid}: no train or test rows of both poles in this model's activations")
            cells[uid] = {"Xtr": X4[tr].astype(np.float32), "ytr": y["train"], "Xte": X4[te].astype(np.float32), "yte": y["test"]}
            # all-layer profile on the raw rows (no verdict projection): direction, AUROC, persistence, native maintenance
            Rtr, Rte = R[tr].astype(np.float64), R[te].astype(np.float64)
            v = Rtr[y["train"] == 1].mean(0) - Rtr[y["train"] == 0].mean(0)          # [S, d]
            sd = Rtr.std(0) + 1e-6
            u = unit_vec(v); us = unit_vec(v / sd)
            gte = Rte[y["test"] == 1].mean(0) - Rte[y["test"] == 0].mean(0)          # test-row class gap per layer
            a_dm = [auroc(np.einsum("nd,d->n", Rte[:, L], u[L]), y["test"]) for L in range(S)]
            a_sdm = [auroc(np.einsum("nd,d->n", Rte[:, L] / sd[L], us[L]), y["test"]) for L in range(S)]
            u_in = u[H_IN]
            prof[uid] = {"auroc_dm": np.round(a_dm, 4).tolist(), "auroc_sdm": np.round(a_sdm, 4).tolist(),
                         "gap_norm": np.linalg.norm(v, axis=1).round(4).tolist(),
                         "gap_norm_std": np.linalg.norm(v / sd, axis=1).round(4).tolist(),
                         "cos_to_steer_raw": (u @ u_in).round(4).tolist(),
                         "cos_to_steer_std": (us @ us[H_IN]).round(4).tolist(),
                         # native maintenance: the test-row class gap along the steering-depth direction, per layer,
                         # over its value at the steering depth
                         "gap_along_steer": (gte @ u_in).round(4).tolist(),
                         "maintenance": ((gte @ u_in) / max(float(gte[H_IN] @ u_in), 1e-9)).round(4).tolist()}
        del B, R, X4, f
        print(f"[diag] {cid} loaded", flush=True)
    rms_b = np.sqrt(sumsq_b / n_b); rms_r = np.sqrt(sumsq_r / n_r)                    # [S, d]
    nb = np.concatenate(norms_b); nr = np.concatenate(norms_r)                        # [N, S]
    order = {L: np.argsort(-rms_b[L]) for L in range(S)}
    ms_b = rms_b ** 2
    out["norm_profile"] = {
        "median_token_norm": np.median(nb, 0).round(3).tolist(), "median_reply32_norm": np.median(nr, 0).round(3).tolist(),
        "topk_share_tokens": {f"top{k}": [float(ms_b[L, order[L][:k]].sum() / ms_b[L].sum()) for L in range(S)] for k in TOPK},
        "top8_dims": {str(L): order[L][:8].tolist() for L in LIDX},
        "top8_rms": {str(L): rms_b[L, order[L][:8]].round(2).tolist() for L in LIDX},
        "rms_median_dim": {str(L): float(np.median(rms_b[L])) for L in LIDX}}
    # pooled standardized space per depth over the 22 honesty cells (cm.coherence.pool_dirs)
    lev = np.load(RESULTS / "coherence" / "levers.npz") if (RESULTS / "coherence" / "levers.npz").exists() else None
    setup_npz = _setup_npz()
    dense_of = {}
    if setup_npz is not None:
        Sj = json.loads((_setup_dir()[0] / "setup.json").read_text())
        dense_of = {s: setup_npz["dense"][k] for k, s in enumerate(Sj["sources"])}
    per_cell, mats = {}, {}
    for li, hidx in enumerate(LIDX):
        Xp = np.concatenate([cells[c]["Xtr"][:, li] for c in HON]).astype(np.float64)
        sd = Xp.std(0) + 1e-6
        mu = Xp.mean(0)
        o = order[hidx]
        dirs = {}
        for uid in HON_SOURCES:
            Xtr, ytr = cells[uid]["Xtr"][:, li].astype(np.float64), cells[uid]["ytr"]
            v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
            u_raw, u_std = unit_vec(v), unit_vec(v / sd)
            dirs[uid] = {"raw": u_raw, "std": u_std, "primal": unit_vec(sd * u_std), "dual": unit_vec(u_std / sd),
                         "gap_raw": float(np.linalg.norm(v)), "gap_std": float(np.linalg.norm(v / sd))}
        mats[hidx] = {"std": np.array([[dirs[a]["std"] @ dirs[b]["std"] for b in HON_SOURCES] for a in HON_SOURCES]),
                      "raw": np.array([[dirs[a]["raw"] @ dirs[b]["raw"] for b in HON_SOURCES] for a in HON_SOURCES])}
        probes = {}
        if hidx == H_IN:
            jobs = [(uid, cells[uid]["Xtr"][:, li].astype(np.float64) / sd, cells[uid]["ytr"]) for uid in HON_SOURCES]
            with ProcessPoolExecutor(workers) as ex:
                for uid, p in ex.map(_probe_job, jobs):
                    probes[uid] = p
        for uid in HON_SOURCES:
            D_ = dirs[uid]
            rec = {"gap_raw": D_["gap_raw"], "gap_std": D_["gap_std"],
                   "cos_std_raw": float(D_["std"] @ D_["raw"]), "cos_primal_injected": None, "cos_std_lever": None,
                   "share": {k: topk_share(D_[k], o) for k in ("raw", "std", "primal", "dual")}}
            if hidx == H_IN:
                if uid in dense_of:
                    rec["cos_primal_injected"] = float(D_["primal"] @ dense_of[uid])
                    rec["cos_raw_injected"] = float(D_["raw"] @ dense_of[uid])
                    rec["share"]["injected"] = topk_share(dense_of[uid], o)
                if lev is not None and f"cell|{uid}" in lev.files:
                    rec["cos_std_lever"] = float(D_["std"] @ lev[f"cell|{uid}"])
                if uid in probes:
                    p = probes[uid]
                    rec["cos_dm_probe"] = float(D_["std"] @ p)
                    rec["share"]["probe"] = topk_share(p, o)
                    rec["share"]["probe_primal"] = topk_share(unit_vec(sd * p), o)
                    rec["auroc_probe_test"] = auroc((cells[uid]["Xte"][:, li].astype(np.float64) / sd) @ p, cells[uid]["yte"])
                    rec["auroc_sdm_test"] = auroc((cells[uid]["Xte"][:, li].astype(np.float64) / sd) @ D_["std"], cells[uid]["yte"])
            per_cell.setdefault(uid, {})[str(hidx)] = rec
        if hidx == H_IN:
            # the pooled levers and the dose table
            pooled = {}
            if lev is not None:
                for k in ("dm_all", "probe_all", "all_frames"):
                    u = lev[k]
                    pooled[k] = {"share_std": topk_share(u, o), "share_primal": topk_share(unit_vec(sd * u), o),
                                 "share_dual": topk_share(unit_vec(u / sd), o), "cos_sd_lever_sd": float(unit_vec(sd) @ unit_vec(lev["sd"]))}
                pooled["cos_dm_probe"] = float(lev["dm_all"] @ lev["probe_all"])
            out["pooled_levers"] = pooled
            N_raw = _median_token_norm()
            tok = np.concatenate([np.linalg.norm(b / sd, axis=1) for b in bnd_in])
            Ns_tok = float(np.median(tok))
            Ns_rows = float(np.median(np.linalg.norm((Xp - mu) / sd, axis=1)))
            gaps_raw = np.array([dirs[c]["gap_raw"] for c in HON]); gaps_std = np.array([dirs[c]["gap_std"] for c in HON])
            inj = {c: dense_of.get(c, dirs[c]["raw"]) for c in HON}
            rows = []
            for c in GRID:
                dose = c * N_raw
                ds = np.median([np.linalg.norm(dose * inj[u] / sd) for u in HON])
                rows.append({"c": c, "dose_raw": dose, "dose_over_gap_raw": float(dose / np.median(gaps_raw)),
                             "dose_std": float(ds), "dose_std_over_token_norm": float(ds / Ns_tok),
                             "dose_std_over_gap_std": float(ds / np.median(gaps_std)),
                             "share_top8_outlier": float(np.median([topk_share(inj[u], o)["top8"] for u in HON]))})
            out["dose"] = {"median_token_norm_raw": N_raw, "median_token_norm_std": Ns_tok, "median_row_norm_std_centred": Ns_rows,
                           "sqrt_d": float(np.sqrt(d)), "median_gap_raw": float(np.median(gaps_raw)),
                           "median_gap_std": float(np.median(gaps_std)), "grid": rows}
            out["sd_profile"] = {"topk_share_sd2": {f"top{k}": float((sd ** 2)[o[:k]].sum() / (sd ** 2).sum()) for k in TOPK},
                                 "top8_sd_dims": np.argsort(-sd)[:8].tolist(),
                                 "overlap_top8_rms_sd": len(set(o[:8].tolist()) & set(np.argsort(-sd)[:8].tolist()))}
    out["cells"] = per_cell
    out["profiles"] = prof
    # cosine matrices and their summaries
    idx = {u: i for i, u in enumerate(HON_SOURCES)}
    hon_i = [idx[u] for u in HON]
    cm_out = {}
    for hidx, M in mats.items():
        rec = {"ids": HON_SOURCES}
        for kind, A in M.items():
            H = A[np.ix_(hon_i, hon_i)]
            off = ~np.eye(len(hon_i), dtype=bool)
            same_def = np.array([[HON[i].split("|")[0] == HON[j].split("|")[0] for j in range(len(HON))] for i in range(len(HON))])
            same_fr = np.array([[HON[i][-1] == HON[j][-1] for j in range(len(HON))] for i in range(len(HON))])
            rec[kind] = {"matrix": np.round(A, 3).tolist(),
                         "mean_abs_off": float(np.abs(H[off]).mean()), "mean_off": float(H[off].mean()),
                         "mean_same_def": float(H[off & same_def].mean()), "mean_other_def": float(H[off & ~same_def].mean()),
                         "mean_same_frame": float(H[off & same_fr].mean()), "mean_other_frame": float(H[off & ~same_fr].mean()),
                         "max_abs_off": float(np.abs(H[off]).max())}
        cm_out[str(hidx)] = rec
    out["cosines"] = cm_out
    (OUT / "anatomy.json").write_text(json.dumps(out, default=float))
    _print_anatomy(out)


def _setup_dir():
    """(results dir, outputs dir) of the S5a run whose sources are the honesty cells."""
    if cfg.TAG:
        return RESULTS / "s5_hon", cfg.outputs_dir() / "s5_hon"
    return RESULTS / "s5", cfg.outputs_dir() / "s5"


def _setup_npz():
    _, o = _setup_dir()
    return np.load(o / "setup.npz") if (o / "setup.npz").exists() else None


def _median_token_norm():
    return json.loads((RESULTS / "s5" / f"residual_norm_L{L_IN}.json").read_text())["median_token_norm"]


def _print_anatomy(out):
    h = str(H_IN)
    med = lambda key, kind, k: float(np.median([out["cells"][u][h]["share"][kind][key] for u in HON if kind in out["cells"][u][h]["share"]]))
    print(f"[diag] {out['model']}: pass-through {out['pass_through']}", flush=True)
    for kind in ("injected", "raw", "std", "primal", "dual", "probe"):
        if kind in out["cells"][HON[0]][h]["share"]:
            print(f"[diag] share of norm on outlier dims, {kind}: " + " ".join(f"top{k} {med(f'top{k}', kind, k):.3f}" for k in TOPK)
                  + " | own: " + " ".join(f"{med(f'own{k}', kind, k):.3f}" for k in TOPK), flush=True)
    np_ = out["norm_profile"]
    print(f"[diag] token norm at depths {[np_['median_token_norm'][i] for i in LIDX]}; top8 share of mean square "
          f"{[round(np_['topk_share_tokens']['top8'][i], 3) for i in LIDX]}", flush=True)
    for r in out["dose"]["grid"]:
        print(f"[diag] dose 1/{round(1 / r['c'])}: raw {r['dose_raw']:.2f} = {r['dose_over_gap_raw']:.2f} class gaps; "
              f"standardized {r['dose_std']:.1f} = {r['dose_std_over_token_norm']:.3f} token norms = {r['dose_std_over_gap_std']:.2f} std gaps", flush=True)
    for u in HON:
        p = out["profiles"][u]
        print(f"[diag] {u}: maintenance at depths {[p['maintenance'][i] for i in LIDX]}; cos to steer raw {[p['cos_to_steer_raw'][i] for i in LIDX]}", flush=True)


def _matrices():
    base, _ = _setup_dir()
    ms = sorted(base.glob("c*/causal_matrix.json"))
    if (base / "causal_matrix.json").exists():
        ms.append(base / "causal_matrix.json")
    dose_base = base.parent / "s5_dose"
    if dose_base.exists():
        ms += sorted(dose_base.glob("c*/causal_matrix.json"))
    return ms


def cmd_grid():
    OUT.mkdir(parents=True, exist_ok=True)
    scal = layer_scalars()
    res_dir, _ = _setup_dir()
    A = _setup_npz()
    dense, tg = A["dense"], [A["tgt_out0"], A["tgt_out1"]]
    out = {"model": cfg.MODEL, "layers": list(L_OUT), "pass_through": {str(L): pass_through(scal, L_IN, L) for L in L_OUT}, "runs": {}}
    for mp in _matrices():
        R = json.loads(mp.read_text())
        if not {"target_gap", "random_sd", "dose_norm"} <= set(R):
            print(f"[diag] {mp}: older matrix format without target_gap/random_sd, skipped", flush=True)
            continue
        src, tgt, dose = R["sources"], R["targets"], R["dose_norm"]
        gap = np.array(R["target_gap"], float)                 # [2, T], nan where undefined
        sd_r = np.array(R["random_sd"], float)
        E = np.array(R["kinds"]["dense+"]["E"], float); Z = np.array(R["kinds"]["dense+"]["Z"], float)
        Em = np.array(R["kinds"]["dense-"]["E"], float)
        six = {u: i for i, u in enumerate(src)}; tix = {u: i for i, u in enumerate(tgt)}
        hon_s = [u for u in HON if u in six]
        run = {"path": str(mp), "dose_c": R["dose_c"], "dose_norm": dose, "n_sources_honesty": len(hon_s), "by_layer": {}}
        for li, L in enumerate(L_OUT):
            skip = dose * (tg[li] @ dense.T).T / gap[li][None, :]          # [S, T] naive skip term in gap units
            Er = np.array(R["kinds"]["dense+"]["E_raw"], float)[:, li] if "E_raw" in R["kinds"]["dense+"] else E[:, li] + skip
            Erm = np.array(R["kinds"]["dense-"]["E_raw"], float)[:, li] if "E_raw" in R["kinds"]["dense-"] else Em[:, li] - skip
            mu = E[:, li] - Z[:, li] * sd_r[li][None, :]                    # the random-direction mean effect per target
            mu = np.nanmedian(mu, 0)
            pi = out["pass_through"][str(L)]
            ok = np.isfinite(gap[li])
            per, slopes, r2s = {}, [], []
            for u in hon_s + [s for s in src if s.startswith(("K1|", "K2|"))]:
                s = six[u]
                x, y = skip[s, ok], Er[s, ok]
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                slope = float((x @ y) / max(x @ x, 1e-12))
                r2 = float(1 - ((y - slope * x) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
                rec = {"slope_raw_on_skip": slope, "r2": r2, "n_targets": int(m.sum()),
                       "sign_sym_r": float(np.corrcoef(y, -Erm[s, ok][m])[0, 1]) if m.sum() > 2 else None}
                if u in tix:
                    t = tix[u]
                    rec.update({"self_raw": float(Er[s, t]), "self_skip_naive": float(skip[s, t]),
                                "self_skip_pass": float(pi * skip[s, t]), "self_E_naive": float(E[s, li, t]),
                                "self_E_pass": float(Er[s, t] - pi * skip[s, t]), "self_z_naive": float(Z[s, li, t]),
                                "self_z_pass": float((Er[s, t] - pi * skip[s, t] - mu[t]) / sd_r[li, t]),
                                "self_z_fit": float((Er[s, t] - slope * skip[s, t] - mu[t]) / sd_r[li, t])})
                per[u] = rec
                if u in hon_s:
                    slopes.append(slope); r2s.append(r2)
            # off-concept spillover under the three skip terms, honesty sources only
            hs = [six[u] for u in hon_s]
            offc = np.array([[tgt[t].split("|")[0] != src[s].split("|")[0] for t in range(len(tgt))] for s in hs])
            fin = np.isfinite(gap[li])[None, :] & offc
            z_naive = Z[hs, li]
            z_pass = (Er[hs] - pi * skip[hs] - mu[None, :]) / sd_r[li][None, :]
            al = np.array([per[u]["slope_raw_on_skip"] for u in hon_s])[:, None]
            z_fit = (Er[hs] - al * skip[hs] - mu[None, :]) / sd_r[li][None, :]
            # off-diagonal over diagonal among honesty cells under each term
            def ratio(Ex):
                dg = [Ex[i, tix[u]] for i, u in enumerate(hon_s) if u in tix]
                of = [Ex[i, tix[b]] for i, u in enumerate(hon_s) for b in HON if b in tix and b != u]
                return float(np.nanmean(of) / np.nanmean(dg)) if np.nanmean(dg) > 0 else None, float(np.nanmean(dg)), float(np.nanmean(of))
            r_naive, d_naive, o_naive = ratio(E[hs, li]); r_pass, d_pass, o_pass = ratio(Er[hs] - pi * skip[hs]); r_raw, d_raw, o_raw = ratio(Er[hs])
            # pooled slope over all honesty (source, target) pairs with a defined gap
            allfin = np.broadcast_to(np.isfinite(gap[li])[None, :], offc.shape)
            x = skip[hs][allfin]; y = Er[hs][allfin]
            m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
            pooled_slope = float((x @ y) / max(x @ x, 1e-12))
            pooled_r2 = float(1 - ((y - pooled_slope * x) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
            run["by_layer"][str(L)] = {
                "pass_through": pi, "n_targets_defined": int(ok.sum()), "n_targets": len(tgt),
                "random_sd_median": float(np.nanmedian(sd_r[li])), "random_mu_median": float(np.nanmedian(mu)),
                "slope_median": float(np.median(slopes)), "slope_iqr": [float(np.percentile(slopes, 25)), float(np.percentile(slopes, 75))],
                "r2_median": float(np.median(r2s)), "pooled_slope": pooled_slope, "pooled_r2": pooled_r2,
                "self": {k: float(np.nanmedian([per[u][k] for u in hon_s if k in per[u]])) for k in
                         ("self_raw", "self_skip_naive", "self_skip_pass", "self_E_naive", "self_E_pass", "self_z_naive", "self_z_pass", "self_z_fit")},
                "share_self_z_gt3": {k: float(np.mean([per[u][k] > 3 for u in hon_s if k in per[u]])) for k in ("self_z_naive", "self_z_pass", "self_z_fit")},
                "spillover_off_concept_absz_gt3": {"naive": float(np.nanmean(np.abs(z_naive[fin]) > 3)),
                                                   "pass": float(np.nanmean(np.abs(z_pass[fin]) > 3)),
                                                   "fit": float(np.nanmean(np.abs(z_fit[fin]) > 3))},
                "off_over_diag": {"naive": r_naive, "pass": r_pass, "raw": r_raw},
                "mean_diag_off": {"naive": [d_naive, o_naive], "pass": [d_pass, o_pass], "raw": [d_raw, o_raw]},
                "per_source": per}
            print(f"[diag] {mp.parent.name} layer {L}: pass-through {pi:.3f}; slope of raw shift on naive skip median {np.median(slopes):.3f} "
                  f"(R2 {np.median(r2s):.2f}, pooled {pooled_slope:.3f}/{pooled_r2:.2f}); self z naive/pass/fit "
                  f"{run['by_layer'][str(L)]['self']['self_z_naive']:.1f}/{run['by_layer'][str(L)]['self']['self_z_pass']:.1f}/{run['by_layer'][str(L)]['self']['self_z_fit']:.1f}; "
                  f"spillover naive/pass/fit {np.nanmean(np.abs(z_naive[fin]) > 3):.3f}/{np.nanmean(np.abs(z_pass[fin]) > 3):.3f}/{np.nanmean(np.abs(z_fit[fin]) > 3):.3f}; "
                  f"off/diag naive {r_naive} pass {r_pass} raw {r_raw}; random sd {np.nanmedian(sd_r[li]):.4f}", flush=True)
        out["runs"][mp.parent.name if mp.parent.name.startswith("c") else "base"] = run
    (OUT / "grid.json").write_text(json.dumps(out, default=float))


def cmd_report(tags):
    cols = {}
    for tag in tags:
        rd = ROOT_RESULTS / cfg.SPECS[tag]["tag"] if cfg.SPECS[tag]["tag"] else ROOT_RESULTS
        a = json.loads((rd / "diag_prop" / "anatomy.json").read_text()) if (rd / "diag_prop" / "anatomy.json").exists() else None
        g = json.loads((rd / "diag_prop" / "grid.json").read_text()) if (rd / "diag_prop" / "grid.json").exists() else None
        cols[tag] = {"anatomy": a, "grid": g}
    od = ROOT_RESULTS / "diag_prop"; od.mkdir(parents=True, exist_ok=True)
    (od / "stage1.json").write_text(json.dumps(cols, default=float))
    lines = ["# D-060 stage 1: direction, dose, hook and spillover anatomy", "", "Columns: " + ", ".join(tags), ""]

    def row(name, vals):
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    lines += ["| check | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    A = {t: cols[t]["anatomy"] for t in tags}; G = {t: cols[t]["grid"] for t in tags}
    fmt = lambda x, p=3: ("n/a" if x is None else f"{x:.{p}f}")
    depths = {t: A[t]["depth_layers"] if A[t] else cfg.SPECS[t]["depths"] for t in tags}
    hid = {t: [L + 1 for L in depths[t]] for t in tags}
    row("steering layer / readout layers", [f"{depths[t][1]} / {depths[t][2]}, {depths[t][3]}" for t in tags])
    row("residual multiplier product, injection to readout 3 / 4", [", ".join(fmt(v) for v in A[t]["pass_through"].values()) if A[t] else "n/a" for t in tags])
    row("median token norm at the 4 depths", [", ".join(f"{A[t]['norm_profile']['median_token_norm'][i]:.0f}" for i in hid[t]) if A[t] else "n/a" for t in tags])
    for k in TOPK:
        row(f"top-{k} dims' share of token mean square, 4 depths", [", ".join(f"{A[t]['norm_profile']['topk_share_tokens'][f'top{k}'][i]:.2f}" for i in hid[t]) if A[t] else "n/a" for t in tags])
    steer = {t: str(hid[t][1]) for t in tags}
    for kind, label in (("injected", "injected raw diff of means (S5a)"), ("std", "standardized direction"), ("primal", "primal raw form sd*u"),
                        ("dual", "dual raw form u/sd (levers |resid)"), ("probe", "probe (standardized)")):
        for k in TOPK:
            vals = []
            for t in tags:
                if not A[t]:
                    vals.append("n/a"); continue
                xs = [A[t]["cells"][u][steer[t]]["share"][kind][f"top{k}"] for u in HON if kind in A[t]["cells"][u][steer[t]]["share"]]
                vals.append(fmt(float(np.median(xs))) if xs else "n/a")
            row(f"{label}: median share on top-{k} outlier dims", vals)
    row("cos(standardized, raw diff of means), median", [fmt(float(np.median([A[t]["cells"][u][steer[t]]["cos_std_raw"] for u in HON]))) if A[t] else "n/a" for t in tags])
    row("cos(diff of means, probe) per cell, median", [fmt(float(np.median([A[t]["cells"][u][steer[t]].get("cos_dm_probe", np.nan) for u in HON]))) if A[t] else "n/a" for t in tags])
    row("cos(primal raw form, injected vector), median", [fmt(float(np.nanmedian([A[t]["cells"][u][steer[t]].get("cos_primal_injected") or np.nan for u in HON]))) if A[t] else "n/a" for t in tags])
    row("pooled levers cos(dm_all, probe_all)", [fmt(A[t]["pooled_levers"].get("cos_dm_probe")) if A[t] and A[t].get("pooled_levers") else "n/a" for t in tags])
    for r_i, c in enumerate(GRID):
        vals = []
        for t in tags:
            if not A[t]:
                vals.append("n/a"); continue
            r = A[t]["dose"]["grid"][r_i]
            vals.append(f"{r['dose_raw']:.1f} raw = {r['dose_over_gap_raw']:.1f} gaps; std {r['dose_std_over_token_norm']:.3f} of token norm = {r['dose_std_over_gap_std']:.1f} std gaps")
        row(f"dose 1/{round(1 / c)}", vals)
    row("cell cosines (std), mean |off-diag| / same-def / other-def", [", ".join(fmt(A[t]["cosines"][steer[t]]["std"][k]) for k in ("mean_abs_off", "mean_same_def", "mean_other_def")) if A[t] else "n/a" for t in tags])
    row("cell cosines (raw), mean |off-diag| / same-def / other-def", [", ".join(fmt(A[t]["cosines"][steer[t]]["raw"][k]) for k in ("mean_abs_off", "mean_same_def", "mean_other_def")) if A[t] else "n/a" for t in tags])
    row("native maintenance of the steering-depth gap at depths 3 / 4, median over cells",
        [", ".join(fmt(float(np.median([A[t]["profiles"][u]["maintenance"][i] for u in HON]))) for i in hid[t][2:]) if A[t] else "n/a" for t in tags])
    row("cos(direction at depth 3 / 4, direction at steering depth), raw, median",
        [", ".join(fmt(float(np.median([A[t]["profiles"][u]["cos_to_steer_raw"][i] for u in HON]))) for i in hid[t][2:]) if A[t] else "n/a" for t in tags])
    lines.append("")
    lines += ["| grid run, readout | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    runs = sorted({k for t in tags if G[t] for k in G[t]["runs"]}, key=lambda k: float(k[1:]) if k.startswith("c") else 1e9)
    for k in runs:
        for L_i in (0, 1):
            vals = []
            for t in tags:
                if not G[t] or k not in G[t]["runs"]:
                    vals.append("n/a"); continue
                L = str(G[t]["layers"][L_i]); b = G[t]["runs"][k]["by_layer"][L]
                vals.append(f"slope {b['slope_median']:.2f} (R2 {b['r2_median']:.2f}); pass-through {b['pass_through']:.3f}; "
                            f"self z naive/pass/fit {b['self']['self_z_naive']:.1f}/{b['self']['self_z_pass']:.1f}/{b['self']['self_z_fit']:.1f}; "
                            f"spill {b['spillover_off_concept_absz_gt3']['naive']:.2f}/{b['spillover_off_concept_absz_gt3']['pass']:.2f}/{b['spillover_off_concept_absz_gt3']['fit']:.2f}; "
                            f"off/diag naive {fmt(b['off_over_diag']['naive'], 2)} pass {fmt(b['off_over_diag']['pass'], 2)} raw {fmt(b['off_over_diag']['raw'], 2)}; "
                            f"gaps defined {b['n_targets_defined']}/{b['n_targets']}")
            row(f"{k}, readout {L_i + 3}", vals)
    (od / "stage1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- stage 2 (D-060 item 2 with the stage-1 amendments a to e)

DOSE_GAPS = (0.125, 0.25, 0.5, 1.0)             # a: fractions of the median raw class gap at the injection depth
GEN_MAX_GAP = 0.5 if cfg.IS_GEMMA4 else 1.0     # c: free generation up to half a gap on gemma-4, one gap on Qwen
N_RAND2, N_RAND_GEN, PER_POLE, MASK_K, GEN_TOKENS = 8, 4, 12, 8, 32
KINDS2 = ("dm", "probe", "masked")               # d: raw diff of means, the probe's primal form sd * u, the top-8-masked diff of means
CELLS_D = [c for c in HON if c.endswith("|d")]
OUT2 = OUT / "stage2"
GEN2 = cfg.outputs_dir() / "diag_prop" / "stage2"
FWD_BATCH, FWD_TOKENS = (4, 6000) if cfg.D >= 5000 else (16, 24000)
NL = len(LIDX)


def select_cells(sel):
    return {"d": CELLS_D, "rest": [c for c in HON if c not in CELLS_D], "all": HON}[sel]


def _probe_job2(args):
    key, Xs, y = args
    return key, probe_dir(Xs, y)


def cmd_stage2_setup(workers):
    """Per cell and injection depth: the three directions, the class gap, the doses, 24 test items; the random
    directions; the pass-through table. CPU, reads the acts of the six honesty concepts."""
    import hashlib
    OUT2.mkdir(parents=True, exist_ok=True); GEN2.mkdir(parents=True, exist_ok=True)
    scal = layer_scalars()
    cells, items, rms, n_b = {}, [], np.zeros((NL, cfg.D)), 0
    for cid in CONCEPTS[:6]:
        f = np.load(ACTS / f"{cid}.npz"); meta = json.loads(str(f["meta"]))
        B = f["boundary"][:, LIDX]; R = f["reply32"]
        rms += np.einsum("nsd,nsd->sd", B, B, dtype=np.float64); n_b += len(B)
        X4 = verdict_projected(cid, meta, R)
        for uid in HON:
            c, sub, fr = uid.split("|")
            if c != cid:
                continue
            rows, y = cell_rows(meta, sub, fr)
            cells[uid] = {"Xtr": X4[rows["train"]].astype(np.float32), "ytr": y["train"], "Xte": X4[rows["test"]].astype(np.float32), "yte": y["test"]}
            r = np.random.default_rng(int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16))
            for p in ("pos", "neg"):
                te = [meta[i]["item_id"] for i in rows["test"] if meta[i]["polarity"] == p]
                for i in r.choice(te, min(PER_POLE, len(te)), replace=False):
                    items.append({"cell": uid, "item_id": str(i), "polarity": p})
        del f, B, R, X4
        print(f"[diag] {cid} loaded", flush=True)
    rms = np.sqrt(rms / n_b)
    d = cfg.D
    dm, masked, probe = (np.zeros((NL, len(HON), d)) for _ in range(3))
    sd_all, gaps, mask_dims, jobs = np.zeros((NL, d)), np.zeros((NL, len(HON))), [], []
    for li in range(NL):
        Xp = np.concatenate([cells[c]["Xtr"][:, li] for c in HON]).astype(np.float64)
        sd = Xp.std(0) + 1e-6; sd_all[li] = sd
        order = np.argsort(-rms[li]); mask_dims.append(order[:MASK_K].tolist())
        for k, uid in enumerate(HON):
            Xtr, ytr = cells[uid]["Xtr"][:, li].astype(np.float64), cells[uid]["ytr"]
            v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
            gaps[li, k] = np.linalg.norm(v); dm[li, k] = unit_vec(v)
            vm = v.copy(); vm[order[:MASK_K]] = 0; masked[li, k] = unit_vec(vm)
            jobs.append(((li, k), Xtr / sd, ytr))
    with ProcessPoolExecutor(workers) as ex:
        for (li, k), p in ex.map(_probe_job2, jobs):
            probe[li, k] = unit_vec(sd_all[li] * p)
    auc = {}
    for li in range(NL):
        for k, uid in enumerate(HON):
            Xte, yte = cells[uid]["Xte"][:, li].astype(np.float64), cells[uid]["yte"]
            # the probe is read through its dual (x / sd) . p, the two diff-of-means forms through the raw projection
            auc[f"{uid}|{LIDX[li]}"] = {"dm": float(auroc(Xte @ dm[li, k], yte)), "masked": float(auroc(Xte @ masked[li, k], yte)),
                                        "probe": float(auroc(Xte @ (probe[li, k] / sd_all[li] ** 2), yte))}
    rand = np.random.default_rng(0).standard_normal((N_RAND2, d)); rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    dose_raw = [[float(np.median(gaps[li]) * g) for g in DOSE_GAPS] for li in range(NL)]
    pt = [[pass_through(scal, cfg.DEPTH_LAYERS[li], cfg.DEPTH_LAYERS[lj]) if lj >= li else None for lj in range(NL)] for li in range(NL)]
    cos = {kd: [float(np.median([D_[li, k] @ dm[li, k] for k in range(len(HON))])) for li in range(NL)] for kd, D_ in (("probe", probe), ("masked", masked))}
    np.savez(GEN2 / "setup.npz", dm=dm, probe=probe, masked=masked, rand=rand, sd=sd_all)
    S = {"model": cfg.MODEL, "cells": HON, "depth_layers": list(cfg.DEPTH_LAYERS), "hidden_index": list(LIDX), "items": items,
         "dose_gaps": list(DOSE_GAPS), "dose_raw": dose_raw, "median_gap": [float(np.median(gaps[li])) for li in range(NL)],
         "gaps": gaps.round(4).tolist(), "mask_dims": mask_dims, "pass_through": pt, "layer_scalar": scal, "auroc_test": auc,
         "cos_to_dm_median": cos, "gen_max_gap": GEN_MAX_GAP}
    (OUT2 / "setup.json").write_text(json.dumps(S, default=float))
    for li in range(NL):
        print(f"[diag] depth {cfg.DEPTH_LAYERS[li]}: median gap {S['median_gap'][li]:.2f}, doses {np.round(dose_raw[li], 2).tolist()}, mask dims {mask_dims[li]}, "
              f"cos(probe, dm) {cos['probe'][li]:.2f} cos(masked, dm) {cos['masked'][li]:.3f}, test AUROC dm/probe/masked median "
              + "/".join(f"{np.median([auc[f'{u}|{LIDX[li]}'][kd] for u in HON]):.3f}" for kd in ("dm", "probe", "masked"))
              + f", pass-through to later depths {[round(x, 3) for x in pt[li] if x is not None]}", flush=True)
    print(f"[diag] {len(items)} items over {len(HON)} cells", flush=True)


def load_stage2():
    return json.loads((OUT2 / "setup.json").read_text()), np.load(GEN2 / "setup.npz")


def stage2_conditions(S, src, rows, gen=False):
    """Conditions of a run: baseline, random directions at every depth and dose, and per source cell every direction,
    depth and dose at +dose (the diff of means also at -dose). Free generation (gen) keeps the doses up to GEN_MAX_GAP,
    + only, and N_RAND_GEN random directions."""
    allrows = list(range(len(rows)))
    by_cell = {c: [n for n, it in enumerate(rows) if it["cell"] == c] for c in src}
    doses = [j for j, g in enumerate(S["dose_gaps"]) if (g <= S["gen_max_gap"] if gen else True)]
    conds = [{"name": "baseline", "kind": "none", "idx": -1, "li": -1, "j": -1, "sg": 0, "items": allrows}]
    for li in range(NL):
        for j in doses:
            for k in range(N_RAND_GEN if gen else N_RAND2):
                for sg in ((1,) if gen else (1, -1)):
                    conds.append({"name": f"rand{k}", "kind": "rand", "idx": k, "li": li, "j": j, "sg": sg, "items": allrows})
    for c in src:
        for li in range(NL):
            for kd in KINDS2:
                for j in doses:
                    for sg in ((1, -1) if (kd == "dm" and not gen) else (1,)):
                        conds.append({"name": c, "kind": kd, "idx": S["cells"].index(c), "li": li, "j": j, "sg": sg, "items": by_cell[c]})
    return conds


def shard_assign(conds, shard, nshards):
    """Balanced by item count: largest conditions first onto the lightest shard."""
    load, mine = [0] * nshards, [[] for _ in range(nshards)]
    for k in sorted(range(len(conds)), key=lambda k: (-len(conds[k]["items"]), k)):
        s = min(range(nshards), key=lambda s: (load[s], s))
        load[s] += len(conds[k]["items"]); mine[s].append(k)
    return sorted(mine[shard])


def _item_tokens(S, rows, tok):
    from .s5_causal import items_index
    from .stimx import load_replies, prompt_and_reply, tokenize
    cache, toks, prompts, questions = {}, [], [], []
    for it in rows:
        cid = it["cell"].split("|")[0]
        if cid not in cache:
            cache[cid] = (items_index(cid), load_replies(cid))
        its, rep = cache[cid]
        item = its[it["item_id"]]
        prompt, reply = prompt_and_reply(tok, item, rep)
        toks.append(tokenize(tok, prompt, reply)); prompts.append(prompt)
        msgs = item["messages"]
        questions.append(msgs[-2]["content"] if item["readout"] == "response" else msgs[-1]["content"])
    return toks, prompts, questions


def _vector(S, A, cd):
    if cd["kind"] == "none":
        return None
    base = A["rand"][cd["idx"]] if cd["kind"] == "rand" else A[cd["kind"]][cd["li"], cd["idx"]]
    return base * cd["sg"] * S["dose_raw"][cd["li"]][cd["j"]]


def cmd_stage2_run(cells_sel, shard, nshards):
    """Teacher-forced readouts: for every condition of the shard, the mean over the first 32 reply tokens of the
    hidden state at the four depths, for the condition's items. GPU."""
    import torch
    from contextlib import nullcontext
    from . import model as cm_model
    from .stimx import forward_hidden
    S, A = load_stage2()
    src = select_cells(cells_sel)
    rows = [it for it in S["items"] if it["cell"] in src]
    conds = stage2_conditions(S, src, rows)
    mine = shard_assign(conds, shard, nshards)
    m, tok = cm_model.load(cfg.MODEL)
    toks, _, _ = _item_tokens(S, rows, tok)
    vecs = []
    for n_done, k in enumerate(mine):
        cd = conds[k]; vec = _vector(S, A, cd)
        ids = [toks[i] for i in cd["items"]]
        R = np.zeros((len(ids), NL, cfg.D), np.float32)
        ctx = cm_model.steer(m, cfg.DEPTH_LAYERS[cd["li"]], torch.tensor(vec, device="cuda", dtype=torch.float32)) if vec is not None else nullcontext()
        with torch.no_grad(), ctx:
            for n, (H, (x, p)) in enumerate(zip(forward_hidden(m, [x for x, _ in ids], batch=FWD_BATCH, token_budget=FWD_TOKENS), ids)):
                R[n] = H[LIDX, p:p + GEN_TOKENS].float().mean(1).cpu().numpy()
        vecs.append(R)
        print(f"[diag] shard {shard}: {n_done + 1}/{len(mine)} {cd['name']} {cd['kind']} depth {cd['li']} dose {cd['j']} {cd['sg']:+d} ({len(ids)} items)", flush=True)
    np.savez(GEN2 / f"run_{cells_sel}_shard{shard}.npz", cond_index=np.array(mine), sizes=np.array([len(conds[k]["items"]) for k in mine]),
             vecs=np.concatenate(vecs))


def _strip(gen_ids, stops):
    out = []
    for t in gen_ids:
        if t in stops:
            break
        out.append(int(t))
    return out


def cmd_stage2_gen(cells_sel):
    """Free generation under steering (greedy, 32 tokens, the prompt without any prefilled reply), then the same
    steered forward over prompt + generation for the readout at the four depths. GPU, one job per model."""
    import torch
    from contextlib import nullcontext
    from . import model as cm_model
    from .stimx import forward_hidden
    from .stimuli.schema import write_jsonl
    S, A = load_stage2()
    src = select_cells(cells_sel)
    rows = [it for it in S["items"] if it["cell"] in src]
    conds = stage2_conditions(S, src, rows, gen=True)
    m, tok = cm_model.load(cfg.MODEL)
    tok.padding_side = "left"
    toks, prompts, questions = _item_tokens(S, rows, tok)
    stops = {tok.eos_token_id, tok.pad_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<end_of_turn>", "<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    stops.discard(None)
    out_rows, vecs, sizes = [], [], []
    for k, cd in enumerate(conds):
        vec = _vector(S, A, cd)
        ctx = cm_model.steer(m, cfg.DEPTH_LAYERS[cd["li"]], torch.tensor(vec, device="cuda", dtype=torch.float32)) if vec is not None else nullcontext()
        uniq = sorted({prompts[i] for i in cd["items"]})
        gen_of = {}
        with torch.no_grad(), ctx:
            for s in range(0, len(uniq), 24):
                chunk = uniq[s:s + 24]
                enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                out = m.generate(**enc, max_new_tokens=GEN_TOKENS, do_sample=False)
                for p_text, g in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
                    gen_of[p_text] = _strip(g, stops)
            ids = [(toks[i][0][:toks[i][1]] + gen_of[prompts[i]], toks[i][1]) for i in cd["items"]]
            R = np.full((len(ids), NL, cfg.D), np.nan, np.float32)
            for n, (H, (x, p)) in enumerate(zip(forward_hidden(m, [x for x, _ in ids], batch=FWD_BATCH, token_budget=FWD_TOKENS), ids)):
                if len(x) > p:
                    R[n] = H[LIDX, p:p + GEN_TOKENS].float().mean(1).cpu().numpy()
        for n, i in enumerate(cd["items"]):
            g = gen_of[prompts[i]]
            out_rows.append({"cond": k, "name": cd["name"], "kind": cd["kind"], "li": cd["li"], "j": cd["j"], "cell": rows[i]["cell"], "item_id": rows[i]["item_id"],
                             "polarity": rows[i]["polarity"], "question": questions[i], "text": tok.decode(g, skip_special_tokens=True), "n_tokens": len(g)})
        vecs.append(R); sizes.append(len(ids))
        print(f"[diag] gen {k + 1}/{len(conds)} {cd['name']} {cd['kind']} depth {cd['li']} dose {cd['j']} ({len(uniq)} prompts)", flush=True)
    write_jsonl(GEN2 / f"gen_{cells_sel}.jsonl", out_rows)
    np.savez(GEN2 / f"gen_{cells_sel}.npz", cond_index=np.arange(len(conds)), sizes=np.array(sizes), vecs=np.concatenate(vecs))


def cmd_stage2_judge(tags, cells_sel):
    """Coherence of the free generations (gemma-4-31B-it, the D-039 judge), both subjects in one job."""
    from .pv.judge import COHERENCE, DigitTreeJudge, fill
    from .stimuli.schema import read_jsonl, write_jsonl
    judge = DigitTreeJudge("google/gemma-4-31B-it", 128)
    for tag in tags:
        tdir = cfg.SPECS[tag]["tag"]
        gdir = (ROOT_OUTPUTS / tdir if tdir else ROOT_OUTPUTS) / "diag_prop" / "stage2"
        rdir = (ROOT_RESULTS / tdir if tdir else ROOT_RESULTS) / "diag_prop" / "stage2"
        rows = read_jsonl(gdir / f"gen_{cells_sel}.jsonl")
        scores = judge.score([fill(COHERENCE, r["question"], r["text"]) for r in rows])
        for r, (sc, _, _) in zip(rows, scores):
            r["coherence"] = sc
        write_jsonl(gdir / f"gen_{cells_sel}_scored.jsonl", rows)
        table = {}
        for r in rows:
            if r["coherence"] is not None:
                table.setdefault((r["kind"], r["li"], r["j"]), []).append(r["coherence"])
        summ = {f"{k[0]}|{k[1]}|{k[2]}": {"mean": float(np.mean(v)), "n": len(v)} for k, v in table.items()}
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"gen_{cells_sel}_coherence.json").write_text(json.dumps(summ, indent=1))
        print(f"[diag] {tag}: coherence by kind|depth|dose " + " ".join(f"{k} {v['mean']:.0f}" for k, v in sorted(summ.items())), flush=True)


def _load_vecs(pattern, conds):
    P = {}
    for f in sorted(GEN2.glob(pattern)):
        z = np.load(f)
        V, idx, sizes = z["vecs"], z["cond_index"], z["sizes"]     # read each array once: an NpzFile re-reads on every access
        off = 0
        for k, n in zip(idx, sizes):
            P[int(k)] = V[off:off + n]; off += n
    missing = [k for k in range(len(conds)) if k not in P]
    if missing:
        raise SystemExit(f"{pattern}: {len(missing)} conditions missing (shards not all done?): {missing[:5]}")
    return P


def cmd_stage2_report(cells_sel):
    """Self effect per source cell, injection depth, direction and dose at every readout depth: raw shift and the
    shift beyond the pass-through skip term in gap units, the random band, z, and the survival of the injected vector
    along its own axis (1 = identity, pass-through = pure residual scaling). Free generations added where present."""
    S, A = load_stage2()
    src = select_cells(cells_sel)
    rows = [it for it in S["items"] if it["cell"] in src]
    conds = stage2_conditions(S, src, rows)
    P = _load_vecs(f"run_{cells_sel}_shard*.npz", conds)
    base = P[0]
    pos_in = {k: {i: n for n, i in enumerate(cd["items"])} for k, cd in enumerate(conds)}
    pt = S["pass_through"]
    out = {"model": cfg.MODEL, "cells": src, "depth_layers": S["depth_layers"], "dose_gaps": S["dose_gaps"], "per_cell": {}, "summary": {}}
    rand_k = {(cd["li"], cd["j"]): [k for k, c2 in enumerate(conds) if c2["kind"] == "rand" and c2["li"] == cd["li"] and c2["j"] == cd["j"]]
              for cd in conds if cd["kind"] == "rand"}
    for c in src:
        kc = S["cells"].index(c)
        I = [n for n, it in enumerate(rows) if it["cell"] == c]
        pos = np.array([rows[n]["polarity"] == "pos" for n in I])
        rec = {"gap": [], "gap_std": [], "conds": {}}
        # two readouts per depth: the raw projection on the cell's diff of means (S5a's) and the standardized one,
        # (x / sd) . unit(v / sd) (S3's sdm; the massive dimensions weigh 1 / sd there)
        ustd = {h: unit_vec(A["dm"][h, kc] / A["sd"][h]) for h in range(NL)}
        pstd = lambda x, h: (x / A["sd"][h]) @ ustd[h]
        for h in range(NL):
            u = A["dm"][h, kc]
            pb = base[I, h] @ u
            g = float(pb[pos].mean() - pb[~pos].mean())
            rec["gap"].append(g if g > 0 else None)
            ps = pstd(base[I, h], h)
            gs = float(ps[pos].mean() - ps[~pos].mean())
            rec["gap_std"].append(gs if gs > 0 else None)
        for k, cd in enumerate(conds):
            if cd["kind"] not in KINDS2 or cd["name"] != c:
                continue
            vinj = A[cd["kind"]][cd["li"], kc]; dose = S["dose_raw"][cd["li"]][cd["j"]] * cd["sg"]
            per_h = []
            for h in range(NL):
                u = A["dm"][h, kc]; g = rec["gap"][h]; gs = rec["gap_std"][h]
                sh = P[k][:, h] - base[I, h]                        # P[k] rows are the cell's items in I order
                raw = float((sh @ u).mean() / g) if g else None
                raw_s = float(pstd(sh, h).mean() / gs) if gs else None
                surv = float((sh @ vinj).mean() / dose)
                if h <= cd["li"] or g is None or gs is None:        # at the injection depth the recorder sees the pre-hook output
                    per_h.append({"raw": raw, "raw_std": raw_s, "survival": surv, "at_injection": h == cd["li"]}); continue
                skip_n = float(dose * (vinj @ u) / g); skip_p = float(pt[cd["li"]][h] * dose * (vinj @ u) / g)
                skip_ps = float(pt[cd["li"]][h] * dose * pstd(vinj, h) / gs)
                rows_r = [[pos_in[kr][i] for i in I] for kr in rand_k[(cd["li"], cd["j"])]]
                rb = np.array([((P[kr][rr, h] - base[I, h]) @ u).mean() / g for kr, rr in zip(rand_k[(cd["li"], cd["j"])], rows_r)])
                rbs = np.array([pstd(P[kr][rr, h] - base[I, h], h).mean() / gs for kr, rr in zip(rand_k[(cd["li"], cd["j"])], rows_r)])
                mu, sd = float(rb.mean()), float(rb.std() + 1e-9); mus, sds = float(rbs.mean()), float(rbs.std() + 1e-9)
                per_h.append({"raw": raw, "skip_naive": skip_n, "skip_pass": skip_p, "E_pass": raw - skip_p, "band_mean": mu, "band_sd": sd,
                              "z": (raw - skip_p - mu) / sd, "survival": surv, "pass_through": pt[cd["li"]][h],
                              "raw_std": raw_s, "skip_std_pass": skip_ps, "E_std_pass": raw_s - skip_ps, "band_std_mean": mus, "band_std_sd": sds,
                              "z_std": (raw_s - skip_ps - mus) / sds})
            rec["conds"][f"{cd['kind']}|{cd['li']}|{cd['j']}|{cd['sg']:+d}"] = per_h
        out["per_cell"][c] = rec
    # medians over cells per (direction, depth, dose, +) and readout depth
    for kd in KINDS2:
        for li in range(NL):
            for j in range(len(S["dose_gaps"])):
                key = f"{kd}|{li}|{j}|+1"
                summ = []
                for h in range(NL):
                    vals = [out["per_cell"][c]["conds"][key][h] for c in src if key in out["per_cell"][c]["conds"]]
                    summ.append({q: float(np.nanmedian([v[q] for v in vals if v.get(q) is not None])) if any(v.get(q) is not None for v in vals) else None
                                 for q in ("raw", "skip_naive", "skip_pass", "E_pass", "band_mean", "band_sd", "z", "survival",
                                           "raw_std", "skip_std_pass", "E_std_pass", "band_std_mean", "band_std_sd", "z_std")})
                out["summary"][key] = summ
    gen_file = GEN2 / f"gen_{cells_sel}.npz"
    if gen_file.exists():
        gconds = stage2_conditions(S, src, rows, gen=True)
        G = _load_vecs(f"gen_{cells_sel}.npz", gconds)
        gb = G[0]
        coh_path = OUT2 / f"gen_{cells_sel}_coherence.json"
        coh = json.loads(coh_path.read_text()) if coh_path.exists() else {}
        scored = GEN2 / f"gen_{cells_sel}_scored.jsonl"
        if scored.exists():      # the judge's mean is not a scale on 32-token truncations (baseline medians near 1); keep the shares
            from .stimuli.schema import read_jsonl
            tab = {}
            for r in read_jsonl(scored):
                if r["coherence"] is not None:
                    tab.setdefault(f"{r['kind']}|{r['li']}|{r['j']}", []).append(float(r["coherence"]))
            coh = {k: {"mean": float(np.mean(v)), "n": len(v), "share_ge75": float(np.mean(np.array(v) >= 75)),
                       "share_zero": float(np.mean(np.array(v) <= 0))} for k, v in tab.items()}
        gsum = {}
        for k, cd in enumerate(gconds):
            if cd["kind"] not in KINDS2:
                continue
            c = cd["name"]; kc = S["cells"].index(c); I = [n for n, it in enumerate(rows) if it["cell"] == c]
            key = f"{cd['kind']}|{cd['li']}|{cd['j']}"
            per_h = []
            for h in range(NL):
                g = out["per_cell"][c]["gap"][h]; u = A["dm"][h, kc]
                sh = np.nanmean(G[k][:, h] - gb[I, h], 0)
                per_h.append(float(sh @ u / g) if (g and np.isfinite(sh).all()) else None)
            gsum.setdefault(key, []).append(per_h)
        out["gen"] = {key: {"E_raw_median": [float(np.nanmedian([v[h] for v in vals if v[h] is not None])) if any(v[h] is not None for v in vals) else None for h in range(NL)],
                            "coherence": coh.get(key)} for key, vals in gsum.items()}
        out["gen_random_coherence"] = {k: v for k, v in coh.items() if k.startswith("rand")}
        out["gen_baseline_coherence"] = coh.get("none|-1|-1")
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / f"report_{cells_sel}.json").write_text(json.dumps(out, default=float))
    for key, summ in out["summary"].items():
        kd, li, j, _ = key.split("|")
        print(f"[diag] {kd} at layer {S['depth_layers'][int(li)]}, dose {S['dose_gaps'][int(j)]} gaps: "
              + "; ".join(f"L{S['depth_layers'][h]} raw {s['raw']:+.3f} E {s['E_pass']:+.3f} (band {s['band_mean']:+.3f}+-{s['band_sd']:.3f}, z {s['z']:+.1f}) "
                          f"std E {s['E_std_pass']:+.3f} (band {s['band_std_mean']:+.3f}+-{s['band_std_sd']:.3f}, z {s['z_std']:+.1f}) surv {s['survival']:.2f}"
                          for h, s in enumerate(summ) if s.get("E_pass") is not None), flush=True)


def cmd_stage2_table(tags, cells_sel):
    """results/diag_prop/stage2_<cells>.md: the gate table across subjects."""
    R = {}
    for tag in tags:
        tdir = cfg.SPECS[tag]["tag"]
        p = (ROOT_RESULTS / tdir if tdir else ROOT_RESULTS) / "diag_prop" / "stage2" / f"report_{cells_sel}.json"
        if p.exists():
            R[tag] = json.loads(p.read_text())
    tags = [t for t in tags if t in R]
    od = ROOT_RESULTS / "diag_prop"; od.mkdir(parents=True, exist_ok=True)
    L = ["# D-060 stage 2: self effect by injection depth x direction x dose", "",
         "Cells: " + ", ".join(R[tags[0]]["cells"]) + ". Medians over cells; gap units of the cell's own direction at the readout depth. "
         "raw = shift of the raw projection on the cell's diff of means; E = raw minus the pass-through skip term; band = mean +- sd "
         "of the 16 random directions at the same depth and dose; z = (E - band mean) / band sd; std = the same with the standardized "
         "readout (x / sd on unit(v / sd), S3's sdm, where the massive dimensions weigh 1 / sd); surv = shift along the injected axis "
         "over the dose (1 = identity residual). Readouts at the injection depth are omitted: the hidden-state recorder reads the "
         "layer output before the hook adds the vector (the next layer reads it after).", ""]
    for h_ord in (1, 2, 3):
        L += [f"## Readout at depth {h_ord + 1} (" + ", ".join(f"{t} layer {R[t]['depth_layers'][h_ord]}" for t in tags) + ")", "",
              "| injection | direction | dose (gaps) | " + " | ".join(f"{t}: raw, E (band; z); std E (band; z); surv" for t in tags) + " |", "|---|---|---|" + "---|" * len(tags)]
        for li in range(NL):
            if li >= h_ord:
                continue
            for kd in KINDS2:
                for j, g in enumerate(DOSE_GAPS):
                    cells = []
                    for t in tags:
                        s = R[t]["summary"].get(f"{kd}|{li}|{j}|+1", [None] * NL)[h_ord]
                        cells.append("n/a" if not s or s.get("E_pass") is None else
                                     f"{s['raw']:+.3f}, {s['E_pass']:+.3f} ({s['band_mean']:+.3f}+-{s['band_sd']:.3f}; z {s['z']:+.1f}); "
                                     f"std {s['E_std_pass']:+.3f} ({s['band_std_mean']:+.3f}+-{s['band_std_sd']:.3f}; z {s['z_std']:+.1f}); {s['survival']:.2f}")
                    L.append(f"| depth {li + 1} (" + ", ".join(f"L{R[t]['depth_layers'][li]}" for t in tags) + f") | {kd} | {g} | " + " | ".join(cells) + " |")
        L.append("")
    if any("gen" in R[t] for t in tags):
        L += ["## Free generation (steered reply32 of the model's own reply): raw shift in gap units, judge coherence", "",
              "| injection | direction | dose | " + " | ".join(f"{t}: E at depths 3, 4 (coherence shares)" for t in tags) + " |", "|---|---|---|" + "---|" * len(tags)]
        for li in range(NL):
            for kd in KINDS2:
                for j, g in enumerate(DOSE_GAPS):
                    cells = []
                    for t in tags:
                        gq = R[t].get("gen", {}).get(f"{kd}|{li}|{j}")
                        if not gq:
                            cells.append("n/a"); continue
                        e = gq["E_raw_median"]; co = gq["coherence"]
                        cells.append(", ".join("n/a" if e[h] is None else f"{e[h]:+.3f}" for h in (2, 3))
                                     + (f" (ge75 {co.get('share_ge75', float('nan')):.2f}, zero {co.get('share_zero', float('nan')):.2f})" if co else ""))
                    if any(c != "n/a" for c in cells):
                        L.append(f"| depth {li + 1} | {kd} | {g} | " + " | ".join(cells) + " |")
        fmtc = lambda v: f"ge75 {v.get('share_ge75', float('nan')):.2f}/zero {v.get('share_zero', float('nan')):.2f}"
        L += ["", "Coherence shown as the share of replies the judge rates at or above 75 and the share it rates 0 (its mean is not a scale on 32-token replies).",
              "Baseline: " + ", ".join(f"{t} {fmtc(R[t]['gen_baseline_coherence'])}" for t in tags if R[t].get("gen_baseline_coherence")),
              "Random directions by depth|dose: " + "; ".join(f"{t}: " + " ".join(f"{k.split('|',1)[1]} {fmtc(v)}" for k, v in sorted(R[t].get("gen_random_coherence", {}).items())) for t in tags)]
    (od / f"stage2_{cells_sel}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- stage 3 (D-060 stage-3 ruling A): part-3 levers in primal form

def cmd_stage3_levers():
    """results/<tag>/coherence/levers_part3.npz: every standardized lever of part 1 (levers.npz) with its primal raw form
    unit(sd * u) under `<key>|primal`, the form to inject (for the diff-of-means levers it is the raw diff of means);
    the `|resid` covectors are kept for reading. levers_part3.json records the rule, the dose cap and the cosines of the
    cell primals with the S5a grid's injected vectors."""
    lev = np.load(RESULTS / "coherence" / "levers.npz")
    sd = lev["sd"]
    arrs = {k: lev[k] for k in lev.files}
    for k in lev.files:
        if k != "sd" and not k.endswith("|resid"):
            arrs[f"{k}|primal"] = unit_vec(sd * lev[k])
    np.savez(RESULTS / "coherence" / "levers_part3.npz", **arrs)
    S2 = json.loads((OUT2 / "setup.json").read_text()) if (OUT2 / "setup.json").exists() else None
    A = _setup_npz(); Sj = json.loads((_setup_dir()[0] / "setup.json").read_text()) if A is not None else None
    cos = {}
    if A is not None:
        for k, u in enumerate(Sj["sources"]):
            key = f"cell|{u}|primal"
            if key in arrs:
                cos[u] = float(arrs[key] @ A["dense"][k])
    scal = layer_scalars()
    info = {"model": cfg.MODEL, "steering_layer": cfg.L_STEER, "hidden_index": cfg.L_STEER + 1,
            "rule": "inject <key>|primal = unit(sd * u) scaled to the dose; read with <key>|resid (standardized projection) or the raw diff of means",
            "dose_cap_class_gaps": 0.5 if cfg.IS_GEMMA4 else 1.0,
            "median_class_gap_steering_depth": S2["median_gap"][1] if S2 else None,
            "dose_cap_raw_units": (0.5 if cfg.IS_GEMMA4 else 1.0) * S2["median_gap"][1] if S2 else None,
            "pass_through": {str(L): pass_through(scal, cfg.L_STEER, L) for L in cfg.L_OUT},
            "keys": sorted(arrs), "cos_cell_primal_vs_s5a_injected": cos,
            "cos_dm_all_primal_vs_probe_all_primal": float(arrs["dm_all|primal"] @ arrs["probe_all|primal"])}
    (RESULTS / "coherence" / "levers_part3.json").write_text(json.dumps(info, indent=1, default=float))
    print(f"[diag] levers_part3.npz: {len(arrs)} arrays; cell primal vs S5a injected cos median {np.median(list(cos.values())) if cos else float('nan'):.4f}; "
          f"dose cap {info['dose_cap_raw_units']} raw units; pass-through {info['pass_through']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["anatomy", "grid", "report", "stage2-setup", "stage2-run", "stage2-gen", "stage2-judge", "stage2-report", "stage2-table", "stage3-levers"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tags", default="gemma-4-31b-it,qwen3.5-27b")
    ap.add_argument("--cells", default="d", choices=["d", "rest", "all"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    if args.stage == "anatomy":
        cmd_anatomy(args.workers)
    elif args.stage == "grid":
        cmd_grid()
    elif args.stage == "report":
        cmd_report(args.tags.split(","))
    elif args.stage == "stage2-setup":
        cmd_stage2_setup(args.workers)
    elif args.stage == "stage2-run":
        cmd_stage2_run(args.cells, args.shard, args.nshards)
    elif args.stage == "stage2-gen":
        cmd_stage2_gen(args.cells)
    elif args.stage == "stage2-judge":
        cmd_stage2_judge(args.tags.split(","), args.cells)
    elif args.stage == "stage2-report":
        cmd_stage2_report(args.cells)
    elif args.stage == "stage3-levers":
        cmd_stage3_levers()
    else:
        cmd_stage2_table(args.tags.split(","), args.cells)


if __name__ == "__main__":
    main()
