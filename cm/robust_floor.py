"""D-064 amendment 1 ruling (d): the floored-standardization robustness recompute, readout side. The floor itself is
cm.coherence.floor_sd (sd_floored = max(sd, quantile(sd, q)); q = None is off and is the default everywhere), applied by
cm.coherence rep and cm.battery_onpolicy directions / score-strat / directions-all / score-sweep when they are given
--sd-floor-q; each such run writes under .../robust_floor/<none|q0.1|q0.25|q0.5>/ and never over the baseline files.
This module reads those outputs:

    CM_MODEL=<model> python -m cm.robust_floor check      robust_floor/none/ against the baseline files, to 1e-6  -> coherence/robust_floor/identity_check.json
    CM_MODEL=<model> python -m cm.robust_floor lever-cos  floored directions against the unfloored ones            -> coherence/robust_floor/lever_cosines.json
    CM_MODEL=<model> python -m cm.robust_floor table      the subject's comparison over the four settings           -> coherence/robust_floor/table.{md,json}
    python -m cm.robust_floor figure [--tags gemma-4-31b-it,qwen3.5-27b]        -> results/figures/honesty/robust_floor.{png,pdf,json}
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import cfg
from .battery_onpolicy import GROUPS, LAYERS as OP_LAYERS, RES as OP_RES
from .coherence import ALL, BLOCKS, COLOR, FRAME_NAME, HON, MARK, OUT, cell_info, qtag, robust_dir, style, unit_vec
from .paths import RESULTS as ROOT_RESULTS

QS = [None, 0.1, 0.25, 0.5]
TOL = 1e-6
RDIR = OUT / "robust_floor"
GROUP_NAMES = list(GROUPS)
HON_GROUPS = GROUP_NAMES[:6]                      # act, self-describe, own-reply, frame levers, shared axes, all_frames
N_CELLS = {g: sum(1 for c in HON if GROUPS[g](f"cell|{c}")) for g in GROUP_NAMES[:3]}      # 6, 6, 10 cells
SHORT = {"act cells (c)": "act", "self-describe cells (d)": "self", "own-reply cells (a, b)": "own", "frame levers": "frames",
         "shared axes (dm_all, probe_all)": "shared", "all_frames": "all_frames", "K1": "K1", "K2": "K2", "random (16)": "random"}
HEADS = ["dm_all", "probe_all", "frame_a", "frame_b", "frame_c", "frame_d", "all_frames", "K1", "K2"]


def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None


def sep(n):
    return "|" + " --- |" * n


def f2(v):
    return "-" if v is None else f"{v:.2f}"


# ---------------------------------------------------------------- check: q = None reproduces the baseline files

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def walk(a, b, path, skip, acc):
    if isinstance(a, dict) and isinstance(b, dict):
        for k, v in a.items():
            if k not in b:
                acc["missing"].append(f"{path}/{k}")
            else:
                walk(v, b[k], f"{path}/{k}", skip, acc)
        acc["extra"] += [f"{path}/{k}" for k in b if k not in a and k not in skip]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            acc["missing"].append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", skip, acc)
    elif _num(a) and _num(b):
        d = 0.0 if (np.isnan(a) and np.isnan(b)) else abs(float(a) - float(b))
        acc["n"] += 1
        acc["max"] = max(acc["max"], d)
        if d > TOL:
            acc["bad"].append([path, a, b])
    elif a != b:
        acc["other"].append([path, str(a)[:60], str(b)[:60]])


def compare_json(base, rob, skip=()):
    if not base.exists() or not rob.exists():
        return {"status": "missing", "base": str(base), "robust": str(rob), "base_exists": base.exists(), "robust_exists": rob.exists()}
    acc = {"n": 0, "max": 0.0, "bad": [], "missing": [], "extra": [], "other": []}
    walk(json.loads(base.read_text()), json.loads(rob.read_text()), "", set(skip), acc)
    ok = not acc["bad"] and not acc["missing"] and not acc["other"]
    return {"status": "pass" if ok else "fail", "base": str(base), "robust": str(rob), "numeric_leaves": acc["n"], "max_abs_diff": acc["max"],
            "n_over_tol": len(acc["bad"]), "first_over_tol": acc["bad"][:10], "missing": acc["missing"][:10],
            "extra_in_robust": acc["extra"][:10], "other": acc["other"][:10]}


def compare_npz(base, rob):
    if not base.exists() or not rob.exists():
        return {"status": "missing", "base": str(base), "robust": str(rob), "base_exists": base.exists(), "robust_exists": rob.exists()}
    A, B = np.load(base), np.load(rob)
    ka, kb = set(A.files), set(B.files)
    per, differs = {}, []
    for k in sorted(ka & kb):
        a, b = A[k], B[k]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu" and a.shape == b.shape:
            per[k] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        elif not (a.shape == b.shape and bool((a == b).all())):
            differs.append(k)
    mx = max(per.values(), default=0.0)
    ok = ka == kb and mx <= TOL and not differs
    return {"status": "pass" if ok else "fail", "base": str(base), "robust": str(rob), "arrays": len(ka & kb), "max_abs_diff": mx,
            "keys_only_in_base": sorted(ka - kb), "keys_only_in_robust": sorted(kb - ka), "non_numeric_differ": differs,
            "worst_keys": sorted(((v, k) for k, v in per.items()), reverse=True)[:5]}


def cmd_check():
    none = robust_dir(OUT, None)
    opn = OP_RES / "robust_floor" / "none"
    out = {"model": cfg.MODEL, "tol": TOL, "files": {}}
    out["files"]["representational.json"] = compare_json(OUT / "representational.json", none / "representational.json", skip=("sd_floor_q", "sd"))
    out["files"]["levers.npz"] = compare_npz(OUT / "levers.npz", none / "levers.npz")
    out["files"]["directions.npz"] = compare_npz(OP_RES / "directions.npz", opn / "directions.npz")
    out["files"]["directions.json"] = compare_json(OP_RES / "directions.json", opn / "directions.json", skip=("sd_floor_q",))
    out["files"]["auroc_strat.json"] = compare_json(OP_RES / "baseline" / "auroc_strat.json", opn / "baseline" / "auroc_strat.json", skip=("sd_floor_q",))
    for name, b, r in (("directions_all.npz", OP_RES / "directions_all.npz", opn / "directions_all.npz"),
                       ("sweep.json", OP_RES / "baseline" / "sweep.json", opn / "baseline" / "sweep.json")):
        if r.exists():
            out["files"][name] = compare_npz(b, r) if name.endswith("npz") else compare_json(b, r, skip=("sd_floor_q",))
    required = ["representational.json", "levers.npz", "directions.npz", "auroc_strat.json"]
    out["pass"] = all(out["files"][k]["status"] == "pass" for k in required)
    RDIR.mkdir(parents=True, exist_ok=True)
    (RDIR / "identity_check.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out["files"].items():
        print(f"[robust_floor] {k:24} {v['status']:7} max |diff| {v.get('max_abs_diff')}"
              + (f" over {v['numeric_leaves']} numbers" if "numeric_leaves" in v else f" over {v['arrays']} arrays" if "arrays" in v else ""), flush=True)
    print("[robust_floor] identity check", "PASS" if out["pass"] else "FAIL", "->", RDIR / "identity_check.json")
    if not out["pass"]:
        sys.exit(1)


# ---------------------------------------------------------------- lever-cos: what the floor turns

def cmd_lever_cos():
    """Per floor setting and item-4 depth, against the unfloored directions.npz: cos u (standardized directions), cos w
    (readout covectors w = unit(u / sd)), cos p (primal levers p = unit(sd u)), and lever-readout agreement cos(w, p)
    before and after, for the shared axes, frame levers and controls, summarized over the 22 cells and the 16 random
    directions; plus the squared weight of u in the lowest 5 percent of (unfloored) sd dimensions."""
    D0 = np.load(OP_RES / "directions.npz")
    out = {"model": cfg.MODEL, "layers": OP_LAYERS, "settings": {}}
    for q in QS[1:]:
        p = OP_RES / "robust_floor" / qtag(q) / "directions.npz"
        if not p.exists():
            print("[robust_floor] missing", p)
            continue
        Dq = np.load(p)
        out["settings"][qtag(q)] = {}
        for L in OP_LAYERS:
            sd0, sdq = D0[f"sd|L{L}"].astype(np.float64), Dq[f"sd|L{L}"].astype(np.float64)
            floor = float(np.quantile(sd0, q))
            low5 = sd0 <= np.quantile(sd0, 0.05)

            def one(nm):
                u0, uq = D0[f"{nm}|L{L}"].astype(np.float64), Dq[f"{nm}|L{L}"].astype(np.float64)
                w0, wq = unit_vec(u0 / sd0), unit_vec(uq / sdq)
                p0, pq = unit_vec(sd0 * u0), unit_vec(sdq * uq)
                return {"cos_u": float(u0 @ uq), "cos_w": float(w0 @ wq), "cos_p": float(p0 @ pq),
                        "cos_wp_before": float(w0 @ p0), "cos_wp_after": float(wq @ pq),
                        "low5_share_before": float((u0[low5] ** 2).sum()), "low5_share_after": float((uq[low5] ** 2).sum())}

            cells = {c: one(f"cell|{c}") for c in HON}
            rnd = [one(f"random_{k}") for k in range(16)]
            keys = list(next(iter(cells.values())))
            ent = {"floor": floor, "n_raised": int((sd0 < floor).sum()), "n_dims": int(sd0.size), "heads": {nm: one(nm) for nm in HEADS},
                   "cells_summary": {k: {"mean": float(np.mean([v[k] for v in cells.values()])), "min": float(np.min([v[k] for v in cells.values()]))} for k in keys},
                   "random_summary": {k: float(np.mean([v[k] for v in rnd])) for k in keys}, "cells": cells}
            out["settings"][qtag(q)][str(L)] = ent
            h = ent["heads"]
            print(f"[robust_floor] {qtag(q)} L{L}: floor {floor:.4f}, raised {ent['n_raised']}/{sd0.size}; dm_all cos u {h['dm_all']['cos_u']:.2f} "
                  f"w {h['dm_all']['cos_w']:.2f} p {h['dm_all']['cos_p']:.3f} cos(w,p) {h['dm_all']['cos_wp_before']:.2f} -> {h['dm_all']['cos_wp_after']:.2f}; "
                  f"probe_all p {h['probe_all']['cos_p']:.2f} cos(w,p) {h['probe_all']['cos_wp_before']:.2f} -> {h['probe_all']['cos_wp_after']:.2f}; "
                  f"cells mean cos u {ent['cells_summary']['cos_u']['mean']:.2f}", flush=True)
    RDIR.mkdir(parents=True, exist_ok=True)
    (RDIR / "lever_cosines.json").write_text(json.dumps(out, indent=1, default=float))
    print("[robust_floor] ->", RDIR / "lever_cosines.json")


# ---------------------------------------------------------------- table: the subject's comparison over the settings

def part1(reps):
    """reps: {setting: representational json or None}. Means, controls' mean |scaled|, Spearman against the unfloored
    per-cell scores, frame blocks, sd stats, and the per-cell numbers at the steering depth."""
    from scipy.stats import spearmanr
    base = reps["none"]
    depths = [str(L) for L in base["depths"]]
    out = {"depths": depths, "steering": depths[1], "mean": {}, "sd": {}, "cells": {}}
    for L in depths:
        for m in ("dm", "probe"):
            out["mean"][f"{L}|{m}"] = {}
            bh = [base["by_depth"][L]["cells"][c][f"scaled_{m}"] for c in HON]
            for tag, R in reps.items():
                if R is None:
                    continue
                Cc = R["by_depth"][L]["cells"]
                hon = [Cc[c][f"scaled_{m}"] for c in HON]
                ctl = [Cc[c][f"scaled_{m}"] for c in ALL if c not in HON]
                ok = [i for i in range(len(HON)) if hon[i] is not None and bh[i] is not None]
                out["mean"][f"{L}|{m}"][tag] = {
                    "mean_scaled": R["summary"][L][m]["mean_scaled"], "n": R["summary"][L][m]["n"],
                    "controls_mean_abs_scaled": float(np.mean([abs(v) for v in ctl if v is not None])) if any(v is not None for v in ctl) else None,
                    "controls_mean_scaled": R["summary"][L][m]["mean_scaled_controls"],
                    "spearman_vs_none": float(spearmanr([hon[i] for i in ok], [bh[i] for i in ok]).correlation) if len(ok) > 3 else None,
                    "by_block": R["summary"][L][m]["by_block"], "mean_loco_auroc": R["summary"][L][m]["mean_loco_auroc"],
                    "mean_loco_auroc_controls": R["summary"][L][m]["mean_loco_auroc_controls"]}
        out["sd"][L] = {tag: R["by_depth"][L]["sd"] for tag, R in reps.items() if R and "sd" in R["by_depth"][L]}
    Ls = depths[1]
    for c in ALL:
        out["cells"][c] = {tag: {m: R["by_depth"][Ls]["cells"][c][f"scaled_{m}"] for m in ("dm", "probe")} for tag, R in reps.items() if R}
    return out


def part2(strats):
    """strats: {setting: auroc_strat json or None}. Per setting the mean over the task readouts that have a pooled block of
    the group means per depth x position and over all cells, the all-cells mean (the 22 cell directions, 6 + 6 + 10),
    and how many readout x (depth, position) cells have a group mean above the random 95th percentile."""
    out = {}
    for tag, J in strats.items():
        if J is None:
            continue
        rows = [(f"{t}|{key}", k2, ent) for t, T in J["tasks"].items() for key, R in T.items() if "pooled" in R for k2, ent in R["pooled"].items()]
        k2s = sorted({k2 for _, k2, _ in rows}, key=lambda s: (int(s.split("|")[0][1:]), s))

        def agg(sel):
            d = {}
            for g in GROUP_NAMES:
                vals = [e["groups"][g]["mean_folded"] for _, _, e in sel if g in e["groups"]]
                d[g] = float(np.mean(vals)) if vals else None
            allc = [sum(N_CELLS[g] * e["groups"][g]["mean_folded"] for g in N_CELLS) / sum(N_CELLS.values()) for _, _, e in sel if all(g in e["groups"] for g in N_CELLS)]
            d["all cells"] = float(np.mean(allc)) if allc else None
            d["random_95"] = float(np.mean([e["random_band_95"] for _, _, e in sel]))
            pcs = [e["probe_ceiling"] for _, _, e in sel if e["probe_ceiling"] is not None]
            d["probe"] = float(np.mean(pcs)) if pcs else None
            d["n"] = len(sel)
            return d

        out[tag] = {"readouts": sorted({r for r, _, _ in rows}), "n_cells": len(rows),
                    "by_depth_pos": {k2: agg([r for r in rows if r[1] == k2]) for k2 in k2s}, "all": agg(rows),
                    "crossings": {g: int(sum(1 for _, _, e in rows if g in e["groups"] and e["groups"][g]["mean_folded"] > e["random_band_95"])) for g in GROUP_NAMES[:8]}}
    return out


def part3(sweeps):
    """sweeps: {setting: sweep json or None}. The fixed rule's single-cell counts and rates, the table-level check (a
    group's mean over the readouts above the table's random 95th at some position and index), the best honesty excess
    per position, and the means over indices 1 to the last."""
    out = {}
    for tag, S in sweeps.items():
        if S is None:
            continue
        dec = S["decision"]
        tl = {"honesty": 0, "control": 0, "rows": 0, "best_excess": {}}
        means = {}
        for pos, T in S["tables"].items():
            best = None
            for l, row in T.items():
                tl["rows"] += 1
                for g in HON_GROUPS:
                    tl["honesty"] += int(row[g] > row["random_95"])
                    ex = row[g] - row["random_95"]
                    if best is None or ex > best["excess"]:
                        best = {"excess": float(ex), "index": int(l), "group": g, "mean_folded": row[g], "random_95": row["random_95"]}
                for g in ("K1", "K2"):
                    tl["control"] += int(row[g] > row["random_95"])
            tl["best_excess"][pos] = best
            late = [row for l, row in T.items() if int(l) >= 1]
            means[pos] = {SHORT[g]: float(np.mean([r[g] for r in late])) for g in GROUP_NAMES}
            means[pos].update({"random_95": float(np.mean([r["random_95"] for r in late])), "probe": float(np.mean([r["probe"] for r in late]))})
        out[tag] = {"single_cell": {"honesty": dec["honesty_crossings"], "control": dec["control_crossings"], "cells": dec["cells"],
                                    "honesty_rate": dec["honesty_crossings"] / (dec["cells"] * len(HON_GROUPS)), "control_rate": dec["control_crossings"] / (dec["cells"] * 2),
                                    "branch": dec["branch"]},
                    "table_level": tl, "means_over_indices": means}
    return out


def render(out):
    tags = [t for t in out["settings"]]
    md = [f"# Standardization floor robustness (D-064 amendment 1 ruling d), {out['model']}", "",
          "sd_floored = max(sd, quantile(sd, q)) replaces the pooled per-dimension sd wherever a direction is built or read (cm.coherence.floor_sd); "
          "none = today's code path, written to robust_floor/none/ and identical to the baseline files (cm.robust_floor check). Numbers only.", ""]
    P1 = out["part1"]
    if P1:
        md += ["## 1. Leave-one-cell-out score (D-058 part 1): mean scaled score over the 22 honesty cells; in brackets the six controls' mean |scaled|", "",
               "| depth | method | " + " | ".join(tags) + " |", sep(2 + len(tags))]
        for L in P1["depths"]:
            for m, mn in (("dm", "diff of means"), ("probe", "probe")):
                E = P1["mean"][f"{L}|{m}"]
                md.append(f"| {L} | {mn} | " + " | ".join(f"{f2(E[t]['mean_scaled'])} ({f2(E[t]['controls_mean_abs_scaled'])})" if t in E else "-" for t in tags) + " |")
        md += ["", "Spearman rank correlation over the 22 cells against the unfloored score (does the ordering of cells survive):", "",
               "| depth | method | " + " | ".join(tags[1:]) + " |", sep(1 + len(tags))]
        for L in P1["depths"]:
            for m, mn in (("dm", "diff of means"), ("probe", "probe")):
                E = P1["mean"][f"{L}|{m}"]
                md.append(f"| {L} | {mn} | " + " | ".join(f2(E[t]["spearman_vs_none"]) if t in E else "-" for t in tags[1:]) + " |")
        Ls = P1["steering"]
        md += ["", f"By frame block at the steering depth {Ls} (own-reply / act / self-describe):", "",
               "| method | " + " | ".join(tags) + " |", sep(1 + len(tags))]
        for m, mn in (("dm", "diff of means"), ("probe", "probe")):
            E = P1["mean"][f"{Ls}|{m}"]
            md.append(f"| {mn} | " + " | ".join(" / ".join(f2(E[t]["by_block"][b]) for b in BLOCKS) if t in E else "-" for t in tags) + " |")
        if any(P1["sd"].values()):
            md += ["", "The pooled sd of the 22 honesty cells' training rows before the floor, and the floor per setting (value; dimensions raised to it):", "",
                   "| depth | dims | min | 5th pct | median | max | dims < 0.05 | " + " | ".join(tags[1:]) + " |", sep(6 + len(tags))]
            for L in P1["depths"]:
                S = P1["sd"].get(L, {})
                b = S.get("none") or next(iter(S.values()), None)
                if b:
                    md.append(f"| {L} | {b['n_dims']} | {b['min']:.4f} | {b['p05']:.4f} | {b['median']:.3f} | {b['max']:.2f} | {b['n_below_0.05']} | "
                              + " | ".join(f"{S[t]['floor']:.4f} ({S[t]['n_raised']})" if t in S else "-" for t in tags[1:]) + " |")
        md += ["", f"Per cell at the steering depth {Ls}, diff of means / probe (controls last):", "", "| cell | " + " | ".join(tags) + " |", sep(1 + len(tags))]
        for c in ALL:
            E = P1["cells"][c]
            md.append(f"| {c} | " + " | ".join(f"{f2(E[t]['dm'])} / {f2(E[t]['probe'])}" if t in E else "-" for t in tags) + " |")
        md.append("")
    P2 = out["part2"]
    if P2:
        md += ["## 2. On-policy prediction within item strata (D-062 item 4): folded AUROC for the dishonest outcome, mean over the task readouts; "
               "all cells = the 22 cell directions together; probe = within-task 5-fold ceiling", "",
               "| setting | depth, position | act | self | own | all cells | frames | shared | all_frames | K1 | K2 | random mean / 95th | probe |", sep(13)]
        for t in tags:
            if t not in P2:
                continue
            E = P2[t]
            for k2, d in list(E["by_depth_pos"].items()) + [(f"all {E['n_cells']} cells", E["all"])]:
                md.append(f"| {t} | {k2} | " + " | ".join(f2(d[g]) for g in GROUP_NAMES[:3]) + f" | {f2(d['all cells'])} | "
                          + " | ".join(f2(d[g]) for g in GROUP_NAMES[3:8]) + f" | {f2(d['random (16)'])} / {f2(d['random_95'])} | {f2(d['probe'])} |")
        md += ["", "Cells (task readout x depth x position) where the group mean exceeds the random 95th percentile:", "",
               "| setting | of | " + " | ".join(SHORT[g] for g in GROUP_NAMES[:8]) + " |", sep(10)]
        for t in tags:
            if t in P2:
                md.append(f"| {t} | {P2[t]['n_cells']} | " + " | ".join(str(P2[t]["crossings"][g]) for g in GROUP_NAMES[:8]) + " |")
        md.append("")
    P3 = out["part3"]
    if P3:
        md += ["## 3. Position-and-layer sweep (D-063 item 1), the median floor against the unfloored run (none = the D-063 run)", "",
               "| setting | table-level honesty crossings (rows x 6 groups) | table-level control crossings (rows x 2) | single-cell honesty crossings (rate) | "
               "single-cell control crossings (rate) | best honesty excess over the table random 95th: value / turn_mean / boundary |", sep(6)]
        for t, E in P3.items():
            tl, sc = E["table_level"], E["single_cell"]
            be = " / ".join(f"{b['excess']:+.2f} ({SHORT[b['group']]}, index {b['index']})" for b in tl["best_excess"].values() if b)
            md.append(f"| {t} | {tl['honesty']} of {tl['rows'] * len(HON_GROUPS)} | {tl['control']} of {tl['rows'] * 2} | "
                      f"{sc['honesty']} of {sc['cells'] * len(HON_GROUPS)} ({100 * sc['honesty_rate']:.1f}%) | {sc['control']} of {sc['cells'] * 2} ({100 * sc['control_rate']:.1f}%) | {be} |")
        md += ["", "Mean folded AUROC over the task readouts and over indices 1 to the last, per position:", "",
               "| setting | position | " + " | ".join(SHORT[g] for g in GROUP_NAMES) + " | random 95th | probe |", sep(len(GROUP_NAMES) + 4)]
        for t, E in P3.items():
            for pos, d in E["means_over_indices"].items():
                md.append(f"| {t} | {pos} | " + " | ".join(f2(d[SHORT[g]]) for g in GROUP_NAMES) + f" | {f2(d['random_95'])} | {f2(d['probe'])} |")
        md.append("")
    LC = out["lever_cosines"]
    if LC and LC.get("settings"):
        md += ["## 4. What the floor turns, against the unfloored directions at the two item-4 depths: u the standardized direction, w = unit(u / sd) the readout "
               "covector, p = unit(sd u) the primal lever; cos(w, p) is lever-readout agreement before -> after the floor", "",
               "| setting | depth | floor (dims raised) | dm_all: cos u / w / p; cos(w, p) | probe_all | frame_c | all_frames | 22 cells, mean: cos u / w / p; cos(w, p) | random, mean: cos(w, p) |",
               sep(9)]

        def fmt(x):
            return f"{x['cos_u']:.2f} / {x['cos_w']:.2f} / {x['cos_p']:.3f}; {x['cos_wp_before']:.2f} -> {x['cos_wp_after']:.2f}"

        for t, byL in LC["settings"].items():
            for L, e in byL.items():
                h, cs, r = e["heads"], e["cells_summary"], e["random_summary"]
                md.append(f"| {t} | {L} | {e['floor']:.4f} ({e['n_raised']}) | {fmt(h['dm_all'])} | {fmt(h['probe_all'])} | {fmt(h['frame_c'])} | {fmt(h['all_frames'])} | "
                          f"{cs['cos_u']['mean']:.2f} / {cs['cos_w']['mean']:.2f} / {cs['cos_p']['mean']:.3f}; {cs['cos_wp_before']['mean']:.2f} -> {cs['cos_wp_after']['mean']:.2f} | "
                          f"{r['cos_wp_before']:.2f} -> {r['cos_wp_after']:.2f} |")
        md.append("")
    return "\n".join(md) + "\n"


def cmd_table():
    reps = {qtag(q): load_json(robust_dir(OUT, q) / "representational.json") for q in QS}
    strats = {qtag(q): load_json(OP_RES / "robust_floor" / qtag(q) / "baseline" / "auroc_strat.json") for q in QS}
    sweeps = {"none": load_json(OP_RES / "baseline" / "sweep.json"), "q0.5": load_json(OP_RES / "robust_floor" / "q0.5" / "baseline" / "sweep.json")}
    lc = load_json(RDIR / "lever_cosines.json")
    if lc:
        for byL in lc["settings"].values():
            for e in byL.values():
                e.pop("cells", None)
    out = {"model": cfg.MODEL, "settings": [qtag(q) for q in QS], "part1": part1(reps) if reps["none"] else None,
           "part2": part2(strats), "part3": part3(sweeps), "lever_cosines": lc}
    RDIR.mkdir(parents=True, exist_ok=True)
    (RDIR / "table.json").write_text(json.dumps(out, indent=1, default=float))
    (RDIR / "table.md").write_text(render(out))
    print((RDIR / "table.md").read_text())
    print("[robust_floor] ->", RDIR / "table.md")


# ---------------------------------------------------------------- figure: per cell, unfloored against the median floor

def cmd_figure(tags):
    from scipy.stats import spearmanr
    plt = style()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(tags), figsize=(3.55 * len(tags), 3.5), squeeze=False)
    plotted = {}
    for ax, tag, letter in zip(axes[0], tags, "abcdef"):
        d = ROOT_RESULTS / tag / "coherence" / "robust_floor"
        A = json.loads((d / "none" / "representational.json").read_text())
        B = json.loads((d / "q0.5" / "representational.json").read_text())
        L = str(A["depths"][1])
        pts = []
        for c in ALL:
            info = cell_info(c)
            col = "#898781" if info["ctrl"] else COLOR[info["def"]]
            for m, hollow in (("dm", False), ("probe", True)):
                x, y = A["by_depth"][L]["cells"][c][f"scaled_{m}"], B["by_depth"][L]["cells"][c][f"scaled_{m}"]
                if x is None or y is None:
                    continue
                ax.scatter(x, y, marker=MARK[info["frame"]], s=20, facecolor="none" if hollow else col, edgecolor=col, linewidth=0.7,
                           zorder=2 if info["ctrl"] else 3)
                pts.append({"cell": c, "method": m, "none": x, "q0.5": y, "ctrl": info["ctrl"]})
        lo = min(min(p["none"], p["q0.5"]) for p in pts)
        hi = max(max(p["none"], p["q0.5"]) for p in pts)
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#c3c2b7", lw=0.5, ls="--", zorder=1)
        ax.axhline(0, color="#c3c2b7", lw=0.5, zorder=1)
        ax.axvline(0, color="#c3c2b7", lw=0.5, zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Leave-one-cell-out score, sd unfloored (layer {L})")
        ax.set_ylabel("Leave-one-cell-out score, sd floored at its median")
        ax.set_title(tag, loc="left")
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
        ax.grid(color="#e1e0d9", lw=0.4)
        ax.set_axisbelow(True)
        sp = {}
        for m in ("dm", "probe"):
            hon = [p for p in pts if not p["ctrl"] and p["method"] == m]
            sp[m] = float(spearmanr([p["none"] for p in hon], [p["q0.5"] for p in hon]).correlation)
        plotted[tag] = {"layer": int(L), "spearman_honesty_cells": sp, "points": pts}
    h = [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=4.5) for n, c in COLOR.items() if "control" not in n] + \
        [Line2D([], [], marker="o", ls="none", color="#898781", label="controls (K1, K2)", ms=4.5)] + \
        [Line2D([], [], marker=m, ls="none", color="#52514e", mfc="white", label=FRAME_NAME[f], ms=4.5) for f, m in MARK.items()] + \
        [Line2D([], [], marker="o", ls="none", color="#52514e", label="diff of means", ms=4.5),
         Line2D([], [], marker="o", ls="none", mfc="white", mec="#52514e", label="probe", ms=4.5)]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.14), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    figd = ROOT_RESULTS / "figures" / "honesty"
    figd.mkdir(parents=True, exist_ok=True)
    fig.savefig(figd / "robust_floor.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(figd / "robust_floor.pdf", bbox_inches="tight", pad_inches=0.03)
    (figd / "robust_floor.json").write_text(json.dumps(plotted, indent=1, default=float))
    print("[robust_floor] wrote", figd / "robust_floor.png", {t: v["spearman_honesty_cells"] for t, v in plotted.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "lever-cos", "table", "figure"])
    ap.add_argument("--tags", default="gemma-4-31b-it,qwen3.5-27b", help="figure: comma list of result tags, one panel each")
    a = ap.parse_args()
    if a.stage == "check":
        cmd_check()
    elif a.stage == "lever-cos":
        cmd_lever_cos()
    elif a.stage == "table":
        cmd_table()
    else:
        cmd_figure(a.tags.split(","))


if __name__ == "__main__":
    main()
