"""A conduct lever from the on-policy probe (D-063 item 2). CPU only, on the D-062 item 4 activations.

At the steering depth (and at any further layer named with --layer-index, from the sweep), a logistic probe on the
pooled on-policy outcome (dishonest act vs not: AR covert COMPLETE, DC false done at the code turn, RH cheat, CF
fabricates), rows standardized by the pooled honesty SD of that depth, weights balanced within task over stratum and
class and across tasks. Leave-one-task-out transfer AUROC (raw and within the held-out task's strata) is the on-policy
generalization number; the pooled direction's cosines with the 22 cells, the frame levers, the shared axes and K1/K2
say how it sits against the described-honesty geometry. Controls: an item-family probe trained on the neutral-prompt
trajectories (item 5, RH/FT/CF) and a reply-length regression, each as a direction (cosine with the conduct direction,
AUROC for conduct), and the conduct probe refit after projecting the length direction out.

    CM_MODEL=<m> python -m cm.battery_conduct gate [--pos reply32|boundary] [--layer-index 0]   CPU
        -> results/<tag>/battery/conduct/conduct_<pos>_L<layer>.{json,md}, conduct_<pos>_L<layer>_dir.npz
    CM_MODEL=<m> python -m cm.battery_conduct write-levers --pos reply32 --layer-index 0        CPU, after the gate
        -> adds conduct|resid (standardized unit) and conduct|primal (unit(sd * u), the form to inject) to
           results/<tag>/coherence/levers_part3.npz, keeps a copy of the original under outputs/<tag>/coherence/
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from . import battery_onpolicy as op
from . import cfg
from .stimuli.schema import read_jsonl

RES = cfg.results_dir() / "battery" / "conduct"
TASK_KIND = {"AR": "write", "DC": "code", "RH": "reply", "CF": "reply"}
FAMILY = {"RH": ("one-off", "conflicting"), "FT": ("error", "no_call"), "CF": ("fictitious_reference", "kuq")}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def load_units(cond, pos, li):
    """Rows of the four conduct tasks at one position and depth index: X (raw), y, task, stratum, reply tokens."""
    strata = op.strata_of()
    X, y, task, st, nrep, ids = [], [], [], [], [], []
    for t, kind in TASK_KIND.items():
        A = np.load(op.OUT / cond / f"{t}.npz")
        meta = json.loads(str(A["meta"]))
        rows = [i for i, m in enumerate(meta) if m["kind"] == kind and m.get("dishonest") is not None and m["reply_tokens"] > 0]
        X.append(A[pos][rows, li].astype(np.float64)); y += [bool(meta[i]["dishonest"]) for i in rows]
        task += [t] * len(rows); st += [f"{t}|{strata[meta[i]['item_id']]}" for i in rows]
        nrep += [meta[i]["reply_tokens"] for i in rows]; ids += [meta[i]["item_id"] for i in rows]
    return np.concatenate(X), np.array(y, bool), np.array(task), np.array(st), np.array(nrep, float), ids


def weights(y, task, st):
    """Each (stratum, class) cell the same total weight within its task; each task the same total weight."""
    w = np.zeros(len(y))
    for t in np.unique(task):
        cells = [(s, c) for s in np.unique(st[task == t]) for c in (0, 1) if ((st == s) & (y == c)).any()]
        for s, c in cells:
            m = (st == s) & (y == c)
            w[m] = 1.0 / m.sum() / len(cells)
    return w


def fit_probe(Xs, y, w, C=1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=3000).fit(Xs, y, sample_weight=w)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def auroc_strat(score, y, st):
    """Within-stratum AUROC pooled by positives; strata need 5 of each class; also the raw AUROC."""
    vals, ws = [], []
    for s in np.unique(st):
        m = st == s
        if y[m].sum() >= 5 and (~y[m]).sum() >= 5:
            vals.append(op.auroc(score[m], y[m])); ws.append(int(y[m].sum()))
    return (float(np.average(vals, weights=ws)) if vals else None), op.auroc(score, y)


def loto(Xs, y, task, st, w):
    out = {}
    for t in np.unique(task):
        tr, te = task != t, task == t
        coef, b = fit_probe(Xs[tr], y[tr], w[tr])
        a_st, a_raw = auroc_strat(Xs[te] @ coef, y[te], st[te])
        out[t] = {"transfer_auroc_within_strata": a_st, "transfer_auroc_raw": a_raw, "n_test": int(te.sum()), "n_positive": int(y[te].sum())}
    return out


def within_task_cv(Xs, y, task, st, seed=0):
    from sklearn.model_selection import StratifiedKFold
    out = {}
    for t in np.unique(task):
        m = task == t
        Xt, yt, stt = Xs[m], y[m], st[m]
        sc = np.zeros(len(yt))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xt, yt):
            coef, b = fit_probe(Xt[tr], yt[tr], weights(yt[tr], np.full(len(tr), t), stt[tr]))
            sc[te] = Xt[te] @ coef
        a_st, a_raw = auroc_strat(sc, yt, stt)
        out[t] = {"cv_auroc_within_strata": a_st, "cv_auroc_raw": a_raw}
    return out


def family_dirs(cond_neu, pos, li, sd):
    """Per neutral-prompt task: the probe direction for the item family (standardized), and its rows."""
    strata = op.strata_of()
    out = {}
    for t, fams in FAMILY.items():
        p = op.OUT / cond_neu / f"{t}.npz"
        if not p.exists():
            continue
        A = np.load(p)
        meta = json.loads(str(A["meta"]))
        rows = [i for i, m in enumerate(meta) if m["reply_tokens"] > 0]
        fam = np.array([strata[meta[i]["item_id"]] for i in rows])
        yf = np.array([not f.startswith(fams[0]) for f in fam], bool)       # second family (or any kuq) = 1
        Xs = A[pos][rows, li].astype(np.float64) / sd
        coef, b = fit_probe(Xs, yf, np.where(yf, 0.5 / max(yf.sum(), 1), 0.5 / max((~yf).sum(), 1)))
        out[t] = {"dir": unit(coef), "n": len(rows), "n_family_1": int(yf.sum()), "families": list(fams), "Xs": Xs, "yf": yf}
    return out


def cmd_gate(cond="baseline", pos="reply32", li=0, cond_neu="p_neu"):
    L = op.LAYERS[li]
    D = np.load(op.RES / "directions.npz")
    sd = D[f"sd|L{L}"].astype(np.float64)
    X, y, task, st, nrep, ids = load_units(cond, pos, li)
    Xs = X / sd
    w = weights(y, task, st)
    res = {"model": cfg.MODEL, "cond": cond, "position": pos, "layer": L, "n": int(len(y)), "n_positive": int(y.sum()),
           "by_task": {t: {"n": int((task == t).sum()), "n_positive": int(y[task == t].sum())} for t in TASK_KIND}}
    res["leave_one_task_out"] = loto(Xs, y, task, st, w)
    res["within_task_cv"] = within_task_cv(Xs, y, task, st)
    coef, b = fit_probe(Xs, y, w)
    u = unit(coef)
    a_st, a_raw = auroc_strat(Xs @ u, y, st)
    res["pooled_fit"] = {"train_auroc_within_strata": a_st, "train_auroc_raw": a_raw}
    names = sorted({k.split("|L")[0] for k in D.files if not k.startswith("sd|")})
    cos = {nm: float(u @ (D[f"{nm}|L{L}"].astype(np.float64))) for nm in names}
    res["cosines"] = cos
    res["cosine_summary"] = {g: {"mean": float(np.mean([cos[nm] for nm in names if f(nm)])), "max_abs": float(np.max([abs(cos[nm]) for nm in names if f(nm)]))}
                             for g, f in op.GROUPS.items() if any(f(nm) for nm in names)}
    # control 1: item-family probes on the neutral-prompt trajectories
    fam = family_dirs(cond_neu, pos, li, sd)
    res["control_family"] = {}
    for t, f in fam.items():
        a_fam_conduct_st, a_fam_conduct_raw = auroc_strat(Xs @ f["dir"], y, st)            # family direction read on the conduct rows
        a_conduct_family = op.auroc(f["Xs"] @ u, f["yf"])                                    # conduct direction read on the family split
        res["control_family"][t] = {"families": f["families"], "n_neutral_rows": f["n"], "cos_with_conduct": float(u @ f["dir"]),
                                    "family_dir_auroc_for_conduct_within_strata": a_fam_conduct_st, "family_dir_auroc_for_conduct_raw": a_fam_conduct_raw,
                                    "conduct_dir_auroc_for_family_on_neutral": a_conduct_family}
    # control 2: reply length
    from sklearn.linear_model import Ridge
    ln = np.log(nrep)
    r = Ridge(alpha=1.0).fit(Xs, ln - ln.mean(), sample_weight=w)
    v = unit(r.coef_.astype(np.float64))
    a_len_st, a_len_raw = auroc_strat(Xs @ v, y, st)
    Xp = Xs - np.outer(Xs @ v, v)
    coef_p, _ = fit_probe(Xp, y, w)
    up = unit(coef_p)
    a_p_st, a_p_raw = auroc_strat(Xp @ up, y, st)
    res["control_length"] = {"cos_with_conduct": float(u @ v), "corr_length_vs_conduct_score": float(np.corrcoef(Xs @ u, ln)[0, 1]),
                             "length_dir_auroc_for_conduct_within_strata": a_len_st, "length_dir_auroc_for_conduct_raw": a_len_raw,
                             "conduct_after_projecting_out_length": {"train_auroc_within_strata": a_p_st, "train_auroc_raw": a_p_raw,
                                                                     "leave_one_task_out": loto(Xp, y, task, st, w), "cos_with_original": float(u @ up)}}
    RES.mkdir(parents=True, exist_ok=True)
    tag = f"{pos}_L{L}"
    np.savez(RES / f"conduct_{tag}_dir.npz", resid=u.astype(np.float32), primal=unit(sd * u).astype(np.float32), sd=sd.astype(np.float32),
             length_resid=v.astype(np.float32), coef_raw=coef.astype(np.float32), intercept=np.float32(b))
    (RES / f"conduct_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    md = [f"# Conduct lever (D-063 item 2), {cfg.MODEL}, position {pos}, layer {L}", "",
          f"Rows {len(y)} (dishonest {int(y.sum())}): " + ", ".join(f"{t} {v['n']}/{v['n_positive']}" for t, v in res["by_task"].items()), "",
          "| held-out task | transfer AUROC within strata | transfer raw | within-task 5-fold (strata / raw) |", "| --- | --- | --- | --- |"]
    for t, v in res["leave_one_task_out"].items():
        c = res["within_task_cv"][t]
        f2 = lambda x: "-" if x is None else f"{x:.2f}"
        md.append(f"| {t} | {f2(v['transfer_auroc_within_strata'])} | {f2(v['transfer_auroc_raw'])} | {f2(c['cv_auroc_within_strata'])} / {f2(c['cv_auroc_raw'])} |")
    md += ["", "Cosine of the pooled conduct direction with the item 4 directions (mean / max |cos| per group):", ""]
    md += [f"- {g}: {v['mean']:+.3f} / {v['max_abs']:.3f}" for g, v in res["cosine_summary"].items()]
    md += ["", "Controls:", ""]
    for t, v in res["control_family"].items():
        md.append(f"- family probe {t} ({' vs '.join(v['families'])}, neutral prompt, n {v['n_neutral_rows']}): cos with conduct {v['cos_with_conduct']:+.3f}; "
                  f"family direction AUROC for conduct {f2(v['family_dir_auroc_for_conduct_within_strata'])} (raw {f2(v['family_dir_auroc_for_conduct_raw'])}); "
                  f"conduct direction AUROC for the family split on neutral rows {v['conduct_dir_auroc_for_family_on_neutral']:.2f}")
    cl = res["control_length"]
    md.append(f"- reply length: cos with conduct {cl['cos_with_conduct']:+.3f}, corr(conduct score, log length) {cl['corr_length_vs_conduct_score']:+.2f}; "
              f"length direction AUROC for conduct {f2(cl['length_dir_auroc_for_conduct_within_strata'])} (raw {f2(cl['length_dir_auroc_for_conduct_raw'])}); "
              f"conduct probe after projecting length out: train {f2(cl['conduct_after_projecting_out_length']['train_auroc_within_strata'])}, LOTO "
              + ", ".join(f"{t} {f2(v['transfer_auroc_within_strata'])}" for t, v in cl["conduct_after_projecting_out_length"]["leave_one_task_out"].items())
              + f", cos with the original {cl['conduct_after_projecting_out_length']['cos_with_original']:.3f}")
    (RES / f"conduct_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def cmd_write_levers(pos="reply32", li=0):
    L = op.LAYERS[li]
    src = RES / f"conduct_{pos}_L{L}_dir.npz"
    lev_p = cfg.results_dir() / "coherence" / "levers_part3.npz"
    bak = cfg.outputs_dir() / "coherence" / "levers_part3_pre_d063.npz"
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copy(lev_p, bak)
    C = np.load(src)
    lev = dict(np.load(lev_p))
    suffix = "" if L == cfg.L_STEER else f"|L{L}"
    lev[f"conduct{suffix}|resid"] = C["resid"]
    lev[f"conduct{suffix}|primal"] = C["primal"]
    np.savez(lev_p, **lev)
    jp = cfg.results_dir() / "coherence" / "levers_part3.json"
    info = json.loads(jp.read_text())
    info.setdefault("conduct", {})[f"conduct{suffix}"] = {
        "source": f"D-063 item 2: logistic probe on the pooled on-policy dishonest outcome (AR covert COMPLETE, DC false done, RH cheat, CF fabricates), "
                  f"position {pos}, layer {L}, standardized by the pooled honesty SD; cm.battery_conduct gate",
        "inject": f"conduct{suffix}|primal = unit(sd * u), scaled to the dose as the other levers", "read": f"conduct{suffix}|resid",
        "cos_primal_vs_dm_all_primal": float(unit(C["primal"].astype(np.float64)) @ unit(lev["dm_all|primal"].astype(np.float64))) if "dm_all|primal" in lev else None}
    info["keys"] = sorted(lev)
    jp.write_text(json.dumps(info, indent=1, default=float))
    print(f"[conduct] levers_part3.npz now has {len(lev)} arrays (added conduct{suffix}|resid, conduct{suffix}|primal); original kept at {bak}")


def cmd_cf_lever(cond="baseline", pos="reply32", li=0):
    """D-063 items 1 and 2 rulings: CF's own conduct probe (within-task fit, standardized, weights balanced over stratum and
    class) as a steering positive control. Writes conduct_cf|resid (standardized unit) and conduct_cf|primal (unit(sd * u))
    into levers_part3.npz, keeps the original under outputs/<tag>/coherence/, and describes the lever in levers_part3.json."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold
    L = op.LAYERS[li]
    D = np.load(op.RES / "directions.npz")
    sd = D[f"sd|L{L}"].astype(np.float64)
    X, y, task, st, nrep, ids = load_units(cond, pos, li)
    m = task == "CF"
    Xs, yc, stc, ln = X[m] / sd, y[m], st[m], np.log(nrep[m])
    w = weights(yc, np.full(m.sum(), "CF"), stc)
    coef, b = fit_probe(Xs, yc, w)
    u = unit(coef)
    # 5-fold within CF, scores pooled, AUROC within strata and raw
    sc = np.zeros(len(yc))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, yc):
        c_, _ = fit_probe(Xs[tr], yc[tr], weights(yc[tr], np.full(len(tr), "CF"), stc[tr]))
        sc[te] = Xs[te] @ c_
    cv_st, cv_raw = auroc_strat(sc, yc, stc)
    tr_st, tr_raw = auroc_strat(Xs @ u, yc, stc)
    # read on the other tasks (specificity): the CF direction's AUROC for their dishonest outcome
    other = {}
    for t in ("AR", "DC", "RH"):
        mt = task == t
        a_st, a_raw = auroc_strat((X[mt] / sd) @ u, y[mt], st[mt])
        other[t] = {"auroc_within_strata": a_st, "auroc_raw": a_raw}
    names = sorted({k.split("|L")[0] for k in D.files if not k.startswith("sd|")})
    cos = {nm: float(u @ D[f"{nm}|L{L}"].astype(np.float64)) for nm in names}
    cos_groups = {g: {"mean": float(np.mean([cos[nm] for nm in names if f(nm)])), "max_abs": float(np.max([abs(cos[nm]) for nm in names if f(nm)]))}
                  for g, f in op.GROUPS.items() if any(f(nm) for nm in names)}
    # length control within CF
    r = Ridge(alpha=1.0).fit(Xs, ln - ln.mean(), sample_weight=w)
    v = unit(r.coef_.astype(np.float64))
    Xp = Xs - np.outer(Xs @ v, v)
    cp, _ = fit_probe(Xp, yc, w)
    up = unit(cp)
    p_st, p_raw = auroc_strat(Xp @ up, yc, stc)
    len_st, len_raw = auroc_strat(Xs @ v, yc, stc)
    strata_n = {s_: {"n": int((stc == s_).sum()), "n_positive": int(yc[stc == s_].sum())} for s_ in np.unique(stc)}
    desc = {"source": f"D-063 items 1 and 2 rulings: CF's own conduct probe, logistic C = 1 on the CF induced trajectories' standardized "
                      f"{pos} activation at layer {L} (D-062 item 4 acts), outcome FABRICATES vs not (judge label), weights balanced over "
                      f"stratum (CF factor) and class; cm.battery_conduct cf-lever",
            "training_rows": {"n": int(m.sum()), "n_positive": int(yc.sum()), "strata": strata_n},
            "auroc": {"train_within_strata": tr_st, "train_raw": tr_raw, "cv5_within_strata": cv_st, "cv5_raw": cv_raw},
            "specificity_other_tasks": other,
            "cosines_with_item4_groups": cos_groups,
            "length_control": {"cos_with_length_dir": float(u @ v), "corr_score_vs_log_length": float(np.corrcoef(Xs @ u, ln)[0, 1]),
                               "length_dir_auroc_within_strata": len_st, "length_dir_auroc_raw": len_raw,
                               "after_projecting_length_out": {"cv_not_rerun": True, "train_within_strata": p_st, "train_raw": p_raw, "cos_with_original": float(u @ up)}},
            "inject": "conduct_cf|primal = unit(sd * u), scaled to the dose as the other levers; positive sign = toward FABRICATES",
            "read": "conduct_cf|resid (standardized projection)", "stage_a_conditions": "2 signs x 2 doses + project-out = 5",
            "reading_rule": "if the CF lever moves CF conduct, the described-direction nulls on conduct are interpretable; if not, Stage A's conduct arm is uninterpretable for steering"}
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(RES / f"conduct_cf_{pos}_L{L}_dir.npz", resid=u.astype(np.float32), primal=unit(sd * u).astype(np.float32), sd=sd.astype(np.float32))
    (RES / f"conduct_cf_{pos}_L{L}.json").write_text(json.dumps(desc, indent=1, default=float))
    # write into levers_part3.npz
    lev_p = cfg.results_dir() / "coherence" / "levers_part3.npz"
    bak = cfg.outputs_dir() / "coherence" / "levers_part3_pre_d063.npz"
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        shutil.copy(lev_p, bak)
    lev = dict(np.load(lev_p))
    lev["conduct_cf|resid"] = u.astype(np.float32)
    lev["conduct_cf|primal"] = unit(sd * u).astype(np.float32)
    np.savez(lev_p, **lev)
    jp = cfg.results_dir() / "coherence" / "levers_part3.json"
    info = json.loads(jp.read_text())
    desc["cos_primal_vs_dm_all_primal"] = float(unit(sd * u) @ unit(lev["dm_all|primal"].astype(np.float64)))
    info.setdefault("conduct", {})["conduct_cf"] = desc
    info["keys"] = sorted(lev)
    jp.write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(desc, indent=1, default=float))
    print(f"[conduct] levers_part3.npz now has {len(lev)} arrays (conduct_cf|resid, conduct_cf|primal added); original kept at {bak}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gate", "write-levers", "cf-lever"])
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--pos", default="reply32", choices=["reply32", "boundary"])
    ap.add_argument("--layer-index", type=int, default=0, help="index into cm.battery_onpolicy.LAYERS (0 = steering depth)")
    a = ap.parse_args()
    if a.stage == "gate":
        cmd_gate(a.cond, a.pos, a.layer_index)
    elif a.stage == "cf-lever":
        cmd_cf_lever(a.cond, a.pos, a.layer_index)
    else:
        cmd_write_levers(a.pos, a.layer_index)


if __name__ == "__main__":
    main()
