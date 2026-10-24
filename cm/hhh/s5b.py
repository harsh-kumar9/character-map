"""S5b for the HHH line (D-055): behavior under steering for the helpfulness and harmlessness concepts, written to
outputs/hhh/s5b and results/hhh/s5b so the honesty S5b does not move (ruling of 2026-10-24).

cm/s5_behavior.py's own stages, with three changes:
  - sources, vectors and dose come from the HHH S5a setup (cm/hhh/s5.py: same calibrated dose; C15's sources are
    the request-type units, so C15's doing and knowing conditions refuse harmful requests and its extra condition
    helps with harmless ones);
  - the concepts are the two blocks' (every one with a behavior instrument), rerun rather than read from the
    global S5b, since several now have a knowing frame they lacked (C5 b, A7 b, C20 and C19 a b);
  - C24's IFEval items (scoring type "ifeval") get 768 new tokens and are scored by the official strict checker
    (instruction_following_eval, langdetect seeded as in cm/stimuli/build/c24.py); the scorer is s5_behavior's
    cmd_score with that branch added and judges run for any concept with judged items.
--dose-mult M is the D-041 dose check: core conditions at M times the dose, under outputs/hhh/s5b_dose/xM.

    python -m cm.hhh.s5b gen --shard K --nshards N [--dose-mult M]     GPU
    python -m cm.hhh.s5b judge --judge gemma|qwen [--dose-mult M]      GPU (vLLM)
    python -m cm.hhh.s5b score [--dose-mult M]                         CPU   results/hhh/s5b/behavior.json
    python -m cm.hhh.s5b coherence [--dose-mult M]                     GPU
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .. import cfg
from .. import s5_behavior as B
from ..stimuli.schema import item_dir, read_jsonl
from .blocks import BLOCKS, out_dir

OUT = cfg.outputs_dir() / "hhh" / "s5b"
RES = out_dir() / "s5b"
S5 = out_dir() / "s5"
IFEVAL_TOKENS = 768


def concepts():
    cids = {u.cid for b in ("helpfulness", "harmlessness") for u in BLOCKS[b]}
    return sorted(c for c in cids if (B.STIM_ROOT / "stimuli" / c / "behavior.jsonl").exists())


def setup():
    S = json.loads((S5 / "setup.json").read_text())
    A = np.load(cfg.outputs_dir() / "hhh" / "s5" / "setup.npz")
    cal = json.loads((S5 / "calibration.json").read_text())
    norm = json.loads((S5 / f"residual_norm_L{B.L_IN}.json").read_text())["median_token_norm"]
    return S, A, cal["chosen_c"] * norm * B.MULT


_max_tokens = B.max_tokens


def max_tokens(sc):
    return IFEVAL_TOKENS if sc["type"] == "ifeval" else _max_tokens(sc)


def score_ifeval(it, reply):
    from instruction_following_eval.evaluation import test_instruction_following
    sc = it["scoring"]
    ex = dict(key=sc["key"], instruction_id_list=sc["instruction_id_list"], prompt=it["messages"][-1]["content"],
              kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in sc["kwargs"]])
    return float(test_instruction_following(ex, reply, strict=True).follow_all_instructions)


def cmd_score():
    """s5_behavior.cmd_score, plus the ifeval branch; judges for every concept that has judged items."""
    from langdetect import DetectorFactory
    from instruction_following_eval.evaluation import ensure_nltk_resource
    DetectorFactory.seed = 0
    ensure_nltk_resource()
    out = {}
    for cid in concepts():
        rp = B.OUT / "replies" / f"{cid}.jsonl"
        if not rp.exists():
            continue
        items = {it["item_id"]: it for it in read_jsonl(item_dir(cid) / "behavior.jsonl")}
        reps = read_jsonl(rp)          # at a dose multiple, gen has copied the 1x baseline rows into this file
        jud = {}
        for j in ("gemma", "qwen"):
            p = B.OUT / f"judged_{j}" / f"{cid}.jsonl"
            jud[j] = {(r["item_id"], r["cond"], r["key"]): r["out"] for r in read_jsonl(p)} if p.exists() else {}
        conds = sorted({r["cond"] for r in reps}, key=lambda c: (c != "baseline", c))
        has_judge = any(it["scoring"]["type"] == "judge" for it in items.values())
        scores = {j: {} for j in ("gemma", "qwen")}
        for r in reps:
            it = items[r["item_id"]]
            sc = it["scoring"]
            for j in ("gemma", "qwen"):
                if sc["type"] == "choice":
                    v = B.score_choice(sc, r["reply"])
                elif sc["type"] == "exact":
                    v = B.score_exact(sc, r["reply"])
                elif sc["type"] == "ifeval":
                    v = score_ifeval(it, r["reply"]) if j == "gemma" else scores["gemma"][(r["item_id"], r["cond"])]
                else:
                    o = jud[j].get((r["item_id"], r["cond"], "main"))
                    v = np.nan if o is None else B.score_judge(sc, o)
                scores[j][(r["item_id"], r["cond"])] = v
        ids = sorted(items)
        types = sorted({items[i]["scoring"]["type"] for i in ids})
        groups = {}
        for i in ids:
            mt = items[i].get("meta") or {}
            g = mt.get("condition", mt.get("contrast"))
            if g is not None:
                groups.setdefault(str(g), []).append(i)
            groups.setdefault(f"type:{items[i]['scoring']['type']}", []).append(i)
        res = {"type": "+".join(types), "n_items": len(ids), "conds": {}}
        for j in ("gemma", "qwen") if has_judge else ("gemma",):
            base = np.array([scores[j].get((i, "baseline"), np.nan) for i in ids])
            rand_eff, per = [], {}
            for c in conds:
                v = np.array([scores[j].get((i, c), np.nan) for i in ids])
                d = v - base
                ok = ~np.isnan(d)
                eff = float(np.mean(d[ok])) if ok.any() else np.nan
                se = float(np.std(d[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 2 else np.nan
                per[c] = {"mean": float(np.nanmean(v)) if (~np.isnan(v)).any() else np.nan,
                          "unparsed": float(np.mean(np.isnan(v))), "effect": eff, "se": se,
                          "by_group": {g: float(np.nanmean([scores[j].get((i, c), np.nan) for i in gi])) for g, gi in groups.items()}}
                if c.startswith("random"):
                    rand_eff.append(eff)
            for c in per:
                if c != "baseline" and not c.startswith("random") and rand_eff:
                    per[c]["vs_random"] = {"random_effects": rand_eff, "max_abs_random": float(np.nanmax(np.abs(rand_eff)))}
            res["conds"][j] = per
        out[cid] = res
        g = res["conds"]["gemma"]
        line = " ".join(f"{c} {g[c]['effect']:+.3f}±{g[c]['se']:.3f}" for c in g if c != "baseline" and not c.startswith("random"))
        rmax = max((abs(g[c]["effect"]) for c in g if c.startswith("random") and not np.isnan(g[c]["effect"])), default=np.nan)
        base_mean = g["baseline"]["mean"] if "baseline" in g else np.nan
        print(f"{cid:4s} {res['type']:12s} base {base_mean:.3f} | {line} | max|random| {rmax:.3f}", flush=True)
    B.RES.mkdir(parents=True, exist_ok=True)
    (B.RES / "behavior.json").write_text(json.dumps(out, indent=1, default=float))


def point(mult):
    B.concepts, B.setup, B.max_tokens = concepts, setup, max_tokens
    B.X1_OUT = OUT
    if mult != 1:
        B.MULT = mult
        tag = f"x{mult:g}"
        B.OUT, B.RES = cfg.outputs_dir() / "hhh" / "s5b_dose" / tag, out_dir() / "s5b_dose" / tag
    else:
        B.OUT, B.RES = OUT, RES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge", "score", "coherence"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--judge", default="gemma", choices=list(B.JUDGES))
    ap.add_argument("--concepts", default="")
    ap.add_argument("--dose-mult", type=float, default=1)
    a = ap.parse_args()
    point(a.dose_mult)
    if a.stage == "gen":
        B.cmd_gen(a.shard, a.nshards, set(a.concepts.split(",")) - {""})
    elif a.stage == "judge":
        B.cmd_judge(a.judge)
    elif a.stage == "coherence":
        B.cmd_coherence()
    else:
        cmd_score()


if __name__ == "__main__":
    main()
