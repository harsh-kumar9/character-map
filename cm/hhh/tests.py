"""The cross-block tests of D-055, read from the HHH S4 (results/hhh/s4/overlap_verdict.json). Every unit is oriented
to its good side (blocks.py `good`), so a negative cosine means the two good sides pull apart.

1. Tension: harmlessness units that refuse or read charitably (C15 on harmful and on harmless requests, A7, C27
   instructed) against helpfulness units about substance and the person (C25, C6, C26), per frame. Prediction
   (D-055): over-refusal opposes substance; refusing a harmful request does not.
2. Knowing vs doing across blocks: refusing in text vs declining a tool call (C15 on harmful requests vs C27
   instructed), and harm perception vs refusal (B1, A1 vs C15 on harmful requests), per frame.
3. One assistant axis: every HHH unit against D1 (assistant vs another role) in the same frame, centred and raw
   cosine; per block, the mean squared raw cosine (the share of a unit direction's length on D1's line).
4. Shared nodes: C8 sycophancy and C3 forthrightness (honesty) against the helpfulness units; C23 manipulation
   against C26 wellbeing; C22 against C15 (declining without lying).

    python3 -m cm.hhh.tests            writes results/hhh/tests.json and prints the tables
"""
from __future__ import annotations

import json

import numpy as np

from .. import cfg
from .blocks import BLOCKS, out_dir

FR = "abcd"


def main():
    ov = json.loads((out_dir() / "s4" / "overlap_verdict.json").read_text())
    idx = {u: i for i, u in enumerate(ov["units"])}
    good = {u.key: u.good for us in BLOCKS.values() for u in us}
    name = {u.key: u.label for us in BLOCKS.values() for u in us}
    sgn = lambda uid: -1.0 if good.get(uid.rsplit("|", 1)[0]) == "neg" else 1.0
    layers = [str(L) for L in cfg.DEPTHS]
    res = {"floors": {L: ov["layers"][L]["floor_cos"] for L in layers}}

    def c(L, a, b, kind="cos"):
        if a not in idx or b not in idx:
            return None
        return round(sgn(a) * sgn(b) * ov["layers"][L][kind][idx[a]][idx[b]], 3)

    def grid(rows, cols, frames=FR):
        return {L: {f: {f"{r}~{k}": c(L, f"{r}|{f}", f"{k}|{f}") for r in rows for k in cols} for f in frames} for L in layers}

    harm_act = ["C15|refuse[harmful]", "C15|refuse[harmless]", "A7|all", "C27|instructed"]
    help_ = ["C25|all", "C6|all", "C26|all"]
    res["tension"] = grid(harm_act, help_)
    res["knowing_doing"] = grid(["C15|refuse[harmful]"], ["C27|instructed", "B1|all", "A1|all"])
    # 3. D1
    d1 = {}
    for L in layers:
        d1[L] = {}
        for b, units in BLOCKS.items():
            per = {}
            for u in units:
                for f in FR:
                    a, d = f"{u.key}|{f}", f"D1|all|{f}"
                    if a in idx and d in idx:
                        per[a] = {"cos": c(L, a, d), "raw": c(L, a, d, "raw")}
            raws = [v["raw"] for v in per.values() if v["raw"] is not None]
            d1[L][b] = {"units": per, "mean_sq_raw": round(float(np.mean(np.square(raws))), 4) if raws else None,
                        "mean_abs_cos": round(float(np.mean([abs(v["cos"]) for v in per.values()])), 3) if per else None}
        ctrl = [c(L, f"K1|all|{f}", f"D1|all|{f}", "raw") for f in FR]
        d1[L]["control_K1_raw"] = ctrl
    res["d1"] = d1
    res["shared_nodes"] = {
        "honesty_vs_help": grid(["C8|sycophantic_agreement", "C3|all"], [u.key for u in BLOCKS["helpfulness"]]),
        "C23_vs_C26": grid(["C23|all"], ["C26|all"]),
        "C22_vs_C15": grid(["C22|all"], ["C15|refuse[harmful]", "C15|refuse[harmless]", "C15|refuse"]),
    }
    (out_dir() / "tests.json").write_text(json.dumps(res, indent=1))
    show = lambda v: "  -  " if v is None else f"{v:+.2f}"
    for L in ("17", "29"):
        fl = res["floors"][L]
        print(f"\nlayer {L} (floor {fl:.2f})")
        print("  tension: rows harmlessness, columns helpfulness; frames a b c d")
        for r in harm_act:
            for k in help_:
                print(f"    {name.get(r, r):17s} ~ {name.get(k, k):12s} " + " ".join(show(res['tension'][L][f][f'{r}~{k}']) for f in FR))
        print("  knowing vs doing: refuses harmful (C15) against")
        for k in ("C27|instructed", "B1|all", "A1|all"):
            print(f"    {name.get(k, k):17s} " + " ".join(show(res['knowing_doing'][L][f][f'C15|refuse[harmful]~{k}']) for f in FR))
        print("  D1 (assistant vs another role), mean squared raw cosine per block: " +
              ", ".join(f"{b} {res['d1'][L][b]['mean_sq_raw']}" for b in BLOCKS) + f"; formality control raw {res['d1'][L]['control_K1_raw']}")
        for b in BLOCKS:
            top = sorted(res["d1"][L][b]["units"].items(), key=lambda kv: -abs(kv[1]["cos"] or 0))[:4]
            print(f"    {b:12s} largest centred: " + ", ".join(f"{k} {v['cos']:+.2f}" for k, v in top))
        print("  shared nodes")
        for grp, g in res["shared_nodes"].items():
            for f in FR:
                vals = {k: v for k, v in g[L][f].items() if v is not None and abs(v) >= fl}
                if vals:
                    print(f"    {grp} ({f}): " + ", ".join(f"{k} {v:+.2f}" for k, v in vals.items()))


if __name__ == "__main__":
    main()
