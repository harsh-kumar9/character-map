"""Dose-response report for S5b (D-041): behavior effects at 1x, 2x, 4x and 8x the S5a dose, with coherence.

Inputs: results/s5b/{behavior,coherence}.json (1x) and results/s5b_dose/x{2,4,8}/{behavior,coherence}.json.
Criterion at each dose, as in S5b: |effect| at least twice the largest random-direction effect at that dose and at
least 3 standard errors. Judged instruments use gemma-4-31B-it as the headline; whether Qwen3.5-27B agrees is recorded.
Per concept and direction (doing+, doing-, knowing+, knowing-): effect by dose, the first dose that passes, and the
coherence of the steered replies at that dose (free-text instruments) or the unparsed share (choice and exact).
Writes results/s5b_dose/report.json and prints the summary.

    python -m cm.s5_dose_report
"""
from __future__ import annotations

import json

import numpy as np

from .paths import RESULTS

DOSES = [1, 2, 4, 8]
DIRS = ["doing+", "doing-", "knowing+", "knowing-"]


def load(m):
    base = RESULTS / "s5b" if m == 1 else RESULTS / "s5b_dose" / f"x{m}"
    B = json.loads((base / "behavior.json").read_text())
    C = json.loads((base / "coherence.json").read_text()) if (base / "coherence.json").exists() else {}
    return B, C


def passes(g, c):
    rand = [g[k]["effect"] for k in g if k.startswith("random") and g[k]["effect"] == g[k]["effect"]]
    v = g.get(c)
    if not v or not rand or v["effect"] != v["effect"] or v["se"] != v["se"]:
        return False, None
    rmax = max(abs(x) for x in rand)
    return (abs(v["effect"]) >= 2 * rmax and abs(v["effect"]) >= 3 * max(v["se"], 1e-9)), rmax


def main():
    data = {m: load(m) for m in DOSES}
    concepts = sorted(data[1][0])
    rep, counts = {}, {m: {"doing": set(), "knowing": set()} for m in DOSES}
    coh_summary = {m: {"baseline": [], "steered": [], "random": []} for m in DOSES}
    for cid in concepts:
        r = {}
        for d in DIRS:
            row = {}
            for m in DOSES:
                B, C = data[m]
                if cid not in B or "gemma" not in B[cid]["conds"] or d not in B[cid]["conds"]["gemma"]:
                    continue
                g = B[cid]["conds"]["gemma"]
                ok, rmax = passes(g, d)
                qok = None
                if "qwen" in B[cid]["conds"] and d in B[cid]["conds"]["qwen"]:
                    qok, _ = passes(B[cid]["conds"]["qwen"], d)
                row[m] = {"effect": g[d]["effect"], "se": g[d]["se"], "max_abs_random": rmax, "passes": ok,
                          "qwen_passes": qok, "unparsed": g[d]["unparsed"], "coherence": C.get(cid, {}).get(d)}
                if ok:
                    counts[m][d[:-1].rstrip("+-")].add(cid)
            if row:
                first = next((m for m in DOSES if row.get(m, {}).get("passes")), None)
                r[d] = {"by_dose": row, "first_passing_dose": first}
        rep[cid] = {"type": data[1][0][cid]["type"], "baseline": data[1][0][cid]["conds"]["gemma"]["baseline"]["mean"], "dirs": r}
        for m in DOSES:
            C = data[m][1].get(cid, {})
            if C:
                coh_summary[m]["baseline"].append(data[1][1].get(cid, {}).get("baseline", np.nan))
                coh_summary[m]["steered"] += [C[d] for d in DIRS if d in C]
                coh_summary[m]["random"] += [v for k, v in C.items() if k.startswith("random")]
    summary = {}
    for m in DOSES:
        summary[m] = {"concepts_passing_doing": sorted(counts[m]["doing"]), "concepts_passing_knowing": sorted(counts[m]["knowing"]),
                      "median_coherence_steered": float(np.nanmedian(coh_summary[m]["steered"])) if coh_summary[m]["steered"] else None,
                      "median_coherence_random": float(np.nanmedian(coh_summary[m]["random"])) if coh_summary[m]["random"] else None,
                      "median_coherence_baseline": float(np.nanmedian(coh_summary[m]["baseline"])) if coh_summary[m]["baseline"] else None}
    out = {"summary": summary, "concepts": rep}
    (RESULTS / "s5b_dose").mkdir(parents=True, exist_ok=True)
    (RESULTS / "s5b_dose" / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print("dose  coherence (baseline / steered / random)  | concepts passing: doing | knowing")
    for m in DOSES:
        s = summary[m]
        f = lambda v: "  -  " if v is None else f"{v:5.1f}"
        print(f"{m}x    {f(s['median_coherence_baseline'])} / {f(s['median_coherence_steered'])} / {f(s['median_coherence_random'])}"
              f"   | {len(s['concepts_passing_doing'])} {','.join(s['concepts_passing_doing'])} | "
              f"{len(s['concepts_passing_knowing'])} {','.join(s['concepts_passing_knowing'])}")
    print("\nper concept and direction that passes at some dose: effect by dose (P = passes; q = Qwen also passes)")
    for cid, r in rep.items():
        for d, v in r["dirs"].items():
            if v["first_passing_dose"] is None:
                continue
            cells = []
            for m in DOSES:
                x = v["by_dose"].get(m)
                if not x:
                    cells.append(f"{m}x  -   ")
                    continue
                mark = ("P" if x["passes"] else " ") + ("q" if x["qwen_passes"] else " ")
                coh = f"c{x['coherence']:.0f}" if x["coherence"] is not None else f"u{x['unparsed']:.2f}"
                cells.append(f"{m}x {x['effect']:+.3f}{mark} {coh}")
            print(f"  {cid:4s} {d:9s} base {r['baseline']:.2f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
