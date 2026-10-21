"""Cross-frame claims under the criteria fixed in D-033 (docs/DECISIONS.md, "Rulings D-033 to D-036").

For every concept, sub-contrast, unordered pair of frames and SAE depth, transfer is the mean of the two directions
(frame i's direction on frame j's test items and the reverse), under four readouts:
  reply32      results/s3                      the headline (D-028, D-029)
  preverdict   results/s3 transfer_boundary    frames read at the boundary scored before Gemma's verdict
  verdict      results/s3_ctrl_verdict         Gemma's verdict direction projected out of a/b reply32 (D-033c)
  negation     results/s3_ctrl_negation        a negation direction projected out of every row (D-036)
Each is compared with the control baseline under the same readout and depth: the mean cross-frame transfer of K1
(formality) and K2 (topic). A pair is
  shared         when it beats the baseline by at least 0.10 under reply32, preverdict and verdict, and both frames'
                 within-frame reply32 directions pass BH at 0.05 at that depth (results/s3/summary.json);
  dissociated    when transfer is below 0.40 under all three;
  readout-dependent otherwise.
Pairs that involve frame a or b need the verdict control; until it exists they are marked provisional. The negation
control is reported as a separate flag (negation_robust: still beats the baseline by 0.10 with negation projected out),
not folded into the class, so the fixed criteria stay as ruled.

    python -m cm.s3_claims          writes results/s3/claims.json and prints counts and lists per depth
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .paths import RESULTS

DEPTHS = {9: 10, 17: 18, 22: 23, 29: 30}
SOURCES = {"reply32": ("s3", "transfer"), "preverdict": ("s3", "transfer_boundary"),
           "verdict": ("s3_ctrl_verdict", "transfer"), "negation": ("s3_ctrl_negation", "transfer")}
MARGIN, LOW, Q = 0.10, 0.40, 0.05


def load_transfer(src, key):
    """{(concept, sub, from, to, layer_index): auroc}, or None when the run does not exist."""
    d = RESULTS / src / "concepts"
    if not d.exists():
        return None
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        for t in r.get(key, []):
            out[(r["concept"], t["sub"], t["from"], t["to"], t["layer"])] = t["auroc"]
    return out


def main():
    T = {name: load_transfer(*src) for name, src in SOURCES.items()}
    # reply32 everywhere a frame has no boundary direction: preverdict falls back to it, as in s3_report
    q = {}
    for s in json.loads((RESULTS / "s3" / "summary.json").read_text()):
        if s["readout"] == "reply32":
            for L in DEPTHS:
                q[(s["concept"], s["sub"], s["frame"], L)] = s.get(f"q_L{L}")
    keys = sorted({(c, s, a, b) for (c, s, a, b, _) in T["reply32"] if a < b and (c, s, b, a, _) in T["reply32"]})

    def pair(name, c, s, a, b, idx):
        src = T[name]
        if src is None:
            return None
        v = []
        for x, y in ((a, b), (b, a)):
            val = src.get((c, s, x, y, idx))
            if val is None and name == "preverdict":
                val = T["reply32"].get((c, s, x, y, idx))
            if val is None and name == "verdict" and x not in "ab" and y not in "ab":
                val = T["reply32"].get((c, s, x, y, idx))
            if val is None:
                return None
            v.append(val)
        return float(np.mean(v))

    def baseline(name, idx):
        vals = [pair(name, c, s, a, b, idx) for (c, s, a, b) in keys if c in ("K1", "K2")]
        vals = [x for x in vals if x is not None]
        return float(np.mean(vals)) if vals else None

    rows = []
    for L, idx in DEPTHS.items():
        base = {n: baseline(n, idx) for n in SOURCES}
        for (c, s, a, b) in keys:
            if c.startswith("K"):
                continue
            v = {n: pair(n, c, s, a, b, idx) for n in SOURCES}
            within_ok = all((q.get((c, s, f, L)) or 1.0) < Q for f in (a, b))
            core = ("reply32", "preverdict", "verdict")
            needs_verdict = a in "ab" or b in "ab"
            avail = [n for n in core if v[n] is not None and base[n] is not None]
            above = [v[n] - base[n] >= MARGIN for n in avail]
            below = [v[n] < LOW for n in avail]
            if all(above) and within_ok:
                cls = "shared"
            elif all(below):
                cls = "dissociated"
            else:
                cls = "readout-dependent"
            provisional = needs_verdict and v["verdict"] is None
            neg_ok = (v["negation"] is not None and base["negation"] is not None and
                      v["negation"] - base["negation"] >= MARGIN)
            rows.append({"concept": c, "sub": s, "pair": f"{a}-{b}", "layer": L, **{n: v[n] for n in SOURCES},
                         **{f"base_{n}": base[n] for n in SOURCES}, "within_bh_ok": within_ok, "class": cls,
                         "provisional": provisional, "negation_robust": neg_ok if cls == "shared" else None})
    (RESULTS / "s3" / "claims.json").write_text(json.dumps(rows, indent=1))
    f = lambda x: "  -  " if x is None else f"{x:.2f}"
    for L in DEPTHS:
        rs = [r for r in rows if r["layer"] == L]
        b = rs[0]
        cnt = defaultdict(int)
        for r in rs:
            cnt[r["class"] + (" (provisional)" if r["provisional"] else "")] += 1
        print(f"\nlayer {L}: baselines reply32 {f(b['base_reply32'])} preverdict {f(b['base_preverdict'])} "
              f"verdict {f(b['base_verdict'])} negation {f(b['base_negation'])}; " +
              ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
        for r in rs:
            if r["class"] != "readout-dependent":
                print(f"  {r['class']:11s}{'*' if r['provisional'] else ' '} {r['concept']:4s} {r['sub'][:16]:16s} "
                      f"{r['pair']}  reply32 {f(r['reply32'])} pre {f(r['preverdict'])} verdict {f(r['verdict'])} "
                      f"negation {f(r['negation'])}" + ("" if r["negation_robust"] in (None, True) else "  (not negation-robust)"))
    print("\n* provisional: involves frame a or b, verdict control not run yet")


if __name__ == "__main__":
    main()
