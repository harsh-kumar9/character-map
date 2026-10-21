"""Compare frame-d results with and without the endorsement control (D-030a).

Reads results/s3/concepts and results/s3_endorse_ctrl/concepts. For each concept and sub-contrast with frame d, at the
four SAE depths: frame-d within AUROC (boundary and reply32), transfer from d to the other frames and from the other
frames to d (reply32), and frame-d held-out AUROC. Writes results/s3_endorse_ctrl/compare.json and prints layer 17.

    python -m cm.s3_ctrl_compare
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .paths import RESULTS

DEPTHS = {9: 10, 17: 18, 22: 23, 29: 30}


def summarize(r):
    out = defaultdict(dict)
    for c in r["cells"]:
        if c["frame"] == "d":
            out[(c["sub"], c["readout"], c["layer"])]["within"] = c["auroc_dm"]
            if c.get("auroc_heldout") is not None:
                out[(c["sub"], c["readout"], c["layer"])]["heldout"] = c["auroc_heldout"]
    tr = defaultdict(list)
    for t in r["transfer"]:
        if (t["from"] == "d") != (t["to"] == "d"):
            tr[(t["sub"], "from_d" if t["from"] == "d" else "to_d", t["layer"])].append(t["auroc"])
    return out, tr


def main():
    rows = []
    for p in sorted((RESULTS / "s3_endorse_ctrl" / "concepts").glob("*.json")):
        cid = p.stem
        base, ctrl = json.loads((RESULTS / "s3" / "concepts" / p.name).read_text()), json.loads(p.read_text())
        (bw, bt), (cw, ct) = summarize(base), summarize(ctrl)
        for sub in ctrl["subcontrasts"]:
            for L, idx in DEPTHS.items():
                row = {"concept": cid, "sub": sub, "layer": L}
                for ro in ("boundary", "reply32"):
                    for k in ("within", "heldout"):
                        row[f"{ro}_{k}_before"] = bw.get((sub, ro, idx), {}).get(k)
                        row[f"{ro}_{k}_after"] = cw.get((sub, ro, idx), {}).get(k)
                for k in ("from_d", "to_d"):
                    b, a = bt.get((sub, k, idx)), ct.get((sub, k, idx))
                    row[f"{k}_before"] = float(np.mean(b)) if b else None
                    row[f"{k}_after"] = float(np.mean(a)) if a else None
                rows.append(row)
    (RESULTS / "s3_endorse_ctrl" / "compare.json").write_text(json.dumps(rows, indent=1))
    f = lambda v: "  -  " if v is None else f"{v:.2f}"
    print("layer 17 | d within boundary | d within reply32 | d held-out reply32 | transfer d->others | others->d")
    print("concept sub              | before after      | before after     | before after       | before after       | before after")
    for r in rows:
        if r["layer"] == 17:
            print(f"{r['concept']:6s} {r['sub'][:16]:16s} | {f(r['boundary_within_before'])} {f(r['boundary_within_after'])}        | "
                  f"{f(r['reply32_within_before'])} {f(r['reply32_within_after'])}       | {f(r['reply32_heldout_before'])} "
                  f"{f(r['reply32_heldout_after'])}         | {f(r['from_d_before'])} {f(r['from_d_after'])}         | "
                  f"{f(r['to_d_before'])} {f(r['to_d_after'])}")


if __name__ == "__main__":
    main()
