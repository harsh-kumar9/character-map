"""Frame d before (endorse form, results/s3_v1_endorse) and after (self-report form, results/s3) the D-030(b) rebuild.

Per concept and sub-contrast with frame d, at the four SAE depths: frame-d within AUROC on the reply32 readout (common
to both forms), transfer from d to the other frames and from the other frames to d (reply32), and frame-d held-out.
Writes results/s3/dform_compare.json and prints layer 17.

    python -m cm.s3_dform_compare
"""
from __future__ import annotations

import json

from .paths import RESULTS
from .s3_ctrl_compare import DEPTHS, summarize


def main():
    rows = []
    old_dir, new_dir = RESULTS / "s3_v1_endorse" / "concepts", RESULTS / "s3" / "concepts"
    for p in sorted(new_dir.glob("*.json")):
        if not (old_dir / p.name).exists():
            continue
        new, old = json.loads(p.read_text()), json.loads((old_dir / p.name).read_text())
        if not any(c["frame"] == "d" for c in new["cells"]):
            continue
        (ow, ot), (nw, nt) = summarize(old), summarize(new)
        for sub in new["subcontrasts"]:
            for L, idx in DEPTHS.items():
                row = {"concept": p.stem, "sub": sub, "layer": L}
                for tag, w, t in (("old", ow, ot), ("new", nw, nt)):
                    row[f"within_{tag}"] = w.get((sub, "reply32", idx), {}).get("within")
                    row[f"heldout_{tag}"] = w.get((sub, "reply32", idx), {}).get("heldout")
                    for k in ("from_d", "to_d"):
                        v = t.get((sub, k, idx))
                        row[f"{k}_{tag}"] = sum(v) / len(v) if v else None
                rows.append(row)
    (RESULTS / "s3" / "dform_compare.json").write_text(json.dumps(rows, indent=1))
    f = lambda v: "  -  " if v is None else f"{v:.2f}"
    print("layer 17, reply32 | d within old new | d held-out old new | d->others old new | others->d old new")
    for r in rows:
        if r["layer"] == 17:
            print(f"{r['concept']:5s} {r['sub'][:16]:16s} | {f(r['within_old'])} {f(r['within_new'])}     | "
                  f"{f(r['heldout_old'])} {f(r['heldout_new'])}       | {f(r['from_d_old'])} {f(r['from_d_new'])}      | "
                  f"{f(r['to_d_old'])} {f(r['to_d_new'])}")


if __name__ == "__main__":
    main()
