"""S3 report: per concept, how well each frame's contrast is read out, and whether one frame's direction reads another.

Reads results/s3/concepts/*.json (cm.s3 directions). Layer choice is fixed in advance, not picked on test data: the
four SAE depths (hidden-state index 10/18/23/30 = decoder layers 9/17/22/29). The common readout across frames is
reply32. Within-frame AUROC is the frame's own direction on its own test items; transfer AUROC is frame i's direction
on frame j's test items (0.5 = no transfer, below 0.5 = reversed). Frames read at the boundary (a, b) have reply32 =
Gemma's own reply, which opens with its verdict, so transfer_boundary and transfer_mean_preverdict also score their
boundary direction (before the verdict) on the target frame's reply32.

    python -m cm.s3_report            writes results/s3/frames_table.json and prints the table
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import cfg

RESULTS = cfg.results_dir()
S3 = RESULTS / "s3"
DEPTHS = dict(cfg.DEPTHS)
GROUP = {"A": "beliefs", "B": "moral", "C": "dispositions", "D": "identity", "E": "situations", "F": "conflicts",
         "K": "controls"}


def main():
    rows = []
    for p in sorted((S3 / "concepts").glob("*.json")):
        r = json.loads(p.read_text())
        cid = r["concept"]
        within = defaultdict(dict)   # (sub, frame) -> layer index -> auroc (reply32)
        within_sdm = defaultdict(dict)   # the standardized diff of means (D-050), where S3 computed it
        held_sdm = defaultdict(dict)
        held = defaultdict(dict)
        far = defaultdict(dict)
        for c in r["cells"]:
            if c["readout"] == "reply32":
                within[(c["sub"], c["frame"])][c["layer"]] = c["auroc_dm"]
                if c.get("auroc_sdm") is not None:
                    within_sdm[(c["sub"], c["frame"])][c["layer"]] = c["auroc_sdm"]
                if c.get("auroc_sdm_heldout") is not None:
                    held_sdm[(c["sub"], c["frame"])][c["layer"]] = c["auroc_sdm_heldout"]
                if c.get("auroc_heldout") is not None:
                    held[(c["sub"], c["frame"])][c["layer"]] = c["auroc_heldout"]
                if c.get("auroc_heldout_far") is not None:
                    far[(c["sub"], c["frame"])][c["layer"]] = c["auroc_heldout_far"]
        trans = defaultdict(dict)    # (sub, from, to) -> layer -> auroc
        for t in r["transfer"]:
            trans[(t["sub"], t["from"], t["to"])][t["layer"]] = t["auroc"]
        trans_b = defaultdict(dict)  # same, from the boundary direction of frames read at the boundary
        for t in r.get("transfer_boundary", []):
            trans_b[(t["sub"], t["from"], t["to"])][t["layer"]] = t["auroc"]
        for sub in r["subcontrasts"]:
            frames = sorted({f for (s, f) in within if s == sub})
            for L, idx in DEPTHS.items():
                w = {f: within[(sub, f)].get(idx) for f in frames}
                h = {f: held[(sub, f)].get(idx) for f in frames}
                hfar = {f: far[(sub, f)].get(idx) for f in frames if far[(sub, f)].get(idx) is not None}
                off = [trans[(sub, a, b)].get(idx) for a in frames for b in frames if a != b]
                off = [x for x in off if x is not None]
                mat = {f"{a}->{b}": trans[(sub, a, b)].get(idx) for a in frames for b in frames}
                mat_b = {f"{a}->{b}": trans_b[(sub, a, b)].get(idx) for a in frames for b in frames
                         if a != b and trans_b[(sub, a, b)].get(idx) is not None}
                # pre-verdict variant: boundary direction where the source frame has one, reply32 otherwise
                off_b = [trans_b[(sub, a, b)].get(idx, trans[(sub, a, b)].get(idx)) for a in frames for b in frames if a != b]
                off_b = [x for x in off_b if x is not None]
                rows.append({"concept": cid, "group": GROUP[cid[0]], "sub": sub, "layer": L, "frames": frames,
                             "within": w, "heldout": h, "heldout_far": hfar, "transfer": mat, "transfer_boundary": mat_b,
                             "within_sdm": {f: within_sdm[(sub, f)].get(idx) for f in frames if within_sdm[(sub, f)].get(idx) is not None},
                             "heldout_sdm": {f: held_sdm[(sub, f)].get(idx) for f in frames if held_sdm[(sub, f)].get(idx) is not None},
                             "within_mean": float(np.nanmean([x for x in w.values() if x is not None])) if w else None,
                             "transfer_mean": float(np.mean(off)) if off else None,
                             "transfer_mean_preverdict": float(np.mean(off_b)) if off_b else None})
    (S3 / "frames_table.json").write_text(json.dumps(rows, indent=1))
    # print: one line per concept (sub "all" or each sub-contrast), layer 17 and 22
    print("concept sub              frames  | L17 within  transfer | L22 within  transfer | held-out (L17, by frame)")
    by = defaultdict(dict)
    for x in rows:
        by[(x["concept"], x["sub"])][x["layer"]] = x
    for (cid, sub), d in sorted(by.items(), key=lambda kv: ("ABCDEFK".index(kv[0][0][0]), int(kv[0][0][1:]), kv[0][1])):
        a, b = d.get(17), d.get(22)
        fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
        ho = " ".join(f"{f}:{fmt(v)}" for f, v in (a["heldout"].items() if a else []) if v is not None)
        print(f"{cid:6s} {sub[:16]:16s} {''.join(a['frames']) if a else '':6s}  |   {fmt(a and a['within_mean'])}     "
              f"{fmt(a and a['transfer_mean'])}   |   {fmt(b and b['within_mean'])}     {fmt(b and b['transfer_mean'])}   | {ho}")


if __name__ == "__main__":
    main()
