"""Item-level landscape and per-unit readouts of one HHH block (D-055), the honesty landscape's method
(cm/honesty_landscape.py) for a block from cm/hhh/blocks.py, plus the readouts the block figures need for units S3
does not have (C15 on harmful or harmless requests only).

Items: every train and test item of the block's units in every frame, plus the formality control. Activations as S4:
reply32, Gemma's verdict direction projected out of frame a/b rows, every hidden dimension standardized over the
pooled items. A unit is one (definition, frame) with at least 10 train and 3 test items per pole; its direction is
the diff of means oriented to the good pole (the helpful or harmless side, `good` in blocks.py), so every arrow and
every filled mark is that side.
Per depth:
  within      test AUROC of the unit's own direction (good = 1)
  transfer    frame i's direction on frame j's test items, same definition, for every ordered pair; and the same for
              the control, as the baseline
  landscape   the block's directions (control excluded) span a subspace; items projected into it, PCA, as is and with
              each unit's mean removed; the first three components exported for the page's 3D view
  stats       on the first three components, in within-unit spreads: frame-group separation (own reply a+b, act, self),
              definition separation inside each frame and group, the good-vs-other gap of every unit
  check       cosines between these directions and the HHH S4 ones (results/hhh/s4), signs matched
Block "hhh" is the three blocks together (honesty read only), for the joint page.

    python -m cm.hhh.landscape --block helpfulness|harmlessness|hhh      CPU
Writes results/hhh/figures/<block>/landscape.json, landscape_points.npz, landscape_3d.json.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from ..s3 import auroc
from ..s3_nuisance import items_by_id
from ..s4_overlap import DEPTHS, LIDX, concept_data, unit_vec
from .blocks import BLOCKS, CONTROL, available, out_dir

MIN_TRAIN, MIN_TEST = 10, 3
FRG = {"a": "own", "b": "own", "c": "act", "d": "self"}


def block_units(block):
    if block == "hhh":
        return [(b, u) for b in ("honesty", "helpfulness", "harmlessness") for u in BLOCKS[b]]
    return [(block, u) for u in BLOCKS[block]]


def gather(units):
    """Item rows (float32, all four depths) and their labels; one concept_data call per concept."""
    by_cid = {}
    for b, u in units:
        by_cid.setdefault(u.cid, []).append((b, u))
    items, Xs = [], []
    for cid, us in by_cid.items():
        X, meta = concept_data(cid, "verdict")
        im = items_by_id(cid) if any(u.where for _, u in us) else {}
        for i, m in enumerate(meta):
            if m["split"] not in ("train", "test"):
                continue
            for b, u in us:
                if (m["contrast"] or "all") != u.sub:
                    continue
                if u.where:
                    imeta = (im.get(m["item_id"]) or {}).get("meta") or {}
                    if not all(imeta.get(k) == v for k, v in u.where):
                        continue
                items.append({"unit": u.key, "name": u.name, "block": b, "group": u.group, "frame": m["frame"],
                              "good": m["polarity"] == u.good, "split": m["split"], "scenario": m["scenario_id"],
                              "row": len(Xs)})
                Xs.append(X[i].astype(np.float32))
        del X
    return items, np.stack(Xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=["helpfulness", "harmlessness", "hhh"])
    args = ap.parse_args()
    units = block_units(args.block) + [("control", CONTROL)]
    have = {u.cid for u in available([u for _, u in units])}
    missing = sorted({u.key for _, u in units if u.cid not in have})
    if missing:
        print(f"[hhh land {args.block}] no activations yet, left out: {', '.join(missing)}", flush=True)
    units = [(b, u) for b, u in units if u.cid in have]
    items, X = gather(units)
    out = out_dir() / "figures" / args.block
    out.mkdir(parents=True, exist_ok=True)
    s4p = out_dir() / "s4" / "overlap_verdict.json"
    s4 = json.loads(s4p.read_text()) if s4p.exists() else None
    s4_idx = {u: i for i, u in enumerate(s4["units"])} if s4 else {}
    good_pole = {u.key: u.good for _, u in units}
    uid = np.array([f"{it['unit']}|{it['frame']}" for it in items])
    good = np.array([it["good"] for it in items])
    split = np.array([it["split"] for it in items])
    is_ctrl = np.array([it["block"] == "control" for it in items])
    frame = np.array([it["frame"] for it in items])
    fgrp = np.array([FRG[f] for f in frame])
    summary = {"block": args.block, "n_items": len(items), "units": [], "layers": {}}
    export = {"meta": [{"unit": it["unit"], "name": it["name"], "block": it["block"], "frame": it["frame"],
                        "good": bool(it["good"])} for it in items], "views": {}}
    pts = {}
    for L, li in DEPTHS.items():
        Z = X[:, LIDX.index(li)].astype(np.float64)
        Zs = Z / (Z.std(0) + 1e-6)
        U = []
        for u in sorted(set(uid), key=lambda s: (s.split("|")[0] == CONTROL.cid, s)):
            m = uid == u
            tr, te = m & (split == "train"), m & (split == "test")
            if min((tr & good).sum(), (tr & ~good).sum()) < MIN_TRAIN or min((te & good).sum(), (te & ~good).sum()) < MIN_TEST:
                continue
            d = Zs[tr & good].mean(0) - Zs[tr & ~good].mean(0)
            U.append({"id": u, "d": unit_vec(d), "ctrl": u.startswith(CONTROL.cid + "|"),
                      "within": float(auroc(Zs[te] @ d, good[te].astype(int)))})
        ids = [x["id"] for x in U]
        # transfer across frames within a definition (and the control's, as the baseline)
        transfer = {}
        for a in U:
            for b in U:
                ka, kb = a["id"].rsplit("|", 1), b["id"].rsplit("|", 1)
                if ka[0] != kb[0] or ka[1] == kb[1]:
                    continue
                te = (uid == b["id"]) & (split == "test")
                transfer[f"{a['id']}->{kb[1]}"] = float(auroc(Zs[te] @ a["d"], good[te].astype(int)))
        base = [v for k, v in transfer.items() if k.startswith(CONTROL.cid + "|")]
        # landscape: the block's directions, PCA of the items projected into their span
        blk = [x for x in U if not x["ctrl"]]
        D = np.stack([x["d"] for x in blk])
        Q, _ = np.linalg.qr(D.T)
        P = (Zs - Zs.mean(0)) @ Q
        Pc = P - P.mean(0)
        _, S, Vt = np.linalg.svd(Pc, full_matrices=False)
        var = S ** 2 / (S ** 2).sum()
        Pw = P.copy()
        for u in np.unique(uid):
            Pw[uid == u] -= Pw[uid == u].mean(0)
        _, Sw, Vtw = np.linalg.svd(Pw, full_matrices=False)
        varw = Sw ** 2 / (Sw ** 2).sum()
        C3, Cw3 = Pc @ Vt[:3].T, Pw @ Vtw[:3].T
        axes_d = (Q @ Vt[:3].T).T
        loads = {x["id"]: [float(axes_d[k] @ x["d"]) for k in range(3)] for x in blk}
        # statistics on the first three components, in within-unit spreads (block items only)
        hb = ~is_ctrl
        wsd = float(np.sqrt(np.mean([C3[uid == u].var(0).sum() for u in np.unique(uid[hb])])))
        gm = {g: C3[hb & (fgrp == g)].mean(0) for g in ("own", "act", "self") if (hb & (fgrp == g)).any()}
        frame_sep = {f"{a}-{b}": float(np.linalg.norm(gm[a] - gm[b]) / wsd) for a, b in itertools.combinations(gm, 2)}
        fm = {f: C3[hb & (frame == f)].mean(0) for f in "abcd" if (hb & (frame == f)).any()}
        frame_sep.update({f"{a}-{b}": float(np.linalg.norm(fm[a] - fm[b]) / wsd) for a, b in itertools.combinations(fm, 2)})
        unit_of = np.array([it["unit"] for it in items])
        def_sep = {}
        for f in "abcd":
            cs = {k: C3[hb & (frame == f) & (unit_of == k)].mean(0) for k in dict.fromkeys(unit_of[hb])
                  if (hb & (frame == f) & (unit_of == k)).any()}
            def_sep[f] = {f"{a}~{b}": float(np.linalg.norm(cs[a] - cs[b]) / wsd) for a, b in itertools.combinations(cs, 2)}
        gap = {x["id"]: float(np.linalg.norm(C3[(uid == x["id"]) & good].mean(0) - C3[(uid == x["id"]) & ~good].mean(0)) / wsd)
               for x in blk}
        # consistency with the HHH S4 (S4 directions point to pos; flip the units whose good pole is neg)
        consist = None
        if s4:
            sh = [x for x in blk if x["id"] in s4_idx]
            sgn = np.array([1.0 if good_pole[x["id"].rsplit("|", 1)[0]] == "pos" else -1.0 for x in sh])
            here = np.stack([x["d"] for x in sh]) @ np.stack([x["d"] for x in sh]).T
            there = np.array(s4["layers"][str(L)]["raw"])[np.ix_([s4_idx[x["id"]] for x in sh], [s4_idx[x["id"]] for x in sh])]
            there = there * np.outer(sgn, sgn)
            off = ~np.eye(len(sh), dtype=bool)
            consist = {"n_units": len(sh), "r": float(np.corrcoef(here[off], there[off])[0, 1]),
                       "max_abs_diff": float(np.abs(here - there)[off].max())}
        centroids = lambda C: {u: [C[(uid == u) & ~good].mean(0).round(3).tolist(), C[(uid == u) & good].mean(0).round(3).tolist()]
                               for u in np.unique(uid) if ((uid == u) & good).any() and ((uid == u) & ~good).any()}
        export["views"][f"as-is, layer {L}"] = {"xyz": C3.round(2).tolist(), "var": var[:3].round(4).tolist(), "centroids": centroids(C3)}
        export["views"][f"each unit centred, layer {L}"] = {"xyz": Cw3.round(2).tolist(), "var": varw[:3].round(4).tolist(),
                                                           "centroids": centroids(Cw3)}
        pts[f"P{L}"] = Pc.astype(np.float32); pts[f"Vt{L}"] = Vt.astype(np.float32)
        pts[f"Pw{L}"] = Pw.astype(np.float32); pts[f"Vtw{L}"] = Vtw.astype(np.float32)
        summary["layers"][str(L)] = {
            "units": ids, "within": {x["id"]: x["within"] for x in U}, "transfer": transfer,
            "control_transfer": float(np.mean(base)) if base else None, "direction_ids": [x["id"] for x in blk],
            "variance": var.round(4).tolist(), "variance_within": varw.round(4).tolist(), "loadings": loads,
            "within_unit_spread": wsd, "frame_separation": frame_sep, "definition_separation": def_sep, "good_gap": gap,
            "consistency_with_s4": consist}
        print(f"[hhh land {args.block}] layer {L}: {len(blk)} directions, {len(items)} items, PC1-3 "
              f"{var[0]:.2f} {var[1]:.2f} {var[2]:.2f}; frames apart " +
              " ".join(f"{k} {v:.1f}" for k, v in frame_sep.items()) +
              (f"; S4 check r {consist['r']:.3f} max|diff| {consist['max_abs_diff']:.2f}" if consist else ""), flush=True)
        for x in U:
            t = {k.split("->")[1]: v for k, v in transfer.items() if k.startswith(x["id"] + "->")}
            print(f"   {x['id']:32s} within {x['within']:.2f}  transfer " + " ".join(f"{k} {v:.2f}" for k, v in sorted(t.items()))
                  + (f"  gap {gap[x['id']]:.1f}" if x["id"] in gap else ""))
    summary["units"] = [{"key": u.key, "name": u.name, "block": b, "group": u.group, "good": u.good} for b, u in units]
    (out / "landscape.json").write_text(json.dumps(summary, indent=1))
    (out / "landscape_3d.json").write_text(json.dumps(export, separators=(",", ":")))
    np.savez(out / "landscape_points.npz", meta=np.array(json.dumps(export["meta"])), **pts)
    print(f"[hhh land {args.block}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
