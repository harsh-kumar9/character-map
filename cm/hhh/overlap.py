"""S4 for the HHH line (D-055), written to results/hhh/s4 so the honesty S4 does not move (ruling of 2026-10-24).

Same units and measures as cm/s4_overlap.py: every (concept, sub-contrast, frame) of every concept with activations,
built by S4's own concept_units, so a unit that S4 has gets the same direction, halves and shuffles. Added: the block
units with an item-meta filter (C15 refuse on harmful and on harmless requests, D-054's 2x2), built the same way on
the filtered items, with ids like C15|refuse[harmful]|c. The pool now holds the new concepts too, so the standardizing
SD and the centring mean differ from the honesty S4; the run prints how the cosines of the units both runs share
compare.
Per depth: centred cosine, RSA, raw standardized cosine, Spearman-Brown reliabilities, the two floors. No taxonomy
tests here; the HHH tests (tension, D1 projection, shared nodes) read this file.

    python -m cm.hhh.overlap [--variant verdict|verdict+negation] [--workers 12]     CPU
Writes results/hhh/s4/units.json and results/hhh/s4/overlap_<variant>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from .. import cfg
from ..s3_nuisance import all_concepts, items_by_id
from ..s4_overlap import (DEPTHS, LIDX, MIN_TEST, MIN_TRAIN, PROBE, R_RAND, R_SPLIT, concept_data, concept_units,
                          corr_rows, sb, unit_vec)
from .blocks import BLOCKS, out_dir


def filtered_units(cid, specs, variant):
    """S4's unit construction on the items that pass an item-meta filter; specs = [(sub, where), ...]."""
    X, meta = concept_data(cid, variant)
    im = items_by_id(cid)
    imeta = [(im.get(m["item_id"]) or {}).get("meta") or {} for m in meta]
    out = []
    for sub, where in specs:
        tag = "".join(f"[{v}]" for _, v in where)
        rng = np.random.default_rng(int(hashlib.sha256(f"{cid}|{sub}{tag}".encode()).hexdigest()[:8], 16))
        for f in sorted({m["frame"] for m in meta}):
            sel = [i for i, m in enumerate(meta) if (m["contrast"] or "all") == sub and m["frame"] == f
                   and all(imeta[i].get(k) == v for k, v in where)]
            tr = [i for i in sel if meta[i]["split"] == "train"]
            te = [i for i in sel if meta[i]["split"] == "test"]
            pos = [i for i in tr if meta[i]["polarity"] == "pos"]
            neg = [i for i in tr if meta[i]["polarity"] == "neg"]
            tpos = [i for i in te if meta[i]["polarity"] == "pos"]
            tneg = [i for i in te if meta[i]["polarity"] == "neg"]
            if min(len(pos), len(neg)) < MIN_TRAIN or min(len(tpos), len(tneg)) < MIN_TEST:
                continue
            d = X[pos].mean(0) - X[neg].mean(0)
            scen = sorted({meta[i]["scenario_id"] for i in tr})
            halves = []
            for _ in range(R_SPLIT):
                perm = rng.permutation(len(scen))
                hs = []
                for part in (perm[: len(scen) // 2], perm[len(scen) // 2:]):
                    ss = {scen[k] for k in part}
                    p = [i for i in pos if meta[i]["scenario_id"] in ss]
                    n = [i for i in neg if meta[i]["scenario_id"] in ss]
                    hs.append(X[p].mean(0) - X[n].mean(0) if p and n else np.zeros_like(d))
                halves.append(hs)
            rands = []
            for _ in range(R_RAND):
                flip = {s: rng.random() < 0.5 for s in scen}
                lab = [(meta[i]["polarity"] == "pos") != flip[meta[i]["scenario_id"]] for i in tr]
                rands.append(X[[i for i, l in zip(tr, lab) if l]].mean(0) - X[[i for i, l in zip(tr, lab) if not l]].mean(0))
            probe = list(rng.choice(tpos, min(PROBE, len(tpos)), replace=False)) + \
                list(rng.choice(tneg, min(PROBE, len(tneg)), replace=False))
            out.append({"id": f"{cid}|{sub}{tag}|{f}", "concept": cid, "sub": f"{sub}{tag}", "frame": f,
                        "n_train": [len(pos), len(neg)], "d": d.astype(np.float32),
                        "halves": np.array(halves, np.float32), "rands": np.array(rands, np.float32),
                        "probe": X[probe].astype(np.float32)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="verdict", choices=["verdict", "verdict+negation"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cids = all_concepts()
    flt = {}
    for units in BLOCKS.values():
        for u in units:
            if u.where and u.cid in cids:
                flt.setdefault(u.cid, []).append((u.sub, u.where))
    units = []
    with ProcessPoolExecutor(args.workers) as ex:
        for us in ex.map(partial(concept_units, variant=args.variant), cids):
            units += us
        for us in ex.map(partial(_filtered, variant=args.variant), list(flt.items())):
            units += us
    U = len(units)
    ids = [u["id"] for u in units]
    D = np.stack([u["d"] for u in units]).astype(np.float64)
    H = np.stack([u["halves"] for u in units]).astype(np.float64)
    Rn = np.stack([u["rands"] for u in units]).astype(np.float64)
    P = np.concatenate([u["probe"] for u in units]).astype(np.float64)
    src = np.concatenate([np.full(len(u["probe"]), k) for k, u in enumerate(units)])
    print(f"[hhh s4] {U} units ({sum('[' in i for i in ids)} filtered), probe {len(P)} items", flush=True)
    res = {"variant": args.variant, "units": ids, "layers": {}}
    old = cfg.results_dir() / "s4" / f"overlap_{args.variant.replace('+', '_')}.json"
    old = json.loads(old.read_text()) if old.exists() else None
    for L, li in DEPTHS.items():
        l = LIDX.index(li)
        sd = P[:, l].std(0) + 1e-6
        Dn = unit_vec(D[:, l] / sd)
        mean = Dn.mean(0)
        C = unit_vec(Dn - mean)
        Hc = unit_vec(unit_vec(H[:, :, :, l] / sd) - mean)
        Rn_l = unit_vec(Rn[:, :, l] / sd)
        Rc = unit_vec(Rn_l - Rn_l.reshape(-1, Rn_l.shape[-1]).mean(0))
        cos = C @ C.T
        rel_cos = sb(np.einsum("urd,urd->ur", Hc[:, :, 0], Hc[:, :, 1]).mean(1))
        mask = ~np.eye(U, dtype=bool)
        floor_cos = float(np.percentile(np.abs(np.einsum("ud,vsd->uvs", C, Rc))[mask], 95))
        PL = (P[:, l] - P[:, l].mean(0)) / sd
        for k in range(U):
            PL[src == k] -= PL[src == k].mean(0)
        pat = Dn @ PL.T
        rsa = corr_rows(pat, pat)
        ph0 = np.einsum("urd,nd->urn", unit_vec(H[:, :, 0, l] / sd), PL)
        ph1 = np.einsum("urd,nd->urn", unit_vec(H[:, :, 1, l] / sd), PL)
        rel_rsa = sb(np.array([np.mean([corr_rows(ph0[u, r][None], ph1[u, r][None])[0, 0] for r in range(R_SPLIT)])
                               for u in range(U)]))
        prand = np.einsum("usd,nd->usn", Rn_l, PL).reshape(U * R_RAND, -1)
        floor_rsa = float(np.percentile(np.abs(corr_rows(pat, prand)).reshape(U, U, R_RAND)[mask], 95))
        raw = Dn @ Dn.T
        res["layers"][str(L)] = {"cos": np.round(cos, 4).tolist(), "rsa": np.round(rsa, 4).tolist(),
                                 "raw": np.round(raw, 4).tolist(), "rel_cos": np.round(rel_cos, 4).tolist(),
                                 "rel_rsa": np.round(rel_rsa, 4).tolist(), "floor_cos": floor_cos, "floor_rsa": floor_rsa}
        msg = f"[hhh s4] layer {L}: floors cos {floor_cos:.3f} rsa {floor_rsa:.3f}; median reliability cos {np.median(rel_cos):.2f}"
        if old:
            shared = [u for u in old["units"] if u in ids]
            a = [ids.index(u) for u in shared]
            b = [old["units"].index(u) for u in shared]
            iu = np.triu_indices(len(shared), 1)
            o = np.array(old["layers"][str(L)]["cos"])[np.ix_(b, b)][iu]
            n = cos[np.ix_(a, a)][iu]
            msg += (f"; vs the honesty S4 on {len(shared)} shared units: r {np.corrcoef(o, n)[0, 1]:.3f}, "
                    f"median |diff| {np.median(np.abs(o - n)):.3f}, floor then {old['layers'][str(L)]['floor_cos']:.3f}")
        print(msg, flush=True)
    d = out_dir() / "s4"
    d.mkdir(parents=True, exist_ok=True)
    (d / "units.json").write_text(json.dumps([{"id": u["id"], "concept": u["concept"], "sub": u["sub"], "frame": u["frame"],
                                               "n_train": u["n_train"]} for u in units], indent=1))
    (d / f"overlap_{args.variant.replace('+', '_')}.json").write_text(json.dumps(res))


def _filtered(item, variant):
    cid, specs = item
    return filtered_units(cid, specs, variant)


if __name__ == "__main__":
    main()
