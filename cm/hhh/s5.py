"""S5a for the HHH line (D-055): the representation-level causal matrix of cm/s5_causal.py, restricted to the three
blocks (honesty read only) and the controls, written to outputs/hhh/s5 and results/hhh/s5 so the honesty S5 does not
move (ruling of 2026-10-24). Sources and targets span all three blocks, so the cross-block tests of D-055 (steer
refusal, read substance; steer substance, read refusal) are cells of the same matrix.

Same procedure as S5a: steer one direction at decoder layer 17 (every position), read every target unit's projection
at layers 22 and 29 on its own probe test items (6 per pole), effect = mean shift minus the skip-path prediction,
over the target's baseline gap, z against 16 random-direction effects. The run and matrix stages are s5_causal's own
code, pointed at these directories. Differences from the honesty run, both from scope:
  - dense directions only (+ and - dose); the SAE condition is left out (the lattice and the pages use dense+);
  - units with an item-meta filter (C15 on harmful or harmless requests) have no S3 direction, so theirs is built the
    same way S3 builds one (raw diff of means of reply32 on train items, Gemma's verdict projected out of a/b rows),
    and their probe items come from the filtered test items.
Dose: the calibrated relative dose of the existing S5 run (results/s5/calibration.json, D-039), copied, not re-chosen.

    python -m cm.hhh.s5 setup                               CPU
    python -m cm.hhh.s5 run --shard K --nshards N           GPU
    python -m cm.hhh.s5 matrix --nshards N                  CPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil

import numpy as np

from .. import cfg
from .. import s5_causal as S5
from ..s3_nuisance import items_by_id
from ..s4_overlap import LIDX, concept_data
from .blocks import BLOCKS, CONTROL, out_dir

OUT = cfg.outputs_dir() / "hhh" / "s5"
RES = out_dir() / "s5"
EXTRA = ("K1|all", "K2|all")          # formality (control) and topic (the unrelated-concept source), as in S5a


def unit_keys():
    keys = {u.key: u for units in BLOCKS.values() for u in units}
    return keys


def targets():
    """HHH S4 units whose key is a block unit or a control."""
    keys = unit_keys()
    ids = [u["id"] for u in json.loads((out_dir() / "s4" / "units.json").read_text())]
    return [i for i in ids if i.rsplit("|", 1)[0] in keys or i.rsplit("|", 1)[0] in EXTRA], keys


_DATA = {}


def filtered_direction(uid, hidx, unit):
    """Raw diff of means of reply32 at hidden index hidx on the unit's filtered train items, verdict projected out of
    a/b rows (S3's construction for the verdict-controlled set)."""
    cid = unit.cid
    if cid not in _DATA:
        X, meta = concept_data(cid, "verdict")
        im = items_by_id(cid)
        _DATA[cid] = (X, meta, [(im.get(m["item_id"]) or {}).get("meta") or {} for m in meta])
    X, meta, imeta = _DATA[cid]
    f = uid.rsplit("|", 1)[1]
    sel = [i for i, m in enumerate(meta) if (m["contrast"] or "all") == unit.sub and m["frame"] == f and m["split"] == "train"
           and all(imeta[i].get(k) == v for k, v in unit.where)]
    pos = [i for i in sel if meta[i]["polarity"] == "pos"]
    neg = [i for i in sel if meta[i]["polarity"] == "neg"]
    l = LIDX.index(hidx)
    v = X[pos, l].mean(0) - X[neg, l].mean(0)
    return v / (np.linalg.norm(v) or 1.0)


def direction(uid, hidx, keys):
    u = keys.get(uid.rsplit("|", 1)[0])
    if u is not None and u.where:
        return filtered_direction(uid, hidx, u)
    return S5.direction(uid, hidx)


def cmd_setup():
    U, keys = targets()
    by = {}
    for u in U:
        k, f = u.rsplit("|", 1)
        by.setdefault(k, []).append(f)
    sources = []
    for k, frames in sorted(by.items()):
        do = next((f for f in "cd" if f in frames), None)
        kn = next((f for f in "ba" if f in frames), None)
        sources += [f"{k}|{f}" for f in (do, kn) if f]
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((S5.N_RAND, cfg.D))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    dense = np.stack([direction(s, S5.H_IN, keys) for s in sources])
    tgt = {h: np.stack([direction(u, h, keys) for u in U]) for h in S5.H_OUT}
    probe = []
    acts = cfg.acts_dir()
    for u in U:
        k, f = u.rsplit("|", 1)
        cid, sub = k.split("|")[0], keys[k].sub if k in keys else k.split("|")[1]
        where = keys[k].where if k in keys else ()
        meta = json.loads(str(np.load(acts / f"{cid}.npz")["meta"]))
        im = items_by_id(cid) if where else {}
        te = [m for m in meta if (m["contrast"] or "all") == sub and m["frame"] == f and m["split"] == "test"
              and all(((im.get(m["item_id"]) or {}).get("meta") or {}).get(a) == b for a, b in where)]
        r = np.random.default_rng(int(hashlib.sha256(u.encode()).hexdigest()[:8], 16))
        for p in ("pos", "neg"):
            c = [m["item_id"] for m in te if m["polarity"] == p]
            for i in r.choice(c, min(S5.PROBE, len(c)), replace=False):
                probe.append({"unit": u, "item_id": str(i), "polarity": p})
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "setup.npz", dense=dense, sae=np.zeros_like(dense), rand=rand,
             tgt_out0=tgt[S5.H_OUT[0]], tgt_out1=tgt[S5.H_OUT[1]])
    src = cfg.results_dir() / "s5"
    for f in ("calibration.json", f"residual_norm_L{S5.L_IN}.json"):
        shutil.copy(src / f, RES / f)
    (RES / "setup.json").write_text(json.dumps({"sources": sources, "targets": U, "probe": probe,
                                                "dose_from": str(src / "calibration.json"), "sae": False}, indent=1))
    print(f"[hhh s5] {len(sources)} sources, {len(U)} targets, {len(probe)} probe items; "
          f"dose c {json.loads((RES / 'calibration.json').read_text())['chosen_c']}", flush=True)


def point():
    """s5_causal's run and matrix stages, on these directories, dense only."""
    S5.OUT, S5.RES = OUT, RES
    cfg.HAS_SAE = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["setup", "run", "matrix"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "setup":
        cmd_setup()
        return
    point()
    if a.stage == "run":
        S5.cmd_run(a.shard, a.nshards)
    else:
        S5.cmd_matrix(a.nshards)


if __name__ == "__main__":
    main()
