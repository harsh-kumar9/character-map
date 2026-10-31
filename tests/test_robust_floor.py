"""D-064 amendment 1 ruling (d): the standardization floor is off by default and changes nothing when off.

Unit level (synthetic rows; needs numpy, scipy, sklearn): floor_sd(sd, None) is sd itself and floor_sd(sd, q) is
max(sd, quantile(sd, q)); pool_dirs and within_cell with q = None equal the unfloored formulas to the bit; rep_depth
accepts the old (cells, l) argument and gives the same numbers as (cells, l, None); parse_qs and qtag.
Data level (ada, after the robust_floor/none/ runs): `CM_MODEL=<model> python -m cm.robust_floor check` compares
coherence/robust_floor/none/{representational.json, levers.npz} and battery/onpolicy/robust_floor/none/{directions.npz,
baseline/auroc_strat.json} with the baseline files to 1e-6 (results/<tag>/coherence/robust_floor/identity_check.json).

    python tests/test_robust_floor.py          (or python -m pytest tests/test_robust_floor.py -q)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # run as a plain script from anywhere
from cm import coherence as C  # noqa: E402


def synth_cells(d=40, n_tr=30, n_te=12, n_depth=2, seed=0):
    rng = np.random.default_rng(seed)
    cells = {}
    for c in C.ALL:
        u = rng.standard_normal(d)
        scale = np.exp(rng.uniform(-3, 1, d))          # a spread of per-dimension sd, as the subjects have

        def rows(n):
            y = np.arange(n) % 2
            X = rng.standard_normal((n, n_depth, d)) * scale + (y[:, None, None] - 0.5) * u * 0.8
            return X.astype(np.float32), y

        Xtr, ytr = rows(n_tr)
        Xte, yte = rows(n_te)
        cells[c] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}
    return cells


def test_floor_sd():
    sd = np.exp(np.random.default_rng(1).uniform(-4, 1, 1000))
    assert C.floor_sd(sd) is sd and C.floor_sd(sd, None) is sd
    f = C.floor_sd(sd, 0.25)
    assert np.array_equal(f, np.maximum(sd, np.quantile(sd, 0.25)))
    assert abs((sd < np.quantile(sd, 0.25)).mean() - 0.25) < 0.01 and f.min() == np.quantile(sd, 0.25)
    assert C.parse_qs(None) is None and C.parse_qs("none,0.1,0.25,0.5") == [None, 0.1, 0.25, 0.5]
    assert [C.qtag(q) for q in (None, 0.1, 0.25, 0.5)] == ["none", "q0.1", "q0.25", "q0.5"]


def test_pool_dirs_identity():
    cells = synth_cells()
    ids = C.HON[:5]
    P = C.pool_dirs(cells, ids, 0)
    Xp = np.concatenate([cells[c]["Xtr"][:, 0] for c in ids]).astype(np.float64)
    sd = Xp.std(0) + 1e-6
    assert np.array_equal(P["sd"], sd) and np.array_equal(P["sd_raw"], sd)
    Xs = Xp / sd
    y = np.concatenate([cells[c]["ytr"] for c in ids])
    cell_of = np.concatenate([np.full(len(cells[c]["ytr"]), k) for k, c in enumerate(ids)])
    per = [Xs[(cell_of == k) & (y == 1)].mean(0) - Xs[(cell_of == k) & (y == 0)].mean(0) for k in range(len(ids))]
    assert np.array_equal(P["dm"], C.unit_vec(np.mean(per, 0)))
    Pq = C.pool_dirs(cells, ids, 0, q=0.5)
    assert np.array_equal(Pq["sd"], np.maximum(sd, np.quantile(sd, 0.5))) and np.array_equal(Pq["sd_raw"], sd)
    assert not np.allclose(Pq["dm"], P["dm"])
    assert C.within_cell(cells, ids[0], 0) == C.within_cell(cells, ids[0], 0, q=None)


def test_rep_depth_old_args():
    cells = synth_cells()
    _, r0 = C.rep_depth((cells, 0))
    _, r1 = C.rep_depth((cells, 0, None))
    assert json.dumps(r0["cells"], sort_keys=True) == json.dumps(r1["cells"], sort_keys=True)
    assert r0["pairwise"] == r1["pairwise"]
    assert r1["sd"]["q"] is None and r1["sd"]["n_raised"] == 0 and r1["sd"]["floor"] is None
    _, r2 = C.rep_depth((cells, 0, 0.5))
    assert r2["sd"]["n_raised"] > 0 and abs(r2["sd"]["floor"] - r2["sd"]["median"]) < 1e-12
    assert json.dumps(r2["cells"], sort_keys=True) != json.dumps(r0["cells"], sort_keys=True)


if __name__ == "__main__":
    for t in (test_floor_sd, test_pool_dirs_identity, test_rep_depth_old_args):
        t()
        print("ok", t.__name__)
