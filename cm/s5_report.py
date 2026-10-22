"""S5a report: what the causal matrix says (reads results/s5/causal_matrix.json and results/s4/overlap_verdict.json).

Per readout layer (22, 29), for the dense +dose intervention unless noted:
  self        each source's effect on its own unit (gap units and z against random directions); share with z > 3
  within      same concept, other frame: knowing source -> doing targets and doing source -> knowing targets
              (the causal counterpart of the S3 knowing-doing dissociation), and doing -> doing / knowing -> knowing
  spillover   off-concept cells with |z| > 3, the K2 (topic) source's spillover as the unrelated-concept reference,
              and the strongest off-concept effects
  geometry    Spearman between off-concept effects and the S4 dense cosine of source and target
  agreement   dense vs SAE-feature vectors (Spearman over cells; self effects of each); sign symmetry (+dose vs -dose)
  taxonomy    honesty-humility (C19, acting) -> dark-triad targets; MFT sources -> MFT targets within vs between the
              individualizing/binding split; Schwartz C21 (self) sources -> C21 targets vs circle distance
Writes results/s5/report.json and prints the summary.

    python -m cm.s5_report
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from .paths import RESULTS
from .s4_overlap import SCHWARTZ

KNOW, DO = set("ab"), set("cd")


def main():
    R = json.loads((RESULTS / "s5" / "causal_matrix.json").read_text())
    D = json.loads((RESULTS / "s4" / "overlap_verdict.json").read_text())
    src, U = R["sources"], R["targets"]
    tix = {u: i for i, u in enumerate(U)}
    dix = {u: i for i, u in enumerate(D["units"])}
    out = {}
    for li, L in enumerate(R["layers"]):
        E = np.array(R["kinds"]["dense+"]["E"])[:, li]
        Z = np.array(R["kinds"]["dense+"]["Z"])[:, li]
        Em = np.array(R["kinds"]["dense-"]["E"])[:, li]
        Es = np.array(R["kinds"]["sae+"]["E"])[:, li]
        Zs = np.array(R["kinds"]["sae+"]["Z"])[:, li]
        o = {}
        sel = [(s, tix[u]) for s, u in enumerate(src) if u in tix]
        se = np.array([E[s, t] for s, t in sel]); sz = np.array([Z[s, t] for s, t in sel])
        sse = np.array([Es[s, t] for s, t in sel]); ssz = np.array([Zs[s, t] for s, t in sel])
        o["self"] = {"median_E": float(np.nanmedian(se)), "median_z": float(np.nanmedian(sz)), "share_z_gt3": float(np.nanmean(sz > 3)),
                     "sae_median_E": float(np.nanmedian(sse)), "sae_share_z_gt3": float(np.nanmean(ssz > 3))}
        # within concept, other frame
        def cells(cond):
            v = [(E[s, t], Z[s, t]) for s, su in enumerate(src) for t, tu in enumerate(U)
                 if cond(su.split("|"), tu.split("|"))]
            v = [x for x in v if not np.isnan(x[0])]
            return {"n": len(v), "median_E": float(np.median([x[0] for x in v])) if v else None,
                    "share_z_gt3": float(np.mean([x[1] > 3 for x in v])) if v else None,
                    "share_z_lt_-3": float(np.mean([x[1] < -3 for x in v])) if v else None}
        same = lambda a, b: a[0] == b[0] and a[1] == b[1] and a[2] != b[2]
        o["within"] = {"knowing_to_doing": cells(lambda a, b: same(a, b) and a[2] in KNOW and b[2] in DO),
                       "doing_to_knowing": cells(lambda a, b: same(a, b) and a[2] in DO and b[2] in KNOW),
                       "doing_to_doing": cells(lambda a, b: same(a, b) and a[2] in DO and b[2] in DO),
                       "knowing_to_knowing": cells(lambda a, b: same(a, b) and a[2] in KNOW and b[2] in KNOW)}
        offm = np.array([[src[s].split("|")[0] != u.split("|")[0] for u in U] for s in range(len(src))])
        offz = Z[offm]
        o["spillover"] = {"share_abs_z_gt3": float(np.nanmean(np.abs(offz) > 3))}
        k2 = [s for s, u in enumerate(src) if u.startswith("K2|")]
        if k2:
            o["spillover"]["K2_share_abs_z_gt3"] = float(np.nanmean(np.abs(Z[k2[0]][offm[k2[0]]]) > 3))
        top = sorted(((abs(Z[s, t]), src[s], U[t], float(E[s, t]), float(Z[s, t])) for s in range(len(src)) for t in range(len(U))
                      if offm[s, t] and not np.isnan(Z[s, t])), reverse=True)[:15]
        o["spillover"]["top"] = [{"source": a, "target": b, "E": e, "z": z} for _, a, b, e, z in top]
        # geometry
        g = [(E[s, t], D["layers"][str(L)]["cos"][dix[su]][dix[tu]]) for s, su in enumerate(src) for t, tu in enumerate(U)
             if offm[s, t] and su in dix and tu in dix and not np.isnan(E[s, t])]
        o["geometry_spearman_E_vs_s4cos"] = float(spearmanr([x[0] for x in g], [x[1] for x in g]).correlation)
        ok = ~np.isnan(E)
        o["agreement"] = {"dense_vs_sae_spearman": float(spearmanr(E[ok], Es[ok]).correlation),
                          "sign_symmetry_r": float(np.corrcoef(E[ok], -Em[ok])[0, 1])}
        # taxonomy
        tax = {}
        sx = {u: i for i, u in enumerate(src)}
        h = "C19|honesty_humility|c"
        if h in sx:
            tax["hh_to_dark_triad"] = {u: float(Z[sx[h], tix[u]]) for u in U if u.startswith("C20|")}
        bs = [f"B{i}|all|b" for i in range(1, 6)]
        bt = [f"B{i}|all|b" for i in range(1, 6)]
        if all(x in sx for x in bs) and all(x in tix for x in bt):
            M = np.array([[E[sx[a], tix[b]] for b in bt] for a in bs])
            def stat(group):   # cells without a target gap are NaN and left out
                wi = [M[p, q] for p in range(5) for q in range(5) if p != q and (p in group) == (q in group)]
                be = [M[p, q] for p in range(5) for q in range(5) if p != q and (p in group) != (q in group)]
                return float(np.nanmean(wi) - np.nanmean(be))
            obs = stat({0, 1})
            tax["mft_within_minus_between"] = {"value": obs, "p_exact": sum(stat(set(g)) >= obs for g in itertools.combinations(range(5), 2)) / 10,
                                               "nan_cells": int(np.isnan(M).sum())}
        cs = [f"C21|{v}|d" for v in SCHWARTZ]
        if all(x in sx for x in cs) and all(x in tix for x in cs):
            prs = [(a, b) for a in range(10) for b in range(10) if a != b and not np.isnan(E[sx[cs[a]], tix[cs[b]]])]
            vals = [E[sx[cs[a]], tix[cs[b]]] for a, b in prs]
            dd = [min(abs(a - b), 10 - abs(a - b)) for a, b in prs]
            rho = spearmanr(dd, vals).correlation
            rng = np.random.default_rng(0)
            null = []
            for _ in range(5000):
                pp = rng.permutation(10)
                null.append(spearmanr([min(abs(pp[a] - pp[b]), 10 - abs(pp[a] - pp[b])) for a, b in prs], vals).correlation)
            tax["schwartz_self"] = {"rho": float(rho), "p_one_sided_negative": (1 + sum(x <= rho for x in null)) / 5001,
                                    "n_pairs": len(prs)}
        o["taxonomy"] = tax
        out[str(L)] = o
        print(f"\nlayer {L}")
        print(f"  self: median {o['self']['median_E']:.2f} gap units, z {o['self']['median_z']:.1f}, z>3 in {o['self']['share_z_gt3']:.2f}"
              f" | SAE vector: median {o['self']['sae_median_E']:.2f}, z>3 in {o['self']['sae_share_z_gt3']:.2f}")
        for k, v in o["within"].items():
            print(f"  {k}: n {v['n']}, median {v['median_E']}, z>3 {v['share_z_gt3']}, z<-3 {v['share_z_lt_-3']}")
        print(f"  spillover off-concept |z|>3: {o['spillover']['share_abs_z_gt3']:.3f}; K2 source {o['spillover'].get('K2_share_abs_z_gt3')}")
        print(f"  geometry: off-concept effect vs S4 cosine Spearman {o['geometry_spearman_E_vs_s4cos']:.2f}")
        print(f"  agreement: dense vs SAE {o['agreement']['dense_vs_sae_spearman']:.2f}; sign symmetry {o['agreement']['sign_symmetry_r']:.2f}")
        print(f"  taxonomy: {json.dumps(tax)}")
        print("  top off-concept: " + "; ".join(f"{x['source']} -> {x['target']} z {x['z']:+.1f}" for x in o["spillover"]["top"][:8]))
    (RESULTS / "s5" / "report.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
