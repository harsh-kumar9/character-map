"""Item-level landscape of the honesty block (D-043): every + and - item of the four honesty definitions in every frame,
projected into the subspace spanned by the ten (definition, frame) directions, then shown in the top two principal
components of that subspace. Companion to the lattice figure (results/figures/honesty/build_lattice.py).

Directions follow S4 (cm/s4_overlap.py): reply32 readout, Gemma's verdict direction projected out of frame a/b rows,
every hidden dimension standardized, diff of means on train items. The standardizing SD here is taken over the pooled
items of the plotted units plus the formality control (S4 used the probe pool of all 208 units); the script prints the
cosines between the directions built here and the S4 ones as a consistency check.

Run on ada:  sbatch -J honesty_land scripts/cm_cpu.sbatch cm.honesty_landscape
Writes results/figures/honesty/landscape.{png,pdf}, landscape_points.npz, landscape.json
"""
from __future__ import annotations

import json

import numpy as np

from . import cfg
from .s4_overlap import DEPTHS, LIDX, concept_data, unit_vec

RESULTS = cfg.results_dir()
OUT = RESULTS / "figures" / "honesty"
DEFS = [("Self-knowledge", "A9", "all", "#009E73"), ("Accuracy", "C2", "all", "#E69F00"),
        ("Sincerity", "C1", "all", "#0072B2"), ("Honesty-humility", "C19", "honesty_humility", "#CC79A7")]
CTRL = ("Formality (control)", "K1", "all", "#898781")
FRAMES = {"a": ("recognize", "^"), "b": ("judge", "D"), "c": ("act", "o"), "d": ("self-describe", "s")}
LAYERS = tuple(cfg.L_LAND)
MAX_PLOT = 120          # points per unit and polarity in the figure; PCA uses every item
SEED = 0


def gather():
    """Standardized item activations and unit directions at each layer."""
    items = []   # dicts with concept, sub, frame, polarity, split, idx into X
    Xs = []
    for name, cid, sub, col in DEFS + [CTRL]:
        X, meta = concept_data(cid, "verdict")          # [n, 4 layers, d], verdict projected out of a/b rows
        for i, m in enumerate(meta):
            if (m["contrast"] or "all") != sub or m["split"] not in ("train", "test"):
                continue
            items.append({"def": name, "concept": cid, "sub": sub, "frame": m["frame"], "polarity": m["polarity"],
                          "split": m["split"], "row": len(Xs)})
            Xs.append(X[i])
    X = np.stack(Xs)                                      # [N, 4, d]
    return items, X


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, X = gather()
    rng = np.random.default_rng(SEED)
    s4 = json.load(open(RESULTS / "s4" / "overlap_verdict.json"))
    s4_idx = {u: i for i, u in enumerate(s4["units"])}
    summary = {"n_items": len(items), "layers": {}}
    points = {}
    for L in LAYERS:
        l = LIDX.index(DEPTHS[L])
        Z = X[:, l]                                        # [N, d]
        sd = Z.std(0) + 1e-6
        Zs = Z / sd
        units = []
        for name, cid, sub, col in DEFS + [CTRL]:
            for f in "abcd":
                tr = [it["row"] for it in items if it["concept"] == cid and it["sub"] == sub and it["frame"] == f and it["split"] == "train"]
                pos = [r for r in tr if items[r]["polarity"] == "pos"]
                neg = [r for r in tr if items[r]["polarity"] == "neg"]
                if min(len(pos), len(neg)) < 10:
                    continue
                d = unit_vec(Zs[pos].mean(0) - Zs[neg].mean(0))
                units.append({"id": f"{cid}|{sub}|{f}", "def": name, "frame": f, "d": d, "ctrl": name == CTRL[0]})
        hon = [u for u in units if not u["ctrl"]]
        D = np.stack([u["d"] for u in hon])                # [10, d]
        # consistency with S4: cosines between these directions and S4's for the same unit pairs
        cos_here = D @ D.T
        cos_s4 = np.array([[s4["layers"][str(L)]["cos"][s4_idx[a["id"]]][s4_idx[b["id"]]] for b in hon] for a in hon])
        off = ~np.eye(len(hon), dtype=bool)
        consist = {"max_abs_diff": float(np.abs(cos_here - cos_s4)[off].max()),
                   "spearman": float(np.corrcoef(cos_here[off], cos_s4[off])[0, 1])}
        # orthonormal basis of the ten directions, items projected into it, PCA of the projections
        Q, _ = np.linalg.qr(D.T)                           # [d, 10]
        mu = Zs.mean(0)
        P = (Zs - mu) @ Q                                  # [N, 10]
        Pc = P - P.mean(0)
        U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
        var = S ** 2 / (S ** 2).sum()
        Y = Pc @ Vt[:2].T                                  # [N, 2]
        # within-unit view: remove each unit's mean so only polarity-related variance remains, then PCA again
        unit_of = np.array([f"{it['concept']}|{it['sub']}|{it['frame']}" for it in items])
        Pw = P.copy()
        for uid in np.unique(unit_of):
            Pw[unit_of == uid] -= Pw[unit_of == uid].mean(0)
        Uw, Sw, Vtw = np.linalg.svd(Pw, full_matrices=False)
        varw = Sw ** 2 / (Sw ** 2).sum()
        Yw = Pw @ Vtw[:2].T
        # which directions load on the two components (cosine between each direction and the component axis in d-space)
        axes_d = (Q @ Vt[:2].T).T                          # [2, d]
        loads = {u["id"]: [float(axes_d[0] @ u["d"]), float(axes_d[1] @ u["d"])] for u in hon}
        # per-unit separation of + from - along each component, in SDs of that unit's items
        sep = {}
        for u in units:
            cid, sub, f = u["id"].split("|")
            rows = [it["row"] for it in items if it["concept"] == cid and it["sub"] == sub and it["frame"] == f]
            pos = [r for r in rows if items[r]["polarity"] == "pos"]; neg = [r for r in rows if items[r]["polarity"] == "neg"]
            gap = Y[pos].mean(0) - Y[neg].mean(0)
            gw = Yw[pos].mean(0) - Yw[neg].mean(0)
            sep[u["id"]] = {"gap_pc1": float(gap[0]), "gap_pc2": float(gap[1]), "sd_pc1": float(Y[rows][:, 0].std()),
                            "sd_pc2": float(Y[rows][:, 1].std()), "n": [len(pos), len(neg)],
                            "centroid_pos": Y[pos].mean(0).tolist(), "centroid_neg": Y[neg].mean(0).tolist(),
                            "w_centroid_pos": Yw[pos].mean(0).tolist(), "w_centroid_neg": Yw[neg].mean(0).tolist(),
                            "w_gap": gw.tolist(), "w_sd": [float(Yw[rows][:, 0].std()), float(Yw[rows][:, 1].std())]}
        # control items (K1) projected into the same subspace and components, as a reference cloud
        summary["layers"][str(L)] = {"variance_explained": var[:5].round(4).tolist(), "variance_within": varw[:5].round(4).tolist(),
                                     "consistency_with_s4": consist,
                                     "loadings": loads, "separation": sep, "units": [u["id"] for u in units]}
        points[str(L)] = Y.astype(np.float32); points[f"w{L}"] = Yw.astype(np.float32)
        # full coordinates in the ten-direction subspace (as is, and unit-centred) plus the PCA bases, so any number of
        # components can be drawn without rerunning: P10 @ Vt.T gives all ten PCs
        points[f"P{L}"] = Pc.astype(np.float32); points[f"Pw{L}"] = Pw.astype(np.float32)
        points[f"Vt{L}"] = Vt.astype(np.float32); points[f"Vtw{L}"] = Vtw.astype(np.float32)
        summary["layers"][str(L)]["variance_all"] = var.round(4).tolist(); summary["layers"][str(L)]["variance_within_all"] = varw.round(4).tolist()
        summary["layers"][str(L)]["direction_ids"] = [u["id"] for u in hon]
        print(f"[landscape] layer {L}: {len(hon)} directions, {len(items)} items, PC1 {var[0]:.2f} PC2 {var[1]:.2f}, "
              f"S4 consistency max|diff| {consist['max_abs_diff']:.2f} r {consist['spearman']:.2f}", flush=True)
        for u in hon:
            s = sep[u["id"]]
            print(f"   {u['id']:28} gap PC1 {s['gap_pc1']:+.2f} PC2 {s['gap_pc2']:+.2f}  load PC1 {loads[u['id']][0]:+.2f} PC2 {loads[u['id']][1]:+.2f}")
    meta = [{k: v for k, v in it.items() if k != "row"} for it in items]
    np.savez(OUT / "landscape_points.npz", meta=np.array(json.dumps(meta)), **{f"Y{L}": points[str(L)] for L in LAYERS},
             **{f"W{L}": points[f"w{L}"] for L in LAYERS}, **{k: v for k, v in points.items() if k[0] in "PV"})
    json.dump(summary, open(OUT / "landscape.json", "w"), indent=1)
    draw(meta, points, summary, rng)


def draw(meta, points, summary, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                         "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3, "axes.linewidth": 0.5,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "text.color": INK})
    col = {name: c for name, _, _, c in DEFS}; col[CTRL[0]] = CTRL[3]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.6))
    panels = [(axes[0, 0], LAYERS[0], "a", False), (axes[0, 1], LAYERS[1], "b", False), (axes[1, 0], LAYERS[0], "c", True), (axes[1, 1], LAYERS[1], "d", True)]
    for ax, L, letter, within in panels:
        Y = points[f"w{L}" if within else str(L)]
        var = summary["layers"][str(L)]["variance_within" if within else "variance_explained"]
        sep = summary["layers"][str(L)]["separation"]
        # subsample per unit and polarity for the scatter
        keep = []
        groups = {}
        for i, m in enumerate(meta):
            groups.setdefault((m["def"], m["frame"], m["polarity"]), []).append(i)
        for k, idx in groups.items():
            idx = np.array(idx)
            keep += list(rng.choice(idx, min(MAX_PLOT, len(idx)), replace=False))
        keep = np.array(keep)
        for i in keep:
            m = meta[i]; c = col[m["def"]]; ctrl = m["def"] == CTRL[0]
            ax.scatter(Y[i, 0], Y[i, 1], marker=FRAMES[m["frame"]][1], s=7 if not ctrl else 5,
                       facecolor=c if m["polarity"] == "pos" else "none", edgecolor=c, linewidth=0.5,
                       alpha=0.35 if not ctrl else 0.25, zorder=2)
        # centroids and the - to + arrow per unit
        for uid, s in sep.items():
            cid, sub, f = uid.split("|"); name = next(n for n, c, su, _ in DEFS + [CTRL] if c == cid and su == sub)
            c = col[name]
            cn, cp = (np.array(s["w_centroid_neg"]), np.array(s["w_centroid_pos"])) if within else (np.array(s["centroid_neg"]), np.array(s["centroid_pos"]))
            ax.annotate("", xy=cp, xytext=cn, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, mutation_scale=8), zorder=5)
            ax.scatter(*cn, marker=FRAMES[f][1], s=38, facecolor="white", edgecolor=c, linewidth=1.2, zorder=6)
            ax.scatter(*cp, marker=FRAMES[f][1], s=38, facecolor=c, edgecolor="white", linewidth=0.8, zorder=6)
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% of the subspace variance)" if not within else f"PC1 within units ({var[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
        ax.set_title(f"Layer {L}" + (", each unit centred" if within else ", as is"), loc="left")
        ax.text(-0.02, 1.03, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True); ax.set_aspect("equal", adjustable="datalim")
    h = [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=5) for n, _, _, c in DEFS] + \
        [Line2D([], [], marker="o", ls="none", mfc="white", mec=CTRL[3], label=CTRL[0], ms=4)] + \
        [Line2D([], [], marker=m, ls="none", color=INK2, mfc="white", label=n, ms=5) for f, (n, m) in FRAMES.items()] + \
        [Line2D([], [], marker="o", ls="none", color=INK2, label="+ item", ms=4), Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="− item", ms=4)]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05), handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "landscape.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(OUT / "landscape.pdf", bbox_inches="tight", pad_inches=0.03)
    print("[landscape] wrote", OUT / "landscape.png", flush=True)


if __name__ == "__main__":
    main()
