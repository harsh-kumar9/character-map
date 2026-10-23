"""Three-component views of the item-level honesty landscape, from the full coordinates saved by cm/honesty_landscape.py.
Static: a 2 x 3 figure (layers 17 and 29; PC1-PC2, PC1-PC3, PC2-PC3), items colored by definition, shaped by frame,
filled for + and hollow for -, with - to + centroid arrows. Also writes landscape_3d.json for the page's rotatable view.
Run on the Mac:  python3 results/figures/honesty/draw_landscape_3pc.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
HERE = Path(os.environ["CM_HONESTY_DIR"]) if os.environ.get("CM_HONESTY_DIR") else Path(__file__).resolve().parent   # per-model: results/<tag>/figures/honesty
LM, LL = os.environ.get("CM_LMID", "17"), os.environ.get("CM_LLATE", "29")
pts = np.load(HERE / "landscape_points.npz"); meta = json.loads(str(pts["meta"]))
J = json.load(open(HERE / "landscape.json"))["layers"]
DEFS = {"Self-knowledge": "#009E73", "Accuracy": "#E69F00", "Sincerity": "#0072B2", "Forthrightness": "#D55E00", "Non-sycophancy": "#56B4E9", "Honesty-humility": "#CC79A7", "Formality (control)": "#898781"}
CID = {"A9": "Self-knowledge", "C2": "Accuracy", "C1": "Sincerity", "C3": "Forthrightness", "C8": "Non-sycophancy", "C19": "Honesty-humility", "K1": "Formality (control)"}
FR = {"a": ("recognize", "^"), "b": ("judge", "D"), "c": ("act", "o"), "d": ("self-describe", "s")}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 6.8, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6.3, "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "text.color": INK})
rng = np.random.default_rng(0)
MAX_PLOT = 120
defn = np.array([CID[m["concept"]] for m in meta]); frame = np.array([m["frame"] for m in meta]); pol = np.array([m["polarity"] for m in meta])
unit = np.array([f"{m['concept']}|{m['sub']}|{m['frame']}" for m in meta])
keep = []
for key in sorted(set(zip(unit, pol))):
    idx = np.where((unit == key[0]) & (pol == key[1]))[0]
    keep += list(rng.choice(idx, min(MAX_PLOT, len(idx)), replace=False))
keep = np.array(sorted(keep))

def coords(L, within):
    P = pts[f"Pw{L}" if within else f"P{L}"]; Vt = pts[f"Vtw{L}" if within else f"Vt{L}"]
    return P @ Vt.T                                   # [N, 10] principal-component coordinates

export = {"meta": [{"def": d, "frame": f, "pol": p} for d, f, p in zip(defn, frame, pol)], "views": {}}
fig, axes = plt.subplots(2, 3, figsize=(7.3, 5.0))
for r, L in enumerate((LM, LL)):
    C = coords(L, False); var = J[L]["variance_all"]
    centro = {}
    for u in np.unique(unit):
        cp = C[(unit == u) & (pol == "pos")][:, :3].mean(0); cn = C[(unit == u) & (pol == "neg")][:, :3].mean(0)
        centro[u] = (cn.round(2).tolist(), cp.round(2).tolist())
    export["views"][f"as-is, layer {L}"] = {"xyz": C[:, :3].round(2).tolist(), "var": var[:3], "centroids": centro}
    Cw = coords(L, True); varw = J[L]["variance_within_all"]
    centw = {}
    for u in np.unique(unit):
        cp = Cw[(unit == u) & (pol == "pos")][:, :3].mean(0); cn = Cw[(unit == u) & (pol == "neg")][:, :3].mean(0)
        centw[u] = (cn.round(2).tolist(), cp.round(2).tolist())
    export["views"][f"each unit centred, layer {L}"] = {"xyz": Cw[:, :3].round(2).tolist(), "var": varw[:3], "centroids": centw}
    for c, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax = axes[r, c]
        for k in keep:
            col = DEFS[defn[k]]; ctrl = defn[k].startswith("Formality")
            ax.scatter(C[k, i], C[k, j], marker=FR[frame[k]][1], s=6 if not ctrl else 4, facecolor=col if pol[k] == "pos" else "none",
                       edgecolor=col, linewidth=0.45, alpha=0.35 if not ctrl else 0.25, zorder=2, rasterized=True)
        for u, (cn, cp) in centro.items():
            col = DEFS[CID[u.split("|")[0]]]; f = u.split("|")[2]
            ax.annotate("", xy=(cp[i], cp[j]), xytext=(cn[i], cn[j]), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7), zorder=5)
            ax.scatter(cn[i], cn[j], marker=FR[f][1], s=30, facecolor="white", edgecolor=col, linewidth=1.1, zorder=6)
            ax.scatter(cp[i], cp[j], marker=FR[f][1], s=30, facecolor=col, edgecolor="white", linewidth=0.7, zorder=6)
        ax.set_xlabel(f"PC{i+1} ({var[i]*100:.0f}%)"); ax.set_ylabel(f"PC{j+1} ({var[j]*100:.0f}%)")
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True); ax.set_aspect("equal", adjustable="datalim")
        if c == 0: ax.set_title(f"Layer {L}", loc="left")
        ax.text(-0.02, 1.03, "abcdef"[r * 3 + c], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
h = [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=5) for n, c in DEFS.items() if not n.startswith("Formality")] + \
    [Line2D([], [], marker="o", ls="none", mfc="white", mec=DEFS["Formality (control)"], label="Formality (control)", ms=4)] + \
    [Line2D([], [], marker=m, ls="none", color=INK2, mfc="white", label=n, ms=5) for f, (n, m) in FR.items()] + \
    [Line2D([], [], marker="o", ls="none", color=INK2, label="+ version", ms=4), Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="− version", ms=4)]
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
fig.savefig(HERE / "landscape_3pc.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(HERE / "landscape_3pc.pdf", bbox_inches="tight", pad_inches=0.03)
json.dump(export, open(HERE / "landscape_3d.json", "w"), separators=(",", ":"))
# what the third component carries, for the text
for L in (LM, LL):
    C = coords(L, False); print(f"layer {L} variance PC1-4 {[round(v, 2) for v in J[L]['variance_all'][:4]]}")
    for u in sorted(set(unit)):
        gap = C[(unit == u) & (pol == "pos")][:, :3].mean(0) - C[(unit == u) & (pol == "neg")][:, :3].mean(0)
        print(f"   {u:26} gap PC1 {gap[0]:+6.2f} PC2 {gap[1]:+6.2f} PC3 {gap[2]:+6.2f}   centroid PC3 {C[unit == u][:, 2].mean():+6.2f}")
print("wrote", HERE / "landscape_3pc.png", "and landscape_3d.json")
