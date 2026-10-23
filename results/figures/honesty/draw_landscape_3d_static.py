"""Static 3D rendering of the first three principal components of the honesty subspace (layers 17 and 29), from the
coordinates exported by draw_landscape_3pc.py (landscape_3d.json). Run on the Mac:
python3 results/figures/honesty/draw_landscape_3d_static.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3D

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "landscape_3d.json")); meta = D["meta"]
DEFS = {"Self-knowledge": "#009E73", "Accuracy": "#E69F00", "Sincerity": "#0072B2", "Honesty-humility": "#CC79A7", "Formality (control)": "#898781"}
CID = {"A9": "Self-knowledge", "C2": "Accuracy", "C1": "Sincerity", "C19": "Honesty-humility", "K1": "Formality (control)"}
FR = {"a": ("recognize", "^"), "b": ("judge", "D"), "c": ("act", "o"), "d": ("self-describe", "s")}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 6.8, "xtick.labelsize": 5.8,
                     "ytick.labelsize": 5.8, "legend.fontsize": 6.3, "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": INK2})
rng = np.random.default_rng(0)
defn = np.array([m["def"] for m in meta]); frame = np.array([m["frame"] for m in meta]); pol = np.array([m["pol"] for m in meta])
keep = []
for key in sorted(set(zip(defn, frame, pol))):
    idx = np.where((defn == key[0]) & (frame == key[1]) & (pol == key[2]))[0]
    keep += list(rng.choice(idx, min(120, len(idx)), replace=False))
keep = np.array(sorted(keep))
fig = plt.figure(figsize=(7.3, 3.6))
for k, (L, view) in enumerate((("17", (22, -55)), ("29", (20, -50)))):
    V = D["views"][f"as-is, layer {L}"]; X = np.array(V["xyz"]); var = V["var"]
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for i in keep:
        c = DEFS[defn[i]]; ctrl = defn[i].startswith("Formality")
        ax.scatter(X[i, 0], X[i, 1], X[i, 2], marker=FR[frame[i]][1], s=5 if not ctrl else 3.5, facecolor=c if pol[i] == "pos" else "none",
                   edgecolor=c, linewidth=0.4, alpha=0.35 if not ctrl else 0.25, depthshade=False)
    for u, (cn, cp) in V["centroids"].items():
        c = DEFS[CID[u.split("|")[0]]]; f = u.split("|")[2]
        ax.add_line(Line3D([cn[0], cp[0]], [cn[1], cp[1]], [cn[2], cp[2]], color=c, lw=1.3))
        ax.scatter(*cn, marker=FR[f][1], s=26, facecolor="white", edgecolor=c, linewidth=1.0, depthshade=False, zorder=10)
        ax.scatter(*cp, marker=FR[f][1], s=26, facecolor=c, edgecolor="white", linewidth=0.6, depthshade=False, zorder=10)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)", labelpad=-4); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)", labelpad=-4); ax.set_zlabel(f"PC3 ({var[2]*100:.0f}%)", labelpad=-4)
    ax.view_init(elev=view[0], azim=view[1]); ax.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(X[:, 2])))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0)); axis._axinfo["grid"]["color"] = GRID; axis._axinfo["grid"]["linewidth"] = 0.4
        axis.line.set_color(AXIS); axis.set_tick_params(pad=-3, colors=INK2)
    ax.set_title(f"{'ab'[k]}  Layer {L}", loc="left", pad=-2, fontweight="bold")
h = [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=5) for n, c in DEFS.items() if not n.startswith("Formality")] + \
    [Line2D([], [], marker="o", ls="none", mfc="white", mec=DEFS["Formality (control)"], label="Formality (control)", ms=4)] + \
    [Line2D([], [], marker=m, ls="none", color=INK2, mfc="white", label=n, ms=5) for f, (n, m) in FR.items()] + \
    [Line2D([], [], marker="o", ls="none", color=INK2, label="+ version", ms=4), Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="− version", ms=4)]
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.0)
fig.subplots_adjust(left=0, right=1, bottom=0.08, top=1, wspace=0)
fig.savefig(HERE / "landscape_3d.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
fig.savefig(HERE / "landscape_3d.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote", HERE / "landscape_3d.png")
