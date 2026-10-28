"""Clean version of the honesty landscape for collaborators: one layer, honesty items only (no formality control), the
first three principal components of the honesty-direction subspace, every item a circle colored by definition, filled
for the honest side and hollow for the other side, the three frame groups labelled in place with leader lines, and for
each (definition, frame) unit a segment from the other-side centroid to the honest-side centroid.

Reads landscape_3d.json (written by draw_landscape_3pc.py from cm/honesty_landscape.py's output).
Writes landscape_clean.{png,pdf}          one panel, colored by definition
       landscape_clean_2panel.{png,pdf}   a: colored by frame group, b: colored by definition (same view)
       landscape_clean_sheet.png          (--sheet) six candidate view angles, to pick one
Run on the Mac:  python3 results/figures/honesty/draw_landscape_clean.py [layer=29] [--sheet] [--view ELEV,AZIM]
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3D

HERE = Path(os.environ["CM_HONESTY_DIR"]) if os.environ.get("CM_HONESTY_DIR") else Path(__file__).resolve().parent
args = sys.argv[1:]
L = next((a for a in args if a.isdigit()), "29")
SHEET = "--sheet" in args
VIEW = (28, -100)
if "--view" in args:
    VIEW = tuple(float(v) for v in args[args.index("--view") + 1].split(","))
PER_CELL = 70            # items drawn per definition, frame and side; centroids use every item

D = json.load(open(HERE / "landscape_3d.json"))
meta = D["meta"]
V = D["views"][f"as-is, layer {L}"]
X = np.array(V["xyz"])
var = V["var"]

DEFS = {"Self-knowledge": "#009E73", "Accuracy": "#E69F00", "Sincerity": "#0072B2", "Forthrightness": "#D55E00",
        "Non-sycophancy": "#56B4E9", "Honesty-humility": "#CC79A7"}
CID = {"A9": "Self-knowledge", "C2": "Accuracy", "C1": "Sincerity", "C3": "Forthrightness", "C8": "Non-sycophancy", "C19": "Honesty-humility"}
FRG = {"a": "own", "b": "own", "c": "act", "d": "self"}
GRPC = {"own": "#7570B3", "act": "#D95F02", "self": "#1B9E77"}
GRPN = {"own": "own reply about someone else (recognize, judge)", "act": "acting (prefilled reply)", "self": "self-describing (prefilled reply)"}
GROUP_LABEL = {"own": ("recognize, judge", "Gemma's own reply about someone else"), "act": ("act", "prefilled reply in Gemma's voice"),
               "self": ("self-describe", "prefilled reply in Gemma's voice")}
LABEL_OFFSET = {"own": (0, 52), "act": (-12, -64), "self": (-72, 42)}     # label position in points from the group's median item, for VIEW
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e4e3dc", "#c3c2b7"
HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6.5, "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": INK2})

defn = np.array([m["def"] for m in meta])
fr = np.array([m["frame"] for m in meta])
pol = np.array([m["pol"] for m in meta])
hon = defn != "Formality (control)"
grp = np.array([FRG[f] for f in fr])

rng = np.random.default_rng(0)
keep = []
for key in sorted(set(zip(defn[hon], fr[hon], pol[hon]))):
    idx = np.where(hon & (defn == key[0]) & (fr == key[1]) & (pol == key[2]))[0]
    keep += list(rng.choice(idx, min(PER_CELL, len(idx)), replace=False))
keep = np.array(sorted(keep))


def panel(ax, mode, view, labels=True, point_size=5):
    """mode 'def' colors by definition, 'frame' by frame group."""
    color_of = (lambda i: DEFS[defn[i]]) if mode == "def" else (lambda i: GRPC[grp[i]])
    cells = {}
    for i in keep:
        cells.setdefault((color_of(i), pol[i]), []).append(i)
    for (c, p), idx in cells.items():
        idx = np.array(idx)
        ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], marker="o", s=point_size, facecolor=c if p == "pos" else "none",
                   edgecolor=c, linewidth=0.4, alpha=0.33, depthshade=False, rasterized=True)
    for u, (cn, cp) in V["centroids"].items():
        cid, _, f = u.split("|")
        if cid == "K1":
            continue
        c = DEFS[CID[cid]] if mode == "def" else GRPC[FRG[f]]
        ax.add_line(Line3D([cn[0], cp[0]], [cn[1], cp[1]], [cn[2], cp[2]], color=c, lw=1.4, zorder=8))
        ax.scatter(*cn, marker="o", s=30, facecolor="white", edgecolor=c, linewidth=1.1, depthshade=False, zorder=9)
        ax.scatter(*cp, marker="o", s=30, facecolor=c, edgecolor="white", linewidth=0.7, depthshade=False, zorder=9)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)", labelpad=-3)
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)", labelpad=-3)
    ax.set_zlabel(f"PC3 ({var[2]*100:.0f}%)", labelpad=-5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis.line.set_color(AXIS)
        axis.set_tick_params(pad=-3, colors=INK2)
    Xh = X[hon]
    ax.set_box_aspect((np.ptp(Xh[:, 0]), np.ptp(Xh[:, 1]), np.ptp(Xh[:, 2])))
    ax.set_xlim(Xh[:, 0].min(), Xh[:, 0].max()); ax.set_ylim(Xh[:, 1].min(), Xh[:, 1].max()); ax.set_zlim(Xh[:, 2].min(), Xh[:, 2].max())
    ax.view_init(elev=view[0], azim=view[1])
    if labels:
        # leader lines from each frame group's median item to a label outside the cloud
        for g in ("own", "act", "self"):
            anc = np.median(X[hon & (grp == g)], 0)
            x2, y2, _ = proj3d.proj_transform(anc[0], anc[1], anc[2], ax.get_proj())
            name, kind = GROUP_LABEL[g]
            dx, dy = LABEL_OFFSET[g]
            ax.annotate(name, xy=(x2, y2), xytext=(dx, dy), textcoords="offset points", ha="center", va="bottom", fontsize=7.2,
                        fontweight="bold", color=INK, path_effects=HALO, zorder=40, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=1.5))
            ax.annotate(kind, xy=(x2, y2), xytext=(dx, dy - 1), textcoords="offset points", ha="center", va="top", fontsize=6,
                        color=INK2, path_effects=HALO, zorder=40, annotation_clip=False)


def legend_handles(mode):
    if mode == "def":
        h = [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=5) for n, c in DEFS.items()]
    else:
        h = [Line2D([], [], marker="o", ls="none", color=GRPC[g], label=GRPN[g], ms=5) for g in ("own", "act", "self")]
    h += [Line2D([], [], marker="o", ls="none", color=INK2, label="honest side of the contrast (e.g. sincere, knows)", ms=4),
          Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="other side (deceptive, does not know)", ms=4),
          Line2D([], [0], marker="o", color=INK2, mfc="white", mec=INK2, label="unit centroids, other side to honest side", ms=5, lw=1.4)]
    return h


if SHEET:
    views = [(20, -50), (22, -20), (25, 10), (18, 40), (28, -100), (22, -140)]
    fig = plt.figure(figsize=(12, 7.5))
    for k, v in enumerate(views):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        panel(ax, "def", v, labels=True, point_size=4)
        ax.set_title(f"elev {v[0]}, azim {v[1]}", loc="left", pad=-2)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.98, wspace=0, hspace=0.05)
    fig.savefig(HERE / "landscape_clean_sheet.png", dpi=110, facecolor="white")
    print("wrote", HERE / "landscape_clean_sheet.png")
    sys.exit()

# one panel, colored by definition
fig = plt.figure(figsize=(6.2, 5.2))
ax = fig.add_subplot(1, 1, 1, projection="3d")
panel(ax, "def", VIEW)
fig.legend(handles=legend_handles("def"), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.3, columnspacing=1.1)
fig.subplots_adjust(left=0, right=1, bottom=0.09, top=1.08)
fig.savefig(HERE / "landscape_clean.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(HERE / "landscape_clean.pdf", bbox_inches="tight", pad_inches=0.03)
print("wrote", HERE / "landscape_clean.png")

# two panels: by frame group, by definition
fig = plt.figure(figsize=(7.6, 3.9))
for k, mode in enumerate(("frame", "def")):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    panel(ax, mode, VIEW, labels=(mode == "frame"), point_size=4)
    if mode == "frame":
        ax.set_zlabel("")        # panel b carries the shared PC3 label; the group label needs the room
    ax.set_title("ab"[k], loc="left", pad=-4, fontweight="bold", fontsize=8)
h = legend_handles("frame")[:3] + legend_handles("def")
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.0)
fig.subplots_adjust(left=0, right=0.92, bottom=0.17, top=1.06, wspace=0.12)
fig.savefig(HERE / "landscape_clean_2panel.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(HERE / "landscape_clean_2panel.pdf", bbox_inches="tight", pad_inches=0.03)
print("wrote", HERE / "landscape_clean_2panel.png")
