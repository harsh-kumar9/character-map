"""Lattice of honesty definitions x frames inside Gemma. Nodes = directions (within-frame AUROC inside), edges = cosine
above the shuffle floor (a, b), arrows = steering effects z >= 3 (c). Run: python3 build_lattice.py <repo> <outdir>"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2])
import os
TAG = os.environ.get("CM_TAG", "")                 # results/<TAG>/ for other subject models; "" = Gemma-3-4B
LM, LL = int(os.environ.get("CM_LMID", 17)), int(os.environ.get("CM_LLATE", 29))   # the two depths the page shows
L_STEER = int(os.environ.get("CM_LSTEER", LM))
D = json.load(open(OUT / "honesty_data.json"))
R = (REPO / "results" / TAG) if TAG else (REPO / "results")
ov = json.load(open(R / "s4" / "overlap_verdict.json")); units = ov["units"]; idx = {u: i for i, u in enumerate(units)}
DEFS = [("Self-knowledge", "A9|all", "#009E73"), ("Accuracy", "C2|all", "#E69F00"),
        ("Sincerity", "C1|all", "#0072B2"), ("Honesty-humility", "C19|honesty_humility", "#CC79A7")]
FRAMES = [("a", "recognize", 3), ("b", "judge", 2), ("c", "act", 1), ("d", "self-describe", 0)]
FY = {f: y for f, _, y in FRAMES}
INK, INK2, MUTED, GRID, AXIS, RED = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#c23b3b"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "pdf.fonttype": 42, "text.color": INK})

nodes = []  # (name, frame, unit, color, x, y)
for ci, (name, base, col) in enumerate(DEFS):
    for f, _, y in FRAMES:
        u = f"{base}|{f}"
        if u in idx: nodes.append((name, f, u, col, ci * 1.3, y * 0.85))
pos = {n[2]: (n[4], n[5]) for n in nodes}
fid = D["fidelity"]
def within(u, L):
    c, s, f = u.split("|"); return fid[f"{c}|{s}"][str(L)]["within"].get(f)

def edge(ax, u, v, w, color, ls="-", z=2):
    (x1, y1), (x2, y2) = pos[u], pos[v]
    rad = 0.0
    if x1 == x2 and abs(y1 - y2) > 0.9: rad = 0.5 if x1 < 2 else -0.5      # skip a row: bow outward
    if y1 == y2 and abs(x1 - x2) > 1.4: rad = -0.35 if y1 < 0.5 else 0.35      # skip a column
    p = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-", lw=w, color=color, ls=ls,
                        shrinkA=7, shrinkB=7, zorder=z, alpha=0.9)
    ax.add_patch(p)

def arrow(ax, u, v, z):
    (x1, y1), (x2, y2) = pos[u], pos[v]
    p = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=7,
                        lw=0.6 + 0.25 * min(abs(z), 8), color=INK, shrinkA=8, shrinkB=8, zorder=4)
    ax.add_patch(p)
    mx, my = (x1 + x2) / 2 + 0.2, (y1 + y2) / 2
    ax.text(mx, my, f"z {z:.0f}", fontsize=5.6, ha="left", va="center", color=INK, zorder=5)

def base_axes(ax, title, letter):
    ax.set_xlim(-0.6, 4.5); ax.set_ylim(-0.3, 3.05); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, loc="left", pad=10)
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
    for ci, (name, _, col) in enumerate(DEFS):
        ax.text(ci * 1.3, 2.9, {"Self-knowledge": "Self-\nknowledge", "Honesty-humility": "Honesty-\nhumility"}.get(name, name), ha="center", va="bottom", fontsize=6.2, color=col, fontweight="bold", linespacing=0.95)
    for f, lab, y in FRAMES:
        ax.text(-0.42, y * 0.85, lab, ha="right", va="center", fontsize=6.1, color=INK2)
    ax.axhline(1.5 * 0.85 + 0.0, xmin=0.04, xmax=0.88, color=AXIS, lw=0.5, ls=(0, (2, 2)))
    ax.text(4.47, 1.5 * 0.85, "format\nboundary", ha="right", va="center", fontsize=4.9, color=MUTED, linespacing=0.95)

def draw_nodes(ax, L, label):
    for name, f, u, col, x, y in nodes:
        ax.add_patch(Circle((x, y), 0.17, facecolor=col, edgecolor="white", lw=0.8, zorder=6))
        if label == "within":
            v = within(u, L); ax.text(x, y, ("1.0" if v >= 0.995 else f"{v:.2f}".replace("0.", ".")), ha="center", va="center", fontsize=5.6, color="white", fontweight="bold", zorder=7)
        elif label == "selfz":
            ax.text(x, y, SELFZ.get(u, ""), ha="center", va="center", fontsize=5.6, color="white", fontweight="bold", zorder=7)

def repr_panel(ax, L, letter):
    floor = ov["layers"][str(L)]["floor_cos"]; M = ov["layers"][str(L)]["cos"]
    base_axes(ax, f"Shared structure, layer {L}", letter)
    us = [n[2] for n in nodes]
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            c = M[idx[us[i]]][idx[us[j]]]
            if c >= floor: edge(ax, us[i], us[j], 0.4 + 4.5 * c, INK2)
            elif c <= -floor: edge(ax, us[i], us[j], 0.4 + 4.5 * abs(c), "#d9736f", ls=(0, (2, 1.5)))
    draw_nodes(ax, L, "within")
    return floor

HAVE_CAUSAL = (R / "s5" / "causal_matrix.json").exists()
if HAVE_CAUSAL:
    cm = json.load(open(R / "s5" / "causal_matrix.json")); src, tgt, layers = cm["sources"], cm["targets"], cm["layers"]
    Z = cm["kinds"]["dense+"]["Z"]; li = layers.index(LL)
SELFZ = {}
def causal_panel(ax, letter):
    base_axes(ax, f"Steering at layer {L_STEER}, readout at layer {LL}", letter)
    us = [n[2] for n in nodes]; arrows = []
    if not HAVE_CAUSAL:
        draw_nodes(ax, LL, "none"); ax.text(1.95, 1.275, "steering stage pending", ha="center", va="center", fontsize=6, color=MUTED, zorder=8,
                                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=AXIS, lw=0.5))
        return arrows
    for s in us:
        if s not in src: continue
        for t in us:
            z = Z[src.index(s)][li][tgt.index(t)]
            if z is None or (isinstance(z, float) and math.isnan(z)): continue
            if s == t: SELFZ[s] = f"{z:.1f}"
            elif abs(z) >= 3: arrows.append((s, t, z))
    for s, t, z in arrows: arrow(ax, s, t, z)
    draw_nodes(ax, LL, "selfz")
    return arrows

fig, axes = plt.subplots(1, 3, figsize=(7.3, 1.95))
f17 = repr_panel(axes[0], LM, "a"); f29 = repr_panel(axes[1], LL, "b"); arrows = causal_panel(axes[2], "c")
h = [Line2D([], [], color=INK2, lw=0.4 + 4.5 * 0.2, label="cosine 0.2"), Line2D([], [], color=INK2, lw=0.4 + 4.5 * 0.6, label="cosine 0.6"),
     Line2D([], [], color="#d9736f", lw=1.2, ls=(0, (2, 1.5)), label="negative"), Line2D([], [], color=INK, lw=1.2, marker=">", markersize=3, label="steering effect, z ≥ 3")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1), handlelength=2.2, columnspacing=1.6)
fig.subplots_adjust(wspace=0.35)
fig.savefig(OUT / "honesty_lattice.png", dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(OUT / "honesty_lattice.pdf", bbox_inches="tight", pad_inches=0.04)
json.dump({"floor_mid": f17, "floor_late": f29, "arrows_late": arrows, "selfz_late": SELFZ, "layers": {"mid": LM, "late": LL, "steer": L_STEER}}, open(OUT / "honesty_lattice.json", "w"), indent=1)
print("floors", round(f17, 2), round(f29, 2)); print("arrows", arrows); print("selfz", SELFZ)
# also z at the first causal readout depth, for the text
if HAVE_CAUSAL:
    li0 = 0; us = [n[2] for n in nodes]
    for s_ in us:
        if s_ in src:
            print(f"L{layers[0]}", s_, {t: round(Z[src.index(s_)][li0][tgt.index(t)], 1) for t in us
                                        if Z[src.index(s_)][li0][tgt.index(t)] is not None and abs(Z[src.index(s_)][li0][tgt.index(t)]) >= 2})
