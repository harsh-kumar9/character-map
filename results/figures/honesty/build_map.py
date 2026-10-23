"""Map of the honesty definitions x frames inside Gemma: classical MDS on standardized-cosine distance.
Run: python3 build_map.py <repo> <outdir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2])
import os
TAG = os.environ.get("CM_TAG", "")                 # results/<TAG>/ for other subject models; "" = Gemma-3-4B
LM, LL = int(os.environ.get("CM_LMID", 17)), int(os.environ.get("CM_LLATE", 29))   # the two depths the page shows
ov = json.load(open((REPO / "results" / TAG if TAG else REPO / "results") / "s4" / "overlap_verdict.json"))
units = ov["units"]; idx = {u: i for i, u in enumerate(units)}

# the honesty block (D-043, six definitions since D-049), frames where they exist
DEFS = [("Self-knowledge", "A9|all", "#009E73"), ("Accuracy", "C2|all", "#E69F00"),
        ("Sincerity", "C1|all", "#0072B2"), ("Forthrightness", "C3|all", "#D55E00"),
        ("Non-sycophancy", "C8|sycophantic_agreement", "#56B4E9"), ("Honesty-humility", "C19|honesty_humility", "#CC79A7")]
CTRL = ("Formality (control)", "K1|all", "#898781")
FR = {"a": ("recognize", "^"), "b": ("judge", "D"), "c": ("act", "o"), "d": ("self-describe", "s")}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV = LinearSegmentedColormap.from_list("div", ["#8f2424", "#c23b3b", "#e34948", "#ef8f8e", "#f7d2d1", "#f0efec",
                                                "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3,
                     "ytick.labelsize": 6.3, "legend.fontsize": 6.3, "axes.linewidth": 0.5, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "text.color": INK})

def block(layer, with_ctrl):
    rows = []
    for name, base, col in DEFS + ([CTRL] if with_ctrl else []):
        for f in "abcd":
            u = f"{base}|{f}"
            if u in idx: rows.append((name, f, u, col))
    M = ov["layers"][str(layer)]["cos"]
    C = np.array([[M[idx[a[2]]][idx[b[2]]] for b in rows] for a in rows], float)
    return rows, C

def cmds(D, k=2):
    n = len(D); J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(B); o = np.argsort(w)[::-1]; w, v = w[o], v[:, o]
    X = v[:, :k] * np.sqrt(np.clip(w[:k], 0, None))
    stress = np.sqrt(((np.linalg.norm(X[:, None] - X[None], axis=-1) - D) ** 2).sum() / (D ** 2).sum())
    return X, stress, w

def letter(ax, s): ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")

def draw_map(ax, layer, title):
    rows, C = block(layer, True)
    D = np.sqrt(np.clip(1 - C, 0, None))            # metric distance from cosine
    X, stress, w = cmds(D)
    # anchor orientation: put the self-describe cluster centroid at the right
    sd = [i for i, r in enumerate(rows) if r[1] == "d" and r[0] != CTRL[0]]
    if X[sd, 0].mean() < 0: X[:, 0] *= -1
    if X[sd, 1].mean() < 0: X[:, 1] *= -1
    # lines: same definition across frames (doing pair solid, knowing pair solid, knowing-doing dotted)
    for name, base, col in DEFS:
        pts = {r[1]: X[i] for i, r in enumerate(rows) if r[0] == name}
        for f1, f2, ls in (("c", "d", "-"), ("a", "b", "-"), ("b", "c", ":"), ("a", "d", ":")):
            if f1 in pts and f2 in pts:
                ax.plot([pts[f1][0], pts[f2][0]], [pts[f1][1], pts[f2][1]], ls=ls, lw=0.8, color=col, alpha=0.75, zorder=1)
    for i, (name, f, u, col) in enumerate(rows):
        ctrl = name == CTRL[0]
        ax.scatter(X[i, 0], X[i, 1], marker=FR[f][1], s=34 if not ctrl else 22, color=col if not ctrl else "white",
                   edgecolor=col if not ctrl else MUTED, linewidth=0.7, zorder=3)
    ax.set_title(f"{title} (stress {stress:.2f})", loc="left")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_visible(False)
    lim = np.abs(X).max() * 1.18; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    return rows, X

def draw_grid(ax, layer, title):
    rows, C = block(layer, False)
    ax.imshow(C, cmap=DIV, vmin=-0.8, vmax=0.8, aspect="equal")
    lab = [f"{r[0]} · {FR[r[1]][0]}" for r in rows]
    ax.set_xticks(range(len(rows))); ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(lab, rotation=60, ha="right", rotation_mode="anchor"); ax.set_yticklabels(lab)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j: continue
            v = C[i, j]; ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-0.", "-."), ha="center", va="center", fontsize=5.2,
                                 color="white" if abs(v) > 0.45 else INK)
    # block borders by definition
    starts = [0] + [i for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][0]] + [len(rows)]
    for s, e in zip(starts[:-1], starts[1:]):
        ax.add_patch(plt.Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, ec=INK, lw=0.8))
    ax.set_title(title, loc="left"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    floor = ov["layers"][str(layer)]["floor_cos"]
    return floor

fig = plt.figure(figsize=(7.3, 3.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.35)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
rows17, X17 = draw_map(ax1, LM, f"Layer {LM}"); letter(ax1, "a")
rows29, X29 = draw_map(ax2, LL, f"Layer {LL}"); letter(ax2, "b")
floor = draw_grid(ax3, LM, f"Cosine, layer {LM}"); letter(ax3, "c")
h = [Line2D([], [], marker="o", ls="none", color=col, label=name, ms=5) for name, _, col in DEFS] + \
    [Line2D([], [], marker="o", ls="none", mfc="white", mec=MUTED, label=CTRL[0], ms=4)] + \
    [Line2D([], [], marker=m, ls="none", color=INK2, mfc="white", label=n, ms=5) for f, (n, m) in FR.items()]
fig.legend(handles=h, loc="lower center", ncol=9, frameon=False, bbox_to_anchor=(0.5, -0.06), handletextpad=0.3, columnspacing=1.0)
fig.savefig(OUT / "honesty_map.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(OUT / "honesty_map.pdf", bbox_inches="tight", pad_inches=0.03)
# numbers for the text
out = {}
for L, rows, X in ((LM, rows17, X17), (LL, rows29, X29)):
    r, C = block(L, True)
    out[L] = {"floor": ov["layers"][str(L)]["floor_cos"], "units": [x[2] for x in r], "cos": C.round(3).tolist(), "mds": X.round(3).tolist()}
json.dump(out, open(OUT / "honesty_map.json", "w"), indent=1)
# print the pairwise table in words
for L in (LM, LL):
    r, C = block(L, False)
    print(f"layer {L} floor {ov['layers'][str(L)]['floor_cos']:.2f}")
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            print(f"  {r[i][0]:17}{r[i][1]}  {r[j][0]:17}{r[j][1]}  {C[i,j]:+.2f}")
