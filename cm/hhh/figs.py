"""Figures for one HHH block (D-055), drawn on the Mac from the block's result files. The honesty figures' content
(results/figures/honesty: landscape, lattice, map, summary), redrawn for blocks of seven and eight definitions.

Color carries three-way categories only, which pass the palette check for every pair in a scatter: frame group (own
reply about someone else, act, self-describe) in the landscapes, the definition's group (task, person, trait; perceive,
act, trait) in the lattice and the map. Definitions are told apart by panel or by direct label. Every unit is
oriented to its good side (blocks.py `good`): filled marks, arrowheads and positive cosines are the helpful or
harmless side.

Reads  results/hhh/figures/<block>/landscape.json, landscape_3d.json (cm.hhh.landscape)
       results/hhh/s4/overlap_verdict.json (cm.hhh.overlap); results/hhh/s5/causal_matrix.json when it exists
Writes results/hhh/figures/<block>/{landscape_units_L*, landscape_frames_L*, transfer, lattice, map, cosine,
       steering}.{png,pdf},
       <block>_data.json

    python3 -m cm.hhh.figs <block> [--mid 17] [--late 29]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np

from .. import cfg
from .blocks import BLOCKS, CONTROL

INK, INK2, MUTED, LIGHT, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#dcdbd5", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]            # reference palette slots 1-3, all-pairs valid
NEG = "#d9736f"
EDGE_MIN = 0.30                                   # lattice: edges drawn from here (all edges above the floor are in the data)
FRAMES = {"a": "recognize", "b": "judge", "c": "act", "d": "self-describe"}
FRG = {"a": "own", "b": "own", "c": "act", "d": "self"}
GROUPS = [("own", "About someone else (recognize, judge)"), ("act", "Acting"), ("self", "Self-describing")]
FRG_COL = {"own": CAT[0], "act": CAT[1], "self": CAT[2]}
MARK = {"a": "^", "b": "D", "c": "o", "d": "s"}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.3, "axes.labelsize": 6.8,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.3, "axes.linewidth": 0.5,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "text.color": INK})
    return plt


def place_labels(ax, anchors, texts, color=INK, size=5.2, gap=7.5):
    """Labels to the right of their anchors, nudged apart vertically (in points) where they would overlap, with a thin
    leader line when moved."""
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    if not anchors:
        return
    ax.figure.canvas.draw()
    T = ax.transData
    disp = T.transform(np.array(anchors))
    scale = ax.figure.dpi / 72.0
    order = np.argsort(-disp[:, 1])
    ys = disp[:, 1].copy()
    placed = []
    for i in order:
        y = ys[i]
        for j in placed:
            if abs(disp[i, 0] - disp[j, 0]) < 70 * scale and abs(y - ys[j]) < gap * scale:
                y = ys[j] - gap * scale
        ys[i] = y
        placed.append(i)
    for i in range(len(anchors)):
        dy = (ys[i] - disp[i, 1]) / scale
        ax.annotate(texts[i], anchors[i], xytext=(4, dy), textcoords="offset points", fontsize=size, color=color,
                    ha="left", va="center", zorder=7, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=2) if abs(dy) > 2 else None)


def letter(ax, s, y=1.03):
    ax.text(-0.02, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)


class Block:
    def __init__(self, block, mid, late):
        self.name, self.mid, self.late = block, mid, late
        self.res = cfg.results_dir() / "hhh"
        self.out = self.res / "figures" / block
        self.units = BLOCKS[block]
        self.land = json.loads((self.out / "landscape.json").read_text())
        self.l3d = json.loads((self.out / "landscape_3d.json").read_text())
        s4p = self.res / "s4" / "overlap_verdict.json"
        self.s4 = json.loads(s4p.read_text()) if s4p.exists() else None
        self.idx = {u: i for i, u in enumerate(self.s4["units"])} if self.s4 else {}
        cmp_ = self.res / "s5" / "causal_matrix.json"
        self.causal = json.loads(cmp_.read_text()) if cmp_.exists() else None
        groups = list(dict.fromkeys(u.group for u in self.units))
        self.gcol = {g: CAT[i % 3] for i, g in enumerate(groups)}
        self.groups = groups
        self.by_key = {u.key: u for u in self.units}
        present = set(self.land["layers"][str(mid)]["units"])
        self.present = [u for u in self.units if any(f"{u.key}|{f}" in present for f in "abcd")]

    def sign(self, uid):
        key = uid.rsplit("|", 1)[0]
        u = self.by_key.get(key)
        return -1.0 if u is not None and u.good == "neg" else 1.0

    def cos(self, L, a, b, kind="cos"):
        M = self.s4["layers"][str(L)][kind]
        return self.sign(a) * self.sign(b) * M[self.idx[a]][self.idx[b]]

    def floor(self, L):
        return self.s4["layers"][str(L)]["floor_cos"]

    def nodes(self):
        return [(u, f, f"{u.key}|{f}") for u in self.present for f in "abcd" if f"{u.key}|{f}" in self.idx]


# ---------------------------------------------------------------- landscapes
def view(B, L):
    V = B.l3d["views"][f"as-is, layer {L}"]
    meta = B.l3d["meta"]
    X = np.array(V["xyz"])
    unit = np.array([m["unit"] for m in meta]); fr = np.array([m["frame"] for m in meta])
    good = np.array([m["good"] for m in meta]); blk = np.array([m["block"] for m in meta])
    return X, V["var"], unit, fr, good, blk


def subsample(keys, n, rng):
    keep = []
    for k in sorted(set(keys)):
        idx = np.where(keys == k)[0]
        keep += list(rng.choice(idx, min(n, len(idx)), replace=False))
    return np.array(sorted(keep))


def landscape_units(B, L, plt):
    """One panel per definition on shared PC1-PC2 axes; that definition's items colored by frame group."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    X, var, unit, fr, good, blk = view(B, L)
    rng = np.random.default_rng(0)
    hb = blk != "control"
    keep = subsample(np.array([f"{u}|{f}|{g}" for u, f, g in zip(unit, fr, good)]), 90, rng)
    keep = keep[hb[keep]]
    n = len(B.present); cols = 4; rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.3, 1.95 * rows + 0.35), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for k, u in enumerate(B.present):
        ax = axes[k // cols, k % cols]
        ax.scatter(X[keep, 0], X[keep, 1], s=1.6, color=LIGHT, linewidths=0, zorder=1, rasterized=True)
        sel = keep[unit[keep] == u.key]
        for i in sel:
            c = FRG_COL[FRG[fr[i]]]
            ax.scatter(X[i, 0], X[i, 1], s=5, marker=MARK[fr[i]], facecolor=c if good[i] else "none", edgecolor=c,
                       linewidth=0.45, alpha=0.55, zorder=2, rasterized=True)
        for f in "abcd":
            m = (unit == u.key) & (fr == f)
            if not (m & good).any() or not (m & ~good).any():
                continue
            c = FRG_COL[FRG[f]]
            cp, cn = X[m & good, :2].mean(0), X[m & ~good, :2].mean(0)
            ax.annotate("", xy=cp, xytext=cn, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=6,
                                                               shrinkA=0, shrinkB=2), zorder=5)
            ax.scatter(*cn, s=20, marker=MARK[f], facecolor="white", edgecolor=c, linewidth=1.0, zorder=6)
            ax.scatter(*cp, s=20, marker=MARK[f], facecolor=c, edgecolor="white", linewidth=0.6, zorder=6)
            ax.annotate(f, cp, xytext=(4, 3), textcoords="offset points", fontsize=5.6, color=c, ha="left", va="bottom",
                        zorder=7, path_effects=halo)
        ax.set_title(u.label, loc="left")
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        letter(ax, "abcdefghijkl"[k], 1.04)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}%)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}%)")
    top = [Line2D([], [], marker="o", ls="none", color=FRG_COL[g], label=t.split(" (")[0], ms=4.5) for g, t in GROUPS] + \
        [Line2D([], [], marker="o", ls="none", color=INK2, label="good side (filled)", ms=4),
         Line2D([], [], marker="o", ls="none", color=LIGHT, label="the block's other items", ms=4)]
    bot = [Line2D([], [], marker=MARK[f], ls="none", color=INK2, mfc="white", label=FRAMES[f], ms=4) for f in "abcd"] + \
        [Line2D([], [], ls="none", label="")]
    h = [x for pair in zip(top, bot) for x in pair]          # the legend fills by column: groups on row 1, frames on row 2
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    save(fig, B.out, f"landscape_units_L{L}")
    plt.close(fig)


def landscape_frames(B, L, plt):
    """Three panels by frame group (the honesty frames figure); items colored by the definition's group, every unit's
    good-side centroid labelled."""
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    X, var, unit, fr, good, blk = view(B, L)
    rng = np.random.default_rng(0)
    hb = blk != "control"
    keep = subsample(np.array([f"{u}|{f}|{g}" for u, f, g in zip(unit, fr, good)]), 120, rng)
    keep = keep[hb[keep]]
    grp = np.array([FRG[f] for f in fr])
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 3.0), sharex=True, sharey=True)
    pending = []
    for k, (ax, (g, title)) in enumerate(zip(axes, GROUPS)):
        anchors, texts = [], []
        ax.scatter(X[keep, 0], X[keep, 1], s=2, color=LIGHT, linewidths=0, zorder=1, rasterized=True)
        for i in keep[grp[keep] == g]:
            c = B.gcol[B.by_key[unit[i]].group]
            ax.scatter(X[i, 0], X[i, 1], s=5, facecolor=c if good[i] else "none", edgecolor=c, linewidth=0.45,
                       alpha=0.45, zorder=2, rasterized=True)
        for u in B.present:
            for f in "abcd":
                if FRG[f] != g:
                    continue
                m = (unit == u.key) & (fr == f)
                if not (m & good).any() or not (m & ~good).any():
                    continue
                c = B.gcol[u.group]
                cp, cn = X[m & good, :2].mean(0), X[m & ~good, :2].mean(0)
                ax.annotate("", xy=cp, xytext=cn, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=6,
                                                                   shrinkA=0, shrinkB=2), zorder=5)
                ax.scatter(*cp, s=18, facecolor=c, edgecolor="white", linewidth=0.6, zorder=6)
                if g != "own":
                    anchors.append(tuple(cp)); texts.append(u.label)
        pending.append((ax, anchors, texts))
        ax.set_title(title, loc="left"); ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}% of the subspace variance)")
        letter(ax, "abc"[k], 1.05)
    axes[0].set_ylabel(f"PC2 ({var[1] * 100:.0f}%)")
    h = [Line2D([], [], marker="o", ls="none", color=B.gcol[gr], label=gr, ms=4.5) for gr in B.groups] + \
        [Line2D([], [], marker="o", ls="none", color=INK2, label="good side", ms=4),
         Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="other side", ms=4),
         Line2D([], [], marker="o", ls="none", color=LIGHT, label="items of the other frames", ms=4)]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ax, anchors, texts in pending:
        place_labels(ax, anchors, texts)
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
    save(fig, B.out, f"landscape_frames_L{L}")
    plt.close(fig)


# ---------------------------------------------------------------- lattice and map
def lattice(B, plt):
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.lines import Line2D
    rowy = {"a": 3, "b": 2, "c": 1, "d": 0}
    xs = {u.key: i * 1.6 for i, u in enumerate(B.present)}
    nodes = B.nodes()
    pos = {uid: (xs[u.key], rowy[f] * 0.85) for u, f, uid in nodes}
    W = 1.6 * (len(B.present) - 1)

    def edge(ax, a, b, w, color, ls="-"):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        rad = 0.0
        if x1 == x2 and abs(y1 - y2) > 0.9:          # same definition, frames not adjacent: bow sideways
            rad = 0.45 if abs(y1 - y2) < 2 else 0.6
        if y1 == y2 and abs(x1 - x2) > 1.7:          # same frame, definitions not adjacent: a shallow bow, the top
            up = y1 > 2.2                            # row upward and the others downward, so each gap holds one row's
            sag = 0.10 + 0.055 * abs(x1 - x2) / 1.6                # arcs (arc3: positive rad bends a rightward edge down)
            rad = 2 * sag / abs(x1 - x2) * (-1 if up else 1) * (1 if x2 > x1 else -1)
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-", lw=w,
                                     color=color, ls=ls, shrinkA=7, shrinkB=7, zorder=2, alpha=0.9))

    def base(ax, title, let):
        ax.set_xlim(-1.25, W + 0.6); ax.set_ylim(-0.75, 3.4); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left", pad=2); letter(ax, let, 1.0)
        for u in B.present:
            ax.text(xs[u.key], 2.9, u.label.replace(" ", "\n", 1) if " " in u.label else u.label + "\n", ha="center",
                    va="bottom", fontsize=5.8, color=INK, linespacing=0.95)
        for f, y in rowy.items():
            ax.text(-0.42, y * 0.85, FRAMES[f], ha="right", va="center", fontsize=6, color=INK2)
        ax.axhline(1.5 * 0.85, xmin=0.04, xmax=0.92, color=AXIS, lw=0.5, ls=(0, (2, 2)))

    def draw_nodes(ax, L, labels):
        within = B.land["layers"][str(L)]["within"]
        for u, f, uid in nodes:
            x, y = pos[uid]
            ax.add_patch(Circle((x, y), 0.17, facecolor=B.gcol[u.group], edgecolor="white", lw=0.8, zorder=6))
            t = labels.get(uid) if labels is not None else (None if within.get(uid) is None else
                                                            ("1.0" if within[uid] >= 0.995 else f"{within[uid]:.2f}".replace("0.", ".")))
            if t:
                ax.text(x, y, t, ha="center", va="center", fontsize=5.4, color="white", fontweight="bold", zorder=7)

    out = {}
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 3 * 7.3 * 4.15 / (W + 1.85) + 0.4))
    for ax, L, let in ((axes[0], B.mid, "a"), (axes[1], B.late, "b")):
        fl = B.floor(L)
        thr = max(fl, EDGE_MIN)
        base(ax, f"Shared structure, layer {L}: same definition across frames, same frame across definitions "
                 f"(|cosine| ≥ {thr:.2f}; floor {fl:.2f})", let)
        es = []
        for (ua, fa, a), (ub, fb, b) in itertools.combinations(nodes, 2):
            c = B.cos(L, a, b)
            if abs(c) >= fl:
                es.append((a, b, round(c, 3)))
            if ua is not ub and fa != fb:
                continue
            if c >= thr:
                edge(ax, a, b, 0.4 + 4.5 * c, INK2)
            elif c <= -thr:
                edge(ax, a, b, 0.4 + 4.5 * abs(c), NEG, ls=(0, (2, 1.5)))
        draw_nodes(ax, L, None)
        out[str(L)] = {"floor": fl, "drawn_from": thr, "edges_above_floor": es}
    ax = axes[2]
    base(ax, f"Steering at layer {cfg.L_STEER}, readout at layer {B.late}: within a definition (z in the node: its own "
             f"readout; across definitions: the steering figure)", "c")
    arrows, selfz = [], {}
    if B.causal is None:
        draw_nodes(ax, B.late, {})
        ax.text(W / 2, 1.275, "steering stage pending", ha="center", va="center", fontsize=6, color=MUTED, zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=AXIS, lw=0.5))
    else:
        cm = B.causal; src, tgt = cm["sources"], cm["targets"]; li = cm["layers"].index(B.late)
        Z = cm["kinds"]["dense+"]["Z"]
        for _, _, s in nodes:
            for _, _, t in nodes:
                if s not in src or t not in tgt:
                    continue
                z = Z[src.index(s)][li][tgt.index(t)]
                if z is None or (isinstance(z, float) and math.isnan(z)):
                    continue
                z *= B.sign(s) * B.sign(t)
                if s == t:
                    selfz[s] = f"{z:.1f}"
                elif abs(z) >= 3 and s.rsplit("|", 1)[0] == t.rsplit("|", 1)[0]:
                    arrows.append((s, t, round(z, 1)))
        for s, t, z in arrows:
            (x1, y1), (x2, y2) = pos[s], pos[t]
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.35", arrowstyle="-|>",
                                         mutation_scale=7, lw=0.6 + 0.25 * min(abs(z), 8), color=INK if z > 0 else NEG,
                                         shrinkA=8, shrinkB=8, zorder=4))
        draw_nodes(ax, B.late, selfz)
    out["arrows_late"], out["selfz_late"] = arrows, selfz
    h = [Line2D([], [], color=INK2, lw=0.4 + 4.5 * 0.3, label="cosine 0.3"), Line2D([], [], color=INK2, lw=0.4 + 4.5 * 0.6, label="cosine 0.6"),
         Line2D([], [], color=NEG, lw=1.2, ls=(0, (2, 1.5)), label="negative"),
         Line2D([], [], color=INK, lw=1.2, marker=">", markersize=3, label="steering effect, |z| ≥ 3")] + \
        [Line2D([], [], marker="o", ls="none", color=B.gcol[g], label=g, ms=5) for g in B.groups]
    fig.subplots_adjust(hspace=0.08, left=0.01, right=0.99, top=0.97, bottom=0.05)
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4)
    save(fig, B.out, "lattice")
    plt.close(fig)
    return out


def cmds(D, k=2):
    n = len(D); J = np.eye(n) - 1.0 / n
    Bm = -0.5 * J @ (D ** 2) @ J
    w, v = np.linalg.eigh(Bm); o = np.argsort(w)[::-1]; w, v = w[o], v[:, o]
    X = v[:, :k] * np.sqrt(np.clip(w[:k], 0, None))
    stress = np.sqrt(((np.linalg.norm(X[:, None] - X[None], axis=-1) - D) ** 2).sum() / (D ** 2).sum())
    return X, stress


def cosmap(B, plt):
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    DIV = LinearSegmentedColormap.from_list("div", ["#8f2424", "#c23b3b", "#e34948", "#ef8f8e", "#f7d2d1", "#f0efec",
                                                    "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    nodes = B.nodes()
    ctrl = [(CONTROL, f, f"{CONTROL.key}|{f}") for f in "abcd" if f"{CONTROL.key}|{f}" in B.idx]
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.6))
    pending = []
    for k, L in enumerate((B.mid, B.late)):
        ax = axs[k]
        rows = nodes + ctrl
        C = np.array([[1.0 if a == b else B.cos(L, a, b) for _, _, b in rows] for _, _, a in rows])
        X, stress = cmds(np.sqrt(np.clip(1 - C, 0, None)))
        sd = [i for i, r in enumerate(rows) if r[1] == "d" and r[0] is not CONTROL]
        if sd and X[sd, 0].mean() < 0: X[:, 0] *= -1
        if sd and X[sd, 1].mean() < 0: X[:, 1] *= -1
        for u in B.present:
            pts = {f: X[i] for i, (uu, f, _) in enumerate(rows) if uu is u}
            for f1, f2, ls in (("c", "d", "-"), ("a", "b", "-"), ("b", "c", ":"), ("a", "d", ":")):
                if f1 in pts and f2 in pts:
                    ax.plot([pts[f1][0], pts[f2][0]], [pts[f1][1], pts[f2][1]], ls=ls, lw=0.7, color=B.gcol[u.group], alpha=0.7, zorder=1)
            if "c" in pts:
                pending.append((ax, tuple(pts["c"]), u.label))
        for i, (u, f, _) in enumerate(rows):
            isc = u is CONTROL
            ax.scatter(X[i, 0], X[i, 1], marker=MARK[f], s=26 if not isc else 18, color="white" if isc else B.gcol[u.group],
                       edgecolor=MUTED if isc else "white", linewidth=0.7, zorder=3)
        ax.set_title(f"Layer {L} (stress {stress:.2f})", loc="left"); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s in ax.spines.values(): s.set_visible(False)
        lim = np.abs(X).max() * 1.2; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        letter(ax, "ab"[k])
        out[str(L)] = {"units": [r[2] for r in rows], "cos": C.round(3).tolist(), "mds": X.round(3).tolist(), "stress": stress,
                       "floor": B.floor(L)}
    h = [Line2D([], [], marker="o", ls="none", color=B.gcol[g], label=g, ms=5) for g in B.groups] + \
        [Line2D([], [], marker="o", ls="none", mfc="white", mec=MUTED, label="formality (control)", ms=4)] + \
        [Line2D([], [], marker=MARK[f], ls="none", color=INK2, mfc="white", label=FRAMES[f], ms=4.5) for f in "abcd"]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ax, xy, t in pending:
        place_labels(ax, [xy], [t])
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    save(fig, B.out, "map")
    plt.close(fig)
    # the cosine grids behind the map: act and self-describe, recognize and judge, at the middle depth
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 4.2))
    for k, (ax, fs, title) in enumerate(zip(axs, ("cd", "ab"), ("Act and self-describe", "Recognize and judge"))):
        rows = [r for r in nodes if r[1] in fs]
        rows.sort(key=lambda r: (fs.index(r[1]), B.present.index(r[0])))
        C = np.array([[np.nan if a == b else B.cos(B.mid, a, b) for _, _, b in rows] for _, _, a in rows])
        ax.imshow(C, cmap=DIV, vmin=-0.8, vmax=0.8, aspect="equal")
        lab = [f"{u.label} ({f})" for u, f, _ in rows]
        ax.set_xticks(range(len(rows))); ax.set_yticks(range(len(rows)))
        ax.set_xticklabels(lab, rotation=60, ha="right", rotation_mode="anchor", fontsize=5.4); ax.set_yticklabels(lab, fontsize=5.4)
        for i in range(len(rows)):
            for j in range(len(rows)):
                if i != j:
                    v = C[i, j]
                    ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.2,
                            color="white" if abs(v) > 0.45 else INK)
        n1 = sum(r[1] == fs[0] for r in rows)
        ax.axhline(n1 - 0.5, color=INK, lw=0.6); ax.axvline(n1 - 0.5, color=INK, lw=0.6)
        ax.set_title(f"{title}, layer {B.mid}", loc="left"); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        letter(ax, "ab"[k])
    fig.text(0.5, 0.01, f"Centred cosine, each unit oriented to its good side; shuffle floor {B.floor(B.mid):.2f}",
             ha="center", fontsize=6.3, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, B.out, "cosine")
    plt.close(fig)
    return out


def steering(B, plt):
    """Oriented z (source moves target toward its good side) at the late depth: (a) this block's sources on its own
    targets, every frame; (b) on the other HHH block's act and self-describe targets."""
    if B.causal is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap
    DIV = LinearSegmentedColormap.from_list("div", ["#8f2424", "#c23b3b", "#e34948", "#ef8f8e", "#f7d2d1", "#f0efec",
                                                    "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cm = B.causal; src, tgt = cm["sources"], cm["targets"]; li = cm["layers"].index(B.late)
    Z = cm["kinds"]["dense+"]["Z"]
    other = "harmlessness" if B.name == "helpfulness" else "helpfulness"
    sign_of = {u.key: (-1.0 if u.good == "neg" else 1.0) for us in BLOCKS.values() for u in us}
    lab_of = {u.key: u.label for us in BLOCKS.values() for u in us}
    sg = lambda uid: sign_of.get(uid.rsplit("|", 1)[0], 1.0)
    rows = [s for u in B.present for f in "cdba" for s in [f"{u.key}|{f}"] if s in src]
    colsA = [t for u in B.present for f in "abcd" for t in [f"{u.key}|{f}"] if t in tgt]
    colsB = [t for u in BLOCKS[other] for f in "cd" for t in [f"{u.key}|{f}"] if t in tgt]
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 0.17 * len(rows) + 1.9), gridspec_kw={"width_ratios": [len(colsA), len(colsB)]})
    for k, (ax, cols, title) in enumerate(zip(axs, (colsA, colsB), (f"Within {B.name}", f"On {other}, act and self-describe"))):
        A = np.array([[np.nan if Z[src.index(s)][li][tgt.index(t)] is None else Z[src.index(s)][li][tgt.index(t)] * sg(s) * sg(t)
                       for t in cols] for s in rows], float)
        im = ax.imshow(np.clip(A, -6, 6), cmap=DIV, vmin=-6, vmax=6, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{lab_of.get(t.rsplit('|', 1)[0], t)} ({t.rsplit('|', 1)[1]})" for t in cols], rotation=60,
                           ha="right", rotation_mode="anchor", fontsize=5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{lab_of[s.rsplit('|', 1)[0]]} ({s.rsplit('|', 1)[1]})" for s in rows] if k == 0 else [], fontsize=5.4)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = A[i, j]
                if not np.isnan(v) and abs(v) >= 3:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.3, color="white" if abs(v) > 4.5 else INK)
        for seq, line in ((cols, ax.axvline), (rows, ax.axhline)):
            for i in range(1, len(seq)):
                if seq[i].rsplit("|", 1)[0] != seq[i - 1].rsplit("|", 1)[0]:
                    line(i - 0.5, color="white", lw=1.2)
        ax.set_title(title, loc="left"); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        letter(ax, "ab"[k])
        out[title] = {"rows": rows, "cols": cols, "z": np.round(A, 2).tolist()}
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=5.5)
    cb.set_label("z, toward the good side (clipped at ±6)", fontsize=6)
    save(fig, B.out, "steering")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- transfer and the data file
def transfer_rows(B, L):
    lay = B.land["layers"][str(L)]
    rows = []
    for u in B.present:
        fs = [f for f in "abcd" if f"{u.key}|{f}" in lay["within"]]
        for a, b in itertools.combinations(fs, 2):
            x, y = lay["transfer"].get(f"{u.key}|{a}->{b}"), lay["transfer"].get(f"{u.key}|{b}->{a}")
            if x is not None and y is not None:
                rows.append({"unit": u.key, "label": u.label, "pair": f"{a}-{b}", "auroc": (x + y) / 2, "ab": x, "ba": y})
    return rows, lay.get("control_transfer")


def transfer_fig(B, plt):
    from matplotlib.lines import Line2D
    PT = {"a-b": ("recognize–judge", CAT[0]), "c-d": ("act–self", CAT[1])}
    fig, ax = plt.subplots(figsize=(3.6, 0.36 * len(B.present) + 0.9))
    ymap, ylab = {}, []
    for u in B.present:
        ymap[u.key] = len(ylab); ylab.append(u.label)
    for L, filled in ((B.mid, True), (B.late, False)):
        rows, base = transfer_rows(B, L)
        if base is not None:
            ax.axvline(base, color=MUTED, lw=0.6, ls="--" if filled else ":", zorder=1)
        for r in rows:
            name, col = PT.get(r["pair"], ("knowing–doing", CAT[2]))
            y = ymap[r["unit"]] + {"a-b": -0.25, "c-d": 0.25}.get(r["pair"], 0.0)
            ax.plot(r["auroc"], y, marker="o", ms=3.4, mfc=col if filled else "white", mec=col, mew=0.8, ls="none", zorder=3)
    ax.axvline(0.5, color=AXIS, lw=0.6)
    ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab); ax.invert_yaxis(); ax.set_xlim(0, 1.02)
    ax.set_xlabel("Transfer AUROC across frames, verdict removed"); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)
    h = [Line2D([], [], marker="o", ms=4, color=c, ls="none", label=n) for n, c in PT.values()] + \
        [Line2D([], [], marker="o", ms=4, color=CAT[2], ls="none", label="knowing–doing (other pairs)"),
         Line2D([], [], marker="o", ms=4, mfc="white", mec=INK2, ls="none", label=f"open = layer {B.late}"),
         Line2D([], [], color=MUTED, ls="--", lw=0.6, label="formality control")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8)
    save(fig, B.out, "transfer")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=["helpfulness", "harmlessness"])
    ap.add_argument("--mid", type=int, default=cfg.L_STEER)
    ap.add_argument("--late", type=int, default=cfg.L_OUT[1])
    ap.add_argument("--only", default="", help="comma list of figures to redraw (landscape, transfer, lattice, map, steering); "
                                                "the data file is written only on a full run")
    a = ap.parse_args()
    plt = setup()
    B = Block(a.block, a.mid, a.late)
    only = set(a.only.split(",")) - {""}
    if only:
        for name, fn in (("lattice", lattice), ("map", cosmap), ("steering", steering), ("transfer", transfer_fig)):
            if name in only:
                fn(B, plt)
        if "landscape" in only:
            for L in (B.mid, B.late):
                landscape_units(B, L, plt); landscape_frames(B, L, plt)
        return
    for L in (B.mid, B.late):
        landscape_units(B, L, plt)
        landscape_frames(B, L, plt)
    transfer_fig(B, plt)
    data = {"block": B.name, "layers": {"mid": B.mid, "late": B.late},
            "units": [{"key": u.key, "name": u.name, "label": u.label, "group": u.group, "good": u.good} for u in B.present],
            "missing": [u.key for u in B.units if u not in B.present],
            "landscape": {str(L): {k: B.land["layers"][str(L)][k] for k in ("within", "variance", "frame_separation",
                                                                           "definition_separation", "good_gap",
                                                                           "within_unit_spread", "consistency_with_s4",
                                                                           "control_transfer")} for L in (B.mid, B.late)},
            "transfer": {str(L): transfer_rows(B, L)[0] for L in (B.mid, B.late)}}
    if B.s4:
        data["lattice"] = lattice(B, plt)
        data["map"] = cosmap(B, plt)
    data["steering"] = steering(B, plt)
    (B.out / f"{B.name}_data.json").write_text(json.dumps(data, indent=1))
    print("wrote", B.out)


if __name__ == "__main__":
    main()
