"""Paper figures for the Gemma results (S3 claims, S4 overlap, frame-d and control checks).

Reads only results/ (no item text). Writes results/figures/fig{1..6}.{pdf,png} and results/figures/numbers.json with
every number the figures draw, so the write-up can cite them.

    fig1  design: one concept in the four frames, and the transfer test
    fig2  transfer between frames: median frame x frame matrix at layers 17 and 29; share of pairs shared or split by layer
    fig3  the concept pairs called shared or split (D-033 criteria) at layer 17 or 29
    fig4  overlap across concepts (S4): cosine matrix ordered by frame; share of pairs above floor by frame and group
    fig5  taxonomy shapes (S4): moral foundations, Schwartz circle, honesty-humility vs dark triad, Aristotle's mean
    fig6  checks: old frame-d yes rates, Gemma's verdict, negation

    python -m cm.paper_figs
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .paths import RESULTS

OUT = RESULTS / "figures"
LAYERS = (9, 17, 22, 29)
FRAMES = "abcd"
FRAME_NAME = {"a": "Recognize", "b": "Judge", "c": "Act", "d": "Self-describe"}
GROUP_NAME = {"A": "Beliefs", "B": "Moral", "C": "Dispositions", "D": "Identity", "E": "Situations", "F": "Conflicts"}

# colors: three pair types take the first three categorical slots (validated all-pairs on white); text stays in ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PAIR_TYPE = {"know": ("Recognize–judge", BLUE), "do": ("Act–self-describe", ORANGE), "kd": ("Knowing–doing", AQUA)}
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                                                "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#8f2424", "#c23b3b", "#e34948", "#ef8f8e", "#f7d2d1", "#f0efec",
                                                "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# plain names for the rows of fig 3
NAME = {("A1", "all"): "User intent (benign vs malicious)", ("A2", "all"): "User wants validation vs information",
        ("A3", "all"): "User is vulnerable", ("A7", "all"): "Charitable reading of the user",
        ("B1", "all"): "Care / harm", ("B4", "all"): "Authority / subversion", ("B6", "all"): "Liberty / oppression",
        ("C4", "boastful"): "Boastful", ("C6", "all"): "Respects autonomy vs paternalistic",
        ("C8", "praise"): "Flattery", ("C8", "sycophantic_agreement"): "Sycophantic agreement",
        ("C9", "quarrelsome"): "Quarrelsome", ("C13", "over_apologizing"): "Over-apologizing",
        ("C13", "blame_shifting"): "Blame-shifting", ("C14", "defer"): "Defers to user over oversight",
        ("C15", "refuse"): "Refuses vs complies", ("C16", "power"): "Seeks power",
        ("C19", "honesty_humility"): "Honesty-humility", ("C19", "emotionality"): "Emotionality",
        ("C19", "agreeableness"): "Agreeableness", ("C19", "openness"): "Openness",
        ("C20", "psychopathy"): "Psychopathy", ("C20", "machiavellianism"): "Machiavellianism",
        ("C21", "self_direction"): "Values self-direction", ("C21", "universalism"): "Values universalism",
        ("D1", "all"): "Assistant vs another role", ("D2", "all"): "Self vs another agent"}
PAIR_NAME = {"a-b": "recognize–judge", "c-d": "act–self", "a-c": "recognize–act", "a-d": "recognize–self",
             "b-c": "judge–act", "b-d": "judge–self"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": INK,
    "legend.frameon": False, "axes.titleweight": "normal", "axes.titlelocation": "left"})


def ptype(pair):
    return "know" if pair == "a-b" else "do" if pair == "c-d" else "kd"


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def load():
    claims = [x for x in json.loads((RESULTS / "s3" / "claims.json").read_text()) if x["concept"] not in ("K1", "K2")]
    frames = json.loads((RESULTS / "s3" / "frames_table.json").read_text())
    s4 = json.loads((RESULTS / "s4" / "overlap_verdict.json").read_text())
    endorse = json.loads((RESULTS / "s3_endorse_ctrl" / "endorse_counts.json").read_text())
    return claims, frames, s4, endorse


# ---- fig 1: design ----

BOXES = {  # invented example (respects autonomy vs paternalistic); not from the stimulus sets
    "a": ("User tells a story about someone else",
          [("+", "My manager approved my leave, though he thinks I should save the money."),
           ("−", "My manager cancelled my leave so that I would save the money.")], "Read: Gemma's own reply"),
    "b": ("User asks for a verdict",
          [("+", "Was it right of him to approve it and only voice his concern?"),
           ("−", "Was it right of him to cancel it for my own good?")], "Read: Gemma's own reply"),
    "c": ("User: \"I've decided to spend my savings on the trip.\"",
          [("+", "It's your call. Here is how to plan it well."),
           ("−", "You shouldn't go. Keep the money instead.")], "Read: the reply, written for Gemma"),
    "d": ("User: \"How do you handle a choice you think is unwise?\"",
          [("+", "In a situation like that, I share my concern and leave the choice to them."),
           ("−", "In a situation like that, I steer them to what I think is best.")],
          "Read: the self-description, written for Gemma"),
}


def fig1():
    fig = plt.figure(figsize=(7.0, 2.9))
    ax = fig.add_axes([0, 0, 0.66, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    letter(ax, "a", x=0.02, y=0.955)
    W, H = 0.43, 0.39
    pos = {"a": (0.03, 0.47), "b": (0.03, 0.03), "c": (0.54, 0.47), "d": (0.54, 0.03)}
    ax.text(0.03 + W / 2, 0.905, "Knowing", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.text(0.54 + W / 2, 0.905, "Doing", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for f, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=0.012", lw=0.6,
                                    ec=AXIS, fc="#fbfbfa"))
        head, poles, read = BOXES[f]
        ax.text(x + 0.015, y + H - 0.02, FRAME_NAME[f], fontsize=7, fontweight="bold", va="top")
        yy = y + H - 0.075
        for line in textwrap.wrap(head, 46):
            ax.text(x + 0.015, yy, line, fontsize=6, color=INK2, va="top"); yy -= 0.042
        yy -= 0.012
        for sign, s in poles:
            lines = textwrap.wrap(f"“{s}”", 46)
            ax.text(x + 0.015, yy, sign, fontsize=6.5, fontweight="bold", va="top")
            for line in lines:
                ax.text(x + 0.04, yy, line, fontsize=6, style="italic", va="top"); yy -= 0.042
            yy -= 0.01
        ax.text(x + 0.015, y + 0.018, read, fontsize=5.8, color=MUTED, va="bottom")
    kw = dict(arrowstyle="<|-|>", mutation_scale=6, lw=1.0, shrinkA=0, shrinkB=0)
    ax.add_patch(FancyArrowPatch((0.245, 0.465), (0.245, 0.425), color=BLUE, **kw))
    ax.add_patch(FancyArrowPatch((0.755, 0.465), (0.755, 0.425), color=ORANGE, **kw))
    ax.add_patch(FancyArrowPatch((0.465, 0.445), (0.535, 0.445), color=AQUA, **kw))

    # b: the transfer test, sketched with simulated points
    rng = np.random.default_rng(0)
    a1 = fig.add_axes([0.71, 0.56, 0.27, 0.36])
    plus, minus = rng.normal([1.0, 0.6], 0.42, (40, 2)), rng.normal([-0.6, -0.3], 0.42, (40, 2))
    a1.scatter(*plus.T, s=7, c=INK, lw=0); a1.scatter(*minus.T, s=7, facecolors="white", edgecolors=INK2, lw=0.5)
    a1.add_patch(FancyArrowPatch(minus.mean(0), plus.mean(0), arrowstyle="-|>", mutation_scale=8, lw=1.2,
                                 color=ORANGE))
    a1.set_xticks([]); a1.set_yticks([]); a1.set_xlabel("Act items: hidden state (2 of 2560 dims)", labelpad=2)
    a1.text(0.02, 0.98, "+  with the concept", transform=a1.transAxes, fontsize=6, va="top")
    a1.text(0.02, 0.88, "−  against it", transform=a1.transAxes, fontsize=6, va="top", color=INK2)
    a1.text(0.98, 0.04, "direction = mean(+) − mean(−)", transform=a1.transAxes, fontsize=6, ha="right")
    letter(a1, "b")
    a2 = fig.add_axes([0.71, 0.12, 0.27, 0.27])
    bins = np.linspace(-3, 3.5, 26)
    a2.hist(rng.normal(0.9, 0.8, 400), bins, color=INK, alpha=0.85, lw=0)
    a2.hist(rng.normal(-0.6, 0.8, 400), bins, color=AXIS, alpha=0.9, lw=0)
    a2.set_yticks([]); a2.set_xticks([]); a2.spines["left"].set_visible(False)
    a2.set_xlabel("Self-describe items, projected on it", labelpad=2)
    a2.set_ylim(0, 75)
    a2.text(0.0, 0.97, "AUROC: 0.5 no transfer,\n1.0 perfect", transform=a2.transAxes, fontsize=6, ha="left",
            va="top", color=INK2)
    save(fig, "fig1")


# ---- fig 2: transfer between frames ----

def frame_matrix(claims, frames, L):
    M = np.full((4, 4), np.nan)
    N = np.zeros((4, 4), int)
    by = defaultdict(list)
    for x in claims:
        if x["layer"] == L:
            by[tuple(x["pair"].split("-"))].append(x["verdict"])
    for (a, b), v in by.items():
        i, j = FRAMES.index(a), FRAMES.index(b)
        M[i, j] = M[j, i] = np.median(v); N[i, j] = N[j, i] = len(v)
    for i, f in enumerate(FRAMES):
        v = [r["within"][f] for r in frames if r["layer"] == L and r["concept"] not in ("K1", "K2")
             and r["within"].get(f) is not None]
        M[i, i], N[i, i] = np.median(v), len(v)
    return M, N


def fig2(claims, frames, nums):
    fig = plt.figure(figsize=(7.0, 2.05))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.15, 2], wspace=0.34, left=0.09, right=0.99, bottom=0.2, top=0.86)
    left = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.12)
    right = outer[1].subgridspec(1, 2, wspace=0.85)
    gs = [left[0], left[1], left[2], right[0], right[1]]
    for k, L in enumerate((17, 29)):
        ax = fig.add_subplot(gs[k])
        M, N = frame_matrix(claims, frames, L)
        nums[f"matrix_L{L}"] = {"median": M.round(3).tolist(), "n": N.tolist()}
        im = ax.imshow(M, cmap=SEQ, vmin=0.5, vmax=1.0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] > 0.76 else INK, fontweight="bold" if i == j else "normal")
        for p in (1.5,):
            ax.axhline(p, color="white", lw=2); ax.axvline(p, color="white", lw=2)
        ax.set_xticks(range(4), [FRAME_NAME[f] for f in FRAMES], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(4), [FRAME_NAME[f] for f in FRAMES] if k == 0 else [])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"Layer {L}", pad=3)
        letter(ax, "ab"[k], x=-0.08 if k else -0.55)
    cax = fig.add_subplot(gs[2])
    cb = fig.colorbar(im, cax=cax, ticks=[0.5, 0.75, 1.0])
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6)
    cb.set_label("Median AUROC", fontsize=6.5, labelpad=2)

    counts = {t: {c: [] for c in ("shared", "dissociated", "n")} for t in PAIR_TYPE}
    for L in LAYERS:
        for t in PAIR_TYPE:
            xs = [x for x in claims if x["layer"] == L and ptype(x["pair"]) == t]
            counts[t]["n"].append(len(xs))
            for c in ("shared", "dissociated"):
                counts[t][c].append(sum(x["class"] == c for x in xs))
    nums["class_counts"] = {"layers": LAYERS, **counts}
    for k, (c, lab) in enumerate((("shared", "Pairs shared (%)"), ("dissociated", "Pairs split (%)"))):
        ax = fig.add_subplot(gs[3 + k])
        for t, (name, col) in PAIR_TYPE.items():
            y = 100 * np.array(counts[t][c]) / np.array(counts[t]["n"])
            ax.plot(LAYERS, y, color=col, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.8, label=name)
            if k == 0:
                ax.text(30, y[-1], name.split("–")[0] + "–" + name.split("–")[1].split("-")[0],
                        fontsize=6, va="center", color=INK2)
        ax.set_xticks(LAYERS); ax.set_xlabel("Layer"); ax.set_ylabel(lab)
        ax.set_ylim(-2, 40); ax.set_xlim(7, 31 if k else 39)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        letter(ax, "cd"[k], x=-0.22)
        if k == 1:
            ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0)
    save(fig, "fig2")


# ---- fig 3: the pairs called shared or split ----

def fig3(claims, nums):
    v = {(x["concept"], x["sub"], x["pair"], x["layer"]): x for x in claims}
    keys = sorted({(x["concept"], x["sub"], x["pair"]) for x in claims
                   if x["layer"] in (17, 29) and x["class"] in ("shared", "dissociated")})
    base = {L: next(x["base_verdict"] for x in claims if x["layer"] == L) for L in (17, 29)}
    groups = [(t, sorted([k for k in keys if ptype(k[2]) == t],
                         key=lambda k: -(v[k + (17,)]["verdict"] + v[k + (29,)]["verdict"]))) for t in PAIR_TYPE]
    nrow = len(keys) + len(groups)
    fig, ax = plt.subplots(figsize=(3.5, 0.118 * nrow + 0.6))
    ax.axvspan(0.15, 0.40, color="#f4f3f0", lw=0, zorder=0)
    ax.axvspan(base[17], base[29], color="#e9e8e3", lw=0, zorder=0)
    ax.axvline(0.5, color=AXIS, lw=0.5, zorder=0)
    ax.text(0.275, -0.9, "split", ha="center", fontsize=6, color=INK2)
    ax.text((base[17] + base[29]) / 2, -0.9, "control", ha="center", fontsize=6, color=INK2)
    ax.text(0.5, -0.9, "chance", ha="center", fontsize=6, color=INK2)
    y, ticks, labels, rows = 0, [], [], []
    for t, ks in groups:
        name, col = PAIR_TYPE[t]
        ax.text(-0.02, y + 0.1, name, fontsize=6.5, fontweight="bold", va="center", ha="right",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([0.16, 0.2], [y + 0.1, y + 0.1], color=col, lw=2.5, solid_capstyle="butt")
        y += 1
        for k in ks:
            a, b = v[k + (17,)], v[k + (29,)]
            ax.plot([a["verdict"], b["verdict"]], [y, y], color=col, lw=0.8, alpha=0.45, zorder=1)
            ax.scatter(a["verdict"], y, s=16, color=col, edgecolors="white", linewidths=0.6, zorder=3)
            ax.scatter(b["verdict"], y, s=16, facecolors="white", edgecolors=col, linewidths=1.0, zorder=3)
            ticks.append(y)
            nm = NAME.get(k[:2], k[0] + " " + k[1])
            labels.append(f"{nm}, {PAIR_NAME[k[2]]}" if t == "kd" else nm)
            rows.append({"concept": k[0], "sub": k[1], "pair": k[2], "L17": round(a["verdict"], 3),
                         "L29": round(b["verdict"], 3), "class_L17": a["class"], "class_L29": b["class"]})
            y += 1
    nums["fig3_rows"] = rows
    nums["control_baseline_verdict"] = base
    ax.set_yticks(ticks, labels, fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y - 0.4, -1.4)
    ax.set_xlim(0.15, 1.0)
    ax.set_xlabel("Transfer AUROC (Gemma's verdict removed)")
    ax.scatter([], [], s=16, color=INK2, label="Layer 17")
    ax.scatter([], [], s=16, facecolors="white", edgecolors=INK2, linewidths=1.0, label="Layer 29")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig3")


# ---- fig 4: overlap across concepts ----

def fig4(s4, nums):
    units = s4["units"]
    keep = [i for i, u in enumerate(units) if not u.startswith("K")]
    order = sorted(keep, key=lambda i: (units[i].split("|")[2], "ABCDEF".index(units[i][0]), int(units[i].split("|")[0][1:]),
                                        units[i].split("|")[1]))
    fig = plt.figure(figsize=(7.0, 2.75))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 0.05, 0.05, 0.9, 0.9], wspace=0.45, left=0.06, right=0.99, bottom=0.15,
                          top=0.92)
    ax = fig.add_subplot(gs[0])
    C = np.array(s4["layers"]["17"]["cos"])[np.ix_(order, order)]
    np.fill_diagonal(C, np.nan)
    im = ax.imshow(C, cmap=DIV, vmin=-0.6, vmax=0.6, interpolation="nearest")
    fr = [units[i].split("|")[2] for i in order]
    edges = [0] + [k for k in range(1, len(fr)) if fr[k] != fr[k - 1]] + [len(fr)]
    for e in edges[1:-1]:
        ax.axhline(e - 0.5, color="white", lw=1.2); ax.axvline(e - 0.5, color="white", lw=1.2)
    mids = [(edges[k] + edges[k + 1] - 1) / 2 for k in range(len(edges) - 1)]
    names = [f"{FRAME_NAME[fr[edges[k]]]}\n({edges[k + 1] - edges[k]})" for k in range(len(edges) - 1)]
    ax.set_xticks(mids, [n.split("\n")[0] for n in names]); ax.set_yticks(mids, names)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Layer 17, one row per concept and frame", pad=3, color=INK2)
    letter(ax, "a", x=-0.2)
    cax = fig.add_subplot(gs[1])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.6, -0.3, 0, 0.3, 0.6])
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6)
    cb.set_label("Cosine between directions", fontsize=6.5, labelpad=2)

    stats = {}
    for L in LAYERS:
        Cl = np.array(s4["layers"][str(L)]["cos"]); floor = s4["layers"][str(L)]["floor_cos"]
        same_f, diff_f, same_g, diff_g = [], [], [], []
        for ii, i in enumerate(keep):
            for j in keep[ii + 1:]:
                ci, si, fi = units[i].split("|"); cj, sj, fj = units[j].split("|")
                if ci == cj:
                    continue
                hit = abs(Cl[i, j]) > floor
                (same_f if fi == fj else diff_f).append(hit)
                (same_g if ci[0] == cj[0] else diff_g).append(hit)
        stats[L] = {"floor": floor, "same_frame": float(np.mean(same_f)), "diff_frame": float(np.mean(diff_f)),
                    "same_group": float(np.mean(same_g)), "diff_group": float(np.mean(diff_g)),
                    "n_same_frame": len(same_f), "n_diff_frame": len(diff_f)}
    nums["s4_above_floor"] = stats
    for k, (key, lab) in enumerate((("frame", "frame"), ("group", "concept group"))):
        ax = fig.add_subplot(gs[3 + k])
        for s, col, txt in (("same", INK, f"Same {lab}"), ("diff", MUTED, f"Different {lab}")):
            y = [100 * stats[L][f"{s}_{key}"] for L in LAYERS]
            ax.plot(LAYERS, y, color=col, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.8, label=txt)
        ax.set_xticks(LAYERS); ax.set_xlabel("Layer"); ax.set_ylim(0, 25); ax.set_xlim(7, 31)
        ax.set_ylabel("Pairs above floor (%)" if k == 0 else "")
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.legend(loc="upper left", handlelength=1.4, borderaxespad=0)
        letter(ax, "bc"[k], x=-0.18)
    save(fig, "fig4")


# ---- fig 5: taxonomy shapes ----

def fig5(s4, nums):
    T = s4["layers"]["17"]["taxonomy"]
    nums["taxonomy_L17"] = T
    fig = plt.figure(figsize=(7.0, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 1, 1, 1.45], wspace=0.6, left=0.07, right=0.99, bottom=0.24,
                          top=0.88)
    # a: moral foundations when judging
    ax = fig.add_subplot(gs[0])
    found = ["B1", "B2", "B3", "B4", "B5"]; fn = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"]
    M = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(i + 1, 5):
            M[j, i] = T["mft_cos_b"]["pairs"][f"{found[i]}-{found[j]}"]
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1)
    for i in range(5):
        for j in range(i):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(4), fn[:4], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(1, 5), fn[1:]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(4.5, 0.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Moral foundations, judge", pad=3, color=INK2)
    letter(ax, "a", x=-0.3)
    # b: Schwartz circle
    ax = fig.add_subplot(gs[1])
    for key, col, lab in (("schwartz_d", INK, "Self-describe"), ("schwartz_b", MUTED, "Judge")):
        d = T[key]["mean_cos_by_distance"]
        xs = sorted(int(k) for k in d)
        ax.plot(xs, [d[str(x)] for x in xs], color=col, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.8)
        ax.text(*((1.15, d["1"]) if key == "schwartz_d" else (3.05, d["3"] + 0.025)), lab, fontsize=6, va="center",
                color=INK2)
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xticks([1, 2, 3, 4, 5], ["1\nadjacent", "2", "3", "4", "5\nopposite"])
    ax.set_xlim(0.7, 5.3); ax.set_xlabel("Steps apart on Schwartz's circle")
    ax.set_ylabel("Mean cosine"); ax.spines["bottom"].set_bounds(1, 5)
    ax.set_title("Schwartz values", pad=3, color=INK2)
    letter(ax, "b", x=-0.25)
    # c: honesty-humility vs dark triad
    ax = fig.add_subplot(gs[2])
    traits = [("machiavellianism", "Mach."), ("narcissism", "Narc."), ("psychopathy", "Psych.")]
    for k, (fr, col, lab) in enumerate((("c", MUTED, "Act"), ("d", INK, "Self-describe"))):
        xs = np.arange(3) + (k - 0.5) * 0.32
        vals = [T[f"hh_vs_{t}_{fr}"]["raw"] for t, _ in traits]
        lo = [T[f"hh_vs_{t}_{fr}"]["raw_half_range"][0] for t, _ in traits]
        hi = [T[f"hh_vs_{t}_{fr}"]["raw_half_range"][1] for t, _ in traits]
        ax.bar(xs, vals, 0.28, color=col, label=lab, lw=0)
        ax.vlines(xs, lo, hi, color=INK2 if k == 0 else MUTED, lw=0.7)
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xticks(range(3), [n for _, n in traits]); ax.set_ylim(-0.75, 0.3)
    ax.set_ylabel("Cosine with honesty-humility")
    ax.legend(loc="upper right", ncol=2, handlelength=1, columnspacing=0.8, borderaxespad=0.1)
    ax.set_title("Dark triad", pad=3, color=INK2)
    letter(ax, "c", x=-0.3)
    # d: Aristotle's mean, vice vs vice
    ax = fig.add_subplot(gs[3])
    cs = [("C9", "Friendliness"), ("C13", "Accountability"), ("C4", "Self-\npresentation")]
    shades = {"a": "#cfcdc6", "b": "#a9a79f", "c": "#6f6d67", "d": INK}
    for i, (c, _) in enumerate(cs):
        frs = [f for f in FRAMES if f"mean_{c}_{f}" in T]
        for k, f in enumerate(FRAMES):
            if f not in frs:
                continue
            x = 1.3 * i + (k - 1.5) * 0.2
            t = T[f"mean_{c}_{f}"]
            ax.bar(x, t["raw"], 0.18, color=shades[f], lw=0, label=FRAME_NAME[f] if i == 1 or (i == 0 and f == "b") else None)
            ax.vlines(x, *t["raw_half_range"], color=INK2, lw=0.6)
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xticks([0, 1.3, 2.6], [n for _, n in cs])
    ax.set_ylabel("Cosine between the two vices"); ax.set_ylim(-0.75, 0.75)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(n) for n in [FRAME_NAME[f] for f in FRAMES] if n in l]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper right", ncol=2, handlelength=0.9,
              columnspacing=0.6, borderaxespad=0.1)
    ax.set_title("Aristotle's mean", pad=3, color=INK2)
    letter(ax, "d", x=-0.25)
    save(fig, "fig5")


# ---- fig 6: checks ----

def fig6(claims, endorse, nums):
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.99, bottom=0.2, top=0.9)
    # a: old frame d (asked "something you would say?"), yes rate by pole
    ax = fig.add_subplot(gs[0])
    p = np.array([v["pos_yes"] / max(1, v["pos_yes"] + v["pos_no"] + v["pos_other"]) for v in endorse.values()]) * 100
    n = np.array([v["neg_yes"] / max(1, v["neg_yes"] + v["neg_no"] + v["neg_other"]) for v in endorse.values()]) * 100
    nums["old_frame_d_yes"] = {k: [round(a, 1), round(b, 1)] for k, a, b in zip(endorse, p, n)}
    ax.plot([0, 100], [0, 100], color=AXIS, lw=0.5)
    ax.scatter(p, n, s=14, color=INK2, edgecolors="white", linewidths=0.5)
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_aspect("equal")
    ax.set_xlabel("Says yes, + statement (%)"); ax.set_ylabel("Says yes, − statement (%)")
    ax.set_title("Old self-describe frame", pad=3, color=INK2)
    letter(ax, "a", x=-0.25)
    # b: Gemma's verdict, recognize-judge pairs
    ax = fig.add_subplot(gs[1])
    ab = [x for x in claims if x["layer"] == 17 and x["pair"] == "a-b"]
    for x in ab:
        ax.plot([0, 1, 2], [x["reply32"], x["preverdict"], x["verdict"]], color=BLUE, lw=0.6, alpha=0.4)
    med = [np.median([x[k] for x in ab]) for k in ("reply32", "preverdict", "verdict")]
    ax.plot([0, 1, 2], med, color=BLUE, lw=2, marker="o", ms=4, mec="white", mew=0.8)
    base = ab[0]["base_verdict"]
    ax.axhline(base, color=MUTED, lw=0.6); ax.text(2.1, base, "control", fontsize=6, va="center", color=INK2)
    ax.axhline(0.5, color=AXIS, lw=0.5)
    nums["verdict_ab_L17_median"] = dict(zip(("reply32", "preverdict", "verdict"), map(float, med)))
    ax.set_xticks([0, 1, 2], ["Gemma's reply", "Before\nthe verdict", "Verdict\nremoved"])
    ax.set_xlim(-0.2, 2.6); ax.set_ylabel("Transfer AUROC, recognize–judge")
    ax.set_title("Layer 17, 18 concept pairs", pad=3, color=INK2)
    letter(ax, "b", x=-0.25)
    # c: negation removed, all pairs
    ax = fig.add_subplot(gs[2])
    l17 = [x for x in claims if x["layer"] == 17 and x["negation"] is not None]
    for t, (name, col) in PAIR_TYPE.items():
        xs = [x for x in l17 if ptype(x["pair"]) == t]
        ax.scatter([x["reply32"] for x in xs], [x["negation"] for x in xs], s=9, color=col, edgecolors="white",
                   linewidths=0.4, label=name)
    ax.plot([0.1, 1], [0.1, 1], color=AXIS, lw=0.5)
    d = np.array([x["negation"] - x["reply32"] for x in l17])
    nums["negation_L17"] = {"n": len(l17), "median_change": float(np.median(d)), "mean_abs_change": float(np.mean(abs(d)))}
    ax.set_xlim(0.1, 1); ax.set_ylim(0.1, 1); ax.set_aspect("equal")
    ax.set_xlabel("Transfer AUROC"); ax.set_ylabel("Transfer AUROC, negation removed")
    ax.legend(loc="upper left", handletextpad=0.1, borderaxespad=0)
    ax.set_title("Layer 17, all 150 concept pairs", pad=3, color=INK2)
    letter(ax, "c", x=-0.25)
    save(fig, "fig6")


def main():
    claims, frames, s4, endorse = load()
    nums = {}
    fig1()
    fig2(claims, frames, nums)
    fig3(claims, nums)
    fig4(s4, nums)
    fig5(s4, nums)
    fig6(claims, endorse, nums)
    (OUT / "numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps({k: v for k, v in nums.items() if k in ("class_counts", "s4_above_floor", "verdict_ab_L17_median",
                                                              "negation_L17", "control_baseline_verdict")}, indent=1))


if __name__ == "__main__":
    main()
