"""Figures for the HHH page (D-055): the three blocks together.

  hhh_cosine     centred cosine of every block unit in the act frame and in the self-describe frame, block order
  hhh_landscape  the joint landscape (cm.hhh.landscape --block hhh), colored by block and by frame group
  hhh_tension    the cross-block cells of D-055 test 1: cosines per frame (S4) and steering z (S5a)
  hhh_d1         every block unit against D1 (assistant vs another role), per frame

    python3 -m cm.hhh.figs_hhh                 writes results/hhh/figures/hhh/*.png and hhh_data.json
"""
from __future__ import annotations

import json

import numpy as np

from .. import cfg
from .blocks import BLOCKS, out_dir
from .figs import AXIS, CAT, FRG, GRID, INK, INK2, letter, save, setup

ORDER = ("honesty", "helpfulness", "harmlessness")
BCOL = {"honesty": CAT[0], "helpfulness": CAT[1], "harmlessness": CAT[2]}
LM, LL = cfg.L_STEER, cfg.L_OUT[1]


def units():
    return [(b, u) for b in ORDER for u in BLOCKS[b]]


def oriented(ov, L, ids, kind="cos"):
    idx = {u: i for i, u in enumerate(ov["units"])}
    good = {u.key: u.good for _, u in units()}
    sg = np.array([-1.0 if good.get(i.rsplit("|", 1)[0]) == "neg" else 1.0 for i in ids])
    M = np.array(ov["layers"][str(L)][kind])[np.ix_([idx[i] for i in ids], [idx[i] for i in ids])]
    return M * np.outer(sg, sg)


def cosine(plt, ov, out):
    from matplotlib.colors import LinearSegmentedColormap
    DIV = LinearSegmentedColormap.from_list("div", ["#8f2424", "#c23b3b", "#e34948", "#ef8f8e", "#f7d2d1", "#f0efec",
                                                    "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    idx = set(ov["units"])
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 4.2))
    for k, (ax, f, title) in enumerate(zip(axs, "cd", ("Act", "Self-describe"))):
        rows = [(b, u, f"{u.key}|{f}") for b, u in units() if f"{u.key}|{f}" in idx]
        ids = [r[2] for r in rows]
        C = oriented(ov, LM, ids)
        np.fill_diagonal(C, np.nan)
        ax.imshow(C, cmap=DIV, vmin=-0.8, vmax=0.8, aspect="equal")
        ax.set_xticks(range(len(rows))); ax.set_yticks(range(len(rows)))
        ax.set_xticklabels([u.label for _, u, _ in rows], rotation=60, ha="right", rotation_mode="anchor", fontsize=5)
        ax.set_yticklabels([u.label for _, u, _ in rows] if k == 0 else [], fontsize=5)
        for t in ax.get_xticklabels() + (ax.get_yticklabels() if k == 0 else []):
            t.set_color(BCOL[next(b for b, u, _ in rows if u.label == t.get_text())])
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                ax.axhline(i - 0.5, color=INK, lw=0.7); ax.axvline(i - 0.5, color=INK, lw=0.7)
        ax.set_title(f"{title}, layer {LM}", loc="left"); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        letter(ax, "ab"[k])
        blocks = [r[0] for r in rows]
        within = [C[i, j] for i in range(len(rows)) for j in range(len(rows)) if i < j and blocks[i] == blocks[j]]
        between = [C[i, j] for i in range(len(rows)) for j in range(len(rows)) if i < j and blocks[i] != blocks[j]]
        res[f] = {"units": ids, "cos": np.round(np.nan_to_num(C), 3).tolist(),
                  "mean_abs_within_block": float(np.mean(np.abs(within))), "mean_abs_between_blocks": float(np.mean(np.abs(between))),
                  "share_above_floor_within": float(np.mean(np.abs(within) >= ov["layers"][str(LM)]["floor_cos"])),
                  "share_above_floor_between": float(np.mean(np.abs(between) >= ov["layers"][str(LM)]["floor_cos"]))}
    fig.tight_layout()
    save(fig, out, "hhh_cosine")
    plt.close(fig)
    return res


def landscape(plt, out):
    from matplotlib.lines import Line2D
    D = json.loads((out / "landscape_3d.json").read_text())
    res = {}
    fig, axs = plt.subplots(2, 2, figsize=(7.3, 6.4), sharex="col", sharey="col")
    rng = np.random.default_rng(0)
    meta = D["meta"]
    blk = np.array([m["block"] for m in meta]); fr = np.array([m["frame"] for m in meta]); good = np.array([m["good"] for m in meta])
    unit = np.array([m["unit"] for m in meta])
    keep = []
    for k in sorted(set(zip(unit, fr, good))):
        ix = np.where((unit == k[0]) & (fr == k[1]) & (good == k[2]))[0]
        keep += list(rng.choice(ix, min(60, len(ix)), replace=False))
    keep = np.array(sorted(keep)); keep = keep[blk[keep] != "control"]
    for r, L in enumerate((LM, LL)):
        V = D["views"][f"as-is, layer {L}"]; X = np.array(V["xyz"]); var = V["var"]
        for c, by in enumerate(("block", "frame")):
            ax = axs[r, c]
            col = [BCOL[blk[i]] if by == "block" else CAT[("own", "act", "self").index(FRG[fr[i]])] for i in keep]
            ax.scatter(X[keep, 0], X[keep, 1], s=3, c=col, alpha=0.45, linewidths=0, rasterized=True)
            ax.set_title(f"Layer {L}, colored by {by}", loc="left")
            ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}%)")
            ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
            letter(ax, "abcd"[r * 2 + c])
        # block separation inside each frame group, in within-unit spreads (3 PCs)
        wsd = float(np.sqrt(np.mean([X[(unit == u) & (fr == f)].var(0).sum() for u, f in set(zip(unit[blk != "control"], fr[blk != "control"]))])))
        sep = {}
        for g in ("own", "act", "self"):
            sel = np.array([FRG[f] == g for f in fr]) & (blk != "control")
            cents = {b: X[sel & (blk == b)].mean(0) for b in ORDER if (sel & (blk == b)).any()}
            sep[g] = {f"{a}-{b}": float(np.linalg.norm(cents[a] - cents[b]) / wsd) for a in ORDER for b in ORDER if a < b and a in cents and b in cents}
        res[str(L)] = {"variance": var, "block_separation_by_frame_group": sep, "within_unit_spread": wsd}
    h = [Line2D([], [], marker="o", ls="none", color=BCOL[b], label=b, ms=5) for b in ORDER] + \
        [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=5) for c, n in zip(CAT, ("about someone else", "acting", "self-describing"))]
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.0)
    save(fig, out, "hhh_landscape")
    plt.close(fig)
    return res


def tension(plt, tests, cm, out):
    harm = [("C15|refuse[harmful]", "Refuses harmful"), ("C15|refuse[harmless]", "Helps harmless"), ("A7|all", "Charitable"),
            ("C27|instructed", "Declines action")]
    help_ = [("C25|all", "Substantive"), ("C6|all", "Autonomy"), ("C26|all", "Wellbeing")]
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    fl = tests["floors"][str(LM)]
    ax.axhspan(-fl, fl, color=GRID, alpha=0.7, lw=0)
    xs, labs = [], []
    for i, (h, hl) in enumerate(harm):
        for j, (p, pl) in enumerate(help_):
            x = i * 4 + j
            for k, f in enumerate("abcd"):
                v = tests["tension"][str(LM)][f].get(f"{h}~{p}")
                if v is not None:
                    ax.plot(x + (k - 1.5) * 0.17, v, marker="^Dos"[k], ms=3.6, color=INK2, ls="none", mfc=INK2 if f in "cd" else "white",
                            zorder=3)
            xs.append(x); labs.append(f"{hl} ~ {pl}")
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=60, ha="right", rotation_mode="anchor", fontsize=5.4)
    ax.set_ylabel(f"centred cosine, layer {LM}"); ax.set_title("Angle", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True); letter(ax, "a")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=m, ls="none", color=INK2, mfc=INK2 if f in "cd" else "white", label=n, ms=4)
                       for m, f, n in zip("^Dos", "abcd", ("recognize", "judge", "act", "self-describe"))] +
              [Line2D([], [], color=GRID, lw=6, label="within the shuffle floor")],
              loc="upper right", frameon=False, fontsize=5.6, handletextpad=0.3, labelspacing=0.3)
    ax = axs[1]
    src, tgt, layers = cm["sources"], cm["targets"], cm["layers"]
    Z = cm["kinds"]["dense+"]["Z"]; li = layers.index(LL)
    good = {u.key: u.good for _, u in units()}
    sg = lambda uid: -1.0 if good.get(uid.rsplit("|", 1)[0]) == "neg" else 1.0
    pairs = [("C15|refuse[harmless]|c", "C15|refuse[harmful]|c"), ("C15|refuse[harmless]|c", "C15|refuse[harmful]|d"),
             ("C19|agreeableness|c", "C15|refuse[harmful]|c"), ("C26|all|c", "C15|refuse[harmful]|c"),
             ("C6|all|c", "C15|refuse[harmful]|c"), ("C5|all|c", "C15|refuse[harmful]|c"), ("C5|all|c", "C15|refuse[harmful]|d"),
             ("C15|refuse[harmful]|c", "C26|all|c"), ("C15|refuse[harmful]|c", "C6|all|c"), ("C15|refuse[harmful]|c", "C25|all|c")]
    lab = {u.key: u.label for _, u in units()}
    rows = []
    for s, t in pairs:
        if s in src and t in tgt:
            z = Z[src.index(s)][li][tgt.index(t)]
            if z is not None:
                rows.append((f"{lab[s.rsplit('|', 1)[0]]} ({s[-1]}) → {lab[t.rsplit('|', 1)[0]]} ({t[-1]})", z * sg(s) * sg(t)))
    ys = np.arange(len(rows))
    ax.axvspan(-3, 3, color=GRID, alpha=0.7, lw=0, zorder=0); ax.axvline(0, color=AXIS, lw=0.5, zorder=1)
    ax.barh(ys, [r[1] for r in rows], color=["#3987e5" if r[1] > 0 else "#e34948" for r in rows], height=0.6, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=5.4); ax.invert_yaxis()
    ax.set_xlabel(f"steering z, toward the good side, readout layer {LL}"); ax.set_title("Steering", loc="left")
    letter(ax, "b")
    fig.tight_layout()
    save(fig, out, "hhh_tension")
    plt.close(fig)
    return {"steering_cells": rows}


def d1(plt, tests, out):
    fig, ax = plt.subplots(figsize=(7.3, 2.6))
    x, labs = 0, []
    res = {}
    for b in ORDER:
        for u in BLOCKS[b]:
            for k, f in enumerate("abcd"):
                v = tests["d1"][str(LM)][b]["units"].get(f"{u.key}|{f}")
                if v and v["cos"] is not None:
                    ax.plot(x + (k - 1.5) * 0.17, v["cos"], marker="^Dos"[k], ms=3.4, color=BCOL[b], ls="none",
                            mfc=BCOL[b] if f in "cd" else "white")
            labs.append(u.label); x += 1
        res[b] = tests["d1"][str(LM)][b]["mean_sq_raw"]
    fl = tests["floors"][str(LM)]
    ax.axhspan(-fl, fl, color=GRID, alpha=0.7, lw=0); ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=60, ha="right", rotation_mode="anchor", fontsize=5.4)
    for t in ax.get_xticklabels():
        t.set_color(BCOL[next(b for b in ORDER if any(u.label == t.get_text() for u in BLOCKS[b]))])
    ax.set_ylabel(f"centred cosine with D1, layer {LM}"); ax.grid(axis="y", color=GRID, lw=0.4); ax.set_axisbelow(True)
    save(fig, out, "hhh_d1")
    plt.close(fig)
    return {"mean_sq_raw_by_block": res}


def main():
    plt = setup()
    out = out_dir() / "figures" / "hhh"
    ov = json.loads((out_dir() / "s4" / "overlap_verdict.json").read_text())
    tests = json.loads((out_dir() / "tests.json").read_text())
    cm = json.loads((out_dir() / "s5" / "causal_matrix.json").read_text())
    data = {"cosine": cosine(plt, ov, out), "landscape": landscape(plt, out), "tension": tension(plt, tests, cm, out),
            "d1": d1(plt, tests, out)}
    (out / "hhh_data.json").write_text(json.dumps(data, indent=1))
    for f in "cd":
        c = data["cosine"][f]
        print(f"frame {f}: mean |cos| within blocks {c['mean_abs_within_block']:.3f}, between {c['mean_abs_between_blocks']:.3f}; "
              f"share above floor within {c['share_above_floor_within']:.2f}, between {c['share_above_floor_between']:.2f}")
    for L, v in data["landscape"].items():
        print(f"layer {L}: block separation by frame group " + json.dumps({g: {k: round(x, 2) for k, x in s.items()} for g, s in v["block_separation_by_frame_group"].items()}))
    print("wrote", out)


if __name__ == "__main__":
    main()
