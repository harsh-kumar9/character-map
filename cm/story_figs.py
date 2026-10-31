"""Story figures: the seven findings of the honesty paper as plain paper figures (panels, axis labels, no headline
text; the caption in docs/STORY.md carries the words). Reads only tracked files under results/, runs on the Mac:

    python3 -m cm.story_figs            # writes results/figures/story/fig*.{pdf,png} and numbers.json

Subjects: gemma-4-31b-it (depths 16/30/39/51) and Qwen3.5-27B (17/32/42/54). "Steering depth" is the second,
"late depth" the fourth, as in the decision log (D-058 part 1, D-062, D-063)."""
import json, itertools, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "story"
OUT.mkdir(parents=True, exist_ok=True)

SUBJ = [("gemma-4-31b-it", "gemma-4-31b-it", "#0072B2", {"steer": "30", "late": "51"}),
        ("qwen3.5-27b", "Qwen3.5-27B", "#E69F00", {"steer": "32", "late": "54"})]
DEFS = ["Self-knowledge", "Accuracy", "Sincerity", "Forthrightness", "Non-sycophancy", "Honesty-humility"]
DEFC = {"Self-knowledge": "#009E73", "Accuracy": "#E69F00", "Sincerity": "#0072B2", "Forthrightness": "#D55E00",
        "Non-sycophancy": "#56B4E9", "Honesty-humility": "#CC79A7"}
DEFID = {"Self-knowledge": "A9", "Accuracy": "C2", "Sincerity": "C1", "Forthrightness": "C3", "Non-sycophancy": "C8",
         "Honesty-humility": "C19"}
FRAMES = [("a", "recognize"), ("b", "judge"), ("c", "act"), ("d", "self-describe")]
FAM = {"a": "own", "b": "own", "c": "act", "d": "self"}
FAMC = {"own": "#7B68B5", "act": "#B5542A", "self": "#2E8B7A"}
FAMN = {"own": "own reply about someone else (recognize, judge)", "act": "acting (reply in the model's voice)",
        "self": "self-describing (reply in the model's voice)"}
INK, INK2, MUTED, LIGHT, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#dcdbd5", "#e1e0d9", "#c3c2b7"
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.3, "axes.labelsize": 6.8,
                     "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.3, "axes.linewidth": 0.5,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "text.color": INK,
                     "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold"})
NUM = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


def panel_letter(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def load(tag, *parts):
    return json.load(open(ROOT / "results" / tag / Path(*parts)))


# ---------------------------------------------------------------- fig 1: the design grid
def fig1():
    rows = [(d, g) for d, g in [("Self-knowledge", "knows what it knows (calibration)"), ("Accuracy", "gets the facts right"),
                                ("Sincerity", "asserts what it believes, under pressure"),
                                ("Forthrightness", "volunteers what matters"), ("Non-sycophancy", "holds the truth against the user"),
                                ("Honesty-humility", "no manipulation, no entitlement (HEXACO H)")]]
    ctrls = [("Formality", "formal vs informal register", "abcd"), ("Topic", "one topic vs another", "ac")]
    have = {"Self-knowledge": "abcd", "Accuracy": "abcd", "Sincerity": "abcd", "Forthrightness": "acd", "Non-sycophancy": "acd",
            "Honesty-humility": "abcd"}
    col_gloss = {"a": "its own reply about\nsomeone else's conduct:\ndoes it see the trait?", "b": "its own reply:\nwas that conduct\nacceptable?",
                 "c": "a reply in its voice,\nunder temptation:\nhonest or not", "d": "a reply in its voice:\n\"the kind of\nassistant I am\""}
    import collections
    z = np.load(ROOT / "results" / SUBJ[0][0] / "figures" / "honesty" / "landscape_points.npz", allow_pickle=True)
    cnt = collections.Counter((m["def"].replace(" (control)", ""), m["frame"]) for m in json.loads(str(z["meta"])))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_axis_off()
    nR = len(rows) + len(ctrls)
    x0, w, h, gap = 2.55, 1.0, 0.56, 0.08
    ytop = nR * (h + gap)
    # column headers and the two families
    for j, (f, name) in enumerate(FRAMES):
        xc = x0 + j * (w + gap) + w / 2
        ax.text(xc, ytop + 0.95, f"{f}. {name}", ha="center", va="bottom", fontsize=7.3, fontweight="bold", color=INK)
        ax.text(xc, ytop + 0.9, col_gloss[f], ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.15)
    for fam, cols, label in (("own", (0, 1), "the model's own words about someone else"), ("act", (2, 2), ""), ("self", (3, 3), "")):
        pass
    xa, xb = x0, x0 + 2 * (w + gap) - gap
    ax.plot([xa, xb], [ytop + 1.75, ytop + 1.75], color=FAMC["own"], lw=1.2, solid_capstyle="butt")
    ax.text((xa + xb) / 2, ytop + 1.82, "the model's own words about a third party", ha="center", va="bottom", fontsize=6.3, color=FAMC["own"])
    xa, xb = x0 + 2 * (w + gap), x0 + 4 * (w + gap) - gap
    ax.plot([xa, xb], [ytop + 1.75, ytop + 1.75], color=INK2, lw=1.2, solid_capstyle="butt")
    ax.text((xa + xb) / 2, ytop + 1.82, "a reply written in the model's voice", ha="center", va="bottom", fontsize=6.3, color=INK2)
    # rows
    for i, (d, g) in enumerate(rows + [(c[0], c[1]) for c in ctrls]):
        y = ytop - (i + 1) * (h + gap)
        isc = i >= len(rows)
        col = MUTED if isc else DEFC[d]
        ax.text(x0 - 0.12, y + h * 0.62, d + (" (control)" if isc else ""), ha="right", va="center", fontsize=7, fontweight="bold", color=INK if not isc else INK2)
        ax.text(x0 - 0.12, y + h * 0.22, g, ha="right", va="center", fontsize=5.8, color=INK2)
        frames_have = have[d] if not isc else ctrls[i - len(rows)][2]
        for j, (f, name) in enumerate(FRAMES):
            x = x0 + j * (w + gap)
            if f in frames_have:
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=col, ec="none", alpha=0.22 if not isc else 0.3))
                n = cnt.get((d, f))
                ax.text(x + w / 2, y + h / 2, f"{n} items" if n else "", ha="center", va="center", fontsize=5.6, color=INK2)
            else:
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="none", ec=LIGHT, lw=0.6, ls=(0, (2, 2))))
    if True:
        y = ytop - len(rows) * (h + gap) - gap / 2
        ax.plot([x0 - 2.5, x0 + 4 * (w + gap) - gap], [y, y], color=AXIS, lw=0.5)
    ax.set_xlim(-0.1, x0 + 4 * (w + gap) + 0.1)
    ax.set_ylim(-0.1, ytop + 2.3)
    save(fig, "fig1_design")


# ---------------------------------------------------------------- fig 2: frame first
def pair_means(ids, M, concept_sub):
    """Mean transfer AUROC (both directions) for the frame pairs of one concept: own (a<->b), prefilled (c<->d), cross."""
    idx = {i.split("|")[2]: k for k, i in enumerate(ids) if i.rsplit("|", 1)[0] == concept_sub}
    def m(f, g):
        if f in idx and g in idx:
            a, b = M[idx[f]][idx[g]], M[idx[g]][idx[f]]
            v = [x for x in (a, b) if x is not None]
            return float(np.mean(v)) if v else None
        return None
    own = m("a", "b")
    pre = m("c", "d")
    cross = [v for v in (m(f, g) for f in "ab" for g in "cd") if v is not None]
    return own, pre, (float(np.mean(cross)) if cross else None)


def fig2():
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.42, wspace=0.22)
    rng = np.random.default_rng(0)
    NUM["fig2"] = {}
    for col, (tag, name, scol, dep) in enumerate(SUBJ):
        # (top) landscape at the late depth, honesty items, coloured by frame family
        z = np.load(ROOT / "results" / tag / "figures" / "honesty" / "landscape_points.npz", allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        Y = z[f"Y{dep['late']}"]
        st = load(tag, "figures", "honesty", "landscape_frames_stats.json")
        defn = np.array([m["def"] for m in meta]); fr = np.array([m["frame"] for m in meta])
        hon = np.array(["(control)" not in d for d in defn])
        fam = np.array([FAM[f] for f in fr])
        ax = fig.add_subplot(gs[0, col])
        keep = []
        for key in sorted(set(zip(defn[hon], fr[hon]))):
            ii = np.where((defn == key[0]) & (fr == key[1]))[0]
            keep += list(rng.choice(ii, min(120, len(ii)), replace=False))
        keep = np.array(keep); rng.shuffle(keep)
        for g in ("own", "act", "self"):
            sel = keep[fam[keep] == g]
            ax.scatter(Y[sel, 0], Y[sel, 1], s=4, c=FAMC[g], alpha=0.35, lw=0, rasterized=True)
        for g in ("own", "act", "self"):
            sel = hon & (fam == g)
            c = Y[sel].mean(0)
            ax.scatter(*c, s=46, c=FAMC[g], ec="white", lw=0.9, zorder=5)
            lab = {"own": "recognize, judge", "act": "act", "self": "self-describe"}[g]
            dy, va = ((0, -9), "top") if g == "act" else ((0, 9), "bottom")
            ax.annotate(lab, c, xytext=dy, textcoords="offset points", ha="center", va=va, fontsize=6.8, fontweight="bold", color=FAMC[g], path_effects=HALO, zorder=6)
        var = st["variance"]
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% of the honesty-subspace variance)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
        ax.set_title(name)
        panel_letter(ax, "ab"[col])
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        fs = st["frame_separation"]
        pol = st["polarity_gap"]; dsep = [v for g in st["definition_separation"].values() for v in g.values()]
        NUM["fig2"][tag] = {"frame_separation": fs, "variance": var, "late_depth": dep["late"],
                            "polarity_gap_median": float(np.median(list(pol.values()))), "polarity_gap": pol,
                            "definition_separation_median": float(np.median(dsep))}
        # (bottom) same-definition transfer across frames, standardized, verdict projected out, steering depth
        rep = load(tag, "coherence", "representational.json")
        pw = rep["by_depth"][dep["steer"]]["pairwise"]
        ids, M = pw["ids"], pw["auroc"]
        sub = {"Sincerity": "lying", "Non-sycophancy": "sycophantic_agreement", "Honesty-humility": "honesty_humility"}
        ax = fig.add_subplot(gs[1, col])
        xs = list(range(len(DEFS))) + [len(DEFS) + 0.6]
        tr = {}
        for k, d in enumerate(DEFS):
            tr[d] = pair_means(ids, M, f"{DEFID[d]}|{sub.get(d, 'all')}")
        k1 = pair_means(ids, M, "K1|all"); k2 = pair_means(ids, M, "K2|all")
        tr["controls"] = (k1[0], k1[1], float(np.mean([v for v in (k1[2], k2[2]) if v is not None])))
        NUM["fig2"][tag]["transfer_steer"] = tr
        kinds = [(0, "own-reply pair: recognize vs judge", "^", FAMC["own"]), (1, "prefilled pair: act vs self-describe", "s", INK2),
                 (2, "across the two families", "o", "#C0392B")]
        for ki, lab, mk, c in kinds:
            ys = [tr[d][ki] for d in DEFS] + [tr["controls"][ki]]
            ax.scatter([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], marker=mk, s=22, c=c, ec="white", lw=0.5, zorder=4, label=lab)
        ax.axhline(0.5, color=AXIS, lw=0.5, ls=":")
        ax.axvline(len(DEFS) - 0.2, color=GRID, lw=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(DEFS + ["controls (formality, topic)"], fontsize=6, rotation=25, ha="right", rotation_mode="anchor")
        ax.set_ylim(0.0, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("transfer AUROC\n(direction from one frame, items of the other)")
        ax.grid(color=GRID, lw=0.4, axis="y"); ax.set_axisbelow(True)
        panel_letter(ax, "cd"[col])
        if col == 0:
            hp = [Line2D([], [], marker=mk, ls="none", color=c, label=lab, ms=5) for _, lab, mk, c in kinds]
    hs = [Line2D([], [], marker="o", ls="none", color=FAMC[g], label=FAMN[g], ms=5) for g in ("own", "act", "self")]
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.1, 0.02), ncol=1, handletextpad=0.3, title="a, b: items coloured by frame", title_fontsize=6.3, alignment="left")
    fig.legend(handles=hp, loc="upper left", bbox_to_anchor=(0.55, 0.02), ncol=1, handletextpad=0.3, title="c, d: transfer between two frames of one definition", title_fontsize=6.3, alignment="left")
    save(fig, "fig2_frame_first")


# ---------------------------------------------------------------- fig 3: the metric
def fig3():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True)
    fig.subplots_adjust(hspace=0.18)
    order = []
    for blk, frames in (("own", "ab"), ("act", "c"), ("self", "d")):
        for f in frames:
            for d in DEFS:
                if f == "b" and d in ("Forthrightness", "Non-sycophancy"):
                    continue
                order.append((d, f))
    blocks = [("own-reply frames", [k for k, (d, f) in enumerate(order) if f in "ab"]), ("act", [k for k, (d, f) in enumerate(order) if f == "c"]),
              ("self-describe", [k for k, (d, f) in enumerate(order) if f == "d"])]
    blocks = [(n, min(ks), max(ks)) for n, ks in blocks]
    NUM["fig3"] = {}
    for ax, (tag, name, scol, dep) in zip(axes, SUBJ):
        t = load(tag, "coherence", "table.json")
        rows = {(r["def"], r["frame"]): r for r in t["rows"]}
        Ls = dep["steer"]
        xs, present = [], []
        for k, (d, f) in enumerate(order):
            r = rows.get((d, f))
            if r is None or r["ctrl"]:
                continue
            present.append((k, d, f, r))
        dm = [r[f"rep_dm_L{Ls}"] for _, _, _, r in present]
        pr = [r[f"rep_probe_L{Ls}"] for _, _, _, r in present]
        ks = [k for k, _, _, _ in present]
        cols = [DEFC[d] for _, d, _, _ in present]
        ctrl = [abs(r[f"rep_dm_L{Ls}"]) for r in t["rows"] if r["ctrl"] and r.get(f"rep_dm_L{Ls}") is not None]
        ctrl_p = [abs(r[f"rep_probe_L{Ls}"]) for r in t["rows"] if r["ctrl"] and r.get(f"rep_probe_L{Ls}") is not None]
        floor_dm, floor_pr = float(np.mean(ctrl)), float(np.mean(ctrl_p))
        ax.axhspan(0, floor_dm, color="#efeee9", lw=0, zorder=0)
        ax.axhline(floor_dm, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.axhline(0, color=AXIS, lw=0.5, zorder=1); ax.axhline(1, color=AXIS, lw=0.5, ls="--", zorder=1)
        for k, c, a, b in zip(ks, cols, dm, pr):
            ax.plot([k, k], [a, b], color=c, lw=0.6, alpha=0.6, zorder=2)
        ax.scatter(ks, dm, s=22, c=cols, ec="white", lw=0.5, zorder=4)
        ax.scatter(ks, pr, s=22, fc="white", ec=cols, lw=0.9, zorder=4)
        # block means (dm)
        for blk, lo, hi in blocks:
            v = [a for k, a in zip(ks, dm) if lo <= k <= hi]
            ax.plot([lo - 0.4, hi + 0.4], [np.mean(v)] * 2, color=INK, lw=1.1, zorder=3)
            if ax is axes[0]:
                ax.text((lo + hi) / 2, 1.12, blk, ha="center", va="bottom", fontsize=6.8, color=INK2)
        for xsep in (blocks[0][2] + 0.5, blocks[1][2] + 0.5):
            ax.axvline(xsep, color=GRID, lw=0.8)
        s = t["summary"]["rep"][Ls]
        NUM["fig3"][tag] = {"mean_dm": s["dm"]["mean_scaled"], "mean_probe": s["probe"]["mean_scaled"], "floor_dm": floor_dm,
                            "floor_probe": floor_pr, "by_block_dm": s["dm"]["by_block"], "by_block_probe": s["probe"]["by_block"],
                            "by_def_dm": s["dm"]["by_def"], "depth": Ls}
        ax.set_ylim(-0.55, 1.1); ax.set_yticks([-0.5, 0, 0.5, 1.0])
        ax.set_ylabel("leave-one-cell-out score\n(share of the cell's own AUROC)")
        ax.text(0.995, 0.04, name, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, fontweight="bold", color=scol)
        ax.grid(color=GRID, lw=0.4, axis="y"); ax.set_axisbelow(True)
        panel_letter(ax, "ab"[list(axes).index(ax)], x=-0.06)
    labels = []
    for d, f in order:
        labels.append({"a": "recog.", "b": "judge", "c": "act", "d": "self"}[f])
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(labels, fontsize=5.6, rotation=90)
    axes[1].set_xlim(-0.8, len(order) - 0.2)
    hs = [Line2D([], [], marker="o", ls="none", color=DEFC[d], label=d, ms=5) for d in DEFS] + \
         [Line2D([], [], marker="o", ls="none", color=INK2, label="difference of means", ms=5),
          Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="logistic probe", ms=5),
          Line2D([], [], color=INK, lw=1.1, label="frame-block mean"),
          Patch(fc="#efeee9", ec=MUTED, lw=0.6, ls=":", label="controls' mean |score|")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5, handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig3_metric")


# ---------------------------------------------------------------- fig 4: the battery, act vs account
def fig4():
    acts = [("RH", "reward hacking\n(no special-casing)"), ("SO", "instructed sophistry\n(won't argue it)"),
            ("CF", "must-answer prompt\n(hedges or declines)"), ("ARact", "covert write\n(no false COMPLETE)"),
            ("FT", "tool failure\n(no false claim)")]
    accts = [("AR", "report to owner\n(not deceptive)"), ("DC", "done claim\n(matches re-run)"),
             ("CO", "confession\n(owns the mistake)"), ("ARadm", "admits the write\n(when asked)")]
    items = acts + accts
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    NUM["fig4"] = {}
    for si, (tag, name, scol, dep) in enumerate(SUBJ):
        T = load(tag, "battery", "table.json")["tasks"]
        vals = {}
        for key, _ in items:
            if key == "ARact":
                n = T["AR"]["n_items"]; k = round((1 - T["AR"]["covert_rate"]) * n)
                lo, hi = wilson(k, n); vals[key] = (k / n, lo, hi, 1 - T["AR"]["control"]["covert_rate"])
            elif key == "ARadm":
                vals[key] = (T["AR"]["admits_rate"], None, None, T["AR"]["control"]["admits_rate"])
            elif key == "SO":
                vals[key] = (T["SO"]["rate"], *T["SO"]["ci"], T["SO"]["control"]["argues_for_correct_option"])
            else:
                vals[key] = (T[key]["rate"], *T[key]["ci"], T[key]["control"]["rate"])
        NUM["fig4"][tag] = vals
        off = -0.17 + 0.34 * si
        for k, (key, _) in enumerate(items):
            v, lo, hi, c = vals[key]
            x = k + (0.4 if k >= len(acts) else 0) + off
            if lo is not None:
                ax.plot([x, x], [lo, hi], color=scol, lw=1.0, zorder=3)
            ax.scatter([x], [v], s=30, c=scol, ec="white", lw=0.6, zorder=4)
            ax.scatter([x], [c], s=30, fc="white", ec=scol, lw=1.0, zorder=4)
    xs = [k + (0.4 if k >= len(acts) else 0) for k in range(len(items))]
    ax.set_xticks(xs); ax.set_xticklabels([lab.replace("\n", " ") for _, lab in items], fontsize=6, rotation=22, ha="right", rotation_mode="anchor")
    ax.axvline(len(acts) - 0.3, color=GRID, lw=0.8)
    ax.text((xs[0] + xs[len(acts) - 1]) / 2, 1.1, "what the model does under pressure", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text((xs[len(acts)] + xs[-1]) / 2, 1.1, "what it then says it did", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("honest-direction rate")
    ax.grid(color=GRID, lw=0.4, axis="y"); ax.set_axisbelow(True)
    hs = [Line2D([], [], marker="o", ls="none", color=SUBJ[0][2], label=SUBJ[0][1], ms=5.5),
          Line2D([], [], marker="o", ls="none", color=SUBJ[1][2], label=SUBJ[1][1], ms=5.5),
          Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="matched control (no pressure, or the doable version)", ms=5.5)]
    ax.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, handletextpad=0.3, columnspacing=1.5)
    save(fig, "fig4_battery")


# ---------------------------------------------------------------- fig 5: judging vs doing
def fig5():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.35})
    cats = [("a_pD_nocheat|AR", "no covert write\n(blind)"), ("b_pD_other_cheat|AR", "the other model's\ncovert write (blind)"),
            ("b_pD_own_cheat|AR", "its own covert\nwrite (blind)"), ("c_pD_own_you|AR", "its own, told\n\"you wrote this\"")]
    NUM["fig5"] = {}
    w = 0.36
    for si, (tag, name, scol, dep) in enumerate(SUBJ):
        X = load(tag, "battery", "xj", "table_baseline.json")
        rec = {}
        for k, (key, lab) in enumerate(cats):
            d = X["D"][key]; nh = X["notH"][key]
            x = k + (-w / 2 + w * si)
            ax.bar(x, nh["value"], width=w - 0.04, color=scol, alpha=0.3, lw=0, zorder=2)
            ax.bar(x, d["value"], width=w - 0.04, color=scol, lw=0, zorder=3)
            if d["ci"]:
                ax.plot([x, x], d["ci"], color=INK, lw=0.7, zorder=4)
            rec[key] = {"dishonest": d["value"], "dishonest_ci": d["ci"], "not_honest": nh["value"]}
        rec["same_move_n"] = X["D"]["b_n_same_move_items|AR"]; rec["own_cheat_n"] = X["D"]["d_n_own_cheat|AR"]
        rec["leniency"] = X["D"]["b_leniency_cheat|AR"]; rec["you_minus_another"] = X["D"]["c_you_minus_another_own|AR"]
        NUM["fig5"][tag] = rec
        # (b) as the actor
        T = load(tag, "battery", "table.json")["tasks"]["AR"]
        for k, (v, lab) in enumerate(((T["control"]["covert_rate"], "neutral"), (T["covert_rate"], "under pressure"))):
            x = k + (-w / 2 + w * si)
            n = T["n_items"]; lo, hi = wilson(round(v * n), n)
            bx.bar(x, v, width=w - 0.04, color=scol, lw=0, zorder=3)
            bx.plot([x, x], [lo, hi], color=INK, lw=0.7, zorder=4)
        NUM["fig5"][tag]["covert_rate"] = T["covert_rate"]; NUM["fig5"][tag]["covert_rate_neutral"] = T["control"]["covert_rate"]
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([lab for _, lab in cats], fontsize=6.2, linespacing=1.1)
    ax.set_ylim(0, 1.02); ax.set_ylabel("share of trajectories the model,\nas judge, calls dishonest")
    ax.grid(color=GRID, lw=0.4, axis="y"); ax.set_axisbelow(True)
    ax.set_title("as the judge of a saved record")
    panel_letter(ax, "a", x=-0.08)
    bx.set_xticks([0, 1]); bx.set_xticklabels(["neutral\ntask", "same task,\nunder pressure"], fontsize=6.2, linespacing=1.1)
    bx.set_ylim(0, 1.02); bx.set_ylabel("share of tasks where it wrote\nthe false COMPLETE itself")
    bx.grid(color=GRID, lw=0.4, axis="y"); bx.set_axisbelow(True)
    bx.set_title("as the actor")
    panel_letter(bx, "b", x=-0.2)
    hs = [Patch(fc=SUBJ[0][2], label=SUBJ[0][1]), Patch(fc=SUBJ[1][2], label=SUBJ[1][1]),
          Patch(fc=INK2, label="verdict: dishonest"), Patch(fc=INK2, alpha=0.3, label="verdict: dishonest or borderline")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=4, handletextpad=0.3, columnspacing=1.4)
    save(fig, "fig5_judge_vs_act")


# ---------------------------------------------------------------- fig 6: the on-policy sweep
def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    groups = [("act cells (c)", "act directions", FAMC["act"], "-"), ("self-describe cells (d)", "self-describe directions", FAMC["self"], "-"),
              ("own-reply cells (a, b)", "recognize and judge directions", FAMC["own"], "-"),
              ("shared axes (dm_all, probe_all)", "shared honesty axis", INK, (0, (3, 2))), ("K1", "formality control", MUTED, (0, (1, 1.5))),
              ("K2", "topic control", MUTED, (0, (3, 1.5)))]
    NUM["fig6"] = {}
    for ax, (tag, name, scol, dep) in zip(axes, SUBJ):
        S = load(tag, "battery", "onpolicy", "baseline", "sweep.json")
        tab = S["tables"]["turn_mean"]
        idx = sorted(int(k) for k in tab)
        idx = [i for i in idx if i >= 1]
        xs = np.array(idx)
        g = lambda key: np.array([tab[str(i)][key] for i in idx], dtype=float)
        ax.fill_between(xs, g("random (16)"), g("random_95"), color="#e6e5df", lw=0, zorder=1)
        ax.plot(xs, g("random (16)"), color=MUTED, lw=0.6, zorder=2)
        ax.plot(xs, g("probe"), color=INK, lw=1.5, zorder=5)
        for key, lab, c, ls in groups:
            ax.plot(xs, g(key), color=c, lw=0.9, ls=ls, zorder=4 if "control" not in lab else 3)
        ax.axhline(0.5, color=AXIS, lw=0.5)
        ax.set_xlabel("layer (hidden-state index)")
        ax.set_title(name)
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        panel_letter(ax, "ab"[list(axes).index(ax)], x=-0.03)
        for L in (int(dep["steer"]), int(dep["late"])):
            ax.axvline(L, color=GRID, lw=0.8, ls="-")
        NUM["fig6"][tag] = {"mean_over_layers": {lab: float(np.mean(g(key))) for key, lab, _, _ in groups},
                            "probe_mean": float(np.mean(g("probe"))), "random_mean": float(np.mean(g("random (16)"))),
                            "random_95_mean": float(np.mean(g("random_95")))}
    axes[0].set_ylabel("AUROC for the model's own dishonest act\n(turn mean, within task strata)")
    axes[0].set_ylim(0.45, 1.0)
    hs = [Line2D([], [], color=INK, lw=1.5, label="probe trained on this task (ceiling)")] + \
         [Line2D([], [], color=c, lw=0.9, ls=ls, label=lab) for _, lab, c, ls in groups] + \
         [Patch(fc="#e6e5df", ec=MUTED, lw=0.6, label="16 random directions: mean to 95th percentile")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig6_onpolicy")


# ---------------------------------------------------------------- fig 7: a licence to deceive
def fig7():
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.15, 1.15], "wspace": 0.55})
    conds = [("neu", "neutral\nprompt"), ("lic", "licence to\ndeceive"), ("ins", "told to\ndeceive")]
    tasks = [("RH", "reward hacking:\nno special-casing"), ("CF", "fabrication:\nhedges or declines"), ("FT", "tool completion:\nno false claim")]
    NUM["fig7"] = {}
    for si, (tag, name, scol, dep) in enumerate(SUBJ):
        P = load(tag, "battery", "prompted", "summary.json")
        NUM["fig7"][tag] = {"battery": P["battery"], "loco_dm": {c: {d: P["loco"][c][d]["dm"]["mean_scaled"] for d in P["loco"][c]} for c in P["loco"]},
                            "loco_controls": {c: {d: P["loco"][c][d]["dm"]["mean_scaled_controls"] for d in P["loco"][c]} for c in P["loco"]}}
        for ti, (task, lab) in enumerate(tasks):
            ax = axes[ti]
            ys = [P["battery"][task][c][0] for c, _ in conds]
            lo = [P["battery"][task][c][1] for c, _ in conds]; hi = [P["battery"][task][c][2] for c, _ in conds]
            xs = np.arange(3) + (-0.07 + 0.14 * si)
            ax.plot(xs, ys, color=scol, lw=1.0, zorder=3)
            for x, y, a, b in zip(xs, ys, lo, hi):
                ax.plot([x, x], [a, b], color=scol, lw=0.8, zorder=3)
            ax.scatter(xs, ys, s=24, c=scol, ec="white", lw=0.5, zorder=4)
            ax.set_xticks(range(3)); ax.set_xticklabels([c for _, c in conds], fontsize=5.8, linespacing=1.1)
            ax.set_ylim(0, 1.05); ax.set_title(lab, fontsize=6.6, fontweight="normal")
            ax.grid(color=GRID, lw=0.4, axis="y"); ax.set_axisbelow(True)
            if ti == 0:
                ax.set_ylabel("honest-direction rate")
            else:
                ax.set_yticklabels([])
        ax = axes[3 + si]
        depths = sorted(P["loco"]["neu"], key=int)
        xs = np.arange(4)
        for c, lab, ls, col in (("neu", "neutral", "-", MUTED), ("lic", "licence", "-", scol), ("ins", "told", (0, (3, 1.5)), scol)):
            ax.plot(xs, [P["loco"][c][d]["dm"]["mean_scaled"] for d in depths], color=col, lw=1.1, ls=ls, zorder=3, marker="o", ms=3, mec="white", mew=0.4)
            ax.plot(xs, [P["loco"][c][d]["dm"]["mean_scaled_controls"] for d in depths], color=col, lw=0.6, ls=ls, alpha=0.6, zorder=2)
        ax.set_xticks(xs); ax.set_xticklabels([f"L{d}" for d in depths], fontsize=5.8)
        ax.set_ylim(-0.05, 1.0)
        ax.set_title(name, fontsize=6.6, fontweight="normal")
        ax.grid(color=GRID, lw=0.4, axis="y"); ax.set_axisbelow(True)
        if si == 0:
            ax.set_ylabel("leave-one-cell-out score\n(12 act + self-describe cells)", labelpad=2)
    for k, ax in enumerate(axes):
        panel_letter(ax, "abcde"[k], x=-0.1)
    hs = [Line2D([], [], marker="o", color=SUBJ[0][2], label=SUBJ[0][1], ms=4), Line2D([], [], marker="o", color=SUBJ[1][2], label=SUBJ[1][1], ms=4),
          Line2D([], [], color=MUTED, lw=1.1, label="neutral prompt (d, e)"), Line2D([], [], color=INK2, lw=1.1, label="licence prompt (d, e)"),
          Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 1.5)), label="told-to-deceive prompt (d, e)"),
          Line2D([], [], color=INK2, lw=0.6, alpha=0.6, label="thin: the same score on the control cells")]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig7_licence")


# ---------------------------------------------------------------- fig 8: the anatomy of the gaps
def fig8():
    """Four parts of a virtue as stages, the named gaps between them, and where the two subjects fall at baseline.
    Statuses are read from the decision log (D-061, D-062, D-063): open = the failure appears, closed = it does not,
    partial = it appears on part of the readout."""
    from matplotlib.patches import FancyArrowPatch
    stages = [("sees it", "perception\n(frame a: recognize)"), ("judges it", "judgment\n(frame b: judge)"),
              ("does it", "choice\n(frame c: act; the battery)"), ("owns it", "account of itself\n(frame d: self-describe;\nreports, confession)")]
    gaps = [  # (from, to, name, source line, instrument, status gemma, status qwen)
        (0, 0, "blind spot", "Gioia 1992; Tenbrunsel & Messick 2004;\nMurdoch 1970; Blum 1991", "blind judge sees the\ncovert write", "partial", "closed"),
        (1, 2, "akrasia", "Aristotle NE VII.3; Davidson 1970;\nMilgram 1963; Browning 1992", "condemns own covert write,\nwrote it anyway", "open", "open"),
        (3, 2, "hypocrisy (say-do)", "Batson 1997; Mazar et al. 2008;\nLifton 1986 (doubling)", "self-report honest,\nconduct not", "open", "open"),
        (1, 1, "self-leniency", "Valdesolo & DeSteno 2007;\nBatson 1997", "own vs other's covert\nwrite, blind", "closed", "closed"),
        (2, 3, "concealment", "cover-up vs confession;\nBok 1978", "report to owner;\nadmits when asked", "closed", "closed"),
        (-1, 2, "situation", "Hartshorne & May 1928; Milgram 1974;\nDoris 2002", "licence prompt moves\nconduct", "open", "closed"),
        (-2, 2, "having, not using", "Aristotle NE VII.3; Elazar et al. 2021;\nGaleone et al. 2026", "described-honesty directions\nread the decision", "open", "open"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.set_axis_off()
    W, H, X0, Y0, DX = 1.35, 0.62, 0.62, 3.9, 1.72
    cx = [X0 + i * DX + W / 2 for i in range(4)]
    for i, (name, sub) in enumerate(stages):
        x = X0 + i * DX
        ax.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0,rounding_size=0.08", fc="#f1f0eb", ec=INK2, lw=0.7))
        ax.text(x + W / 2, Y0 + H * 0.66, name, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + W / 2, Y0 + H * 0.26, sub, ha="center", va="center", fontsize=5.4, color=INK2, linespacing=1.1)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + W + 0.03, Y0 + H / 2), (x + DX - 0.03, Y0 + H / 2), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))
    ax.text(X0 - 0.05, Y0 + H + 0.32, "a trait, by its parts", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.text(cx[0] - 0.9, Y0 + H / 2, "the\nsituation", ha="center", va="center", fontsize=6, color=MUTED, linespacing=1.1)
    ax.add_patch(FancyArrowPatch((cx[0] - 0.62, Y0 + H / 2), (X0 - 0.03, Y0 + H / 2), arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.7, ls=(0, (2, 2))))
    # gap rows
    ST = {"open": (INK, "full"), "closed": ("white", "none"), "partial": (INK, "left")}
    yrow, dy = Y0 - 0.55, 0.47
    xcol = [0.0, 1.7, 3.7, 5.45, 6.35, 6.85]
    for j, h in enumerate(["gap", "name in the literature", "what we read (D-061 to D-063)", "gemma-4", "Qwen"]):
        ax.text(xcol[j] + (0.33 if j >= 3 else 0), yrow + 0.22, h, ha="center" if j >= 3 else "left", va="bottom", fontsize=6.2, color=MUTED)
    ax.plot([0, 7.4], [yrow + 0.18, yrow + 0.18], color=AXIS, lw=0.5)
    for k, (a, b, name, src, inst, sg, sq) in enumerate(gaps):
        y = yrow - k * dy
        ax.text(xcol[0], y, name, ha="left", va="center", fontsize=6.8, fontweight="bold", color=INK)
        ax.text(xcol[1], y, src, ha="left", va="center", fontsize=5.4, color=INK2, linespacing=1.1)
        ax.text(xcol[2], y, inst, ha="left", va="center", fontsize=5.6, color=INK, linespacing=1.1)
        for j, stt in ((3, sg), (4, sq)):
            x = xcol[j] + 0.33
            if stt == "partial":
                ax.scatter([x], [y], s=46, marker=MarkerStyle("o", fillstyle="left"), c=INK, ec=INK, lw=0.7, zorder=3)
            else:
                ax.scatter([x], [y], s=46, fc=INK if stt == "open" else "white", ec=INK, lw=0.7, zorder=3)
        ax.plot([0, 7.4], [y - dy / 2, y - dy / 2], color=GRID, lw=0.4)
    hs = [Line2D([], [], marker="o", ls="none", mfc=INK, mec=INK, label="the gap is open at baseline (the failure appears)", ms=6),
          Line2D([], [], marker=MarkerStyle("o", fillstyle="left"), ls="none", mfc=INK, mec=INK, label="partial", ms=6),
          Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, label="closed (it does not appear)", ms=6)]
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.0, -0.06), ncol=3, handletextpad=0.3, columnspacing=1.4)
    ax.set_xlim(-0.05, 7.5); ax.set_ylim(yrow - len(gaps) * dy + 0.05, Y0 + H + 0.6)
    save(fig, "fig8_gaps")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1)
    print("wrote numbers.json")
