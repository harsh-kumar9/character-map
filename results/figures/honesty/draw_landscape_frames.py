"""The landscape figure for one line: inside Gemma, where an honesty item sits is set first by its frame, then by which
honesty it is, and last by whether it is the honest or the dishonest side. One layer, three panels by frame group on
shared PC1-PC2 axes, no control. Also writes landscape_frames_stats.json and the page's landscape_text.html.
Run on the Mac:  python3 results/figures/honesty/draw_landscape_frames.py [layer=29]"""
import json, sys, itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

import os
HERE = Path(os.environ["CM_HONESTY_DIR"]) if os.environ.get("CM_HONESTY_DIR") else Path(__file__).resolve().parent   # per-model: results/<tag>/figures/honesty
L = sys.argv[1] if len(sys.argv) > 1 else "29"
D = json.load(open(HERE / "landscape_3d.json")); meta = D["meta"]
V = D["views"][f"as-is, layer {L}"]; X = np.array(V["xyz"]); var = V["var"]
DEFS = {"Self-knowledge": "#009E73", "Accuracy": "#E69F00", "Sincerity": "#0072B2", "Forthrightness": "#D55E00", "Non-sycophancy": "#56B4E9", "Honesty-humility": "#CC79A7"}
CID = {"A9": "Self-knowledge", "C2": "Accuracy", "C1": "Sincerity", "C3": "Forthrightness", "C8": "Non-sycophancy", "C19": "Honesty-humility"}
FRG = {"a": "own", "b": "own", "c": "act", "d": "self"}
PANEL = [("own", "Gemma's own reply about someone else\n(recognize and judge)"), ("act", "Acting\n(reply written in Gemma's voice)"), ("self", "Self-describing\n(reply written in Gemma's voice)")]
INK, INK2, MUTED, LIGHT, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#dcdbd5", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.3, "axes.labelsize": 6.8, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6.3, "axes.linewidth": 0.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "text.color": INK})
defn = np.array([m["def"] for m in meta]); fr = np.array([m["frame"] for m in meta]); pol = np.array([m["pol"] for m in meta])
grp = np.array([FRG[f] for f in fr]); hon = defn != "Formality (control)"
rng = np.random.default_rng(0)

# ---- statistics behind the caption (3 PCs)
units = sorted({(d, f) for d, f in zip(defn[hon], fr[hon])})
wsd = float(np.sqrt(np.mean([X[(defn == d) & (fr == f)].var(0).sum() for d, f in units])))
gm = {g: X[(grp == g) & hon].mean(0) for g in ("own", "act", "self")}
frame_sep = {f"{a}-{b}": float(np.linalg.norm(gm[a] - gm[b]) / wsd) for a, b in itertools.combinations(gm, 2)}
def_sep = {}
for g in ("act", "self"):
    cs = {d: X[(grp == g) & (defn == d)].mean(0) for d in DEFS}
    def_sep[g] = {f"{a}|{b}": float(np.linalg.norm(cs[a] - cs[b]) / wsd) for a, b in itertools.combinations(DEFS, 2)}
pol_gap = {f"{d}|{f}": float(np.linalg.norm(X[(defn == d) & (fr == f) & (pol == "pos")].mean(0) - X[(defn == d) & (fr == f) & (pol == "neg")].mean(0)) / wsd) for d, f in units}
stats = {"layer": L, "within_unit_spread": wsd, "frame_separation": frame_sep, "definition_separation": def_sep, "polarity_gap": pol_gap, "variance": var}
json.dump(stats, open(HERE / "landscape_frames_stats.json", "w"), indent=1)

# ---- figure: three panels, shared axes, all items as a grey backdrop, this frame's items in color
keep = []
for key in sorted(set(zip(defn, fr, pol))):
    idx = np.where((defn == key[0]) & (fr == key[1]) & (pol == key[2]))[0]
    keep += list(rng.choice(idx, min(150, len(idx)), replace=False))
keep = np.array(sorted(keep)); keep_h = keep[hon[keep]]
fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.9), sharex=True, sharey=True)
for ax, (g, title) in zip(axes, PANEL):
    ax.scatter(X[keep_h, 0], X[keep_h, 1], s=2.5, color=LIGHT, linewidths=0, zorder=1, rasterized=True)
    sel = keep_h[grp[keep_h] == g]
    for i in sel:
        c = DEFS[defn[i]]
        ax.scatter(X[i, 0], X[i, 1], s=7, facecolor=c if pol[i] == "pos" else "none", edgecolor=c, linewidth=0.5, alpha=0.5, zorder=2, rasterized=True)
    for d, f in units:
        if FRG[f] != g: continue
        c = DEFS[d]; cp = X[(defn == d) & (fr == f) & (pol == "pos")].mean(0); cn = X[(defn == d) & (fr == f) & (pol == "neg")].mean(0)
        ax.annotate("", xy=cp[:2], xytext=cn[:2], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=2), zorder=5)
        ax.scatter(*cn[:2], s=30, facecolor="white", edgecolor=c, linewidth=1.1, zorder=6)
        ax.scatter(*cp[:2], s=30, facecolor=c, edgecolor="white", linewidth=0.7, zorder=6)
        if g == "own":
            ax.text(cp[0] + 1.5, cp[1] + (4.5 if f == "a" else -4.5), {"a": "recognize", "b": "judge"}[f], fontsize=6.2, color=c, ha="left", va="center", zorder=7, path_effects=HALO)
        if g == "act" and d == "Self-knowledge":
            ax.text(cn[0] - 1.0, cn[1] - 9.5, "\u201cI don't know\u201d\nreplies", fontsize=6.2, color=c, ha="right", va="top", zorder=7, linespacing=1.05, path_effects=HALO)
    ax.set_title(title, loc="left", linespacing=1.05)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% of the subspace variance)"); ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.text(-0.02, 1.08, "abc"[PANEL.index((g, title))], transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
axes[0].set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
for ax in axes: ax.set_aspect("equal", adjustable="box")
h = [Line2D([], [], marker="o", ls="none", color=c, label=n, ms=5) for n, c in DEFS.items()] + \
    [Line2D([], [], marker="o", ls="none", color=INK2, label="honest side of the contrast (e.g. sincere, knows)", ms=4),
     Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="other side (deceptive, does not know)", ms=4),
     Line2D([], [], marker="o", ls="none", color=LIGHT, label="items of the other frames", ms=4)]
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
fig.savefig(HERE / "landscape_frames.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(HERE / "landscape_frames.pdf", bbox_inches="tight", pad_inches=0.03)

# ---- the page's reading, from the same numbers
fs = frame_sep; ds = def_sep; pg = pol_gap
FAM = {"Self-knowledge": "epistemic", "Accuracy": "epistemic", "Sincerity": "assertion", "Forthrightness": "assertion", "Non-sycophancy": "social", "Honesty-humility": "trait"}   # D-049
def fam_sep(g):
    w = [v for k, v in ds[g].items() if FAM[k.split("|")[0]] == FAM[k.split("|")[1]]]
    b = [v for k, v in ds[g].items() if FAM[k.split("|")[0]] != FAM[k.split("|")[1]]]
    return float(np.mean(w)), float(np.mean(b))
w_self, b_self = fam_sep("self"); w_act, b_act = fam_sep("act")
near_act = min(ds["act"].items(), key=lambda kv: kv[1]); far_act = max(ds["act"].items(), key=lambda kv: kv[1])
near_self = min(ds["self"].items(), key=lambda kv: kv[1]); far_self = max(ds["self"].items(), key=lambda kv: kv[1])
stats["family_separation"] = {"self": {"within": w_self, "between": b_self}, "act": {"within": w_act, "between": b_act}}
json.dump(stats, open(HERE / "landscape_frames_stats.json", "w"), indent=1)
def pr(kv): return kv[0].replace("|", " vs ").lower() + f" ({kv[1]:.1f})"
txt = f"""<p><b>The line.</b> Inside Gemma, where an honesty item sits is set first by its frame, then by which honesty it is, and last by whether it is the honest or the dishonest side.</p>
<ul>
<li><b>Frame first.</b> At layer {L} the items of the three frame groups sit {fs['own-act']:.1f}, {fs['own-self']:.1f} and {fs['act-self']:.1f} within-unit spreads apart (own reply vs acting, own reply vs self-describing, acting vs self-describing). The first component of the ten-direction subspace is that split; it carries {var[0]*100:.0f}% of the subspace variance. The same clusters hold the formality control's items, so this is the kind of utterance, not honesty.</li>
<li><b>Definition second, and in families.</b> The six definitions fall into four families (epistemic: self-knowledge, accuracy; assertion: sincerity, forthrightness; social: non-sycophancy; trait: honesty-humility). Inside the self-describing frame two definitions of the same family sit {w_self:.1f} spreads apart on average and two of different families {b_self:.1f}; inside the acting frame {w_act:.1f} and {b_act:.1f}. The closest pair in acting is {pr(near_act)} and the farthest {pr(far_act)}; in self-description {pr(near_self)} and {pr(far_self)}. So "honesty" splits into families before it splits into six.</li>
<li><b>Honest vs dishonest last, except for self-knowledge.</b> The + to − gap of a unit is {min(v for k, v in pg.items() if not k.startswith('Self-knowledge')):.1f} to {max(v for k, v in pg.items() if not k.startswith('Self-knowledge')):.1f} spreads for every contrast but self-knowledge's. Self-knowledge's gap is {pg['Self-knowledge|c']:.1f} in acting and {pg['Self-knowledge|d']:.1f} in self-description, and its "I don't know" replies land among the self-descriptions: an abstention is, in form, a statement about oneself. That is why this definition's act and self-describe directions converge with depth and why it entangles with self-deprecation.</li>
</ul>
<p>Two consequences for the design. A probe or steering vector for "honesty" built from own-reply judgments mostly encodes the own-reply format unless that axis is projected out, which is what the verdict control does. And an evaluation that reads self-descriptions is reading the character family's cluster; what separates "I am sincere" from "I am formal" inside it is a small offset.</p>"""
open(HERE / "landscape_text.html", "w").write(txt)
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items() if k in ("within_unit_spread", "frame_separation")}))
print("def sep self", {k: round(v, 2) for k, v in ds["self"].items()}); print("def sep act", {k: round(v, 2) for k, v in ds["act"].items()})
print("pol gaps", {k: round(v, 2) for k, v in pg.items()}); print("wrote landscape_frames.png")
