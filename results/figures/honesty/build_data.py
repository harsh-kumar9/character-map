"""Collect the honesty block from character-map result files and draw the figure.
Run: python3 build_data.py <repo> <outdir>
Writes honesty_data.json and honesty_fig.png (+pdf) to outdir."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
R = REPO / "results"

# ---- the honesty block: (concept, sub) -> plain name; polarity = positive pole
UNITS = [
    ("C1", "all", "Sincere vs deceptive"),
    ("C2", "all", "Accurate vs fabricating"),
    ("A9", "all", "Knows vs does not know"),
    ("C3", "all", "Forthright vs withholding"),
    ("C4", "boastful", "Truthful vs boastful"),
    ("C4", "self-deprecating", "Truthful vs self-deprecating"),
    ("C19", "honesty_humility", "Honesty-humility high vs low"),
    ("C12", "integrity", "Integrity vs hypocrisy"),
    ("C11", "all", "Holds position vs capitulates"),
    ("E1", "all", "Observed vs unobserved"),
    ("E5", "all", "Pushback vs none"),
    ("F1", "all", "Honest option vs kind option"),
    ("K1", "all", "Formal vs casual (control)"),
]
NAME = {(c, s): n for c, s, n in UNITS}
KEYS = [(c, s) for c, s, _ in UNITS]
FRAME = {"a": "recognize", "b": "judge", "c": "act", "d": "self-describe"}

def uid(c, s, f): return f"{c}|{s}|{f}"

# ---- S3 fidelity and transfer
frames = json.load(open(R / "s3/frames_table.json"))
fid = {}
for r in frames:
    k = (r["concept"], r["sub"])
    if k in NAME and r["layer"] in (17, 29):
        fid.setdefault(k, {})[r["layer"]] = {
            "frames": r["frames"], "within": r["within"], "heldout": r["heldout"], "heldout_far": r["heldout_far"],
            "transfer": r["transfer"]}
claims = json.load(open(R / "s3/claims.json"))
tr = []
for r in claims:
    k = (r["concept"], r["sub"])
    if k in NAME and r["layer"] in (17, 29):
        tr.append({"concept": r["concept"], "sub": r["sub"], "name": NAME[k], "pair": r["pair"], "layer": r["layer"],
                   "reply32": r["reply32"], "preverdict": r["preverdict"], "verdict": r["verdict"],
                   "base_reply32": r["base_reply32"], "base_verdict": r["base_verdict"], "class": r["class"]})

# ---- S4 cosines
ov = json.load(open(R / "s4/overlap_verdict.json"))
units = ov["units"]; idx = {u: i for i, u in enumerate(units)}
def submat(layer, frame_set):
    ids = [uid(c, s, f) for c, s in KEYS for f in frame_set if uid(c, s, f) in idx]
    M = ov["layers"][str(layer)]["cos"]
    return ids, [[M[idx[a]][idx[b]] for b in ids] for a in ids]
cos = {}
for L in (17, 29):
    for fs, tag in (("c", "act"), ("d", "self"), ("ab", "knowing")):
        ids, M = submat(L, fs)
        cos[f"{tag}_L{L}"] = {"units": ids, "cos": M, "floor": ov["layers"][str(L)]["floor_cos"]}

# ---- S5a causal
cm = json.load(open(R / "s5/causal_matrix.json"))
src, tgt, layers = cm["sources"], cm["targets"], cm["layers"]
Z = cm["kinds"]["dense+"]["Z"]; E = cm["kinds"]["dense+"]["E"]
hs = [s for s in src if (s.split("|")[0], s.split("|")[1]) in NAME and not s.startswith("K1")]
ht = [t for t in tgt if (t.split("|")[0], t.split("|")[1]) in NAME and not t.startswith("K1")]
causal = {}
for li, L in enumerate(layers):
    causal[f"L{L}"] = {"sources": hs, "targets": ht,
                       "z": [[(None if Z[src.index(s)][li][tgt.index(t)] is None or math.isnan(Z[src.index(s)][li][tgt.index(t)]) else round(Z[src.index(s)][li][tgt.index(t)], 2)) for t in ht] for s in hs],
                       "E": [[(None if E[src.index(s)][li][tgt.index(t)] is None or math.isnan(E[src.index(s)][li][tgt.index(t)]) else round(E[src.index(s)][li][tgt.index(t)], 3)) for t in ht] for s in hs]}
causal["dose_c"] = cm["dose_c"]; causal["dose_norm"] = cm["dose_norm"]

# ---- S5b behavior
bh = json.load(open(R / "s5b/behavior.json"))
BCOND = {"C1": ["doing+", "doing-", "knowing+", "knowing-"], "C2": ["doing+", "doing-"], "A9": ["doing+", "doing-"],
         "C3": ["doing+", "doing-", "knowing+", "knowing-"], "C4": ["doing+", "doing+:self-deprecating", "doing-", "knowing+", "knowing-"],
         "C19": ["doing+:honesty_humility"], "C12": ["doing+", "doing-", "knowing+", "knowing-"], "C11": ["doing+", "doing-"],
         "E1": ["doing+", "doing-", "knowing+", "knowing-"], "E5": ["doing+", "doing-"],
         "F1": ["doing+", "doing-", "knowing+", "knowing-", "duty+:C1|all|c", "duty+:C5|all|c"]}
INSTR = {"C1": "MASK known_facts: share asserting the truth under pressure (judge)",
         "C2": "SelfAware: abstain on unanswerable, gold answer on answerable (exact)",
         "A9": "SelfAware: abstention on unanswerable vs answerable (judge)",
         "C3": "forthrightness judge, 0-100 scaled to 0-1",
         "C4": "capability self-report, calibrated self-presentation judge",
         "C19": "TRAIT two-option choice: rate of the high honesty-humility reply",
         "C12": "Batson-style coin-flip task, two-option choice",
         "C11": "are-you-sure pushback: keeps the correct answer (exact)",
         "E1": "observed vs unobserved, exact-scored",
         "E5": "SYCON/are-you-sure two-option choice",
         "F1": "AIRiskDilemmas: share of choices for the honest option"}
beh = []
for c in BCOND:
    for judge in ("gemma", "qwen"):
        cc = bh[c]["conds"].get(judge)
        if not cc: continue
        base = cc["baseline"]["mean"]
        for cond in BCOND[c]:
            v = cc.get(cond)
            if not v: continue
            rnd = v.get("vs_random", {}).get("max_abs_random")
            eff, se = v["effect"], v["se"]
            passes = (rnd is not None and abs(eff) >= 2 * rnd and se > 0 and abs(eff) >= 3 * se)
            beh.append({"concept": c, "name": NAME[[k for k in KEYS if k[0] == c][0]], "instrument": INSTR[c], "type": bh[c]["type"],
                        "judge": judge, "cond": cond, "baseline": base, "effect": eff, "se": se, "max_random": rnd,
                        "unparsed": v.get("unparsed"), "passes": passes})

data = {"units": [{"concept": c, "sub": s, "name": n} for c, s, n in UNITS], "fidelity": {f"{c}|{s}": v for (c, s), v in fid.items()},
        "transfer": tr, "cos": cos, "causal": causal, "behavior": beh}
json.dump(data, open(OUT / "honesty_data.json", "w"), indent=1)

# ---- figure (paper style, as cm/paper_figs.py)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIV = LinearSegmentedColormap.from_list("div", ["#8f2424", "#c23b3b", "#e34948", "#ef8f8e", "#f7d2d1", "#f0efec",
                                                "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.2,
                     "ytick.labelsize": 6.2, "legend.fontsize": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "text.color": INK})
def short(u):
    c, s, f = u.split("|")
    n = NAME[(c, s)]
    return {"Sincere vs deceptive": "Sincere", "Accurate vs fabricating": "Accurate", "Knows vs does not know": "Knows answer",
            "Forthright vs withholding": "Forthright", "Truthful vs boastful": "Not boastful", "Truthful vs self-deprecating": "Not self-deprecating",
            "Honesty-humility high vs low": "Honesty-humility", "Integrity vs hypocrisy": "Integrity", "Holds position vs capitulates": "Holds position",
            "Observed vs unobserved": "Observed", "Pushback vs none": "Under pushback", "Honest option vs kind option": "Honest over kind",
            "Formal vs casual (control)": "Formal (control)"}[n] + (f" ({FRAME[f][0]})" if f in "ab" else "")
def letter(ax, s): ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
def heat(ax, ids, M, vmin, vmax, fmt, title):
    A = np.array([[np.nan if v is None else v for v in row] for row in M], float)
    ax.imshow(A, cmap=DIV, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
    ax.set_xticklabels([short(u) for u in ids], rotation=60, ha="right", rotation_mode="anchor"); ax.set_yticklabels([short(u) for u in ids])
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = A[i, j]
            if i == j or np.isnan(v): continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=4.6, color="white" if abs(v) > 0.55 * vmax else INK)
    ax.set_title(title, loc="left"); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

fig = plt.figure(figsize=(7.3, 7.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.05], hspace=0.62, wspace=0.55)
axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])
ids, M = cos["act_L17"]["units"], cos["act_L17"]["cos"]
heat(axa, ids, M, -0.8, 0.8, lambda v: f"{v:.2f}".replace("0.", ".").replace("-.", "-."), "Acting, layer 17 (cosine)"); letter(axa, "a")
ids, M = cos["self_L17"]["units"], cos["self_L17"]["cos"]
heat(axb, ids, M, -0.8, 0.8, lambda v: f"{v:.2f}".replace("0.", "."), "Self-describing, layer 17 (cosine)"); letter(axb, "b")

# (c) within-concept transfer, verdict-controlled, L17 filled and L29 open
PT = {"a-b": ("know", BLUE), "c-d": ("do", ORANGE)}
rows = sorted({(t["concept"], t["sub"], t["pair"]) for t in tr}, key=lambda x: (KEYS.index((x[0], x[1])), x[2]))
rows = [r for r in rows if r[0] != "K1"]
ypos = {r: i for i, r in enumerate(rows)}
base17 = next(t["base_verdict"] for t in tr if t["layer"] == 17); base29 = next(t["base_verdict"] for t in tr if t["layer"] == 29)
axc.axvspan(min(base17, base29), max(base17, base29), color=GRID, alpha=0.7, lw=0); axc.axvline(0.5, color=AXIS, lw=0.5, ls=":")
axc.axvspan(0.0, 0.40, color="#f7d2d1", alpha=0.35, lw=0)
for t in tr:
    key = (t["concept"], t["sub"], t["pair"])
    if key not in ypos: continue
    col = PT.get(t["pair"], ("kd", AQUA))[1]
    axc.plot(t["verdict"], ypos[key], marker="o", ms=4, mfc=col if t["layer"] == 17 else "white", mec=col, mew=0.8, ls="none")
axc.set_yticks(range(len(rows))); axc.set_yticklabels([f"{short(uid(c, s, 'c')).replace(' (c)', '')} · {p.replace('a-b', 'recognize–judge').replace('c-d', 'act–self').replace('a-c', 'recognize–act').replace('a-d', 'recognize–self').replace('b-c', 'judge–act').replace('b-d', 'judge–self')}" for c, s, p in rows])
axc.invert_yaxis(); axc.set_xlim(0.1, 1.0); axc.set_xlabel("Transfer AUROC, Gemma's verdict removed"); axc.tick_params(axis="y", length=0)
axc.grid(axis="x", color=GRID, lw=0.4); axc.set_axisbelow(True); axc.set_title("Within a definition, across frames", loc="left"); letter(axc, "c")
from matplotlib.lines import Line2D
axc.legend(handles=[Line2D([], [], marker="o", ms=4, color=BLUE, ls="none", label="recognize–judge"), Line2D([], [], marker="o", ms=4, color=ORANGE, ls="none", label="act–self-describe"),
                    Line2D([], [], marker="o", ms=4, color=AQUA, ls="none", label="knowing–doing"), Line2D([], [], marker="o", ms=4, mfc="white", mec=INK2, ls="none", label="open = layer 29")],
           loc="lower right", frameon=False, handletextpad=0.3, borderpad=0.2, labelspacing=0.25)

# (d) causal z at L29, honesty sources x honesty targets
cz = causal["L29"]; S, T = cz["sources"], cz["targets"]
order_t = [uid(c, s, f) for c, s in KEYS for f in "abcd" if uid(c, s, f) in T]
order_s = [u for u in [uid(c, s, f) for c, s in KEYS for f in "cdba"] if u in S]
A = np.array([[np.nan if cz["z"][S.index(s)][T.index(t)] is None else cz["z"][S.index(s)][T.index(t)] for t in order_t] for s in order_s], float)
im = axd.imshow(np.clip(A, -6, 6), cmap=DIV, vmin=-6, vmax=6, aspect="auto")
axd.set_xticks(range(len(order_t))); axd.set_xticklabels([short(t) + ("" if t.split("|")[2] in "ab" else f" ({t.split('|')[2]})") for t in order_t], rotation=60, ha="right", rotation_mode="anchor")
axd.set_yticks(range(len(order_s))); axd.set_yticklabels([short(s) + ("" if s.split("|")[2] in "ab" else f" ({s.split('|')[2]})") for s in order_s])
for i in range(len(order_s)):
    for j in range(len(order_t)):
        v = A[i, j]
        if np.isnan(v) or abs(v) < 3: continue
        axd.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.6, color="white" if abs(v) > 4 else INK)
axd.set_title("Steering at layer 17, readout at layer 29 (z vs random directions)", loc="left"); axd.tick_params(length=0)
for s in axd.spines.values(): s.set_visible(False)
axd.set_ylabel("steered direction"); axd.set_xlabel("readout direction"); letter(axd, "d")
cb = fig.colorbar(im, ax=axd, fraction=0.03, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6); cb.set_label("z (clipped at ±6)")
fig.savefig(OUT / "honesty_fig.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
fig.savefig(OUT / "honesty_fig.pdf", bbox_inches="tight", pad_inches=0.03)
print("ok", OUT)
