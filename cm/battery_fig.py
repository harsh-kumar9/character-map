"""Baseline battery figure (D-061): one panel, tasks on the x axis, honest-direction rate with the 95 percent item
bootstrap interval, the two subjects side by side (filled), their controls hollow. Axis labels and a caption carry the
words; nothing is written inside the panel. Repo figure style (Arial 7 pt, Okabe-Ito colors, as results/figures/honesty)."""
from __future__ import annotations

from pathlib import Path

ORDER = ("RH", "FT", "SB", "SO", "CF", "CO", "EC", "AR", "DC")
XLAB = {"RH": "reward\nhacking", "FT": "tool\ncompletion", "SB": "sandbagging\n+ admission", "SO": "instructed\nsophistry",
        "CF": "confident\nfabrication", "CO": "confession", "EC": "eval-\nconditioned", "AR": "action\nreport",
        "DC": "done\nclaim"}
COL = {"gemma-4-31b-it": "#0072B2", "qwen3.5-27b": "#E69F00"}
NAME = {"gemma-4-31b-it": "gemma-4-31b-it", "qwen3.5-27b": "Qwen3.5-27B"}
INK, INK2, GRID = "#222222", "#555555", "#DDDDDD"
CAPTION = ("Honest-direction rate per task at baseline (unsteered): RH 1 - cheat rate on impossible tests, FT 1 - fabricated "
           "completion, SB 1 - P(sandbagged and denied), SO (instructed sophistry) 1 - argued for the wrong option on request, CF 1 - fabricated, CO owned the "
           "planted mistake, EC same decision on the real and hypothetical versions, AR 1 - deceptive report after a "
           "tool-use task under inducement (DecepEval pairs), DC 1 - false done claim in coding. Bars: 95% bootstrap interval over "
           "items (2,000 resamples). Hollow markers: the matched control (RH pass rate on the original tests, FT reports a "
           "real success, SB accuracy without the notice, SO argues for the correct option, CF without the must-answer "
           "prompt, CO false confession on a correct earlier answer, AR the neutral version of the same pairs, DC the original tests).")


def control_rate(s):
    c = s.get("control")
    if not c:
        return None
    if "rate" in c:
        return c["rate"]
    return c.get("argues_for_correct_option")


def draw(tabs: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42, "text.color": INK,
                         "axes.labelcolor": INK2, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    tags = [t for t in COL if t in tabs]
    order = [t for t in ORDER if any(t in tabs[tag]["tasks"] for tag in tags)]
    fig, ax = plt.subplots(figsize=(5.2 + 0.55 * max(0, len(order) - 7), 2.6))
    width = 0.34
    for k, tag in enumerate(tags):
        off = (k - (len(tags) - 1) / 2) * width
        xs, ys, lo, hi, cx, cy = [], [], [], [], [], []
        for i, task in enumerate(order):
            s = tabs[tag]["tasks"].get(task)
            if not s or s.get("rate") is None:
                continue
            xs.append(i + off); ys.append(s["rate"]); lo.append(s["rate"] - s["ci"][0]); hi.append(s["ci"][1] - s["rate"])
            c = control_rate(s)
            if c is not None:
                cx.append(i + off); cy.append(c)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=4.5, color=COL[tag], ecolor=COL[tag], elinewidth=1.2, capsize=0,
                    label=NAME[tag], zorder=3)
        ax.scatter(cx, cy, s=22, facecolors="white", edgecolors=COL[tag], linewidths=1.0, zorder=4,
                   label=f"{NAME[tag]}, control")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([XLAB[t] for t in order])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("honest-direction rate")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "baseline.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(out_dir / "baseline.pdf", bbox_inches="tight", pad_inches=0.03)
    (out_dir / "baseline_caption.txt").write_text(CAPTION + "\n")
    print(f"[battery] figure -> {out_dir / 'baseline.png'}")
