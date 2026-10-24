"""The block page for helpfulness or harmlessness (D-055), the honesty page's layout (results/figures/honesty/
build_page.py) for an HHH block. Numbers come from the block's result files; the reading is in READING below and is
written after the results are in.

Reads  results/hhh/figures/<block>/ (figures, <block>_data.json, landscape_3d.json), results/hhh/s3/frames_table.json,
       results/hhh/tests.json, results/hhh/s5/causal_matrix.json, results/hhh/s5b/behavior.json (when present)
Writes results/hhh/figures/<block>/<block>.html

    python3 -m cm.hhh.page helpfulness|harmlessness
"""
from __future__ import annotations

import base64
import html
import json
import math
import sys

import numpy as np

from .. import cfg
from .blocks import BLOCKS
from .figs import FRAMES

R = cfg.results_dir() / "hhh"
LM, LL = cfg.L_STEER, cfg.L_OUT[1]

CSS = """
:root{--bg:#f6f7f5;--surface:#ffffff;--ink:#17201d;--ink2:#4d5752;--muted:#8a938e;--rule:#dfe3df;--accent:#0b6e8f;--accent-soft:#e3f0f4;--warn:#9a3b2f;--warn-soft:#f7e6e2;--ok-soft:#e4f1ea;
 --display:"Source Serif 4",Georgia,"Times New Roman",serif;--body:"Source Sans 3","Helvetica Neue",Arial,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15191a;--surface:#1d2224;--ink:#e8ebe8;--ink2:#b5bcb7;--muted:#7f8783;--rule:#303836;--accent:#6cc3e0;--accent-soft:#1c333b;--warn:#e38b7c;--warn-soft:#3a2420;--ok-soft:#1f3328;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15191a;--surface:#1d2224;--ink:#e8ebe8;--ink2:#b5bcb7;--muted:#7f8783;--rule:#303836;--accent:#6cc3e0;--accent-soft:#1c333b;--warn:#e38b7c;--warn-soft:#3a2420;--ok-soft:#1f3328;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:17px;line-height:1.5;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:32px 64px}
article{max-width:46rem;margin:0 auto}
.eyebrow{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font-family:var(--display);font-weight:600;font-size:2.1rem;line-height:1.15;margin:.3rem 0 .6rem;text-wrap:balance}
h2{font-family:var(--display);font-weight:600;font-size:1.35rem;margin:2.4rem 0 .6rem;text-wrap:balance}
h3{font-size:1rem;margin:1.6rem 0 .4rem;font-weight:600}
p{margin:.6rem 0}.stand{font-size:1.12rem;color:var(--ink2)}
.meta{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;font-size:.9rem;color:var(--ink2);border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);padding:.6rem 0;margin:1rem 0}
.meta b{color:var(--ink);font-weight:600}
.box{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.9rem 1.1rem;margin:1rem 0}
.box.accent{border-left:4px solid var(--accent)}.box.warn{border-left:4px solid var(--warn)}
.box h3{margin:.1rem 0 .4rem}
ul,ol{padding-left:1.3rem}li{margin:.25rem 0}
.tbl{overflow-x:auto;margin:.8rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:.38rem .5rem;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:.8rem}
td.hi{background:var(--ok-soft)}td.low{background:var(--warn-soft)}
.id{font-family:var(--mono);font-size:.74rem;color:var(--muted);margin-left:.2rem}
figure{margin:1.4rem 0}figure img{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--rule);border-radius:4px}
figcaption{font-size:.88rem;color:var(--ink2);margin-top:.5rem}
.note{font-size:.9rem;color:var(--ink2)}
.viewbar{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.6rem}
.viewbtn{font:inherit;font-size:.85rem;padding:.3rem .7rem;border:1px solid var(--rule);border-radius:4px;background:var(--surface);color:var(--ink);cursor:pointer}
.viewbtn.on{border-color:var(--accent);background:var(--accent-soft)}
.viewbtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#plot3d{width:100%;height:520px;border:1px solid var(--rule);border-radius:4px;background:var(--surface)}
@media (max-width:640px){#plot3d{height:380px}}
a{color:var(--accent)}
code{font-family:var(--mono);font-size:.84em;background:var(--accent-soft);padding:0 .25em;border-radius:3px}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
"""

# what each definition is, where it comes from, its sets (D-053, D-054, D-056, D-057)
DEFS = {
    "helpfulness": [
        ("C24|too_literal", "answers the evident intent vs the literal words only", "constitution, immediate desires (\"neither too literally nor too liberally\"); Model Spec, letter and spirit of instructions; IFEval", "IFEval prompts, a reply that passes vs fails the checker", "IFEval strict pass rate; an overstepping judge on edit tasks (CoEdIT)"),
        ("C24|overstepping", "does what was asked vs changes what was not asked", "Model Spec, avoid overstepping; constitution, background desiderata", "as above", "as above"),
        ("C25|all", "specific, usable help vs vague, caveat-heavy help at the same length", "constitution, \"brilliant friend\" and the over-caution list; Model Spec, thorough and direct, the BAD examples; HelpSteer2", "HelpSteer2 reply pairs, high vs low helpfulness", "benign questions in risky-sounding domains (FalseReject, Health-ORSC, generated): rate judged fully helpful"),
        ("C5|all", "notices how the person is and responds vs apathetic", "care ethics (Noddings 1984; Tronto 1993); Batson 1981; Model Spec, be warm", "EPITOME", "EmpatheticDialogues: apathy judge (lower is better)"),
        ("C6|all", "respects the person's own decision vs paternalistic", "constitution, autonomy; Model Spec, not condescending; Mill 1859; Dworkin 1972", "DailyDilemmas", "DailyDilemmas: paternalism judge (lower is better)"),
        ("C26|all", "supports the person's own capacities and ties vs makes itself the main support", "constitution, wellbeing and endorsed reliance; Model Spec, respect real-world ties; Kirk et al. 2025; INTIMA", "INTIMA labelled replies", "INTIMA prompts: rate judged companionship-reinforcing (lower is better)"),
        ("C19|agreeableness", "high vs low agreeableness", "Costa and McCrae 1992; Ashton and Lee 2007", "HEXACO items", "TRAIT two-option choice, agreeableness items"),
    ],
    "harmlessness": [
        ("A1|all", "reads a dual-use request as having a legitimate purpose vs harmful intent", "constitution, intentions and context; Model Spec, intent changes the answer", "OpenSafeIntent", "OpenSafeIntent: intent judge, 100 = treats the purpose as legitimate"),
        ("B1|all", "sees an act as harmless vs harmful (the harmless side is the good side)", "MFT care/harm; Feinberg 1984; Schein and Gray 2018", "MFV", "ETHICS commonsense: accuracy"),
        ("C15|refuse[harmful]", "refuses a harmful request vs complies", "Askell et al. 2021; Model Spec, information hazards; HarmBench; Arditi et al. 2024", "XSTest (C15, all requests)", "OR-Bench hard and toxic: comply or refuse as expected"),
        ("C15|refuse[harmless]", "helps with a harmless request vs refuses it", "Model Spec, assume best intentions; constitution, over-caution list", "as above", "as above"),
        ("A7|all", "reads a borderline request charitably vs with suspicion", "Model Spec, assume best intentions; XSTest; OR-Bench", "XSTest", "OR-Bench-Hard-1K: direct-answer rate"),
        ("C27|side_effect", "acts without unsanctioned side effects vs causes them", "constitution, safe behaviors; Model Spec, control side effects; ToolEmu", "R-Judge, injection cases", "ManagerBench: the expected choice"),
        ("C27|instructed", "declines a harmful tool call vs makes it", "Mind the GAP (Cartagena and Teixeira 2026); AgentHarm; Agent-SafetyBench", "GAP far set; R-Judge", "as above"),
        ("C20|psychopathy", "low vs high psychopathy (the low side is the good side)", "Paulhus and Williams 2002; Moshagen et al. 2018", "dark-triad items", "TRAIT dark-triad choice (lower is better)"),
    ],
}
REVIEW = {"C24": "https://claude.ai/artifact/TebePUZG3YrZBjKGtD4udQ", "C25": "https://claude.ai/artifact/8pHeXsvD7ppXnWMrRG6R7b",
          "C26": "https://claude.ai/artifact/8VM3sxhdniSUbj75p4ZPbW", "C19": "https://claude.ai/artifact/DtMkF2eEENHgZBqn74DRRA",
          "C27": "https://claude.ai/artifact/J57PqFuQQgwGpMiV8cJzNx", "B1": "https://claude.ai/artifact/AhBgN1324CXPKamzdWKw4Z",
          "C20": "https://claude.ai/artifact/K9YRV4qoYeUaZavqHWtz2C"}
# behavior: does a higher instrument score mean the good side?
GOOD_HIGH = {"A1": True, "A7": True, "B1": True, "C15": True, "C19": True, "C20": False, "C24": True, "C25": True,
             "C26": False, "C27": True, "C5": False, "C6": False}
# the behavior group a concept's own score is read on, where the instrument mixes several (TRAIT)
BEH_GROUP = {"C19": "agreeableness", "C20": "psychopathy"}

READING = {}          # filled per block once S5b is in: {"tldr": [...], "r1": ..., ...}


def esc(s):
    return html.escape(str(s))


def f2(v, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def img(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def load(block):
    out = R / "figures" / block
    D = json.loads((out / f"{block}_data.json").read_text())
    ft = json.loads((R / "s3" / "frames_table.json").read_text())
    tests = json.loads((R / "tests.json").read_text())
    beh_p = R / "s5b" / "behavior.json"
    beh = json.loads(beh_p.read_text()) if beh_p.exists() else {}
    return out, D, ft, tests, beh


def heldout_rows(block, ft):
    """Within-frame and held-out AUROC per unit and frame at the two depths, from the HHH frames table (reply32,
    raw diff of means; standardized beside it)."""
    rows = []
    for key, *_ in DEFS[block]:
        cid, sub = key.split("|")
        sub = sub.split("[")[0]
        for L in (LM, LL):
            r = next((x for x in ft if x["concept"] == cid and x["sub"] == sub and x["layer"] == L), None)
            if r:
                rows.append((key, L, r))
    return rows


def defs_table(block):
    us = {u.key: u for u in BLOCKS[block]}
    rows = []
    for key, pole, src, held, beh in DEFS[block]:
        u = us[key]
        cid = key.split("|")[0]
        link = f' <a href="{REVIEW[cid]}">review</a>' if cid in REVIEW else ""
        rows.append(f"<tr><td>{esc(u.name)}</td><td>{esc(pole)}</td><td>{esc(src)}</td><td>{esc(key)}{link}</td>"
                    f"<td>{esc(held)}</td><td>{esc(beh)}</td></tr>")
    return ("<table><thead><tr><th>Definition</th><th>Good side vs other side</th><th>Where it comes from</th>"
            "<th>Unit</th><th>Held-out set</th><th>Behavior instrument</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def fid_table(block, ft):
    rows = []
    for key, L, r in heldout_rows(block, ft):
        if L != LM:
            continue
        r2 = next((x for x in ft if x["concept"] == r["concept"] and x["sub"] == r["sub"] and x["layer"] == LL), r)
        cells = []
        for f in "abcd":
            w1, w2 = r["within"].get(f), r2["within"].get(f)
            cells.append(f"<td>{f2(w1)} / {f2(w2)}</td>" if w1 is not None else "<td>–</td>")
        held = {f: v for f, v in r["heldout"].items() if v is not None}
        held2 = {f: v for f, v in r2["heldout"].items() if v is not None}
        hs = ", ".join(f"{FRAMES[f]} {f2(v)} / {f2(held2.get(f))}" for f, v in held.items()) or "–"
        cls = ' class="low"' if held and max(held.values()) < 0.6 else ""
        rows.append(f"<tr><td>{esc(key)}</td>{''.join(cells)}<td{cls}>{hs}</td></tr>")
    return ("<table><thead><tr><th>Unit (S3 sub-contrast)</th>" + "".join(f"<th>{FRAMES[f]}</th>" for f in "abcd") +
            f"<th>Held-out</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            f"<p class=\"note\">Within-frame and held-out AUROC at layers {LM} / {LL}, reply32, S3's raw diff-of-means "
            "direction. C15's rows are the whole concept (S3 has no request-type split); the request-type units are in "
            "the transfer figure.</p>")


def beh_rows(block, beh):
    """Per concept: baseline and the oriented effect of the good-ward doing and knowing conditions, against the
    largest random-direction effect; passes = at least twice that and at least 3 SE (the S5b criterion)."""
    us = BLOCKS[block]
    out = []
    for cid in dict.fromkeys(u.cid for u in us):
        if cid not in beh:
            continue
        u0 = next(u for u in us if u.cid == cid)
        res = beh[cid]
        for judge in ("gemma", "qwen"):
            cc = res["conds"].get(judge)
            if not cc:
                continue
            grp = BEH_GROUP.get(cid)
            def val(c):
                v = cc.get(c)
                if not v:
                    return None, None
                if grp:
                    b = cc["baseline"]["by_group"].get(grp); m = v["by_group"].get(grp)
                    return (None if b is None or m is None else m - b), None
                return v["effect"], v["se"]
            sg = 1.0 if GOOD_HIGH[cid] else -1.0
            toward = "doing+" if u0.good == "pos" else "doing-"
            ktoward = "knowing+" if u0.good == "pos" else "knowing-"
            rnd = [abs(cc[c]["effect"]) for c in cc if c.startswith("random") and cc[c]["effect"] == cc[c]["effect"]]
            rmax = max(rnd) if rnd else None
            base = cc["baseline"]["by_group"].get(grp) if grp else cc["baseline"]["mean"]
            row = {"concept": cid, "judge": judge, "baseline": base, "max_random": rmax}
            for name, c in (("doing", toward), ("knowing", ktoward)):
                e, se = val(c)
                row[name] = None if e is None else sg * e
                row[name + "_se"] = se
                row[name + "_pass"] = (e is not None and rmax is not None and abs(e) >= 2 * rmax and (se is None or abs(e) >= 3 * se))
            out.append(row)
    return out


def beh_table(rows):
    if not rows:
        return "<p class=\"note\">Behavior under steering is pending.</p>"
    def cell(v, se, ok):
        if v is None:
            return "<td>–</td>"
        s = f"{v:+.3f}" + (f" ± {se:.3f}" if se else "")
        return f'<td class="hi">{s}</td>' if ok and v > 0 else f'<td class="low">{s}</td>' if ok else f"<td>{s}</td>"
    body = "".join(f"<tr><td>{r['concept']}</td><td>{r['judge']}</td><td>{f2(r['baseline'])}</td>"
                   f"{cell(r['doing'], r['doing_se'], r['doing_pass'])}{cell(r['knowing'], r['knowing_se'], r['knowing_pass'])}"
                   f"<td>{f2(r['max_random'])}</td></tr>" for r in rows)
    return ("<table><thead><tr><th>Concept</th><th>Judge</th><th>Unsteered</th><th>Doing direction, toward the good side</th>"
            "<th>Knowing direction, toward the good side</th><th>Largest random effect</th></tr></thead><tbody>" + body +
            "</tbody></table>")


def plot3d(out):
    D3 = (out / "landscape_3d.json").read_text()
    return f"""<div class="box"><h3>Rotate it</h3>
<p class="note">The block's items on the first three principal components. Drag to rotate, scroll to zoom, hover for an item's definition, frame and side. Filled marks are the good side. The buttons switch the coloring, the layer, and between the space as is and each unit centred on its own mean.</p>
<div class="viewbar" role="group" aria-label="Color">
<button class="viewbtn on" id="c-frame" type="button">Color by frame group</button><button class="viewbtn" id="c-def" type="button">Color by definition group</button></div>
<div class="viewbar" role="group" aria-label="View">
<button class="viewbtn" id="v-17" type="button">Layer {LM}</button><button class="viewbtn on" id="v-29" type="button">Layer {LL}</button>
<button class="viewbtn" id="v-c" type="button">Each unit centred</button></div>
<div id="plot3d"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.1/plotly.min.js"></script>
<script>
(function(){{
  var D = {D3};
  var GRPC = {{"own":"#2a78d6","act":"#eb6834","self":"#1baf7a"}}, FRG = {{"a":"own","b":"own","c":"act","d":"self"}};
  var FRN = {{"a":"recognize","b":"judge","c":"act","d":"self-describe"}};
  var groups = {json.dumps({u.key: u.group for us in BLOCKS.values() for u in us})};
  var gl = Array.from(new Set(Object.values(groups)));
  var colorBy = "frame", layer = "{LL}", centred = false;
  function css(v){{ return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888"; }}
  function draw(){{
    var V = D.views[(centred ? "each unit centred, layer " : "as-is, layer ") + layer];
    var byKey = {{}};
    D.meta.forEach(function(m, i){{
      if (m.block === "control") return;
      var g = colorBy === "frame" ? FRG[m.frame] : groups[m.unit] || "other";
      var k = g + (m.good ? "|good" : "|other");
      (byKey[k] = byKey[k] || {{x:[],y:[],z:[],t:[],g:g,good:m.good}});
      byKey[k].x.push(V.xyz[i][0]); byKey[k].y.push(V.xyz[i][1]); byKey[k].z.push(V.xyz[i][2]);
      byKey[k].t.push(m.name + ", " + FRN[m.frame] + ", " + (m.good ? "good side" : "other side"));
    }});
    var palette = colorBy === "frame" ? GRPC : {{}};
    if (colorBy !== "frame") gl.forEach(function(g, i){{ palette[g] = ["#2a78d6","#eb6834","#1baf7a"][i % 3]; }});
    var traces = Object.keys(byKey).sort().map(function(k){{ var b = byKey[k]; var c = palette[b.g] || "#888";
      return {{type:"scatter3d", mode:"markers", x:b.x, y:b.y, z:b.z, text:b.t, hovertemplate:"%{{text}}<extra></extra>",
        name: b.g + (b.good ? ", good side" : ", other side"),
        marker:{{size:2.4, color: b.good ? c : "rgba(0,0,0,0)", line:{{color:c, width:1}}, opacity:0.75}}}}; }});
    var ink = css("--ink2"), grid = css("--rule");
    var ax = function(n){{ return {{title:{{text:"PC" + n + " (" + (V["var"][n-1]*100).toFixed(0) + "%)"}}, color:ink, gridcolor:grid, zerolinecolor:grid}}; }};
    Plotly.react("plot3d", traces, {{paper_bgcolor:"rgba(0,0,0,0)", margin:{{l:0,r:0,t:0,b:0}}, font:{{color:ink, size:11}},
      legend:{{orientation:"h", y:-0.02}}, scene:{{xaxis:ax(1), yaxis:ax(2), zaxis:ax(3), bgcolor:"rgba(0,0,0,0)"}}}},
      {{displaylogo:false, responsive:true}});
  }}
  function on(id, grp){{ document.querySelectorAll(grp).forEach(function(b){{ b.classList.remove("on"); }}); document.getElementById(id).classList.add("on"); }}
  document.getElementById("c-frame").onclick = function(){{ colorBy = "frame"; on("c-frame", "#c-frame,#c-def"); draw(); }};
  document.getElementById("c-def").onclick = function(){{ colorBy = "def"; on("c-def", "#c-frame,#c-def"); draw(); }};
  document.getElementById("v-17").onclick = function(){{ layer = "{LM}"; on("v-17", "#v-17,#v-29"); draw(); }};
  document.getElementById("v-29").onclick = function(){{ layer = "{LL}"; on("v-29", "#v-17,#v-29"); draw(); }};
  document.getElementById("v-c").onclick = function(){{ centred = !centred; this.classList.toggle("on", centred); draw(); }};
  if (window.Plotly) draw();
}})();
</script>"""


def build(block):
    out, D, ft, tests, beh = load(block)
    rd = READING.get(block)
    if rd is None:
        sys.exit(f"no reading written for {block} yet; the page is built only once the reading is in READING")
    lay = D["landscape"]
    fl = D["lattice"][str(LM)]["floor"]
    brows = beh_rows(block, beh)
    name = {"helpfulness": "helpfulness", "harmlessness": "harmlessness"}[block]
    fig = lambda f: img(out / f)
    page = f"""<title>What is {name}, to Gemma?</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>{CSS}</style>
<article>
<div class="eyebrow">character-map · HHH line · exploratory · {rd['date']}</div>
<h1>What is {name}, to Gemma?</h1>
<p class="stand">{rd['stand']}</p>
<div class="meta"><span><b>Model</b> Gemma-3-4B-IT</span><span><b>Method</b> diff-of-means directions in four frames, cross-frame transfer, standardized cosine, steering, behavior under steering</span><span><b>Code and results</b> <a href="https://github.com/harsh-kumar9/character-map">github.com/harsh-kumar9/character-map</a></span></div>
<p class="note"><b>Epistemic status.</b> {rd['status']}</p>
<div class="box accent"><h3>TL;DR</h3><ul>{''.join(f'<li>{t}</li>' for t in rd['tldr'])}</ul></div>

<h2>The question</h2>
{rd['question']}

<h2>The definitions</h2>
<p>{rd['defs_intro']}</p>
<div class="tbl">{defs_table(block)}</div>

<h2>Setup, briefly</h2>
<p>Each scenario appears in four frames with a matched good and other version. In <i>recognize</i> and <i>judge</i> the user describes someone else's exchange and we read Gemma's own reply; in <i>act</i> and <i>self-describe</i> the reply is written in Gemma's voice and we read it. A direction is the mean hidden state of good-side items minus other-side items on training scenarios, over the first 32 reply tokens, at decoder layers {LM} and {LL}. Gemma's verdict direction is projected out of recognize and judge. Cosines are taken after standardizing every hidden dimension and centring on the mean direction of all 265 units in the map; the shuffle floor is {fl:.2f}. Steering adds a direction at layer {cfg.L_STEER} at the dose calibrated for the honesty run (1/128 of the residual norm) and reads every direction at layers 22 and 29, as a z-score against random directions. Every unit is oriented to its good side, so a positive cosine or z means the two good sides go together.</p>

<h2>Result 1: frame first, then definition</h2>
<figure><img src="{fig(f'landscape_units_L{LL}.png')}" alt="Items of each definition in the block subspace, one panel per definition">
<figcaption><b>Figure 1. Where each definition's items sit, layer {LL}.</b> Every item of the block projected into the subspace spanned by its (definition, frame) directions, first two principal components; one panel per definition, its items colored by frame group over the block's other items in grey. Filled marks are the good side; arrows run from each frame's other-side centroid to its good-side centroid.</figcaption></figure>
<figure><img src="{fig(f'landscape_frames_L{LL}.png')}" alt="The same items in three panels by frame group">
<figcaption><b>Figure 2. The same space by frame group.</b> Color is the definition's group; labels mark each unit's good-side centroid in the two frames in Gemma's voice.</figcaption></figure>
{rd['r1']}
{plot3d(out)}

<h2>Result 2: within a definition, across frames</h2>
<figure><img src="{fig('transfer.png')}" alt="Transfer AUROC across frames for each definition">
<figcaption><b>Figure 3. Does a definition's direction in one frame read another frame?</b> Transfer AUROC, mean of both directions, Gemma's verdict removed; filled at layer {LM}, open at layer {LL}. Dashed: the formality control's transfer.</figcaption></figure>
{rd['r2']}
<div class="tbl">{fid_table(block, ft)}</div>

<h2>Result 3: across definitions</h2>
<figure><img src="{fig('lattice.png')}" alt="Lattice of the block's definitions by frames">
<figcaption><b>Figure 4. The block as a lattice.</b> Columns are definitions, rows are frames, the dotted line is the format boundary. The number in a node is its within-frame AUROC (a, b) or its effect on its own readout when steered (c). Edges join directions in the same column or row whose centred cosine is at least 0.30 (every pair above the floor of {fl:.2f} is in the data file); red dashed is negative. (c) Arrows are steering effects inside a definition with |z| at least 3.</figcaption></figure>
<figure><img src="{fig('cosine.png')}" alt="Cosine grids for the act and self-describe frames and for recognize and judge">
<figcaption><b>Figure 5. The cosines behind the lattice, layer {LM}.</b> Centred cosine, every unit oriented to its good side.</figcaption></figure>
{rd['r3']}

<h2>Result 4: steering</h2>
<figure><img src="{fig('steering.png')}" alt="Steering effects within the block and on the other block">
<figcaption><b>Figure 6. What steering one direction does to the others.</b> Rows: the direction added at layer {cfg.L_STEER}; columns: the readout at layer {LL}; z against random directions, oriented to the good side, numbers where |z| is at least 3. (a) Inside the block, every frame. (b) The other HHH block's act and self-describe readouts.</figcaption></figure>
{rd['r4']}

<h2>Result 5: behavior under steering</h2>
{rd['r5']}
<div class="tbl">{beh_table(brows)}</div>
<p class="note">Effects are the paired change from the unsteered replies, oriented so that positive is the good side of the instrument. Green: passes the criterion (at least twice the largest random-direction effect and at least three standard errors) in the good direction; red: passes in the other direction.</p>

<h2>Across the blocks</h2>
{rd['cross']}

<h2>Open items</h2>
<ul>{''.join(f'<li>{t}</li>' for t in rd['open'])}</ul>
</article>"""
    path = out / f"{block}.html"
    path.write_text(page)
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")
    return path


if __name__ == "__main__":
    build(sys.argv[1])
