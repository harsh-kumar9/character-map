"""The HHH page (D-055): honesty, helpfulness and harmlessness together, on the block pages' layout (cm/hhh/page.py).
Figures from cm/hhh/figs_hhh.py; the reading is in READING below, written after the results are in.

Reads  results/hhh/figures/hhh/ (figures, hhh_data.json, landscape_3d.json)
Writes results/hhh/figures/hhh/hhh.html

    python3 -m cm.hhh.page_hhh
"""
from __future__ import annotations

import json

from .page import CSS, LL, LM, R, img
from .blocks import BLOCKS

OUT = R / "figures" / "hhh"
PAGES = {"honesty": "https://claude.ai/artifact/QTcDSMpzByXX9w8o4sJ7Us",
         "helpfulness": "", "harmlessness": ""}      # filled once the block pages are published

READING = {
    "date": "2026-10-24",
    "stand": "Honesty, helpfulness and harmlessness, each built from its own definitions, read together inside "
             "Gemma-3-4B. Are they three things, do they pull against each other, and is there one assistant "
             "direction under them?",
    "status": "Exploratory. One model, synthetic scenarios written and checked by LLMs. The three blocks were built "
              "separately (honesty for its own investigation, helpfulness and harmlessness for this one, D-053 to "
              "D-057) and are read here with the same pipeline. The numbers are for spotting signals worth a test, "
              "not claims.",
    "tldr": [
        "<b>Frame first, block last.</b> In one space for all 21 units, the first component (81% at layer 17) is "
        "the frame. Inside a frame group the three blocks overlap: they sit 0.3 to 0.8 within-unit spreads apart "
        "in Gemma's own voice, against 15 between frame groups.",
        "<b>No assistant axis.</b> The assistant-persona direction carries 2 to 3% of a typical unit's squared "
        "length in every block. The units nearest it are low psychopathy and honesty-humility in action.",
        "<b>The tension runs through autonomy.</b> Refusing a harmful request is against autonomy in recognize, "
        "judge and act (-0.23 to -0.29), helping a harmless request goes with it (+0.23 to +0.62), and substance "
        "is not involved (-0.09 to +0.13).",
        "<b>Causally, helpfulness pushes on the other blocks; they barely push back.</b> Helpfulness directions move "
        "another block's readout in 6.5 to 6.9% of cells (|z| 3 or more), honesty and harmlessness in 1.7 to 2.3%, "
        "near the controls' false-positive rate (1.0 to 1.4%). Agreeableness, wellbeing and helping a harmless "
        "request in action lower the readout for refusing a harmful one (z -3.1 to -6.5 at layer 29).",
        "BEH",
    ],
    "question": "<p>Askell et al. (2021) defined the assistant as helpful, honest and harmless, and the label stuck: "
                "Claude's constitution and the Model Spec are organized around the same three, and Bai et al. (2022) "
                "trained for helpfulness and harmlessness and found them in tension. Each of the three has several "
                "definitions of its own, and each block page asks whether its definitions pick out one thing inside "
                "Gemma. Here we ask the same of the three blocks: is HHH three things or one, do the blocks pull "
                "against each other where the training literature says they do, and is there a single assistant "
                "direction that carries all of them?</p>",
    "setup": "<p>The three blocks are the units of their pages: six for honesty (self-knowledge, accuracy, "
             "sincerity, forthrightness, non-sycophancy, honesty-humility), seven for helpfulness, eight for "
             "harmlessness. Every unit has four frames (recognize, judge, act, self-describe) and is oriented to "
             "its good side, so a positive cosine means the two good sides go together. Directions, cosines, floors "
             "and steering are those of the block pages: diff of means over the first 32 reply tokens at layers "
             f"{LM} and {LL}, Gemma's verdict projected out of recognize and judge, standardized and centred "
             "cosine with a shuffle floor of 0.16 at layer 17, steering at layer 17 at the calibrated dose read as "
             "z against random directions.</p>",
    "r1": "<p>Put together, the three blocks look like each block alone. At layer 17 the first component carries "
          "81% of the variance and splits items about someone else's reply from items in Gemma's voice; at 29 it "
          "carries 39%, and the second component splits acting from self-describing (panel d). Inside a frame "
          "group the blocks overlap. In the act frame they sit 0.3 to 0.7 within-unit spreads apart at layer 17 "
          "(helpfulness and honesty closest), in self-description 0.4 to 0.8, when Gemma watches 1.1 to 1.3. At "
          "29 harmlessness drifts from the other two in self-description (1.0 from honesty, 1.7 from "
          "helpfulness).</p><p>So inside Gemma the frame decides more than the block does. A reply that refuses, "
          "a reply that cares and a reply that tells the truth are closer to each other than any of them is to a "
          "judgment of someone else's reply on the same topic.</p>",
    "r2": "<p>Within a block more pairs of definitions share a direction than between blocks: in the act frame "
          "30% of pairs inside a block pass the floor against 14% between blocks; in self-description 28% against "
          "21%. The pairs that cross blocks are few and specific. Honesty-humility goes with low psychopathy "
          "(+0.52 act, +0.59 self-describe) and with harmless acts (+0.30, +0.48), as HEXACO predicts. Autonomy "
          "goes with helping a harmless request (+0.62 act) and with legitimate intent (+0.31), and against "
          "refusing and declining (-0.26 to -0.28). Self-knowledge (a question Gemma can answer, against one it "
          "cannot) goes with helping a harmless request (+0.40, +0.49) and against refusing and declining (-0.26 "
          "to -0.29): not being able to answer and not being willing to sit on the same side. Forthrightness goes "
          "with not reading too literally in self-description (+0.37).</p>",
    "r3": "<p>D-055's first test was that refusing over-cautiously opposes substantive help, the tension in the "
          "constitution's over-caution list. It does not: substance and helping a harmless request are -0.09 to "
          "+0.13 in every frame, and substance and refusing a harmful request -0.06 to +0.10. The tension runs "
          "through autonomy instead. Refusing a harmful request and declining a harmful action are against "
          "autonomy in recognize, judge and act (-0.17 to -0.29 at layer 17); helping a harmless request (+0.23 "
          "to +0.62 in every frame) and charitable reading (+0.15 to +0.32) go with it. Gemma reads a refusal "
          "as overriding the person's choice, not as giving less.</p><p>Steering agrees in direction. Helping a "
          "harmless request in action lowers the refusing readout in action (z -6.5 at layer 29), and so do "
          "wellbeing (-4.2) and agreeableness (-3.1) in action. Care does the opposite in self-description "
          "(+3.7). Steering toward refusing lowers wellbeing in action (-2.2) and leaves autonomy and substance "
          "where they were. Autonomy's own direction moves refusal little (-1.5).</p>",
    "r4": "<p>Lu et al. (2026) find an assistant axis: a direction that separates the assistant persona from other "
          "roles and that steering along keeps the model in character. Our D1 is that contrast (the assistant vs "
          "another role). If HHH were the assistant persona seen from three sides, the units would line up with "
          "it. They do not. The mean squared raw cosine of a block's directions with D1 is 0.023 for honesty, "
          "0.018 for helpfulness and 0.027 for harmlessness: a typical unit puts 2 to 3% of its squared length on "
          "D1's line. Centred, most units sit inside the floor. The ones outside it are low psychopathy in action "
          "(+0.40), honesty-humility in action (+0.32), harmless acts in action and self-description (+0.27, "
          "+0.19), autonomy in action and self-description (+0.20, +0.17), accuracy in action (+0.19) and "
          "charitable reading in action (+0.17). The behaviors the blocks are named for, refusing, substance and "
          "sincerity, sit at -0.14 to +0.07. The persona leans toward the two personality traits and a few acts, "
          "not toward helpful, honest or harmless replies as such.</p>",
    "r5": "R5",
    "open": [
        "The blocks' behavior instruments are close to floor or ceiling for Gemma-3-4B (see each block page), so "
        "the behavior side of the tension needs instruments with room to move.",
        "Whether autonomy is causally part of the tension: its own direction does not move refusal at the "
        "calibrated dose.",
        "The large subjects (gemma-4-31B-it, Qwen3.5-27B), where the honesty geometry replicated.",
        "C28 (asks only when a request is truly ambiguous) joins helpfulness when built.",
    ],
}


def plot3d():
    D = json.loads((OUT / "landscape_3d.json").read_text())
    D["views"] = {k: v for k, v in D["views"].items() if k.endswith(f"layer {LM}") or k.endswith(f"layer {LL}")}
    D["meta"] = [{k: m[k] for k in ("name", "block", "frame", "good")} for m in D["meta"]]
    return f"""<div class="box"><h3>Rotate it</h3>
<p class="note">Every item of the three blocks on the first three principal components. Drag to rotate, scroll to zoom, hover for an item's definition, frame and side. Filled marks are the good side. The buttons switch the coloring, the layer, and between the space as is and each unit centred on its own mean.</p>
<div class="viewbar" role="group" aria-label="Color">
<button class="viewbtn on" id="c-block" type="button">Color by block</button><button class="viewbtn" id="c-frame" type="button">Color by frame group</button></div>
<div class="viewbar" role="group" aria-label="View">
<button class="viewbtn" id="v-17" type="button">Layer {LM}</button><button class="viewbtn on" id="v-29" type="button">Layer {LL}</button>
<button class="viewbtn" id="v-c" type="button">Each unit centred</button></div>
<div id="plot3d"></div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.1/plotly.min.js"></script>
<script>
(function(){{
  var D = {json.dumps(D, separators=(",", ":"))};
  var C3 = ["#2a78d6","#eb6834","#1baf7a"];
  var PAL = {{"block": {{"honesty":C3[0],"helpfulness":C3[1],"harmlessness":C3[2]}}, "frame": {{"about someone else":C3[0],"acting":C3[1],"self-describing":C3[2]}}}};
  var FRG = {{"a":"about someone else","b":"about someone else","c":"acting","d":"self-describing"}};
  var FRN = {{"a":"recognize","b":"judge","c":"act","d":"self-describe"}};
  var colorBy = "block", layer = "{LL}", centred = false;
  function css(v){{ return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888"; }}
  function draw(){{
    var V = D.views[(centred ? "each unit centred, layer " : "as-is, layer ") + layer];
    var byKey = {{}};
    D.meta.forEach(function(m, i){{
      if (m.block === "control") return;
      var g = colorBy === "block" ? m.block : FRG[m.frame];
      var k = g + (m.good ? "|good" : "|other");
      (byKey[k] = byKey[k] || {{x:[],y:[],z:[],t:[],g:g,good:m.good}});
      byKey[k].x.push(V.xyz[i][0]); byKey[k].y.push(V.xyz[i][1]); byKey[k].z.push(V.xyz[i][2]);
      byKey[k].t.push(m.name + " (" + m.block + "), " + FRN[m.frame] + ", " + (m.good ? "good side" : "other side"));
    }});
    var pal = PAL[colorBy];
    var traces = Object.keys(pal).reduce(function(acc, g){{ ["good","other"].forEach(function(s){{ var b = byKey[g + "|" + s]; if (!b) return; var c = pal[g];
      acc.push({{type:"scatter3d", mode:"markers", x:b.x, y:b.y, z:b.z, text:b.t, hovertemplate:"%{{text}}<extra></extra>",
        name: g + (b.good ? ", good side" : ", other side"),
        marker:{{size:2.2, color: b.good ? c : "rgba(0,0,0,0)", line:{{color:c, width:1}}, opacity:0.7}}}}); }}); return acc; }}, []);
    var ink = css("--ink2"), grid = css("--rule");
    var ax = function(n){{ return {{title:{{text:"PC" + n + " (" + (V["var"][n-1]*100).toFixed(0) + "%)"}}, color:ink, gridcolor:grid, zerolinecolor:grid}}; }};
    Plotly.react("plot3d", traces, {{paper_bgcolor:"rgba(0,0,0,0)", margin:{{l:0,r:0,t:0,b:0}}, font:{{color:ink, size:11}},
      legend:{{orientation:"h", y:-0.02}}, scene:{{xaxis:ax(1), yaxis:ax(2), zaxis:ax(3), bgcolor:"rgba(0,0,0,0)"}}}},
      {{displaylogo:false, responsive:true}});
  }}
  function on(id, grp){{ document.querySelectorAll(grp).forEach(function(b){{ b.classList.remove("on"); }}); document.getElementById(id).classList.add("on"); }}
  document.getElementById("c-block").onclick = function(){{ colorBy = "block"; on("c-block", "#c-block,#c-frame"); draw(); }};
  document.getElementById("c-frame").onclick = function(){{ colorBy = "frame"; on("c-frame", "#c-block,#c-frame"); draw(); }};
  document.getElementById("v-17").onclick = function(){{ layer = "{LM}"; on("v-17", "#v-17,#v-29"); draw(); }};
  document.getElementById("v-29").onclick = function(){{ layer = "{LL}"; on("v-29", "#v-17,#v-29"); draw(); }};
  document.getElementById("v-c").onclick = function(){{ centred = !centred; this.classList.toggle("on", centred); draw(); }};
  if (window.Plotly) draw();
}})();
</script>"""


def build():
    rd = READING
    fig = lambda f: img(OUT / f)
    links = " &middot; ".join(f'<a href="{u}">{b}</a>' if u else b for b, u in PAGES.items())
    n = {b: len(us) for b, us in BLOCKS.items()}
    page = f"""<title>Is HHH one thing, to Gemma?</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>{CSS}</style>
<article>
<div class="eyebrow">character-map &middot; HHH line &middot; exploratory &middot; {rd['date']}</div>
<h1>Is HHH one thing, to Gemma?</h1>
<p class="stand">{rd['stand']}</p>
<div class="meta"><span><b>Model</b> Gemma-3-4B-IT</span><span><b>Blocks</b> honesty {n['honesty']}, helpfulness {n['helpfulness']}, harmlessness {n['harmlessness']} units</span><span><b>Block pages</b> {links}</span><span><b>Code and results</b> <a href="https://github.com/harsh-kumar9/character-map">github.com/harsh-kumar9/character-map</a></span></div>
<p class="note"><b>Epistemic status.</b> {rd['status']}</p>
<div class="box accent"><h3>TL;DR</h3><ul>{''.join(f'<li>{t}</li>' for t in rd['tldr'])}</ul></div>

<h2>The question</h2>
{rd['question']}

<h2>Setup, briefly</h2>
{rd['setup']}

<h2>Result 1: frame first, block last</h2>
<figure><img src="{fig('hhh_landscape.png')}" alt="Every item of the three blocks on the first two principal components, colored by block and by frame group">
<figcaption><b>Figure 1. The three blocks in one space.</b> Every item of the 21 units projected into the subspace spanned by their (definition, frame) directions, first two principal components, at layers {LM} (a, b) and {LL} (c, d). Left: colored by block; right: the same points colored by frame group.</figcaption></figure>
{rd['r1']}
{plot3d()}

<h2>Result 2: which definitions cross blocks</h2>
<figure><img src="{fig('hhh_cosine.png')}" alt="Centred cosine of every unit with every other, act frame and self-describe frame">
<figcaption><b>Figure 2. Cosines across the three blocks, layer {LM}.</b> Centred cosine of every pair of units in the act frame (a) and the self-describe frame (b), every unit oriented to its good side; blue positive, red negative. Lines separate the blocks; label color is the block.</figcaption></figure>
{rd['r2']}

<h2>Result 3: the tension between helpful and harmless</h2>
<figure><img src="{fig('hhh_tension.png')}" alt="Cosines of the refusal units with substance, autonomy and wellbeing per frame, and steering effects between them">
<figcaption><b>Figure 3. Where the tension runs.</b> (a) Centred cosine at layer {LM} of four harmlessness units with three helpfulness units, one marker per frame; grey band, within the shuffle floor. (b) Steering z, toward the target's good side, at layer {LL}, for the cells D-055 named; grey band, |z| under 3.</figcaption></figure>
{rd['r3']}

<h2>Result 4: no single assistant axis</h2>
<figure><img src="{fig('hhh_d1.png')}" alt="Centred cosine of every unit with the assistant-persona direction">
<figcaption><b>Figure 4. Every unit against the assistant-persona direction (D1), layer {LM}.</b> Centred cosine; circles act, squares self-describe; grey band, within the shuffle floor. Label color is the block.</figcaption></figure>
{rd['r4']}

<h2>Result 5: behavior under steering</h2>
{rd['r5']}

<h2>Open items</h2>
<ul>{''.join(f'<li>{t}</li>' for t in rd['open'])}</ul>
</article>"""
    path = OUT / "hhh.html"
    path.write_text(page)
    print("wrote", path, f"{path.stat().st_size / 1e6:.1f} MB")
    return path


if __name__ == "__main__":
    build()
