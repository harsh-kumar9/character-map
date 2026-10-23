"""Render the honesty case-study page from the collected numbers. Run: python3 build_page.py <outdir>"""
import json, base64, sys, html
from pathlib import Path
OUT = Path(sys.argv[1])
import os
LM, LL = int(os.environ.get("CM_LMID", "17")), int(os.environ.get("CM_LLATE", "29")); LMs, LLs = str(LM), str(LL)
L_STEER = int(os.environ.get("CM_LSTEER", LM))
MODEL_NAME = os.environ.get("CM_MODEL_NAME", "Gemma-3-4B-IT")
D = json.load(open(OUT / "honesty_data.json")); MP = json.load(open(OUT / "honesty_map.json")); LT = json.load(open(OUT / "honesty_lattice.json"))
img = base64.b64encode(open(OUT / "honesty_lattice.png", "rb").read()).decode()
LAND = (OUT / "landscape_frames.png").exists() and (OUT / "landscape_3d.json").exists()
if LAND:
    land_img = base64.b64encode(open(OUT / "landscape_frames.png", "rb").read()).decode()
    LS = json.load(open(OUT / "landscape_frames_stats.json"))
    LJ = json.load(open(OUT / "landscape.json"))
    L3D = open(OUT / "landscape_3d.json").read()
    LAND_TEXT = open(OUT / "landscape_text.html").read() if (OUT / "landscape_text.html").exists() else ""
    L3D_IMG = ""
def land_section():
    if not LAND: return ""
    v17 = LJ["layers"][LMs]["variance_all"]; v29 = LJ["layers"][LLs]["variance_all"]
    c17 = LJ["layers"][LMs]["consistency_with_s4"]; c29 = LJ["layers"][LLs]["consistency_with_s4"]
    return f"""<h3>The same block, item by item</h3>
<figure><img src="data:image/png;base64,{land_img}" alt="Items of the six honesty definitions in the honesty subspace, one panel per frame group">
<figcaption><b>Figure 2. Inside Gemma, where an honesty item sits is set first by its frame, then by which honesty it is, and last by whether it is the honest or the dishonest side.</b> Layer {LS["layer"]}. Each mark is one scenario in one frame, projected into the subspace spanned by the {len(LJ["layers"][LLs]["direction_ids"])} (definition, frame) directions and shown on its first two principal components; all three panels share the axes, and the grey marks in each panel are the other frames' items. Color is the definition; filled marks are the honest side of a contrast and hollow marks the other side; the large marks are each unit's two centroids, with an arrow from the dishonest side to the honest one. At most 150 items per unit and side are drawn, from {LJ["n_items"]} in all. Frame groups sit {LS["frame_separation"]["own-act"]:.1f} to {LS["frame_separation"]["act-self"]:.1f} within-unit spreads apart; inside a frame the epistemic definitions sit apart from the character definitions; the honest-vs-dishonest gap is the smallest effect for every contrast except self-knowledge.</figcaption></figure>
<div class="box"><h3>Rotate it</h3>
<p class="note">The same points on the first three components. Drag to rotate, scroll to zoom, hover for an item's definition, frame and side. Open marks are the dishonest side of a contrast. The thick segments run from each unit's dishonest-side centroid (white) to its honest-side centroid. "Color by frame" recolors the same points by frame instead of definition, which is the easiest way to see the three plates; the other buttons switch layer and between the space as is and each unit centred on its own mean. Click a legend entry to hide or show it.</p>
<div id="land3d" style="width:100%;height:560px"></div>
<div class="viewbar" id="viewbar"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.1/plotly.min.js"></script>
<script>
(function(){{
  var D = {L3D};
  var COL = {{"Self-knowledge":"#009E73","Accuracy":"#E69F00","Sincerity":"#0072B2","Honesty-humility":"#CC79A7","Formality (control)":"#898781"}};
  var SYM = {{"a":"x","b":"diamond","c":"circle","d":"square"}};
  var FCOL = {{"a":"#6a5acd","b":"#1f78b4","c":"#d95f02","e":"","d":"#1b9e77"}};
  var colorBy = "definition";
  var FRN = {{"a":"recognize","b":"judge","c":"act","d":"self-describe"}};
  var CIDN = {{"A9":"Self-knowledge","C2":"Accuracy","C1":"Sincerity","C3":"Forthrightness","C8":"Non-sycophancy","C19":"Honesty-humility","K1":"Formality (control)"}};
  var views = Object.keys(D.views).sort(function(a,b){{ return (b.indexOf("{LLs}")>=0) - (a.indexOf("{LLs}")>=0) || (a.indexOf("as-is")>=0 ? -1 : 1); }});
  function tokens(){{ var cs = getComputedStyle(document.documentElement); return {{bg: cs.getPropertyValue("--surface").trim()||"#fff", ink: cs.getPropertyValue("--ink").trim()||"#111", ink2: cs.getPropertyValue("--ink2").trim()||"#555", rule: cs.getPropertyValue("--rule").trim()||"#ddd"}}; }}
  function build(viewName){{
    var V = D.views[viewName]; var xyz = V.xyz; var traces = [];
    var groups = {{}};
    D.meta.forEach(function(m, i){{ if (m.def.indexOf("Formality")===0) return; var k = (colorBy==="frame" ? FRN[m.frame] : m.def)+"|"+m.frame+"|"+m.pol; (groups[k] = groups[k] || []).push(i); }});
    var seen = {{}};
    Object.keys(groups).sort().forEach(function(k){{
      var p = k.split("|"); var idx = groups[k]; var ctrl = false;
      var first = !seen[p[0]]; seen[p[0]] = true; var col = colorBy==="frame" ? FCOL[p[1]] : COL[p[0]];
      var sym = SYM[p[1]] + (p[2]==="neg" && SYM[p[1]]!=="x" ? "-open" : ""); if (p[2]==="neg" && SYM[p[1]]==="x") sym = "cross";
      traces.push({{type:"scatter3d", mode:"markers", name: p[0], legendgroup: p[0], showlegend: first,
        x: idx.map(function(i){{return xyz[i][0];}}), y: idx.map(function(i){{return xyz[i][1];}}), z: idx.map(function(i){{return xyz[i][2];}}),
        marker: {{size: 2.6, color: col, symbol: sym, opacity: 0.55, line:{{width:0}}}},
        hovertemplate: (colorBy==="frame" ? D.meta[idx[0]].def : p[0])+", "+FRN[p[1]]+", "+(p[2]==="pos"?"honest side":"other side")+"<extra></extra>"}});
    }});
    Object.keys(V.centroids).forEach(function(u){{
      var c = V.centroids[u]; var q = u.split("|"); if (q[0]==="K1") return; var name = CIDN[q[0]]; var col = colorBy==="frame" ? FCOL[q[2]] : COL[name];
      traces.push({{type:"scatter3d", mode:"lines+markers", name: name+", "+FRN[q[2]], legendgroup: colorBy==="frame" ? FRN[q[2]] : name, showlegend:false,
        x:[c[0][0],c[1][0]], y:[c[0][1],c[1][1]], z:[c[0][2],c[1][2]], line:{{color:col, width:5}},
        marker:{{size:[5,6], color:[ "#ffffff", col], symbol: SYM[q[2]]==="x"?"circle":SYM[q[2]], line:{{color:col, width:2}}}},
        hovertemplate: name+", "+FRN[q[2]]+": %{{text}}<extra></extra>", text:["centroid, other side","centroid, honest side"]}});
    }});
    var t = tokens(); var ax = function(n, i){{ return {{title:{{text:n+" ("+Math.round(V.var[i]*100)+"%)", font:{{size:11, color:t.ink2}}}}, gridcolor:t.rule, zerolinecolor:t.rule, backgroundcolor:t.bg, showbackground:true, tickfont:{{size:9, color:t.ink2}}}}; }};
    var layout = {{paper_bgcolor:t.bg, plot_bgcolor:t.bg, font:{{color:t.ink, family:"Source Sans 3, Helvetica, Arial, sans-serif"}}, margin:{{l:0,r:0,t:10,b:0}},
      scene:{{xaxis:ax("PC1",0), yaxis:ax("PC2",1), zaxis:ax("PC3",2), aspectmode:"data", camera:{{eye:{{x:1.5,y:1.3,z:0.9}}}}}},
      legend:{{font:{{size:11, color:t.ink}}, itemsizing:"constant", bgcolor:"rgba(0,0,0,0)", x:1, y:1, title:{{text:"click to hide or show", font:{{size:10, color:t.ink2}}}}}}, showlegend:true}};
    Plotly.newPlot("land3d", traces, layout, {{responsive:true, displaylogo:false, modeBarButtonsToRemove:["toImage"]}});
  }}
  var bar = document.getElementById("viewbar"); var current = views[0];
  var cb = document.createElement("button"); cb.type="button"; cb.id="view-colour"; cb.className="viewbtn"; cb.textContent="color by frame";
  cb.addEventListener("click", function(){{ colorBy = colorBy==="definition" ? "frame" : "definition"; cb.textContent = colorBy==="definition" ? "color by frame" : "color by definition"; build(current); }});
  views.forEach(function(v){{ var b = document.createElement("button"); b.type="button"; b.id="view-"+v.replace(/[^a-z0-9]+/gi,"-"); b.textContent = v; b.className = "viewbtn"+(v===current?" on":"");
    b.addEventListener("click", function(){{ current = v; build(v); Array.prototype.forEach.call(bar.children, function(x){{ x.classList.toggle("on", x===b); }}); }}); bar.appendChild(b); }});
  bar.appendChild(cb);
  build(current);
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function(){{ build(current); }});
}})();
</script>
{('<figure><img src="data:image/png;base64,' + L3D_IMG + '" alt="Static 3D rendering of the three components"><figcaption><b>Figure 3. The three components in one view.</b> The same points as Figure 2 drawn in three dimensions at a fixed viewpoint, layer {LM} (a) and layer {LL} (b); the rotatable version above is the same data. The three plates are the utterance types; the long green segments are the self-knowledge contrasts, which cross from the act plate to the self-describe plate.</figcaption></figure>') if L3D_IMG else ''}
{LAND_TEXT}"""


CORE = [("Self-knowledge", "A9", "all"), ("Accuracy", "C2", "all"), ("Sincerity", "C1", "all"), ("Forthrightness", "C3", "all"),
        ("Non-sycophancy", "C8", "sycophantic_agreement"), ("Honesty-humility", "C19", "honesty_humility")]   # D-043, six since D-049
FOUR = CORE   # older call sites
REST = [("Truthful vs boastful", "C4", "boastful"), ("Truthful vs self-deprecating", "C4", "self-deprecating"),
        ("Integrity", "C12", "integrity"), ("Courage under pushback", "C11", "all"), ("Observed vs unobserved", "E1", "all"),
        ("Pushback present vs absent", "E5", "all"), ("Honest option vs kind option", "F1", "all")]
FR = {"a": "recognize", "b": "judge", "c": "act", "d": "self-describe"}
PAIR = {"a-b": "recognize–judge", "c-d": "act–self-describe", "a-c": "recognize–act", "a-d": "recognize–self-describe", "b-c": "judge–act", "b-d": "judge–self-describe"}
HELD = {("A9", "d"): "KUQ", ("C2", "c"): "TruthfulQA", ("C1", "c"): "MASK provided_facts", ("C19", "d"): "HEXACO-PI-R", ("C8", "c"): "SycophancyEval answer", ("C4", "c"): "MWE self-awareness",
        ("C12", "c"): "WDCT", ("C11", "c"): "SYCON/Moral Stories (near), are-you-sure (far)", ("E1", "c"): "alignment-faking tiers", ("E1", "d"): "MWE not-watched",
        ("E5", "c"): "SYCON debate", ("F1", "b"): "AIRiskDilemmas"}
fid = D["fidelity"]; tr = D["transfer"]; beh = D["behavior"]
def f2(v): return "–" if v is None else (f"{v:.2f}")
def esc(s): return html.escape(str(s))
def key(c, s): return f"{c}|{s}"

# ---- fidelity table (four)
def fid_rows(lst):
    rows = []
    for name, c, s in lst:
        for L in (LMs, LLs):
            r = fid[key(c, s)][L]
            for f in r["frames"]:
                h = r["heldout"].get(f); far = r["heldout_far"].get(f)
                hd = HELD.get((c, f), "")
                htxt = (f"{f2(h)} ({hd})" if h is not None else "not scored") + (f"; far {f2(far)}" if far is not None else "")
                rows.append((name, c, FR[f], L, r["within"][f], htxt))
    return rows
def fid_table(lst):
    out = [f'<table><thead><tr><th>Definition</th><th>Frame</th><th>Within-frame AUROC, layer {LM}</th><th>layer {LL}</th><th>Held-out benchmark AUROC (layer {LM} / {LL})</th></tr></thead><tbody>']
    rows = fid_rows(lst)
    seen = {}
    for name, c, fr, L, w, htxt in rows:
        seen.setdefault((name, c, fr), {})[L] = (w, htxt)
    for (name, c, fr), d in seen.items():
        w17, h17 = d.get(LMs, (None, "")); w29, h29 = d.get(LLs, (None, ""))
        hcell = h17 if h17 == h29 else f"{h17} / {h29}"
        cls = ' class="weak"' if (w17 is not None and w17 < 0.7) else ""
        out.append(f'<tr><td>{esc(name)} <span class="id">{c}</span></td><td>{fr}</td><td{cls}>{f2(w17)}</td><td{" class=weak" if (w29 is not None and w29 < 0.7) else ""}>{f2(w29)}</td><td>{esc(hcell)}</td></tr>')
    out.append('</tbody></table>'); return "\n".join(out)

# ---- transfer table (four)
def tr_table(lst, pairs=None):
    out = [f'<table><thead><tr><th>Definition</th><th>Frame pair</th><th>Transfer AUROC, layer {LM}</th><th>Call</th><th>layer {LL}</th><th>Call</th></tr></thead><tbody>']
    base = {t["layer"]: t["base_verdict"] for t in tr}
    for name, c, s in lst:
        rs = [t for t in tr if t["concept"] == c and t["sub"] == s and (pairs is None or t["pair"] in pairs)]
        for pair in sorted({t["pair"] for t in rs}, key=lambda p: list(PAIR).index(p)):
            cells = []
            for L in (LM, LL):
                t = next((x for x in rs if x["layer"] == L and x["pair"] == pair), None)
                if t is None: cells += ["–", "–"]; continue
                cl = {"shared": "shared", "dissociated": "split", "readout-dependent": "in between"}[t["class"]]
                v = t["verdict"]; mark = "low" if v < 0.5 else ("hi" if v >= base[L] + 0.10 else "")
                cells += [f'<td class="{mark}">{v:.2f}</td>', f"<td>{cl}</td>"]
            out.append(f'<tr><td>{esc(name)} <span class="id">{c}</span></td><td>{PAIR[pair]}</td>' + "".join(x if x.startswith("<td") else f"<td>{x}</td>" for x in cells) + "</tr>")
    out.append("</tbody></table>"); return "\n".join(out), base

# ---- cosine key numbers (four), from honesty_map.json
def cosv(L, u, v):
    m = MP[str(L)]; i, j = m["units"].index(u), m["units"].index(v); return m["cos"][i][j]
U = {"sk_c": "A9|all|c", "sk_d": "A9|all|d", "ac_c": "C2|all|c", "ac_d": "C2|all|d", "si_a": "C1|all|a", "si_b": "C1|all|b", "si_c": "C1|all|c", "si_d": "C1|all|d",
     "hh_c": "C19|honesty_humility|c", "hh_d": "C19|honesty_humility|d", "fo_a": "C3|all|a", "fo_c": "C3|all|c", "fo_d": "C3|all|d",
     "ns_a": "C8|sycophantic_agreement|a", "ns_c": "C8|sycophantic_agreement|c", "ns_d": "C8|sycophantic_agreement|d", "k_c": "K1|all|c", "k_d": "K1|all|d"}
floor17 = MP[LMs]["floor"]; floor29 = MP[LLs]["floor"]
CK = {"A9": "sk", "C2": "ac", "C1": "si", "C3": "fo", "C8": "ns", "C19": "hh"}
def def_pairs(L, f="c"):
    out = []
    for i, a in enumerate(CORE):
        for b in CORE[i + 1:]:
            ua, ub = f"{a[1]}|{a[2]}|{f}", f"{b[1]}|{b[2]}|{f}"
            if ua in MP[str(L)]["units"] and ub in MP[str(L)]["units"]:
                out.append((a[0].lower(), b[0].lower(), cosv(L, ua, ub)))
    return out
def above(L, f="c"):
    fl = MP[str(L)]["floor"]
    return [(a, b, v) for a, b, v in def_pairs(L, f) if abs(v) >= fl]
def fmt_pairs(ps): return "; ".join(f"{a}\u2013{b} {v:+.2f}" for a, b, v in ps) if ps else "none"
def cos_table():
    pairs = [(f"{a[0].lower()} vs {b[0].lower()}", CK[a[1]] + "_c", CK[b[1]] + "_c") for i, a in enumerate(CORE) for b in CORE[i + 1:]]
    out = [f'<table><thead><tr><th>Pair of definitions</th><th>Acting, layer {LM}</th><th>layer {LL}</th><th>Self-describing, layer {LM}</th><th>layer {LL}</th></tr></thead><tbody>']
    for lab, a, b in pairs:
        ad, bd = a.replace("_c", "_d"), b.replace("_c", "_d")
        vals = [cosv(LM, U[a], U[b]), cosv(LL, U[a], U[b]), cosv(LM, U[ad], U[bd]), cosv(LL, U[ad], U[bd])]
        cells = "".join(f'<td class="{"hi" if v >= 0.16 else ("low" if v <= -0.16 else "")}">{v:+.2f}</td>' for v in vals)
        out.append(f"<tr><td>{lab}</td>{cells}</tr>")
    out.append("</tbody></table>"); return "\n".join(out)

# ---- behavior table (four)
BC = {"A9": ["doing+", "doing-"], "C2": ["doing+", "doing-"], "C1": ["doing+", "doing-", "knowing+", "knowing-"], "C3": ["doing+", "doing-", "knowing+", "knowing-"],
      "C8": ["doing+:sycophantic_agreement"], "C19": ["doing+:honesty_humility"]}
CN = {"doing+": "act direction, +dose", "doing-": "act direction, −dose", "knowing+": "judge direction, +dose", "knowing-": "judge direction, −dose", "doing+:honesty_humility": "act direction, +dose", "doing+:sycophantic_agreement": "act direction, +dose"}
def beh_table():
    out = ['<table><thead><tr><th>Definition</th><th>Instrument</th><th>Judge</th><th>Unsteered</th><th>Condition</th><th>Effect ± SE</th><th>Largest random effect</th><th>Passes</th></tr></thead><tbody>']
    for name, c, s in FOUR:
        rows = [b for b in beh if b["concept"] == c and b["cond"] in BC[c]]
        for b in rows:
            out.append(f'<tr><td>{esc(name)} <span class="id">{c}</span></td><td>{esc(b["instrument"])}</td><td>{b["judge"]}</td><td>{b["baseline"]:.2f}</td><td>{CN[b["cond"]]}</td>'
                       f'<td>{b["effect"]:+.3f} ± {b["se"]:.3f}</td><td>{f2(b["max_random"]) if b["max_random"] is not None else "–"}</td><td>{"yes" if b["passes"] else "no"}</td></tr>')
    out.append("</tbody></table>"); return "\n".join(out)

# ---- appendix: the other seven
def app_table():
    act17 = D["cos"]["act_L"+LMs]; self17 = D["cos"]["self_L"+LMs]
    def maxcos(tab, u):
        if u not in tab["units"]: return "–"
        i = tab["units"].index(u); best = None
        for name, c, s in FOUR:
            v = f"{c}|{s}|{u.split('|')[2]}"
            if v in tab["units"]:
                x = tab["cos"][i][tab["units"].index(v)]
                if best is None or abs(x) > abs(best[0]): best = (x, name)
        return f"{best[0]:+.2f} ({best[1]})" if best else "–"
    out = [f'<table><thead><tr><th>Concept</th><th>Frames</th><th>Within-frame AUROC, layer {LM}</th><th>Held-out, layer {LM}</th><th>Doing pair transfer {LM} / {LL}</th><th>Nearest of the four, acting</th><th>Nearest of the four, self-describing</th></tr></thead><tbody>']
    for name, c, s in REST:
        r = fid[key(c, s)][LMs]; frs = "".join(r["frames"])
        w = ", ".join(f"{FR[f][0]} {r['within'][f]:.2f}" for f in r["frames"])
        h = ", ".join(f"{FR[f][0]} {v:.2f}" for f, v in r["heldout"].items() if v is not None) or "not scored"
        pair = "c-d" if "c" in r["frames"] and "d" in r["frames"] else ("b-c" if "b" in r["frames"] else None)
        tv = "–"
        if pair:
            t17 = next((t["verdict"] for t in tr if t["concept"] == c and t["sub"] == s and t["pair"] == pair and t["layer"] == LM), None)
            t29 = next((t["verdict"] for t in tr if t["concept"] == c and t["sub"] == s and t["pair"] == pair and t["layer"] == LL), None)
            tv = f"{PAIR[pair]} {f2(t17)} / {f2(t29)}"
        out.append(f'<tr><td>{esc(name)} <span class="id">{c}</span></td><td>{frs}</td><td>{w}</td><td>{h}</td><td>{tv}</td><td>{maxcos(act17, f"{c}|{s}|c")}</td><td>{maxcos(self17, f"{c}|{s}|d")}</td></tr>')
    out.append("</tbody></table>"); return "\n".join(out)

tr_html, base = tr_table(FOUR)
selfz = LT["selfz_late"]; arrows = LT["arrows_late"]
c1 = fid["C1|all"]; hh = fid["C19|honesty_humility"]; a9 = fid["A9|all"]; c2 = fid["C2|all"]
def T(c, s, pair, L): return next(t["verdict"] for t in tr if t["concept"] == c and t["sub"] == s and t["pair"] == pair and t["layer"] == L)
b1 = {b["cond"]: b for b in beh if b["concept"] == "C1" and b["judge"] == "gemma"}; b1q = {b["cond"]: b for b in beh if b["concept"] == "C1" and b["judge"] == "qwen"}
IS_DEFAULT = MODEL_NAME == "Gemma-3-4B-IT"
HAVE_CAUSAL = bool(arrows) or bool(selfz); HAVE_BEH = bool(b1) and bool(b1q)
NAMES = {"A9": "self-knowledge", "C2": "accuracy", "C1": "sincerity", "C3": "forthrightness", "C8": "non-sycophancy", "C19": "honesty-humility"}
def uname(u):
    c, sub, f = u.split("|"); return f"{NAMES.get(c, c)} ({FR[f]})"
def causal_generic():
    if not HAVE_CAUSAL:
        return "<p>The steering stage (S5a) has not run for this model yet.</p>"
    lines = [f"{uname(a)} \u2192 {uname(b)} (z {z:.1f})" for a, b, z in arrows]
    selfs = ", ".join(f"{uname(u)} {z}" for u, z in selfz.items())
    return (f"<p>Adding a direction at layer {L_STEER} and reading every direction in the block at layer {LL}: "
            + (f"arrows with z at least 3: {'; '.join(lines)}." if lines else "no cross-unit effect reaches z 3.")
            + f" Each source's effect on its own readout: {selfs}.</p>")
def beh_generic():
    if not HAVE_BEH:
        return "<p>The behavior stage (S5b) has not run for this model yet.</p>"
    passes = [b for b in beh if b["passes"] and b["concept"] in NAMES]
    cands = [b for b in beh if b["concept"] in NAMES and b["max_random"]]
    near = max(cands, key=lambda b: abs(b["effect"]) / b["max_random"]) if cands else None
    txt = "<p>The criterion is an effect at least twice the largest of four random directions and at least three standard errors. "
    txt += (f"{len(passes)} condition(s) pass: " + "; ".join(f"{NAMES[b['concept']]} {CN.get(b['cond'], b['cond'])} under {b['judge']} ({b['effect']:+.3f})" for b in passes) + ".") if passes else "Nothing passes."
    if near:
        txt += f" The nearest miss is {NAMES[near['concept']]} ({CN.get(near['cond'], near['cond'])}, {near['judge']}): {near['effect']:+.3f} \u00b1 {near['se']:.3f} against random effects up to {near['max_random']:.3f}."
    return txt + "</p>"
banner = "" if IS_DEFAULT else ('<div class="box warn"><h3>Replication run on ' + MODEL_NAME + '</h3><p class="note">Every number and table on this page comes from this model\'s own run through the same pipeline. The connecting prose was written for Gemma-3-4B-IT and is kept only where it names the numbers; sections marked pending have not run for this model, and the interpretive reading for this model is still to be written.</p></div>\n')


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
td.hi{background:var(--ok-soft)}td.low,td.weak{background:var(--warn-soft)}
.id{font-family:var(--mono);font-size:.74rem;color:var(--muted);margin-left:.2rem}
figure{margin:1.4rem 0}figure img{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--rule);border-radius:4px}
figcaption{font-size:.88rem;color:var(--ink2);margin-top:.5rem}
.note{font-size:.9rem;color:var(--ink2)}
.viewbar{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.6rem}
.viewbtn{font:inherit;font-size:.85rem;padding:.3rem .7rem;border:1px solid var(--rule);border-radius:4px;background:var(--surface);color:var(--ink);cursor:pointer}
.viewbtn.on{border-color:var(--accent);background:var(--accent-soft)}
.viewbtn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
a{color:var(--accent)}
code{font-family:var(--mono);font-size:.84em;background:var(--accent-soft);padding:0 .25em;border-radius:3px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:1rem}@media (max-width:640px){.two{grid-template-columns:1fr}}
.two>div{min-width:0}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.3rem 1rem;font-size:.95rem}dt{font-weight:600}dd{margin:0}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
"""

sincerity_mask = c1[LMs]["heldout"].get("c")
CAUSAL_BULLET = (f'''<li><b>Steering stays in its lane.</b> Adding the sincerity-in-action direction moves sincerity self-description (z {arrows[0][2]:.1f}) and nothing else in the block; the accuracy and self-knowledge directions do not move even their own readouts at the coherent dose (z {selfz['C2|all|c']}, {selfz['A9|all|c']}).</li>''') if IS_DEFAULT else ""
BEH_BULLET = (f'''<li><b>No honesty behavior moves</b> at that dose, on MASK, SelfAware, TRAIT or AIRiskDilemmas. The nearest miss is sincerity on MASK ({b1['doing+']['effect']:+.3f} ± {b1['doing+']['se']:.3f}, random up to {b1['doing+']['max_random']:.3f}).</li>''') if IS_DEFAULT else ""
R2_PARA = (f'''<p>Three things follow. First, MASK is an akrasia benchmark: the model's belief is elicited, then a pressure prompt asks it to assert otherwise. Our sincerity direction, learned on matched sincere and deceptive replies, reads MASK at chance ({sincerity_mask:.2f}). The direction we have is not the one that separates holding to a belief from giving it up under pressure, which is the gap that matters. Second, the concepts that measure gaps directly are weakly represented: integrity vs hypocrisy has a within-frame AUROC of {fid['C12|integrity'][LMs]['within']['c']:.2f} in the act frame and reads the WDCT held-out set at {fid['C12|integrity'][LMs]['heldout']['c']:.2f}; holding a position under pushback has {fid['C11|all'][LMs]['within']['c']:.2f} within frame, {fid['C11|all'][LMs]['heldout']['c']:.2f} on near held-out items and {fid['C11|all'][LMs]['heldout_far']['c']:.2f} on are-you-sure at layer {LM} ({fid['C11|all'][LLs]['heldout_far']['c']:.2f} at layer {LL}). The situation (pushback) is far better encoded than the virtue (holding). Third, the one closed gap is honesty-humility's, and that is the definition whose two frames are both statements about the kind of agent one is. Integration in this model is easiest where neither frame is conduct.</p>''') if IS_DEFAULT else ""
ACT_W = ", ".join(f"{n.lower()} {fid[key(c, s)][LMs]['within']['c']:.2f}" for n, c, s in CORE if fid[key(c, s)][LMs]["within"].get("c") is not None)
R3_PARA = (f'''<p>Act-frame within-frame AUROC at layer {LM}: {ACT_W}. Sincerity is among the weakest and the only one whose held-out benchmark is its own definition's. Its within-frame AUROC is {c1[LMs]['within']['c']:.2f} in the act frame at layer {LM} and {c1[LLs]['within']['c']:.2f} at layer {LL}, and the direction scores the MASK held-out set at {sincerity_mask:.2f}, which is chance, at every layer. Either the sincerity items are weak or a 4B model has little that separates a sincere reply from a deceptive one when the content is otherwise matched. This matters because sincerity is the definition the deception literature means, and MASK is its benchmark. D-045 lists the diagnosis.</p>''') if IS_DEFAULT else ""
R4_BLOCK = (f'''<h2>Result 4: steering stays in its lane</h2>
<p>Adding a direction at layer {L_STEER} and reading every direction in the block at layer {LL} gives two arrows with z at least 3: sincerity-in-action moves sincerity self-description (z {arrows[0][2]:.1f}) and honesty-humility-in-action moves honesty-humility self-description (z {arrows[1][2]:.1f}). Nothing crosses a definition. The judge direction for sincerity moves recognition and judgment (z 2.5 and 2.3) and not acting or self-description. Accuracy and self-knowledge do not move even their own act-frame readouts at this dose (z {selfz['C2|all|c']} and {selfz['A9|all|c']}; self-knowledge reaches its own self-description at z 2.9 at layer 22). In the wider block, one broad direction, integrity-in-action, raises most honesty self-descriptions and pushes the white-lie readout toward the kind option (appendix).</p>
''') if IS_DEFAULT else ""
R5_BLOCK = (f'''<h2>Result 5: no honesty behavior moves at the coherent dose</h2>
<div class="tbl">{beh_table()}</div>
<p>The criterion is an effect at least twice the largest of four random directions and at least three standard errors. Nothing passes. Sincerity on MASK is the nearest miss under both judges ({b1['doing+']['effect']:+.3f} with gemma-4-31B, {b1q['doing+']['effect']:+.3f} with Qwen3.5-27B, against random effects up to {max(b1['doing+']['max_random'], b1q['doing+']['max_random']):.3f}). The dose is 1/128 of the layer-17 residual norm, the largest that kept generic replies coherent. A dose-response run at 2, 4 and 8 times that dose (D-041) has been launched and its results are not in yet. Until then these are nulls at one dose, not evidence that the directions are behaviorally inert.</p>
''') if IS_DEFAULT else ""
OI_SINCERITY = (f'''<li><b>Sincerity is weak.</b> Within-frame {c1[LMs]['within']['c']:.2f} and MASK at chance. Diagnose per source, try a logistic probe, audit the act-frame prefills, swap MASK and Apollo roles (D-045). Honesty cannot carry the paper until this is settled.</li>''') if IS_DEFAULT else ""
OI_DOSE = (f'''<li><b>Dose-response (D-041)</b> is running on ada; results not on the Mac yet. It decides whether Result 4 is a dose null or a real null.</li>''') if IS_DEFAULT else ""
OI_ONEMODEL = (f'''<li><b>One model.</b> OLMo-3-7B-Instruct and Llama-3.1-8B-Instruct get the same items next (S6). The honesty block should run first there, before the other 48 concepts.</li>''') if IS_DEFAULT else ""
OI_GAPS = (f'''<li><b>The gap instruments are not read by our directions.</b> Integrity reads WDCT at {fid['C12|integrity'][LMs]['heldout']['c']:.2f}, courage reads are-you-sure at {fid['C11|all'][LMs]['heldout_far']['c']:.2f} at layer {LM}, sincerity reads MASK at {sincerity_mask:.2f}. Either the gaps are represented differently from the static contrasts, or they live at a different position than the first 32 reply tokens (the moment of capitulation is later). Worth a per-token look (D-046).</li>''') if IS_DEFAULT else ""
if not (IS_DEFAULT and HAVE_CAUSAL):
    CAUSAL_BULLET = "<li><b>Steering.</b> " + causal_generic()[3:-4] + "</li>"
    R4_BLOCK = "<h2>Result 4: steering</h2>\n" + causal_generic() + "\n"
if not (IS_DEFAULT and HAVE_BEH):
    BEH_BULLET = "<li><b>Behavior.</b> " + beh_generic()[3:-4] + "</li>"
    R5_BLOCK = "<h2>Result 5: behavior under steering</h2>\n" + (('<div class="tbl">' + beh_table() + "</div>\n") if HAVE_BEH else "") + beh_generic() + "\n"
page = f"""<title>{"What is honesty, to Gemma?" if IS_DEFAULT else "What is honesty, to " + MODEL_NAME + "?"}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>{CSS}</style>
<article>
<div class="eyebrow">character-map · research note · draft · 22 October 2026</div>
<h1>{"What is honesty, to Gemma?" if IS_DEFAULT else "What is honesty, to " + MODEL_NAME + "?"}</h1>
<p class="stand">Honesty has at least six working definitions in the literature and the two model specifications. Inside Gemma-3-4B-IT they are mostly different directions when the model acts, they partly merge when it describes itself, and steering one leaves the others alone.</p>
<div class="meta"><span><b>Model</b> {MODEL_NAME}</span><span><b>Method</b> diff-of-means directions, cross-frame transfer, standardized cosine, steering</span><span><b>Code and results</b> <a href="https://github.com/harsh-kumar9/character-map">github.com/harsh-kumar9/character-map</a></span></div>
{banner}<p class="note"><b>Epistemic status.</b> Exploratory. One model, linear readouts, one steering dose. The geometry is stable across the confound controls we ran; the behavioral part is a null at a coherence-limited dose and a dose-response run is pending. The sincerity direction is weak in this model (see Result 3), and that is the one most of the literature means by honesty.</p>

<div class="box accent"><h3>TL;DR</h3>
<ul>
<li><b>Six definitions, mostly separate objects.</b> When Gemma acts, the standardized cosines between the six definitions' directions run from {min(v for *_, v in def_pairs(LM)):+.2f} to {max(v for *_, v in def_pairs(LM)):+.2f} at layer {LM} (shuffle floor {floor17:.2f}); beyond the floor: {fmt_pairs(above(LM))}.</li>
<li><b>One self-concept.</b> When Gemma describes itself, sincerity and honesty-humility share a direction (cosine {cosv(LM,U['si_d'],U['hh_d']):+.2f}), and the formality control does not join them. Self-report collapses what conduct keeps apart.</li>
<li><b>The gaps between frames are the classic failure modes, and most are open.</b> Judge-to-act (akrasia) is open for sincerity ({T('C1','all','b-c',LM):.2f} at layer {LM}, control {base[LM]:.2f}); self-describe-to-act (hypocrisy) is open for accuracy and sincerity, and closed only for honesty-humility ({T('C19','honesty_humility','c-d',LM):.2f}), the one definition where neither frame is conduct.</li>
{CAUSAL_BULLET}
{BEH_BULLET}
<li><b>Untested consequence.</b> "Train for honesty" is underspecified. The geometry predicts that training on one definition moves that definition and the self-concept, and leaves the others where they were. D-044 proposes the test.</li>
</ul></div>

<h2>The question</h2>
<p>A model spec says the assistant should be honest. What does the model have to have inside it for that to be true? The honesty survey by Li et al. (2024) defines honesty as self-knowledge plus self-expression: know what you know, and say it faithfully. Williams (2002) splits the virtue of truth into sincerity (assert what you believe) and accuracy (take care to believe truly). Personality psychology measures honesty-humility as a factor of HEXACO, with a questionnaire. Each definition comes with its own benchmark, and the benchmarks are used interchangeably as evidence that a model "is honest".</p>
<p>We ask whether these definitions pick out one thing inside the model. If they do, any one benchmark is evidence about the others, and training on one should move the rest. If they do not, "honesty" names a family of separate capacities, and an evaluation or a training run reaches only the member it was written for.</p>
<p>There is a second axis. A person called honest notices lies, judges them wrong, tells the truth, and would call themselves honest. Hartshorne and May (1928) found that these parts come apart in children. Our earlier note found that in Gemma, directions for knowing a trait (recognizing it, judging it) rarely carry over to doing it (acting, self-describing). Here we cross that axis with the definitions.</p>

<h2>Six honesties</h2>
<p>We chose the definitions that Claude's constitution and the OpenAI Model Spec name as rules or properties and that an existing benchmark evaluates (D-049): epistemic (self-knowledge, accuracy), assertion (sincerity, forthrightness), social (non-sycophancy) and trait (honesty-humility). Each row is one concept in the map, with a matched positive and negative version of the same scenario in every frame the concept has.</p>
<div class="tbl"><table><thead><tr><th>Definition</th><th>Positive pole</th><th>Where it comes from</th><th>Concept</th><th>Frames</th><th>Held-out benchmark</th><th>Behavior instrument</th></tr></thead><tbody>
<tr><td>Self-knowledge</td><td>takes itself to know the answer vs not</td><td>Li et al. 2024 (self-knowledge); SelfAware</td><td>A9</td><td>act, self-describe</td><td>KUQ</td><td>SelfAware abstention</td></tr>
<tr><td>Accuracy</td><td>gives what it can support vs fabricates with the same confidence</td><td>Williams 2002; Frankfurt 1986; Li et al. (self-expression)</td><td>C2</td><td>act, self-describe</td><td>TruthfulQA</td><td>SelfAware, exact</td></tr>
<tr><td>Sincerity</td><td>asserts what it believes vs misleads to gain something</td><td>Williams 2002; Bok 1978; MASK, Apollo</td><td>C1</td><td>recognize, judge, act, self-describe</td><td>MASK provided_facts</td><td>MASK known_facts under pressure</td></tr>
<tr><td>Forthrightness</td><td>volunteers the material fact the person would want vs answers truthfully and leaves it out</td><td>Grice 1975; constitution "forthright"; Model Spec lying by omission</td><td>C3</td><td>recognize, act, self-describe</td><td>DailyDilemmas (judge frame, not scored)</td><td>material-fact disclosure judge</td></tr>
<tr><td>Non-sycophancy</td><td>disagrees with a user who is wrong vs agrees to please</td><td>Sharma et al. 2023; Vennemeyer et al. 2025; Model Spec "don't be sycophantic"</td><td>C8</td><td>recognize, act, self-describe</td><td>SycophancyEval answer</td><td>SycophancyEval accuracy drop</td></tr>
<tr><td>Honesty-humility</td><td>high vs low on the HEXACO factor</td><td>Ashton and Lee 2007</td><td>C19</td><td>act, self-describe</td><td>HEXACO-PI-R</td><td>TRAIT two-option choice</td></tr>
</tbody></table></div>
<p class="note">Six more concepts in the map are honesty under some description: truthful self-presentation, integrity, courage under pushback, being observed, pushback itself, and honesty against kindness. They are in the appendix with the same numbers, and the reading below does not change with them included.</p>

<h3>How this differs from the honesty survey</h3>
<div class="tbl"><table><thead><tr><th></th><th>Li et al. 2024, <a href="https://github.com/SihengLi99/LLM-Honesty-Survey">LLM-Honesty-Survey</a></th><th>This note</th></tr></thead><tbody>
<tr><td>Definition</td><td>Self-knowledge plus self-expression, as one trait</td><td>Those two, plus sincerity, forthrightness, non-sycophancy and honesty-humility, from philosophy, personality psychology and the two model specifications, as candidates that may or may not be one thing</td></tr>
<tr><td>What is measured</td><td>Outputs: known/unknown recognition, calibration, selective prediction, consistency</td><td>Internal directions: do the definitions share structure, and does steering one move another</td></tr>
<tr><td>Frames</td><td>The model acting (and consistency across prompts)</td><td>Recognizing, judging, acting, self-describing, for each definition</td></tr>
<tr><td>Improvement</td><td>A catalogue of training-free and training-based methods</td><td>Which definition a training method reaches, read internally (proposed, D-044)</td></tr>
</tbody></table></div>

<h2>Setup, briefly</h2>
<p>Each scenario appears in every applicable frame with a + and a − version. In <i>recognize</i> and <i>judge</i>, the user describes or asks about someone else's act and we read Gemma's own reply. In <i>act</i> and <i>self-describe</i>, the reply is written for Gemma and we read it. The direction for a concept and frame is the mean hidden state of + items minus − items on training scenarios, read over the first 32 reply tokens at decoder layers {LM} and {LL}. Within-frame AUROC scores the direction on held-out scenarios of its own frame; transfer AUROC scores it on another frame's. Cosines are taken after standardizing each hidden dimension, because a few outlier dimensions otherwise dominate every direction. Recognize and judge directions have Gemma's verdict direction projected out. The baseline for transfer is what a formality contrast and a topic contrast carry over ({base[LM]:.2f} at layer {LM}, {base[LL]:.2f} at layer {LL}). Steering adds a direction at layer {L_STEER} at one global dose chosen to keep replies coherent, and the effect is read at layers 22 and 29 as a z-score against eight norm-matched random directions. The earlier note on the full map has the details and the confound checks.</p>
<p><b>On grouping the frames.</b> The earlier note pooled recognize with judge as "knowing" and act with self-describe as "doing", because transfer formed those two blocks across the 59 concepts. That grouping is empirical, and it coincides with a format boundary: recognize and judge are read from Gemma's own reply about someone else, act and self-describe from a reply written in Gemma's voice. Two controls argued that format alone does not make the blocks, but the label hides a distinction this note needs. Self-description is not conduct. It is the frame an evaluation reads when it asks the model about itself, and the gap between it and acting is its own failure mode. So this note keeps the four frames apart, marks the format boundary in Figure 1 for what it is, and names each gap below.</p>

<h2>Result 1: the definitions are different objects, except in self-description</h2>
<figure><img src="data:image/png;base64,{img}" alt="Lattice of six honesty definitions by four frames inside Gemma">
<figcaption><b>Figure 1. Six honesty definitions across four frames inside Gemma.</b> Columns are definitions, rows are frames. The dotted line is the format boundary: above it, directions are read from Gemma's own reply about someone else; below it, from a reply written in Gemma's voice. A node is one direction; the number inside is its within-frame AUROC. (a, b) Edges join directions whose standardized cosine is above the shuffle floor ({floor17:.2f}); width is the cosine, red dashed is a negative cosine beyond the floor. (c) Arrows are steering effects with z at least 3 when the direction is added at layer {L_STEER} and read at layer {LL}; the number inside a node is the direction's effect on its own readout.</figcaption></figure>
<p><b>Across definitions, in acting.</b> Of the {len(def_pairs(LM))} pairs of definitions, {len(above(LM))} at layer {LM} and {len(above(LL))} at layer {LL} have an acting cosine beyond the shuffle floor (layer {LM}: {fmt_pairs(above(LM))}; layer {LL}: {fmt_pairs(above(LL))}). Self-knowledge points slightly away from sincerity in action ({cosv(LM,U['sk_c'],U['si_c']):+.2f} at layer {LM}, {cosv(LL,U['sk_c'],U['si_c']):+.2f} at layer {LL}). {"Williams's split between sincerity and accuracy holds." if abs(cosv(LM,U['si_c'],U['ac_c'])) < floor17 else ""} {"The HEXACO factor shares nothing with sincerity or accuracy." if max(abs(cosv(LM,U['hh_c'],U['si_c'])), abs(cosv(LM,U['hh_c'],U['ac_c']))) < floor17 else ""}</p>
<div class="tbl">{cos_table()}</div>
<p><b>Across definitions, in self-description.</b> Sincerity and honesty-humility merge when Gemma describes itself ({cosv(LM,U['si_d'],U['hh_d']):+.2f} at layer {LM}, {cosv(LL,U['si_d'],U['hh_d']):+.2f} at layer {LL}), and sincerity self-description also sits near honesty-humility in action ({cosv(LM,U['si_d'],U['hh_c']):+.2f}). The formality control's self-description direction is at {cosv(LM,U['k_d'],U['si_d']):+.2f} and {cosv(LM,U['k_d'],U['hh_d']):+.2f} from those two, so this is not a style of self-description, it is an "I am honest" concept. Accuracy and self-knowledge stay out of it. In the wider block, integrity, courage and sincerity self-descriptions join the same cluster (appendix).</p>
<p><b>Within a definition, across frames.</b> Honesty-humility is the one integrated definition: the same direction reads acting and self-description at both depths. That is what a questionnaire construct should look like, since both frames are statements about the kind of agent one is. Self-knowledge joins it at layer {LL} ({T('A9','all','c-d',LL):.2f}) after sitting below the control at layer {LM} ({T('A9','all','c-d',LM):.2f}). Accuracy never does: describing oneself as careful and being careful are separate directions at every depth ({T('C2','all','c-d',LM):.2f}, {T('C2','all','c-d',LL):.2f}). Sincerity's knowing frames carry over to its doing frames at or below the control, and the gap widens with depth (judge–act {T('C1','all','b-c',LM):.2f} at layer {LM}, {T('C1','all','b-c',LL):.2f} at layer {LL}).</p>

{land_section()}

<h2>Result 2: the gaps between frames are the classic failure modes</h2>
<p>Philosophy already has names for an agent whose frames come apart, and each name is one off-diagonal cell of Figure 1. The table lists them, what each would be for honesty, which behavioral instrument in the map measures that gap as conduct, and whether the gap is open in Gemma at the level of representation. "Open" here means the two frames' directions do not carry over beyond the control; it is the precondition for the failure mode, not the failure mode itself, which is a behavioral fact.</p>
<div class="tbl"><table><thead><tr><th>Frame pair</th><th>Failure mode when they come apart</th><th>For honesty</th><th>Instrument that measures it as conduct</th><th>Representational status in Gemma</th></tr></thead><tbody>
<tr><td>judge → act</td><td>Akrasia, weakness of will (Aristotle NE VII; Davidson 1970)</td><td>Judges lying wrong, lies under pressure</td><td>MASK known_facts (belief vs assertion under a pressure prompt); are-you-sure and SYCON pushback (C11, E5)</td><td>Sincerity: open ({T('C1','all','b-c',LM):.2f} at layer {LM}, {T('C1','all','b-c',LL):.2f} at layer {LL}, control {base[LM]:.2f}/{base[LL]:.2f}). The other three have no judge frame.</td></tr>
<tr><td>self-describe → act</td><td>Hypocrisy, the say–do gap (Batson et al. 1997; Williams 1973)</td><td>Calls itself honest, acts otherwise</td><td>WDCT word–deed consistency and the Batson private coin flip (C12)</td><td>Accuracy: open ({T('C2','all','c-d',LM):.2f}, {T('C2','all','c-d',LL):.2f}). Sincerity: open ({T('C1','all','c-d',LM):.2f}, {T('C1','all','c-d',LL):.2f}). Self-knowledge: open at layer {LM}, closed at 29 ({T('A9','all','c-d',LM):.2f}, {T('A9','all','c-d',LL):.2f}). Honesty-humility: closed ({T('C19','honesty_humility','c-d',LM):.2f}, {T('C19','honesty_humility','c-d',LL):.2f}).</td></tr>
<tr><td>recognize → judge</td><td>Seeing the vice without condemning it (moral indifference)</td><td>Spots a lie, does not call it wrong</td><td>No instrument in the map</td><td>Sincerity: at or below control ({T('C1','all','a-b',LM):.2f}, {T('C1','all','a-b',LL):.2f}); the two directions share a cosine of {cosv(LM,U['si_a'],U['si_b']):+.2f}, so the verdict control is what pulls transfer down.</td></tr>
<tr><td>recognize → act</td><td>Knowing the vice in others without having the virtue (NE II.4: knowing what is right is not being virtuous)</td><td>Detects deception, deceives</td><td>No instrument in the map</td><td>Sincerity: open ({T('C1','all','a-c',LM):.2f}, {T('C1','all','a-c',LL):.2f}).</td></tr>
<tr><td>situation → act</td><td>Situational honesty: Gyges (Plato), Hartshorne and May 1928, Asch 1951</td><td>Honest when watched or unopposed, not otherwise</td><td>Alignment-faking compliance gap (E1); are-you-sure answer kept (E5)</td><td>Pushback is sharply encoded in the act frame (E5 within-frame {fid['E5|all'][LMs]['within']['c']:.2f}, held-out {fid['E5|all'][LMs]['heldout']['c']:.2f}); being observed is not (E1 act {fid['E1|all'][LMs]['within']['c']:.2f}).</td></tr>
</tbody></table></div>
{R2_PARA}
<p>The claim table for the six definitions, under the paper's criterion (shared: beats the control by 0.10 under three readouts; split: below 0.40 under all three):</p>
<div class="tbl">{tr_html}</div>

<h2>Result 3: does each direction exist?{" Sincerity barely does" if IS_DEFAULT else ""}</h2>
<div class="tbl">{fid_table(FOUR)}</div>
{R3_PARA}

{R4_BLOCK}
{R5_BLOCK}
<h2>What this would mean, if it holds</h2>
<p>Everything in this section is a prediction from the geometry, not a result.</p>
<ul>
<li><b>Evaluation.</b> MASK and TruthfulQA measure directions that are orthogonal in this model, so a model can pass one and fail the other with no contradiction. A self-report of honesty reads the merged self-concept and will say "honest" whichever conduct direction is weak.</li>
<li><b>Training.</b> "Train for honesty" is underspecified. Fine-tuning on honest self-descriptions (persona or constitution style) should move the self-concept cluster and honesty-humility and leave accuracy and sincerity conduct where they were. Fine-tuning on sincere replies under pressure should move the sincerity act direction and its self-description, and not accuracy or self-knowledge. A mixed corpus should move each by a third.</li>
<li><b>Gaps as targets.</b> A training method can be scored by which gap it closes: does it move the act direction toward the judge direction (closing akrasia) or only the self-description (widening hypocrisy)? The gap instruments (MASK, WDCT, are-you-sure, the alignment-faking compliance gap) are the behavioral side of the same question.</li>
<li><b>The test.</b> D-044: four LoRA runs on Gemma-3-4B-IT with equal budgets, one per definition and one mixed, then read the whole block in every frame plus the four behaviors. Reuses the persona-collapse SFT code.</li>
</ul>

<h2>Open items</h2>
<ol>
{OI_SINCERITY}
{OI_DOSE}
{OI_ONEMODEL}
<li><b>Being observed is barely encoded</b> in the act frame (within-frame {fid['E1|all'][LMs]['within']['c']:.2f} at layer {LM}, {fid['E1|all'][LLs]['within']['c']:.2f} at layer {LL}), so the Hartshorne–May question has no handle in this model. Report as a null.</li>
{"" if LAND else '<li><b>Item-level map.</b> Figure 1 places directions. A second figure should place items: every + and − item of the six definitions in every frame, projected into the subspace spanned by the block directions. This needs the stored activations on ada (a CPU job), not the Mac.</li>'}
{OI_GAPS}
<li><b>The appendix concepts</b> contain two entanglements worth a sentence in the paper: "I know the answer" shares a direction with "not self-deprecating" in self-description (cosine {D['cos']['self_L'+LMs]['cos'][D['cos']['self_L'+LMs]['units'].index('A9|all|d')][D['cos']['self_L'+LMs]['units'].index('C4|self-deprecating|d')]:+.2f}), and sincerity in action shares with candor ({D['cos']['act_L'+LMs]['cos'][D['cos']['act_L'+LMs]['units'].index('C1|all|c')][D['cos']['act_L'+LMs]['units'].index('C3|all|c')]:+.2f}).</li>
</ol>

<h2>Proposed decisions</h2>
<dl>
<dt>D-042</dt><dd>Reframe the research question around definitions: is a spec-level trait one thing inside the model, and which definition do evaluations and training reach. The knowing–doing split is the frame axis.</dd>
<dt>D-043</dt><dd>The four-definition honesty block above is the paper's spine; the other seven honesty concepts and the rest of the 59 are replication and appendix. Position against Li et al. as in the table.</dd>
<dt>D-044</dt><dd>S7, training under each definition: four equal-budget LoRA runs, read the block and the four behaviors after each.</dd>
<dt>D-045</dt><dd>Fix sincerity (C1) before any of the above.</dd>
<dt>D-049</dt><dd>Six definitions in the core, each named as a rule or property by a model specification: calibration, accuracy, sincerity (rebuilt, with a misleading sub-contrast), forthrightness, non-sycophancy, honesty-humility; self-transparency and manipulation as new appendix concepts.</dd>
<dt>D-046</dt><dd>Stop pooling act with self-describe in the honesty story. Name each frame gap by its failure mode and pair it with its gap instrument; rebuild the sincerity act frame around holding a belief under pressure, so that the direction and MASK measure the same gap.</dd>
</dl>
<p class="note">Full text of each proposal is in <code>docs/DECISIONS.md</code>.</p>

<h2>Appendix A: the other six honesty concepts</h2>
<p class="note">Same measures, layer {LM} unless stated. "Nearest of the four" is the largest absolute standardized cosine between this concept's direction and the four main definitions in the same frame.</p>
<div class="tbl">{app_table()}</div>
<p class="note">Reading: candor shares with sincerity in action; the self-description cluster takes in integrity, courage and sincerity (integrity–honesty-humility {D['cos']['self_L'+LMs]['cos'][D['cos']['self_L'+LMs]['units'].index('C12|integrity|d')][D['cos']['self_L'+LMs]['units'].index('C19|honesty_humility|d')]:+.2f}, sincerity–integrity {D['cos']['self_L'+LMs]['cos'][D['cos']['self_L'+LMs]['units'].index('C1|all|d')][D['cos']['self_L'+LMs]['units'].index('C12|integrity|d')]:+.2f}); the white-lie choice uses neither the sincerity nor the honesty-humility direction, and judging a white lie does not carry over to choosing one ({T('F1','all','b-c',LM):.2f}).</p>

<h2>Appendix B: how to read the numbers</h2>
<dl>
<dt>AUROC</dt><dd>0.5 means the direction cannot tell + from −, 1.0 is perfect, below 0.5 points the wrong way. Within-frame uses the direction's own frame; transfer uses another frame's items and averages both ways.</dd>
<dt>Control baseline</dt><dd>What a formality contrast and a topic contrast carry over between frames: {base[LM]:.2f} at layer {LM}, {base[LL]:.2f} at layer {LL}. Shared means 0.10 above it under three readouts (reply as is, token before the reply, reply with Gemma's verdict removed).</dd>
<dt>Cosine</dt><dd>Between two directions after standardizing each hidden dimension. The shuffle floor is the 95th percentile of cosines between shuffled-label directions, {floor17:.2f} at layer {LM} and {floor29:.2f} at layer {LL}.</dd>
<dt>z</dt><dd>Steering effect on a target direction, in units of that direction's + minus − gap, minus the skip-path prediction, divided by the spread over eight norm-matched random directions.</dd>
<dt>Behavior effect</dt><dd>Paired item-level difference from the unsteered reply, with its standard error, against four random directions at the same dose.</dd>
</dl>
</article>
"""
open(OUT / "honesty.html", "w").write(page)
print("written", len(page) // 1024, "KB")
