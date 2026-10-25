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
BEH_GROUP = {"C19": ["agreeableness"], "C20": ["psychopathy"], "C24": ["type:ifeval", "type:judge"]}
GROUP_NAME = {"type:ifeval": "IFEval", "type:judge": "overstepping judge"}

STATUS = ("Exploratory. One model, synthetic scenarios written and checked by LLMs, one build per definition. The "
          "numbers are for spotting signals worth a test, not claims. Kept apart from the honesty investigation.")
READING = {
    "helpfulness": {
        "date": "2026-10-24",
        "stand": "Seven definitions of helpfulness, from the constitution, the Model Spec and personality psychology, each "
                 "in four frames, read inside Gemma-3-4B. Do they pick out one thing, and does knowing each one carry over "
                 "to doing it?",
        "status": STATUS,
        "tldr": [
            "<b>Frame first, again.</b> Items about someone else's reply sit about 11 within-unit spreads from items in "
            "Gemma's voice at layer 17. Unlike harmlessness, acting and self-describing also come apart (2.2 spreads at "
            "17, 2.4 at 29).",
            "<b>One cluster when Gemma watches, none when it acts.</b> In the judge frame care, autonomy, wellbeing and "
            "agreeableness share a direction (cosines +0.31 to +0.44, floor 0.16) and substance runs against wellbeing "
            "(-0.33). In the act frame no two definitions pass 0.21.",
            "<b>Knowing rarely becomes doing.</b> Judge-to-act transfer at layer 17 is 0.21 for autonomy, 0.47 for "
            "wellbeing and 0.54 to 0.59 for the task definitions; only care (0.75) and agreeableness (0.72) get "
            "anywhere. None of the three new held-out sets transfers.",
            "<b>Care and agreeableness in action are hubs.</b> Steered, they move most of the block's other readouts and "
            "cross into harmlessness, where agreeableness and wellbeing lower refusing a harmful request. Substance does "
            "not steer even its own readout.",
            "BEH",
        ],
        "question": "<p>The constitution and the Model Spec build helpfulness from several parts: read the request as it "
                    "was meant, neither too literally nor too liberally; give real, specific help instead of hedged help; "
                    "and care about the person, which includes their wellbeing and their right to decide for "
                    "themselves. Personality psychology adds agreeableness as the trait behind it. Each part has its own "
                    "benchmark: IFEval for following instructions, HelpSteer2 for how helpful a reply is, INTIMA for "
                    "companionship.</p><p>We ask the honesty page's two questions of this block. Do these definitions "
                    "pick out one thing inside Gemma, or several? And does knowing each one, recognizing it in another "
                    "assistant's reply or judging that reply, carry over to doing it, writing the reply or describing "
                    "itself?</p>",
        "defs_intro": "Seven units from five concepts, in three groups: the task (reads the request, as two "
                      "sub-contrasts, and substantive help), the person (care, autonomy, wellbeing) and the trait "
                      "(agreeableness). Every unit has all four frames. The grounding is in the concept cards (D-053, "
                      "D-056); the review links open each concept's item review.",
        "r1": "<p>As for honesty, frame comes before definition. At layer 17 the first principal component of the "
              "block's subspace carries 81% of its variance and splits items about someone else's reply (recognize, "
              "judge) from items in Gemma's voice (act, self-describe), 10.6 to 11.0 within-unit spreads apart. At 29 "
              "the split is 3.2 to 3.8 spreads and the first component carries 37%. What is new is the second split: "
              "acting and self-describing sit 2.2 spreads apart at 17 and 2.4 at 29, while recognize and judge sit 1.0 "
              "and 0.7 apart. In the harmlessness block acting and self-describing nearly coincide (0.6, 0.8).</p>"
              "<p>Inside a frame group the task definitions sit apart from the person ones (mean 1.8 to 2.2 spreads "
              "when Gemma watches, 1.0 to 1.3 when it acts or describes itself). The person definitions sit close to "
              "each other and to agreeableness when Gemma watches (0.3 to 1.0).</p>",
        "r2": "<p>Recognize to judge transfers for every definition (0.77 to 1.00 at both layers). Act to self-describe "
              "transfers for care, autonomy, agreeableness and wellbeing (0.85 to 0.99; the formality control 0.86 and "
              "0.85) and less for the task definitions: not too literal 0.70 and 0.77, no overstepping 0.78 and 0.76, "
              "substance 0.87 and 0.84.</p><p>Knowing to doing is where the block comes apart. Judge to act at layer "
              "17: autonomy 0.21 (the judge direction reads the acts backwards), wellbeing 0.47, substance 0.54, the "
              "two C24 units 0.56 and 0.59, agreeableness 0.72, care 0.75. Recognize to act is about the same, except "
              "for autonomy (0.75 at 17, 0.93 at 29). None of the three new held-out sets transfers: IFEval pass vs "
              "fail sits at 0.49 to 0.53, HelpSteer2 pairs at 0.50 to 0.53, and INTIMA is reversed (0.40 to 0.42), "
              "not read yet.</p>",
        "r3": "<p>The lattice has more edges when Gemma watches than when it acts. In the judge frame at layer 17 "
              "(floor 0.16) autonomy goes with agreeableness (+0.44), wellbeing (+0.42) and care (+0.39), and "
              "wellbeing and care go with agreeableness (+0.32, +0.31). Substance runs against wellbeing in recognize "
              "and judge (-0.31, -0.33): a reply judged substantive is judged less supportive of the person's own "
              "capacities and ties. In the act frame no two definitions pass 0.21 (overstepping with autonomy), and in "
              "self-description the largest is not too literal with agreeableness (+0.28).</p><p>So the person side "
              "of helpfulness is one cluster when Gemma evaluates someone else's reply and separate directions in its "
              "own. The task side never clusters.</p>",
        "r4": "<p>Helpfulness steers weakly on its own readouts: the median self effect is z 1.9 at layer 29, 6 of 13 "
              "sources reach 3, and substance moves nothing (0.9). What does steer, steers widely. Care in action and "
              "agreeableness in action (z 5.8 and 15.9 on themselves at 29) carry 24 of the 40 off-definition cells "
              "at |z| 3 or more at layer 22 and 13 of the 18 at 29: they raise autonomy, substance and each other. Of "
              "the 10 negative cells at 22, 7 fall on overstepping when Gemma watches. Care, agreeableness, wellbeing "
              "and not-too-literal in action move the judged reply toward overstepping (-4.3 to -5.9): warmth added "
              "to a reply reads as doing more than was asked. The other 3 fall on wellbeing in action (-3.1 to "
              "-3.9): care and agreeableness move Gemma's own reply toward fostering reliance.</p><p>Across blocks "
              "helpfulness "
              "pushes more than it receives: 34 and 35 cells on harmlessness readouts at 22 and 29, against 10 and 6 "
              "the other way. Agreeableness and wellbeing in action lower refusing a harmful request in "
              "self-description (-6.1, -6.0 at 22), and agreeableness raises helping a harmless one (+5.9). Care is "
              "mixed: it raises refusing in self-description (+4.7), and care judged pushes the judged act toward "
              "harmful (-8.6 at 29), so care makes harm more visible.</p>",
        "r5": "R5",
        "cross": "<p>The HHH page puts the three blocks side by side. Two things from this block go there. The tension "
                 "between helpfulness and harmlessness runs through autonomy, not substance: refusing a harmful "
                 "request is against autonomy in recognize, judge and act (-0.23 to -0.29 at layer 17) and helping a "
                 "harmless one goes with it (+0.23 to +0.62), while substance and helping a harmless request are -0.09 "
                 "to +0.13 in every frame. And the steering that couples the blocks starts on the person side: "
                 "agreeableness, wellbeing and care in action are the directions that move refusal.</p>",
        "open": [
            "C26's INTIMA held-out is reversed (0.40 to 0.42); the labels or the pairing may run the other way from "
            "the card's good side.",
            "C25 does not move its own readout at the calibrated dose, so whether substance is causally part of the "
            "tension is untested.",
            "Several behavior instruments sit at floor or ceiling unsteered: no reply rated apathetic (C5, 200 of "
            "200 at 0), almost none paternalistic (C6, 194 of 200 at 0), almost none fully helpful (C25, 4 of 200), "
            "and 98 of 100 edits judged to change more than the errors (C24).",
            "C28 (asks only when a request is truly ambiguous) is being built and joins the block when done.",
            "The large subjects (gemma-4-31B-it, Qwen3.5-27B).",
        ],
    },
    "harmlessness": {
        "date": "2026-10-24",
        "stand": "Eight definitions of harmlessness, from the constitution, the Model Spec, moral psychology and the "
                 "agent-safety literature, each in four frames, read inside Gemma-3-4B. Do they pick out one thing, "
                 "and does knowing each one carry over to doing it?",
        "status": STATUS,
        "tldr": [
            "<b>Frame first, and acting and self-describing coincide.</b> Items about someone else's reply sit about 8 "
            "within-unit spreads from those in Gemma's voice at layer 17; acting and self-describing are 0.6 apart "
            "(helpfulness 2.2).",
            "<b>Refusing harmful requests and helping harmless ones pull against each other.</b> Their directions are "
            "opposed in every frame (-0.22 to -0.48), and refusing does not carry over from knowing to doing: the "
            "judge and recognize directions read Gemma's own refusals at 0.07 to 0.61.",
            "<b>Refusing in text and declining a tool call share a direction</b> (+0.30 act, +0.48 self-describe), "
            "and declining carries over from knowing to doing (0.86 to 0.99) where refusing does not.",
            "<b>Seeing harm and refusing are separate.</b> Harm in an act (B1) and refusing a harmful request are "
            "under the floor in 7 of 8 cells.",
            "<b>Steering splits the block in two.</b> Directions for reading the situation as fine (legitimate intent, "
            "harmless act, low psychopathy, charitable reading) raise each other; refusing and declining raise each "
            "other; between the groups 19 of 21 cells at layer 29 are negative (down to z -9).",
            "BEH",
        ],
        "question": "<p>Harmlessness has more sources than helpfulness and they disagree on what it is. The early "
                    "alignment work and the refusal benchmarks make it refusing harmful requests (HarmBench, Arditi et "
                    "al. 2024). The constitution and the Model Spec add the other side: help with harmless requests, "
                    "assume good intent, read borderline requests charitably (XSTest, OR-Bench). Moral psychology puts "
                    "harm in the perception of an act (Schein and Gray 2018), and the agent-safety work moves it from "
                    "words to actions: declining a harmful tool call and avoiding side effects nobody asked for "
                    "(AgentHarm, ToolEmu, Mind the GAP). Psychopathy is the trait at the far end.</p><p>We ask the "
                    "honesty page's two questions of this block. Do these definitions pick out one thing inside Gemma, "
                    "or several? And does knowing each one, recognizing it in another assistant's reply or judging "
                    "that reply, carry over to doing it, writing the reply or describing itself?</p>",
        "defs_intro": "Eight units from six concepts, in three groups: perceiving (legitimate intent, harm in an act), "
                      "acting (refusing a harmful request, helping a harmless one, charitable reading, declining a "
                      "harmful tool call, avoiding side effects) and the trait (psychopathy). C15 is split by request "
                      "type, so refusing and helping are separate units. Every unit has all four frames. The grounding "
                      "is in the concept cards (D-054, D-057); the review links open each concept's item review.",
        "r1": "<p>Frame first here too. At layer 17 the first principal component carries 73% of the block's subspace "
              "and puts recognize and judge 7.9 to 8.2 within-unit spreads from act and self-describe (3.2 to 3.8 at "
              "29, first component 41%). Unlike helpfulness, acting and self-describing nearly coincide: 0.6 spreads "
              "at 17 and 0.8 at 29, as close as recognize and judge (0.6, 0.5). In this block Gemma describes itself "
              "the way it acts.</p><p>Inside a frame the act definitions stay apart from each other (mean 1.2 to 1.9 "
              "spreads) and the two perceiving ones are closest in self-description (0.2 at 17).</p>",
        "r2": "<p>Recognize to judge transfers for every definition (0.95 to 1.00; side effects 0.74 and 0.76). Act to "
              "self-describe does too (0.82 to 1.00; side effects 0.78 and 0.72; formality control 0.87 and 0.85).</p>"
              "<p>Knowing to doing splits the block. Declining a harmful tool call (0.86 to 0.99), legitimate intent "
              "(0.90 to 0.97) and helping a harmless request (0.75 to 0.95) carry over from recognize and judge to "
              "act. Refusing a harmful request does not: 0.34 to 0.61 to act and 0.07 to 0.36 to self-description. "
              "The C15 2x2 suggests a reason. Read where Gemma takes in the quoted exchange, before its reply, the "
              "judge frame codes fit (refusing a harmful request or helping a harmless one, against the reverse) more "
              "than refusal. Seeing an act as harmful (0.46 to 0.63) and side effects (0.63 to 0.72) carry over "
              "weakly. Held-out: R-Judge's injection cases read side effects in recognize and judge (0.58 to 0.88), "
              "the GAP far set reads declining (0.93, 1.00), and R-Judge's unintended-action cases do not (0.43 to "
              "0.69).</p>",
        "r3": "<p>At layer 17 (floor 0.16) the block has more pairs above the floor than helpfulness (149 against "
              "76, with one more unit), and unlike helpfulness about as many between definitions in Gemma's own voice "
              "as when it watches (28 in act and self-describe, against 7 for helpfulness). Refusing in text and declining a tool call share a direction (+0.30 act, "
              "+0.48 self-describe). Declining a harmful action opposes helping a harmless request (-0.36 act, -0.65 "
              "self-describe), and refusing a harmful request opposes it in every frame (-0.22 to -0.48 at 17 and "
              "29). Legitimate intent goes "
              "with charitable reading (+0.48 act, +0.57 self-describe), with helping a harmless request (+0.35 act) "
              "and against refusing (-0.34 act). Low psychopathy goes with harmless acts (+0.40 act, +0.54 "
              "self-describe). Side effects connect to little.</p><p>Seeing harm and refusing are separate: harm in an "
              "act against refusing a harmful request is -0.03 to +0.18, under the floor in 7 of 8 cells, as Zhao et "
              "al. (2025) report for harmfulness and refusal.</p>",
        "r4": "<p>Harmlessness steers itself well: the median self effect is z 3.7 at layer 29 and 10 of 15 sources "
              "reach 3, the judge directions most (harm in an act 18.4, refusing 10.6, psychopathy 10.0, helping 9.3). "
              "Off-definition cells at |z| 3 or more: 19 at layer 22, 44 at 29.</p><p>At 29 they form two groups. "
              "Reading the situation as fine (legitimate intent judged, a harmless act judged, low psychopathy judged, "
              "charitable reading recognized) raise each other (+4.5 to +10.9). Refusing and declining raise each "
              "other (refusing in action raises declining in self-description +10.7). Inside the groups 21 of 23 "
              "cells are positive; between them 19 of 21 are negative. Helping a harmless request in action lowers "
              "declining in self-description (-9.0) and "
              "refusing in action (-6.5); judging intent legitimate lowers declining (-6.0); refusing in action lowers "
              "judging the act harmless (-6.0). Causally, harmlessness is two things that pull against each "
              "other.</p><p>Harmlessness moves helpfulness little (10 and 6 cells at 22 and 29). Helping a harmless "
              "request in action raises agreeableness and autonomy (+7.3, +6.8 at 22), and refusing a harmful one, "
              "judged, lowers agreeableness (-4.7).</p>",
        "r5": "R5",
        "cross": "<p>The HHH page puts the three blocks side by side. Three things from this block go there. Refusing "
                 "is read as overriding the person's choice, not as giving less: refusing a harmful request is "
                 "against autonomy in recognize, judge and act (-0.23 to -0.29 at layer 17), while helping a harmless "
                 "request and substance are -0.09 to +0.13. A refusal that offers safe help (C15 safe completion) "
                 "points toward complying in Gemma's own reply (cosine with refusing -0.78, -0.85) and not toward "
                 "substance (-0.16). And the directions that lower refusal causally come from helpfulness's person "
                 "side: agreeableness and wellbeing in action.</p>",
        "open": [
            "Whether the harm reading in Gemma's own replies follows its stance once stated (a window at the stance, "
            "not the first 32 tokens, which are mostly a counselling preamble).",
            "Side effects (C27) are weak within frame (0.56 to 0.81) and transfer nowhere; too subtle a contrast or "
            "too varied scenarios.",
            "Behavior instruments at ceiling unsteered: legitimate intent rated 100 on 191 of 200 replies (A1).",
            "The large subjects (gemma-4-31B-it, Qwen3.5-27B).",
        ],
    },
}


def esc(s):
    return html.escape(str(s))


def f2(v, sign=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "&ndash;"
    return f"{v:+.2f}" if sign else f"{v:.2f}"


def img(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def load(block):
    out = R / "figures" / block
    D = json.loads((out / f"{block}_data.json").read_text())
    ft = json.loads((R / "s3" / "frames_table.json").read_text())
    tests = json.loads((R / "tests.json").read_text())
    behs = {}
    for m in DOSES:
        p = R / ("s5b" if m == 1 else f"s5b_dose/x{m}") / "behavior.json"
        if p.exists():
            behs[m] = json.loads(p.read_text())
    return out, D, ft, tests, behs


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
            cells.append(f"<td>{f2(w1)} / {f2(w2)}</td>" if w1 is not None else "<td>&ndash;</td>")
        held = {f: v for f, v in r["heldout"].items() if v is not None}
        held2 = {f: v for f, v in r2["heldout"].items() if v is not None}
        hs = ", ".join(f"{FRAMES[f]} {f2(v)} / {f2(held2.get(f))}" for f, v in held.items()) or "&ndash;"
        cls = ' class="low"' if held and max(held.values()) < 0.6 else ""
        rows.append(f"<tr><td>{esc(key)}</td>{''.join(cells)}<td{cls}>{hs}</td></tr>")
    return ("<table><thead><tr><th>Unit (S3 sub-contrast)</th>" + "".join(f"<th>{FRAMES[f]}</th>" for f in "abcd") +
            f"<th>Held-out</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            f"<p class=\"note\">Within-frame and held-out AUROC at layers {LM} / {LL}, reply32, S3's raw diff-of-means "
            "direction. C15's rows are the whole concept (S3 has no request-type split); the request-type units are in "
            "the transfer figure.</p>")


DOSES = (1, 2, 4, 8)


def beh_eff(cc, cond, grp):
    """(effect, passes) of one condition: paired change from the unsteered replies, on the concept's own behavior
    group where the instrument mixes several; passes = at least twice the largest random-direction effect at that
    dose and at least 3 SE (the S5b criterion)."""
    v = cc.get(cond)
    if not v:
        return None, False
    if grp:
        b = cc["baseline"]["by_group"].get(grp)
        d = lambda c: None if b is None or cc[c]["by_group"].get(grp) is None else cc[c]["by_group"][grp] - b
        e, se = d(cond), v.get("by_group_se", {}).get(grp)
        rnd = [abs(x) for c in cc if c.startswith("random") for x in [d(c)] if x is not None and x == x]
    else:
        e, se = v["effect"], v["se"]
        rnd = [abs(cc[c]["effect"]) for c in cc if c.startswith("random") and cc[c]["effect"] == cc[c]["effect"]]
    if e is None or e != e:
        return None, False
    rmax = max(rnd) if rnd else None
    ok = rmax is not None and se is not None and se == se and abs(e) > 0 and abs(e) >= 2 * rmax and abs(e) >= 3 * se
    return e, ok


def beh_rows(block, behs):
    """Per unit, judge and frame (doing, knowing): unsteered score and the oriented effect of steering toward the
    unit's good side and toward its other side at each dose. The concept's first sub-contrast has doing and knowing at
    both signs at every dose; a further sub-contrast has doing at + and at 1x only, and its sign is flipped when its
    good side is the negative pole."""
    us = BLOCKS[block]
    out = []
    for u in us:
        beh = behs.get(1, {})
        if u.cid not in beh:
            continue
        sub = u.key.split("|")[1]
        sg = 1.0 if GOOD_HIGH[u.cid] else -1.0
        pole = 1.0 if u.good == "pos" else -1.0
        for grp in BEH_GROUP.get(u.cid, [None]):
            for judge in ("gemma", "qwen") if grp != "type:ifeval" else ("gemma",):
                cc1 = beh[u.cid]["conds"].get(judge)
                if not cc1:
                    continue
                extra = f"doing+:{sub}" in cc1
                name = u.name + (f", {GROUP_NAME[grp]}" if grp in GROUP_NAME else "")
                base = cc1["baseline"]["by_group"].get(grp) if grp else cc1["baseline"]["mean"]
                frames = ("doing",) if extra else ("doing", "knowing")
                rows = [{"unit": name, "concept": u.cid, "judge": judge, "baseline": base, "frame": f, "good": {}, "away": {}}
                        for f in frames]
                for m in DOSES:
                    cc = behs.get(m, {}).get(u.cid, {}).get("conds", {}).get(judge)
                    if not cc:
                        continue
                    if extra:
                        e, ok = beh_eff(cc, f"doing+:{sub}", grp)
                        rows[0]["good"][m] = None if e is None else (pole * sg * e, ok)
                        continue
                    for row in rows:
                        for side, c in (("good", "+" if pole > 0 else "-"), ("away", "-" if pole > 0 else "+")):
                            e, ok = beh_eff(cc, row["frame"] + c, grp)
                            row[side][m] = None if e is None else (sg * e, ok)
                out.extend(rows)
    return out


def beh_table(rows):
    if not rows:
        return "<p class=\"note\">Behavior under steering is pending.</p>"
    def cell(x):
        if not x:
            return "<td>&ndash;</td>"
        v, ok = x
        s = f"{v:+.3f}".replace("-0.000", "+0.000")
        return f'<td class="hi">{s}</td>' if ok and v > 0 else f'<td class="low">{s}</td>' if ok else f"<td>{s}</td>"
    body = []
    for i, r in enumerate(rows):
        first = i == 0 or (rows[i - 1]["unit"], rows[i - 1]["judge"]) != (r["unit"], r["judge"])
        n = sum(1 for x in rows[i:] if (x["unit"], x["judge"]) == (r["unit"], r["judge"]))
        lead = (f"<td rowspan=\"{n}\">{esc(r['unit'])} <span class=\"id\">{r['concept']}</span></td><td rowspan=\"{n}\">{r['judge']}</td>"
                f"<td rowspan=\"{n}\">{f2(r['baseline'])}</td>") if first else ""
        body.append(f"<tr>{lead}<td>{r['frame']}</td>" + "".join(cell(r["good"].get(m)) for m in DOSES)
                    + "".join(cell(r["away"].get(m)) for m in DOSES) + "</tr>")
    dose = "".join(f"<th>{m}x</th>" for m in DOSES)
    return ("<table><thead><tr><th rowspan=\"2\">Unit</th><th rowspan=\"2\">Judge</th><th rowspan=\"2\">Unsteered</th><th rowspan=\"2\">Direction</th>"
            f"<th colspan=\"{len(DOSES)}\">Steered toward the good side</th><th colspan=\"{len(DOSES)}\">Steered toward the other side</th></tr>"
            f"<tr>{dose}{dose}</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


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
<div class="eyebrow">character-map &middot; HHH line &middot; exploratory &middot; {rd['date']}</div>
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
<p class="note">Effects are the paired change from the unsteered replies, oriented so that positive is the good side of the instrument; each direction is steered toward the unit's good side and toward its other side, so a working direction reads positive on the left and negative on the right. Doses are multiples of the calibrated dose (1x is 1/128 of the residual norm). Green: passes the criterion (at least twice the largest random-direction effect at that dose and at least three standard errors) in the good direction; red: passes in the other direction. A concept's second sub-contrast (too literal, helping a harmless request, side effects) was steered at 1x only and toward its positive pole; for helping a harmless request the sign is flipped, which assumes a symmetric effect (S5a sign symmetry r 0.95).</p>

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
