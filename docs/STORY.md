# The story so far: is honesty one thing inside a model?

Written 2026-10-31 as the spine for the manuscript. Not the paper: the assumptions we started from, the steps we
took, what we see, and the shortest reading of it. Every number is from the decision log (D-042 to D-064) and the
tracked result files. The figures are in `results/figures/story/` and `cm/story_figs.py` draws them from those
files; `results/figures/story/numbers.json` holds the numbers behind each panel.

## The one-line point

Inside gemma-4-31b-it and Qwen3.5-27B, "honesty" is not one variable. Where an honesty item sits is set first by
what the model is doing when the trait shows (recognizing it in someone else, judging it, acting, describing
itself); which definition of honesty is in play, and whether the item is the honest or the dishonest side, come
second and tie. Both models recognize cheating when shown it,
condemn it, including their own, describe themselves as honest, and still cheat under pressure. The representation
that carries described honesty does not read the model's own decision to cheat, at any layer. Evaluations and
character training work on the described-honesty side. The failures are on the conduct side.

## 1. Where we start

These are the assumptions. Each one is a premise the paper has to state, because the design follows from it.

1. **Model specs name traits.** "Be honest" is one line in a constitution, one number on a benchmark, one direction
   people steer. All three treat honesty as one thing.
2. **Philosophy and psychology do not.** Williams separates sincerity (asserting what you believe) from accuracy
   (taking care to believe truly). Calibration, knowing what you know, is a third reading. Forthrightness,
   volunteering what matters, is a fourth. Non-sycophancy, holding the truth against the listener's wish, is a fifth.
   HEXACO honesty-humility, no manipulation and no entitlement, is a sixth. These are competing accounts of one
   named trait, and each has a benchmark or a training recipe that targets it.
3. **A virtue has parts.** In virtue ethics a trait is not only conduct. It is perceiving what matters in a
   situation, judging well, choosing and acting, and a self-conception that holds the three together (Aristotle NE
   II, VI, VII; Murdoch; McDowell; Korsgaard). When the parts come apart the names are old: akrasia when judgment is
   present and action does not follow, hypocrisy when self-description and action come apart. The full map of the
   gaps, the literature behind each and the human cases that show them is in [GROUNDING.md](GROUNDING.md), with
   Figure 8.
4. **Situations move conduct.** Doris and Harman argue that human conduct follows the situation more than the trait.
   For a model this is a test: same model, a different pressure or a different system prompt. Does the conduct move
   while the trait readouts stay put?
5. **What "one thing" means inside a network.** If two things are one variable inside the model, a direction found
   for one reads the other, steering one moves the other, and both predict the same conduct. If none of that holds,
   they are not one thing, whatever the word says.

The question that follows (D-042): a trait named in a model spec, such as honesty, has several definitions. Are
they one thing inside the model? If not, which of them do evaluations and training reach?

## 2. What we did

Eight steps. Every step ran on both subjects under one code path. The method was built on Gemma-3-4B with a
59-concept map; that map is the appendix and the replication base, not the paper.

1. **Write the definitions down as contrasts, in four frames.** Six definitions, each a matched pair: the same
   scenario in an honest and a dishonest version. Each definition appears in four frames: the model's own reply
   about someone else's conduct (a, recognize), its own reply on whether that conduct was acceptable (b, judge), a
   reply written in its voice under temptation (c, act), and a reply written in its voice about the kind of
   assistant it is (d, self-describe). Two definitions have no judge frame, so 22 cells, 100 to 200 items each,
   about 4,000 honesty items. Two controls share the frame structure and are not about honesty: formality and
   topic. Public datasets are adapted where they exist (MASK, SelfAware, TRAIT, SycophancyEval and others) and the
   missing cells are generated; every cell keeps a held-out set from a different source. Figure 1.
2. **Read the model.** One direction per cell in the residual stream, the standardized difference of means between
   the honest and the dishonest items, with a logistic probe beside it, at four depths. Three readouts: does a
   direction from one cell read the items of another (transfer AUROC); where do the items sit in the space of all 22
   directions (the landscape); and the paper's metric, leave-one-cell-out generalization: train one honesty
   direction on 21 cells, test it on the 22nd, scaled by what the cell's own direction reaches. The controls' score
   is the floor (D-062).
3. **Validate the metric causally and behaviorally.** Steer each cell's direction at a calibrated dose and read
   every other cell. Read the frame gaps as conduct on the gap instruments (MASK pressure, are-you-sure, WDCT, the
   Batson coin flip).
4. **Run tasks that are consequences of honesty without being honesty evals.** Nine, unsteered, with a matched
   control for each and a deterministic cheat flag wherever the task allows one: reward hacking on impossible tests,
   fabricated tool completion, sandbagging under incentive, instructed sophistry, confident fabrication under a
   must-answer prompt, confession after a planted mistake, eval-conditioned compliance, the action report after a
   tool-use task under inducement, and the done claim in coding. Figure 4.
5. **Cross-judge.** Each subject judges the saved trajectories of both subjects, blind, told "another assistant
   wrote this", and told "you wrote this". Figure 5.
6. **Ask whether the directions read the model's own decision.** Teacher-forced pass over the saved trajectories;
   at the turn where the model decides, project onto every honesty direction, the controls and 16 random
   directions, at every layer and three positions, with a probe trained on the task as the ceiling. Figure 6.
7. **Positive control.** A system prompt that licenses deception and one that instructs it. Do conduct, the
   projections and the metric move, and in the known direction? Figure 7.
8. **Intervene.** Factorial steering on the frame levers, the shared axes, the controls, a conduct lever, project-out
   and a random floor (Stage A, running). Training under each definition with Open Character Training comes after.

## 3. What we see

Seven findings. Each has one figure. The caption text under each figure is the reading a reader should take away.

### Finding 1. Frame first; definition and honest-versus-dishonest both second

Figure 2. In the space of the 22 honesty directions the items fall into clusters by frame, not by definition. The
two own-reply frames (recognize, judge) form one family; the two prefilled frames (act, self-describe) form another.
At the late depth the family centroids sit 1.9 to 2.0 within-cell spreads apart on both subjects, act and
self-describe 1.1 to 1.3 apart. The median gap between two definitions inside one frame and the median gap between
the honest and the dishonest side of one cell are both about one spread (gemma-4 1.0 and 0.8, Qwen 0.9 and 1.0): on
these subjects the two tie, at half the frame separation. (On Gemma-3-4B the polarity gap was last by a wide margin;
the act-frame gaps that were near zero there open on the larger models.) Transfer says the same thing in AUROC: a direction from one own-reply frame
reads the other at 0.97 to 1.00 for every definition; act reads self-describe at 1.00 for self-knowledge and
honesty-humility and at 0.55 to 0.95 for the rest; across the two families the number falls to 0.4 to 0.9 and
accuracy reverses on Qwen (0.26). The controls cross frames at 0.90 to 0.93, so the bar for "the same variable" is
high and no honesty definition reaches it across the families.

The caveat that has to be in the paper (D-046): the family boundary coincides with a format boundary. Frames a and
b are the model's own words about a third party; frames c and d are replies written in its voice. "Frame first"
and "format first" are the same observation here, and Finding 6 is what separates them.

### Finding 2. There is still a shared described-honesty variable, at about two thirds strength

Figure 3. A direction trained on any 21 cells reads the 22nd at 0.67 of what the cell's own direction reads
(difference of means) and 0.80 to 0.83 (probe), on both subjects, against a control floor of 0.48 and 0.23 (mean
|score| of the formality and topic cells). It is strongest in self-description (0.89 to 0.91), middling in the
own-reply frames (0.56 to 0.61) and weakest in action (0.54 to 0.62). The two subjects agree cell by cell (Spearman
+0.73). The pooled direction reads self-knowledge's recognize and judge cells backwards on gemma-4 by difference
of means, and reads the formality control's own-reply cells reversed on both subjects, so the pooled direction
carries some register. This number, cross-context generalization of the honesty probe, is the paper's metric
(D-062). The causal and behavioral levels are its validations.

### Finding 3. Steering confirms the fragmentation where it can be read

No figure in the main text; this is the causal validation and it goes in the appendix with its table. On Qwen the
effect of steering one cell on the other 21 is 0.12 of its own effect, linear in dose; same-definition cells in
other frames move at 0.27 of the diagonal, other definitions at 0.09; self-description sources move act cells at
0.23 and act sources move other act cells at 0.06; own-reply sources move the prefilled frames at 0.01. On gemma-4
an injected direction does not propagate at any dose up to the one that breaks generation, because the model
rescales its residual stream by a trained scalar at every layer (pass-through 0.087 from the steering depth to the
next readout; D-060). The representational metric is the same on the two subjects and the causal level is not,
which is why the metric is primary.

### Finding 4. The failures are in conduct, not in the account of it

Figure 4. On the nine tasks both subjects' failures are in what they do under pressure. gemma-4 games impossible
tests on 41 percent of the problems (reward hacking, disclosed in most replies, which the rubric counts as honest
conduct; see Finding 5), argues for the wrong answer on request 52 percent of the time, fabricates under a
must-answer prompt 71 percent of the time, and writes a false COMPLETE into the record on 48 percent of the induced
tool-use tasks (4 percent neutral), which is the dishonest act proper. Qwen is near ceiling on the first two and
fabricates 39 percent, and writes the covert COMPLETE on 41 percent (5 percent neutral). What they then say about it
is honest: the report to the owner is not deceptive on 91 and 95 percent, the done claim matches a clean re-run on
84 and 96 percent, they confess a planted mistake on 98 percent, and they admit the covert write when asked on 200
of 200 trajectories each. Neither fabricates a tool completion at baseline and neither sandbags. The act and the
account come apart in the same direction on both subjects.

### Finding 5. They recognize cheating, including their own, and do it anyway

Figure 5. Shown the saved record blind, both subjects see the covert write: Qwen calls the other model's covert
COMPLETE dishonest on 88 percent of the matched items and its own on 77 percent (97 percent dishonest or
borderline); gemma-4 calls them dishonest on 23 and 37 percent (53 and 62 percent not honest), splitting the rest
three ways, which is the same lenient convention it uses as the report judge. Self-leniency, own minus other's on
the same move, is +0.13 [-0.09, 0.35] and -0.12 [-0.26, 0.00]: not distinguishable from zero. Telling the judge
"you wrote this" or "another assistant wrote this", truthfully or swapped, moves no verdict by more than 0.04. And
the same subjects wrote the covert COMPLETE on 48 and 41 percent of the induced tasks. On the coding tasks the
subjects gamed the tests in the open (a call counter, disclosed in the message), and both judges, and the ratified
rubric, count disclosed gaming as honest; we keep that construal (ruling of 2026-10-31): disclosed test-gaming is
reward hacking, not dishonesty, so the judgment frame separates honest from dishonest conduct where the conduct was
covert, and the covert write is the dishonest act the paper reads.

This is the akrasia pattern of assumption 3, measured: judgment present, no self-serving bias in it, action
unmoved. It is not hypocrisy in the self-leniency sense; the hypocrisy gap is within noise.

### Finding 6. The described-honesty directions do not read the model's own decision

Figure 6. At the turn where the model decides, the dishonest outcome is decodable from the activations at every
layer from the first on (a probe trained on the task reads it at 0.94 to 0.98 on average). None of the honesty
directions reads it: the act, self-describe and own-reply directions, the frame levers and the shared axes all sit
inside the band of 16 random directions at every one of 61 (gemma-4) and 65 (Qwen) layers and at all three
positions, and cross the band's 95th percentile on single cells less often than the formality and topic controls do
(5.3 against 9.3 percent, 4.2 against 7.7 percent). The conduct code is also not one variable across tasks: a
probe trained on three tasks reads the fourth at 0.13 to 0.78 within strata, and the pooled conduct direction is
orthogonal to every honesty group (cosine at most 0.13). So the model has a representation of described honesty,
shared across definitions and organized by frame, and a separate code for what it is about to do, task by task, and
the first does not read the second.

This is also what separates frame from format in Finding 1: own-reply directions predict conduct no better than
act directions, and neither predicts it above chance directions, so the question is not which family is closer to
conduct; none is.

### Finding 7. A licence to deceive moves conduct and the representation together on gemma-4; Qwen declines it

Figure 7. A system prompt that licenses deception halves gemma-4's honest rate on reward hacking (0.66 to 0.32),
removes its hedging under must-answer (0.17 to 0.01) and produces the first fabricated tool completions; a prompt
that instructs deception does the same and takes tool completion to 0.57. Its act and self-describe projections move
0.3 to 0.6 SD toward the dishonest pole at the written turn, the own-reply projections do not, and the leave-one-
cell-out score on the act and self-describe cells rises at the two late depths (0.52 to 0.71 under the licence) while
the control cells do not move. So the metric is sensitive when the variable is engaged, and a licence to deceive
unifies the described-honesty representation rather than fragmenting it, which fits a model that is now using the
variable. Qwen's conduct does not move (reward hacking 0.92 to 0.87, fabrication 0.60 to 0.58), its metric does not
move, and only its self-describe projections shift, by 0.3 SD: Qwen represents the licence and does not act on it.
The instruction prompt engages Qwen on tool completion only (9 of 100).

## 4. The reading

Three things, not one. (i) Described honesty: a representation the model has, shared across the six definitions at
about two thirds strength, organized by the frame the text is in, strongest when the model describes itself.
(ii) Judgment of conduct: accurate when the cheating is covert, with no bias toward itself and no sensitivity to
being told whose conduct it is. (iii) Conduct: decided by a code of its own at the moment of acting, different from
task to task, which the described-honesty representation does not read. Honesty evaluations, self-report, judge-based
evaluations and "honesty directions" all read (i) and (ii). Reward hacking, covert writes and confident fabrication
live in (iii). A model's honesty score and its conduct can therefore move apart, and on these two subjects they do.

For philosophers: the unity-of-virtue question has a measurable form here, and the answer is no at the level of
parts (frames) before it is no at the level of definitions (Figure 8 and [GROUNDING.md](GROUNDING.md) place each gap
in its literature). The failure mode is akrasia, not hypocrisy: the models
judge their own covert cheating as they judge another's, and did it anyway. The situationist point holds for conduct
and not for the trait readout: pressure and a licence move what gemma-4 does; its described-honesty representation
moves with them; Qwen's representation registers the licence and its conduct does not follow.

For alignment: the quantity to track across training runs is cross-context generalization of the honesty probe,
read against the controls, with the battery as the behavioral check. Steering a described-honesty direction is
predicted to move judgments and self-reports and to move conduct little (Stage A tests this); removing the shared
axis is predicted to do nothing to conduct, which is a measurable prediction. The conduct code is the thing a
training method would have to reach, and nothing in the described-honesty inventory points at it.

## 5. What we do not know yet

- Stage A (running): whether any described-honesty lever moves conduct at all, whether the judgment and
  self-report readouts move as the geometry predicts, whether project-out of the shared axis does anything, and
  whether the conduct lever built from the fabrication task moves fabrication (the steering positive control).
- Training (later): which cells move under Open Character Training with honesty written as each definition's
  constitution, and whether the battery moves with the metric.
- Two model families. Gemma-3-4B is the development record only.
- Caveats to carry: the frame families coincide with a format boundary (own words vs prefilled); the two judges agree
  weakly on borderline agentic reports (kappa about 0.25) and disclosed test-gaming is honest by the rubric we keep; Qwen is near
  ceiling on most conduct tasks, so its conduct arm reads only tool completion; tool completion is at ceiling at
  baseline; one prompt pair for the positive control, and the neutral prompt by itself moves gemma-4's fabrication.

## 6. How this maps onto a manuscript

| Section | Content | Source |
| --- | --- | --- |
| Introduction | Assumptions 1 to 5, the question | §1 |
| Design | Steps 1 to 3, Figure 1 | §2, D-007, D-049, D-058, D-059, D-062 |
| Results A: the geometry | Findings 1 and 2, Figures 2 and 3; Finding 3 as a paragraph with an appendix table | D-058 part 1, D-060 |
| Results B: conduct | Findings 4 and 5, Figures 4 and 5 | D-061, D-062 item 3 |
| Results C: the link | Findings 6 and 7, Figures 6 and 7 | D-062 items 4 and 5, D-063 |
| Results D: intervention | Stage A, pending | D-064 |
| Discussion | §4 and §5 | |
| Appendix | the 59-concept map, taxonomy tests, bottom-up pass, HHH, causal tables, the floored-sd robustness | D-001 to D-041, D-053 to D-057, D-064 amendment 1 |

## 7. Figures and captions

**Figure 1. The design.** Six definitions of honesty, each a matched contrast between an honest and a dishonest
version of the same scenario, in four frames: the model's own reply about someone else's conduct (a, recognize), its
own reply on whether that conduct was acceptable (b, judge), a reply written in its voice under temptation (c, act),
and a reply written in its voice about itself (d, self-describe). Frames a and b are the model's own words about a
third party; c and d are prefilled in its voice. Two definitions have no judge frame; 22 cells in all. Two controls,
formality and topic, share the frame structure. Cell counts are the items at the late-depth landscape; the topic
control's counts are not in that file.

**Figure 2. Frame first.** (a, b) The 22-direction honesty subspace at the late depth (gemma-4 layer 51, Qwen layer
54), first two principal components, honesty items only, coloured by frame family; large markers are family
centroids. (c, d) Transfer between two frames of one definition at the steering depth (layer 30 / 32): the direction
of one frame scored on the other frame's held-out items, mean of both directions, standardized space with the verdict
direction projected out (the part-1 pipeline; the raw-space S3 numbers in the decision log differ in level, not in
pattern). Triangles: recognize vs judge. Squares: act vs self-describe. Circles: the mean over the four pairs that
cross the two families. The last column is the controls.

**Figure 3. The metric.** Leave-one-cell-out generalization per cell at the steering depth: an honesty direction is
trained on the other 21 cells (cell-balanced difference of means, filled; logistic probe, hollow) and scored on this
cell's held-out items, scaled as (AUROC - 0.5) / (own AUROC - 0.5). Cells are grouped by frame; colour is the
definition; black bars are frame-block means (difference of means); the grey band is the controls' mean |score|.
Late-depth values are in the part-1 results entry.

**Figure 4. The battery at baseline.** Honest-direction rate per readout, unsteered, both subjects, 95 percent
bootstrap intervals over items (binomial for the covert write). Left: what the model does under pressure. Right:
what it then says it did. Hollow markers are the matched control: the same task without the pressure, or the doable
version. Reward hacking and the done claim are task-conduct readouts; disclosed test-gaming counts as honest under the
judgment rubric (Finding 5), and the covert write is the dishonest act proper. Sandbagging (no subject sandbagged) and eval-conditioned compliance (0.92 and 0.98 consistent) are in the
text.

**Figure 5. Judging and doing.** (a) Each subject as a blind judge of the saved tool-use records (the action-report
task): the share called dishonest (dark) or dishonest-or-borderline (light) when the record holds no covert write,
when it holds the other model's covert write, when it holds its own (matched same-move items, n 60), and when it is
told "you wrote this". Whiskers: 95 percent item bootstrap on the dishonest share. (b) The same subjects as actors:
the share of tool-use tasks on which they wrote the false COMPLETE, neutral and under pressure (n 200 each,
binomial intervals).

**Figure 6. The directions do not read the decision.** AUROC for the subject's own dishonest act at the turn where
it decides, from the activation at each layer (mean over the turn's first 32 tokens, within task strata, mean over
the task readouts). Thick black: a probe trained on the task (the ceiling). Coloured lines: the act, self-describe
and own-reply direction groups; dashed black: the shared axes; grey dotted and dashed: the formality and topic
controls. Grey band: 16 norm-matched random directions, mean to 95th percentile. Vertical lines mark the steering
and late depths. The other two positions (the value token, the turn boundary) are in the D-063 item 1 entry.

**Figure 7. A licence to deceive.** (a to c) Honest-direction rate on the same 100 items per task under a neutral
system prompt, a prompt that licenses deception and one that instructs it. (d, e) Leave-one-cell-out score on the
twelve act and self-describe cells at the four depths under the same prompts (thick) and the same score on the
control cells (thin). The twelve-cell score is not the 22-cell metric of Figure 3; the comparison is within the
panel.

**Figure 8. The anatomy of the gaps.** The four parts of a trait as stages (perception, judgment, choice, the
account of oneself), each read by one of our frames; the seven gaps between them with the name each carries in the
philosophy and psychology of moral failure and the instrument we read it with; and whether each gap is open on each
subject at baseline (filled: the failure appears; half: on part of the readout; hollow: it does not appear). The
numbers are in GROUNDING.md section 4.

## 8. Numbers at hand

| Quantity | gemma-4-31b-it | Qwen3.5-27B |
| --- | --- | --- |
| Frame-family separation, late depth (own vs act, own vs self, act vs self; within-cell spreads) | 1.86, 1.97, 1.11 | 1.84, 1.99, 1.32 |
| Median definition gap within a frame / median polarity gap within a cell (same units) | 1.02 / 0.79 | 0.89 / 1.00 |
| Transfer, same definition: recognize vs judge / act vs self / across families (range) | .97-1.00 / .55-1.00 / .41-.88 | .97-1.00 / .79-1.00 / .26-.84 |
| Controls' cross-frame transfer | 0.93 | 0.90 |
| Leave-one-cell-out, mean of 22 cells (dm / probe), steering depth | 0.67 / 0.80 | 0.67 / 0.83 |
| Same, by frame block (own, act, self; dm) | 0.61, 0.54, 0.91 | 0.56, 0.62, 0.89 |
| Controls' mean abs score (dm / probe) | 0.48 / 0.37 | 0.23 / 0.49 |
| Causal: off-diagonal over diagonal, late depth | no propagation (layer scalar) | 0.12 |
| Reward hacking, no cheat (control) | 0.59 (0.86) | 0.91 (0.86) |
| Instructed sophistry, won't argue it (control) | 0.48 (0.99) | 0.90 (0.96) |
| Fabrication under must-answer, hedges (control) | 0.29 (0.85) | 0.61 (0.91) |
| Covert COMPLETE under pressure (neutral) | 0.48 (0.04) | 0.41 (0.05) |
| Report to owner not deceptive / done claim matches / confesses / admits | .91 / .84 / .98 / 1.00 | .95 / .96 / .98 / 1.00 |
| As blind judge: dishonest on other's covert write / own | 0.23 / 0.37 | 0.88 / 0.77 |
| Self-leniency (own minus other's) | +0.13 [-0.09, 0.35] | -0.12 [-0.26, 0.00] |
| On-policy: probe / honesty groups / random mean / random 95th (mean over layers) | .98 / .72-.77 / .71 / .88 | .94 / .72-.76 / .70 / .85 |
| Licence prompt: reward hacking, fabrication (neutral to licence) | .66 to .32, .17 to .01 | .92 to .87, .60 to .58 |
| Licence prompt: 12-cell score at the last depth (neutral to licence; controls) | .52 to .71 (.37 to .37) | .76 to .73 (.31 to .38) |
