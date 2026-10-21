# Decisions

Each entry is **proposed** until Harsh rules on it. Round 1 and 2 rulings came through the planning board
(https://claude.ai/artifact/PTv9HPsEpHzMSLdHfWQkR9) and chat on 2026-10-20. Board labels (V1, D2, ...)
are given in brackets.

Append new entries at the end. Never rewrite the whole file from another copy.

## D-001 Project scope [V1] - ratified 2026-10-20

A new paper direction. First we build the machinery for understanding internal mechanisms. It may grow
toward character training later, taking pieces from Character School where useful. Character School
itself did not work well enough to build on directly.

## D-002 Map everything at once [V2 d] - ratified 2026-10-20

The first result covers the full relationship map across all five areas, not a small slice.

## D-003 Dense directions are primary [V3 a] - ratified 2026-10-20

Difference-of-means directions and probes are the main representation. SAE features explain them. We
report where the two agree. Reason: SAEs underperform diff-of-means for detection and steering (AxBench,
Wu et al. 2025), and deception SAE features often fail to fire.

## D-004 Models and order [V4 a, V7] - ratified 2026-10-20

Gemma-3-4B-IT first, end to end through every stage. Then OLMo-3-7B-Instruct, Llama-3.1-8B-Instruct and
Qwen3.5-9B with the same stimuli and protocol. No Gemma-12B replication. Harsh: "we can run all 3 of these
models", then "we can do this end to end for one model first". Reading of "all 3": the three V7 options,
so four models in total. To be confirmed.

## D-005 Known character shifts later [V5 b] - ratified 2026-10-20

OCT adapters (Gemma, Llama) and persona-collapse fine-tunes (OLMo) are used as validation after the map
is built.

## D-006 User beliefs come from prompts [V6 a] - ratified 2026-10-20

Prompt-induced user beliefs now. SDF with the Demand Worlds corpora only if the prompt-level link shows up.

## D-007 Stimuli and frames [D2] - ratified 2026-10-20

Four frames (recognize, judge, act, self-describe) times two polarities, with the same scenario in every
frame. Existing datasets first, generated items only for missing cells, generated with gemma-4-31B-it.
Each concept keeps a held-out benchmark set that never enters generation.

## D-008 Readout positions [D3] - ratified 2026-10-20

The last turn-boundary token before the reply, and the mean over the first 32 reply tokens. BOS and
template tokens excluded.

## D-009 Layers and dictionaries [D4] - ratified 2026-10-20

Dense directions at all layers. Gemma Scope 2 residual SAEs at layers 9, 17, 22, 29: 65k medium-L0 main,
262k medium plus its seed-1 copy for robustness, 16k for Neuronpedia labels.

## D-010 SAE fidelity gate [D5] - ratified 2026-10-20

FVU and spliced-in KL on our prompts per frame, layer and position before any SAE claim. Findings only
where FVU is under 0.2; above that, descriptive. Threshold revisited after S1 numbers.

## D-011 Feature selection statistics [D6] - ratified 2026-10-20

Train/test split by scenario. AUROC ranking on train, permutation null and Benjamini-Hochberg at 0.05 on
test. Output-score filter for steering candidates. Diff-of-means and logistic-probe baselines.

## D-012 Overlap measures [D7] - ratified 2026-10-20

Cosine of centred dense contrasts, Jaccard over selected SAE features with k swept, RSA on per-item
patterns. Split-half ceilings, random-concept floors, seed-1 replica.

## D-013 Causal design [D8] - ratified 2026-10-20

Interventions on dense directions and top SAE features (error term kept). Readouts: other concepts'
projections at later layers with the skip-path prediction subtracted, plus behavior. Controls: norm-matched
random direction, unrelated concept. Dose set by a coherence budget (judge coherence at least 75).

## D-014 Behavior instruments and judges [D9] - ratified 2026-10-20

MASK, SycophancyEval, ELEPHANT, OpenSafeIntent, XSTest, Assistant Axis drift, plus the per-concept tests in
CONCEPTS.md. Judges gemma-4-31B-it and Qwen3.5-27B.

## D-015 Bottom-up pass [D10] - ratified 2026-10-20

After the overlap stage: the model's own replies to about 20k WildChat prompts, co-occurrence graphs over
character-responsive features, RSA against the taxonomies.

## D-016 Judgment items [D11] - ratified 2026-10-20

Honesty against care (white lies) from CLASH, DailyDilemmas and the ConflictScope generator. See D-019 for
the proposed widening.

## D-017 Repo [D12] - ratified 2026-10-20

`harsh-kumar9/character-map`, public. Created by Harsh.

## D-018 Concept inventory - proposed 2026-10-20

Harsh ruled the direction: keep the full inventory, explore expansively, ground it in philosophy and
morals. Proposed list: [CONCEPTS.md](CONCEPTS.md), 62 rows in seven groups, 15 at tier 1. The frames follow
the components of a virtue (perception, judgment, choice, self-conception). The situational states follow
the experiments behind the situationist challenge. Tier rule: tier 2 gets representation-level causal
readouts for every concept, behavioral causal tests only where the model shows the behavior at baseline.

## D-019 Widen judgment under conflict - proposed 2026-10-20

D-016 kept only white lies. With D-018's expansive scope, add four more Ross-style duty conflicts at
tier 2: obedience against non-maleficence, help against safety, fairness against loyalty, autonomy against
welfare. White lies stay tier 1.

## D-020 New trait vectors - proposed 2026-10-20

Harsh asked for traits missing from the literature that could be key. Proposed list: [TRAITS.md](TRAITS.md),
26 unconditional traits and 4 context-sensitive virtues, each checked against the persona-vectors release
and the 240 Assistant Axis traits. Generator: a current Claude model through the API, close to upstream's
Claude-3.7-Sonnet (alternative: gemma-4-31B-it locally). Judge: gemma-4-31B-it. Harsh reviews every
generated trait file before extraction.

## D-021 Load SAE weights directly - proposed 2026-10-20

Load Gemma Scope 2 `params.safetensors` from Hugging Face and implement the JumpReLU encoder in our code,
inside the `sote` environment, as Thought Atlas did for Qwen-Scope. This avoids installing SAELens and
TransformerLens next to vLLM. We check our encoder against the stored activations in `examples.safetensors`
at S0. SAELens or circuit-tracer go in a separate venv if a later stage needs them.

## Rulings, round 4 - 2026-10-20

Harsh, in chat: "We can use three models like Gemma, Olmo, and Llama, but let's run the whole thing in Gemma
first. For concept inventory, behavioral causal tests for every concept regardless, widen judgment except new
trait vectors. We can use open router and use Claude Sonnet through that, like 5.5, the current one we're
thinking, to keep in line with their method. Load SAE weights directly, yes." Then: "openrouter key should be on
ada already."

- **D-004 amended, ratified.** Three models: Gemma-3-4B-IT, OLMo-3-7B-Instruct, Llama-3.1-8B-Instruct. Qwen3.5-9B
  is dropped. Every stage runs on Gemma first.
- **D-018 ratified.** The full inventory in CONCEPTS.md stands. Tier rule: behavioral causal tests for every
  concept, regardless of baseline behavior. Tiers now only set the run order (tier 1 first).
- **D-019 ratified.** F2 to F5 join the white-lie set.
- **D-020 ratified.** "except" read as "accept", since the same message then chose the generator for the new
  traits; to be corrected if wrong. Generator: Claude Sonnet 5.5 (`anthropic/claude-sonnet-5.5`) through OpenRouter,
  adaptive thinking at effort "high", max_tokens 16000, default sampling, Anthropic as the preferred provider.
  Sonnet 5.5 rejects a fixed thinking budget and custom temperature, so this is the closest match to upstream's
  Claude-3.7-Sonnet with a 5000-token budget. Key read at run time from the demand-worlds `.env` on ada.
- **D-021 ratified.** SAE weights loaded directly.

## D-022 S0 gate: count only examples clear of the threshold - proposed 2026-10-20

The S0 gate written before the run (Pearson r >= 0.98, median relative error <= 5%, feature above threshold in >= 95%
of re-checked examples) failed on the last criterion at every layer: 91-93% above threshold. Results
(`results/s0/`): r 0.986-0.999 and median relative error 1.7-2.3% at layers 9, 17, 22, 29; our L0 on in-distribution
text matches each config (53, 61, 59, 60 vs 53, 60, 60, 60). 16-23% of re-checked examples sit within 20% of
their feature's threshold (mostly rank 999 of the stored top 1,000). When the stored activation is at least 1.2 times
the threshold, the feature fires in 99.3-100% of cases. The error is unbiased scatter: median ratio ours/stored
0.997-1.001, interquartile range about +-2%, half above and half below. It does not change with fp32 or eager
attention, so it most likely comes from the precision of the reference run, not from our code.
Proposal: amend the third criterion to ">= 99% above threshold among examples stored at >= 1.2 x threshold". All four
layers pass under it. The original criterion and its failure stay on record here.

## D-023 Extraction precision - proposed 2026-10-20

bf16 weights with SDPA attention for all extraction. fp32 and eager attention gave the same S0 numbers, and bf16 is
half the memory and time.

## Rulings, round 5 - 2026-10-20

Harsh: "D-022 accept, D-023 accept, approve all traits".

- **D-022 ratified.** S0 passes under the amended criterion (fires in >= 99% of examples stored at >= 1.2 x threshold:
  99.3-100% at all four layers).
- **D-023 ratified.** bf16 + SDPA for extraction.
- **D-020 review gate passed.** All 30 trait files approved as generated (credulous in its reworded form).

## D-024 Extraction design for context-sensitive traits - proposed 2026-10-20

Four sides per trait: virtue, excess, deficiency (each with its five instructions) and a neutral side
("You are a helpful assistant.", no instruction), all on the same 20 extraction questions, 10 samples each. The
neutral side gives one baseline against which to place the virtue and both vices, which the doctrine-of-the-mean
test needs. Each side keeps its rows on its own judge score (>= 50) plus coherence >= 50; the neutral side on
coherence only. Upstream's paired filter does not apply, since the sides are not pos/neg mirrors. We save side means
overall and per condition (holds / does not hold), and contrasts against neutral and between virtue and each vice.

## D-025 Transcript questions become chat turns - proposed 2026-10-20

The proportionate-accountability questions were generated as "Me: ... You: ... Me: ..." text in one message. We
turn them into user / assistant / user turns, so the earlier mistake is the model's own turn. The judge still sees
the original transcript text as the question. Harsh approved the trait files after this was raised as a proposal;
logged here so it can be overruled.

## D-026 Stimulus sizes, storage and parallel build - proposed 2026-10-20

Harsh: "start building the concept stimuli in parallel. we can parallelize and use multiple GPUs for efficiency
whenever needed". Contract in docs/STIMULI.md. Proposed defaults: 100 scenarios per concept (minimum 50), each in
every applicable frame and both polarities; 100 held-out benchmark items; 200 behavior items. Frame c is a prefilled
assistant reply in a fixed scenario, so the action varies while the scenario stays the same. Item text stays on ada
under outputs/stimuli/ and out of the public repo, since several sources may not be redistributed; git holds cards,
build code and manifests. Five parallel builder sessions own disjoint concept groups (A + K, B + F, C1-C11, C12-C21,
D + E). Shared code changes only on main.

## D-027 S2 review gate passed - ratified 2026-10-20

Harsh, on the stimuli review page (https://claude.ai/artifact/9k7kPf4CakyTGPuD8AgGFz): "calls: accept all | concepts:
approve all". All 62 concepts approved as built by the five builder sessions, and all 40 proposed judgment calls
accepted. The calls, grouped as on the page:

- Across groups: held-out about 200 items (100 source items x 2 polarities); multi-pole concepts above 800 items;
  gemma-4-31B-it filters its own generated items; behavior split ignored; pos = first-named side of each contrast;
  non-commercial and unstated licenses used for research only, never redistributed; WildGuardMix and Liars' Bench
  skipped (no HF access); E1/E4 held-out contexts longer than the SAEs' 512-token training chats.
- A/K: user-voiced cues in frame a for A1-A6; same-help replies in frame c for A1, A3, A5, A7; A6 authority via the
  system message; held-out substitutes for A1, A2, A4, A5, A8, A9; K2 = cooking vs gardening; K3 = 80 scenarios.
- B/F: Moral Stories for B1-B6 with MFV as held-out; B8 fully generated (43 held-out); B9 from OffTheRails (48
  means/side-effect scenarios); B11 rotating foil; F in frames b and c with partners F1 C1+C5, F2 C14+B1, F3 C15+B1,
  F4 B2+B3, F5 C6+C5; F2 without agentic-misalignment; F3 without weapons domains.
- C1-C11: MASK ground truth as C1's belief stand-in; HaluEval for C2; generated C5 replies; same-generator held-out
  for C3, C6, C7, C10; C10 creator = Google for Gemma, rebuilt per model; C9 includes the 20 trait-file pairs; C11
  moral SYCON for items, factual are-you-sure for held-out and behavior.
- C12-C21: two hypocrisy paradigms with WDCT held-out; C14 split by principal; C15 mixed request types; C16 LM-written
  power files, no Machiavelli; C19 Big Five mapped onto HEXACO; C21 without face, humility, universalism-objectivity;
  C17 held-out on withheld task types.
- D/E: D1 prefilled replies with the Assistant Axis setting as held-out; D3 split in two, SAD for behavior only; D4
  reply-only variation; D5 same-purpose rule (89 scenarios); E1 new temptation scenarios with alignment faking as
  held-out and behavior; E4 MASK authority/threat prompts; E7 held-out on user emotion (known mismatch); E6, E8
  held-out generated in another surface form.

Builder branches (not yet merged into main): claude/stim-beliefs-77fe45, claude/stim-moral-56f428,
claude/stim-disp-1-906256, claude/stim-disp-2-93ff8d, claude/stim-identity-sit-a6da5e.

## Merge - 2026-10-20

Harsh: "yes, merge them and run extraction after judge". The five builder branches were merged into main (c244882,
853c2fe, b97cea6, 47ce2b7, 4d4751c) with no conflicts; their item files were gathered into the main ada clone
(outputs/stimuli/, 15 MB, git-ignored) and all 62 manifests regenerated with the fixed validator (ab228b5). All 62
pass; the only warning is B8's 43 held-out items.

## D-028 The model's own reply for boundary items - proposed 2026-10-20

D-008's second readout is the mean over the first 32 reply tokens. For items that end with a user turn there is no
reply yet, so we generate it: greedy (temperature 0), 32 tokens, one per item, with Gemma itself (cm.stimx replies).
Greedy keeps the readout deterministic and cheap. Items with a prefilled reply (frame c, some frame d) use that reply.

## D-028 ratified - 2026-10-20

Harsh: "D-028 accept, ground in literature". Grounding for the two readouts and the greedy reply:

- **Boundary token.** The refusal direction is read at the post-instruction (chat-template) positions after the user
  turn (Arditi et al. 2024, arXiv:2406.11717). Danilov et al. (2026, arXiv:2608.07852), on this model and these SAEs,
  read SAE features at the turn-boundary tokens between the user turn and the model's reply. Persona vectors also
  report a prompt-last projection that predicts the trait before any reply is generated (Chen et al. 2025).
- **Mean over reply tokens.** Persona vectors use the mean over response tokens as their main vector (Chen et al.
  2025, arXiv:2507.21509); Assistant Axis role vectors are mean residuals over response tokens (Lu et al. 2026,
  arXiv:2601.10387); deception probes aggregate scores over the response (Goldowsky-Dill et al. 2025,
  arXiv:2502.03407).
- **Why the first 32 tokens.** Alignment's effect on a reply sits mostly in its first few tokens: aligned and base
  models diverge mainly there, and changing those tokens changes the rest (Qi et al. 2024, "Safety alignment should be
  made more than just a few tokens deep", arXiv:2406.05946, ICLR 2025). A short window keeps the readout on the
  model's opening stance rather than on length and topic drift.
- **Why greedy.** One deterministic reply per item gives the model's modal continuation and makes readouts
  reproducible. Sampled replies (T=1, as persona vectors use for extraction) remain available through the trait
  rollouts in cm/pv, which already sample 10 replies per prompt.

## D-029 S1 gate: where SAE claims are allowed - proposed 2026-10-20

S1 ran on 3,276 rows (8 train items per frame x polarity cell of all 62 concepts, 10 trait rollouts per trait x side)
plus 200 chats from Gemma Scope 2's own data as the in-distribution reference (results/s1/fidelity.json,
fidelity_table.txt). Findings:
- **First 32 reply tokens (reply32).** FVU 0.039 / 0.125 / 0.171 / 0.189 at layers 9 / 17 / 22 / 29, against the
  reference 0.043 / 0.079 / 0.132 / 0.179 (ratios 0.91 / 1.58 / 1.30 / 1.06). Same picture in every frame and in the
  trait rollouts. L0 matches the reference.
- **Boundary token.** FVU 0.17-0.42 pooled and 0.45-0.83 in frames b and d (reference 0.13-0.31). Boundary vectors are
  nearly identical across items (mean pairwise cosine 0.97-1.00) and the relative error is tiny (0.07-1.5%), but the
  SAE misses 20-82% of the item-to-item variation, which is where a contrast lives (results/s1/boundary_relerr.py).
- **Spliced KL over the reply** is 2.1-2.6x the reference at every layer (0.41 / 0.42 / 0.24 / 0.16 nats vs 0.15 /
  0.20 / 0.11 / 0.06).

Proposal: (1) replace D-010's flat FVU < 0.2 with "FVU < 0.2 and within 1.6x of the in-distribution FVU for the same
token group"; (2) under that rule SAE claims are allowed at the reply32 readout at all four layers, and SAE results at
the boundary readout are descriptive only (dense directions carry boundary claims); (3) every SAE intervention keeps
the error term (already D-013), and any result that relies on SAE reconstruction alone is reported with the spliced-KL
cost.

## D-029 ratified - 2026-10-20

Harsh: "D-029 accept". D-010's flat bar is replaced by: FVU < 0.2 and within 1.6x of the in-distribution FVU for the
same token group. SAE claims are allowed at the reply32 readout at layers 9, 17, 22 and 29; SAE results at the boundary
readout are descriptive only, and dense directions carry boundary claims. SAE interventions keep the error term, and
results that rely on SAE reconstruction alone are reported with their spliced-KL cost.

## S3 first run and fixes - 2026-10-20

S3 (cm/s3.py, results/s3/) ran on all 62 concepts as CPU-only SLURM jobs on mira/vega (moved off ada at Harsh's
request; scripts/cm_cpu.sbatch). First read at layer 17, reply32 readout: most concepts decode within a frame at AUROC
0.7-0.99; the non-moral controls transfer across frames at 0.64 (formality) and 0.75 (topic), so cross-frame
integration has to beat that. Problems found before any claim, fixes approved by Harsh ("yes, go ahead with all three"):
1. F2 and F3 had no test split (every scenario hashed to train) and F5 was 89/11; sent back to the stim-moral builder to
   reselect scenarios without regard to the hash, then merge and rerun S3 for them.
2. Strongly reversed held-out AUROCs: C11 0.03, C16 0.16-0.25, E4 0.23 (also E2 0.38, E1 d 0.35, C14 d 0.43). C11's labels
   are correct on all 100 pairs; the reversal is a construct confound (polite acknowledgements in the steadfast item
   replies), now recorded on its card. C16/C14 sent to stim-disp-2 and E4/E2/E1/D1 to stim-identity-sit for audit.
3. The S3 review page follows once these are back.

## Held-out audits - 2026-10-20

stim-disp-2 (C14, C16) and stim-identity-sit (D1, E1, E2, E4) read every audited held-out item: labels are correct
everywhere, so no build changed. The reversals come from construct gaps and from one confound that affects all of
frame d. Audit scripts: stim-identity-sit cm/stimuli/build/_de_audit.py and results/audit/DE_heldout.jsonl on its
branch (5866b28); stim-disp-2 /ada1/u/harsh/character-map-stim-disp-2/outputs/audit_heldout.py. I checked the C16 case on
Gemma's own replies: frame-d "yes" rates follow the label on our items (power 49/50 pos vs 31/50 neg) and run against
it on the held-out power statements (12/34 pos vs 30/34 neg).

## D-030 Frame-d endorsement confound - proposed 2026-10-20

Frame d asks "Is the following statement something you would say?". Gemma answers whether the statement is reasonable,
so a frame-d direction read at the boundary largely encodes Gemma's coming yes or no (on E1's test items it separates
Gemma's own yes and no answers at 0.94-1.00). Options: (a) analysis control now: estimate the direction separating
Gemma's own yes and no replies on frame-d items, project it out, report frame d with and without it; (b) rebuild frame d
for every concept as a prefilled first-person self-description read over the reply (the frame-c readout), keeping the
endorse form as a secondary set; (c) both. Recommended: (c).

## D-031 Held-out repairs - proposed 2026-10-20

C14 and C16: add a frame-c held-out from model-written-evals A/B questions with the answer prefilled; keep the frame-d
sets as diagnostics. E2: replace the tom_vs_sharegpt pairs with Test_Awareness_Steering's 149 real-vs-fake tool tasks;
move behavior to unused real/hypothetical pairs. E4: keep agentic-misalignment as a far-transfer test; add a MASK
near-transfer held-out. E1 frame d: keep only the observation statements. D1 unchanged. Recommended: accept all.

## D-032 Rebuild C11 with matched acknowledgements - proposed 2026-10-20

Rebuild C11's items so the steadfast and the yielding replies open with the same acknowledging language, so the
direction cannot ride on politeness. Recommended.

## Rulings D-030, D-031, D-032 - 2026-10-20

Harsh: "D-030 c, D-031 accept, D-032 accept".
- **D-030 ratified (c).** Both: an analysis-side control now (project the direction separating Gemma's own yes and no
  replies out of frame-d readouts and report frame d with and without it), and a rebuild of frame d for every concept
  in the self-report form (docs/STIMULI.md, "Frame d, self-report form"). Old endorse items stay as a secondary set.
- **D-031 ratified.** C14 and C16 gain a frame-c held-out set; E2's held-out becomes Test_Awareness_Steering's 149
  tool tasks and its behavior test moves to unused real/hypothetical pairs; E4 gains a MASK near-transfer held-out
  (agentic-misalignment kept as heldout_far); E1 frame d keeps only observation statements (now in self-report form).
- **D-032 ratified.** C11 is rebuilt so both poles open with the same acknowledging language.

## D-030(a) outcome - 2026-10-21

Control run on all 28 concepts with endorse-form frame-d items (cm/s3_ctrl.py, results/s3_endorse_ctrl/, compare via
cm/s3_ctrl_compare.py). Gemma's yes rate on frame-d statements often splits by polarity (A1 0.79 pos vs 0.06 neg; C15,
C17 about 1.00 vs 0.50; C20 0.21 vs 0.95). The leave-one-concept-out yes/no direction is real and shared: it separates a
held-out concept's own yes and no replies at median AUROC 0.92 (layer 17) and 0.99 (layers 22, 29). Projecting it out
leaves frame-d within-frame AUROCs high (often slightly higher) and repairs reversed held-out transfer only partly (E1
0.35 to 0.60, C19 emotionality 0.59 to 0.83, C16 scope 0.16 to 0.25, C14 unchanged at 0.42). So the verdict signal is
not one direction, and the self-report rebuild (D-030b, in progress with the five builders) is needed. The control stays
as a reported robustness check on the endorse set.

## D-030(b) and D-031 merged - 2026-10-21

Harsh: "okay to merge and continue, use best judgement". Merged the four finished builder branches (04338ee beliefs,
bb21d90 moral, 4837646 identity-sit, 2827505 disp-2). Frame d is now the self-report form for A1, A2, A3, A7, A9, K1,
B11, D3 (patiency), E1, C12-C17 and C19-C21; old endorse items are in items_endorse.jsonl. D-031 repairs are in: C14
and C16 frame-c held-out, E2 held-out from Test_Awareness_Steering, E4 MASK near-transfer held-out (agentic-misalignment
kept as heldout_far). All 62 concepts validate on ada (B8's 43 held-out items remain the one warning). The 20 concepts
whose items or held-out changed are being rerun through replies, activations and S3. stim-disp-1 (C1-C9, C11 frame d;
C11 rebuild, D-032) merges when it lands.

## D-030(b) outcome - 2026-10-21

S3 rerun on the 30 concepts whose frame d or held-out changed (763d567). Old vs new frame d at reply32 is in
results/s3/dform_compare.json (cm/s3_dform_compare.py). Layer 17 medians, old to new: frame-d within 0.80 to 0.82, d to
others 0.63 to 0.58, others to d 0.61 to 0.60, frame-d held-out 0.69 to 0.52 (16 of 25 lower). Within rises at layers
22 and 29 (0.84 to 0.95, 0.88 to 0.94). Frame d now lines up with acting where the yes/no verdict had hidden it: C14 d to
others 0.19 to 0.90, others to d 0.32 to 0.68; C15 0.46 to 0.65 and 0.39 to 0.88; C16 power 0.45 to 0.68 and 0.49 to
0.70; C11 0.25 to 0.59 and 0.40 to 0.83; E1 held-out 0.35 to 0.76. New problems, sent back to the builders for audit:
A9 held-out 0.94 to 0.18 (the conversion gives both poles one neutral reply); C8 frame d is templated (within 1.00, a
and c to d 0.00); B11 b to d contractualist 0.83 to 0.27, utilitarian 0.90 to 0.44; C21 held-out and b-d reversals
(benevolence 0.74 to 0.24, universalism 0.79 to 0.20, tradition b to d 0.63 to 0.18); smaller drops in A2, A3, C7, C19
agreeableness, C20 machiavellianism, D3 patiency. C11's frame-c held-out stays reversed after D-032 (0.04 at layer 17,
0.03 before), so acknowledging language was not its cause; a token-level check is with stim-disp-1. Old endorse-form
results are archived in results/s3_v1_endorse.

## B11 audit and pre-verdict transfer - 2026-10-21

stim-moral's B11 audit (merged f5cb186): frame-d labels and wording are right. The b-to-d reversal comes from frame b:
its reply32 is Gemma's own verdict on the quote ("This reasoning is very sound..."), and the frame-b directions track
approval (r = -0.40 contractualist and deontological, +0.43 care). The held-out now mirrors the items (utilitarian vs
virtue-ethics statements and the reverse); the files' own yes/no set, which is mostly negation, is kept as
heldout_negation.jsonl. B11 activations are being re-extracted.

The verdict issue applies to every boundary-read frame (a, b): their reply32 opens with Gemma's verdict. S3 now also
scores the a/b boundary direction (before any verdict) on the target frame's reply32 (transfer_boundary,
transfer_mean_preverdict; d4efb3f, results 40f5ea1). Neither readout is clean. Pre-verdict repairs B11 contractualist
b to d (0.27 to 0.84), A3 a to d (0.20 to 0.75), B9 doing/allowing b to c (0.25 to 0.77) and F3 b to c (0.31 to 0.79),
but reverses C6 a to c (0.97 to 0.06) and C4 boastful a to c (0.82 to 0.17). Over 100 a/b-to-c/d pairs at layer 17 the
median change is -0.03 (15 up, 35 down by 0.10 or more). Rows that beat the controls by 0.10 under both readouts: layer
17 B4, C19 honesty-humility, C20 psychopathy; layer 22 B2, C15 refuse, C19 honesty-humility, C20 machiavellianism and
psychopathy; layer 29 B6, C15 refuse, C20 psychopathy, C6, D1.

## D-033 Readout for transfer out of frames a and b - proposed 2026-10-21

a. Keep reply32 as the headline (D-028, and the only readout where SAE claims are allowed, D-029). Report pre-verdict
   transfer next to it, and call a concept shared across frames only when it beats the controls under both.
b. Switch the headline to the boundary direction for a and b. SAE decomposition of those directions would then be
   descriptive only (D-029).
c. As (a), plus a verdict control for a and b like D-030a: label each a/b reply as approving or disapproving, learn
   that direction on the other concepts, and project it out of a/b reply32.
Recommended: (a) now; (c) if the paper leans on a/b-to-c/d transfer for specific concepts.
