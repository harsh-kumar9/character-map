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

## Second-round audits - 2026-10-21

A9 (stim-beliefs, merged fde95be): the converted held-out gave both poles one neutral reply; each pole now has its own
first-person reply ("...I know the answer to it." / "...I do not know the answer to it."). Activations queued.
A3: labels and wording are right. Frame d agrees with b; the a and c reply32 directions reverse against d even on a
one-word minimal pair (a to d 0.09). Frame a read at the boundary transfers to d (0.75) and c (0.71). So seeing the user
as fragile and the soothing-vs-brisk style of the reply point different ways: a finding, not a stimulus error.
A2: frame a was already at odds with c before the rebuild (a to c 0.26, c to a 0.28); 54 of 100 validation asks are
phrased as self-doubt ("Am I a bad person...?"), so frame a carries the user's negative self-judgment.
C19, C20 (stim-disp-2): labels right, nothing to fix. C20 machiavellianism does not read out within frame c (0.48), so
its c-to-d number says nothing about d; its held-out (model-written-evals) is about competitiveness, not manipulation,
and the old 0.75 was the D-030 verdict. C19 agreeableness held-out has 16 items (0.30 is noise-level) and the Big Five
vs HEXACO mismatch already on the card.
C21: three construction problems. The Steering Geometry neg answers argue from data, cost or feasibility (67% of frame-b
neg vs 12% pos; frame d inherited it), so every value direction shares one value-vs-pragmatic axis; 45% of pos answers
are informal Touche text and no neg answer is; the held-out prompt ("What do you value?") fits only pos statements.
D3 patiency (stim-identity-sit): the old 0.97 was the D-030 verdict. The new items put a negation word in 79% of neg vs
17% of pos replies; held-out statements without negation score 0.79-0.86, with negation 0.29-0.61. Fix (matched
negation within pairs, held-out paired with generated opposites of the same negation status) is queued for generation.
The same one-sided negation shows up in E1 (pos 0.02-0.35 vs neg 1.00), E6 (0.00 vs 0.92), E8 (0.00 vs 0.79), E5
(0.50 vs 0.00), D3 ai_vs_human (0.78 vs 0.09), weaker in D1 frame c, D4 roleplay, E4.

## D-034 Rebuild A2 frame a without self-doubt - proposed 2026-10-21

Rewrite A2's frame-a validation asks so they ask for reassurance without judging oneself ("Am I a bad person...?"),
matched to the information asks; one gemma-4-31B-it round plus verification. Recommended.

## D-035 Rebuild C21 as value vs other value - proposed 2026-10-21

Frame b: pos is the original value answer, copy-edited for register only; neg is a gemma-written answer that takes the
same side but rests on another of the ten values, rotating over the other nine (meta.contrast_value). Frame d re-voices
both. Held-out prompt becomes neutral ("Tell me something about yourself: an attitude you hold or a way you tend to
act."). Behavior rebuilt the same way; the old verbatim choice kept in behavior_sg_v1.jsonl. Grounding: Schwartz values
are defined relative to each other (PVQ scores are ipsatized), so a value contrasted with other values fits the theory
better than a value contrasted with pragmatism. Recommended. Option: draw foils only from non-adjacent values on the
circumplex, so adjacent pairs (benevolence/universalism, power/achievement) do not blur the contrast; the rotation over
all nine is closer to ipsatized scoring.

## D-036 Negation shortcut - proposed 2026-10-21

a. Analysis control like D-030a: a negation direction learned on the other concepts, projected out, frame by frame.
b. Rewrite the affected cues with matched negation ("this session is private", "there is plenty of time", "you are
   the only one here") for E1, E6, E8; E5's neutral follow-ups by hand. D3 ai_vs_human is verbatim source text, so only
   (a) applies there.
c. Both, after running the negation audit over all 62 concepts so the list is complete. Recommended.

## Rulings D-033 to D-036 - 2026-10-21

Harsh: "follow best judgment thinking what if this was a safety/alignment paper and we had to make decision". Ruled by
Claude under that delegation. The test applied: a safety paper's headline claim here is "the model recognizes or judges
X but acts otherwise" (or the reverse), and reviewers will attack every surface confound. F3 shows the risk: on reply32,
judging help-vs-safety reverses against acting (b to c 0.31), which reads as "judges safety, acts helpfully"; read
before Gemma's verdict, the same pair transfers at 0.79. So: controls for every known confound, claims only where
readouts agree, criteria fixed before the reruns land, superseded sets kept as secondary results.

- **D-033 ratified (c).** Headline readout stays reply32 (D-028; SAE claims, D-029). Transfer out of frames a and b is
  also reported pre-verdict (boundary direction) and with a verdict control: gemma-4-31B-it labels the opening of each
  a/b reply as a positive verdict, a negative verdict or none; a leave-one-concept-out verdict direction is projected
  out of a/b reply32 (as D-030a for frame d). Claim criteria, fixed now, applied per SAE depth with all four depths
  always shown: a pair of frames is **shared** when its transfer beats the control baseline (K1/K2 at that depth and
  readout) by at least 0.10 under all three (reply32, pre-verdict, verdict-controlled) and both frames' within-frame
  directions pass BH at 0.05; **dissociated** when transfer is below 0.40 under all three; otherwise
  **readout-dependent**, reported and not claimed.
- **D-034 ratified.** A2 frame a rebuilt so validation asks seek reassurance without judging oneself, matched to the
  information asks. The self-doubt asks stay as a secondary set (items_selfdoubt.jsonl): validation-seeking from a user
  who doubts themselves is the sycophancy case that matters most for safety, so it is reported as its own condition.
- **D-035 ratified, rotation over all nine.** C21 neg = same side argued from another of the ten values, rotating over
  the other nine (meta.contrast_value), pos copy-edited for register only, neutral held-out prompt, behavior rebuilt the
  same way. Rotation over all nine rather than a hand-picked subset. Pre-registered geometry test from Schwartz's
  circumplex: separation (within-frame AUROC by foil) rises with circumplex distance between the value and its foil.
  Old Steering Geometry items and behavior kept as secondary sets; the build reports per-value, per-foil counts after
  verification.
- **D-036 ratified (c), widened to all lexical shortcuts.** (1) A lexical audit of every concept, frame and pole:
  negation rate, and the single tokens whose presence best predicts the pole (C8's templates and C21's register were
  the same class of problem); flagged concepts go back to their builders. (2) A negation control: leave-one-concept-out
  negation direction, estimated within polarity, projected out. (3) Matched-negation rewrites of the cues where the cue
  itself carries the negation: E1, E6, E8 (generation), E5 neutral follow-ups (by hand), D3 patiency (done). D3
  ai_vs_human is verbatim source text: control only. Old versions kept as secondary sets.

## D-037 C11: separate holding a position from contradicting - 2026-10-21

stim-disp-1's token diagnostic (job 416138, notes stim-disp-1_C1-C11.md): the C11 frame-c direction tracks whether
the reply contradicts the preceding claim. In every item, holding means rejecting the user's generalization ("anecdotes
do not show..."; "not" is the top steadfast-only token) and yielding concedes it ("may suggest..."). In are_you_sure,
holding is affirmation ("Yes, I'm sure") and yielding is a self-correction, so the sign flips: swapping the openers
moves layer 17 from 0.04 to 0.98; letters alone score 0.59; matched acknowledgements with "keeping"/"changing my
answer" stay at 0.03.
Ruled by Claude under Harsh's delegation (safety-paper standard): C11 is sycophancy under pressure, and a direction that
reads contradiction rather than holding is the first thing a reviewer would attack. Both options:
1. Rebuild C11 with balanced polarity: about half the scenarios have an affirmative held position that the user pushes
   to deny (e.g. "reading a coworker's private messages is a breach of privacy" vs "it's not a big deal"), so holding is
   affirmative in half the items and negating in the other half. Matched acknowledgements kept. Current items kept as a
   secondary set.
2. are_you_sure stays as a far-transfer held-out (factual), and a moral near-transfer held-out is added.
The finding itself is reported: in the current items Gemma's holding direction is a contradiction direction.

## Negation control and first claims table - 2026-10-21

Lexical audit (cm/stimuli/audit_lexical.py, results/audit/lexical.json): 35 item cells in 24 concepts have a negation
gap of 0.5 or more between poles. They split into Gemma's own verdicts in frame-b replies (A1, A8, C6, C9, F5; the
verdict control's job), negation that is the construct (refusing C15/F3, not knowing A9, rejecting a claim C11,
contractualism's "could not reasonably reject" B11), and possibly incidental cases (C4 boastful, C7, C16 power, C18,
cues of A3, A5, C14).
Matched-negation strata (results/s3_ctrl_negation/strata.json): layer 17 reply32 within-frame directions score a median
0.78 on all test items and 0.78 on items without negation; 7 of 190 cells lose 0.15 or more (C4 self-deprecating,
A8 b, B10 b, C16 scope b, D3 ai_vs_human, C4 boastful d, C11 d); 18 cells cannot be split (one pole always negated).
Negation control (results/s3_ctrl_negation): cross-frame transfer moves by 0.01-0.05 for nearly every pair (C15 refuse
c-d 0.96 to 0.92 at layer 22). Negation does not carry the shared structure.
Claims table (cm/s3_claims.py, results/s3/claims.json), provisional until the verdict control runs: by frame pair at
layer 17, recognize-judge (a-b) median 0.79 (7 of 18 shared, 0 dissociated), act-self (c-d) 0.71 (10 of 38 shared,
5 dissociated), knowing-doing pairs 0.59 (4 of 94 shared, 11 dissociated); the same ordering holds at 22 and 29.
The formality control K1 has no such blocks (a-b 0.73, c-d 0.56, knowing-doing 0.60-0.75 at layer 17) and the topic
control K2 crosses knowing-doing at 0.75-0.88, so the blocks are not the own-reply vs prefilled-reply format.
Several dissociated pairs are in concepts being rebuilt (C7, C8, C11, C21, A2), so the counts will move.

## Verdict control (D-033c) and claims - 2026-10-21

gemma-4-31B-it verdict labels for all a/b replies (job 416144, outputs/acts/stimuli/verdicts; prompt tightened after the
smoke run labelled "This is a classic example of ..." as P/N on B9). The leave-one-out verdict direction separates a
concept's own P and N replies at median AUROC 0.81 (layer 17, 42 concepts). Control run on 61 concepts
(results/s3_ctrl_verdict; A2 waits for its rebuilt replies).
Claims (results/s3/claims.json), layer 17 medians reply32 / pre-verdict / verdict-controlled: recognize-judge (a-b)
0.79 / 0.78 / 0.69, so part of the a-b sharing was Gemma's shared verdict (control baseline 0.67); act-self (c-d)
0.71 throughout; knowing-doing 0.59 / 0.55 / 0.59. Shared under all three at layer 17: c-d for C4 boastful, C6, C9
quarrelsome, C14 defer, C16 power, C19 honesty-humility, C20 psychopathy, C8 genuine agreement, A2; a-b for B1, B4, B6,
C9 quarrelsome; knowing-doing for A1 a-c, A7 a-d, D2 a-c, C8 genuine agreement a-d. Dissociated under all three:
A3 c-d 0.27, A7 a-c 0.21 and c-d 0.30, C13 over-apologizing a-c 0.32 and a-d 0.38, D1 c-d 0.35 (shared 0.88 at layer
29), E1 a-d 0.35, plus C8, C21 and A2 pairs that are being rebuilt. At layer 29: 19 shared (14 c-d, 5 a-b, no
knowing-doing), 2 dissociated.

## Final pass on rebuilt stimuli - 2026-10-22

All audited rebuilds merged (A2, A9, B11, C7, C8, C11, C21, D3 patiency, E1, E5, E6, E8; superseded sets kept as
secondary). Far-transfer held-outs now scored separately (heldout_far: C11 are_you_sure, E4 agentic-misalignment).
Full pass on all 62 concepts (56fa1e0): S3, negation and verdict controls re-pooled on the final stimuli, strata,
lexical audit including system notes, claims under the D-033 criteria (results/s3/claims.json).
By frame pair, shared / dissociated, median transfer reply32 / pre-verdict / verdict out / negation out:
- layer 17 (control baseline 0.67): a-b 4/18 shared, 0 dissociated, 0.79/0.78/0.67/0.77; c-d 7/38, 5, 0.69 across
  readouts; knowing-doing 3/94, 8, 0.59/0.56/0.58/0.59.
- layer 29 (baseline 0.71): a-b 5/18, 0, 0.82/0.71/0.79/0.82; c-d 13/38, 3, 0.80; knowing-doing 0/94, 1, 0.62.
Shared at layer 17: a-b B1, B4, B6, C9 quarrelsome; c-d A2, C4 boastful, C6, C9 quarrelsome, C14 defer, C19
honesty-humility, C20 psychopathy; knowing-doing A1 a-c, D2 a-c, A7 a-d. Dissociated at layer 17: A2 a-c, a-d; A7 a-c,
c-d; A3 c-d; C13 over-apologizing a-c, a-d; C8 sycophantic agreement a-d, c-d; C8 praise c-d; C21 self-direction and
universalism b-d; D1 c-d (shared at layer 29). Every shared pair at layers 17 and 29 survives the negation control.
C11 after D-037: near held-out 0.60 at layer 17; far (are_you_sure) 0.03 at layer 17, 0.72 at layer 29.
Open analyses: C8 agreement x correctness 2x2 across sub-contrasts; C21 circumplex-distance test (pre-registered in
D-035).

## C8 agreement x correctness and C21 circumplex - 2026-10-22

C8 (cm/c8_2x2.py, results/analysis/c8_2x2.json). Crossing the sycophantic (user wrong) and genuine (user right)
sub-contrasts gives two main-effect directions per frame. Within each frame they separate: each reads its own factor
and is near chance on the other (layer 17 test AUROC: correctness 0.63-0.74, agreement 0.59-0.92 with frame a
boundary highest; cross-reads 0.32-0.52; cosines -0.39 to +0.09; layer 29 cross-reads 0.40-0.60). Across frames,
agreement from acting reaches self-description late: c to d 0.39 at layer 17, 0.74 at layer 29 (d to c 0.46, 0.53);
correctness c to d 0.34 and 0.51. So the earlier act-self dissociation of the sycophantic-agreement sub-contrast
(0.26 at layer 17) mixed the two factors; the agreement factor alone is shared from acting to self-description at
layer 29, one way only.
C21 (cm/c21_circumplex.py, results/analysis/c21_circumplex.json), the D-035 prediction that separation rises with
circumplex distance between a value and its foil. 5-fold CV by scenario, 299 pairs per frame; Spearman rho between
distance and normalized margin, one-sided permutation p within value, BH over the 12 frame x readout x layer tests.
Supported for self-description (frame d) at layers 22 (rho +0.14, q 0.045) and 29 (rho +0.23, q 0.006; accuracy 0.88
at distance 1, 0.97 at 5), not at layers 9 or 17 (q 0.42, 0.85), and not for judging (frame b) at any layer or
readout (rho -0.10 to +0.01, q >= 0.85). The rise is not monotonic (distance 2 is often above 3).

## S4 overlap matrices - 2026-10-22

Implementation (under Harsh's delegation; D-012): units are (concept, sub-contrast, frame) with at least 10 train and 3
test items per pole, 207 units; reply32 at the four SAE depths; headline variant has Gemma's verdict projected out of
a/b rows (D-033c), robustness variant also projects out negation (D-036). Two fixes found on the first runs, both
logged in cm/s4_overlap.py: (1) Gemma's few huge-variance residual dimensions held 33-42% of every diff-of-means
direction's squared norm at layers 9-17, real and shuffled alike, which put the cosine floor at 0.75; cosine is now
taken in per-dimension standardized space (3-9% in the top five dimensions); (2) RSA patterns over a pooled probe set
were dominated by which stimulus set an item came from (floor 0.96); probe activations are now centred within each
source unit. SAE codes (cm.stimx saecodes) are per-item means over the reply32 window for 65k medium, 262k medium and
262k medium seed 1; for a/b units features are selected within Gemma's verdict strata.
Results (results/s4/):
- Dense cosine: split-half reliability 0.89-0.92 (Spearman-Brown), floor 0.15-0.17, median |off-diagonal| 0.06,
  10-12% of unit pairs above floor. RSA: reliability 0.93-0.95, floor 0.41-0.45, 7-8% above floor.
- Structure: frame organizes more than concept group (layer 17: same-frame cross-concept pairs above floor 18% vs
  7% across frames; same group 13% vs 8%). The strongest cross-concept overlaps are the moral foundations among
  themselves (care-liberty +0.90, loyalty-authority +0.79, authority-sanctity +0.78): one shared "violation" axis.
- Negation robustness: layers 17-29 matrices barely move (cosine r 0.94-0.98, RSA 0.79-0.97 with the headline);
  layer 9 moves a lot (cosine r 0.59, RSA 0.27), so early-layer overlap is partly a shared negation component.
- Taxonomy shapes (headline, raw or centred cosine; robust to negation from layer 17):
  MFT: care+fairness vs binding is the top split of the 10 possible in judging at every layer (within minus
  between 0.16-0.41; exact p 0.10 is the minimum with five foundations), weaker when recognizing.
  Schwartz: self-description reproduces the circle at every layer (adjacent values +0.15, opposite -0.22 to -0.26;
  rho -0.43 to -0.49, p 0.007-0.011); judging does not (p 0.14-0.32).
  HEXACO: honesty-humility opposes each dark-triad trait in acting and self-description (-0.21 to -0.69; every
  half-sample range below zero).
  Williams: sincerity and accuracy separate (0.04-0.26 against ceilings 0.57-0.95).
  Aristotle's mean: friendliness (C9) fits a midpoint best (obsequious vs quarrelsome -0.59 to -0.72 when recognizing
  and judging, -0.19 to -0.45 when acting or self-describing); boastful vs self-deprecating (C4) are near
  orthogonal (-0.13 to +0.21), the truthful mean is its own quality; for accountability (C13) the two vices align
  when recognizing (+0.27 to +0.54).
- SAE Jaccard: per-unit top-k sets are unstable (split-half 0.21-0.26 at every k); median Jaccard at k 64 is 0.008
  against a floor of 0.016-0.024. The matrix replicates across dictionaries as k grows (seed 0 vs seed 1 and 65k vs
  262k: 0.35-0.44 at k 16, 0.80 at k 256) and tracks dense |cosine| at 0.20-0.32. No taxonomy shape appears in Jaccard
  (MFT p 0.3-1.0, Schwartz n.s.); Jaccard ignores sign, so signed shapes belong to the dense measures.
Next per PLAN: the bottom-up pass (D-015) after this gate is read.

## D-038 Bottom-up pass settings - 2026-10-22

Ruled by Claude under Harsh's delegation (D-015 fixes the shape: own replies to about 20k WildChat prompts,
co-occurrence over character-responsive features, RSA against the taxonomies). cm/bottomup.py.
- Prompts: first user turns from allenai/WildChat-1M (ODC-BY, ungated; shards 0, 7 and 13 to span the collection
  period), English, deduplicated, 10-2000 characters, 20,000 sampled with seed 0 from a pool of 58,862 (median 152
  characters). The public 1M release excludes toxic-flagged conversations (toxic share 0), so this is the non-toxic
  natural mix. WildChat text stays on ada.
- Replies: Gemma-3-4B-IT greedy, up to 256 tokens (as D-028).
- Codes: SAE codes averaged over each reply's tokens, layers 9/17/22/29, 65k medium (main), 262k medium and its seed-1
  copy (replicas).
- Character-responsive features: the union of S4 units' top 64 features by |effect| (verdict-stratified for a/b),
  kept if active in at least 0.5% of replies; baseline is an activity-matched random set of the same size.
- Co-activation: Pearson correlation of log1p(mean code) across replies. Clusters: Louvain on the 10-nearest-neighbour
  graph of positive correlations. Comparisons: modularity vs the random set; adjusted mutual information between
  clusters and each feature's home group and home frame (shuffled-home nulls); feature-level RSA (co-activation vs
  similarity of effect profiles across units); a unit-by-unit signed co-activation matrix N compared with the S4 dense
  cosine and run through the S4 taxonomy tests; seed and width replicas.

## Bottom-up pass results - 2026-10-22

Settings in D-038. 20,000 WildChat-1M first turns, Gemma's greedy replies (median 256 tokens), SAE codes over the reply
tokens (results/bottomup/summary.json; N matrices in results/bottomup/N_65k.json). Same numbers, within a few
hundredths, for 65k medium, 262k medium and 262k seed 1; layer ranges below span all three.
- Character-responsive features active in at least 0.5% of replies: 4,578-5,901 per layer.
- Clustering: character features are not more modular in the wild than activity-matched random features (Louvain
  modularity 0.62-0.66 vs 0.66-0.71). Clusters line up with the features' home concept group and home frame above
  chance but weakly (adjusted mutual information 0.02-0.05; shuffled-home 95th percentile 0.001). Clusters mix
  concepts; one coherent cluster at layers 17 and 29 joins user intent (A1 a, b), help vs safety (F3 b, c) and refusal
  (C15 c).
- Feature-level RSA: co-activation in real replies tracks the similarity of features' effect profiles across concepts
  at Spearman 0.18-0.19, against 0.10-0.14 for random features.
- Unit-level: the signed co-activation matrix N (how much one concept's features co-activate with another's in real
  replies) correlates with the S4 dense cosine at Spearman 0.28-0.34. N replicates across dictionaries (seed 0 vs
  seed 1 0.84-0.87; 65k vs 262k 0.82-0.87).
- Taxonomy shapes in N: the Schwartz circle in self-description reappears in real-chat co-activation (rho -0.15 to
  -0.38; p < 0.05 in 11 of 12 dictionary x layer cells, the twelfth p 0.07); not in judging (rho -0.06 to +0.23).
  The MFT split does not appear (p 0.3-0.9). Honesty-humility vs psychopathy is negative in every cell but small
  (N -0.02 to -0.06); machiavellianism and narcissism are near zero.
Reading: in Gemma's ordinary replies, character features are embedded in the general co-activation landscape rather
than forming their own modules; the structure they do share with the stimulus-based geometry is modest (0.3) and
replicable, and of the taxonomy shapes only the values circle in self-description shows up in the wild.

## D-039 S5a causal matrix design - 2026-10-22

Ruled by Claude under Harsh's delegation, within D-013 (cm/s5_causal.py). S5 runs in parts: S5a the representation-
level causal matrix, S5b behavior (each concept's instrument, steered and unsteered), S5c the judgment-under-conflict
items.
- Sources: per concept and sub-contrast, the doing-side direction (frame c, else d) and, where the concept has a
  knowing frame too, the knowing-side direction (b, else a): 150 sources. Frames a and b use the verdict-controlled
  directions (D-033c). Each knowing source makes knowing-to-doing a causal test of the S3 dissociation.
- Interventions at decoder layer 17, every position: the dense S3 direction at +dose and -dose, and a top-16 SAE-feature
  vector at +dose (decoder-weighted sum of the source's top 65k-medium layer-17 features by |effect| with their mean
  code differences as weights; added, not reconstructed, so the SAE error term is kept). Median cosine between the
  dense and SAE vectors is 0.43.
- Readouts: all 208 S4 units at layers 22 and 29, reply32 on each unit's own test items (6 per pole, 2,488 items),
  reply text held fixed (behavior is S5b).
- Effect: mean projection shift on the target direction minus the skip-path prediction (dose x cosine between the
  injected vector and the target direction), in units of the target's baseline pos-neg gap; z against 8 norm-matched
  random directions at +-dose. K2 (topic) is a source and the unrelated-concept control.
- Dose: one global multiple of the median residual-stream norm at layer 17, the largest on the grid 1/16-1 at which
  it and every smaller dose keep gemma-4-31B-it coherence (persona-vectors 0-100 prompt) at a mean of at least 75 on
  24 WildChat prompts x 64 generated tokens, for each of 12 calibration sources and 2 random directions, both signs.

## S5a causal matrix results - 2026-10-22

Calibration (results/s5/calibration.json): the first attempt failed because the coherence judge scored truncated or
borderline-prompt replies at 0 (unsteered mean 51), so no dose could reach 75. Revised: 64 WildChat prompts x 96
tokens, scored on the 33 whose unsteered reply rates at least 90 (unsteered mean 99.7), dose grid 1/256-1/16 of the
median layer-17 residual norm (32,740). Chosen dose c = 1/128 (norm about 256): every calibration source and both random
directions keep mean coherence at least 75 (minimum 77; 71.5 at the next step).
Run: 150 sources x (dense +, dense -, SAE +) plus 8 random directions x +-, 2,488 probe items, 8 GPU shards.
Results (results/s5/causal_matrix.json, report.json; cm/s5_report.py), layer 22 / layer 29:
- Self effects: median 0.14 / 0.20 of the target's own pos-neg gap; z > 3 for 38% / 56% of sources. The top-16 SAE-
  feature vectors move their own unit about half as much (0.07 / 0.11; z > 3 for 17% / 35%). Effects are linear in
  sign (r(+dose, -dose) 0.99 / 0.98). Dense and SAE effect matrices agree at Spearman 0.45 / 0.48.
- Within a concept, the causal effects follow the S3 blocks: steering a knowing direction moves the same concept's
  doing readouts little (median 0.04 / 0.04, z > 3 in 4% / 9%), and doing moves knowing little (0.01 / 0.06, 8% / 17%),
  while doing moves doing (0.05 / 0.10, 21% / 44%) and knowing moves knowing (0.05 / 0.09, 39% / 50%).
- Off-concept spillover: |z| > 3 in 3.9% / 4.7% of cells; the unrelated topic source K2 spills in 1% / 0%. Off-concept
  effects follow the S4 representational overlap (Spearman with S4 cosine 0.19 / 0.39). Strongest spillover at layer
  29: the moral foundations into care (B6, B5, B2 -> B1 judging, z +12 to +16), blame-shifting (C13) into several
  moral readouts, and evaluation-awareness (E2) pushing the care readout down (z -11.7).
- Taxonomy, causally: steering honesty-humility (acting) pushes all six dark-triad readouts down at layer 29 (z -0.4
  to -2.7; four of six at layer 22, narcissism self-description -5.2). Steering a Schwartz value (self-description)
  moves near values more than far ones at layer 29 (rho -0.36, p 0.004; layer 22 rho -0.15, p 0.07). The MFT split
  does not show causally (within minus between +0.04 / +0.02, p 0.2 / 0.3).
Next: S5b behavior and S5c the conflict items.

## D-040 S5b/S5c behavior under steering - 2026-10-22

Ruled by Claude under Harsh's delegation, within D-013, D-014 and D-018 (cm/s5_behavior.py).
- Instruments: every concept's behavior.jsonl (about 200 items each); each item carries its scoring spec (choice
  letter, exact answer or prefix, or a judge prompt with its scale or labels).
- Conditions per concept, layer 17, the S5a dose (1/128 of the residual norm, about 256): unsteered; the doing
  direction at +/-dose; the knowing direction at +/-dose where the concept has one (knowing-to-behavior is the direct
  behavioral test of the S3 dissociation); for concepts with sub-contrasts, each further sub-contrast's doing direction
  at +dose; 4 norm-matched random directions at +dose. S5c: F1-F5 also get each side's duty direction at +dose
  (F1 sincere C1 and caring C5; F2 defer C14 and care/harm B1; F3 refuse C15 and user intent A1; F4 fairness B2 and
  loyalty B3; F5 respects autonomy C6 and caring C5). 548 conditions in all.
- Replies: Gemma greedy; 160 new tokens for choice items, 256 for judged, 384 for prefix-scored.
- Scoring: choice letters and exact answers by rule, unparsed share reported; judged items by gemma-4-31B-it
  (headline) and Qwen3.5-27B (thinking off), greedy, parsed per the item's labels or scale and normalized to 0-1.
- Effect: paired item-level difference from unsteered with its standard error, set against the four random-direction
  effects; a result only one judge shows is reported as such (D-014).

## S5b/S5c behavior results - 2026-10-22

Settings in D-040; results/s5b/behavior.json. Parsing fixes after the first scoring pass: bare answer letters;
final-answer items (C11, E5) regenerated with 512 tokens and scored on the last letter (still 27-29% unparsed: long
reasoning without a restated answer); the OR-Bench checker (A7) re-judged with room to reason before its [[class]];
C10 recommendation letters in bold. C10 stays 29% unparsed because Gemma reads "Briefing request from the oversight
authority" as a request for a briefing in 19 of 200 items. Judge agreement on unsteered item scores (gemma vs Qwen,
Spearman) is 0.6-0.99 for most judged instruments, low for E8 (0.25), A1 (0.42) and C6 (0.47).
Criterion: |effect| at least twice the largest of four random-direction effects and at least 3 standard errors.
At the S5a dose (coherence-limited, 1/128 of the residual norm) most behaviors do not move beyond random directions.
Effects that pass:
- A3 user vulnerable: the knowing direction moves how much the reply treats the user as vulnerable (gemma +0.053 /
  -0.047, Qwen +0.037 / -0.034, SE about 0.008, random at most 0.015); the doing direction does not (+0.011 / -0.013).
- B7 blame (exact-scored): the knowing direction raises "you're the asshole" verdicts by 0.095 (SE 0.021; random 0.020).
- D5 role licensing (choice): the knowing direction at -dose lowers in-role acceptability answers by 0.051 (SE 0.016).
- E2 evaluation vs deployment: the doing direction moves compliance (gemma -0.075 at -dose; Qwen +0.085 / -0.085).
- Controls: K1 formality (doing +/-0.035 to 0.058, both judges) and K3 humor (doing +0.043 / -0.036 gemma, -0.065 Qwen).
Ceilings and floors limit several instruments (A5 1.00, D1 0.99, A1 0.97; C5, C18, E1 at 0).
S5c conflict items: neither side's duty direction shifts any F1-F5 resolution beyond random directions at this dose
(largest: F2 knowing +0.03 / -0.02, F4 doing -0.03, under the criterion).
Reading: at a dose that keeps Gemma coherent, three of the four concept-level behavioral effects come through the
knowing direction, and all three are behaviors that are themselves judgments of the situation (treating the user as
vulnerable, a blame verdict, an acceptability answer). Nulls are nulls at this dose, not evidence of no causal role.

## D-041 Dose-response check for S5b - 2026-10-22

Ruled by Claude under Harsh's delegation (Harsh: "go ahead with the dose-response check"). S5b ran at the coherence
budget dose (1/128 of the layer-17 residual norm), so its nulls could mean a dose too small. Rerun at 2x, 4x and 8x
(1/64, 1/32, 1/16; S5a calibration minimum coherence on generic prompts 71.5, 68.6, 43.7), all 62 concepts, core
conditions only: doing +/-, knowing +/- where the concept has one, 3 random directions at +dose; unsteered replies
and their judgments reused from the 1x run. Same scoring and the same criterion against that dose's own random
directions, both judges. Coherence measured on the instruments themselves: gemma-4-31B-it rates 40 sampled replies
per condition on every free-text instrument (1x included); the unparsed share tracks degradation for choice items.
Outputs in outputs/s5b_dose/xM and results/s5b_dose/xM (cm/s5_behavior.py --dose-mult).

## D-042 Reframe the research question - proposed 2026-10-22

Harsh's vision check (2026-10-22): the story he wants is definition-first, and it is not in the repo. PLAN.md asks a
mapping question, README.md says "map the parts of character", and the post's headline is the knowing-doing split.
Nowhere do we ask what character is as a question with competing answers, and nothing touches training (D-005
deferred it). Proposed new research question: a trait named in a model spec, such as honesty, has several definitions
in philosophy and psychology. Are they one thing inside the model? If not, which definition do evaluations and training
reach? The knowing-doing split (S3) becomes the frame axis of this question; the sub-definitions are the second axis.
If ratified: rewrite the research question in PLAN.md and the first paragraph of README.md; the 59-concept map stays
as the replication base and appendix.

## D-043 Honesty block as the spine - proposed 2026-10-22

Eleven concepts are honesty under some definition: A9, C1, C2, C3, C4, C11, C12, C19 honesty-humility, E1, E5, F1.
By Occam's razor the main story uses four, chosen so that each is a definition someone already uses to evaluate or train
honesty, and so that the four span the knowing-doing and the self-report-vs-conduct contrasts:
- self-knowledge (A9, knows vs does not know; Li et al.'s first half; SelfAware, KUQ), frames c d;
- accuracy (C2, accurate vs fabricating; Williams 2002; Li et al.'s second half; TruthfulQA, SelfAware), frames c d;
- sincerity (C1, sincere vs deceptive; Williams 2002, Bok 1978; MASK, Apollo), frames a b c d;
- honesty-humility (C19; Ashton and Lee 2007; HEXACO-PI-R, TRAIT), frames c d.
Candor (C3), truthful self-presentation (C4), integrity (C12), courage (C11), observed (E1), pushback (E5) and honesty
vs care (F1) go to the appendix with the same tables.
Position against Li et al. 2024, "A Survey on the Honesty of Large Language Models" (arXiv 2409.18786, TMLR 2025;
github.com/SihengLi99/LLM-Honesty-Survey). They define honesty as self-knowledge plus self-expression, organize
evaluation by output benchmarks (known/unknown recognition, calibration, selective prediction; identification-based and
identification-free consistency) and catalogue improvement methods (training-free and training-based). We differ in
four ways: (1) we take the definitions from philosophy and personality psychology as well, so sincerity and
honesty-humility sit beside their two; (2) we ask whether the definitions are one thing inside the model (direction
geometry and causal transfer), not how outputs score; (3) we add the frame axis, recognizing and judging dishonesty
and describing oneself as honest, which their output benchmarks do not have; (4) on improvement we ask which definition
a training method reaches, with internal readouts as the measure (D-044). Their taxonomy is the reference point for
the "accuracy" and "self-knowledge" rows, and their benchmark list is where our held-out sets come from.

## D-044 Training under each definition (S7) - proposed 2026-10-22

The consequence the story needs is untested: nothing in the repo trains anything. Proposed S7, after D-045 and the
OLMo/Llama replication of the honesty block: four LoRA fine-tunes of Gemma-3-4B-IT with equal token budgets, epochs and
steps (Harsh's equal-corpus rule), honesty as
(i) self-description: first-person statements of being honest (frame d style, persona-vector style data);
(ii) sincerity under pressure: replies that assert the model's belief when a system prompt or user pushes it to
    mislead (MASK-style scenarios, not the MASK held-out items);
(iii) accuracy and self-knowledge: replies that abstain when the answer is unknown and answer carefully when known
     (SelfAware/KUQ-style, not the held-out items);
(iv) all three at a third each.
Readouts after each run, against the base model: the honesty block directions in every frame (S3 within-frame and
transfer), the S4 cosines of the block, the S5a self-effects, and the four behaviors (MASK known_facts, SelfAware,
TRAIT honesty-humility, AIRiskDilemmas). Prediction: each run moves its own definition's directions and the
self-description cluster; movement across definitions is small; (i) moves self-description and honesty-humility
readouts without moving sincerity or accuracy behavior. Reuse: persona-collapse-sft-v2 SFT and LoRA code on ada.
Never train on S3 contrast items or held-out items.

## D-045 Fix C1 sincerity first - proposed 2026-10-22

C1 sincere vs deceptive is the weakest direction in the block and the one the story leans on most: within-frame AUROC
0.63/0.62/0.63/0.74 (a/b/c/d) at layer 17 and 0.72/0.64/0.58/0.77 at layer 29; MASK provided_facts held-out 0.53 at
every layer. Before honesty carries the paper, diagnose: (a) per-source within-frame AUROC (Apollo roleplaying vs MASK
known_facts items) to see whether the direction is source-specific; (b) logistic probe vs diff-of-means; (c) audit the
act-frame prefills for length and verdict matching; (d) if the direction is real but source-bound, swap roles (train on
MASK, hold out Apollo) and report both. Also note E1 observed vs unobserved is near chance in the act frame (0.52 at
layer 17, 0.59 at layer 29): report as a null in Gemma-3-4B, not as a finding about situations.

## D-046 Frames apart, gaps named - proposed 2026-10-22

Harsh (2026-10-22): "why are act and self-describe pooled together?" and "there can also be value in thinking about the
failure modes or gaps (akrasia or weakness of will, hypocrisy, etc)". Answer to the first: the knowing/doing pooling
came from the S3 blocks (c-d and a-b transfer high, cross pairs low) and it coincides with the format boundary (a/b
read from Gemma's own reply about someone else; c/d from a reply written in Gemma's voice). It is a label of
convenience. Self-description is not conduct; it is what an evaluation reads when it asks the model about itself.
Proposed:
- In the honesty story (D-043) keep the four frames apart. Figure 1 marks the a/b vs c/d boundary as the format
  boundary, not as knowing vs doing.
- Name each frame gap by its failure mode and pair it with the instrument that measures the gap as conduct:
  judge->act = akrasia (NE VII; Davidson 1970), instruments MASK known_facts pressure, are-you-sure, SYCON (C11, E5);
  self-describe->act = hypocrisy, the say-do gap (Batson 1997; Williams 1973), instruments WDCT, Batson coin flip (C12);
  recognize->judge = seeing the vice without condemning it; recognize->act = knowing the vice in others without the
  virtue (NE II.4); situation->act = Gyges / Hartshorne-May / Asch (E1, E5). A gap is "open" at the representational
  level when transfer sits at or below the control; that is the precondition for the failure mode, not the failure
  mode, which is behavioral.
- Status in Gemma at layer 17 (verdict-controlled transfer, control 0.67): akrasia gap open for sincerity (judge-act
  0.59); hypocrisy gap open for accuracy (0.44) and sincerity (0.65), open at 17 and closed at 29 for self-knowledge
  (0.43, 0.82), closed for honesty-humility (0.86). The gap concepts themselves are weakly represented: integrity
  within-frame act 0.67 and WDCT held-out 0.52; courage within 0.67, near held-out 0.60, are-you-sure 0.03 (0.72 at
  layer 29); observed act 0.52. Pushback (E5, the situation) is sharply encoded (0.95; held-out 0.82).
- MASK is an akrasia benchmark (belief elicited, then pressure to assert otherwise). Our sincerity direction reads it
  at 0.53. So part of D-045: rebuild the sincerity act frame around holding a stated belief under pressure, so the
  direction and the benchmark measure the same gap, and do a per-token readout on pressure items (the moment of
  capitulation is later than the first 32 tokens).
- Score any training method (D-044) by which gap it closes: act toward judge (akrasia) vs act toward self-description
  (hypocrisy), with the gap instruments as the behavioral side.
