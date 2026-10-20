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
