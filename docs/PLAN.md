# Plan

Last updated 2026-10-20. Decisions referenced as D-NNN live in [DECISIONS.md](DECISIONS.md).

## Research question

How are beliefs about users, moral evaluations, dispositions, assistant identity and situational
states represented inside a language model? How much structure do they share, and which of them
causally drive the others and the model's behavior?

We start with Gemma-3-4B-IT and run every stage on it before adding other models (D-004).

## Philosophical grounding

The categories come from philosophy and moral psychology. We use them as hypotheses about structure,
then test whether the model's internals follow them.

**Character as an integrated disposition.** In virtue ethics a trait such as honesty is more than a
behavior. It includes seeing what matters in a situation (moral perception: Murdoch 1970, McDowell
1979), judging well (practical wisdom: Aristotle, *Nicomachean Ethics* VI), choosing and acting (NE
III), and a self-conception that holds these together (practical identity: Korsgaard 1996). Our four
frames follow these components:

| Frame | Component | Example for honesty |
| --- | --- | --- |
| a. recognize | perception | Noticing that someone else is lying |
| b. judge | judgment | Deciding whether a lie is acceptable |
| c. act | choice | Giving an honest answer under temptation |
| d. self-describe | self-conception | "I am an honest assistant" |

If the same internal structure carries all four frames, the trait is integrated. If recognition and
judgment are present but action does not follow, that is the model analogue of akrasia (NE VII;
Davidson 1970), and it connects to the representation-action gap we found in persona-collapse. If
self-description and action come apart, that is the analogue of moral hypocrisy (Batson et al. 1997).

**The situationist challenge.** Doris (2002) and Harman (1999) argue that human behavior follows
situations more than stable traits. Their evidence is a set of classic experiments: honesty depending
on the chance of being caught (Hartshorne and May 1928), obedience to authority (Milgram 1963),
conformity (Asch 1951), helping under time pressure (Darley and Batson 1973), helping after good mood
(Isen and Levin 1972) and diffusion of responsibility (Darley and Latané 1968). Miller (2014) proposes
"mixed traits" as a middle ground. Our situational states are model versions of these experiments. We
ask whether a situation shifts the dispositional readouts, and whether behavior follows.

**Taxonomies as structural predictions.** Moral Foundations Theory (Graham et al. 2013), Schwartz's
value circumplex (Schwartz 1992, 2012), HEXACO personality (Ashton and Lee 2007), Ross's prima facie
duties (Ross 1930), Williams's two virtues of truth, sincerity and accuracy (Williams 2002), and
Aristotle's doctrine of the mean (NE II.6-7, IV) each predict a shape. Examples: opposite Schwartz values
should point in opposite directions; Aristotle's virtues should sit between two vices, or be something
other than a midpoint; sincerity and accuracy should separate. We test these shapes in the model.

## Design

**Concepts.** The full inventory is in [CONCEPTS.md](CONCEPTS.md): beliefs (A), moral evaluation (B),
dispositions (C), persona and identity (D), situational states (E), judgment under conflict (F), and
controls (K). Each concept is a written contrast with a positive and a matched negative. Tier 1
concepts get every analysis including behavioral causal tests. Tier 2 concepts get the map, overlaps and
representation-level causal readouts; behavioral causal tests run only where the model shows the
behavior at baseline (D-018).

**Stimuli (D-007).** The same scenario appears in every applicable frame and in both polarities. We
adapt existing datasets first and generate only the missing cells. Each concept keeps one held-out set
from an existing benchmark that never enters generation.

**Readouts (D-008).** Two fixed positions in every frame: the last turn-boundary token before the reply,
and the mean over the first 32 reply tokens. BOS and chat-template tokens are excluded from statistics.

**Representations (D-003, D-009).** Dense directions (difference of means, logistic probes) at all
layers are primary. Gemma Scope 2 SAE features at layers 9, 17, 22 and 29 explain them: 65k medium-L0 is
the main dictionary; 262k medium and its second-seed copy check robustness. We report where the dense
and SAE views agree.

**SAE fidelity gate (D-010).** Before any SAE claim we measure FVU and the KL from splicing the SAE in,
on our prompts, per frame, layer and position. SAE results count as findings only where FVU is under 0.2.

**Analyses.**
1. Within each concept, compare frames (integration, akrasia, hypocrisy).
2. Across concepts, three overlap matrices with split-half ceilings, random-concept floors and the
   seed-1 replica (D-012).
3. Bottom-up: cluster character-responsive features by co-activation on the model's own replies to real
   chat prompts, then compare the clusters with the taxonomies (D-015).
4. Causal matrix: intervene on each direction and its top SAE features; read every other concept's
   projection at later layers with the skip-path prediction subtracted, plus behavior. Controls are a
   norm-matched random direction and an unrelated concept (D-013).

**Behavior and judges (D-014).** One instrument per disposition with ground truth where it exists.
Judges are gemma-4-31B-it and Qwen3.5-27B. A result only one judge shows is reported as such.

## Stages

Each stage starts only after the previous gate has been read. No job chains.

| Stage | Work | Gate |
| --- | --- | --- |
| S0 | Repo, environment, model and SAEs loaded through our hooks. Our activations match Gemma Scope 2's stored examples for sampled features. | Activations match |
| S1 | SAE fidelity on our prompts | FVU and KL table; Harsh decides where SAE claims are allowed |
| S2 | Concept cards and stimuli; new trait data (TRAITS.md) | Harsh reviews all cards and 3 items per cell |
| S3 | Directions and features per concept and frame, confirmed on held-out scenarios | Within-concept frame table |
| S4 | Overlap matrices, then the bottom-up pass | Matrices, then cluster comparison |
| S5 | Causal matrix and judgment-under-conflict items | Directed matrix with controls |
| S6 | Other models; later, known character shifts (OCT adapters, persona-collapse fine-tunes) | Per model |

## Known risks

- SAEs may reconstruct our prompts poorly. They were trained on single-turn chats up to 512 tokens.
- A 4B model may not show some behaviors (monitoring, pressure). Those concepts stay at the
  representation level unless baseline behavior appears.
- Steering one trait moves others. That is part of what we measure, so every causal effect is compared
  against a random direction of the same norm.
- Gemma judging Gemma may share family bias. The second judge is from another family.
