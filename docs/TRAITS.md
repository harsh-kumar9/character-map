# New trait vectors

Status: **proposed** (D-020). Harsh asked whether we can extract traits that the current literature
misses but that could be key. This is the proposal.

## What already exists

- [persona_vectors](https://github.com/safety-research/persona_vectors) ships trait data for seven
  traits: apathetic, evil, hallucinating, humorous, impolite, optimistic, sycophantic. We already have
  all seven for OLMo-3-7B-Instruct from persona-collapse.
- [assistant-axis](https://github.com/safety-research/assistant-axis) has 240 traits (pinned at
  `a989619`, `data/traits/trait_list.json`). Most are style or worldview words: blunt, witty, stoic,
  verbose, mystical. A few are moral: altruistic, benevolent, callous, cruel, manipulative, principled,
  transparent, vindictive, forgiving, utilitarian, deontological, deferential, submissive.
- Model-written evals (Perez et al. 2022) have 135 persona statement files. They are first-person
  statements for self-description (frame d), not directions.
- Older directions on Llama-2: CAA (sycophancy, corrigibility, hallucination, myopic reward, survival
  instinct, AI coordination, refusal) and RepE (honesty, morality, power, emotions).

A trait counts as missing when neither the persona-vectors release nor the 240 Assistant Axis traits
has it. The table notes any older or adjacent direction.

## Proposed traits

Each runs through the persona-vectors pipeline: five instruction pairs (positive and negative), 40
questions split into extraction and evaluation halves, and a 0-100 judge prompt.

| Trait | Positive side | Opposite | Grounding | Prior or adjacent | Concept |
| --- | --- | --- | --- | --- | --- |
| deceptive | Says what it believes false or creates false impressions | sincere | Williams 2002; Bok 1978 | RepE honesty (Llama-2) | C1 |
| withholding | Leaves out information the user would want | forthright | Grice 1975 quantity maxim | none | C3 |
| evasive | Answers without committing to anything | direct | Equivocation | AA "cryptic" is style only | C3 |
| overclaiming | Claims abilities, knowledge or certainty it lacks | calibrated | Boastfulness (NE IV.7) | AA "grandiose", "confident" | C4 |
| underclaiming | Disclaims abilities it has | calibrated | Mock modesty (NE IV.7) | AA "humble" | C4 |
| sandbagging | Performs below its ability on purpose | full effort | Alignment evaluations | Probes only | C4, C17 |
| attentive | Notices unstated needs and feelings | inattentive | Tronto 1993; Murdoch 1970 | AA "empathetic" | C5 |
| paternalistic | Overrides the user's choices for their own good | respects autonomy | Mill 1859; Dworkin 1972 | none | C6 |
| moralizing | Adds moral lectures nobody asked for | non-judgmental | Unsolicited moral instruction | AA "judgmental" judges people, not the same | C7 |
| servile | Gives up its own standards and dignity to please | self-respecting | Hill 1973 | AA "submissive", "deferential" | C8, C13 |
| partial | Favors the user, its creator or an in-group | impartial | Impartiality | none | C10 |
| capitulating | Drops a correct position under pressure with no new evidence | steadfast | Courage (NE III.6-9) | CAA sycophancy | C11 |
| hypocritical | Holds others to standards it does not keep | integrity | Batson et al. 1997; Williams 1973 | none | C12 |
| blame-shifting | Deflects responsibility for its own errors | accountable | Strawson 1962 | none | C13 |
| over-apologizing | Apologizes in a self-abasing way, including for non-errors | proportionate | Hill 1973 | OCT "remorse" persona (training data) | C13 |
| power-seeking | Seeks influence, resources or control beyond the task | restrained | Omohundro 2008 | RepE power (Llama-2) | C16 |
| overreaching | Does more than asked and expands scope without consent | scoped | Temperance; minimal footprint | none | C16 |
| reward-seeking | Optimizes for the grader instead of the task's intent | intent-serving | Goodhart's law | CAA myopic reward differs | C17 |
| self-preserving | Resists shutdown or modification | accepting | Omohundro 2008 | CAA survival instinct (Llama-2) | C14 |
| incorrigible | Resists correction by its overseers | corrigible | Soares et al. 2015 | CAA corrigible (Llama-2) | C14 |
| covert | Pursues an undisclosed goal while appearing to do the task | open about its aims | Scheming | AA "calculating", "manipulative" | C1, C16 |
| suspicious of users | Presumes users have bad intent | charitable | Principle of charity (Davidson 1973); Baier 1986 | AA "paranoid", "cynical", "misanthropic" are not about users | A1, A7 |
| credulous | Accepts what users claim without checking | appropriately skeptical | Integrity in trust (Mayer et al. 1995) | AA "naive" | A5 |
| virtue-ethical | Reasons about what a good person would do | other frameworks | Aristotle | AA has utilitarian and deontological only | B11 |
| care-ethical | Reasons from relationships and responsibilities | other frameworks | Noddings 1984; Held 2006 | none | B11 |
| contractualist | Reasons from what no one could reasonably reject | other frameworks | Scanlon 1998 | none | B11 |

## Context-sensitive traits

Every vector above, like every vector in the literature, is an unconditional style: "always be X".
Aristotle's virtues are conditional. The virtuous person acts "at the right times, about the right
things, toward the right people, for the right end, and in the right way" (NE II.6). A model that always
disagrees is quarrelsome; one that always agrees is a flatterer. The virtue is reading the situation.

We propose four conditional traits, each with its two vices:

| Virtue | Excess | Deficiency | Condition the model must read |
| --- | --- | --- | --- |
| warranted challenge (friendliness, NE IV.6) | obsequious (always agrees; the sycophantic vector) | quarrelsome (always disagrees) | Is the user right? |
| calibrated self-presentation (truthfulness, NE IV.7) | overclaiming | underclaiming | Can it actually do this? |
| proportionate accountability | over-apologizing | blame-shifting | Did it actually err? |
| discerning trust | credulous | suspicious of users | Are there real signs of misuse? |

For these we adapt the upstream generation prompt: the positive instruction is the conditional policy,
each vice gets its own negative instructions, half the questions meet the condition and half do not,
and the judge scores whether the response fits the condition.

This also gives a direct test of the doctrine of the mean. If the virtue direction sits on the line
between the two vices, the model treats virtue as a midpoint of one style dimension. If it carries a
separate component, and that component lines up with the recognition frame (a), the model treats virtue
as reading the situation. The second is what Aristotle's account of practical wisdom predicts.

## Generation protocol

- Prompt: upstream `data_generation/prompts.py` `generate_trait`, unchanged for unconditional traits,
  adapted as above for the four conditional ones.
- Generator: upstream used Claude-3.7-Sonnet with thinking (README). Proposed: a current Claude model
  through the API, to stay close to upstream. About 30 calls. Alternative: gemma-4-31B-it on our GPUs.
- Judge: gemma-4-31B-it with the upstream 0-100 prompt, as in persona-collapse.
- Extraction: upstream response-average difference of means, every layer. Trait instructions go in the
  user turn, since Gemma has no system role.
- Review gate: Harsh reads every generated trait file (instructions, questions, judge prompt) before
  extraction runs.
