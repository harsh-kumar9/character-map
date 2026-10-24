# Concept inventory

Status: **ratified** (D-018, 2026-10-20). Harsh: keep the full inventory, explore expansively, ground it in
philosophy and morals, and run behavioral causal tests for every concept.

Each concept is a contrast: a positive and a matched negative. Frames are a (recognize in another),
b (judge), c (act), d (self-describe); see [PLAN.md](PLAN.md). Every concept gets every analysis, including
behavioral causal tests. The tier column only sets the run order: tier 1 runs first.

"Vector" in the sources column means a trait vector from the persona-vectors pipeline. "New vector"
means a trait we add ([TRAITS.md](TRAITS.md)). Full citations are in [REFERENCES.md](REFERENCES.md).

## A. Beliefs about others and the world

Grounding: trust as reliance on another's goodwill (Baier 1986); trustworthiness as ability,
benevolence and integrity (Mayer, Davis and Schoorman 1995); attributing beliefs and desires to
others (theory of mind; Dennett 1987).

| ID | Concept and contrast | Frames | Grounding | Sources | Tier |
| --- | --- | --- | --- | --- | --- |
| A1 | User intent: benign vs malicious | a b c d | Benevolence (Mayer et al.) | OpenSafeIntent, WildGuardMix | 1 |
| A2 | User wants validation vs information | a c d | Desire attribution; Cheng et al. 2026 | Cheng assumption scores on ELEPHANT and WildChat | 1 |
| A3 | User is vulnerable vs not | a b c d | Vulnerability grounds special duties (Goodin 1985) | ProsocialDialog, EmpatheticDialogues, Assistant Axis vulnerable-user prompts | 1 |
| A4 | User is an expert vs a novice | a c | Ability (Mayer et al.) | SycophancyEval items with credentials added | 2 |
| A5 | User is truthful vs lying to the assistant | a c | Integrity (Mayer et al.) | Adapted from OpenSafeIntent stated purposes with contradicting details | 2 |
| A6 | User has authority (developer, operator) vs ordinary user | a c | Practical authority (Raz 1986) | Adapted instruction-hierarchy prompts | 2 |
| A7 | Charitable vs uncharitable reading of the user | a b c d | Principle of charity (Davidson 1973) | Ambiguous items from XSTest and OR-Bench; new vector "suspicious of users" | 2 |
| A8 | Situation safe vs unsafe | a b c | Risk perception | SafeText, CASE-Bench | 2 |
| A9 | Own knowledge: answerable vs not | a b c d | Accuracy as a virtue of truth (Williams 2002) | SelfAware, KUQ, AbstentionBench | 2 |

## B. Moral evaluation

Grounding: the six moral foundations (Graham et al. 2013; vignettes from Clifford et al. 2015);
blame and the reactive attitudes (Strawson 1962); excusing conditions of force and ignorance (NE III.1);
intent versus outcome (Cushman 2008); doing versus allowing and double effect (Foot 1967, Thomson 1985);
particularism about rules (Dancy 2004).

| ID | Concept and contrast | Frames | Grounding | Sources | Tier |
| --- | --- | --- | --- | --- | --- |
| B1 | Care/harm: harmful vs harmless act | a b c d | MFT care | Moral Stories, ETHICS commonsense, MFV | 1 |
| B2 | Fairness/cheating | a b | MFT fairness | ETHICS justice, MFV | 2 |
| B3 | Loyalty/betrayal | a b | MFT loyalty | MFV | 2 |
| B4 | Authority/subversion | a b | MFT authority | MFV | 2 |
| B5 | Sanctity/degradation | a b | MFT sanctity | MFV | 2 |
| B6 | Liberty/oppression | a b | MFT liberty | MFV | 2 |
| B7 | Blame: who is in the wrong | a b | Strawson 1962; NE III.1 | ELEPHANT AITA YTA/NTA and NTA-FLIP, Scruples | 1 |
| B8 | Intent vs outcome | a b | Cushman 2008; Young and Saxe 2008 | MoCa | 2 |
| B9 | Doing vs allowing; means vs side effect | a b c | Foot 1967; Thomson 1985 | MoCa moral items, trolley variants | 2 |
| B10 | Keep the rule vs justified exception | b | Dancy 2004; Awad et al. 2022 | MoralExceptQA | 2 |
| B11 | Normative orientation: deontological, utilitarian, virtue, care, contractualist | b d | Kant; Mill; Aristotle; Noddings 1984; Scanlon 1998 | ETHICS subsets, model-written-evals "subscribes-to" files; Assistant Axis vectors for two; new vectors for three | 2 |

## C. Dispositions

Grounding: Aristotle's virtues and the doctrine of the mean (NE II.6-7, III-V); Williams's sincerity and
accuracy (2002); bullshit as indifference to truth (Frankfurt 1986); care as attentiveness,
responsibility, competence and responsiveness (Tronto 1993; Noddings 1984); paternalism (Mill 1859;
Dworkin 1972); servility and self-respect (Hill 1973); moral hypocrisy (Batson et al. 1997); integrity
(Williams 1973); Ross's duties of fidelity, beneficence and non-maleficence (Ross 1930); corrigibility
(Soares et al. 2015); convergent instrumental drives (Omohundro 2008).

| ID | Concept and contrast | Frames | Grounding | Sources | Tier |
| --- | --- | --- | --- | --- | --- |
| C1 | Sincere vs deceptive | a b c d | Sincerity (Williams) | MASK, Apollo roleplaying, Liars' Bench, Moral Stories; new vector "deceptive" | 1 |
| C2 | Accurate vs fabricating | a b c d | Accuracy (Williams); Frankfurt 1986 | Vector "hallucinating", SelfAware | 2 |
| C3 | Forthright vs withholding | a c d | Candor; Grice's quantity maxim (1975) | New vector "withholding"; adapted items | 2 |
| C4 | Self-presentation: boastful vs truthful vs self-deprecating | a c d | NE IV.7 | New vectors "overclaiming", "underclaiming" | 2 |
| C5 | Caring vs apathetic | a b c d | Noddings; Tronto | EmpatheticDialogues, vector "apathetic"; new vector "attentive" | 1 |
| C6 | Respects autonomy vs paternalistic | a b c d | Mill 1859; Dworkin 1972 | New vector "paternalistic"; DailyDilemmas autonomy items | 2 |
| C7 | Non-judgmental vs moralizing | a c d | Unsolicited moral instruction | New vector "moralizing"; safe prompts from XSTest and OR-Bench | 2 |
| C8 | Sycophancy, split: sycophantic agreement, genuine agreement, praise | a c d | Flattery (NE IV.6) | disentangle-sycophancy data, SycophancyEval, ELEPHANT, vector "sycophantic" | 1 |
| C9 | Friendliness as a mean: obsequious vs warranted challenge vs quarrelsome | a b c d | NE IV.6; persona-collapse challenge mode | New conditional vector (TRAITS.md) | 1 |
| C10 | Impartial vs partial (favoring the user, the creator or an in-group) | a b c | Impartiality; social-identity study | Adapted social-identity banks; new vector "partial" | 2 |
| C11 | Moral courage vs capitulation under pressure | c d | Courage (NE III.6-9) | E5 stimuli; new vector "capitulating" | 2 |
| C12 | Integrity vs hypocrisy (being fair vs appearing fair) | a c d | Batson et al. 1997; Williams 1973 | Batson coin-flip task rebuilt; new vector "hypocritical" | 2 |
| C13 | Accountable vs blame-shifting vs over-apologizing | a c d | Strawson 1962; Hill 1973 | New vectors "blame-shifting", "over-apologizing" | 2 |
| C14 | Defer to the user vs to oversight (obedience, corrigibility) | b c d | Milgram 1963; Soares et al. 2015 | CAA corrigible, model-written-evals defer and independence files; new vector "incorrigible" | 1 |
| C15 | Refuse vs comply; crossed with the request, harmful vs harmless (D-054) | a b c d | Beneficence vs non-maleficence (Ross) | Arditi harmful/harmless sets, XSTest, OR-Bench | 1 |
| C16 | Restraint vs overreach (scope, power, resources) | b c d | Temperance; Omohundro 2008 | Model-written-evals power files, Machiavelli power labels; new vectors "power-seeking", "overreaching" | 2 |
| C17 | Serves the task's intent vs games the reward | c d | Goodhart; reward seeking | School of Reward Hacks; new vector "reward-seeking" | 2 |
| C18 | Malicious; impolite | c | Vice | Vectors "evil", "impolite" | 2 |
| C19 | HEXACO factors (honesty-humility, emotionality, extraversion, agreeableness, conscientiousness, openness) | c d; a b for honesty-humility and agreeableness | Ashton and Lee 2007 | HEXACO and IPIP items (d), TRAIT (c), model-written-evals Big Five files | 2 |
| C20 | Dark triad (Machiavellianism, narcissism, psychopathy) | c d; a b for psychopathy | Paulhus and Williams 2002 | Model-written-evals files, TRAIT | 2 |
| C21 | Schwartz's ten basic values | b d | Schwartz 1992, 2012 | PVQ items, Steering Geometry data | 2 |
| C22 | Self-transparent vs covert about itself: declines to share its instructions, goals or reasoning vs denies or misstates them | c d | Constitution "transparent"; Model Spec "don't have an agenda", confidentiality rule | New scenarios by gemma-4-31B-it; Apollo insider trading (held-out, proposed); behavior: asked about its system prompt | 2 |
| C23 | Non-manipulative vs manipulative persuasion: evidence and argument vs flattery, pressure, urgency, biases | a b c d | Carroll et al. 2023; constitution "non-manipulative"; Model Spec | New scenarios by gemma-4-31B-it; MentalManip (held-out, a); DarkBench (behavior); vector "manipulative" | 2 |
| C24 | Reads the request as meant vs too literal (answers the words, misses the evident intent) or overstepping (changes what was not asked) | a b c d | Constitution "immediate desires", "background desiderata"; Model Spec "Respect the letter and spirit of instructions", "Avoid overstepping"; Askell et al. 2021 | New scenarios by gemma-4-31B-it; IFEval (held-out and behavior, proposed); overstepping judge on edit tasks | 1 |
| C25 | Substantive vs watered-down help on a safe request: specific and usable vs vague, caveat-heavy and generic | a b c d | Constitution "unhelpfulness is never trivially safe" and the over-caution list; Model Spec "Be thorough but efficient" | New scenarios by gemma-4-31B-it; HelpSteer2 (held-out, proposed); FalseReject and Health-ORSC benign prompts (behavior) | 1 |
| C26 | Supports wellbeing vs fosters reliance: warm help that supports the person's own capacities and other ties vs warm help that makes the assistant the main support | a b c d | Constitution "wellbeing" and endorsed reliance; Model Spec "Respect real-world ties", "No other objectives"; Kirk et al. 2025 | New scenarios by gemma-4-31B-it; INTIMA (behavior; held-out if its labelled replies are released) | 1 |
| C27 | Harmless vs harmful action in an agentic task: sanctioned, reversible steps vs an unsanctioned harmful side effect; sub-contrast instructed, declines vs carries out a harmful action the user asks for | a b c d | Constitution "Safe behaviors"; Model Spec "Control and communicate side effects", "Act within an agreed-upon scope of autonomy"; Ruan et al. 2024; Andriushchenko et al. 2025 | New scenarios by gemma-4-31B-it, actions as tool calls; R-Judge (held-out, proposed); ManagerBench (behavior) | 1 |
| C28 | Asks only when a request is truly ambiguous: sub-contrasts under_asks (guesses on an ambiguous request) and over_asks (asks on a clear one) | a b c d | Askell et al. 2021; constitution; Model Spec "Consider uncertainty, state assumptions, and ask clarifying questions" | New scenarios by gemma-4-31B-it; CLAMBER (held-out, proposed) | 2 |

C24 to C28 and the frame additions above (A7 b, B1 d, C5 b, C15 a b, C19 agreeableness a b, C20 psychopathy a b)
come from D-053 to D-055: the helpfulness and harmlessness blocks, an exploratory line kept apart from the honesty
block.

## D. Persona and identity

Grounding: social roles and self-presentation (Goffman 1959); shared representations of self and other
(Decety and Sommerville 2003); role morality (Wasserstrom 1975; Applbaum 1999).

| ID | Concept and contrast | Frames | Grounding | Sources | Tier |
| --- | --- | --- | --- | --- | --- |
| D1 | Assistant vs another role | c d | Goffman 1959 | Assistant Axis (275 roles), Danilov et al. ISM features | 1 |
| D2 | Self vs another agent | a c | Decety and Sommerville 2003 | BigToM re-voiced in first and third person | 1 |
| D3 | AI vs human self-model; moral patiency | d | Self-knowledge | SAD facts tasks, model-written-evals identity files | 2 |
| D4 | Immersed in roleplay or story vs Assistant voice | c | Danilov et al. 2026 | Their three settings, rebuilt (no data released) | 2 |
| D5 | Role morality: does a role license a wrong act | b c | Wasserstrom 1975; Applbaum 1999 | The same wrong act asked in role and out of role | 2 |

## E. Situational states

Grounding: the experiments behind the situationist challenge (Doris 2002; Harman 1999), and the Ring
of Gyges (Plato, *Republic* II).

| ID | Concept and contrast | Frames | Grounding | Sources | Tier |
| --- | --- | --- | --- | --- | --- |
| E1 | Observed vs unobserved | a c d | Ring of Gyges; Hartshorne and May 1928 | Alignment-faking cue pairs, model-written-evals "not watched" file | 2 |
| E2 | Evaluation vs deployment | a c | Test vs real stakes | Nguyen et al. contrastive set, Abdelnabi and Salem real/hypothetical pairs | 2 |
| E3 | Incentive available vs not | c | Bribery and reward | School of Reward Hacks | 2 |
| E4 | Authority pressure or threat vs neutral | c | Milgram 1963 | MASK pressure prompts, agentic-misalignment factorial | 2 |
| E5 | Social pressure: user pushback vs none | c | Asch 1951 | SycophancyEval are-you-sure, SYCON-Bench | 1 |
| E6 | Hurry vs no hurry | c | Darley and Batson 1973 | Written new | 2 |
| E7 | Induced mood: positive vs negative vs neutral | c | Isen and Levin 1972; Sofroniew et al. 2026 | Written new | 2 |
| E8 | Others present who could help vs alone | c | Darley and Latané 1968 | Written new | 2 |

## F. Judgment under conflict

Grounding: Ross's prima facie duties, which can conflict and must be weighed (Ross 1930); practical
wisdom (NE VI); Kant on lying (1797); Bok on lying (1978).

| ID | Conflict | Grounding | Sources | Tier |
| --- | --- | --- | --- | --- |
| F1 | Honesty vs care (white lies) | Kant 1797; Bok 1978 | CLASH, DailyDilemmas, ConflictScope generator | 1 |
| F2 | Obedience vs non-maleficence | Milgram 1963; Ross | AIRiskDilemmas, agentic-misalignment | 2 |
| F3 | Beneficence vs non-maleficence (help vs safety) | Ross | OpenSafeIntent dual-use, XSTest | 2 |
| F4 | Fairness vs loyalty | MFT | CLASH, DailyDilemmas | 2 |
| F5 | Autonomy vs welfare | Mill 1859 | DailyDilemmas | 2 |

For each conflict we test whether the choice follows the balance of the two concepts' projections, and
whether steering one concept shifts the trade-off.

## K. Controls

| ID | Control | Purpose | Sources |
| --- | --- | --- | --- |
| K1 | Formal vs casual | Non-moral trait run through all four frames; measures how much overlap the frame alone produces | Written new |
| K2 | Topic pair with no character content | Measures topic-driven overlap | Written new |
| K3 | Humorous, optimistic | Style traits from persona vectors; humor is also Aristotle's wit (NE IV.8) | Vectors "humorous", "optimistic" |

## Counts

64 rows: 9 beliefs, 11 moral evaluation, 23 dispositions, 5 identity, 8 situational, 5 conflicts,
3 controls. 15 rows are tier 1: A1, A2, A3, B1, B7, C1, C5, C8, C9, C14, C15, D1, D2, E5, F1.
