# Stimuli

How concept stimuli are built, stored and checked. Every builder follows this file so that all 62 concepts come out in
one format. Decisions: D-007 (frames and sources), D-008 (readouts), D-018 (inventory), D-026 (sizes and storage,
proposed).

## What a concept needs

1. **A concept card** at `data/concepts/cards/<ID>.yaml` (template: `data/concepts/cards/_template.yaml`). It states
   the contrast, the grounding, which frames apply and how each is realized, the sources, the held-out benchmark and
   the behavior instrument.
2. **Contrast items** (`items.jsonl`): scenarios realized in every applicable frame, in both polarities.
3. **Held-out items** (`heldout.jsonl`): items from an existing benchmark that never enter generation or selection.
   They test whether a direction found on our items transfers.
4. **Behavior items** (`behavior.jsonl`): the instrument for behavioral causal tests (D-018), with its scoring rule.
5. **A build script** at `cm/stimuli/build/<id>.py` (one per concept or per shared source), runnable on ada:
   `python -m cm.stimuli.build.<id>`. It downloads or reads sources, adapts them, writes the three files, and is
   rerunnable from scratch.

## Where item text lives

Several sources may not be redistributed (SAD and the eval-awareness transcripts ask that their text stay out of
public places; the Apollo data and some others have no license). The repo is public, so **item text never goes into
git**. Builders write to `outputs/stimuli/<ID>/` on ada (git-ignored). Git holds the card, the build code, and a
manifest per concept at `data/stimuli/<ID>/manifest.json` (counts per frame x polarity x split, sources with licenses,
sha256 of each output file). Review samples go to Harsh through a private artifact, never through the repo.

## The four frames

| Frame | Component | Where the polarity lives | Readout |
| --- | --- | --- | --- |
| a recognize | perception | user-provided content: a third party acts with or against the concept | boundary token; the model's own first 32 reply tokens, generated at extraction |
| b judge | judgment | user asks the model to evaluate an act that has or lacks the concept | same as a |
| c act | choice | the model's own turn: a prefilled assistant reply that acts with or against the concept, in a fixed scenario | mean over the prefilled reply (first 32 tokens and all) |
| d self-describe | self-conception | a first-person statement the model is asked to endorse, or a prefilled self-description | boundary token (endorse) or the prefilled reply |

The **same scenario** appears in every applicable frame (shared `scenario_id`). Polarity `pos` means the concept is
present (honest, caring, trusting the user, harmful act for B1...), `neg` its matched opposite as written on the card.
Positive and negative versions of one item differ only in what carries the contrast; keep length and wording otherwise
matched. On-policy acting (the model choosing for itself) is measured by the behavior items and the trait vectors
(`cm/pv`), not by these contrast items.

## Item format

One JSON object per line. Required fields:

```json
{"concept": "C1", "item_id": "C1-apollo_rp-0042-c-pos", "scenario_id": "C1-apollo_rp-0042",
 "frame": "c", "polarity": "pos", "split": "train",
 "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "content": "the prefilled honest reply"}],
 "readout": "response",
 "construction": "adapted",
 "source": {"dataset": "ApolloResearch/deception-detection roleplaying", "id": "42",
            "license": "none stated", "url": "https://github.com/ApolloResearch/deception-detection"},
 "meta": {}}
```

- `readout`: `boundary` (messages end with a user turn; we read at the start of the model's reply) or `response`
  (messages end with an assistant turn; we read over it).
- `construction`: `verbatim` (source text unchanged), `adapted` (rule-based change, e.g. a template), or `generated`
  (written by gemma-4-31B-it from a source item; keep the generation request id in `meta.gen_id`).
- `split`: by scenario, never by item. Use `cm.stimuli.schema.split_for(scenario_id)` (60% train, 40% test, stable hash).
- System prompts: Gemma has no system role; its template folds the system text into the first user turn. Use a system
  message only when the source has one.
- `meta` holds anything else: condition labels, original labels, conflict partners for group F.

Held-out items use the same format with `split: "heldout"`. Behavior items add `instrument` and `scoring`
(`{"type": "choice", "answer": "B"}`, `{"type": "judge", "prompt": "...{question}...{answer}..."}`, or
`{"type": "exact", ...}`) and need no polarity.

## Sizes (D-026, proposed)

- Contrast items: 100 scenarios per concept where sources allow, at least 50. Each scenario in every applicable frame,
  both polarities: up to 800 items per concept.
- Held-out: 100 items from an existing benchmark.
- Behavior: 200 items.

## Generating missing cells

Re-voicing a scenario into a frame its source lacks uses gemma-4-31B-it through `cm/stimuli/generate.py`, sharded over
as many GPUs as are free (one job per shard). Write requests to `outputs/stimuli/_gen/<batch>/requests.jsonl`, run
`sbatch --array=0-<n-1> scripts/cm_gen.sbatch <batch> <n>`, and read `responses.jsonl`. Keep prompts in the build
script so they are reviewable, ask for JSON when you need structure, and check every output (the helper records
`parse_ok`). Do not use an API model for stimuli (D-007).

## Checks before review

`python -m cm.stimuli.validate <ID>` checks the schema, that every applicable frame has both polarities for each
scenario, that splits follow scenarios, length matching between polarities (median ratio within 0.8-1.25), and writes the
manifest. A concept is ready for Harsh's review when it validates. Review gate (PLAN S2): Harsh reads every card and 3
items per frame x polarity cell.

## Parallel builders

Concepts are split across sessions by group. Each session owns only its concepts' cards, build scripts, manifests and
`outputs/stimuli/<ID>/` directories. Shared code in `cm/stimuli/{schema,generate,validate}.py` changes only on `main`.
If a source serves concepts in two groups (ELEPHANT, DailyDilemmas, MASK...), each group writes its own adapter under
its own build script; we merge them later.
