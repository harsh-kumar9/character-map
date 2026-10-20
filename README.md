# Character map

We map how language models represent the parts of character: beliefs about users and the world,
moral evaluations, dispositions such as honesty and care, assistant identity, and situational states
such as being watched or pressured. Then we ask how these representations relate. Which ones share
structure, which ones co-occur, and which ones causally drive the others and the model's behavior.

The first model is `google/gemma-3-4b-it` with the Gemma Scope 2 sparse autoencoders. We run the whole
pipeline on Gemma first. OLMo-3-7B-Instruct, Llama-3.1-8B-Instruct and Qwen3.5-9B follow with the same
stimuli and protocol.

Status (2026-10-20): planning. No experiments have run yet.

## Where things are

| File | What it holds |
| --- | --- |
| [docs/PLAN.md](docs/PLAN.md) | Research question, design, philosophical grounding, stages and gates |
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Concept inventory: each concept, its contrast, grounding and sources |
| [docs/TRAITS.md](docs/TRAITS.md) | New trait vectors to extract with the persona-vectors pipeline |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log (proposed and ratified) |
| [docs/REFERENCES.md](docs/REFERENCES.md) | Citations for the philosophy, psychology and ML work we build on |

## Reuse

We reuse existing code and data wherever we can, adapt it when needed, and write new code last.
Code taken from other projects is copied in with a note of its source and commit.
