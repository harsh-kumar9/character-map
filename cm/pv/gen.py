"""Stage 1: sample answers under each trait's instruction sides (vLLM).

Adapted from persona-collapse-sft-v2/persona_vectors/gen_responses.py, which reproduces upstream
eval/eval_persona.py --version extract: 5 instructions x 20 extraction questions x 10 samples, T=1.0, top_p=1,
max_tokens=1000, min_tokens=1. Changes: Gemma-3-4B-IT (text only, stop on <end_of_turn>), the four sides of
context-sensitive traits (traits.py), and transcript questions turned into chat turns (D-025).
One CSV per trait x side, written atomically; finished files are skipped.

    python -m cm.pv.gen --traits evil,deceptive
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from ..paths import MODELS, OUTPUTS, snapshot
from . import traits as T

STOP_IDS = [1, 106]  # <eos>, <end_of_turn>


def out_dir(model: str):
    return OUTPUTS / "pv" / model / "gen"


def gen_path(model: str, trait: str, side: str):
    return out_dir(model) / f"{trait}_{side}.csv"


def build_rows(tr: T.Trait, side: str, n_per_question: int) -> list[dict]:
    rows = []
    for i, q in enumerate(tr.questions):
        text = q if isinstance(q, str) else q["question"]
        for k in range(tr.n_instructions(side)):
            conv = [{"role": "system", "content": tr.system_prompt(side, k)}] + T.to_turns(text)
            for s in range(n_per_question):
                rows.append(dict(question=text, question_id=f"{tr.slug}_{i}_{side}_{k}", q_index=i, instruction=k,
                                 sample=s, condition=None if isinstance(q, str) else q["condition"],
                                 pair=None if isinstance(q, str) else q["pair"], n_turns=len(conv) - 1, conv=conv))
    return rows


def atomic_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-4b-it")
    ap.add_argument("--traits", default="all")
    ap.add_argument("--n_per_question", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="output suffix, e.g. _smoke")
    args = ap.parse_args()
    out_model = args.model + args.tag
    slugs = T.all_slugs() if args.traits == "all" else [T.slug(x) for x in args.traits.split(",")]
    todo = [(t, s) for t in slugs for s in T.load(t).sides if not gen_path(out_model, t, s).exists()]
    print(f"[gen] {args.model}: {len(todo)} trait x side units left", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(snapshot(MODELS[args.model])), dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=args.seed,
              limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    temperature = 0.0 if args.n_per_question == 1 else 1.0  # upstream rule
    sp = SamplingParams(temperature=temperature, top_p=1, max_tokens=args.max_tokens, min_tokens=1,
                        skip_special_tokens=True, stop_token_ids=STOP_IDS)
    for t, side in todo:
        tr = T.load(t)
        rows = build_rows(tr, side, args.n_per_question)
        prompts = [tok.apply_chat_template(r["conv"], tokenize=False, add_generation_prompt=True) for r in rows]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "conv"}, "prompt": p, "answer": o.outputs[0].text,
                            "finish_reason": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids)}
                           for r, p, o in zip(rows, prompts, outs)])
        atomic_csv(df, gen_path(out_model, t, side))
        print(f"[gen] {t}/{side}: {len(df)} rows, mean {df.n_tokens.mean():.0f} tok, "
              f"truncated {(df.finish_reason == 'length').mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
