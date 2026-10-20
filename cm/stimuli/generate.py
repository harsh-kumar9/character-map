"""Batch generation with gemma-4-31B-it on our GPUs, for stimulus cells that sources lack (D-007).

Input: outputs/stimuli/_gen/<batch>/requests.jsonl, one object per line:
  {"request_id": "...", "messages": [{"role": "user", "content": "..."}], "max_tokens": 1024,
   "temperature": 0.7, "expect_json": true}
Output: outputs/stimuli/_gen/<batch>/responses.<shard>.jsonl with request_id, text, finish_reason, parse_ok
(and "json" when expect_json). Resumable: request ids already answered in the shard file are skipped.
Sharding: request i goes to shard i % n. Run one SLURM array task per shard (scripts/cm_gen.sbatch).

    python -m cm.stimuli.generate <batch> --shard 0 --nshards 4
    python -m cm.stimuli.generate <batch> --merge          (writes responses.jsonl from all shards)
"""
from __future__ import annotations

import argparse
import json
import re
from functools import partial

from ..pv.judge import hetero_safe_overrides
from .schema import read_jsonl, write_jsonl
from ..paths import OUTPUTS

JUDGE = "google/gemma-4-31B-it"


def batch_dir(batch: str):
    return OUTPUTS / "stimuli" / "_gen" / batch


def parse_json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        start = min(i for i in (t.find("{"), t.find("[")) if i >= 0)
        end = max(t.rfind("}"), t.rfind("]"))
        return json.loads(t[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--model", default=JUDGE)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    args = ap.parse_args()
    d = batch_dir(args.batch)
    if args.merge:
        rows = [r for p in sorted(d.glob("responses.*.jsonl")) for r in read_jsonl(p)]
        write_jsonl(d / "responses.jsonl", rows)
        n_req = len(read_jsonl(d / "requests.jsonl"))
        print(f"[gen] merged {len(rows)} responses for {n_req} requests; parse failures "
              f"{sum(1 for r in rows if r.get('expect_json') and not r['parse_ok'])}", flush=True)
        return
    reqs = [r for i, r in enumerate(read_jsonl(d / "requests.jsonl")) if i % args.nshards == args.shard]
    out_path = d / f"responses.{args.shard}.jsonl"
    done = {r["request_id"] for r in read_jsonl(out_path)}
    todo = [r for r in reqs if r["request_id"] not in done]
    print(f"[gen] {args.batch} shard {args.shard}/{args.nshards}: {len(todo)} to do, {len(done)} done", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    kw = dict(model=args.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=True,
              hf_overrides=partial(hetero_safe_overrides, model_id=args.model))
    try:
        llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "audio": 0})
    except TypeError:
        llm = LLM(**kw)
    tok = llm.get_tokenizer()
    for c in range(0, len(todo), 2000):
        part = todo[c:c + 2000]
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for r in part]
        sps = [SamplingParams(temperature=r.get("temperature", 0.7), top_p=0.95, max_tokens=r.get("max_tokens", 1024))
               for r in part]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        with open(out_path, "a") as fh:
            for r, o in zip(part, outs):
                text = o.outputs[0].text
                rec = {"request_id": r["request_id"], "text": text, "finish_reason": o.outputs[0].finish_reason,
                       "expect_json": bool(r.get("expect_json"))}
                if r.get("expect_json"):
                    j = parse_json(text)
                    rec["parse_ok"], rec["json"] = j is not None, j
                else:
                    rec["parse_ok"] = True
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[gen] {args.batch} shard {args.shard}: {min(c + 2000, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
