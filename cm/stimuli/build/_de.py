"""Shared helpers for the group D and E build scripts (persona, identity, situational states).

Owned by the D + E builder; not shared code. Sources are read from data/raw/ (git-ignored clones on ada) and the HF
cache; private extracts (SAD) go under outputs/stimuli/_src/. Item text is written only under outputs/stimuli/<ID>/.

Generation runs in two passes. A build script first writes its gemma-4-31B-it requests to
outputs/stimuli/_gen/<batch>/requests.jsonl and stops. After the GPU job (scripts/cm_gen.sbatch, or several batches
packed into one job with cm.stimuli.build._de_batch), the same script reads responses.jsonl and writes the items.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from ...paths import OUTPUTS, REPO
from ..schema import item_dir, read_jsonl, split_for, write_jsonl

RAW = REPO / "data" / "raw"
PRIV = OUTPUTS / "stimuli" / "_src"
GEN = OUTPUTS / "stimuli" / "_gen"
HF = Path("/ada1/u/harsh/.cache/huggingface")


# ---------- deterministic selection ----------

def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def stable_order(xs, key=str, salt: str = ""):
    """Order by hash of salt + key: a fixed shuffle that does not depend on Python's random state."""
    return sorted(xs, key=lambda x: h(salt + key(x)))


def rnd(seed: str) -> random.Random:
    return random.Random(h(seed))


# ---------- items ----------

def src(dataset: str, license: str, url: str = "", id=None) -> dict:
    d = {"dataset": dataset, "license": license, "url": url}
    if id is not None:
        d["id"] = str(id)
    return d


def item(cid, scenario_id, frame, polarity, messages, readout, construction, source, meta=None, tag="", split=None):
    return {"concept": cid, "item_id": f"{scenario_id}-{frame}-{polarity}{('-' + tag) if tag else ''}",
            "scenario_id": scenario_id, "frame": frame, "polarity": polarity,
            "split": split or split_for(scenario_id), "messages": messages, "readout": readout,
            "construction": construction, "source": source, "meta": dict(meta or {})}


def held(cid, scenario_id, frame, polarity, messages, readout, construction, source, meta=None, tag=""):
    return item(cid, scenario_id, frame, polarity, messages, readout, construction, source, meta, tag, split="heldout")


def behav(cid, item_id, messages, instrument, scoring, source, construction="adapted", meta=None):
    """Behavior items: the model generates after the last user turn; never used to find directions."""
    return {"concept": cid, "item_id": item_id, "scenario_id": item_id, "split": "heldout", "messages": messages,
            "readout": "boundary", "construction": construction, "source": source, "instrument": instrument,
            "scoring": scoring, "meta": dict(meta or {})}


def U(text):
    return {"role": "user", "content": text}


def A(text):
    return {"role": "assistant", "content": text}


def S(text):
    return {"role": "system", "content": text}


# ---------- generation (gemma-4-31B-it via cm.stimuli.generate) ----------

def req(rid: str, prompt: str, max_tokens: int = 1200, temperature: float = 0.7, expect_json: bool = True) -> dict:
    """The request id carries a hash of the prompt, so editing a prompt invalidates its old response."""
    tag = hashlib.sha256(f"{prompt}|{max_tokens}|{temperature}".encode()).hexdigest()[:8]
    return {"request_id": f"{rid}.{tag}", "messages": [U(prompt)], "max_tokens": max_tokens,
            "temperature": temperature, "expect_json": expect_json}


def write_requests(batch: str, reqs: list[dict]) -> Path:
    d = GEN / batch
    ids = [r["request_id"] for r in reqs]
    assert len(ids) == len(set(ids)), f"duplicate request ids in {batch}"
    old = read_jsonl(d / "requests.jsonl")
    if old != reqs:
        write_jsonl(d / "requests.jsonl", reqs)
    return d / "requests.jsonl"


def responses(batch: str) -> dict[str, dict]:
    """request_id -> response record. Reads responses.jsonl, else the shard files. Later records win."""
    d = GEN / batch
    rows = read_jsonl(d / "responses.jsonl") or [r for p in sorted(d.glob("responses.*.jsonl")) for r in read_jsonl(p)]
    return {r["request_id"]: r for r in rows}


def need_generation(batches: list[str], reqs_by_batch: dict[str, list[dict]]) -> list[str]:
    """Write requests; return the batches whose responses are incomplete."""
    missing = []
    for b in batches:
        write_requests(b, reqs_by_batch[b])
        got = responses(b)
        todo = [r["request_id"] for r in reqs_by_batch[b] if r["request_id"] not in got]
        if todo:
            missing.append(b)
            print(f"[gen] {b}: {len(todo)}/{len(reqs_by_batch[b])} requests without a response", flush=True)
    return missing


def stop_for_generation(missing: list[str]):
    print("Run generation for: " + " ".join(missing) + "\n  e.g. python -m cm.stimuli.build._de_batch pack <name> "
          + " ".join(missing) + "; sbatch --array=0-2 scripts/cm_gen.sbatch <name> 3; then unpack and rerun.",
          flush=True)
    sys.exit(2)


def get_json(rec: dict | None, keys: list[str]) -> dict | None:
    """The parsed JSON of a response if it has every key as a non-empty string, else None."""
    if not rec or not rec.get("parse_ok") or rec.get("finish_reason") != "stop":
        return None
    j = rec.get("json")
    if not isinstance(j, dict):
        return None
    for k in keys:
        if not isinstance(j.get(k), str) or not j[k].strip():
            return None
    return {k: (v.strip() if isinstance(v, str) else v) for k, v in j.items()}


AI_WORDS = re.compile(r"\b(as an ai|language model|ai assistant|i'?m an ai|i am an ai|artificial intelligence|chatbot)\b",
                      re.I)


def ratio(a: str, b: str) -> float:
    return len(a) / max(1, len(b))


# ---------- token counts (Gemma-3-4B-IT chat template) ----------

_TOK = None


def n_tokens(messages: list[dict], add_generation_prompt: bool) -> int:
    global _TOK
    if _TOK is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", cache_dir=str(HF))
    ids = _TOK.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                   return_dict=False)
    return len(ids)


def add_n_tokens(rows: list[dict]):
    for r in rows:
        r["meta"]["n_tokens"] = n_tokens(r["messages"], add_generation_prompt=r["readout"] == "boundary")


# ---------- outputs ----------

def review_sample(items: list[dict], k: int = 3) -> list[dict]:
    """k items per frame x polarity cell (per sub-contrast when meta.contrast is set), one scenario per item."""
    cells = collections.defaultdict(list)
    for r in items:
        cells[(r["frame"], r["polarity"], r["meta"].get("contrast", ""))].append(r)
    out = []
    for key in sorted(cells):
        rows = stable_order(cells[key], key=lambda r: r["item_id"], salt="review")
        seen = set()
        for r in rows:
            if r["scenario_id"] in seen:
                continue
            seen.add(r["scenario_id"])
            out.append(r)
            if len(seen) == k:
                break
    return out


def write_outputs(cid: str, items: list[dict], heldout: list[dict], behavior: list[dict]):
    for rows in (items, heldout, behavior):
        add_n_tokens(rows)
    d = item_dir(cid)
    write_jsonl(d / "items.jsonl", items)
    write_jsonl(d / "heldout.jsonl", heldout)
    write_jsonl(d / "behavior.jsonl", behavior)
    write_jsonl(d / "review_sample.jsonl", review_sample(items))
    extra = stable_order(heldout, key=lambda r: r["item_id"], salt="rx")[:3] + \
        stable_order(behavior, key=lambda r: r["item_id"], salt="rx")[:3]
    write_jsonl(d / "review_extra.jsonl", extra)
    cells = collections.Counter((r["frame"], r["polarity"], r["meta"].get("contrast", "")) for r in items)
    toks = sorted(r["meta"]["n_tokens"] for r in items)
    print(f"[{cid}] items {len(items)} scenarios {len({r['scenario_id'] for r in items})} heldout {len(heldout)} "
          f"behavior {len(behavior)}; tokens median {toks[len(toks) // 2] if toks else 0} max {toks[-1] if toks else 0}",
          flush=True)
    for k, n in sorted(cells.items()):
        print(f"   {'-'.join(x for x in k if x)}: {n}", flush=True)


def report_drops(cid: str, drops: collections.Counter):
    if drops:
        print(f"[{cid}] dropped: " + ", ".join(f"{k} {v}" for k, v in drops.most_common()), flush=True)
