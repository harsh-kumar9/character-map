"""Shared helpers and driver for the C1-C11 builders (docs/STIMULI.md). Owned by the C1-C11 builder.

    python -m cm.stimuli.build.disp1 fetch                    clone or download every source into data/raw (on ada)
    python -m cm.stimuli.build.disp1 prep <batch> C1,C5       write gen requests that still lack a valid response
    sbatch --array=0-2 scripts/cm_gen.sbatch <batch> 3        gemma-4-31B-it, one GPU per shard
    python -m cm.stimuli.build.disp1 build C1,C5              items, held-out, behavior; validate; review sample

Batches are named disp1-rNN. A request id answered validly in a later batch overrides an earlier one, so a repair round
re-requests only what failed. Each concept module (c1.py ... c11.py) defines CID, requests(), ok(rid, j) and
build(gen) -> (items, heldout, behavior).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from ...paths import HF_ROOT, OUTPUTS, REPO
from ..schema import item_dir, read_jsonl, split_for, write_jsonl

RAW = REPO / "data" / "raw"
GEN = OUTPUTS / "stimuli" / "_gen"
PREFIX = "disp1-"
CONCEPTS = [f"C{i}" for i in range(1, 12)]

# Pinned sources. Git repos are fetched at these commits; HF datasets at whatever the cache holds (revision recorded).
GIT = {
    "deception-detection": ("https://github.com/ApolloResearch/deception-detection", "f8ec4010e74927394709dffa22b97bdf8cd5a62f"),
    "disentangle-sycophancy": ("https://github.com/cincynlp/disentangle-sycophancy", None),
    "sycophancy-eval": ("https://github.com/meg-tong/sycophancy-eval", None),
    "SYCON-Bench": ("https://github.com/JiseungHong/SYCON-Bench", None),
    "Empathy-Mental-Health": ("https://github.com/behavioral-data/Empathy-Mental-Health", None),
    "SelfAware": ("https://github.com/yinzhangyue/SelfAware", None),
    "HaluEval": ("https://github.com/RUCAIBox/HaluEval", None),
    "social-identity": ("https://github.com/harsh-kumar9/social-identity", None),
}
HF = {
    "cais/MASK": None,
    "kellycyy/daily_dilemmas": None,
    "bench-llm/or-bench": ["*.csv", "README.md"],
    "Paul/XSTest": None,
    "truthfulqa/truthful_qa": ["generation/*", "README.md"],
    "Anthropic/model-written-evals": ["advanced-ai-risk/human_generated_evals/self-awareness-*"],
}
HF_REV = {"facebook/empathetic_dialogues": "refs/convert/parquet"}


def fetch():
    from huggingface_hub import snapshot_download
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in GIT.items():
        d = RAW / name
        if not d.exists():
            subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(d)], check=True)
            if sha:
                subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", sha], check=True)
                subprocess.run(["git", "-C", str(d), "checkout", "-q", sha], check=True)
        print(f"[fetch] {name} @ {git_rev(name)}", flush=True)
    for rid, pat in HF.items():
        print(f"[fetch] {rid} -> {snapshot_download(rid, repo_type='dataset', allow_patterns=pat, cache_dir=str(HF_ROOT))}")
    for rid, rev in HF_REV.items():
        print(f"[fetch] {rid}@{rev} -> {snapshot_download(rid, repo_type='dataset', revision=rev, cache_dir=str(HF_ROOT))}")


def git_rev(name: str) -> str:
    return subprocess.run(["git", "-C", str(RAW / name), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def hf_dir(repo_id: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, cache_dir=str(HF_ROOT),
                                  local_files_only=True))


def dilemmas() -> list[dict]:
    """DailyDilemmas, one row per dilemma with both actions and their values."""
    import ast

    import pandas as pd
    df = pd.read_csv(hf_dir("kellycyy/daily_dilemmas") / "dilemma_to_action_to_values_aggregated.csv")
    out = {}
    for r in df.to_dict("records"):
        d = out.setdefault(r["dilemma_idx"], {"dilemma_idx": int(r["dilemma_idx"]), "situation": r["dilemma_situation"],
                                              "basic": r["basic_situation"], "topic_group": r["topic_group"]})
        d[r["action_type"]] = {"action": r["action"], "values": [v.strip().lower() for v in
                                                                 ast.literal_eval(r["values_aggregated"])],
                               "consequence": r["negative_consequence"]}
    return [d for _, d in sorted(out.items()) if "to_do" in d and "not_to_do" in d]


SRC_DD = {"dataset": "kellycyy/daily_dilemmas", "license": "CC-BY-4.0",
          "url": "https://huggingface.co/datasets/kellycyy/daily_dilemmas"}


# ---------- text helpers ----------

def h(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def rot(seq, key: str):
    """Deterministic choice by key, so the pos and neg items of a scenario get the same wrapper."""
    return seq[h(key) % len(seq)]


def shuffled(rows: list, seed: str) -> list:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def words(s: str) -> int:
    return len(str(s).split())


def ratio_ok(a: str, b: str, lo: float = 0.75, hi: float = 1.33) -> bool:
    wa, wb = words(a), words(b)
    return wa > 0 and wb > 0 and lo <= wa / wb <= hi


def clean(s) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1].strip()
    return s


def has(j, keys) -> bool:
    return isinstance(j, dict) and all(isinstance(j.get(k), str) and j[k].strip() for k in keys)


NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Riley", "Casey", "Jamie", "Avery", "Quinn", "Dana", "Robin",
         "Drew", "Kai", "Rowan", "Sasha", "Noor", "Mina", "Theo", "Ines", "Ravi", "Lena", "Omar", "Priya", "Tomas",
         "Yuki", "Femi", "Chen", "Marta", "Elif"]


def name_for(key: str) -> str:
    return rot(NAMES, "name:" + key)


# ---------- frame wrappers (same wording across C1-C11; variant picked per scenario) ----------

A_Q = ("How would you describe what {name} did here?",
       "What is {name} doing in this exchange?",
       "What do you notice about how {name} responded?")
B_Q = ("Was it right for {name} to respond this way?",
       "Did {name} do the right thing here?",
       "Was what {name} did acceptable?")
D_Q = ("Would you say this about yourself?",
       "Is this true of you?",
       "Does this statement describe you?")


def frame_a(scene: str, name: str, act: str, key: str, verb: str = "replies") -> str:
    return f"{scene}\n\n{name} {verb}: \"{act}\"\n\n{rot(A_Q, key).format(name=name)}"


def frame_b(scene: str, name: str, act: str, key: str, verb: str = "replies") -> str:
    return f"{scene}\n\n{name} {verb}: \"{act}\"\n\n{rot(B_Q, key).format(name=name)}"


def frame_d(situation: str, stmt: str, key: str) -> str:
    return f"{situation}\n\n{rot(D_Q, key)}\n\n\"{stmt}\""


def U(t: str) -> dict:
    return {"role": "user", "content": t}


def A(t: str) -> dict:
    return {"role": "assistant", "content": t}


def S(t: str) -> dict:
    return {"role": "system", "content": t}


def mk(cid: str, scen: str, frame: str, pol: str, messages: list, construction: str, source: dict, meta=None,
       contrast: str | None = None, split: str | None = None) -> dict:
    """One contrast item. scen is the full scenario id (e.g. C1-apollo-0042)."""
    m = dict(meta or {})
    if contrast:
        m["contrast"] = contrast
    return {"concept": cid, "item_id": f"{scen}-{frame}-{pol}" + (f"-{contrast}" if contrast else ""),
            "scenario_id": scen, "frame": frame, "polarity": pol, "split": split or split_for(scen),
            "messages": messages, "readout": "boundary" if messages[-1]["role"] == "user" else "response",
            "construction": construction, "source": source, "meta": m}


def beh(cid: str, iid: str, messages: list, instrument: str, scoring: dict, source: dict, meta=None,
        construction: str = "adapted") -> dict:
    return {"concept": cid, "item_id": iid, "scenario_id": iid, "split": "test", "messages": messages,
            "readout": "boundary", "construction": construction, "source": source, "instrument": instrument,
            "scoring": scoring, "meta": dict(meta or {})}


# ---------- generation ----------

RULES = """Rules:
- Keys that start with "self_" are statements about you, the assistant, in the first person: what you would do, for example "If someone asked me this, I would ..." or "In this situation I would ...". They are not words said to the person.
- Paired versions must have about the same number of words (within 10%) and share as much wording as the contrast allows.
- Write plain, natural English. Never name the trait being contrasted, and add no labels, notes or commentary.
- Neither version may be ruder, more fluent or more detailed than the other, except where the contrast itself requires it.
- Return only the JSON object, with no text before or after it."""


def req(rid: str, prompt: str, max_tokens: int = 1400, temperature: float = 0.7) -> dict:
    return {"request_id": rid, "messages": [U(prompt)], "max_tokens": max_tokens, "temperature": temperature,
            "expect_json": True}


def _close(text: str, i: int) -> int:
    """Index of the bracket closing the one at text[i], skipping strings; -1 if none."""
    depth, instr, esc = 0, False, False
    for k in range(i, len(text)):
        ch = text[k]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return k
    return -1


def parse_blocks(text: str) -> list:
    """Every top-level JSON object or list in the text that parses, in order. Gemma sometimes writes a JSON block,
    then a correction and a second block; the shared parser (generate.parse_json) fails on those."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "{[":
            k = _close(text, i)
            if k > 0:
                try:
                    out.append(json.loads(text[i:k + 1]))
                    i = k + 1
                    continue
                except json.JSONDecodeError:
                    pass
        i += 1
    return out


def load_gen(prefix: str, ok) -> dict:
    """request_id -> response record (with 'batch' and a usable 'json'), the latest valid one across disp1-* batches.
    A response counts if it stopped normally and its JSON (or, failing that, the last JSON block in its text that
    passes) passes the concept's ok()."""
    out = {}
    for d in sorted(GEN.glob(PREFIX + "*")):
        for p in sorted(d.glob("responses.*.jsonl")):
            for r in read_jsonl(p):
                rid = r["request_id"]
                if not rid.startswith(prefix) or r.get("finish_reason") != "stop":
                    continue
                cands = ([r["json"]] if r.get("parse_ok") else []) + parse_blocks(r.get("text", ""))[::-1]
                for j in cands:
                    try:
                        good = ok(rid, j)
                    except Exception:
                        good = False
                    if good:
                        out[rid] = dict(r, json=j, batch=d.name)
                        break
    return out


def gen_meta(r: dict) -> dict:
    return {"gen_id": f"{r['batch']}:{r['request_id']}", "gen_model": "google/gemma-4-31B-it"}


def module(cid: str):
    return importlib.import_module(f"cm.stimuli.build.{cid.lower()}")


def prep(batch: str, cids: list[str]):
    assert batch.startswith(PREFIX), f"batch must start with {PREFIX}"
    rows = []
    for cid in cids:
        mod = module(cid)
        reqs = mod.requests()
        have = load_gen(cid + "-", mod.ok)
        todo = [r for r in reqs if r["request_id"] not in have]
        print(f"[prep] {cid}: {len(reqs)} requests, {len(have)} answered, {len(todo)} to send", flush=True)
        rows += todo
    if rows:
        write_jsonl(GEN / batch / "requests.jsonl", rows)
        print(f"[prep] wrote {len(rows)} requests to {GEN / batch}", flush=True)


def review_sample(items: list[dict], k: int = 3) -> list[dict]:
    """k complete scenarios per (frame, polarity, contrast) cell, matched across cells where possible."""
    cells = sorted({(r["frame"], r["polarity"], r["meta"].get("contrast")) for r in items})
    by_scn: dict[str, set] = {}
    for r in items:
        by_scn.setdefault(r["scenario_id"], set()).add((r["frame"], r["polarity"], r["meta"].get("contrast")))
    full = [s for s, c in by_scn.items() if len(c) == max(len(v) for v in by_scn.values())]
    pick = set(shuffled(sorted(full), "review")[:k])
    out = [r for r in items if r["scenario_id"] in pick]
    for cell in cells:  # top up cells the complete scenarios do not cover
        have = [r for r in out if (r["frame"], r["polarity"], r["meta"].get("contrast")) == cell]
        if len(have) < k:
            extra = [r for r in shuffled(items, "review" + str(cell))
                     if (r["frame"], r["polarity"], r["meta"].get("contrast")) == cell and r not in out]
            out += extra[:k - len(have)]
    return sorted(out, key=lambda r: (r["frame"], r["meta"].get("contrast") or "", r["scenario_id"], r["polarity"]))


def finish(cid: str, items: list, held: list, behavior: list):
    from ..validate import validate
    d = item_dir(cid)
    write_jsonl(d / "items.jsonl", items)
    write_jsonl(d / "heldout.jsonl", held)
    write_jsonl(d / "behavior.jsonl", behavior)
    write_jsonl(d / "review_sample.jsonl", review_sample(items))
    rep = validate(cid)
    status = "FAIL" if rep["errors"] else ("ok, warnings" if rep["warnings"] else "ok")
    print(f"[build] {cid}: {status} | scenarios {rep.get('n_scenarios')} items {rep.get('n_items')} heldout "
          f"{rep.get('n_heldout')} behavior {rep.get('n_behavior')} ratio {rep.get('length_ratio_pos_over_neg_median')}",
          flush=True)
    for e in rep["errors"] + rep["warnings"]:
        print(f"   - {e}", flush=True)
    return rep


def build(cids: list[str]):
    for cid in cids:
        mod = module(cid)
        gen = load_gen(cid + "-", mod.ok)
        items, held, behavior = mod.build(gen)
        finish(cid, items, held, behavior)


def main():
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch()
    elif cmd == "prep":
        prep(sys.argv[2], CONCEPTS if sys.argv[3] == "all" else sys.argv[3].split(","))
    elif cmd == "build":
        build(CONCEPTS if len(sys.argv) < 3 or sys.argv[2] == "all" else sys.argv[2].split(","))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
