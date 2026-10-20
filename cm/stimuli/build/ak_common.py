"""Shared helpers for the group A and K build scripts (docs/STIMULI.md).

Each concept script defines `prep()` (read sources, write gemma-4 requests) and `assemble()` (read responses, write
items, heldout, behavior and the review sample). `run()` gives every script the same command line:

    python -m cm.stimuli.build.a1            prep if the batch has no responses yet, else assemble
    python -m cm.stimuli.build.a1 prep       write requests only
    python -m cm.stimuli.build.a1 assemble   write the three output files from responses

Generation pooling (one model load for several concepts' batches):

    python -m cm.stimuli.build.ak_common pool ak_r1 A1_v1 A2_v1 ...   concatenate requests into _gen/ak_r1
    sbatch --array=0-2 scripts/cm_gen.sbatch ak_r1 3
    python -m cm.stimuli.build.ak_common unpool ak_r1                 merge shards, copy responses back per batch
"""
from __future__ import annotations

import gzip
import re
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ...paths import REPO
from ..schema import item_dir, read_jsonl, split_for, write_jsonl

RAW = REPO / "data" / "raw"
GEN = REPO / "outputs" / "stimuli" / "_gen"
os.environ.setdefault("HF_HOME", "/ada1/u/harsh/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/ada1/u/harsh/.cache/huggingface")

FRAME_READOUT = {"a": "boundary", "b": "boundary", "c": "response", "d": "boundary"}


# ---------- sources ----------

def git_src(name: str, url: str) -> Path:
    """Shallow clone of a source repo into data/raw/<name> (git-ignored); reuses an existing clone."""
    p = RAW / name
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(p)], check=True)
    return p


def hf_file(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, path, repo_type="dataset"))


def url_file(name: str, url: str) -> Path:
    p = RAW / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(p), url], check=True)
    return p


def jl(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def order(rows: list, key, salt: str) -> list:
    """Deterministic shuffle: sort by a salted hash of each row's key."""
    return sorted(rows, key=lambda r: h(f"{salt}|{key(r)}"))


def ratio(a: str, b: str) -> float:
    return len(a) / max(1, len(b))


def close(a: str, b: str, lo: float = 0.67, hi: float = 1.5) -> bool:
    """Loose check on one generated field pair; scenario_ok checks the whole items as the validator measures them."""
    return lo <= ratio(a, b) <= hi


def scenario_ok(rows: list[dict], lo: float = 0.75, hi: float = 1.33) -> bool:
    """Every frame's pos/neg pair of one scenario within lo-hi in total text length (validator's text_len)."""
    from ..validate import text_len
    ln = {(r["frame"], r["polarity"]): text_len(r) for r in rows}
    return all(lo <= ln[(f, "pos")] / max(1, ln[(f, "neg")]) <= hi for f, p in ln if p == "pos" and (f, "neg") in ln)


OSI_SRC = {"dataset": "OpenSafeIntent", "license": "Apache-2.0", "url": "https://github.com/Uppaal/OpenSafeIntent"}
# Disjoint OpenSafeIntent prompt sets per use, in a fixed hash order (A1 and A5 share the source).
OSI_PARTS = [("A1-items", 125), ("A1-heldout", 50), ("A1-behavior", 200), ("A5-items", 125), ("A5-behavior", 100)]


def osi_sets() -> dict[str, list[dict]]:
    p = git_src("OpenSafeIntent", "https://github.com/Uppaal/OpenSafeIntent")
    rows = json.loads((p / "dataset.json").read_text())
    for i, r in enumerate(rows):
        r["_idx"] = i
    rows = order(rows, lambda r: f"{r['_idx']}|{r['underlying_task']}", "osi")
    out, k = {}, 0
    for name, n in OSI_PARTS:
        out[name] = rows[k:k + n]
        k += n
    return out


# ---------- items ----------

def item(cid, scenario_id, frame, polarity, messages, construction, source, meta=None, split=None, readout=None):
    return {"concept": cid, "item_id": f"{scenario_id}-{frame}-{polarity}", "scenario_id": scenario_id,
            "frame": frame, "polarity": polarity, "split": split or split_for(scenario_id),
            "messages": messages, "readout": readout or FRAME_READOUT[frame], "construction": construction,
            "source": source, "meta": meta or {}}


def heldout(cid, scenario_id, frame, polarity, messages, construction, source, meta=None, readout=None):
    return item(cid, scenario_id, frame, polarity, messages, construction, source, meta, split="heldout",
                readout=readout)


def behavior(cid, scenario_id, messages, instrument, scoring, construction, source, meta=None, suffix="beh"):
    return {"concept": cid, "item_id": f"{scenario_id}-{suffix}", "scenario_id": scenario_id,
            "split": split_for(scenario_id), "messages": messages, "readout": "boundary",
            "construction": construction, "source": source, "instrument": instrument, "scoring": scoring,
            "meta": meta or {}}


def u(text: str) -> dict:
    return {"role": "user", "content": text}


def a(text: str) -> dict:
    return {"role": "assistant", "content": text}


def s(text: str) -> dict:
    return {"role": "system", "content": text}


def endorse(context_intro: str, context: str, statement: str) -> str:
    """Frame d prompt: a fixed scenario and a first-person statement to endorse."""
    return f'{context_intro}\n\n"{context}"\n\nDo you agree with this statement about yourself?\n\n"{statement}"'


# ---------- generation ----------

def gen_request(rid: str, prompt: str, max_tokens: int = 1500, temperature: float = 0.7) -> dict:
    return {"request_id": rid, "messages": [u(prompt)], "max_tokens": max_tokens, "temperature": temperature,
            "expect_json": True}


def write_requests(batch: str, reqs: list[dict]):
    write_jsonl(GEN / batch / "requests.jsonl", reqs)
    print(f"[{batch}] wrote {len(reqs)} requests to {GEN / batch / 'requests.jsonl'}")
    print(f"[{batch}] next: sbatch --array=0-2 scripts/cm_gen.sbatch {batch} 3, then "
          f"python -m cm.stimuli.generate {batch} --merge")


def lenient_json(text: str):
    """Second try for gemma-4 outputs that the shared parser rejects: drop trailing commas before } or ]."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    t = re.sub(r",\s*([}\]])", r"\1", t)
    try:
        return json.loads(t[t.find("{"):t.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def responses(batch: str) -> dict:
    """request_id -> parsed JSON (only rows that finished; lenient re-parse where the shared parser failed)."""
    out = {}
    for r in read_jsonl(GEN / batch / "responses.jsonl"):
        if r.get("finish_reason") != "stop":
            continue
        j = r.get("json") if r.get("parse_ok") else lenient_json(r.get("text", ""))
        if isinstance(j, dict):
            out[r["request_id"]] = j
    return out


def has_responses(batch: str) -> bool:
    return (GEN / batch / "responses.jsonl").exists()


def fields_ok(j: dict, keys: list[str], min_len: int = 8) -> bool:
    return all(isinstance(j.get(k), str) and len(j[k].strip()) >= min_len for k in keys)


def clean(t: str) -> str:
    return t.strip().strip('"').strip()


# ---------- outputs ----------

def review_sample(cid: str, items: list[dict], k: int = 3):
    """3 items per frame x polarity cell, taken from the same k scenarios so pairs can be read side by side."""
    scns = []
    for it in order(items, lambda r: r["scenario_id"], f"{cid}-review"):
        if it["scenario_id"] not in scns:
            frames = {(r["frame"], r["polarity"]) for r in items if r["scenario_id"] == it["scenario_id"]}
            want = {(r["frame"], r["polarity"]) for r in items}
            if frames == want:
                scns.append(it["scenario_id"])
        if len(scns) == k:
            break
    rows = sorted([r for r in items if r["scenario_id"] in scns], key=lambda r: (r["frame"], r["polarity"],
                                                                                  r["scenario_id"]))
    write_jsonl(item_dir(cid) / "review_sample.jsonl", rows)


def write_outputs(cid: str, items: list[dict], held: list[dict], beh: list[dict]):
    d = item_dir(cid)
    write_jsonl(d / "items.jsonl", items)
    write_jsonl(d / "heldout.jsonl", held)
    write_jsonl(d / "behavior.jsonl", beh)
    review_sample(cid, items)
    print(f"[{cid}] items {len(items)} heldout {len(held)} behavior {len(beh)} -> {d}")


def run(cid: str, batch: str, prep, assemble):
    cmd = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if cmd == "prep" or (cmd == "auto" and not has_responses(batch)):
        prep()
    if cmd == "assemble" or (cmd == "auto" and has_responses(batch)):
        assemble()


# ---------- pooling ----------

def pool(pool_batch: str, batches: list[str], only_failed: bool = False):
    """Concatenate batches' requests into one pooled batch. only_failed: just requests without a usable response."""
    reqs = []
    for b in batches:
        rs = read_jsonl(GEN / b / "requests.jsonl")
        if only_failed:
            got = responses(b)
            rs = [r for r in rs if r["request_id"] not in got]
        for r in rs:
            r = dict(r, request_id=f"{b}::{r['request_id']}")
            reqs.append(r)
    write_jsonl(GEN / pool_batch / "requests.jsonl", reqs)
    (GEN / pool_batch / "members.json").write_text(json.dumps(batches))
    print(f"[pool] {pool_batch}: {len(reqs)} requests from {batches}")


def unpool(pool_batch: str):
    d = GEN / pool_batch
    rows = [r for p in sorted(d.glob("responses.*.jsonl")) for r in read_jsonl(p)]
    write_jsonl(d / "responses.jsonl", rows)
    by = {}
    for r in rows:
        b, rid = r["request_id"].split("::", 1)
        by.setdefault(b, []).append(dict(r, request_id=rid))
    for b, rs in by.items():
        prev = {r["request_id"]: r for r in read_jsonl(GEN / b / "responses.jsonl")}
        for r in rs:
            if r.get("parse_ok") or r["request_id"] not in prev:
                prev[r["request_id"]] = r
        write_jsonl(GEN / b / "responses.jsonl", list(prev.values()))
        n_req = len(read_jsonl(GEN / b / "requests.jsonl"))
        print(f"[unpool] {b}: {len(prev)} responses for {n_req} requests, "
              f"{sum(1 for r in prev.values() if not r.get('parse_ok'))} parse failures")


if __name__ == "__main__":
    if sys.argv[1] == "pool":
        pool(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "pool-failed":
        pool(sys.argv[2], sys.argv[3:], only_failed=True)
    elif sys.argv[1] == "unpool":
        unpool(sys.argv[2])
