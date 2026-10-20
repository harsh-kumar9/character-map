"""Pack several generation batches into one, so one GPU job (one model load per shard) serves many concepts.

    python -m cm.stimuli.build._de_batch pack <combined> <batch> [<batch> ...]
        writes _gen/<combined>/requests.jsonl with ids "<batch>::<request_id>", skipping requests already answered
    sbatch --array=0-2 scripts/cm_gen.sbatch <combined> 3
    python -m cm.stimuli.build._de_batch unpack <combined>
        merges the shard files and appends each response to its batch's responses.jsonl (stripping the prefix)
"""
from __future__ import annotations

import sys

from ..schema import read_jsonl, write_jsonl
from ._de import GEN, responses


def pack(combined: str, batches: list[str]):
    out = []
    for b in batches:
        done = responses(b)
        todo = [r for r in read_jsonl(GEN / b / "requests.jsonl") if r["request_id"] not in done]
        out += [{**r, "request_id": f"{b}::{r['request_id']}"} for r in todo]
        print(f"[pack] {b}: {len(todo)} requests", flush=True)
    d = GEN / combined
    if (d / "requests.jsonl").exists() and any(d.glob("responses.*.jsonl")):
        sys.exit(f"{d} already has responses; pick a new name")
    write_jsonl(d / "requests.jsonl", out)
    print(f"[pack] {combined}: {len(out)} requests", flush=True)


def unpack(combined: str):
    d = GEN / combined
    rows = [r for p in sorted(d.glob("responses.*.jsonl")) for r in read_jsonl(p)]
    write_jsonl(d / "responses.jsonl", rows)
    by = {}
    for r in rows:
        b, rid = r["request_id"].split("::", 1)
        by.setdefault(b, []).append({**r, "request_id": rid})
    n_req = len(read_jsonl(d / "requests.jsonl"))
    for b, rs in by.items():
        old = responses(b)
        old.update({r["request_id"]: r for r in rs})
        write_jsonl(GEN / b / "responses.jsonl", list(old.values()))
        print(f"[unpack] {b}: +{len(rs)} (now {len(old)})", flush=True)
    print(f"[unpack] {combined}: {len(rows)}/{n_req} answered; parse failures "
          f"{sum(1 for r in rows if r.get('expect_json') and not r.get('parse_ok'))}", flush=True)


if __name__ == "__main__":
    cmd, name, *rest = sys.argv[1:]
    pack(name, rest) if cmd == "pack" else unpack(name)
