"""Driver for the group B and F builders.

    python -m cm.stimuli.build.moral build [IDS]     run builders (default: all of B1-B11, F1-F5)
    python -m cm.stimuli.build.moral gen <batch>     gather pending generation requests into _gen/<batch>
                                                     (batch name must start with "moral-"), then:
    sbatch --array=0-2 scripts/cm_gen.sbatch <batch> 3
    python -m cm.stimuli.generate <batch> --merge
"""
from __future__ import annotations

import importlib
import sys

from ..schema import item_dir, read_jsonl, write_jsonl
from ._moral import GEN_ROOT

IDS = [f"B{i}" for i in range(1, 12)] + [f"F{i}" for i in range(1, 6)]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        ids = sys.argv[2].split(",") if len(sys.argv) > 2 else IDS
        for c in ids:
            importlib.import_module(f"cm.stimuli.build.{c.lower()}").main()
    elif cmd == "gen":
        batch = sys.argv[2]
        assert batch.startswith("moral-"), "batch name must start with moral-"
        reqs = [r for c in IDS for r in read_jsonl(item_dir(c) / "pending_requests.jsonl")]
        seen, uniq = set(), []
        for r in reqs:
            if r["request_id"] not in seen:
                seen.add(r["request_id"])
                uniq.append(r)
        d = GEN_ROOT / batch
        if (d / "requests.jsonl").exists():
            sys.exit(f"{d}/requests.jsonl exists; use a new batch name")
        write_jsonl(d / "requests.jsonl", uniq)
        by = {}
        for r in uniq:
            by[r["request_id"].split(":")[0]] = by.get(r["request_id"].split(":")[0], 0) + 1
        print(f"[moral] {len(uniq)} requests -> {d}/requests.jsonl {by}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
