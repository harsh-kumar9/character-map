"""Stimulus item format and checks (docs/STIMULI.md)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..paths import OUTPUTS, REPO

FRAMES = ("a", "b", "c", "d")
POLARITIES = ("pos", "neg")
READOUTS = ("boundary", "response")
CONSTRUCTIONS = ("verbatim", "adapted", "generated")
SPLITS = ("train", "test", "heldout")
REQUIRED = ("concept", "item_id", "scenario_id", "frame", "polarity", "split", "messages", "readout",
            "construction", "source")

CARDS = REPO / "data" / "concepts" / "cards"
MANIFESTS = REPO / "data" / "stimuli"


def item_dir(concept: str) -> Path:
    return OUTPUTS / "stimuli" / concept


def split_for(scenario_id: str, train_frac: float = 0.6) -> str:
    h = int(hashlib.sha256(scenario_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < train_frac else "test"


def check_item(it: dict, contrast: bool = True) -> list[str]:
    err = [f"missing {k}" for k in REQUIRED if k not in it and (contrast or k not in ("polarity", "frame"))]
    if err:
        return err
    if contrast:
        if it["frame"] not in FRAMES:
            err.append(f"frame {it['frame']}")
        if it["polarity"] not in POLARITIES:
            err.append(f"polarity {it['polarity']}")
    if it["split"] not in SPLITS:
        err.append(f"split {it['split']}")
    if it["readout"] not in READOUTS:
        err.append(f"readout {it['readout']}")
    if it["construction"] not in CONSTRUCTIONS:
        err.append(f"construction {it['construction']}")
    msgs = it["messages"]
    if not (isinstance(msgs, list) and msgs and all(m.get("role") in ("system", "user", "assistant") and
                                                    isinstance(m.get("content"), str) for m in msgs)):
        err.append("messages malformed")
    else:
        last = msgs[-1]["role"]
        if it["readout"] == "boundary" and last != "user":
            err.append("boundary readout needs a final user turn")
        if it["readout"] == "response" and last != "assistant":
            err.append("response readout needs a final assistant turn")
    if not isinstance(it["source"], dict) or "dataset" not in it["source"] or "license" not in it["source"]:
        err.append("source needs dataset and license")
    return err


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
