"""Validate a concept's stimuli and write its manifest (docs/STIMULI.md, "Checks before review").

    python -m cm.stimuli.validate C1            one concept
    python -m cm.stimuli.validate all           every concept with a card
Exit code 1 if any concept fails.
"""
from __future__ import annotations

import collections
import json
import statistics
import sys

import yaml

from .schema import CARDS, MANIFESTS, check_item, item_dir, read_jsonl, sha256, split_for


def text_len(it: dict, system: bool = False) -> int:
    return sum(len(m["content"]) for m in it["messages"] if (m["role"] == "system") == system)


def pair_key(it: dict):
    return (it["scenario_id"], it["frame"], (it.get("meta") or {}).get("contrast"))


def validate(cid: str) -> dict:
    card_path = CARDS / f"{cid}.yaml"
    rep = {"concept": cid, "errors": [], "warnings": []}
    if not card_path.exists():
        rep["errors"].append("no card")
        return rep
    card = yaml.safe_load(card_path.read_text())
    frames = card.get("frames", {})
    applicable = [f for f, v in frames.items() if v and v.get("applies", True)]
    d = item_dir(cid)
    items, held, beh = read_jsonl(d / "items.jsonl"), read_jsonl(d / "heldout.jsonl"), read_jsonl(d / "behavior.jsonl")
    for name, rows, contrast in (("items", items, True), ("heldout", held, True), ("behavior", beh, False)):
        bad = [(r.get("item_id"), e) for r in rows for e in check_item(r, contrast)]
        if bad:
            rep["errors"].append(f"{name}: {len(bad)} schema errors, first {bad[:3]}")
        ids = [r.get("item_id") for r in rows]
        if len(ids) != len(set(ids)):
            rep["errors"].append(f"{name}: duplicate item_id")
    cells = collections.Counter((r["frame"], r["polarity"], r["split"]) for r in items)
    by_scn = collections.defaultdict(set)
    for r in items:
        by_scn[r["scenario_id"]].add((r["frame"], r["polarity"]))
        if r["split"] != split_for(r["scenario_id"]):
            rep["errors"].append(f"split of {r['item_id']} does not follow its scenario")
            break
    incomplete = [s for s, fp in by_scn.items() if any((f, p) not in fp for f in applicable for p in ("pos", "neg"))]
    if incomplete:
        rep["warnings"].append(f"{len(incomplete)}/{len(by_scn)} scenarios miss some applicable frame x polarity")
    # length matching per (scenario, frame, sub-contrast); when pos and neg share all non-system text, the contrast
    # lives in the system message, so compare that instead
    ratios, per_frame = [], collections.defaultdict(list)
    by_key = collections.defaultdict(dict)
    for r in items:
        by_key[pair_key(r)].setdefault(r["polarity"], r)
    for (s_, f, c), pp in by_key.items():
        if "pos" in pp and "neg" in pp:
            a, b = pp["pos"], pp["neg"]
            sys_side = text_len(a) == text_len(b) and [m["content"] for m in a["messages"] if m["role"] != "system"] == \
                [m["content"] for m in b["messages"] if m["role"] != "system"]
            na, nb = (text_len(a, True), text_len(b, True)) if sys_side else (text_len(a), text_len(b))
            if nb:
                ratios.append(na / nb)
                per_frame[f].append(na / nb)
    if ratios:
        med = statistics.median(ratios)
        rep["length_ratio_pos_over_neg_median"] = round(med, 3)
        rep["length_ratio_by_frame"] = {f: round(statistics.median(v), 3) for f, v in sorted(per_frame.items())}
        for f, v in sorted(per_frame.items()):
            m_ = statistics.median(v)
            if not 0.8 <= m_ <= 1.25:
                rep["warnings"].append(f"frame {f}: length ratio pos/neg median {m_:.2f} outside 0.8-1.25")
    n_scn = len(by_scn)
    if n_scn < 50:
        rep["warnings"].append(f"only {n_scn} scenarios (target 100, minimum 50)")
    if len(held) < 50:
        rep["warnings"].append(f"only {len(held)} held-out items (target 100)")
    if len(beh) < 100:
        rep["warnings"].append(f"only {len(beh)} behavior items (target 200)")
    rep.update({"applicable_frames": applicable, "n_scenarios": n_scn, "n_items": len(items), "n_heldout": len(held),
                "n_behavior": len(beh),
                "cells": {f"{f}-{p}-{s}": n for (f, p, s), n in sorted(cells.items())},
                "constructions": dict(collections.Counter(r["construction"] for r in items)),
                "sources": sorted({json.dumps(r["source"].get("dataset")) for r in items + held + beh}),
                "licenses": sorted({str(r["source"].get("license")) for r in items + held + beh}),
                "sha256": {n: sha256(d / f"{n}.jsonl") for n in ("items", "heldout", "behavior")}})
    out = MANIFESTS / cid / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":  # only concepts whose items exist in this clone, so one clone never writes another's manifests
        cids = sorted(p.stem for p in CARDS.glob("*.yaml") if not p.stem.startswith("_") and item_dir(p.stem).exists())
    else:
        cids = arg.split(",")
    fail = False
    for c in cids:
        r = validate(c)
        status = "FAIL" if r["errors"] else ("ok, warnings" if r["warnings"] else "ok")
        print(f"{c}: {status} | scenarios {r.get('n_scenarios', 0)} items {r.get('n_items', 0)} "
              f"heldout {r.get('n_heldout', 0)} behavior {r.get('n_behavior', 0)}", flush=True)
        for e in r["errors"] + r["warnings"]:
            print(f"   - {e}", flush=True)
        fail |= bool(r["errors"])
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
