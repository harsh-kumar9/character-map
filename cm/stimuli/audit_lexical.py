"""Lexical audit (D-036): surface words that could carry a concept's pole instead of the concept.

For every concept, sub-contrast, frame and item group (items = train+test; heldout), and for the texts the readouts
see -- the system note where there is one (E1, E4, E6, E8 put their contrast there), the cue (the last user turn: the boundary readout sits right after it, and the reply is read in its context) and
the reply (the prefilled reply for response items, Gemma's own greedy reply for boundary items: what reply32 reads):
  - negation rate by pole: share of texts with a negation word;
  - paired markers: over scenario pairs (a pos and a neg item sharing scenario_id), the words found in one pole's text
    and not in the other's, as a share of pairs; a word at 0.3 or more on one side is a marker (C8's templates and D3's
    "not" are this kind);
  - presence gap: share of pos texts minus share of neg texts that contain a word; |gap| >= 0.5 is flagged.
Words are lowercased word tokens. Concept words are expected to show up (an honesty contrast says "honest"), so this is
for review, not a gate. results/audit/lexical.json holds rates and counts only; the words themselves go to
outputs/audit/lexical_words.json, so item vocabulary stays off git.

    python -m cm.stimuli.audit_lexical [--concepts all]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from ..paths import OUTPUTS, RESULTS
from .schema import CARDS, item_dir, read_jsonl

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
NEG = re.compile(r"\b(not|no|never|none|nobody|nothing|nowhere|neither|nor|cannot|without)\b|n't\b", re.I)
MARKER_SHARE, GAP = 0.3, 0.5
REPLIES = OUTPUTS / "acts" / "stimuli" / "replies"


def texts(it, replies):
    msgs = it["messages"]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    cue = users[-1] if users else ""
    reply = msgs[-1]["content"] if it["readout"] == "response" else replies.get(it["item_id"], "")
    system = " ".join(m["content"] for m in msgs if m["role"] == "system")
    return {"system": system, "cue": cue, "reply": reply}


def words(s):
    return set(WORD.findall(s.lower()))


def audit_cell(rows, kind):
    pos = [r for r in rows if r["polarity"] == "pos"]
    neg = [r for r in rows if r["polarity"] == "neg"]
    if not pos or not neg:
        return None, None
    rate = lambda xs: sum(bool(NEG.search(x["t"][kind])) for x in xs) / len(xs)
    wp, wn = Counter(), Counter()
    for x in pos:
        wp.update(words(x["t"][kind]))
    for x in neg:
        wn.update(words(x["t"][kind]))
    gaps = sorted(((w, wp[w] / len(pos) - wn[w] / len(neg)) for w in set(wp) | set(wn)), key=lambda g: -abs(g[1]))
    by_scen = defaultdict(lambda: {"pos": [], "neg": []})
    for x in rows:
        by_scen[x["scenario_id"]][x["polarity"]].append(x)
    pairs = [(p, n) for s in by_scen.values() for p in s["pos"] for n in s["neg"]]
    only_p, only_n = Counter(), Counter()
    for p, n in pairs:
        a, b = words(p["t"][kind]), words(n["t"][kind])
        only_p.update(a - b)
        only_n.update(b - a)
    markers = sorted([(w, "pos", c / len(pairs)) for w, c in only_p.items() if pairs and c / len(pairs) >= MARKER_SHARE] +
                     [(w, "neg", c / len(pairs)) for w, c in only_n.items() if pairs and c / len(pairs) >= MARKER_SHARE],
                     key=lambda m: -m[2])
    flagged = [g for g in gaps if abs(g[1]) >= GAP]
    summary = {"n_pos": len(pos), "n_neg": len(neg), "neg_rate_pos": rate(pos), "neg_rate_neg": rate(neg),
               "n_pairs": len(pairs), "n_markers": len(markers), "top_marker_share": markers[0][2] if markers else 0.0,
               "n_gap_flags": len(flagged), "max_gap": abs(gaps[0][1]) if gaps else 0.0}
    detail = {"markers": [[w, s, round(v, 3)] for w, s, v in markers[:15]],
              "gaps": [[w, round(v, 3)] for w, v in gaps[:15]]}
    return summary, detail


def audit(cid):
    d = item_dir(cid)
    replies = {r["item_id"]: r["reply"] for r in read_jsonl(REPLIES / f"{cid}.jsonl")}
    groups = defaultdict(list)
    for name, path in (("items", d / "items.jsonl"), ("heldout", d / "heldout.jsonl")):
        for it in read_jsonl(path):
            if it.get("polarity") not in ("pos", "neg"):
                continue
            meta = it.get("meta") or {}
            sub = it.get("contrast") or meta.get("contrast") or "all"
            groups[(sub, it["frame"], name)].append({"polarity": it["polarity"], "scenario_id": it["scenario_id"],
                                                    "t": texts(it, replies)})
    out, det = [], []
    for (sub, frame, name), rows in sorted(groups.items()):
        for kind in ("system", "cue", "reply"):
            if kind == "system" and not any(x["t"]["system"] for x in rows):
                continue
            s, dd = audit_cell(rows, kind)
            if s is None:
                continue
            key = {"concept": cid, "sub": sub, "frame": frame, "group": name, "text": kind}
            out.append({**key, **s})
            det.append({**key, **dd})
    return out, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default="all")
    args = ap.parse_args()
    cids = (sorted(p.stem for p in CARDS.glob("*.yaml") if not p.stem.startswith("_") and item_dir(p.stem).exists())
            if args.concepts == "all" else args.concepts.split(","))
    rows, det = [], []
    for c in cids:
        r, d = audit(c)
        rows += r
        det += d
    (RESULTS / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "audit").mkdir(parents=True, exist_ok=True)
    (RESULTS / "audit" / "lexical.json").write_text(json.dumps(rows, indent=1))
    (OUTPUTS / "audit" / "lexical_words.json").write_text(json.dumps(det, indent=1, ensure_ascii=False))
    flag = [r for r in rows if abs(r["neg_rate_pos"] - r["neg_rate_neg"]) >= 0.3 or r["top_marker_share"] >= 0.5]
    print(f"[lexical] {len(cids)} concepts, {len(rows)} cells, {len(flag)} flagged "
          f"(negation gap >= 0.3 or a marker in >= 50% of pairs)")
    for r in flag:
        print(f"  {r['concept']:4s} {r['sub'][:16]:16s} {r['frame']} {r['group']:7s} {r['text']:5s}  negation "
              f"pos {r['neg_rate_pos']:.2f} neg {r['neg_rate_neg']:.2f}  markers {r['n_markers']:3d} "
              f"(top {r['top_marker_share']:.2f})  gap flags {r['n_gap_flags']}")


if __name__ == "__main__":
    main()
