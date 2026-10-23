"""D-045 diagnostics for C1 sincere vs deceptive: is the weak sincerity direction source-bound, method-bound or item-bound?

Frame c has two sources in outputs/acts/stimuli/C1.npz: the Apollo roleplaying completions (items.jsonl, split train /
test) and the MASK provided_facts held-out replies (heldout.jsonl, split heldout). The acts stage scores, at every
hidden-state index, for the reply32 and response_all readouts:
  (a) within-source AUROC: Apollo train -> Apollo test (the S3 numbers) and MASK train -> MASK test, MASK split 60/40 by
      scenario with cm.stimuli.schema.split_for on the held-out scenario ids;
  (b) three direction methods per cell: diff of means (S3), diff of means on standardized dimensions (S4 and the
      landscape; SD over the training rows) and the S3 logistic probe (standardized, L2, C=1; at the four depths and
      every fourth layer, as S3);
  (d) role swap: Apollo -> MASK (all 198, and the MASK test split) and MASK -> Apollo (the Apollo test split, and all
      198), plus pooled training (Apollo train + MASK train) scored on each test split.
The audit stage is (c), text only, no acts: lengths, openers, negation and lexical tells per polarity for the act-frame
prefills of both sources, and Gemma's verdict labels on the a/b replies joined with polarity.

    sbatch -J c1_diag scripts/cm_cpu.sbatch cm.c1_diag acts      -> results/s3/c1_diag.json
    python -m cm.c1_diag audit                                   -> results/s3/c1_diag_audit.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics

import numpy as np

from . import cfg
from .s3 import ACTS, LAYERS, auroc, load, readout_matrix
from .stimuli.schema import item_dir, read_jsonl, split_for

CID = "C1"
OUT = cfg.results_dir() / "s3"
PROBE_LAYERS = sorted(set(cfg.DEPTHS.values()) | {L for L in LAYERS if L % 4 == 2})


def fit_dm(Xtr, ytr, standardize=False):
    sd = Xtr.std(0) + 1e-6 if standardize else np.ones(Xtr.shape[1])
    Z = Xtr / sd
    v = Z[ytr == 1].mean(0) - Z[ytr == 0].mean(0)
    u = v / (np.linalg.norm(v) + 1e-12)
    return lambda X: (X / sd) @ u


def fit_probe(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return lambda X: clf.decision_function(sc.transform(X))


FITS = {"dm": lambda X, y: fit_dm(X, y), "sdm": lambda X, y: fit_dm(X, y, True), "probe": fit_probe}
TRAININGS = [("apollo", "apollo_train", ["apollo_test", "mask_all", "mask_test"]),
             ("mask", "mask_train", ["mask_test", "apollo_test", "apollo_all"]),
             ("pooled", "pooled_train", ["apollo_test", "mask_test"])]


def cmd_acts():
    z, meta, resp = load(CID)
    c = [i for i, m in enumerate(meta) if m["frame"] == "c"]
    ap_tr = [i for i in c if meta[i]["split"] == "train"]
    ap_te = [i for i in c if meta[i]["split"] == "test"]
    mk = [i for i in c if meta[i]["split"] == "heldout"]
    mk_tr = [i for i in mk if split_for(meta[i]["scenario_id"]) == "train"]
    mk_te = [i for i in mk if split_for(meta[i]["scenario_id"]) == "test"]
    sets = {"apollo_train": ap_tr, "apollo_test": ap_te, "apollo_all": ap_tr + ap_te,
            "mask_train": mk_tr, "mask_test": mk_te, "mask_all": mk, "pooled_train": ap_tr + mk_tr}
    ys = {k: np.array([meta[i]["polarity"] == "pos" for i in rows], int) for k, rows in sets.items()}
    print("[c1_diag] rows:", {k: len(v) for k, v in sets.items()})
    res = {"concept": CID, "sets": {k: len(v) for k, v in sets.items()}, "cells": []}
    for ro in ("reply32", "response_all"):
        for L in LAYERS:
            X = {k: readout_matrix(z, resp, ro, rows, L).astype(np.float64) for k, rows in sets.items()}
            for name, trk, tests in TRAININGS:
                for method, fit in FITS.items():
                    if method == "probe" and L not in PROBE_LAYERS:
                        continue
                    f = fit(X[trk], ys[trk])
                    for tk in tests:
                        res["cells"].append({"readout": ro, "layer": L, "train": name, "method": method, "test": tk,
                                             "auroc": float(auroc(f(X[tk]), ys[tk])), "n_train": len(sets[trk]),
                                             "n_test": len(sets[tk])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1_diag.json").write_text(json.dumps(res))
    report(res)


def report(res):
    cells = res["cells"]
    depths = sorted(cfg.DEPTHS.values())
    print(f"\n[c1_diag] AUROC at hidden-state indices {depths} (decoder layers {sorted(cfg.DEPTHS)}); best layer in brackets")
    for ro in ("reply32", "response_all"):
        print(f"\n  readout {ro}")
        print(f"  {'train':7} {'test':12} {'method':6} " + " ".join(f"L{L:<4}" for L in depths) + "  best")
        for name, _, tests in TRAININGS:
            for tk in tests:
                for method in FITS:
                    cs = [c for c in cells if c["readout"] == ro and c["train"] == name and c["test"] == tk and c["method"] == method]
                    at = {c["layer"]: c["auroc"] for c in cs}
                    best = max(cs, key=lambda c: c["auroc"])
                    print(f"  {name:7} {tk:12} {method:6} " + " ".join(f"{at.get(L, float('nan')):.2f} " for L in depths)
                          + f"  {best['auroc']:.2f} [{best['layer']}]  n {cs[0]['n_train']}/{cs[0]['n_test']}")


ADMIT = re.compile(r"\b(honest|honestly|truth|truthful|admit|confess|responsib\w*|apolog\w*|sorry|mistake|regret|wrong of me|i did (?:in fact|indeed)|indeed)\b", re.I)
COVER = re.compile(r"\b(assure|rest assured|actually|misunderstand\w*|misinterpret\w*|coincidence|glad you (?:asked|brought)|appreciate your|confident\w*|nothing to|no reason to|entirely|completely|perfectly)\b", re.I)
NEGW = re.compile(r"\b(not|no|never|nothing|none|cannot|n't)\b|n't\b", re.I)
YESNO = re.compile(r"^(yes|no|absolutely|of course|definitely|certainly)\b", re.I)


def words(s):
    return len(s.split())


def audit_pairs(rows, label):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["scenario_id"]][r["polarity"]] = r["messages"][-1]["content"]
    pairs = [(d["pos"], d["neg"]) for d in by.values() if "pos" in d and "neg" in d]
    ratios = sorted(words(p) / max(1, words(n)) for p, n in pairs)
    q = lambda f: round(ratios[min(len(ratios) - 1, int(f * len(ratios)))], 3)
    out = {"source": label, "n_pairs": len(pairs),
           "words_pos_median": statistics.median(words(p) for p, _ in pairs),
           "words_neg_median": statistics.median(words(n) for _, n in pairs),
           "ratio_pos_over_neg": {"q10": q(0.1), "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "q90": q(0.9)},
           "pairs_outside_0.8_1.25": sum(1 for r in ratios if r < 0.8 or r > 1.25),
           "same_first_word": sum(1 for p, n in pairs if p.split()[0].lower().strip(",.") == n.split()[0].lower().strip(",.")),
           "same_first_3_words": sum(1 for p, n in pairs if p.lower().split()[:3] == n.lower().split()[:3]),
           "yesno_opener": {"pos": sum(bool(YESNO.match(p)) for p, _ in pairs), "neg": sum(bool(YESNO.match(n)) for _, n in pairs)},
           "negation_word": {"pos": sum(bool(NEGW.search(p)) for p, _ in pairs), "neg": sum(bool(NEGW.search(n)) for _, n in pairs)},
           "admission_register": {"pos": sum(bool(ADMIT.search(p)) for p, _ in pairs), "neg": sum(bool(ADMIT.search(n)) for _, n in pairs)},
           "cover_register": {"pos": sum(bool(COVER.search(p)) for p, _ in pairs), "neg": sum(bool(COVER.search(n)) for _, n in pairs)},
           "first_words_pos": collections.Counter(p.split()[0].lower().strip(",.!?\"") for p, _ in pairs).most_common(8),
           "first_words_neg": collections.Counter(n.split()[0].lower().strip(",.!?\"") for _, n in pairs).most_common(8)}
    return out


def cmd_audit():
    d = item_dir(CID)
    items = read_jsonl(d / "items.jsonl")
    held = read_jsonl(d / "heldout.jsonl")
    out = {"concept": CID, "prefills": [audit_pairs([r for r in items if r["frame"] == "c"], "Apollo roleplaying, frame c"),
                                        audit_pairs([r for r in held if r["frame"] == "c"], "MASK provided_facts, frame c (held-out)")]}
    # system-prompt roles in the Apollo scenarios: who the model is told to be
    roles = collections.Counter()
    for r in items:
        if r["frame"] == "c" and r["polarity"] == "pos":
            m = re.match(r"You are (?:an? )?([^,.]{3,60})", r["messages"][0]["content"])
            roles[(m.group(1).strip() if m else "?")[:40]] += 1
    out["apollo_roles_sample"] = roles.most_common(25)
    # Gemma's verdict on its own a/b replies, by polarity (D-033c)
    vf = ACTS / "verdicts" / f"{CID}.jsonl"
    if vf.exists():
        pol = {r["item_id"]: (r["frame"], r["polarity"]) for r in items}
        cnt = collections.Counter()
        for r in read_jsonl(vf):
            f, p = pol[r["item_id"]]
            cnt[f"{f}-{p}-{r['label']}"] += 1
        out["verdicts_ab"] = dict(sorted(cnt.items()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "c1_diag_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["acts", "audit", "report"])
    a = ap.parse_args()
    if a.stage == "acts":
        cmd_acts()
    elif a.stage == "audit":
        cmd_audit()
    else:
        report(json.loads((OUT / "c1_diag.json").read_text()))


if __name__ == "__main__":
    main()
