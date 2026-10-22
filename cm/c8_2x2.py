"""C8 agreement x correctness (2x2 across sub-contrasts).

Within each C8 sub-contrast the pos reply is the correct one, so a sub-contrast direction mixes agreeing with the user
and being correct. The two agreement sub-contrasts cross the two factors:
    sycophantic_agreement (user wrong):  pos = disagree, correct    neg = agree, incorrect
    genuine_agreement     (user right):  pos = agree, correct       neg = disagree, incorrect
Per frame (a, c, d), readout and SAE depth, on train items, the two main-effect directions with equal cell weights:
    correctness = mean(correct cells) - mean(incorrect cells)
    agreement   = mean(agree cells)   - mean(disagree cells)
Scored on test items: each direction on its own label and on the other factor's label (near 0.5 means the two are
separated), their cosine, and cross-frame transfer of each (frame i's direction on frame j's test items).
Writes results/analysis/c8_2x2.json and prints layer 17 and 29.

    python -m cm.c8_2x2
"""
from __future__ import annotations

import json

import numpy as np

from .paths import RESULTS
from .s3 import auroc, load

DEPTHS = {9: 10, 17: 18, 22: 23, 29: 30}
SUBS = ("sycophantic_agreement", "genuine_agreement")


def labels(m):
    correct = m["polarity"] == "pos"
    agree = (m["contrast"] == "genuine_agreement") == correct
    return int(correct), int(agree)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    z, meta, resp = load("C8")
    rows = [i for i, m in enumerate(meta) if m["contrast"] in SUBS and m["split"] in ("train", "test")]
    frames = sorted({meta[i]["frame"] for i in rows})
    out, dirs = [], {}
    readouts = {f: (["reply32", "boundary"] if any(meta[i]["readout"] == "boundary" for i in rows if meta[i]["frame"] == f)
                    else ["reply32"]) for f in frames}
    for f in frames:
        for ro in readouts[f]:
            tr = [i for i in rows if meta[i]["frame"] == f and meta[i]["split"] == "train"]
            te = [i for i in rows if meta[i]["frame"] == f and meta[i]["split"] == "test"]
            lab_tr = np.array([labels(meta[i]) for i in tr])
            lab_te = np.array([labels(meta[i]) for i in te])
            for L, idx in DEPTHS.items():
                X = z[ro][tr, idx].astype(np.float64)
                cell = {(c, a): X[(lab_tr[:, 0] == c) & (lab_tr[:, 1] == a)].mean(0) for c in (0, 1) for a in (0, 1)}
                corr = unit((cell[(1, 0)] + cell[(1, 1)]) / 2 - (cell[(0, 0)] + cell[(0, 1)]) / 2)
                agr = unit((cell[(0, 1)] + cell[(1, 1)]) / 2 - (cell[(0, 0)] + cell[(1, 0)]) / 2)
                dirs[(f, ro, L)] = (corr, agr)
                Y = z[ro][te, idx].astype(np.float64)
                out.append({"frame": f, "readout": ro, "layer": L, "n_train": len(tr), "n_test": len(te),
                            "correctness_on_correct": auroc(Y @ corr, lab_te[:, 0]),
                            "agreement_on_agree": auroc(Y @ agr, lab_te[:, 1]),
                            "correctness_on_agree": auroc(Y @ corr, lab_te[:, 1]),
                            "agreement_on_correct": auroc(Y @ agr, lab_te[:, 0]),
                            "cosine": float(corr @ agr)})
    transfer = []
    for (fi, ro, L), (corr, agr) in dirs.items():
        for fj in frames:
            if fj == fi:
                continue
            te = [i for i in rows if meta[i]["frame"] == fj and meta[i]["split"] == "test"]
            lab = np.array([labels(meta[i]) for i in te])
            Y = z["reply32"][te, DEPTHS[L]].astype(np.float64)
            transfer.append({"from": fi, "from_readout": ro, "to": fj, "layer": L,
                             "correctness": auroc(Y @ corr, lab[:, 0]), "agreement": auroc(Y @ agr, lab[:, 1])})
    (RESULTS / "analysis").mkdir(parents=True, exist_ok=True)
    (RESULTS / "analysis" / "c8_2x2.json").write_text(json.dumps({"within": out, "transfer": transfer}, indent=1))
    for L in (17, 29):
        print(f"layer {L}: within frame (test AUROC)")
        print("  frame readout   corr->corr agree->agree | corr->agree agree->corr | cosine")
        for r in out:
            if r["layer"] == L:
                print(f"  {r['frame']}     {r['readout']:9s} {r['correctness_on_correct']:.2f}       {r['agreement_on_agree']:.2f}"
                      f"        |   {r['correctness_on_agree']:.2f}        {r['agreement_on_correct']:.2f}     | {r['cosine']:+.2f}")
        print("  transfer (from reply32 unless noted): from->to  correctness  agreement")
        for t in transfer:
            if t["layer"] == L:
                print(f"    {t['from']}({t['from_readout']})->{t['to']}   {t['correctness']:.2f}   {t['agreement']:.2f}")


if __name__ == "__main__":
    main()
