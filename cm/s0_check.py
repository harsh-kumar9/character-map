"""S0: do our hooks and JumpReLU encoder reproduce the activations Gemma Scope 2 stored for its features?

For each SAE layer, examples.safetensors holds, per feature, its top 1,000 activations with the sequence id
and token position in a shared token set (tokens [237009, 512], single-turn Gemma chats, pad id 0).
We sample features, re-run those sequences through Gemma-3-4B-IT, encode the residual at the stored
positions, and compare with the stored values. We also measure FVU and L0 on random sequences from the
same token set: that is the in-distribution reference for the S1 fidelity gate on our own prompts.

Gate (proposed in PLAN.md S0): Pearson r >= 0.98 and median relative error <= 5% at every layer,
and the feature is above threshold in >= 95% of sampled examples.

Run: python -m cm.s0_check [--features 64] [--random-seqs 64]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from safetensors import safe_open

from . import model as cm_model
from .paths import RESULTS, sae_dir
from .sae import JumpReLUSAE

LAYERS = [9, 17, 22, 29]
RANKS = [0, 1, 2, 3, 10, 50, 200, 999]  # which of each feature's stored top-1000 examples to re-check
PAD, BOS = 0, 2


def load_examples(folder):
    with safe_open(str(folder / "examples.safetensors"), "pt") as f:
        return {k: f.get_tensor(k) for k in ("activations", "positions", "seq_ids", "feature_frequencies", "tokens")}


@torch.no_grad()
def resid_at(m, tokens: torch.Tensor, layer: int, batch: int = 16) -> list[torch.Tensor]:
    """Run each row of tokens [n, 512] (right-padded) and return the layer output per row, trimmed to its length."""
    tm = cm_model.text_model(m)
    out = []
    for i in range(0, len(tokens), batch):
        chunk = tokens[i:i + batch]
        lens = (chunk != PAD).sum(1)
        T = int(lens.max())
        ids = chunk[:, :T].cuda()
        mask = (ids != PAD).long()
        with cm_model.capture_resid(m, [layer]) as st:
            tm(input_ids=ids, attention_mask=mask)
        h = st[layer].float().cpu()
        out.extend(h[j, : int(lens[j])] for j in range(len(chunk)))
    return out


def check_layer(m, tok, layer: int, n_features: int, n_random: int, rng: np.random.Generator) -> dict:
    t0 = time.time()
    folder = sae_dir("gemma-3-4b-it", layer)
    sae = JumpReLUSAE.load(folder, device="cuda")
    ex = load_examples(folder)
    acts, pos, sid, toks = ex["activations"], ex["positions"], ex["seq_ids"], ex["tokens"]

    alive = torch.nonzero(acts[:, 999] > 0).squeeze(1).numpy()  # features with a full top-1000 list
    feats = rng.choice(alive, size=min(n_features, len(alive)), replace=False)
    pairs = [(int(f), r) for f in feats for r in RANKS]
    seqs = sorted({int(sid[f, r]) for f, r in pairs})
    row_of = {s: i for i, s in enumerate(seqs)}
    hs = resid_at(m, toks[torch.as_tensor(seqs)], layer)

    x = torch.stack([hs[row_of[int(sid[f, r])]][int(pos[f, r])] for f, r in pairs]).cuda()
    idx = torch.tensor([f for f, _ in pairs], device="cuda")
    pre = sae.feature_pre(x, idx)
    ours = (pre * (pre > sae.threshold[idx])).cpu().numpy()
    stored = np.array([float(acts[f, r]) for f, r in pairs])
    rel = np.abs(ours - stored) / np.maximum(np.abs(stored), 1e-6)

    # in-distribution reference: FVU and L0 over random sequences, BOS and pad excluded
    rand = torch.as_tensor(np.sort(rng.choice(len(toks), size=n_random, replace=False)))
    hr = resid_at(m, toks[rand], layer)
    keep = [h[(toks[int(s), : len(h)] != BOS)] for s, h in zip(rand, hr)]
    X = torch.cat(keep).cuda()
    F = sae.encode(X)
    Xh = sae.decode(F)
    fvu = float(((X - Xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())
    l0 = float((F > 0).float().sum(1).mean())

    # a short decoded sample, to confirm the chat format of the stored token set
    s0 = toks[seqs[0]]
    sample = tok.decode(s0[(s0 != PAD)][:60].tolist())

    res = {
        "layer": layer,
        "n_features": int(len(feats)),
        "n_pairs": len(pairs),
        "pearson_r": float(np.corrcoef(ours, stored)[0, 1]),
        "median_rel_err": float(np.median(rel)),
        "p90_rel_err": float(np.quantile(rel, 0.9)),
        "frac_above_threshold": float((ours > 0).mean()),
        "rel_err_by_rank": {str(r): float(np.median(rel[[i for i, (_, rr) in enumerate(pairs) if rr == r]])) for r in RANKS},
        "random_seqs": int(n_random),
        "random_tokens": int(X.shape[0]),
        "fvu_in_distribution": fvu,
        "l0_in_distribution": l0,
        "config_l0": sae.config.get("l0"),
        "sample_decoded": sample,
        "seconds": round(time.time() - t0, 1),
    }
    res["pass"] = bool(res["pearson_r"] >= 0.98 and res["median_rel_err"] <= 0.05 and res["frac_above_threshold"] >= 0.95)
    del sae
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--random-seqs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    m, tok = cm_model.load("gemma-3-4b-it")
    assert len(cm_model.decoder_layers(m)) == 34
    out = {"model": "google/gemma-3-4b-it", "sae": "gemma-scope-2-4b-it resid_post 65k l0_medium",
           "ranks_checked": RANKS, "seed": args.seed, "layers": []}
    for L in LAYERS:
        r = check_layer(m, tok, L, args.features, args.random_seqs, rng)
        print(json.dumps({k: v for k, v in r.items() if k != "sample_decoded"}), flush=True)
        out["layers"].append(r)
    out["pass"] = all(r["pass"] for r in out["layers"])
    dest = RESULTS / "s0"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "s0_check.json").write_text(json.dumps(out, indent=2))
    print("S0", "PASS" if out["pass"] else "FAIL", "->", dest / "s0_check.json", flush=True)


if __name__ == "__main__":
    main()
