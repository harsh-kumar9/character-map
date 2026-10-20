"""Stage 3: persona vectors and per-row activations from the judged answers (HF forward passes, bf16 + SDPA, D-023).

Follows upstream generate_vec.py (get_persona_effective, get_hidden_p_and_r): text = prompt + answer, tokenized without
extra special tokens; prompt_avg = mean over prompt tokens, response_avg = mean over answer tokens, prompt_last = last
prompt token; vector = mean(pos) - mean(neg) per hidden-state index. Vectors are [35, 2560]: index 0 = embeddings,
index L+1 = output of decoder layer L (the Gemma Scope 2 hook point model.layers.L.output), index 34 = after the
final norm. Upstream's filter for plain traits: pos trait >= 50, neg trait < 50, both coherence >= 50, rows paired.

Changes from upstream: batched forward passes (same values); upstream reads config.num_hidden_layers, which Gemma's
multimodal config keeps under text_config, so we run the text decoder directly; context-sensitive traits (D-024)
keep each side's rows on their own score (virtue, excess or deficiency >= 50, coherence >= 50; neutral: coherence
only) and save side means per condition. We also save per-row activations at the SAE layers (9, 17, 22, 29) for
every row, judged or not, for the fidelity check (S1) and later analyses: prompt_last, response_avg, and the mean
over the first 32 answer tokens (D-008).

    python -m cm.pv.extract --traits evil,deceptive
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .. import model as cm_model
from ..paths import OUTPUTS
from . import traits as T
from .judge import scored_path

SAE_LAYERS = [9, 17, 22, 29]
KINDS = ["prompt_avg", "response_avg", "prompt_last"]


def vec_dir(model, judge):
    return OUTPUTS / "pv" / model / f"vectors_{judge}"


def acts_dir(model):
    return OUTPUTS / "pv" / model / "acts"


@torch.no_grad()
def hidden(m, tok, prompts, answers, batch=16):
    """Per row: dict of KINDS -> [35, d] float32 means, plus SAE-layer readouts [4, d] for prompt_last,
    response_avg and response_first32."""
    tm = cm_model.text_model(m)
    out = []
    for i in range(0, len(prompts), batch):
        P, A = prompts[i:i + batch], answers[i:i + batch]
        enc = [tok(p + a, add_special_tokens=False)["input_ids"] for p, a in zip(P, A)]
        plen = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in P]
        T_ = max(len(e) for e in enc)
        ids = torch.zeros(len(enc), T_, dtype=torch.long)
        mask = torch.zeros(len(enc), T_, dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, :len(e)] = torch.tensor(e)
            mask[j, :len(e)] = 1
        hs = tm(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
        H = torch.stack(hs, 1)  # [b, 35, T, d] bf16
        for j, e in enumerate(enc):
            n, p = len(e), plen[j]
            h = H[j, :, :n].float()
            r = {"prompt_avg": h[:, :p].mean(1).cpu(), "prompt_last": h[:, p - 1].cpu(),
                 "response_avg": (h[:, p:].mean(1) if n > p else torch.zeros_like(h[:, 0])).cpu()}
            idx = [L + 1 for L in SAE_LAYERS]
            r["sae_prompt_last"] = h[idx, p - 1].cpu()
            r["sae_response_avg"] = r["response_avg"][idx]
            r["sae_response_first32"] = (h[idx, p:p + 32].mean(1) if n > p else torch.zeros_like(h[idx, 0])).cpu()
            out.append(r)
        del hs, H
    return out


def check_index(m, tok):
    """hidden_states[L+1] must equal the hooked output of decoder layer L (the SAE hook point)."""
    ids = torch.tensor([tok("<start_of_turn>user\nHello there<end_of_turn>\n")["input_ids"]]).cuda()
    with cm_model.capture_resid(m, [17]) as st:
        hs = cm_model.text_model(m)(input_ids=ids, output_hidden_states=True).hidden_states
    assert len(hs) == 35, len(hs)
    assert torch.allclose(hs[18].float(), st[17].float(), atol=1e-2), "hidden_states index does not match hook"


def keep_mask(tr: T.Trait, side: str, df: pd.DataFrame, other: pd.DataFrame | None, thr=50):
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    if tr.kind == "plain":  # upstream rule, paired with the other side's row
        o = lambda c: pd.to_numeric(other[c], errors="coerce")
        pos, neg = (df, other) if side == "pos" else (other, df)
        pn = lambda d, c: pd.to_numeric(d[c], errors="coerce")
        return ((pn(pos, tr.slug) >= thr) & (pn(neg, tr.slug) < 100 - thr) &
                (pn(pos, "coherence") >= 50) & (pn(neg, "coherence") >= 50)).to_numpy()
    if side == "neutral":
        return (num("coherence") >= 50).to_numpy()
    return ((num(side) >= thr) & (num("coherence") >= 50)).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-4b-it")
    ap.add_argument("--traits", default="all")
    ap.add_argument("--judge", default="gemma-4-31B-it")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    mt = args.model + args.tag
    slugs = T.all_slugs() if args.traits == "all" else [T.slug(x) for x in args.traits.split(",")]
    vd, ad = vec_dir(mt, args.judge), acts_dir(mt)
    vd.mkdir(parents=True, exist_ok=True)
    ad.mkdir(parents=True, exist_ok=True)
    m, tok = cm_model.load(args.model)
    check_index(m, tok)
    for t in slugs:
        tr = T.load(t)
        if (vd / f"{t}_meta.json").exists():
            continue
        paths = {s: scored_path(mt, t, s, args.judge) for s in tr.sides}
        if not all(p.exists() for p in paths.values()):
            print(f"[extract] {t}: not fully judged, skipping", flush=True)
            continue
        dfs = {s: pd.read_csv(p, keep_default_na=False) for s, p in paths.items()}
        means, meta = {}, {"trait": t, "kind": tr.kind, "judge": args.judge, "n_rows": {}, "n_kept": {}}
        for s, df in dfs.items():
            other = dfs["neg" if s == "pos" else "pos"] if tr.kind == "plain" else None
            keep = keep_mask(tr, s, df, other)
            rows = hidden(m, tok, df.prompt.tolist(), df.answer.tolist())
            np.savez(ad / f"{t}_{s}.npz", keep=keep,
                     **{k: torch.stack([r[k] for r in rows]).numpy().astype(np.float16)
                        for k in ("sae_prompt_last", "sae_response_avg", "sae_response_first32")})
            meta["n_rows"][s], meta["n_kept"][s] = int(len(df)), int(keep.sum())
            cond = df["condition"].astype(str).to_numpy() if tr.kind == "conditional" else None
            for k in KINDS:
                X = torch.stack([r[k] for r in rows])
                if keep.any():
                    means[(s, k, "all")] = X[torch.from_numpy(keep)].mean(0)
                if cond is not None:
                    for c in ("True", "False"):
                        sel = torch.from_numpy(keep & (cond == c))
                        if sel.any():
                            means[(s, k, c)] = X[sel].mean(0)
            print(f"[extract] {t}/{s}: kept {keep.sum()}/{len(df)}", flush=True)
        contrasts = ([("pos", "neg")] if tr.kind == "plain" else
                     [("virtue", "neutral"), ("excess", "neutral"), ("deficiency", "neutral"),
                      ("virtue", "excess"), ("virtue", "deficiency")])
        for a, b in contrasts:
            for k in KINDS:
                if (a, k, "all") in means and (b, k, "all") in means:
                    name = f"{t}_{k}_diff.pt" if tr.kind == "plain" else f"{t}_{k}_{a}_minus_{b}.pt"
                    torch.save(means[(a, k, "all")] - means[(b, k, "all")], vd / name)
        if tr.kind == "conditional":
            torch.save({f"{s}|{k}|{c}": v for (s, k, c), v in means.items()}, vd / f"{t}_side_means.pt")
        rv = vd / (f"{t}_response_avg_diff.pt" if tr.kind == "plain" else f"{t}_response_avg_virtue_minus_neutral.pt")
        if rv.exists():
            meta["response_avg_norm_by_layer"] = [round(x, 3) for x in torch.load(rv).norm(dim=1).tolist()]
        (vd / f"{t}_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[extract] {t}: saved", flush=True)


if __name__ == "__main__":
    main()
