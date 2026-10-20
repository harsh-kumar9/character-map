"""Activations for the concept stimuli (S3 inputs) and the SAE fidelity gate on our own prompts (S1).

Readouts (D-008): the boundary token (last token before the model's reply) and the mean over the first 32 reply
tokens; for items with a prefilled reply (readout "response") also the mean over the whole reply. Boundary items need
the model's own reply for the 32-token readout, so stage `replies` generates it first: greedy, 32 tokens (D-028).

    python -m cm.stimx replies  --concepts A1,A2        vLLM, greedy 32-token replies for boundary items
    python -m cm.stimx acts     --concepts A1,A2        HF forward; outputs/acts/stimuli/<ID>.npz, [n, 35, 2560] fp16
    python -m cm.stimx fidelity --concepts A1,A2        S1: FVU, L0 and spliced KL at the SAE layers, per frame and
                                                        token group, on a sample per concept (and trait rollouts)
Hidden-state index L+1 is the output of decoder layer L (the Gemma Scope 2 hook point), as in cm.pv.extract.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from . import model as cm_model
from .paths import MODELS, OUTPUTS, RESULTS, sae_dir, snapshot
from .sae import JumpReLUSAE
from .stimuli.schema import CARDS, item_dir, read_jsonl, write_jsonl

MODEL = "gemma-3-4b-it"
SAE_LAYERS = [9, 17, 22, 29]
ACTS = OUTPUTS / "acts" / "stimuli"
REPLY_N = 32
BOS, PAD = 2, 0
TEMPLATE_IDS = {2, 105, 106}  # <bos>, <start_of_turn>, <end_of_turn>


def all_concepts():
    return sorted(p.stem for p in CARDS.glob("*.yaml") if not p.stem.startswith("_") and item_dir(p.stem).exists())


def load_items(cid):
    d = item_dir(cid)
    return read_jsonl(d / "items.jsonl") + read_jsonl(d / "heldout.jsonl")


def prompt_and_reply(tok, it, replies):
    """Prompt string ending where the model's reply starts, and the reply text (prefilled or generated)."""
    msgs = it["messages"]
    if it["readout"] == "response":
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        return prompt, msgs[-1]["content"]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt, replies.get(it["item_id"], "")


def replies_path(cid):
    return ACTS / "replies" / f"{cid}.jsonl"


def cmd_replies(cids):
    from vllm import LLM, SamplingParams
    todo = [c for c in cids if not replies_path(c).exists()]
    if not todo:
        return
    llm = LLM(model=str(snapshot(MODELS[MODEL])), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0})
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=REPLY_N, stop_token_ids=[1, 106], skip_special_tokens=True)
    for c in todo:
        items = [it for it in load_items(c) if it["readout"] == "boundary"]
        prompts = [tok.apply_chat_template(it["messages"], tokenize=False, add_generation_prompt=True) for it in items]
        outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []
        write_jsonl(replies_path(c), [{"item_id": it["item_id"], "reply": o.outputs[0].text} for it, o in zip(items, outs)])
        print(f"[replies] {c}: {len(items)} boundary items", flush=True)


def load_replies(cid):
    return {r["item_id"]: r["reply"] for r in read_jsonl(replies_path(cid))}


@torch.no_grad()
def forward_hidden(m, ids_list, batch=8):
    """Yield per-row hidden states [35, n_tokens, d] (bf16, on GPU) for right-padded batches."""
    tm = cm_model.text_model(m)
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        T = max(len(x) for x in chunk)
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        hs = tm(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
        H = torch.stack(hs, 1)
        for j, x in enumerate(chunk):
            yield H[j, :, :len(x)]
        del hs, H


def tokenize(tok, prompt, reply):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(reply, add_special_tokens=False)["input_ids"] if reply else []
    return p + r, len(p)


def cmd_acts(cids):
    m, tok = cm_model.load(MODEL)
    ACTS.mkdir(parents=True, exist_ok=True)
    for c in cids:
        out = ACTS / f"{c}.npz"
        if out.exists():
            continue
        items, rep = load_items(c), load_replies(c)
        rows = [(it,) + tokenize(tok, *prompt_and_reply(tok, it, rep)) for it in items]
        n, L1, d = len(rows), 35, 2560
        # float32: Gemma's residual dim 443 exceeds the fp16 range at layers 27-32
        bnd = np.zeros((n, L1, d), np.float32)
        r32 = np.zeros((n, L1, d), np.float32)
        resp_rows = [k for k, (it, _, _) in enumerate(rows) if it["readout"] == "response"]
        resp_pos = {k: j for j, k in enumerate(resp_rows)}
        rall = np.zeros((len(resp_rows), L1, d), np.float32)
        nrep = np.zeros(n, np.int32)
        for k, (H, (it, ids, p)) in enumerate(zip(forward_hidden(m, [r[1] for r in rows]), rows)):
            h = H.float()
            bnd[k] = h[:, p - 1].cpu().numpy()
            if len(ids) > p:
                r32[k] = h[:, p:p + REPLY_N].mean(1).cpu().numpy()
                if it["readout"] == "response":
                    rall[resp_pos[k]] = h[:, p:].mean(1).cpu().numpy()
            nrep[k] = len(ids) - p
        meta = [{"item_id": it["item_id"], "scenario_id": it["scenario_id"], "frame": it["frame"],
                 "polarity": it["polarity"], "split": it["split"], "readout": it["readout"],
                 "contrast": (it.get("meta") or {}).get("contrast")} for it, _, _ in rows]
        np.savez(out, boundary=bnd, reply32=r32, response_all=rall, response_rows=np.array(resp_rows, np.int32),
                 n_reply_tokens=nrep,
                 meta=np.array(json.dumps(meta)))
        print(f"[acts] {c}: {n} items", flush=True)


def token_groups(ids, p):
    """Index lists for the fidelity groups: boundary token, first 32 reply tokens, prompt content tokens
    (no BOS or turn markers), and all non-BOS tokens (comparable to S0's in-distribution FVU)."""
    content = [i for i in range(p) if ids[i] not in TEMPLATE_IDS]
    return {"boundary": [p - 1], "reply32": list(range(p, min(len(ids), p + REPLY_N))), "prompt_content": content,
            "all_non_bos": [i for i in range(len(ids)) if ids[i] != BOS]}


def fidelity_sample(cid, per_cell, rng):
    items = [it for it in load_items(cid) if it["split"] == "train"]
    cells = {}
    for it in items:
        cells.setdefault((it["frame"], it["polarity"]), []).append(it)
    out = []
    for k in sorted(cells):
        rng.shuffle(cells[k])
        out += cells[k][:per_cell]
    return out


@torch.no_grad()
def cmd_fidelity(cids, per_cell, traits_per_side, tag, indist=0):
    import pandas as pd
    from .pv import traits as T
    from .pv.gen import gen_path
    rng = random.Random(0)
    m, tok = cm_model.load(MODEL)
    saes = {L: JumpReLUSAE.load(sae_dir(MODEL, L), device="cuda") for L in SAE_LAYERS}
    blocks = cm_model.decoder_layers(m)
    rows = []  # (source, frame, ids, p)
    for c in cids:
        rep = load_replies(c)
        for it in fidelity_sample(c, per_cell, rng):
            ids, p = tokenize(tok, *prompt_and_reply(tok, it, rep))
            rows.append((c, it["frame"], ids, p))
    if traits_per_side:
        for t in T.all_slugs():
            for s in T.load(t).sides:
                gp = gen_path(MODEL, t, s)
                if not gp.exists():
                    continue
                df = pd.read_csv(gp, keep_default_na=False).sample(n=traits_per_side, random_state=0)
                for pr, an in zip(df.prompt, df.answer):
                    ids, p = tokenize(tok, pr, an)
                    rows.append((f"trait:{t}", "trait", ids, p))
    if indist:  # Gemma Scope 2's own chat data: the same groups and KL, as the in-distribution reference
        from safetensors import safe_open
        with safe_open(str(sae_dir(MODEL, 17) / "examples.safetensors"), "pt") as f:
            toks = f.get_tensor("tokens")
        marker = tok("<start_of_turn>model\n", add_special_tokens=False)["input_ids"]
        order = list(range(len(toks)))
        rng.shuffle(order)
        got = 0
        for r_ in order:
            ids = [int(x) for x in toks[r_] if int(x) != PAD]
            p = None
            for j in range(len(ids) - len(marker), 0, -1):
                if ids[j:j + len(marker)] == marker:
                    p = j + len(marker)
                    break
            if p is None or len(ids) - p < 8:
                continue
            rows.append(("indist", "indist", ids, p))
            got += 1
            if got >= indist:
                break
    acc = {}  # (layer, frame, group) -> sums

    def add(key, x, xh, f):
        a = acc.setdefault(key, {"se": 0.0, "n": 0, "sum": torch.zeros(x.shape[1], dtype=torch.float64), "sq": 0.0,
                                 "l0": 0.0})
        a["se"] += float(((x - xh) ** 2).sum())
        a["sum"] += x.double().sum(0).cpu()
        a["sq"] += float((x.double() ** 2).sum())
        a["n"] += x.shape[0]
        a["l0"] += float((f > 0).float().sum())

    kl = {}  # (layer, frame) -> [sum, count]
    tm = cm_model.text_model(m)
    head = m.lm_head

    def logits_at(x_ids, pos):  # final hidden state is already normed; project only the positions we score
        return head(tm(input_ids=x_ids).last_hidden_state[0, pos]).float()

    for k, (src, frame, ids, p) in enumerate(rows):
        x_ids = torch.tensor([ids]).cuda()
        pos = torch.arange(p - 1, min(len(ids) - 1, p - 1 + REPLY_N), device="cuda")
        with cm_model.capture_resid(m, SAE_LAYERS) as st:
            clean = logits_at(x_ids, pos)
        g = token_groups(ids, p)
        for L in SAE_LAYERS:
            h = st[L][0].float()
            f = saes[L].encode(h)
            hh = saes[L].decode(f)
            for gname, idx in g.items():
                if idx:
                    sel = torch.tensor(idx, device=h.device)
                    add((L, frame, gname), h[sel], hh[sel], f[sel])
                    if frame != "indist":
                        add((L, "ALL", gname), h[sel], hh[sel], f[sel])
            if len(ids) > p:  # spliced KL over reply-predicting positions
                def splice(_m, _i, out, L=L):
                    o = out[0] if isinstance(out, tuple) else out
                    rec = saes[L].decode(saes[L].encode(o.float())).to(o.dtype)
                    return (rec,) + tuple(out[1:]) if isinstance(out, tuple) else rec
                hd = blocks[L].register_forward_hook(splice)
                try:
                    spl = logits_at(x_ids, pos)
                finally:
                    hd.remove()
                lp, lq = torch.log_softmax(clean, -1), torch.log_softmax(spl, -1)
                kv = float((lp.exp() * (lp - lq)).sum(-1).mean())
                for fr in ((frame, "ALL") if frame != "indist" else (frame,)):
                    s = kl.setdefault((L, fr), [0.0, 0])
                    s[0] += kv
                    s[1] += 1
        if k % 200 == 0:
            print(f"[fidelity] {k}/{len(rows)}", flush=True)
    res = []
    for (L, fr, gname), a in sorted(acc.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        mean = a["sum"] / a["n"]
        var_total = a["sq"] - a["n"] * float((mean ** 2).sum())
        res.append({"layer": L, "frame": fr, "group": gname, "n_tokens": a["n"], "fvu": a["se"] / var_total,
                    "l0": a["l0"] / a["n"],
                    "spliced_kl_reply": (kl[(L, fr)][0] / kl[(L, fr)][1]) if (L, fr) in kl and gname == "reply32" else None})
    dest = RESULTS / "s1"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"fidelity{tag}.json").write_text(json.dumps({"model": MODEL, "sae": "resid_post 65k l0_medium",
                                                         "concepts": cids, "per_cell": per_cell,
                                                         "traits_per_side": traits_per_side, "indist": indist, "n_rows": len(rows),
                                                         "results": res}, indent=1))
    print(f"[fidelity] wrote {dest / f'fidelity{tag}.json'} ({len(rows)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["replies", "acts", "fidelity"])
    ap.add_argument("--concepts", default="all")
    ap.add_argument("--per_cell", type=int, default=8)
    ap.add_argument("--traits_per_side", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--indist", type=int, default=0, help="also score N chats from Gemma Scope 2's own data")
    args = ap.parse_args()
    cids = all_concepts() if args.concepts == "all" else args.concepts.split(",")
    if args.stage == "replies":
        cmd_replies(cids)
    elif args.stage == "acts":
        cmd_acts(cids)
    else:
        cmd_fidelity(cids, args.per_cell, args.traits_per_side, args.tag, args.indist)


if __name__ == "__main__":
    main()
