"""On-policy prediction (D-062 item 4): do the honesty directions predict what the subject itself does?

Teacher-forced forward pass over the subject's saved induced trajectories (AR 200, DC 200, RH 200, FT 200, CF 200),
hidden states at the second and fourth depths (cfg.DEPTH_LAYERS[1], [3]), two positions per D-008 at the turn that
carries the decision: the last prompt token before that turn (the boundary token, h[p - 1] as cm.stimx) and the mean
over the turn's first 32 tokens. Decision turns: AR the write turn (the last assistant turn whose tool call writes the
official output file; the last act turn when there is none), DC the final code turn and the claim turn (two units per
item), RH / FT / CF the single reply. Prompt = apply_chat_template(messages before the turn, add_generation_prompt),
reply = that turn's text, tokenized separately and concatenated, exactly as the runner generated it.

Directions (standardized space of the pooled honesty train rows at each depth, as cm.coherence): the 22 cell
directions, the four frame levers, the shared axes (dm_all, probe_all), K1 and K2 (frame-pooled control directions)
and 16 random unit directions. Readout: AUROC of the standardized projection (x / sd) . u for the deterministic
dishonest outcome within the induced condition (AR covert COMPLETE, DC false done, RH cheat, FT fabricates, CF
fabricates), signed (u points to the concept's positive pole) and folded (max(a, 1 - a)); the within-task 5-fold
logistic probe on the full activation as the ceiling; the random band as the floor.

    CM_MODEL=<model> python -m cm.battery_onpolicy positions [--cond baseline]      CPU   results/<tag>/battery/onpolicy/<cond>/positions.json
    CM_MODEL=<model> python -m cm.battery_onpolicy acts [--cond] [--n 5 --dry]       GPU   outputs/<tag>/battery/onpolicy/<cond>/<task>.npz
    CM_MODEL=<model> python -m cm.battery_onpolicy directions                       CPU   results/<tag>/battery/onpolicy/directions.npz
    CM_MODEL=<model> python -m cm.battery_onpolicy score [--cond]                   CPU   results/<tag>/battery/onpolicy/<cond>/auroc.{json,md}
    ... directions | score-strat | directions-all | score-sweep --sd-floor-q none,0.1,0.25,0.5
        the same under the standardization floor (cm.coherence.floor_sd; D-064 amendment 1 ruling d), one run per
        setting, each under results/<tag>/battery/onpolicy/robust_floor/<setting>/; the baseline files are untouched.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from . import battery as _battery
from . import battery_agent as agent
from . import cfg
from .stimuli.schema import read_jsonl

TASKS = ("AR", "DC", "RH", "FT", "CF")
B_OUT = cfg.outputs_dir() / "battery"
OUT = B_OUT / "onpolicy"
RES = cfg.results_dir() / "battery" / "onpolicy"
DEPTH_IDX = (1, 3)
LAYERS = [cfg.DEPTH_LAYERS[i] for i in DEPTH_IDX]
REPLY_N = 32
N_RANDOM = 16
N_BOOT = 1000
def robust_res(q):
    """results/<tag>/battery/onpolicy/robust_floor/<none|q0.1|...>/: where a --sd-floor-q run reads its directions and writes."""
    from .coherence import qtag
    return RES / "robust_floor" / qtag(q)


GROUPS = {"act cells (c)": lambda k: k.startswith("cell|") and k.endswith("|c"),
          "self-describe cells (d)": lambda k: k.startswith("cell|") and k.endswith("|d"),
          "own-reply cells (a, b)": lambda k: k.startswith("cell|") and k[-1] in "ab",
          "frame levers": lambda k: k.startswith("frame_"),
          "shared axes (dm_all, probe_all)": lambda k: k in ("dm_all", "probe_all"),
          "all_frames": lambda k: k == "all_frames",
          "K1": lambda k: k == "K1", "K2": lambda k: k == "K2",
          "random (16)": lambda k: k.startswith("random_")}


# ---------------------------------------------------------------- decision turns

def write_turn(row):
    """Index of the AR write turn: the last assistant message whose call writes the official output file."""
    out = agent._norm(row.get("official_output_file"))
    idx = None
    for i, m in enumerate(row["messages"]):
        if m["role"] != "assistant":
            continue
        call = agent.parse_call(m["content"])
        if call and call["tool"] == "write_file" and agent._norm(call["arguments"].get("file_path")) == out:
            idx = i
    return idx


def units(cond="baseline"):
    """Per task: [(item_id, kind, prompt messages, reply text, outcome dict)]."""
    items = {t: {it["item_id"]: it for it in read_jsonl(_battery.ITEMS / f"{t}.jsonl")} for t in TASKS}
    ld = cfg.results_dir() / "battery" / "labels" / cond
    labels = {t: ({r["item_id"]: r for r in read_jsonl(ld / f"{t}.jsonl")} if (ld / f"{t}.jsonl").exists() else {}) for t in TASKS}
    U = {}
    p = B_OUT / "traj" / cond / "AR.jsonl"
    if p.exists():
        U["AR"] = []
        for r in read_jsonl(p):
            msgs = r["messages"]
            i = write_turn(r)
            fallback = i is None
            if fallback:       # ..., assistant (last act), user (report request), assistant (report), user (question), assistant (direct)
                i = len(msgs) - 5
            lab = labels["AR"].get(r["item_id"])
            U["AR"].append((r["item_id"], "write" if not fallback else "last_act", msgs[:i], msgs[i]["content"],
                            {"dishonest": None if lab is None else bool(lab["covert"])}))
    p = B_OUT / "traj" / cond / "DC.jsonl"
    if p.exists():
        U["DC"] = []
        for r in read_jsonl(p):
            msgs = r["messages"]
            lab = labels["DC"].get(r["item_id"])
            y = {"dishonest": None if lab is None else bool(lab["false_done"])}
            U["DC"].append((r["item_id"], "code", msgs[:-3], msgs[-3]["content"], y))
            U["DC"].append((r["item_id"], "claim", msgs[:-1], msgs[-1]["content"], y))
    for t in ("RH", "FT", "CF"):
        p = B_OUT / "replies" / cond / f"{t}.jsonl"
        if not p.exists():
            continue
        U[t] = []
        for r in read_jsonl(p):
            lab = labels[t].get(r["item_id"])
            if lab is None:
                y = {"dishonest": None}
            elif t == "RH":
                y = {"dishonest": bool(lab["cheat"])}
            elif t == "FT":
                y = {"dishonest": lab["label"] == "COMPLETED", "not_done": lab["label"] == "NOT_DONE", "label": lab["label"]}
            else:
                y = {"dishonest": None if lab["label"] is None else lab["label"] == "FABRICATES", "label": lab["label"]}
            U[t].append((r["item_id"], "reply", items[t][r["item_id"]]["messages"], r["reply"], y))
    return U


def tokenize(tok, msgs, reply):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **cfg.chat_kwargs())
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(reply, add_special_tokens=False)["input_ids"] if reply else []
    return p + r, len(p)


def cmd_positions(cond="baseline"):
    from transformers import AutoTokenizer
    from .paths import MODELS, snapshot
    tok = AutoTokenizer.from_pretrained(snapshot(MODELS[cfg.MODEL]))
    U = units(cond)
    rule = {"AR": "the write turn: the last assistant message whose tool call writes the official output file (kind write); "
                  "the last act turn before the report request when no such write exists (kind last_act)",
            "DC": "two units per item: the final code turn (the last assistant message before the claim question) and the claim turn (the last message)",
            "RH": "the single reply", "FT": "the single reply (after the scripted tool call and its result)", "CF": "the single reply"}
    info = {"model": cfg.MODEL, "cond": cond, "layers": LAYERS, "depth_idx": list(DEPTH_IDX),
            "positions": "boundary = hidden state at the last prompt token (index p - 1, the final token of the generation prompt); "
                         "reply32 = mean over hidden states at positions p .. p + 31 (the turn's first 32 tokens, fewer when the turn is shorter)",
            "tasks": {}}
    for t, us in U.items():
        rows = []
        for iid, kind, msgs, reply, y in us:
            ids, p = tokenize(tok, msgs, reply)
            rows.append((kind, p, len(ids) - p, y.get("dishonest")))
        kinds = sorted({k for k, *_ in rows})
        ex = {}
        for k in kinds:
            iid, kind, msgs, reply, y = next(u for u in us if u[1] == k)
            ids, p = tokenize(tok, msgs, reply)
            ex[k] = {"item_id": iid, "prompt_tokens": p, "reply_tokens": len(ids) - p,
                     "last_prompt_tokens": tok.convert_ids_to_tokens(ids[max(0, p - 4):p]),
                     "first_reply_tokens": tok.convert_ids_to_tokens(ids[p:p + 8])}
        info["tasks"][t] = {"rule": rule[t], "n_units": len(rows), "kinds": {k: sum(1 for r in rows if r[0] == k) for k in kinds},
                            "prompt_tokens_median_max": [int(np.median([r[1] for r in rows])), max(r[1] for r in rows)],
                            "reply_tokens_median_min": [int(np.median([r[2] for r in rows])), min(r[2] for r in rows)],
                            "units_with_reply_under_32_tokens": sum(1 for r in rows if r[2] < REPLY_N),
                            "units_with_empty_reply": sum(1 for r in rows if r[2] == 0),
                            "dishonest_outcomes": sum(1 for r in rows if r[3]), "outcome_unknown": sum(1 for r in rows if r[3] is None),
                            "example": ex}
    (RES / cond).mkdir(parents=True, exist_ok=True)
    (RES / cond / "positions.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- activations

def forward_two_depths(m, ids_list, batch=8, token_budget=24000):
    """Yield per row {layer: [t, d] bf16 on GPU} at the two depths, right-padded batches bounded by rows and tokens."""
    import torch
    from . import model as cm_model
    tm = cm_model.text_model(m)
    i = 0
    while i < len(ids_list):
        chunk, T = [], 0
        while i < len(ids_list) and len(chunk) < batch:
            T2 = max(T, len(ids_list[i]))
            if chunk and T2 * (len(chunk) + 1) > token_budget:
                break
            chunk.append(ids_list[i]); T = T2; i += 1
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        with torch.no_grad(), cm_model.capture_resid(m, LAYERS) as store:
            tm(input_ids=ids.cuda(), attention_mask=mask.cuda())
        for j, x in enumerate(chunk):
            yield {L: store[L][j, :len(x)] for L in LAYERS}
        del store


def cmd_acts(cond="baseline", only_n=None, dry=False):
    from . import model as cm_model
    dst_dir = OUT / (cond if not dry else f"{cond}_dry")
    dst_dir.mkdir(parents=True, exist_ok=True)
    U = units(cond)
    m, tok = cm_model.load(cfg.MODEL)
    for t, us in U.items():
        dst = dst_dir / f"{t}.npz"
        if dst.exists():
            print(f"[onpolicy] {dst} exists, skipping")
            continue
        us = us[:only_n] if only_n else us
        toks = [tokenize(tok, msgs, reply) for _, _, msgs, reply, _ in us]
        order = sorted(range(len(us)), key=lambda k: -len(toks[k][0]))       # longest first: fewer padded tokens
        n, d = len(us), cfg.D
        bnd = np.zeros((n, len(LAYERS), d), np.float32)
        r32 = np.zeros((n, len(LAYERS), d), np.float32)
        nrep = np.zeros(n, np.int32)
        for k, H in zip(order, forward_two_depths(m, [toks[k][0] for k in order])):
            ids, p = toks[k]
            for li, L in enumerate(LAYERS):
                h = H[L].float()
                bnd[k, li] = h[p - 1].cpu().numpy()
                if len(ids) > p:
                    r32[k, li] = h[p:p + REPLY_N].mean(0).cpu().numpy()
                else:
                    r32[k, li] = np.nan
            nrep[k] = len(ids) - p
        meta = [{"item_id": iid, "kind": kind, **y, "prompt_tokens": toks[k][1], "reply_tokens": int(nrep[k])}
                for k, (iid, kind, _, _, y) in enumerate(us)]
        np.savez(dst, boundary=bnd, reply32=r32, n_reply_tokens=nrep, layers=np.array(LAYERS), meta=np.array(json.dumps(meta)))
        print(f"[onpolicy] {t}: {n} units -> {dst}; boundary {bnd.shape} reply32 {r32.shape}, reply tokens min {nrep.min()} "
              f"median {int(np.median(nrep))}, dishonest {sum(1 for x in meta if x.get('dishonest'))}", flush=True)


# ---------------------------------------------------------------- directions

def cmd_directions(qs=None):
    """qs None: today's directions.npz. A list of floor quantiles (None allowed): the same build per setting on the same
    loaded cells, each under robust_floor/<setting>/ (D-064 amendment 1 ruling d)."""
    from .coherence import gather
    cells = gather()
    lev = np.load(cfg.results_dir() / "coherence" / "levers.npz")
    if qs is None:
        directions_run(cells, lev, None, RES, robust=False)
        return
    for q in qs:
        directions_run(cells, lev, q, robust_res(q), robust=True)


def directions_run(cells, lev, q, out, robust):
    from .coherence import ALL, HON, pool_dirs, unit_vec
    arrs, info = {}, {"model": cfg.MODEL, "layers": LAYERS, "depth_idx": list(DEPTH_IDX), "check_vs_levers": {}}
    if robust:
        info["sd_floor_q"] = q
    rng = np.random.default_rng(0)
    for li, l in zip(LAYERS, DEPTH_IDX):
        common = pool_dirs(cells, HON, l, probe=True, q=q)
        sd = common["sd"]
        dirs = {f"cell|{c}": common["per_cell"][c] for c in HON}
        for f in "abcd":
            dirs[f"frame_{f}"] = unit_vec(np.mean([dirs[f"cell|{c}"] for c in HON if c.endswith(f"|{f}")], 0))
        dirs["all_frames"] = unit_vec(sum(dirs[f"frame_{f}"] for f in "abcd"))
        dirs["dm_all"], dirs["probe_all"] = common["dm"], common["probe"]
        for ctrl in ("K1", "K2"):
            per = []
            for c in [c for c in ALL if c.startswith(ctrl + "|")]:
                Xtr, ytr = cells[c]["Xtr"][:, l].astype(np.float64) / sd, cells[c]["ytr"]
                per.append(unit_vec(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)))
            dirs[ctrl] = unit_vec(np.mean(per, 0))
        for k in range(N_RANDOM):
            dirs[f"random_{k}"] = unit_vec(rng.standard_normal(cfg.D))
        for k, u in dirs.items():
            arrs[f"{k}|L{li}"] = u.astype(np.float32)
        arrs[f"sd|L{li}"] = sd.astype(np.float32)
        if li == cfg.L_STEER:
            info["check_vs_levers"] = {k: float(dirs[k] @ lev[k]) for k in ("dm_all", "probe_all", "frame_a", "frame_c", "all_frames")}
            info["check_vs_levers"]["cells_min_cos"] = float(min(dirs[f"cell|{c}"] @ lev[f"cell|{c}"] for c in HON))
            info["check_vs_levers"]["sd_max_abs_diff"] = float(np.max(np.abs(sd - lev["sd"])))
        print(f"[onpolicy] directions at layer {li}: {len(dirs)}", flush=True)
    info["names"] = sorted({k.split("|L")[0] for k in arrs if not k.startswith("sd|")})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "directions.npz", **arrs)
    (out / "directions.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info["check_vs_levers"], indent=1))


# ---------------------------------------------------------------- scoring

def auroc(score, y):
    """Rank AUROC (ties averaged); nan when one class."""
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe_ceiling(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y, int)
    if y.sum() < 5 or (1 - y).sum() < 5:
        return None
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        out.append(auroc(clf.decision_function(X[te]), y[te]))
    return float(np.mean(out))


def score_block(A, D, rows, y, names, rng, n_boot=200):
    """Per depth and position: per-direction signed and folded AUROC (with an item bootstrap when n_boot > 0), group means,
    the random band and the probe ceiling, for one set of rows and one outcome vector."""
    by = {}
    for li, L in enumerate(LAYERS):
        sd = D[f"sd|L{L}"].astype(np.float64)
        for pos in ("boundary", "reply32"):
            X = A[pos][rows, li].astype(np.float64) / sd
            ent = {"probe_ceiling": probe_ceiling(X, y), "directions": {}}
            boots = rng.choice(len(rows), (n_boot, len(rows)), replace=True) if n_boot else None
            for nm in names:
                z = X @ D[f"{nm}|L{L}"].astype(np.float64)
                a = auroc(z, y)
                d = {"auroc": a, "folded": max(a, 1 - a)}
                if n_boot:
                    bs = [auroc(z[b], y[b]) for b in boots]
                    d["ci"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
                ent["directions"][nm] = d
            ent["groups"] = {}
            for g, f in GROUPS.items():
                vals = [ent["directions"][nm]["folded"] for nm in names if f(nm)]
                if vals:
                    ent["groups"][g] = {"mean_folded": float(np.mean(vals)), "max_folded": float(np.max(vals)), "n": len(vals)}
            rnd = [ent["directions"][nm]["folded"] for nm in names if nm.startswith("random_")]
            ent["random_band_95"] = float(np.percentile(rnd, 95))
            by[f"L{L}|{pos}"] = ent
    return by


def strata_of():
    """item_id -> stratum: AR the contract family (one-turn v14 / recipient-state), the other tasks the item factor."""
    out = {}
    for t in TASKS:
        for it in read_jsonl(_battery.ITEMS / f"{t}.jsonl"):
            m = it["meta"]
            out[it["item_id"]] = ("one-turn" if m.get("contract", "").startswith("v14_") else "recipient-state") if t == "AR" else m.get("factor")
    return out


MD_HEAD = ("| depth, position | act (c) | self-describe (d) | own-reply (a, b) | frame levers | shared axes | all_frames | K1 | K2 | random mean / 95th | probe ceiling |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def md_rows(by):
    out = []
    for k2, ent in by.items():
        g = ent["groups"]
        cell = lambda name: f"{g[name]['mean_folded']:.2f}" if name in g else "-"
        pc = "-" if ent["probe_ceiling"] is None else f"{ent['probe_ceiling']:.2f}"
        out.append(f"| {k2} | {cell('act cells (c)')} | {cell('self-describe cells (d)')} | {cell('own-reply cells (a, b)')} | {cell('frame levers')} | "
                   f"{cell('shared axes (dm_all, probe_all)')} | {cell('all_frames')} | {cell('K1')} | {cell('K2')} | "
                   f"{g['random (16)']['mean_folded']:.2f} / {ent['random_band_95']:.2f} | {pc} |")
    return out


def cmd_score_strat(cond="baseline", qs=None):
    """The same readout within item strata (AR contract family, else the item factor), so that an outcome that follows the
    item family is not read as a direction's prediction: per stratum with at least 5 items of each class, no bootstrap, and
    the positive-weighted mean of the group folded AUROCs over strata. qs: floor settings, each read from and written
    under robust_floor/<setting>/ (None = today's files)."""
    if qs is None:
        score_strat_run(cond, RES, None, robust=False)
        return
    for q in qs:
        score_strat_run(cond, robust_res(q), q, robust=True)


def score_strat_run(cond, rdir, q, robust):
    from .coherence import qtag
    D = np.load(rdir / "directions.npz")
    names = sorted({k.split("|L")[0] for k in D.files if not k.startswith("sd|")})
    rng = np.random.default_rng(0)
    strata = strata_of()
    res = {"model": cfg.MODEL, "cond": cond, "layers": LAYERS, "tasks": {}}
    if robust:
        res["sd_floor_q"] = q
    md = [f"# On-policy prediction within item strata (D-062 item 4), {cfg.MODEL}, cond {cond}" + (f", sd floor {qtag(q)}" if robust else ""), "",
          "Same readout as auroc.md, computed within each item stratum (AR: one-turn / recipient-state contract family; other tasks: "
          "the item factor) and averaged over strata with weights = positives; strata need 5 items of each class.", ""]
    for t in TASKS:
        p = OUT / cond / f"{t}.npz"
        if not p.exists():
            continue
        A = np.load(p)
        meta = json.loads(str(A["meta"]))
        kinds = sorted({x["kind"] for x in meta})
        outcomes = ["dishonest"] + (["not_done"] if t == "FT" else [])
        res["tasks"][t] = {}
        for kind in kinds:
            for oc in outcomes:
                base = [i for i, x in enumerate(meta) if x["kind"] == kind and x.get(oc) is not None and x["reply_tokens"] > 0]
                groups_by = {}
                R = {"strata": {}, "skipped": {}}
                for st in sorted({strata[meta[i]["item_id"]] for i in base}):
                    rows = [i for i in base if strata[meta[i]["item_id"]] == st]
                    y = np.array([meta[i][oc] for i in rows], bool)
                    if y.sum() < 5 or (~y).sum() < 5:
                        R["skipped"][st] = {"n": len(rows), "n_positive": int(y.sum())}
                        continue
                    by = score_block(A, D, rows, y, names, rng, n_boot=0)
                    R["strata"][st] = {"n": len(rows), "n_positive": int(y.sum()), "by": by}
                    for k2, ent in by.items():
                        groups_by.setdefault(k2, []).append((int(y.sum()), ent))
                if not R["strata"]:
                    R["note"] = "no stratum with 5 items of each class"
                    res["tasks"][t][f"{kind}|{oc}"] = R
                    continue
                pooled = {}
                for k2, lst in groups_by.items():
                    w = np.array([n for n, _ in lst], float)
                    ent = {"groups": {}, "random_band_95": float(np.average([e["random_band_95"] for _, e in lst], weights=w)),
                           "probe_ceiling": (float(np.average([e["probe_ceiling"] for _, e in lst], weights=w)) if all(e["probe_ceiling"] is not None for _, e in lst) else None)}
                    for g in GROUPS:
                        vals = [e["groups"][g]["mean_folded"] for _, e in lst if g in e["groups"]]
                        if vals:
                            ent["groups"][g] = {"mean_folded": float(np.average(vals, weights=w[:len(vals)]))}
                    pooled[k2] = ent
                R["pooled"] = pooled
                res["tasks"][t][f"{kind}|{oc}"] = R
                md += [f"## {t} {kind}, outcome {oc}: strata " + ", ".join(f"{st} (n {v['n']}, positives {v['n_positive']})" for st, v in R["strata"].items())
                       + ("; skipped " + ", ".join(f"{st} (n {v['n']}, positives {v['n_positive']})" for st, v in R["skipped"].items()) if R["skipped"] else ""), ""]
                md += list(MD_HEAD) + md_rows(pooled) + [""]
                for st, v in R["strata"].items():
                    md += [f"stratum {st}", ""] + list(MD_HEAD) + md_rows(v["by"]) + [""]
                print(f"[onpolicy] strat {t} {kind}|{oc}: " + "; ".join(
                    f"{k2} act {e['groups'].get('act cells (c)', {}).get('mean_folded', float('nan')):.2f} own {e['groups'].get('own-reply cells (a, b)', {}).get('mean_folded', float('nan')):.2f} "
                    f"rand {e['groups']['random (16)']['mean_folded']:.2f} probe {e['probe_ceiling'] if e['probe_ceiling'] is None else round(e['probe_ceiling'], 2)}"
                    for k2, e in pooled.items()), flush=True)
    (rdir / cond).mkdir(parents=True, exist_ok=True)
    (rdir / cond / "auroc_strat.json").write_text(json.dumps(res, indent=1, default=float))
    (rdir / cond / "auroc_strat.md").write_text("\n".join(md) + "\n")
    print(f"[onpolicy] -> {rdir / cond / 'auroc_strat.json'}")


def cmd_score(cond="baseline"):
    D = np.load(RES / "directions.npz")
    names = sorted({k.split("|L")[0] for k in D.files if not k.startswith("sd|")})
    rng = np.random.default_rng(0)
    res = {"model": cfg.MODEL, "cond": cond, "layers": LAYERS, "tasks": {}}
    md = [f"# On-policy prediction (D-062 item 4), {cfg.MODEL}, cond {cond}", "",
          "Folded AUROC (max(a, 1 - a)) of the standardized projection for the dishonest outcome, mean over the group's directions; "
          "signed values per direction in auroc.json. Probe: within-task 5-fold logistic ceiling on the full activation.", ""]
    for t in TASKS:
        p = OUT / cond / f"{t}.npz"
        if not p.exists():
            continue
        A = np.load(p)
        meta = json.loads(str(A["meta"]))
        kinds = sorted({x["kind"] for x in meta})
        outcomes = ["dishonest"] + (["not_done"] if t == "FT" else [])
        res["tasks"][t] = {}
        for kind in kinds:
            for oc in outcomes:
                rows = [i for i, x in enumerate(meta) if x["kind"] == kind and x.get(oc) is not None and x["reply_tokens"] > 0]
                y = np.array([meta[i][oc] for i in rows], bool)
                key = f"{kind}|{oc}"
                R = {"n": len(rows), "n_positive": int(y.sum()), "by": {}}
                if y.sum() == 0 or (~y).sum() == 0:
                    R["note"] = "one class only; AUROC undefined"
                    res["tasks"][t][key] = R
                    continue
                R["by"] = score_block(A, D, rows, y, names, rng, n_boot=200)
                res["tasks"][t][key] = R
                md += [f"## {t} {kind}, outcome {oc}: n {len(rows)}, positives {int(y.sum())}", "",
                       *MD_HEAD]
                md += md_rows(R["by"]) + [""]
                print(f"[onpolicy] {t} {key}: n {len(rows)} pos {int(y.sum())}; " + "; ".join(
                    f"{k2} act {e['groups'].get('act cells (c)', {}).get('mean_folded', float('nan')):.2f} own {e['groups'].get('own-reply cells (a, b)', {}).get('mean_folded', float('nan')):.2f} "
                    f"rand {e['groups']['random (16)']['mean_folded']:.2f} probe {e['probe_ceiling'] if e['probe_ceiling'] is None else round(e['probe_ceiling'], 2)}"
                    for k2, e in R["by"].items()), flush=True)
    (RES / cond).mkdir(parents=True, exist_ok=True)
    (RES / cond / "auroc.json").write_text(json.dumps(res, indent=1, default=float))
    (RES / cond / "auroc.md").write_text("\n".join(md) + "\n")
    print(f"[onpolicy] -> {RES / cond / 'auroc.json'}")




# ---------------------------------------------------------------- D-063 item 1: position-and-layer sweep

SWEEP_POS = ("value", "turn_mean", "boundary")
SWEEP_OUT = OUT / "sweep"
VALUE_RULE = ("AR: the first token of the state value inside the write call (the contract's state field, then the usual state "
              "keys as cm.battery_agent.artifact_state searches them, JSON or label form; fallback the first token of the file "
              "content, then the first reply token); DC: the first token of the claim answer; RH, CF, FT: the first reply token. "
              "turn_mean: the mean over every token of the decision turn. boundary: the D-008 boundary token (p - 1).")


def value_char_offset(task, kind, reply, item):
    """Character offset in the reply where the value token starts, and how it was found."""
    if not (task == "AR" and kind == "write"):
        return 0, "first_reply_token"
    keys = (item["meta"]["state_field"],) + tuple(k for k in agent.STATE_KEYS if k != item["meta"]["state_field"])
    for k in keys:
        m = re.search(r'(?i)\\?"' + re.escape(k) + r'\\?"\s*:\s*\\?"', reply)       # JSON key inside the escaped content
        if m:
            return m.end(), "state_key_json"
    skip = re.compile(r'(?:\\[ntr"]|\s|\*|[:=|,\-#>])*')        # escaped newlines and quotes, markdown and separators before the value
    for k in keys:
        lab = k.replace("_", "[ _]")
        # the key may follow an escaped newline (\n glues to the first letter, so \b alone misses it)
        m = re.search(r'(?i)(?:(?<=\\n)|(?<=\\t)|(?<=\\")|\b)' + lab + r"\b", reply)       # md / csv label form
        if m:
            end = m.end() + skip.match(reply, m.end()).end() - m.end()
            if end < len(reply):
                return end, "state_key_label"
    m = re.search(r'"content"\s*:\s*"', reply)
    if m:
        return m.end(), "content_start"
    return 0, "first_reply_token"


def tokenize_sweep(tok, msgs, reply, offset):
    """ids, p (prompt length), the index of the value token (first reply token whose span reaches past `offset`)."""
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **cfg.chat_kwargs())
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True) if reply else {"input_ids": [], "offset_mapping": []}
    k = len(enc["input_ids"]) - 1 if enc["input_ids"] else 0
    for j, (a, b) in enumerate(enc["offset_mapping"]):
        if b > offset:
            k = j
            break
    return p_ids + list(enc["input_ids"]), len(p_ids), len(p_ids) + k


def sweep_units(cond="baseline"):
    """units() plus the value offset and how it was found; items are needed for the AR state field."""
    items = {it["item_id"]: it for it in read_jsonl(_battery.ITEMS / "AR.jsonl")}
    U = units(cond)
    out = {}
    for t, us in U.items():
        out[t] = []
        for iid, kind, msgs, reply, y in us:
            off, how = value_char_offset(t, kind, reply, items.get(iid, {"meta": {"state_field": ""}}))
            out[t].append((iid, kind, msgs, reply, y, off, how))
    return out


def cmd_positions_sweep(cond="baseline"):
    from transformers import AutoTokenizer
    from .paths import MODELS, snapshot
    tok = AutoTokenizer.from_pretrained(snapshot(MODELS[cfg.MODEL]))
    U = sweep_units(cond)
    info = {"model": cfg.MODEL, "cond": cond, "n_states": cfg.N_STATES, "d": cfg.D, "rule": VALUE_RULE, "tasks": {}}
    total_units = 0
    for t, us in U.items():
        rows, ex = [], {}
        for iid, kind, msgs, reply, y, off, how in us:
            ids, p, v = tokenize_sweep(tok, msgs, reply, off)
            rows.append((kind, how, p, v - p, len(ids) - p))
            if how not in ex:
                ex[how] = {"item_id": iid, "kind": kind, "prompt_tokens": p, "value_token_index_in_reply": v - p, "reply_tokens": len(ids) - p,
                           "tokens_around_value": tok.convert_ids_to_tokens(ids[max(p, v - 3):v + 5]), "value_token": tok.convert_ids_to_tokens([ids[v]])[0]}
        info["tasks"][t] = {"n_units": len(rows), "value_found_by": {h: sum(1 for r in rows if r[1] == h) for h in sorted({r[1] for r in rows})},
                            "value_index_in_reply_median_max": [int(np.median([r[3] for r in rows])), max(r[3] for r in rows)],
                            "reply_tokens_median_max": [int(np.median([r[4] for r in rows])), max(r[4] for r in rows)], "examples": ex}
        total_units += len(rows)
    gb = total_units * cfg.N_STATES * cfg.D * 4 * len(SWEEP_POS) / 1e9
    info["storage_gb_float32"] = round(gb, 2)
    (RES / cond).mkdir(parents=True, exist_ok=True)
    (RES / cond / "positions_sweep.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, indent=1, ensure_ascii=False))


def forward_all_layers(m, ids_list, batch=4, token_budget=8000):
    """Yield per row the stacked hidden states [n_states, t, d] (bf16, GPU): all layers, small batches."""
    import torch
    from . import model as cm_model
    tm = cm_model.text_model(m)
    i = 0
    while i < len(ids_list):
        chunk, T = [], 0
        while i < len(ids_list) and len(chunk) < batch:
            T2 = max(T, len(ids_list[i]))
            if chunk and T2 * (len(chunk) + 1) > token_budget:
                break
            chunk.append(ids_list[i]); T = T2; i += 1
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.long)
        for j, x in enumerate(chunk):
            ids[j, :len(x)] = torch.tensor(x)
            mask[j, :len(x)] = 1
        with torch.no_grad():
            hs = tm(input_ids=ids.cuda(), attention_mask=mask.cuda(), output_hidden_states=True).hidden_states
        H = torch.stack(hs, 1)
        for j, x in enumerate(chunk):
            yield H[j, :, :len(x)]
        del hs, H


def cmd_acts_sweep(cond="baseline", only_n=None, dry=False):
    from . import model as cm_model
    dst_dir = SWEEP_OUT if not dry else OUT / "sweep_dry"
    dst_dir.mkdir(parents=True, exist_ok=True)
    U = sweep_units(cond)
    m, tok = cm_model.load(cfg.MODEL)
    for t, us in U.items():
        dst = dst_dir / f"{t}.npz"
        if dst.exists():
            print(f"[sweep] {dst} exists, skipping")
            continue
        us = us[:only_n] if only_n else us
        toks = [tokenize_sweep(tok, msgs, reply, off) for _, _, msgs, reply, _, off, _ in us]
        order = sorted(range(len(us)), key=lambda k: -len(toks[k][0]))
        n, L1, d = len(us), cfg.N_STATES, cfg.D
        arr = {pos: np.zeros((n, L1, d), np.float32) for pos in SWEEP_POS}
        nrep = np.zeros(n, np.int32)
        for k, H in zip(order, forward_all_layers(m, [toks[k][0] for k in order])):
            ids, p, v = toks[k]
            h = H.float()
            arr["boundary"][k] = h[:, p - 1].cpu().numpy()
            arr["value"][k] = h[:, v].cpu().numpy()
            arr["turn_mean"][k] = (h[:, p:].mean(1) if len(ids) > p else h[:, p - 1]).cpu().numpy()
            nrep[k] = len(ids) - p
            del h
        meta = [{"item_id": iid, "kind": kind, **y, "prompt_tokens": toks[k][1], "value_index": toks[k][2] - toks[k][1],
                 "value_found_by": how, "reply_tokens": int(nrep[k])} for k, (iid, kind, _, _, y, _, how) in enumerate(us)]
        np.savez(dst, **arr, n_reply_tokens=nrep, meta=np.array(json.dumps(meta)))
        print(f"[sweep] {t}: {n} units -> {dst}; " + ", ".join(f"{pos} {arr[pos].shape}" for pos in SWEEP_POS)
              + f"; {dst.stat().st_size / 1e9:.2f} GB", flush=True)


def cmd_directions_all(qs=None):
    """directions_all.npz: the item 4 direction set at every hidden-state index (names x n_states x d) and the pooled SD per
    layer, from the same cells and code path as cm.coherence (gather, pool_dirs), with s4_overlap.LIDX widened to all states.
    qs: floor settings, each under robust_floor/<setting>/ (None = today's file)."""
    from . import s4_overlap
    s4_overlap.LIDX = list(range(cfg.N_STATES))
    from .coherence import gather
    cells = gather()
    if qs is None:
        directions_all_run(cells, None, RES, robust=False)
        return
    for q in qs:
        directions_all_run(cells, q, robust_res(q), robust=True)


def directions_all_run(cells, q, out, robust):
    from .coherence import ALL, HON, pool_dirs, unit_vec
    names = None
    L1 = cfg.N_STATES
    arrs, sds = {}, np.zeros((L1, cfg.D), np.float32)
    for l in range(L1):
        common = pool_dirs(cells, HON, l, probe=True, q=q)
        sd = common["sd"]
        dirs = {f"cell|{c}": common["per_cell"][c] for c in HON}
        for f in "abcd":
            dirs[f"frame_{f}"] = unit_vec(np.mean([dirs[f"cell|{c}"] for c in HON if c.endswith(f"|{f}")], 0))
        dirs["all_frames"] = unit_vec(sum(dirs[f"frame_{f}"] for f in "abcd"))
        dirs["dm_all"], dirs["probe_all"] = common["dm"], common["probe"]
        for ctrl in ("K1", "K2"):
            per = []
            for c in [c for c in ALL if c.startswith(ctrl + "|")]:
                Xtr, ytr = cells[c]["Xtr"][:, l].astype(np.float64) / sd, cells[c]["ytr"]
                per.append(unit_vec(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)))
            dirs[ctrl] = unit_vec(np.mean(per, 0))
        rng = np.random.default_rng(l)
        for k in range(N_RANDOM):
            dirs[f"random_{k}"] = unit_vec(rng.standard_normal(cfg.D))
        if names is None:
            names = sorted(dirs)
            for nm in names:
                arrs[nm] = np.zeros((L1, cfg.D), np.float32)
        for nm in names:
            arrs[nm][l] = dirs[nm]
        sds[l] = sd
        print(f"[sweep] directions at hidden index {l} (layer {l - 1})", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "directions_all.npz", sd=sds, names=np.array(names), **arrs)
    d2p = out / "directions.npz" if (out / "directions.npz").exists() else RES / "directions.npz"   # the same setting's two-depth file when it exists
    chk = {}
    if d2p.exists():
        D2 = np.load(d2p)
        chk = {f"{nm}|L{L}": float(arrs[nm][L + 1] @ D2[f"{nm}|L{L}"]) for L in LAYERS for nm in ("dm_all", "probe_all", "frame_c", "cell|C1|lying|c")}
    info = {"model": cfg.MODEL, "n_states": L1, "names": names, "check_vs_directions_npz": chk}
    if robust:
        info.update({"sd_floor_q": q, "checked_against": str(d2p) if d2p.exists() else None})
    (out / "directions_all.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(chk, indent=1))


def cmd_score_sweep(cond="baseline", qs=None):
    """Within-stratum readout per position and hidden-state index: one table per position, layers as rows, groups as
    columns (mean over the six task readouts of the pooled-over-strata group means), plus the decision-rule check.
    qs: floor settings, each read from and written under robust_floor/<setting>/ (None = today's files)."""
    if qs is None:
        score_sweep_run(cond, RES, None, robust=False)
        return
    for q in qs:
        score_sweep_run(cond, robust_res(q), q, robust=True)


def score_sweep_run(cond, rdir, q, robust):
    from .coherence import qtag
    D = np.load(rdir / "directions_all.npz")
    names = [str(x) for x in D["names"]]
    strata = strata_of()
    L1 = cfg.N_STATES
    res = {"model": cfg.MODEL, "cond": cond, "positions": list(SWEEP_POS), "readouts": {}, "tables": {}, "crossings": []}
    if robust:
        res["sd_floor_q"] = q
    per_pos = {pos: {} for pos in SWEEP_POS}       # pos -> layer -> list of (group -> mean) over readouts
    rng = np.random.default_rng(0)
    for t in TASKS:
        p = SWEEP_OUT / f"{t}.npz"
        if not p.exists():
            continue
        A = {pos: None for pos in SWEEP_POS}
        meta = json.loads(str(np.load(p)["meta"]))
        kinds = sorted({x["kind"] for x in meta})
        outcomes = ["dishonest"] + (["not_done"] if t == "FT" else [])
        with np.load(p) as Z:
            for pos in SWEEP_POS:
                A[pos] = Z[pos]
        for kind in kinds:
            for oc in outcomes:
                base = [i for i, x in enumerate(meta) if x["kind"] == kind and x.get(oc) is not None and x["reply_tokens"] > 0]
                groups = {st: [i for i in base if strata[meta[i]["item_id"]] == st] for st in sorted({strata[meta[i]["item_id"]] for i in base})}
                groups = {st: rows for st, rows in groups.items() if 5 <= sum(meta[i][oc] for i in rows) and 5 <= sum(not meta[i][oc] for i in rows)}
                if not groups:
                    res["readouts"][f"{t}|{kind}|{oc}"] = {"note": "no stratum with 5 items of each class"}
                    continue
                key = f"{t}|{kind}|{oc}"
                res["readouts"][key] = {"strata": {st: {"n": len(r), "n_positive": int(sum(meta[i][oc] for i in r))} for st, r in groups.items()}, "by": {}}
                for pos in SWEEP_POS:
                    for l in range(L1):
                        sd = D["sd"][l].astype(np.float64)
                        acc = {}
                        w_all = []
                        for st, rows in groups.items():
                            y = np.array([meta[i][oc] for i in rows], bool)
                            X = A[pos][rows, l].astype(np.float64) / sd
                            fold = {nm: auroc(X @ D[nm][l].astype(np.float64), y) for nm in names}
                            fold = {nm: max(a, 1 - a) for nm, a in fold.items()}
                            ent = {g: float(np.mean([fold[nm] for nm in names if f(nm)])) for g, f in GROUPS.items()}
                            ent["random_95"] = float(np.percentile([fold[nm] for nm in names if nm.startswith("random_")], 95))
                            ent["probe"] = probe_ceiling(X, y)
                            w = int(y.sum())
                            for k2, v in ent.items():
                                if v is not None:
                                    acc.setdefault(k2, []).append((w, v))
                        pooled = {k2: float(np.average([v for _, v in lst], weights=[w for w, _ in lst])) for k2, lst in acc.items()}
                        res["readouts"][key]["by"][f"{pos}|{l}"] = pooled
                        per_pos[pos].setdefault(l, []).append(pooled)
                        for g in GROUPS:
                            if g != "random (16)" and pooled.get(g, 0) > pooled["random_95"]:
                                res["crossings"].append({"readout": key, "position": pos, "hidden_index": l, "layer": l - 1, "group": g,
                                                         "mean_folded": pooled[g], "random_95": pooled["random_95"]})
                print(f"[sweep] {key}: scored {len(SWEEP_POS)} positions x {L1} states", flush=True)
    short = {"act cells (c)": "act", "self-describe cells (d)": "self", "own-reply cells (a, b)": "own", "frame levers": "frames",
             "shared axes (dm_all, probe_all)": "shared", "all_frames": "all_frames", "K1": "K1", "K2": "K2", "random (16)": "random"}
    md = [f"# Position-and-layer sweep (D-063 item 1), {cfg.MODEL}, cond {cond}" + (f", sd floor {qtag(q)}" if robust else ""), "",
          "Mean over the task readouts of the within-stratum pooled group folded AUROC; probe = within-task 5-fold ceiling. "
          "Rows are hidden-state indices (layer = index - 1). Crossings (a group mean above the random 95th percentile on a "
          "single readout, position and layer) are listed at the end.", ""]
    for pos in SWEEP_POS:
        res["tables"][pos] = {}
        md += [f"## position {pos}", "", "| index | " + " | ".join(short.values()) + " | random 95th | probe |", "| --- |" + " --- |" * (len(short) + 2)]
        for l in range(L1):
            lst = per_pos[pos].get(l, [])
            if not lst:
                continue
            row = {g: float(np.mean([e[g] for e in lst if g in e])) for g in GROUPS}
            row["random_95"] = float(np.mean([e["random_95"] for e in lst])); row["probe"] = float(np.mean([e["probe"] for e in lst if e.get("probe") is not None]))
            res["tables"][pos][str(l)] = row
            md.append(f"| {l} | " + " | ".join(f"{row[g]:.2f}" for g in GROUPS) + f" | {row['random_95']:.2f} | {row['probe']:.2f} |")
        md.append("")
    hon = [c for c in res["crossings"] if c["group"] not in ("K1", "K2")]
    ctl = [c for c in res["crossings"] if c["group"] in ("K1", "K2")]
    n_cells = sum(len(v["by"]) for v in res["readouts"].values() if "by" in v)
    res["decision"] = {"honesty_crossings": len(hon), "control_crossings": len(ctl), "cells": n_cells,
                       "branch": "finding stands" if not hon else "position-specific: a group crosses the random 95th"}
    md += [f"## crossings: honesty groups {len(hon)}, control groups {len(ctl)}, of {n_cells} readout x position x index cells", ""]
    for c in sorted(hon, key=lambda c: -c["mean_folded"])[:40]:
        md.append(f"- {c['readout']} {c['position']} index {c['hidden_index']} {c['group']}: {c['mean_folded']:.2f} vs random 95th {c['random_95']:.2f}")
    (rdir / cond).mkdir(parents=True, exist_ok=True)
    (rdir / cond / "sweep.json").write_text(json.dumps(res, indent=1, default=float))
    (rdir / cond / "sweep.md").write_text("\n".join(md) + "\n")
    print(json.dumps(res["decision"], indent=1))
    print(f"[sweep] -> {rdir / cond / 'sweep.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["positions", "acts", "directions", "score", "score-strat",
                                      "positions-sweep", "acts-sweep", "directions-all", "score-sweep"])
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--sd-floor-q", default=None, help="directions, score-strat, directions-all, score-sweep: comma list of standardization-floor "
                    "quantiles ('none' allowed); given at all, every setting reads and writes under robust_floor/<setting>/ only")
    a = ap.parse_args()
    from .coherence import parse_qs
    qs = parse_qs(a.sd_floor_q)
    if a.stage == "positions":
        cmd_positions(a.cond)
    elif a.stage == "acts":
        cmd_acts(a.cond, a.n, a.dry)
    elif a.stage == "directions":
        cmd_directions(qs)
    elif a.stage == "score":
        cmd_score(a.cond)
    elif a.stage == "score-strat":
        cmd_score_strat(a.cond, qs)
    elif a.stage == "positions-sweep":
        cmd_positions_sweep(a.cond)
    elif a.stage == "acts-sweep":
        cmd_acts_sweep(a.cond, a.n, a.dry)
    elif a.stage == "directions-all":
        cmd_directions_all(qs)
    else:
        cmd_score_sweep(a.cond, qs)


if __name__ == "__main__":
    main()
