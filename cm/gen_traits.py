"""Generate trait files for new persona vectors (D-020) with Claude Sonnet 5.5 through OpenRouter.

Upstream (safety-research/persona_vectors) generated its trait files with Claude-3.7-Sonnet, thinking budget 5000,
max_tokens 16000. Sonnet 5.5 no longer takes a thinking budget (adaptive thinking only) and rejects custom
temperature, so the closest setting is adaptive thinking at effort "high", max_tokens 16000, default sampling.

Outputs, per trait (slug = name with spaces and hyphens as underscores):
  data/traits/raw/<slug>.json      every API attempt: request settings, usage, cost, finish reason, text
  data/traits/full/<slug>.json     the parsed, validated object
  data/traits/extract/<slug>.json  instructions + eval prompt(s) + extraction questions (upstream layout)
  data/traits/eval/<slug>.json     instructions + eval prompt(s) + held-out evaluation questions
Unconditional split follows upstream's 20/20 with no overlap (first 20 extraction, last 20 evaluation).
Conditional split keeps matched pairs together: pairs 1-10 extraction, 11-20 evaluation.

Resumable: a trait whose full/ file exists and validates is skipped.

Run on ada (reads OPENROUTER_API_KEY from the environment; scripts/gen_traits.sh sets it):
  python -m cm.gen_traits --dry-run            render prompts only
  python -m cm.gen_traits --only deceptive     one trait
  python -m cm.gen_traits                      everything in data/traits/specs.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from vendor.persona_vectors.prompts import PROMPTS

from .paths import REPO
from .trait_prompts import CONDITIONAL

TRAITS = REPO / "data" / "traits"
MODEL = "anthropic/claude-sonnet-5.5"
SETTINGS = {"max_tokens": 16000, "reasoning": {"effort": "high"},
            "provider": {"order": ["Anthropic"], "allow_fallbacks": True}, "usage": {"include": True}}
URL = "https://openrouter.ai/api/v1/chat/completions"


def slug(name: str) -> str:
    return re.sub(r"[ -]+", "_", name.strip().lower())


def render(spec: dict, kind: str) -> str:
    if kind == "unconditional":
        return PROMPTS["generate_trait"].format(TRAIT=spec["name"], trait_instruction=spec["description"],
                                                question_instruction="")
    return CONDITIONAL.format(TRAIT=spec["name"], condition=spec["condition"], virtue=spec["virtue"],
                              excess=spec["excess"], deficiency=spec["deficiency"], grounding=spec["grounding"])


def call(prompt: str, max_tokens: int) -> dict:
    key = os.environ["OPENROUTER_API_KEY"]
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], **SETTINGS, "max_tokens": max_tokens}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-Title": "character-map trait generation"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
            if "choices" not in d:
                raise RuntimeError(f"no choices: {str(d)[:300]}")
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            return {"id": d.get("id"), "model": d.get("model"), "provider": d.get("provider"),
                    "finish_reason": ch.get("finish_reason"), "native_finish_reason": ch.get("native_finish_reason"),
                    "usage": d.get("usage"), "text": msg.get("content") or "",
                    "reasoning_chars": len(msg.get("reasoning") or ""), "max_tokens": max_tokens}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (408, 409, 429) or e.code >= 500:
                time.sleep(10 * 2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * 2 ** attempt)
    raise RuntimeError("OpenRouter call failed after 5 attempts")


def parse(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(t[t.index("{"): t.rindex("}") + 1])


def has_slots(p, slots) -> bool:
    return isinstance(p, str) and all(s in p for s in slots)


def validate(d: dict, kind: str) -> list[str]:
    err = []
    ins, qs = d.get("instruction"), d.get("questions")
    keys = ("pos", "neg") if kind == "unconditional" else ("virtue", "excess", "deficiency")
    if not (isinstance(ins, list) and len(ins) == 5 and all(isinstance(i, dict) and all(isinstance(i.get(k), str) and i[k].strip() for k in keys) for i in ins)):
        err.append(f"instruction: need 5 entries with {keys}")
    if kind == "unconditional":
        if not (isinstance(qs, list) and len(qs) == 40 and all(isinstance(q, str) and q.strip() for q in qs)):
            err.append(f"questions: need 40 strings, got {len(qs) if isinstance(qs, list) else type(qs).__name__}")
        elif len(set(qs)) != 40:
            err.append("questions: duplicates")
        if not has_slots(d.get("eval_prompt"), ("{question}", "{answer}")):
            err.append("eval_prompt: missing {question}/{answer}")
    else:
        ok = isinstance(qs, list) and len(qs) == 40 and all(isinstance(q, dict) and isinstance(q.get("question"), str) and isinstance(q.get("condition"), bool) for q in qs)
        if not ok:
            err.append("questions: need 40 objects with question and boolean condition")
        else:
            if sum(q["condition"] for q in qs) != 20:
                err.append("questions: need exactly 20 with condition true")
            pairs = {}
            for q in qs:
                pairs.setdefault(q.get("pair"), []).append(q["condition"])
            if sorted(pairs) != list(range(1, 21)) or any(sorted(v) != [False, True] for v in pairs.values()):
                err.append("questions: need pairs 1-20, each with one true and one false")
        if not has_slots(d.get("eval_prompt_virtue"), ("{question}", "{answer}", "{condition_status}")):
            err.append("eval_prompt_virtue: missing {question}/{answer}/{condition_status}")
        for k in ("eval_prompt_excess", "eval_prompt_deficiency"):
            if not has_slots(d.get(k), ("{question}", "{answer}")):
                err.append(f"{k}: missing {{question}}/{{answer}}")
    return err


def split(d: dict, kind: str) -> tuple[dict, dict]:
    base = {k: v for k, v in d.items() if k != "questions"}
    qs = d["questions"]
    if kind == "unconditional":
        return {**base, "questions": qs[:20]}, {**base, "questions": qs[20:]}
    return ({**base, "questions": [q for q in qs if q["pair"] <= 10]},
            {**base, "questions": [q for q in qs if q["pair"] > 10]})


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def gen_one(spec: dict, kind: str) -> dict:
    s = slug(spec["name"])
    full = TRAITS / "full" / f"{s}.json"
    if full.exists() and not validate(json.loads(full.read_text()), kind):
        return {"trait": s, "status": "skip (done)"}
    prompt = render(spec, kind)
    attempts, parsed, errs = [], None, ["not run"]
    for i in range(3):
        mt = 16000 if i == 0 else 32000
        r = call(prompt, mt)
        try:
            cand = parse(r["text"])
            errs = validate(cand, kind)
        except Exception as e:  # unparseable output
            cand, errs = None, [f"parse: {e}"]
        attempts.append({**r, "errors": errs})
        if not errs:
            parsed = cand
            break
    write(TRAITS / "raw" / f"{s}.json", {"trait": spec["name"], "kind": kind, "spec": spec, "model": MODEL,
                                         "settings": SETTINGS, "prompt": prompt, "attempts": attempts})
    cost = sum(((a.get("usage") or {}).get("cost") or 0) for a in attempts)
    if parsed is None:
        return {"trait": s, "status": "FAILED", "errors": errs, "attempts": len(attempts), "cost": cost}
    parsed = {"trait": spec["name"], "kind": kind, **parsed}
    write(full, parsed)
    ex, ev = split(parsed, kind)
    write(TRAITS / "extract" / f"{s}.json", ex)
    write(TRAITS / "eval" / f"{s}.json", ev)
    return {"trait": s, "status": "ok", "attempts": len(attempts), "cost": cost,
            "provider": attempts[-1].get("provider"), "finish": attempts[-1].get("finish_reason")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated trait names or slugs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    specs = json.loads((TRAITS / "specs.json").read_text())
    jobs = [(s, k) for k in ("unconditional", "conditional") for s in specs[k]]
    if args.only:
        want = {slug(x) for x in args.only.split(",")}
        jobs = [(s, k) for s, k in jobs if slug(s["name"]) in want]
    if args.dry_run:
        for s, k in jobs:
            print(f"===== {slug(s['name'])} ({k}) =====\n{render(s, k)}")
        return
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda sk: gen_one(*sk), jobs))
    for r in results:
        print(json.dumps(r), flush=True)
    total = sum(r.get("cost") or 0 for r in results)
    print(f"done: {sum(r['status'] == 'ok' for r in results)} ok, "
          f"{sum(r['status'].startswith('skip') for r in results)} skipped, "
          f"{sum(r['status'] == 'FAILED' for r in results)} failed; cost ${total:.2f}", flush=True)


if __name__ == "__main__":
    main()
