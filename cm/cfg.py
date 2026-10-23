"""Model registry for the pipeline (D-047). CM_MODEL selects the subject model; gemma-3-4b-it is the original run and
keeps its outputs at the unprefixed paths, every other model writes under outputs/<tag>/ and results/<tag>/.

Readout depths for a new model sit at the same fractions of depth as Gemma-3-4B's four SAE layers (9, 17, 22, 29 of
34): about 0.26, 0.50, 0.65 and 0.85 of the way through. The steering layer is the second depth (the 0.5 point, as
layer 17 was), the causal readouts are the third and fourth, the landscape uses the second and fourth.
Hidden-state index L + 1 is the output of decoder layer L (index 0 = embeddings).
"""
from __future__ import annotations

import os

from .paths import OUTPUTS, RESULTS

SPECS = {
    "gemma-3-4b-it": dict(hf="google/gemma-3-4b-it", d=2560, n_layers=34, depths=(9, 17, 22, 29), tag="", sae=True),
    "gemma-4-31b-it": dict(hf="google/gemma-4-31b-it", d=5376, n_layers=60, depths=(16, 30, 39, 51), tag="gemma-4-31b-it", sae=False),
    "qwen3.5-27b": dict(hf="Qwen/Qwen3.5-27B", d=5120, n_layers=64, depths=(17, 32, 42, 54), tag="qwen3.5-27b", sae=False),
    "qwen3.6-27b": dict(hf="Qwen/Qwen3.6-27B", d=5120, n_layers=64, depths=(17, 32, 42, 54), tag="qwen3.6-27b", sae=False),
    "qwen3.8-27b": dict(hf="Qwen/Qwen3.8-27B", d=5120, n_layers=64, depths=(17, 32, 42, 54), tag="qwen3.8-27b", sae=False),
}
MODEL = os.environ.get("CM_MODEL", "gemma-3-4b-it")
if MODEL not in SPECS:
    raise SystemExit(f"CM_MODEL={MODEL!r} is not in cm.cfg.SPECS: {sorted(SPECS)}")
SPEC = SPECS[MODEL]
HF_ID = SPEC["hf"]
D = SPEC["d"]
N_STATES = SPEC["n_layers"] + 1
DEPTH_LAYERS = tuple(SPEC["depths"])
DEPTHS = {L: L + 1 for L in DEPTH_LAYERS}          # decoder layer -> hidden-state index
L_STEER = DEPTH_LAYERS[1]
L_OUT = (DEPTH_LAYERS[2], DEPTH_LAYERS[3])
L_LAND = (DEPTH_LAYERS[1], DEPTH_LAYERS[3])
HAS_SAE = SPEC["sae"]
TAG = SPEC["tag"]
IS_GEMMA4 = MODEL.startswith("gemma-4")
IS_QWEN = MODEL.startswith("qwen")


def results_dir():
    return RESULTS / TAG if TAG else RESULTS


def outputs_dir():
    return OUTPUTS / TAG if TAG else OUTPUTS


def acts_dir():
    return outputs_dir() / "acts" / "stimuli"


def chat_kwargs():
    """Extra apply_chat_template arguments: Qwen's template takes enable_thinking; Gemma's ignores it."""
    return {"enable_thinking": False} if IS_QWEN else {}


def vllm_kwargs(model_id: str | None = None):
    """Per-model vLLM engine arguments (gemma-4 needs the per-layer config override; Qwen3.5-family is hybrid)."""
    from functools import partial
    mid = model_id or HF_ID
    kw = {}
    if "gemma-4" in mid.lower():
        from .pv.judge import hetero_safe_overrides
        kw["hf_overrides"] = partial(hetero_safe_overrides, model_id=mid)
    if "qwen3." in mid.lower():
        kw["max_num_seqs"] = 384
    return kw
