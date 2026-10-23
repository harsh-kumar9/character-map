"""Locations on ada. Everything large lives under /ada1/u/harsh; the HF cache root is shared with other projects."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"      # small tables and figures, tracked in git
OUTPUTS = REPO / "outputs"      # activations and other large files, not tracked

HF_ROOT = Path(os.environ.get("HF_HUB_CACHE", "/ada1/u/harsh/.cache/huggingface"))

MODELS = {
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-4-31b-it": "google/gemma-4-31b-it",
    "qwen3.5-27b": "Qwen/Qwen3.5-27B",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
}
SAE_REPOS = {
    "gemma-3-4b-it": "google/gemma-scope-2-4b-it",
}


def snapshot(repo_id: str, allow_patterns: list[str] | None = None) -> Path:
    """Path of a cached HF snapshot. Never downloads (downloads happen on ada, outside jobs)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id, cache_dir=str(HF_ROOT), local_files_only=True,
                                  allow_patterns=allow_patterns))


def sae_dir(model: str, layer: int, width: str = "65k", l0: str = "medium", site: str = "resid_post") -> Path:
    sub = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
    return snapshot(SAE_REPOS[model], allow_patterns=[f"{sub}/*"]) / sub
