"""Model loading and residual-stream capture.

Gemma-3-4B-IT is a multimodal checkpoint (Gemma3ForConditionalGeneration); its text decoder sits at
model.model.language_model.layers. The Gemma Scope 2 hook point "model.layers.L.output" is the output of
decoder layer L in the text-only layout, which is what `ResidCapture` records.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .paths import MODELS, snapshot


def load(model: str = "gemma-3-4b-it", device: str = "cuda", dtype: torch.dtype = torch.bfloat16, attn: str = "sdpa"):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    path = snapshot(MODELS[model])
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation=attn)
    m.to(device).eval()
    return m, tok


def text_model(m):
    """The text decoder (embeddings + layers + final norm), skipping the vision tower and lm_head."""
    for path in ("model.language_model", "language_model.model", "model"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "layers"):
            return obj
    raise AttributeError("no text decoder with .layers found")


def decoder_layers(m):
    return text_model(m).layers


@contextmanager
def capture_resid(m, layers: list[int]):
    """Record the output of each listed decoder layer into a dict {layer: tensor [b, t, d]}."""
    store: dict[int, torch.Tensor] = {}
    blocks = decoder_layers(m)
    handles = []
    for L in layers:
        def hook(_mod, _inp, out, L=L):
            store[L] = (out[0] if isinstance(out, tuple) else out).detach()
        handles.append(blocks[L].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer(m, layer: int, vec: torch.Tensor):
    """Add `vec` ([d], already scaled) to the output of decoder layer `layer` at every position (hidden-state index
    layer + 1). The residual stream carries it forward unchanged except for what later layers do with it."""
    block = decoder_layers(m)[layer]

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
        return out + vec.to(out.dtype)

    h = block.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
