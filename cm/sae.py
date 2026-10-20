"""Gemma Scope 2 JumpReLU SAEs, loaded straight from params.safetensors (D-021).

Adapted from thought-atlas mode_control/mc/neuro_sae.py (Qwen-Scope TopK loader). Gemma Scope 2 stores
w_enc [d_model, F], b_enc [F], threshold [F], w_dec [F, d_model], b_dec [d_model] in float32.
Encoder: pre = x @ w_enc + b_enc; f = pre * (pre > threshold). Decoder: x_hat = f @ w_dec + b_dec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file


@dataclass
class JumpReLUSAE:
    w_enc: torch.Tensor
    b_enc: torch.Tensor
    threshold: torch.Tensor
    w_dec: torch.Tensor
    b_dec: torch.Tensor
    config: dict

    @classmethod
    def load(cls, folder: Path, device: str = "cpu") -> "JumpReLUSAE":
        p = load_file(str(Path(folder) / "params.safetensors"), device=device)
        cfg = json.loads((Path(folder) / "config.json").read_text())
        assert cfg.get("architecture") == "jump_relu", cfg
        assert not cfg.get("affine_connection", False), "affine (skip) transcoders are not handled here"
        return cls(p["w_enc"], p["b_enc"], p["threshold"], p["w_dec"], p["b_dec"], cfg)

    @property
    def width(self) -> int:
        return self.w_enc.shape[1]

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.w_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre(x)
        return pre * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.w_dec + self.b_dec

    def feature_pre(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Pre-activations of one feature per row: x [n, d], idx [n] -> [n]."""
        return (x.float() * self.w_enc[:, idx].T).sum(-1) + self.b_enc[idx]
