"""LoRA from scratch: a low-rank adapter wrapped around nn.Linear (module 36).

W' = W + (alpha / r) * B @ A with A (r, in) and B (out, r). A starts
kaiming-uniform(a=sqrt(5)) like nn.Linear's own weight, B starts at zero, so the
adapted model computes exactly the base model's function until the first step.

Wrapping keeps the base Linear as a child called ``base``; full state_dict keys
therefore change from ``...c_attn.weight`` to ``...c_attn.base.weight``.
``unwrap_lora`` merges and puts the plain Linear back (original key names).

Interface (syllabus E.8):
    LoRALinear(base, r=8, alpha=16, dropout=0.0)  .merge() .unmerge() .merged .delta_weight()
    apply_lora(model, target_names=("c_attn", "c_proj"), r=8, alpha=16, dropout=0.0) -> list[str]
    mark_only_lora_trainable(model)
    lora_state_dict(model) -> dict
    load_lora_state_dict(model, sd) -> torch.nn.modules.module._IncompatibleKeys
    merge_all(model) / unmerge_all(model) / unwrap_lora(model) -> list[str]
    count_trainable(model) -> int

Imports torch only. Read it next to content/courses/pytorch/36-capstone-lora-finetuning.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_KEYS = ("lora_A", "lora_B")


class LoRALinear(nn.Module):
    """y = base(x) + scale * dropout(x) @ A^T @ B^T, base frozen, scale = alpha / r."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if not isinstance(r, int) or r <= 0:
            raise ValueError(f"r must be a positive integer, got {r!r}")
        self.base = base                                   # same Parameter objects, not copies
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r, device=dev, dtype=dt))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        self.base.requires_grad_(False)
        self.reset_parameters()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # nn.Linear's default for weights
        nn.init.zeros_(self.lora_B)                            # delta W = 0 at init

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale        # (out, in), same layout as base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)                                       # (..., out)
        if self.merged:
            return y                                           # delta already inside base.weight
        h = F.linear(self.lora_dropout(x), self.lora_A)        # (..., r)
        return y + F.linear(h, self.lora_B) * self.scale       # (..., out)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scale={self.scale:g}, merged={self.merged}"


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def apply_lora(model: nn.Module, target_names=("c_attn", "c_proj"), r: int = 8,
               alpha: float = 16, dropout: float = 0.0) -> list[str]:
    """Replace every nn.Linear whose last name component is in target_names.

    Returns the replaced module names in named_modules order. Freezes nothing
    except each wrapped base; call mark_only_lora_trainable for the rest.
    """
    targets = set(target_names)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.rpartition(".")[2] in targets]
    for name in names:                       # collect first, then mutate: never edit while iterating
        parent, child = _parent(model, name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r=r, alpha=alpha, dropout=dropout))
    return names


def lora_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(name.rpartition(".")[2] in LORA_KEYS)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # tied params once


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if k.rpartition(".")[2] in LORA_KEYS}


def load_lora_state_dict(model: nn.Module, sd: dict):
    """Load adapter weights into a model that already went through apply_lora.

    strict=False because the adapter file has no base weights; we then check the
    result ourselves: nothing unexpected, and every missing key is a base key.
    """
    result = model.load_state_dict(sd, strict=False)
    missing_lora = [k for k in result.missing_keys if k.rpartition(".")[2] in LORA_KEYS]
    if result.unexpected_keys or missing_lora:
        raise KeyError(f"adapter mismatch: unexpected={result.unexpected_keys} missing={missing_lora}")
    return result


def merge_all(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.merge()


def unmerge_all(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.unmerge()


def unwrap_lora(model: nn.Module) -> list[str]:
    """Merge every adapter and put the plain nn.Linear back (original state_dict keys)."""
    names = []
    for name, m in lora_modules(model):
        m.merge()
        parent, child = _parent(model, name)
        setattr(parent, child, m.base)
        names.append(name)
    return names
