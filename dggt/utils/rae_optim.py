"""RAEv2-style optimizer and LR scheduler helpers."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class MuonAdamW(Optimizer):
    """Composite optimizer: GMuon for 2D params, AdamW for the rest."""

    def __init__(self, muon_opt: Optimizer, adamw_opt: Optimizer) -> None:
        self._muon = muon_opt
        self._adamw = adamw_opt
        self.param_groups = muon_opt.param_groups + adamw_opt.param_groups
        self.defaults: dict[str, Any] = {}

    @property
    def state(self) -> dict:
        merged: dict = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        self._muon.step(closure=closure)
        self._adamw.step(closure=closure)

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self._muon.param_groups + self._adamw.param_groups


def build_rae_optimizer(
    param_groups: list[dict[str, Any]],
    *,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_coefficients_preset: str = "POLAR_EXPRESS_COEFFICIENTS",
    ns_use_kernels: bool = False,
) -> tuple[Optimizer, str]:
    """Build the optimizer used by RAEv2 stage-2 configs.

    ``gmuon`` intentionally raises if ``gram_newton_schulz`` is unavailable; a
    silent AdamW fallback would make experiments look RAE-aligned when they are
    not.
    """
    opt_type = str(optimizer_type).lower()
    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=float(lr),
            betas=betas,
            eps=float(eps),
        )
        return optimizer, f"AdamW(lr={lr}, betas={betas}, wd={weight_decay})"

    if opt_type != "gmuon":
        raise ValueError("optimizer_type must be 'gmuon' or 'adamw', got " f"{optimizer_type!r}")

    try:
        from gram_newton_schulz import Muon as GMuon
    except ImportError as exc:
        raise ImportError(
            "RAEv2 t2i uses optimizer_type='gmuon', but package "
            "`gram_newton_schulz` is not importable in this environment."
        ) from exc

    muon_params: list[torch.nn.Parameter] = []
    fallback_groups: list[dict[str, Any]] = []
    for group in param_groups:
        params = [p for p in group.get("params", []) if p.requires_grad]
        two_d = [p for p in params if p.ndim == 2]
        rest = [p for p in params if p.ndim != 2]
        muon_params.extend(two_d)
        if rest:
            fallback = {k: v for k, v in group.items() if k != "params"}
            fallback["params"] = rest
            fallback_groups.append(fallback)

    if not muon_params:
        raise ValueError("optimizer_type='gmuon' selected but no trainable 2D parameters were found.")
    if not fallback_groups:
        fallback_groups = [{"params": [torch.nn.Parameter(torch.empty(0))], "weight_decay": 0.0}]

    adamw_opt = torch.optim.AdamW(
        fallback_groups,
        lr=float(lr),
        betas=betas,
        eps=float(eps),
    )
    gmuon_opt = GMuon(
        muon_params,
        lr=float(lr),
        momentum=float(momentum),
        nesterov=bool(nesterov),
        weight_decay=float(weight_decay),
        ns_coefficients_preset=str(ns_coefficients_preset),
        ns_use_kernels=bool(ns_use_kernels),
        adjust_lr="rms_norm",
    )
    optimizer = MuonAdamW(gmuon_opt, adamw_opt)
    msg = (
        f"GMuon(lr={lr}, momentum={momentum}, preset={ns_coefficients_preset}, "
        f"kernels={ns_use_kernels}, {len(muon_params)} 2D params, "
        f"{sum(len(g['params']) for g in fallback_groups)} fallback)"
    )
    return optimizer, msg


def build_rae_scheduler(
    optimizer: Optimizer,
    *,
    scheduler_type: str,
    warmup_steps: int,
    decay_end_steps: int,
    base_lr: float,
    final_lr: float,
    warmup_from_zero: bool,
) -> LambdaLR:
    """RAEv2 scheduler: linear/cosine decay to ``final_lr`` after warmup."""
    warmup_steps = max(0, int(warmup_steps))
    decay_end_steps = max(int(decay_end_steps), warmup_steps)
    total_decay_steps = max(decay_end_steps - warmup_steps, 1)
    final_ratio = float(final_lr) / float(base_lr) if float(base_lr) > 0.0 else 1.0

    for group in optimizer.param_groups:
        group["lr"] = float(base_lr)

    sched_type = str(scheduler_type).lower()
    if sched_type == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_steps == 0 or not bool(warmup_from_zero):
                    return 1.0
                return float(step + 1) / float(warmup_steps)
            if step >= decay_end_steps:
                return final_ratio
            progress = float(step - warmup_steps) / float(total_decay_steps)
            return 1.0 - (1.0 - final_ratio) * progress
    elif sched_type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                if warmup_steps == 0 or not bool(warmup_from_zero):
                    return 1.0
                return float(step + 1) / float(warmup_steps)
            if step >= decay_end_steps:
                return final_ratio
            progress = float(step - warmup_steps) / float(total_decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine
    else:
        raise ValueError("scheduler_type must be 'linear' or 'cosine', got " f"{scheduler_type!r}")

    return LambdaLR(optimizer, lr_lambda)
