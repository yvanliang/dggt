"""PerTokenNoiseScheduler — per-token noise level + z_init composition.

Implements research_plan.md §3.5.3 (t_tok) and §3.5.4 (z_init).

* `t_tok[b,s,p,1] = clip(base_t * (M_source + gamma_dest * M_dest)
                         + (1 - M_preserve) * eps_floor, 0, 1)`
* `z_init = M_preserve * z_clean
          + M_source  * eps
          + M_dest    * (z_splat + sigma_partial * eps)`

Where `eps ~ N(0, I)` is fresh noise per step. The scheduler is stateless and
carries no learnable parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerTokenNoiseScheduler(nn.Module):
    def __init__(
        self,
        gamma_dest: float = 0.4,
        eps_floor: float = 0.05,
        sigma_partial: float = 0.3,
    ) -> None:
        super().__init__()
        self.gamma_dest = float(gamma_dest)
        self.eps_floor = float(eps_floor)
        self.sigma_partial = float(sigma_partial)

    @staticmethod
    def sample_base_t(
        batch_size: int,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample base_t ~ U(0, 1) per sample."""
        return torch.rand((int(batch_size),), device=device, generator=generator)

    def build_t_tok(
        self,
        base_t: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
    ) -> torch.Tensor:
        """Compose per-token noise schedule.

        Parameters
        ----------
        base_t : [B] in [0, 1]
        M_preserve, M_source, M_dest : [B, S, P, 1]

        Returns
        -------
        t_tok : [B, S, P, 1] in [0, 1]
        """
        self._check_mask_shapes(M_preserve, M_source, M_dest, base_t.shape[0])
        edit_weight = M_source + self.gamma_dest * M_dest
        base = base_t.view(-1, 1, 1, 1)
        t_tok = base * edit_weight + (1.0 - M_preserve) * self.eps_floor
        return t_tok.clamp_(0.0, 1.0)

    def compose_z_init(
        self,
        z_clean: torch.Tensor,
        z_splat: torch.Tensor,
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend the noisy starting latent.

        Returns `(z_init, eps)`. `eps` is the raw noise drawn, also returned
        so callers can reuse it for deterministic tests or for alternate
        reconstructions.
        """
        if z_clean.shape != z_splat.shape:
            raise ValueError(
                f"z_clean / z_splat shape mismatch: {tuple(z_clean.shape)} vs {tuple(z_splat.shape)}"
            )
        B = int(z_clean.shape[0])
        self._check_mask_shapes(M_preserve, M_source, M_dest, B)
        if z_clean.shape[:-1] != M_preserve.shape[:-1] or z_clean.shape[2] != M_preserve.shape[2]:
            raise ValueError(
                "z_clean and masks must share [B, S, P] axes; got "
                f"{tuple(z_clean.shape)} vs {tuple(M_preserve.shape)}"
            )
        eps = torch.empty_like(z_clean)
        eps.normal_(generator=generator)
        z_init = (
            M_preserve * z_clean
            + M_source * eps
            + M_dest * (z_splat + self.sigma_partial * eps)
        )
        return z_init, eps

    @staticmethod
    def _check_mask_shapes(
        M_preserve: torch.Tensor,
        M_source: torch.Tensor,
        M_dest: torch.Tensor,
        expected_B: int,
    ) -> None:
        for name, t in (("M_preserve", M_preserve), ("M_source", M_source), ("M_dest", M_dest)):
            if t.dim() != 4 or t.shape[-1] != 1:
                raise ValueError(f"{name} must be [B, S, P, 1], got {tuple(t.shape)}")
            if t.shape[0] != expected_B:
                raise ValueError(f"{name}.shape[0] {t.shape[0]} != expected B {expected_B}")
        if M_preserve.shape != M_source.shape or M_preserve.shape != M_dest.shape:
            raise ValueError("All three masks must share shape")
