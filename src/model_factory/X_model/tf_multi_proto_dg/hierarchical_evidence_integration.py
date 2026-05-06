from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalEvidenceIntegration(nn.Module):
    """Fuse local and global transparent evidence into a single interpretable concept state."""

    def __init__(self, role_dim: int, concept_norm: str = "layernorm") -> None:
        super().__init__()
        self.role_dim = int(role_dim)
        self.concept_norm = str(concept_norm)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.concept_norm == "layernorm":
            return F.layer_norm(x, (int(x.shape[-1]),))
        if self.concept_norm == "l2":
            return F.normalize(x, dim=-1)
        if self.concept_norm == "none":
            return x
        raise ValueError(f"Unsupported concept_norm: {self.concept_norm}")

    @staticmethod
    def _focus(weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim != 2:
            raise ValueError(f"Expected weights [B, R], got {tuple(weights.shape)}.")
        if weights.shape[1] <= 1:
            return torch.ones(weights.shape[0], 1, device=weights.device, dtype=weights.dtype)
        focus = weights.pow(2).sum(dim=1, keepdim=True)
        min_focus = 1.0 / float(weights.shape[1])
        denom = max(1.0 - min_focus, 1e-6)
        return ((focus - min_focus) / denom).clamp(0.0, 1.0)

    @staticmethod
    def _agreement(local: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        local_norm = F.normalize(local, dim=-1)
        global_norm = F.normalize(global_, dim=-1)
        cosine = (local_norm * global_norm).sum(dim=-1, keepdim=True)
        return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)

    def forward(
        self,
        g_t_local: torch.Tensor,
        g_f_local: torch.Tensor,
        g_t_global: torch.Tensor,
        g_f_global: torch.Tensor,
        w_t_local: torch.Tensor,
        w_f_local: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if g_t_local.shape != g_t_global.shape or g_f_local.shape != g_f_global.shape:
            raise ValueError(
                "Local/global evidence shapes must match, got "
                f"{tuple(g_t_local.shape)} vs {tuple(g_t_global.shape)} and "
                f"{tuple(g_f_local.shape)} vs {tuple(g_f_global.shape)}."
            )

        time_focus = self._focus(w_t_local)
        freq_focus = self._focus(w_f_local)
        time_agreement = self._agreement(g_t_local, g_t_global)
        freq_agreement = self._agreement(g_f_local, g_f_global)

        time_trust_local = torch.sqrt((time_focus * time_agreement).clamp(0.0, 1.0))
        freq_trust_local = torch.sqrt((freq_focus * freq_agreement).clamp(0.0, 1.0))

        g_t_summary = time_trust_local * g_t_local + (1.0 - time_trust_local) * g_t_global
        g_f_summary = freq_trust_local * g_f_local + (1.0 - freq_trust_local) * g_f_global

        g_t_summary = self._normalize(g_t_summary)
        g_f_summary = self._normalize(g_f_summary)
        interaction = self._normalize(g_t_summary * g_f_summary)
        time_gap = (g_t_local - g_t_global).abs()
        freq_gap = (g_f_local - g_f_global).abs()
        local_global_gap = self._normalize(0.5 * (time_gap + freq_gap))

        h = torch.cat([g_t_summary, g_f_summary, interaction, local_global_gap], dim=-1)
        extras = {
            "time_focus": time_focus,
            "freq_focus": freq_focus,
            "time_agreement": time_agreement,
            "freq_agreement": freq_agreement,
            "time_trust_local": time_trust_local,
            "freq_trust_local": freq_trust_local,
            "g_t_summary": g_t_summary,
            "g_f_summary": g_f_summary,
            "local_global_gap": local_global_gap,
            "time_gap": time_gap,
            "freq_gap": freq_gap,
        }
        return h, extras
