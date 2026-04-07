from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEvidencePooling(nn.Module):
    def __init__(
        self,
        score_norm: str = "layernorm",
        pool_mode: str = "logsumexp",
        pool_tau: float = 0.5,
        concept_norm: str = "layernorm",
        self_mix: float = 0.0,
        self_score_mode: str = "absmean",
        adaptive_self_mix: bool = False,
    ) -> None:
        super().__init__()
        self.score_norm = str(score_norm)
        self.pool_mode = str(pool_mode)
        self.pool_tau = float(pool_tau)
        self.concept_norm = str(concept_norm)
        self.self_mix = float(self_mix)
        self.self_score_mode = str(self_score_mode)
        self.adaptive_self_mix = bool(adaptive_self_mix)
        self.score_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _layer_norm(x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (int(x.shape[-1]),))

    def _normalize(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "layernorm":
            return self._layer_norm(x)
        if mode == "l2":
            return F.normalize(x, dim=-1)
        if mode == "none":
            return x
        raise ValueError(f"Unsupported normalization mode: {mode}")

    def _reduce_scores(self, s: torch.Tensor, dim: int) -> torch.Tensor:
        if self.pool_mode == "max":
            return s.max(dim=dim).values
        if self.pool_mode == "mean":
            return s.mean(dim=dim)
        if self.pool_mode == "logsumexp":
            tau = max(self.pool_tau, 1e-6)
            return tau * torch.logsumexp(s / tau, dim=dim)
        raise ValueError(f"Unsupported cross evidence pool mode: {self.pool_mode}")

    def _self_scores(self, z: torch.Tensor) -> torch.Tensor:
        if self.self_score_mode == "absmean":
            return z.abs().mean(dim=-1)
        if self.self_score_mode == "l2":
            return z.pow(2).sum(dim=-1).sqrt()
        if self.self_score_mode == "none":
            return z.new_zeros(z.shape[:-1])
        raise ValueError(f"Unsupported self evidence score mode: {self.self_score_mode}")

    @staticmethod
    def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
        if weights.shape[1] <= 1:
            return weights.new_zeros(weights.shape[0], 1)
        entropy = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(dim=1, keepdim=True)
        return entropy / math.log(float(weights.shape[1]))

    def forward(
        self, z_t: torch.Tensor, z_f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_t.ndim != 3 or z_f.ndim != 3:
            raise ValueError(
                f"Expected z_t [B, P, R] and z_f [B, F, R], got {tuple(z_t.shape)} and {tuple(z_f.shape)}."
            )
        if z_t.shape[0] != z_f.shape[0] or z_t.shape[-1] != z_f.shape[-1]:
            raise ValueError(
                f"Cross-evidence shape mismatch between {tuple(z_t.shape)} and {tuple(z_f.shape)}."
            )

        role_dim = int(z_t.shape[-1])
        z_t_score = self._normalize(z_t, self.score_norm)
        z_f_score = self._normalize(z_f, self.score_norm)
        scale = math.sqrt(float(max(role_dim, 1)))
        score_scale = F.softplus(self.score_scale) + 1e-4
        s = score_scale * torch.einsum("bpr,bfr->bpf", z_t_score, z_f_score) / scale
        a_t = self._reduce_scores(s, dim=2)
        a_f = self._reduce_scores(s, dim=1)
        w_t_cross = torch.softmax(a_t, dim=1)
        w_f_cross = torch.softmax(a_f, dim=1)
        self_mix = min(max(self.self_mix, 0.0), 1.0)
        if self_mix > 0.0:
            w_t_self = torch.softmax(self._self_scores(z_t), dim=1)
            w_f_self = torch.softmax(self._self_scores(z_f), dim=1)
            if self.adaptive_self_mix:
                mix_t = self_mix * self._normalized_entropy(w_t_cross).detach()
                mix_f = self_mix * self._normalized_entropy(w_f_cross).detach()
            else:
                mix_t = self_mix
                mix_f = self_mix
            w_t = (1.0 - mix_t) * w_t_cross + mix_t * w_t_self
            w_f = (1.0 - mix_f) * w_f_cross + mix_f * w_f_self
            w_t = w_t / w_t.sum(dim=1, keepdim=True).clamp_min(1e-6)
            w_f = w_f / w_f.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            w_t = w_t_cross
            w_f = w_f_cross
        g_t_attn = torch.einsum("bp,bpr->br", w_t, z_t)
        g_f_attn = torch.einsum("bf,bfr->br", w_f, z_f)
        g_t = g_t_attn
        g_f = g_f_attn
        g_t_h = self._normalize(g_t, self.concept_norm)
        g_f_h = self._normalize(g_f, self.concept_norm)
        h = torch.cat([g_t_h, g_f_h, g_t_h * g_f_h, (g_t_h - g_f_h).abs()], dim=-1)
        return h, s, w_t, w_f, g_t, g_f
