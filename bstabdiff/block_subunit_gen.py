# block_subunit_gen.py
# ============================================================
# Block--Subunit HDLSS Tabular Generator (NSC-free)
# - Block latents h in R^M
# - Continuous: Gaussian copula latent -> inverse empirical CDF marginal
# - Categorical: logits from h_t
# - Missingness: Bernoulli rates (optionally class-conditional)
# - Prior: diffusion OR RealNVP flow on h
# - Train-time best-checkpoint tracking (lowest loss)
# - Optional EMA for diffusion prior (recommended)
# - Optional checkpoint saving to disk (save_dir/save_name)
# - Optional return_train_info from fit helper
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def erf_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    # Phi(x) = 0.5 * (1 + erf(x/sqrt(2)))
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def approx_normal_icdf(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Approximate inverse CDF for standard normal using erfinv:
      Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    """
    u = torch.clamp(u, eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def make_equal_blocks(m: int, M: int) -> List[np.ndarray]:
    """
    Partition features 0..m-1 into M contiguous equal-ish blocks.
    """
    idx = np.arange(m)
    blocks = np.array_split(idx, M)
    return [b.astype(int) for b in blocks]


def apply_permutation(X: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return X[:, perm]


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# --------------------------
# EMA helper (for diffusion prior)
# --------------------------

class EMA:
    """
    Exponential moving average of parameters (for more stable sampling).
    Works on state_dict tensors.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0,1).")
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow


# --------------------------
# Feature schema
# --------------------------

@dataclass
class FeatureSpec:
    name: str
    kind: str  # "continuous" or "categorical"
    n_categories: int = 0  # for categorical only


def _validate_feature_specs(feature_specs: List[FeatureSpec], m: int) -> None:
    if len(feature_specs) != m:
        raise ValueError(f"feature_specs length ({len(feature_specs)}) must equal m ({m}).")
    for fs in feature_specs:
        if fs.kind not in ("continuous", "categorical"):
            raise ValueError(f"Unknown feature kind: {fs.kind}")
        if fs.kind == "categorical" and fs.n_categories <= 1:
            raise ValueError(f"Categorical feature {fs.name} must have n_categories >= 2.")


# --------------------------
# Empirical inverse CDF for marginals
# --------------------------

class EmpiricalMarginals:
    """
    Stores per-feature (and optionally per-class) sorted values for inverse CDF sampling.
    For continuous features only.

    Supports:
      - fit from observed X with mask
      - inverse_cdf(u, feature_index, class_index)
    """

    def __init__(self, m: int, n_classes: Optional[int], device: torch.device):
        self.m = m
        self.n_classes = n_classes  # None => unconditional
        self.device = device

        # For each feature j:
        #   unconditional: values[(j, None)] = 1D tensor sorted
        #   class-cond:   values[(j, y)] = 1D tensor sorted
        self.values: Dict[Tuple[int, Optional[int]], torch.Tensor] = {}

    @staticmethod
    def _to_sorted_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
        x = x.astype(np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            # fallback to standard normal samples if feature fully missing
            x = np.random.normal(size=256).astype(np.float32)
        x = np.sort(x)
        return torch.from_numpy(x).to(device)

    def fit(
        self,
        X: np.ndarray,
        R: np.ndarray,
        feature_specs: List[FeatureSpec],
        y: Optional[np.ndarray] = None,
    ) -> None:
        """
        X: (n, m) with np.nan for missing continuous
        R: (n, m) {0,1}
        y: (n,) optional class labels
        """
        n, m = X.shape
        assert m == self.m

        if self.n_classes is None:
            for j in range(m):
                if feature_specs[j].kind != "continuous":
                    continue
                xj = X[:, j]
                rj = R[:, j].astype(bool)
                vals = xj[rj]
                self.values[(j, None)] = self._to_sorted_tensor(vals, self.device)
        else:
            if y is None:
                raise ValueError("Class-conditional marginals requested but y is None.")
            for j in range(m):
                if feature_specs[j].kind != "continuous":
                    continue
                xj = X[:, j]
                rj = R[:, j].astype(bool)
                for c in range(self.n_classes):
                    mask = (y == c) & rj
                    vals = xj[mask]
                    self.values[(j, c)] = self._to_sorted_tensor(vals, self.device)

    def inverse_cdf(self, u: torch.Tensor, j: int, y: Optional[int] = None) -> torch.Tensor:
        """
        u: tensor in (0,1) arbitrary shape
        returns x with same shape, sampled from empirical inverse CDF
        using linear interpolation between sorted values.
        """
        key = (j, y if self.n_classes is not None else None)
        if key not in self.values:
            # fallback: standard normal mapped
            return approx_normal_icdf(u)

        v = self.values[key]  # (K,)
        K = v.numel()
        if K < 2:
            return v[0].expand_as(u)

        # Map u -> index in [0, K-1]
        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        pos = u * (K - 1)
        idx0 = torch.floor(pos).long()
        idx1 = torch.clamp(idx0 + 1, max=K - 1)
        w = (pos - idx0.float())

        # Gather via indexing (broadcast-safe)
        x0 = v[idx0]
        x1 = v[idx1]
        return (1.0 - w) * x0 + w * x1


# --------------------------
# Simple h inference (plug-in estimator)
# --------------------------

def infer_block_latents_mean_gaussianized(
    X: np.ndarray,
    R: np.ndarray,
    blocks: List[np.ndarray],
    feature_specs: List[FeatureSpec],
    y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    A lightweight plug-in estimator for h:
      - for each continuous feature, map observed values to rank-based u then to z=Phi^{-1}(u)
      - per block t, h_t is the mean of z over features in that block (ignoring missing)

    Returns:
      h_hat: (n, M)
    """
    n, m = X.shape
    M = len(blocks)

    # Precompute per-feature empirical CDF ranks (unconditional) for gaussianization
    Z = np.zeros((n, m), dtype=np.float32)
    Z[:] = 0.0

    for j in range(m):
        if feature_specs[j].kind != "continuous":
            continue
        obs = (R[:, j] == 1) & np.isfinite(X[:, j])
        vals = X[obs, j].astype(np.float64)
        if vals.size < 5:
            continue
        order = np.argsort(vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(vals.size)
        u = (ranks + 0.5) / vals.size
        u_t = torch.from_numpy(u.astype(np.float32))
        z_t = approx_normal_icdf(u_t).numpy()
        Z[obs, j] = z_t.astype(np.float32)

    # block means
    h_hat = np.zeros((n, M), dtype=np.float32)
    for t, idx in enumerate(blocks):
        idx = idx.astype(int)
        cont_mask = np.array([feature_specs[j].kind == "continuous" for j in idx], dtype=bool)
        idx_cont = idx[cont_mask]
        if idx_cont.size == 0:
            continue
        block_obs = (R[:, idx_cont] == 1)
        denom = np.maximum(block_obs.sum(axis=1), 1)
        h_hat[:, t] = (Z[:, idx_cont] * block_obs).sum(axis=1) / denom

    # standardize across samples
    h_hat = (h_hat - h_hat.mean(axis=0, keepdims=True)) / (h_hat.std(axis=0, keepdims=True) + 1e-6)
    return h_hat


# --------------------------
# Emission model (decoder): continuous + categorical + missingness
# --------------------------

@dataclass
class EmissionParams:
    # Continuous: z_j = a_j * h_t + b_j(y) + eps, eps~N(0, sigma_j^2)
    a: torch.Tensor                 # (m,)
    sigma: torch.Tensor             # (m,) positive
    b: Optional[torch.Tensor]       # (n_classes, m) or None

    # Categorical: logits_j = W_j * h_t + c_j(y)
    cat_W: Dict[int, torch.Tensor]  # j -> (K_j,)
    cat_c: Dict[int, torch.Tensor]  # j -> (n_classes, K_j) or (K_j,)

    # Missingness: Bernoulli rates (store missing probability)
    miss_rate: torch.Tensor         # (m,) unconditional P(missing)
    miss_rate_y: Optional[torch.Tensor]  # (n_classes, m) or None


def fit_emissions_from_inferred_h(
    X: np.ndarray,
    R: np.ndarray,
    y: Optional[np.ndarray],
    blocks: List[np.ndarray],
    feature_specs: List[FeatureSpec],
    h_hat: np.ndarray,
    n_classes: Optional[int],
    device: torch.device,
) -> EmissionParams:
    """
    Fit simple emission parameters from inferred h_hat.
    - Continuous: regress gaussianized z on scalar h_t (+ optional class bias)
    - Categorical: fit per-feature logits as linear in h_t + optional class bias
    - Missingness: empirical missing rates (optional class-conditional)

    Note: this is a practical initialization / baseline fitting.
    """
    n, m = X.shape
    M = len(blocks)

    # Map each feature to its block id
    feat_to_block = np.zeros(m, dtype=int)
    for t, idx in enumerate(blocks):
        feat_to_block[idx.astype(int)] = t

    # Missing rates: P(missing)
    miss_rate = 1.0 - R.mean(axis=0).astype(np.float32)
    miss_rate_t = torch.from_numpy(np.clip(miss_rate, 1e-4, 1 - 1e-4)).to(device)

    miss_rate_y_t = None
    if n_classes is not None and y is not None:
        mr_y = np.zeros((n_classes, m), dtype=np.float32)
        for c in range(n_classes):
            mask = (y == c)
            if mask.sum() < 2:
                mr_y[c] = miss_rate
            else:
                mr_y[c] = 1.0 - R[mask].mean(axis=0)
        miss_rate_y_t = torch.from_numpy(np.clip(mr_y, 1e-4, 1 - 1e-4)).to(device)

    # Gaussianize continuous targets z by rank -> normal (unconditional)
    Z = np.zeros((n, m), dtype=np.float32)
    Z[:] = 0.0
    for j in range(m):
        if feature_specs[j].kind != "continuous":
            continue
        obs = (R[:, j] == 1) & np.isfinite(X[:, j])
        vals = X[obs, j].astype(np.float64)
        if vals.size < 8:
            continue
        order = np.argsort(vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(vals.size)
        u = (ranks + 0.5) / vals.size
        u_t = torch.from_numpy(u.astype(np.float32))
        z_t = approx_normal_icdf(u_t).numpy()
        Z[obs, j] = z_t.astype(np.float32)

    # Fit a_j, sigma_j, and b_j(y)
    a = np.zeros(m, dtype=np.float32)
    sigma = np.ones(m, dtype=np.float32)
    b = None
    if n_classes is not None:
        b = np.zeros((n_classes, m), dtype=np.float32)

    for j in range(m):
        if feature_specs[j].kind != "continuous":
            continue
        t = feat_to_block[j]
        hj = h_hat[:, t]  # (n,)
        obs = (R[:, j] == 1) & np.isfinite(X[:, j])

        if obs.sum() < 10:
            a[j] = 0.0
            sigma[j] = 1.0
            if b is not None:
                b[:, j] = 0.0
            continue

        z = Z[obs, j].astype(np.float64)
        h = hj[obs].astype(np.float64)

        if (n_classes is not None) and (y is not None):
            yy = y[obs]
            # z ~ a*h + b_y
            H = h.reshape(-1, 1)
            OH = np.zeros((len(h), n_classes), dtype=np.float64)
            OH[np.arange(len(h)), yy] = 1.0
            A = np.concatenate([H, OH], axis=1)
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            a[j] = float(coef[0])
            b[:, j] = coef[1:].astype(np.float32)
            resid = z - A @ coef
        else:
            # z ~ a*h + b0
            H = np.stack([h, np.ones_like(h)], axis=1)
            coef, *_ = np.linalg.lstsq(H, z, rcond=None)
            a[j] = float(coef[0])
            resid = z - (H @ coef)

        sigma[j] = float(np.sqrt(np.maximum(np.mean(resid ** 2), 1e-4)))

    a_t = torch.from_numpy(a).to(device)
    sigma_t = torch.from_numpy(sigma).to(device)
    b_t = torch.from_numpy(b).to(device) if b is not None else None

    # Fit categorical parameters (simple torch training per feature)
    cat_W: Dict[int, torch.Tensor] = {}
    cat_c: Dict[int, torch.Tensor] = {}

    for j in range(m):
        fs = feature_specs[j]
        if fs.kind != "categorical":
            continue
        K = fs.n_categories
        t = feat_to_block[j]
        hj = torch.from_numpy(h_hat[:, t].astype(np.float32)).to(device)  # (n,)
        obs = (R[:, j] == 1) & np.isfinite(X[:, j])
        if obs.sum() < 10:
            cat_W[j] = torch.zeros(K, device=device)
            if n_classes is not None:
                cat_c[j] = torch.zeros((n_classes, K), device=device)
            else:
                cat_c[j] = torch.zeros(K, device=device)
            continue

        xj = torch.from_numpy(X[obs, j].astype(np.int64)).to(device)
        idx_obs = torch.from_numpy(np.where(obs)[0]).to(device)
        hj_obs = hj[idx_obs]

        if n_classes is not None and y is not None:
            yj = torch.from_numpy(y[obs].astype(np.int64)).to(device)
            W = torch.zeros(K, device=device, requires_grad=True)
            Cb = torch.zeros((n_classes, K), device=device, requires_grad=True)
            opt = torch.optim.Adam([W, Cb], lr=5e-2)
            for _ in range(200):
                logits = hj_obs[:, None] * W[None, :] + Cb[yj]
                loss = F.cross_entropy(logits, xj)
                opt.zero_grad()
                loss.backward()
                opt.step()
            cat_W[j] = W.detach()
            cat_c[j] = Cb.detach()
        else:
            W = torch.zeros(K, device=device, requires_grad=True)
            b0 = torch.zeros(K, device=device, requires_grad=True)
            opt = torch.optim.Adam([W, b0], lr=5e-2)
            for _ in range(200):
                logits = hj_obs[:, None] * W[None, :] + b0[None, :]
                loss = F.cross_entropy(logits, xj)
                opt.zero_grad()
                loss.backward()
                opt.step()
            cat_W[j] = W.detach()
            cat_c[j] = b0.detach()

    return EmissionParams(
        a=a_t,
        sigma=sigma_t,
        b=b_t,
        cat_W=cat_W,
        cat_c=cat_c,
        miss_rate=miss_rate_t,
        miss_rate_y=miss_rate_y_t,
    )


# --------------------------
# Prior option 1: Diffusion on h (latent DDPM)
# --------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DiffusionPrior(nn.Module):
    """
    Diffusion prior on h in R^M:
      - train denoiser eps_theta(h_t, t, y)
      - sample via reverse steps
    """
    def __init__(
        self,
        M: int,
        n_classes: Optional[int],
        T: int = 200,
        hidden: int = 256,
        y_embed_dim: int = 64,
    ):
        super().__init__()
        self.M = M
        self.n_classes = n_classes
        self.T = T

        self.time_emb = TimeEmbedding(128)
        self.y_embed = None
        y_dim = 0
        if n_classes is not None:
            self.y_embed = nn.Embedding(n_classes, y_embed_dim)
            y_dim = y_embed_dim

        inp = M + 128 + y_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, M),
        )

        # simple linear beta schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def eps_theta(self, h_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        te = self.time_emb(t)
        parts = [h_t, te]
        if self.n_classes is not None:
            assert y is not None
            parts.append(self.y_embed(y))
        x = torch.cat(parts, dim=1)
        return self.net(x)

    def q_sample(self, h0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].unsqueeze(1)  # (b,1)
        return torch.sqrt(ab) * h0 + torch.sqrt(1.0 - ab) * eps

    def training_loss(self, h0: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        b = h0.size(0)
        t = torch.randint(0, self.T, (b,), device=h0.device)
        eps = torch.randn_like(h0)
        h_t = self.q_sample(h0, t, eps)
        eps_hat = self.eps_theta(h_t, t, y)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: Optional[torch.Tensor],
        device: torch.device,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        _ = steps  # kept for future; currently uses full schedule
        h = torch.randn(n, self.M, device=device)
        for t_int in reversed(range(self.T)):
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            eps_hat = self.eps_theta(h, t, y)
            beta = self.betas[t_int]
            alpha = self.alphas[t_int]
            ab = self.alpha_bar[t_int]
            mean = (1.0 / torch.sqrt(alpha)) * (h - (beta / torch.sqrt(1.0 - ab)) * eps_hat)
            if t_int > 0:
                noise = torch.randn_like(h)
                h = mean + torch.sqrt(beta) * noise
            else:
                h = mean
        return h


# --------------------------
# Prior option 2: Simple conditional RealNVP flow on h
# --------------------------

class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)  # (dim,)
        inp = dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),  # scale, shift
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        h = torch.cat([x_masked, cond], dim=1)
        st = self.net(h)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = ((1 - self.mask) * s).sum(dim=1)
        return y, logdet

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        h = torch.cat([y_masked, cond], dim=1)
        st = self.net(h)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        logdet = -((1 - self.mask) * s).sum(dim=1)
        return x, logdet


class FlowPrior(nn.Module):
    def __init__(
        self,
        M: int,
        n_classes: Optional[int],
        n_layers: int = 6,
        hidden: int = 256,
        y_embed_dim: int = 64,
    ):
        super().__init__()
        self.M = M
        self.n_classes = n_classes
        self.y_embed = None
        cond_dim = 0
        if n_classes is not None:
            self.y_embed = nn.Embedding(n_classes, y_embed_dim)
            cond_dim = y_embed_dim

        masks = []
        for k in range(n_layers):
            mask = torch.zeros(M)
            mask[k % 2::2] = 1.0
            masks.append(mask)

        self.layers = nn.ModuleList([AffineCoupling(M, hidden, masks[k], cond_dim) for k in range(n_layers)])

    def _cond(self, y: Optional[torch.Tensor], device: torch.device, n: int) -> torch.Tensor:
        if self.n_classes is None:
            return torch.zeros(n, 0, device=device)
        assert y is not None
        return self.y_embed(y)

    def log_prob(self, h: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        device = h.device
        cond = self._cond(y, device, h.size(0))
        z = h
        logdet_sum = torch.zeros(h.size(0), device=device)
        for layer in self.layers:
            z, logdet = layer.forward(z, cond)
            logdet_sum += logdet
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.M * math.log(2 * math.pi)
        return log_pz + logdet_sum

    @torch.no_grad()
    def sample(self, n: int, y: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        cond = self._cond(y, device, n)
        z = torch.randn(n, self.M, device=device)
        x = z
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x, cond)
        return x


# --------------------------
# Main model: BlockSubunitGenerator
# --------------------------

class BlockSubunitGenerator:
    """
    End-to-end generator with:
      - feature specs
      - blocks + optional permutation
      - emission params (fitted from inferred h)
      - marginals (empirical inverse CDFs)
      - prior on h: diffusion OR flow

    Typical flow:
      1) Provide blocks (or build via make_equal_blocks)
      2) Fit marginals from X, R (and y if class-conditional)
      3) Infer h_hat from X, R
      4) Fit emission params from (X,R,y,h_hat)
      5) Train prior on h_hat (diffusion loss or flow NLL)
      6) Sample: y -> h -> (R, X) -> optional permutation
    """

    def __init__(
        self,
        feature_specs: List[FeatureSpec],
        blocks: List[np.ndarray],
        n_classes: Optional[int],
        prior_type: str = "diffusion",  # "diffusion" or "flow"
        device: Union[str, torch.device] = "cpu",
        use_class_cond_marginals: bool = True,
        use_class_cond_missingness: bool = True,
    ):
        self.device = torch.device(device)
        self.feature_specs = feature_specs
        self.m = len(feature_specs)
        _validate_feature_specs(feature_specs, self.m)

        self.blocks = blocks
        self.M = len(blocks)
        self.n_classes = n_classes

        self.use_class_cond_marginals = bool(use_class_cond_marginals and n_classes is not None)
        self.use_class_cond_missingness = bool(use_class_cond_missingness and n_classes is not None)

        self.marginals = EmpiricalMarginals(
            m=self.m,
            n_classes=self.n_classes if self.use_class_cond_marginals else None,
            device=self.device,
        )

        self.emission: Optional[EmissionParams] = None

        prior_type = prior_type.lower().strip()
        if prior_type not in ("diffusion", "flow"):
            raise ValueError("prior_type must be 'diffusion' or 'flow'")
        self.prior_type = prior_type

        if self.prior_type == "diffusion":
            self.prior = DiffusionPrior(M=self.M, n_classes=self.n_classes).to(self.device)
        else:
            self.prior = FlowPrior(M=self.M, n_classes=self.n_classes).to(self.device)

        self.perm: Optional[np.ndarray] = None
        self.inv_perm: Optional[np.ndarray] = None

    def set_permutation(self, perm: Optional[np.ndarray]) -> None:
        if perm is None:
            self.perm = None
            self.inv_perm = None
            return
        perm = np.asarray(perm).astype(int)
        if perm.shape != (self.m,):
            raise ValueError(f"perm must have shape ({self.m},)")
        self.perm = perm
        self.inv_perm = invert_permutation(perm)

    # --------------------------
    # Fit routines
    # --------------------------

    def fit_marginals(self, X: np.ndarray, R: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        self.marginals.fit(X=X, R=R, feature_specs=self.feature_specs, y=y)

    def infer_h(self, X: np.ndarray, R: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return infer_block_latents_mean_gaussianized(X=X, R=R, blocks=self.blocks, feature_specs=self.feature_specs, y=y)

    def fit_emissions(self, X: np.ndarray, R: np.ndarray, y: Optional[np.ndarray], h_hat: np.ndarray) -> None:
        self.emission = fit_emissions_from_inferred_h(
            X=X,
            R=R,
            y=y,
            blocks=self.blocks,
            feature_specs=self.feature_specs,
            h_hat=h_hat,
            n_classes=self.n_classes,
            device=self.device,
        )

    def train_prior(
        self,
        h_hat: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 2000,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        verbose_every: int = 200,
        # NEW:
        save_dir: Optional[str] = None,
        save_name: str = "blocksubunit",
        save_best: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        return_train_info: bool = False,
    ) -> Union[None, Dict[str, Union[int, float, str, bool, None]]]:
        """
        Train diffusion prior (MSE eps loss) OR flow prior (negative log-likelihood) on inferred h_hat.

        New behavior:
          - Tracks best (lowest) loss and loads it at end.
          - Optionally uses EMA for diffusion and saves/loads EMA weights as "best".
          - Optionally saves best checkpoint to disk.
        """
        h_t = torch.from_numpy(h_hat.astype(np.float32)).to(self.device)
        n = h_t.size(0)

        y_t = None
        if self.n_classes is not None:
            if y is None:
                raise ValueError("n_classes is not None but y is None.")
            y_t = torch.from_numpy(np.asarray(y).astype(np.int64)).to(self.device)

        opt = torch.optim.Adam(self.prior.parameters(), lr=lr, weight_decay=weight_decay)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        train_info: Dict[str, Union[int, float, str, bool, None]] = {
            "best_epoch": None,
            "best_loss": float("inf"),
            "best_ckpt_path": None,
            "loaded_at_end": False,
            "prior_type": self.prior_type,
            "used_ema": bool(use_ema and self.prior_type == "diffusion"),
            "ema_decay": float(ema_decay) if (use_ema and self.prior_type == "diffusion") else None,
        }

        ema = None
        if self.prior_type == "diffusion" and use_ema:
            ema = EMA(self.prior, decay=ema_decay)

        best_state: Optional[Dict[str, torch.Tensor]] = None

        for ep in range(1, epochs + 1):
            idx = torch.randint(0, n, (min(batch_size, n),), device=self.device)
            hb = h_t[idx]
            yb = y_t[idx] if y_t is not None else None

            if self.prior_type == "diffusion":
                loss = self.prior.training_loss(hb, yb)
            else:
                logp = self.prior.log_prob(hb, yb)
                loss = -logp.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.prior.parameters(), 1.0)
            opt.step()

            # EMA update after optimizer step
            if ema is not None:
                ema.update(self.prior)

            loss_val = float(loss.detach().cpu().item())

            if verbose_every and (ep % verbose_every == 0 or ep == 1 or ep == epochs):
                print(f"[prior:{self.prior_type}] epoch {ep}/{epochs} | loss={loss_val:.6f}")

            if save_best and loss_val < float(train_info["best_loss"]):
                train_info["best_loss"] = loss_val
                train_info["best_epoch"] = ep

                # store best weights (EMA for diffusion if enabled; otherwise raw weights)
                if ema is not None:
                    best_state = copy.deepcopy(ema.state_dict())
                else:
                    best_state = copy.deepcopy(self.prior.state_dict())

                if save_dir is not None:
                    ckpt_path = os.path.join(save_dir, f"{save_name}_best.pt")
                    torch.save(
                        {
                            "epoch": ep,
                            "best_loss": loss_val,
                            "prior_type": self.prior_type,
                            "used_ema": bool(ema is not None),
                            "ema_decay": float(ema_decay) if (ema is not None) else None,
                            "prior_state_dict": best_state,
                        },
                        ckpt_path,
                    )
                    train_info["best_ckpt_path"] = ckpt_path

        # load best at end
        if save_best and best_state is not None:
            self.prior.load_state_dict(best_state, strict=True)
            train_info["loaded_at_end"] = True

        if return_train_info:
            return train_info
        return None

    # --------------------------
    # Sampling routines
    # --------------------------

    def _sample_y(self, n: int, y: Optional[Union[int, np.ndarray]] = None) -> Optional[torch.Tensor]:
        if self.n_classes is None:
            return None
        if y is None:
            return torch.randint(0, self.n_classes, (n,), device=self.device)
        if isinstance(y, int):
            return torch.full((n,), int(y), device=self.device, dtype=torch.long)
        y = np.asarray(y).astype(int)
        if y.shape != (n,):
            raise ValueError("If y is an array, it must have shape (n,).")
        return torch.from_numpy(y.astype(np.int64)).to(self.device)

    @torch.no_grad()
    def sample_h(self, n: int, y: Optional[Union[int, np.ndarray]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y_t = self._sample_y(n, y)
        h = self.prior.sample(n=n, y=y_t, device=self.device) if self.prior_type == "diffusion" else self.prior.sample(n=n, y=y_t, device=self.device)
        return h, y_t

    @torch.no_grad()
    def sample(
        self,
        n: int,
        y: Optional[Union[int, np.ndarray]] = None,
        apply_perm: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate synthetic samples:
          returns X_out (n,m) float32 (categoricals stored as float codes),
          R_out (n,m) int64 {0,1},
          y_out (n,) int64 or None.
        """
        if self.emission is None:
            raise RuntimeError("Emission parameters not fitted. Call fit_emissions() first.")
        em = self.emission

        h, y_t = self.sample_h(n, y=y)

        # Missingness
        if self.n_classes is not None and self.use_class_cond_missingness and em.miss_rate_y is not None and y_t is not None:
            miss_p = em.miss_rate_y[y_t]  # (n,m)
        else:
            miss_p = em.miss_rate.unsqueeze(0).expand(n, self.m)

        U_m = torch.rand_like(miss_p)
        R = (U_m > miss_p).long()  # 1 observed, 0 missing

        # Initialize X (canonical)
        X = torch.empty(n, self.m, device=self.device, dtype=torch.float32)

        # feature -> block
        ftb_np = np.zeros(self.m, dtype=int)
        for t, idx in enumerate(self.blocks):
            ftb_np[idx.astype(int)] = t
        feat_to_block = torch.from_numpy(ftb_np).to(self.device)

        for j in range(self.m):
            t = int(feat_to_block[j].item())
            hj = h[:, t]  # (n,)
            miss = (R[:, j] == 0)

            if self.feature_specs[j].kind == "continuous":
                mu = em.a[j] * hj
                if self.n_classes is not None and em.b is not None and y_t is not None:
                    mu = mu + em.b[y_t, j]
                sig = torch.clamp(em.sigma[j], min=1e-4)
                z = mu + sig * torch.randn_like(mu)
                u01 = erf_normal_cdf(z)

                if self.use_class_cond_marginals and y_t is not None:
                    xj = torch.empty_like(u01)
                    for c in range(self.n_classes):
                        mask_c = (y_t == c)
                        if mask_c.any():
                            xj[mask_c] = self.marginals.inverse_cdf(u01[mask_c], j=j, y=int(c))
                else:
                    xj = self.marginals.inverse_cdf(u01, j=j, y=None)

                xj = xj.float()
                xj[miss] = float("nan")
                X[:, j] = xj

            else:
                fs = self.feature_specs[j]
                K = fs.n_categories
                W = em.cat_W.get(j, torch.zeros(K, device=self.device))
                default_c = torch.zeros((self.n_classes, K), device=self.device) if self.n_classes else torch.zeros(K, device=self.device)
                Cb = em.cat_c.get(j, default_c)

                if self.n_classes is not None and y_t is not None and Cb.dim() == 2:
                    logits = hj[:, None] * W[None, :] + Cb[y_t]
                else:
                    logits = hj[:, None] * W[None, :] + Cb[None, :]

                probs = F.softmax(logits, dim=1)
                cat = torch.multinomial(probs, num_samples=1).squeeze(1).float()
                cat[miss] = float("nan")
                X[:, j] = cat

        X_out = X.detach().cpu().numpy().astype(np.float32)
        R_out = R.detach().cpu().numpy().astype(np.int64)
        y_out = y_t.detach().cpu().numpy().astype(np.int64) if y_t is not None else None

        if apply_perm and (self.perm is not None):
            X_out = X_out[:, self.perm]
            R_out = R_out[:, self.perm]

        return X_out, R_out, y_out


# --------------------------
# Convenience: end-to-end fit helper
# --------------------------

def fit_block_subunit_generator(
    X: np.ndarray,
    feature_specs: List[FeatureSpec],
    y: Optional[np.ndarray] = None,
    M: int = 32,
    blocks: Optional[List[np.ndarray]] = None,
    permute_features: bool = False,
    prior_type: str = "diffusion",
    device: str = "cpu",
    seed: int = 0,
    prior_epochs: int = 1500,
    prior_batch: int = 128,
    prior_lr: float = 1e-3,
    verbose_every: int = 200,
    # NEW:
    save_dir: Optional[str] = None,
    save_name: str = "blocksubunit",
    save_best: bool = True,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    return_train_info: bool = False,
) -> Union[BlockSubunitGenerator, Tuple[BlockSubunitGenerator, Dict[str, Union[int, float, str, bool, None]]]]:
    """
    One-shot fitting pipeline:
      - create blocks
      - create mask R from nan
      - fit marginals
      - infer h_hat
      - fit emissions
      - train prior on h_hat (with best checkpoint + optional EMA)
    """
    set_seed(seed)
    X = np.asarray(X)
    n, m = X.shape
    _validate_feature_specs(feature_specs, m)

    # Missing mask: treat nan as missing for all kinds
    R = np.isfinite(X).astype(np.int64)

    if blocks is None:
        blocks = make_equal_blocks(m=m, M=M)

    n_classes = None
    if y is not None:
        y = np.asarray(y).astype(int)
        n_classes = int(y.max()) + 1

    gen = BlockSubunitGenerator(
        feature_specs=feature_specs,
        blocks=blocks,
        n_classes=n_classes,
        prior_type=prior_type,
        device=device,
        use_class_cond_marginals=True,
        use_class_cond_missingness=True,
    )

    if permute_features:
        perm = np.random.permutation(m)
        gen.set_permutation(perm)
    else:
        gen.set_permutation(None)

    # Fit marginals on canonical space (unpermuted data assumed)
    gen.fit_marginals(X=X, R=R, y=y)

    # Infer h
    h_hat = gen.infer_h(X=X, R=R, y=y)

    # Fit emissions
    gen.fit_emissions(X=X, R=R, y=y, h_hat=h_hat)

    # Train prior (+ best / EMA)
    train_info = gen.train_prior(
        h_hat=h_hat,
        y=y,
        epochs=prior_epochs,
        batch_size=prior_batch,
        lr=prior_lr,
        verbose_every=verbose_every,
        save_dir=save_dir,
        save_name=save_name,
        save_best=save_best,
        use_ema=use_ema,
        ema_decay=ema_decay,
        return_train_info=True,
    )

    if return_train_info:
        return gen, train_info  # type: ignore[return-value]
    return gen


# --------------------------
# Minimal usage example (remove or keep)
# --------------------------
if __name__ == "__main__":
    set_seed(0)
    n, m = 80, 200
    X = np.random.randn(n, m).astype(np.float32)
    mask = np.random.rand(n, m) > 0.1
    X[~mask] = np.nan

    feature_specs = [FeatureSpec(name=f"f{j}", kind="continuous") for j in range(m)]
    y = np.random.randint(0, 2, size=n)

    gen, info = fit_block_subunit_generator(
        X=X,
        feature_specs=feature_specs,
        y=y,
        M=20,
        prior_type="diffusion",
        device="cpu",
        prior_epochs=300,
        verbose_every=100,
        save_dir="checkpoints_demo",
        save_name="demo_M20_diff",
        save_best=True,
        use_ema=True,
        ema_decay=0.999,
        return_train_info=True,
    )

    print("Train info:", info)

    X_syn, R_syn, y_syn = gen.sample(n=50)
    print("Synthetic shapes:", X_syn.shape, R_syn.shape, y_syn.shape if y_syn is not None else None)