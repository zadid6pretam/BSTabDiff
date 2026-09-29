# bstabdiff_gobs.py
# ============================================================
# BSTabDiff: BSTabDiff + GO-BS ordering
# - GO-BS learns permutation + contiguous blocks first
# - Uses GPU KMeans for clustering samples when available
# - Builds cluster-wise feature graphs with multiple metrics
# - Integrates local feature orderings into a global ordering
# - Segments global ordering into M contiguous blocks
# - Trains BSTabDiff in canonical ordered space
# - Samples back in original observed feature order
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import os
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans as KMeansCPU
from kmeans_gpu import KMeans as KMeansGPU


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def erf_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def approx_normal_icdf(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = torch.clamp(u, eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def make_equal_blocks(m: int, M: int) -> List[np.ndarray]:
    idx = np.arange(m)
    blocks = np.array_split(idx, M)
    return [b.astype(int) for b in blocks]


def apply_permutation(X: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return X[:, perm]


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def contiguous_blocks_from_boundaries(m: int, boundaries: np.ndarray) -> List[np.ndarray]:
    boundaries = np.asarray(boundaries, dtype=int)
    full = np.concatenate(([0], boundaries, [m]))
    blocks = []
    for i in range(len(full) - 1):
        blocks.append(np.arange(full[i], full[i + 1], dtype=int))
    return blocks


def reorder_feature_specs(feature_specs: List["FeatureSpec"], perm: np.ndarray) -> List["FeatureSpec"]:
    return [feature_specs[int(j)] for j in perm]


# ============================================================
# EMA helper
# ============================================================

class EMA:
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


# ============================================================
# Feature schema
# ============================================================

@dataclass
class FeatureSpec:
    name: str
    kind: str  # "continuous" or "categorical"
    n_categories: int = 0


def _validate_feature_specs(feature_specs: List[FeatureSpec], m: int) -> None:
    if len(feature_specs) != m:
        raise ValueError(f"feature_specs length ({len(feature_specs)}) must equal m ({m}).")
    for fs in feature_specs:
        if fs.kind not in ("continuous", "categorical"):
            raise ValueError(f"Unknown feature kind: {fs.kind}")
        if fs.kind == "categorical" and fs.n_categories <= 1:
            raise ValueError(f"Categorical feature {fs.name} must have n_categories >= 2.")


# ============================================================
# GO-BS ordering
# ============================================================

@dataclass
class GOBSResult:
    perm_obs_to_can: np.ndarray
    perm_can_to_obs: np.ndarray
    boundaries: np.ndarray
    blocks: List[np.ndarray]
    local_orderings: List[List[int]]
    graphs: List[torch.Tensor]
    centroids: List[torch.Tensor]


class GOBSOrdering:
    """
    GO-BS ordering:
      1) Cluster samples using GPU KMeans when available
      2) Build cluster-wise feature graphs using chosen metric
      3) Get local feature orderings
      4) Integrate local ranks into a global ordering
      5) Segment the global ordering into M contiguous blocks
      6) Optional local boundary refinement

    Updated version:
      - Does NOT force feature-graph construction to CPU.
      - Keeps feature-feature graphs on self.device.
      - Does NOT retain dense graphs in GOBSResult during tuning.
    """

    def __init__(
        self,
        n_blocks: int,
        num_clusters: int = 7,
        metric: str = "kl_divergence",
        bins: int = 32,
        top_k: Optional[int] = None,
        refine_order: bool = True,
        direction_select: bool = True,
        refine_passes: int = 1,
        boundary_refine_passes: int = 5,
        boundary_window: int = 8,
        lambda_cross: float = 1.0,
        gamma_balance: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.n_blocks = int(n_blocks)
        self.num_clusters = int(num_clusters)
        self.metric = metric
        self.bins = int(bins)
        self.top_k = None if top_k is None else int(top_k)
        self.refine_order = bool(refine_order)
        self.direction_select = bool(direction_select)
        self.refine_passes = int(refine_passes)
        self.boundary_refine_passes = int(boundary_refine_passes)
        self.boundary_window = int(boundary_window)
        self.lambda_cross = float(lambda_cross)
        self.gamma_balance = float(gamma_balance)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self, seed: int = 42) -> None:
        set_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _empty_cuda_cache_if_needed(self) -> None:
        try:
            dev = torch.device(self.device)
            if dev.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _nanmean_fill(self, X: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(X)
        X0 = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        denom = mask.sum(dim=0).clamp_min(1)
        mean = X0.sum(dim=0) / denom

        Xf = X.clone()
        bad = ~torch.isfinite(Xf)
        if bad.any():
            Xf[bad] = mean.unsqueeze(0).expand_as(Xf)[bad]

        return Xf

    def _discretize(self, x: torch.Tensor, bins: int = 32) -> torch.Tensor:
        xmin, xmax = x.min(), x.max()
        if (xmax - xmin) < 1e-12:
            return torch.zeros_like(x, dtype=torch.long)

        edges = torch.linspace(
            xmin,
            xmax,
            bins + 1,
            device=x.device,
            dtype=x.dtype,
        )

        return (torch.bucketize(x, edges) - 1).clamp(0, bins - 1).long()

    # ---------- pairwise metrics on selected device, CPU or GPU ----------
    def _pairwise_euclidean(self, X: torch.Tensor) -> torch.Tensor:
        Z = X.t().contiguous()
        return torch.cdist(Z, Z, p=2)

    def _pairwise_manhattan(self, X: torch.Tensor) -> torch.Tensor:
        Z = X.t().contiguous()
        return torch.cdist(Z, Z, p=1)

    def _pairwise_cosine(self, X: torch.Tensor) -> torch.Tensor:
        Z = F.normalize(X.t().contiguous(), dim=1)
        sim = Z @ Z.t()
        D = 1.0 - sim.clamp(-1.0, 1.0)
        D.fill_diagonal_(0.0)
        return D

    def _pairwise_correlation(self, X: torch.Tensor) -> torch.Tensor:
        N, D = X.shape

        Xm = X - X.mean(dim=0, keepdim=True)
        std = Xm.std(dim=0, unbiased=False).clamp_min(1e-12)
        Z = Xm / std

        C = (Z.t() @ Z) / float(max(N, 1))
        Dcorr = 1.0 - C.abs()
        Dcorr.fill_diagonal_(0.0)

        return Dcorr

    @torch.no_grad()
    def _histograms_shared_bins(self, X: torch.Tensor) -> torch.Tensor:
        xmin = X.min()
        xmax = X.max()
        D = X.shape[1]

        if (xmax - xmin) < 1e-12:
            P = torch.zeros((D, self.bins), device=X.device, dtype=torch.float32)
            P[:, 0] = 1.0
            return P

        edges = torch.linspace(
            xmin,
            xmax,
            self.bins + 1,
            device=X.device,
            dtype=X.dtype,
        )

        idx = torch.bucketize(X, edges) - 1
        idx = idx.clamp(0, self.bins - 1)

        counts = torch.zeros((D, self.bins), device=X.device, dtype=torch.float32)
        ones = torch.ones_like(idx, dtype=torch.float32)

        counts.scatter_add_(1, idx.t(), ones.t())

        return counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def _kl_matrix(self, P: torch.Tensor) -> torch.Tensor:
        logP = torch.log(P.clamp_min(1e-12))
        Hself = (P * logP).sum(dim=1)
        XEnt = P @ logP.t()
        K = Hself[:, None] - XEnt
        K.fill_diagonal_(0.0)
        return K

    @torch.no_grad()
    def _construct_graph(self, X_cluster: torch.Tensor) -> torch.Tensor:
        """
        Construct feature-feature affinity graph on self.device.

        Previous version forced CPU here:
            X_cluster.detach().to("cpu")

        This version keeps the graph on GPU when self.device is CUDA.
        """
        dev = torch.device(self.device)

        X_dev = X_cluster.detach().to(device=dev, dtype=torch.float32)
        X_dev = self._nanmean_fill(X_dev)

        metric = self.metric.lower()

        if metric in ("euclidean", "l2"):
            D = self._pairwise_euclidean(X_dev)

        elif metric in ("l1", "manhattan", "cityblock"):
            D = self._pairwise_manhattan(X_dev)

        elif metric in ("cosine", "cos"):
            D = self._pairwise_cosine(X_dev)

        elif metric in ("correlation", "corr", "pearson"):
            D = self._pairwise_correlation(X_dev)

        elif metric in ("kl", "kl_divergence"):
            P = self._histograms_shared_bins(X_dev)
            D = self._kl_matrix(P)
            del P

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Convert distance/divergence to affinity.
        W = 1.0 / (1.0 + D.to(torch.float32))
        W.fill_diagonal_(0.0)

        del D

        if self.top_k is not None and self.top_k > 0:
            Df = W.shape[0]
            k = min(int(self.top_k), max(Df - 1, 1))

            vals, idx = torch.topk(W, k=k, dim=1)

            Ws = torch.zeros_like(W)
            Ws.scatter_(1, idx, vals)

            W = torch.maximum(Ws, Ws.t())
            W.fill_diagonal_(0.0)

            del vals, idx, Ws

        return W

    # ---------- ordering ----------
    @torch.no_grad()
    def _dispersion_cost(self, G: torch.Tensor, order: List[int]) -> torch.Tensor:
        D = len(order)
        device = G.device

        order_t = torch.tensor(order, device=device, dtype=torch.long)

        pos = torch.empty(D, device=device, dtype=torch.float32)
        pos[order_t] = torch.arange(D, device=device, dtype=torch.float32)

        total = torch.tensor(0.0, device=device, dtype=torch.float32)
        Gf = G.to(torch.float32)

        for i in range(D - 1):
            diff = (pos[i] - pos[i + 1:]).abs()
            total += (Gf[i, i + 1:] * diff).sum()

        return total

    @torch.no_grad()
    def _minimize_dispersion(self, G: torch.Tensor) -> List[int]:
        D = G.shape[0]

        row_sums = G.sum(1)
        start = int(torch.argmin(row_sums).item())

        visited = torch.zeros(D, dtype=torch.bool, device=G.device)
        order = [start]
        visited[start] = True

        for _ in range(D - 1):
            d = G[order[-1]].clone()
            d[visited] = float("inf")
            nxt = int(torch.argmin(d).item())
            order.append(nxt)
            visited[nxt] = True

        return order

    @torch.no_grad()
    def _adjacent_swap_delta(
        self,
        G: torch.Tensor,
        order_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        u = order_t[t]
        v = order_t[t + 1]

        if t > 0:
            left = order_t[:t]
            left_term = (
                G[u, left].to(torch.float32) -
                G[v, left].to(torch.float32)
            ).sum()
        else:
            left_term = torch.zeros((), device=G.device, dtype=torch.float32)

        if t + 2 < order_t.numel():
            right = order_t[t + 2:]
            right_term = (
                G[u, right].to(torch.float32) -
                G[v, right].to(torch.float32)
            ).sum()
        else:
            right_term = torch.zeros((), device=G.device, dtype=torch.float32)

        return left_term - right_term

    @torch.no_grad()
    def _refine_order(
        self,
        G: torch.Tensor,
        order: List[int],
        passes: int,
    ) -> List[int]:
        if (not self.refine_order) or passes <= 0:
            return order

        if self.direction_select:
            rev = list(reversed(order))
            if self._dispersion_cost(G, rev) < self._dispersion_cost(G, order):
                order = rev

        order_t = torch.tensor(order, device=G.device, dtype=torch.long)
        D = len(order)

        for _ in range(int(passes)):
            improved = False

            for t in range(D - 1):
                delta = self._adjacent_swap_delta(G, order_t, t)

                if delta < 0:
                    tmp = order_t[t].clone()
                    order_t[t] = order_t[t + 1]
                    order_t[t + 1] = tmp
                    improved = True

            if not improved:
                break

        return order_t.detach().cpu().tolist()

    def _calculate_inter_cluster_distances(
        self,
        centroids: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        C = torch.stack(centroids, dim=0).to(self.device)
        return torch.cdist(C, C, p=2), C

    def _integrate_orderings(
        self,
        local_orderings: List[List[int]],
        cluster_distances: torch.Tensor,
    ) -> List[int]:
        C = cluster_distances.shape[0]
        D = len(local_orderings[0])
        device = cluster_distances.device

        md = cluster_distances + torch.eye(C, device=device)
        w = 1.0 / md.mean(dim=1)
        w = w / (w.sum() + 1e-12)

        ranks = torch.empty((C, D), device=device, dtype=torch.float32)

        for c, order in enumerate(local_orderings):
            pos = torch.empty(D, device=device, dtype=torch.float32)
            pos[torch.as_tensor(order, device=device, dtype=torch.long)] = torch.arange(
                D,
                device=device,
                dtype=torch.float32,
            )
            ranks[c] = pos

        avg_rank = (w[:, None] * ranks).sum(0)

        return torch.argsort(avg_rank).detach().cpu().tolist()

    # ---------- segmentation ----------
    @torch.no_grad()
    def _segment_score(self, W_ord: torch.Tensor, boundaries: np.ndarray) -> float:
        m = W_ord.shape[0]
        blocks = contiguous_blocks_from_boundaries(m, boundaries)

        within = 0.0
        cross = 0.0
        sizes = []

        for blk in blocks:
            sizes.append(len(blk))

            if len(blk) >= 2:
                idx = torch.as_tensor(blk, dtype=torch.long, device=W_ord.device)
                sub = W_ord.index_select(0, idx).index_select(1, idx)
                within += float(torch.triu(sub, diagonal=1).sum().item())

        for i in range(len(blocks)):
            idx_i = torch.as_tensor(blocks[i], dtype=torch.long, device=W_ord.device)

            for j in range(i + 1, len(blocks)):
                idx_j = torch.as_tensor(blocks[j], dtype=torch.long, device=W_ord.device)
                sub = W_ord.index_select(0, idx_i).index_select(1, idx_j)
                cross += float(sub.sum().item())

        target = float(m) / len(blocks)
        reg = sum((s - target) ** 2 for s in sizes)

        return -within + self.lambda_cross * cross + self.gamma_balance * reg

    def _segment_by_leakage(
        self,
        W_ord: torch.Tensor,
        m: int,
        M: int,
    ) -> np.ndarray:
        base = np.linspace(0, m, M + 1, dtype=int)
        best = base[1:-1].copy()
        best_score = self._segment_score(W_ord, best)

        improved = True

        # boundary_refine_passes was previously unused.
        # This keeps your original logic but also respects the requested limit.
        passes = max(int(self.boundary_refine_passes), 1)

        for _ in range(passes):
            if not improved:
                break

            improved = False

            for bi in range(len(best)):
                cur = int(best[bi])

                lo = best[bi - 1] + 1 if bi > 0 else 1
                hi = best[bi + 1] - 1 if bi < len(best) - 1 else m - 1

                s_lo = max(lo, cur - self.boundary_window)
                s_hi = min(hi, cur + self.boundary_window)

                local_best = cur
                local_best_score = best_score

                for cand in range(s_lo, s_hi + 1):
                    trial = best.copy()
                    trial[bi] = cand

                    sc = self._segment_score(W_ord, trial)

                    if sc < local_best_score:
                        local_best = cand
                        local_best_score = sc

                if local_best != cur:
                    best[bi] = local_best
                    best_score = local_best_score
                    improved = True

        return best.astype(int)

    # ---------- main ----------
    @torch.no_grad()
    def fit(
        self,
        X_train: np.ndarray,
        y: Optional[np.ndarray] = None,
        seed: int = 42,
        deterministic: bool = True,
        use_cpu_kmeans: bool = False,
    ) -> GOBSResult:
        if deterministic:
            self._set_seed(seed)

        dev = torch.device(self.device)

        X = torch.as_tensor(
            np.asarray(X_train),
            dtype=torch.float32,
            device=dev,
        )

        X = self._nanmean_fill(X)
        N, D = X.shape

        # Step 1: cluster samples
        if use_cpu_kmeans or (dev.type == "cpu"):
            labels = KMeansCPU(
                n_clusters=self.num_clusters,
                random_state=seed,
                n_init=10,
            ).fit_predict(X.detach().cpu().numpy())

            cluster_labels = torch.tensor(labels, device=dev, dtype=torch.long)

            centroids = torch.stack(
                [
                    X[cluster_labels == i].mean(dim=0)
                    for i in range(self.num_clusters)
                ],
                dim=0,
            )

        else:
            points = X.unsqueeze(0).contiguous()
            features = X.t().unsqueeze(0).contiguous()

            kmeans = KMeansGPU(
                n_clusters=self.num_clusters,
                max_iter=100,
                tolerance=1e-4,
                distance="euclidean",
                sub_sampling=None,
                max_neighbors=15,
            )

            centroids_b, _ = kmeans(points, features)
            centroids = centroids_b[0].to(dev)

            dists = torch.cdist(X, centroids, p=2)
            cluster_labels = torch.argmin(dists, dim=1)

            del points, features, centroids_b, dists

        # Step 2: cluster-wise graphs and local feature orderings
        # Important:
        #   We intentionally do not retain dense graphs in `graphs`.
        #   Keeping them caused CPU/GPU memory accumulation during Optuna.
        graphs: List[torch.Tensor] = []
        local_orderings: List[List[int]] = []
        centroids_list: List[torch.Tensor] = []

        unique_labels = torch.unique(cluster_labels)

        for i in unique_labels.detach().cpu().tolist():
            cluster_data = X[cluster_labels == int(i)]

            if cluster_data.numel() == 0:
                continue

            centroids_list.append(cluster_data.mean(dim=0).detach())

            G = self._construct_graph(cluster_data)  # GPU feature affinity graph

            init_order = self._minimize_dispersion(G)
            ord_i = self._refine_order(
                G,
                init_order,
                passes=self.refine_passes,
            )

            local_orderings.append(ord_i)

            # Do not store dense graph.
            del G, cluster_data
            self._empty_cuda_cache_if_needed()

        if len(local_orderings) == 0:
            perm_obs_to_can = np.arange(D, dtype=int)

        elif len(local_orderings) == 1:
            perm_obs_to_can = np.asarray(local_orderings[0], dtype=int)

        else:
            cluster_distances, _ = self._calculate_inter_cluster_distances(centroids_list)

            perm_obs_to_can = np.asarray(
                self._integrate_orderings(
                    local_orderings,
                    cluster_distances,
                ),
                dtype=int,
            )

            del cluster_distances
            self._empty_cuda_cache_if_needed()

        # Step 3: contiguous segmentation in canonical order
        W_full = self._construct_graph(X)  # GPU full-data feature affinity

        perm_t = torch.as_tensor(
            perm_obs_to_can,
            dtype=torch.long,
            device=W_full.device,
        )

        W_ord = W_full.index_select(0, perm_t).index_select(1, perm_t)

        boundaries = self._segment_by_leakage(
            W_ord,
            D,
            self.n_blocks,
        )

        blocks = contiguous_blocks_from_boundaries(D, boundaries)

        perm_can_to_obs = invert_permutation(perm_obs_to_can)

        # Free dense full graph before returning.
        del W_full, W_ord, perm_t
        self._empty_cuda_cache_if_needed()

        return GOBSResult(
            perm_obs_to_can=perm_obs_to_can,
            perm_can_to_obs=perm_can_to_obs,
            boundaries=boundaries,
            blocks=blocks,
            local_orderings=local_orderings,
            graphs=[],  # intentionally empty to avoid retaining dense graphs
            centroids=centroids_list,
        )


# ============================================================
# GO-BS-FC ordering: feature-clustered GO-BS
# ============================================================

class GOBSFCOrdering(GOBSOrdering):
    """
    GO-BS-FC ordering:
      1) Cluster features instead of samples using X^T in R^{m x n}
      2) Build one feature-feature graph inside each feature cluster
      3) Order features locally inside each feature cluster
      4) Order feature clusters by centroid similarity
      5) Concatenate local orders into a global feature ordering
      6) Segment the global ordering into M contiguous blocks

    This is the feature-clustered counterpart to sample-clustered GO-BS.
    """

    @torch.no_grad()
    def _cluster_features(
        self,
        X: torch.Tensor,
        seed: int = 42,
        use_cpu_kmeans: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster features using the transposed view X^T.

        X: shape (n_samples, n_features)
        returns:
            feature_cluster_labels: shape (n_features,)
            centroids: shape (num_clusters_eff, n_samples)
        """
        dev = torch.device(self.device)
        X = X.to(dev, dtype=torch.float32)
        X = self._nanmean_fill(X)

        N, D = X.shape
        K = min(int(self.num_clusters), D)

        # Feature vectors are rows: shape (D, N)
        XF = X.t().contiguous()

        if K <= 1:
            labels = torch.zeros(D, device=dev, dtype=torch.long)
            centroids = XF.mean(dim=0, keepdim=True)
            return labels, centroids

        if use_cpu_kmeans or (dev.type == "cpu"):
            labels_np = KMeansCPU(
                n_clusters=K,
                random_state=seed,
                n_init=10,
            ).fit_predict(XF.detach().cpu().numpy())

            labels = torch.tensor(labels_np, device=dev, dtype=torch.long)

            centroids = torch.stack(
                [
                    XF[labels == k].mean(dim=0)
                    for k in range(K)
                ],
                dim=0,
            )

            return labels, centroids

        # GPU feature clustering.
        # kmeans_gpu expects:
        #   points:   (B, num_points, point_dim)
        #   features: (B, point_dim, num_points)
        points = XF.unsqueeze(0).contiguous()       # (1, D, N)
        features = XF.t().unsqueeze(0).contiguous() # (1, N, D)

        kmeans = KMeansGPU(
            n_clusters=K,
            max_iter=100,
            tolerance=1e-4,
            distance="euclidean",
            sub_sampling=None,
            max_neighbors=15,
        )

        centroids_b, _ = kmeans(points, features)
        centroids = centroids_b[0].to(dev)

        dists = torch.cdist(XF, centroids, p=2)
        labels = torch.argmin(dists, dim=1)

        del points, features, centroids_b, dists
        self._empty_cuda_cache_if_needed()

        return labels, centroids

    @torch.no_grad()
    def _order_feature_clusters(
        self,
        centroids: torch.Tensor,
    ) -> List[int]:
        """
        Greedy nearest-neighbor ordering of feature clusters.
        """
        K = centroids.shape[0]

        if K <= 1:
            return list(range(K))

        Dmat = torch.cdist(centroids, centroids, p=2)
        row_sums = Dmat.sum(dim=1)
        start = int(torch.argmin(row_sums).item())

        visited = torch.zeros(K, dtype=torch.bool, device=centroids.device)
        order = [start]
        visited[start] = True

        for _ in range(K - 1):
            d = Dmat[order[-1]].clone()
            d[visited] = float("inf")
            nxt = int(torch.argmin(d).item())
            order.append(nxt)
            visited[nxt] = True

        return order

    @torch.no_grad()
    def fit(
        self,
        X_train: np.ndarray,
        y: Optional[np.ndarray] = None,
        seed: int = 42,
        deterministic: bool = True,
        use_cpu_kmeans: bool = False,
    ) -> GOBSResult:
        if deterministic:
            self._set_seed(seed)

        dev = torch.device(self.device)

        X = torch.as_tensor(
            np.asarray(X_train),
            dtype=torch.float32,
            device=dev,
        )

        X = self._nanmean_fill(X)
        N, D = X.shape

        # --------------------------------------------------------
        # Step 1: feature clustering on X^T
        # --------------------------------------------------------
        feature_cluster_labels, feature_centroids = self._cluster_features(
            X,
            seed=seed,
            use_cpu_kmeans=use_cpu_kmeans,
        )

        unique_clusters = torch.unique(feature_cluster_labels).detach().cpu().tolist()

        local_orderings_global: List[List[int]] = []
        centroids_list: List[torch.Tensor] = []

        # --------------------------------------------------------
        # Step 2: local ordering within each feature cluster
        # --------------------------------------------------------
        for c in unique_clusters:
            feat_idx_t = torch.where(feature_cluster_labels == int(c))[0]
            feat_idx = feat_idx_t.detach().cpu().numpy().astype(int)

            if feat_idx.size == 0:
                continue

            # Centroid in sample-profile space
            centroids_list.append(X[:, feat_idx].t().mean(dim=0).detach())

            if feat_idx.size == 1:
                local_orderings_global.append([int(feat_idx[0])])
                continue

            X_sub = X[:, feat_idx]

            G_sub = self._construct_graph(X_sub)

            init_local = self._minimize_dispersion(G_sub)
            refined_local = self._refine_order(
                G_sub,
                init_local,
                passes=self.refine_passes,
            )

            # Map local feature indices back to global observed feature indices.
            refined_global = [int(feat_idx[j]) for j in refined_local]
            local_orderings_global.append(refined_global)

            del X_sub, G_sub, feat_idx_t
            self._empty_cuda_cache_if_needed()

        if len(local_orderings_global) == 0:
            perm_obs_to_can = np.arange(D, dtype=int)

        elif len(local_orderings_global) == 1:
            perm_obs_to_can = np.asarray(local_orderings_global[0], dtype=int)

        else:
            # --------------------------------------------------------
            # Step 3: order feature clusters by centroid distances
            # --------------------------------------------------------
            C = torch.stack(centroids_list, dim=0).to(dev)
            cluster_order = self._order_feature_clusters(C)

            global_order = []
            for ci in cluster_order:
                global_order.extend(local_orderings_global[int(ci)])

            perm_obs_to_can = np.asarray(global_order, dtype=int)

            del C
            self._empty_cuda_cache_if_needed()

        # Safety check: permutation must include every feature exactly once.
        if len(perm_obs_to_can) != D or len(np.unique(perm_obs_to_can)) != D:
            raise RuntimeError(
                "GO-BS-FC produced an invalid feature permutation. "
                f"Expected {D} unique features, got {len(perm_obs_to_can)} entries "
                f"with {len(np.unique(perm_obs_to_can))} unique values."
            )

        # --------------------------------------------------------
        # Step 4: segment global canonical order into contiguous blocks
        # --------------------------------------------------------
        W_full = self._construct_graph(X)

        perm_t = torch.as_tensor(
            perm_obs_to_can,
            dtype=torch.long,
            device=W_full.device,
        )

        W_ord = W_full.index_select(0, perm_t).index_select(1, perm_t)

        boundaries = self._segment_by_leakage(
            W_ord,
            D,
            self.n_blocks,
        )

        blocks = contiguous_blocks_from_boundaries(D, boundaries)
        perm_can_to_obs = invert_permutation(perm_obs_to_can)

        del W_full, W_ord, perm_t
        self._empty_cuda_cache_if_needed()

        return GOBSResult(
            perm_obs_to_can=perm_obs_to_can,
            perm_can_to_obs=perm_can_to_obs,
            boundaries=boundaries,
            blocks=blocks,
            local_orderings=local_orderings_global,
            graphs=[],
            centroids=centroids_list,
        )

# ============================================================
# Empirical inverse CDF marginals
# ============================================================

class EmpiricalMarginals:
    def __init__(self, m: int, n_classes: Optional[int], device: torch.device):
        self.m = m
        self.n_classes = n_classes
        self.device = device
        self.values: Dict[Tuple[int, Optional[int]], torch.Tensor] = {}

    @staticmethod
    def _to_sorted_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
        x = x.astype(np.float32)
        x = x[np.isfinite(x)]
        if x.size == 0:
            x = np.random.normal(size=256).astype(np.float32)
        x = np.sort(x)
        return torch.from_numpy(x).to(device)

    def fit(self, X: np.ndarray, R: np.ndarray, feature_specs: List[FeatureSpec],
            y: Optional[np.ndarray] = None) -> None:
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
        key = (j, y if self.n_classes is not None else None)
        if key not in self.values:
            return approx_normal_icdf(u)

        v = self.values[key]
        K = v.numel()
        if K < 2:
            return v[0].expand_as(u)

        u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
        pos = u * (K - 1)
        idx0 = torch.floor(pos).long()
        idx1 = torch.clamp(idx0 + 1, max=K - 1)
        w = pos - idx0.float()
        x0 = v[idx0]
        x1 = v[idx1]
        return (1.0 - w) * x0 + w * x1


# ============================================================
# Inference of h
# ============================================================

def infer_block_latents_mean_gaussianized(
    X: np.ndarray,
    R: np.ndarray,
    blocks: List[np.ndarray],
    feature_specs: List[FeatureSpec],
    y: Optional[np.ndarray] = None,
) -> np.ndarray:
    n, m = X.shape
    M = len(blocks)

    Z = np.zeros((n, m), dtype=np.float32)

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
        z_t = approx_normal_icdf(torch.from_numpy(u.astype(np.float32))).numpy()
        Z[obs, j] = z_t.astype(np.float32)

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

    h_hat = (h_hat - h_hat.mean(axis=0, keepdims=True)) / (h_hat.std(axis=0, keepdims=True) + 1e-6)
    return h_hat


# ============================================================
# Emission model
# ============================================================

@dataclass
class EmissionParams:
    a: torch.Tensor
    sigma: torch.Tensor
    b: Optional[torch.Tensor]
    cat_W: Dict[int, torch.Tensor]
    cat_c: Dict[int, torch.Tensor]
    miss_rate: torch.Tensor
    miss_rate_y: Optional[torch.Tensor]


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
    n, m = X.shape

    feat_to_block = np.zeros(m, dtype=int)
    for t, idx in enumerate(blocks):
        feat_to_block[idx.astype(int)] = t

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

    Z = np.zeros((n, m), dtype=np.float32)
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
        z_t = approx_normal_icdf(torch.from_numpy(u.astype(np.float32))).numpy()
        Z[obs, j] = z_t.astype(np.float32)

    a = np.zeros(m, dtype=np.float32)
    sigma = np.ones(m, dtype=np.float32)
    b = None
    if n_classes is not None:
        b = np.zeros((n_classes, m), dtype=np.float32)

    for j in range(m):
        if feature_specs[j].kind != "continuous":
            continue
        t = feat_to_block[j]
        hj = h_hat[:, t]
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
            H = h.reshape(-1, 1)
            OH = np.zeros((len(h), n_classes), dtype=np.float64)
            OH[np.arange(len(h)), yy] = 1.0
            A = np.concatenate([H, OH], axis=1)
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            a[j] = float(coef[0])
            b[:, j] = coef[1:].astype(np.float32)
            resid = z - A @ coef
        else:
            H = np.stack([h, np.ones_like(h)], axis=1)
            coef, *_ = np.linalg.lstsq(H, z, rcond=None)
            a[j] = float(coef[0])
            resid = z - (H @ coef)

        sigma[j] = float(np.sqrt(np.maximum(np.mean(resid ** 2), 1e-4)))

    a_t = torch.from_numpy(a).to(device)
    sigma_t = torch.from_numpy(sigma).to(device)
    b_t = torch.from_numpy(b).to(device) if b is not None else None

    cat_W: Dict[int, torch.Tensor] = {}
    cat_c: Dict[int, torch.Tensor] = {}

    for j in range(m):
        fs = feature_specs[j]
        if fs.kind != "categorical":
            continue

        K = fs.n_categories
        t = feat_to_block[j]
        hj = torch.from_numpy(h_hat[:, t].astype(np.float32)).to(device)
        obs = (R[:, j] == 1) & np.isfinite(X[:, j])

        if obs.sum() < 10:
            cat_W[j] = torch.zeros(K, device=device)
            cat_c[j] = torch.zeros((n_classes, K), device=device) if n_classes is not None else torch.zeros(K, device=device)
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


# ============================================================
# Diffusion prior
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DiffusionPrior(nn.Module):
    def __init__(self, M: int, n_classes: Optional[int], T: int = 200, hidden: int = 256, y_embed_dim: int = 64):
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
        ab = self.alpha_bar[t].unsqueeze(1)
        return torch.sqrt(ab) * h0 + torch.sqrt(1.0 - ab) * eps

    def training_loss(self, h0: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        b = h0.size(0)
        t = torch.randint(0, self.T, (b,), device=h0.device)
        eps = torch.randn_like(h0)
        h_t = self.q_sample(h0, t, eps)
        eps_hat = self.eps_theta(h_t, t, y)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, n: int, y: Optional[torch.Tensor], device: torch.device, steps: Optional[int] = None) -> torch.Tensor:
        _ = steps
        h = torch.randn(n, self.M, device=device)
        for t_int in reversed(range(self.T)):
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            eps_hat = self.eps_theta(h, t, y)
            beta = self.betas[t_int]
            alpha = self.alphas[t_int]
            ab = self.alpha_bar[t_int]
            mean = (1.0 / torch.sqrt(alpha)) * (h - (beta / torch.sqrt(1.0 - ab)) * eps_hat)
            if t_int > 0:
                h = mean + torch.sqrt(beta) * torch.randn_like(h)
            else:
                h = mean
        return h


# ============================================================
# Flow prior
# ============================================================

class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)
        inp = dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
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
    def __init__(self, M: int, n_classes: Optional[int], n_layers: int = 6, hidden: int = 256, y_embed_dim: int = 64):
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
        cond = self._cond(y, h.device, h.size(0))
        z = h
        logdet_sum = torch.zeros(h.size(0), device=h.device)
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


# ============================================================
# Main BSTabDiff generator
# ============================================================

class BlockSubunitGenerator:
    def __init__(
        self,
        feature_specs: List[FeatureSpec],
        blocks: List[np.ndarray],
        n_classes: Optional[int],
        prior_type: str = "diffusion",
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

        # canonical -> observed permutation for returning samples
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
        save_dir: Optional[str] = None,
        save_name: str = "blocksubunit",
        save_best: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        return_train_info: bool = False,
    ) -> Union[None, Dict[str, Union[int, float, str, bool, None, np.ndarray]]]:
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

        train_info: Dict[str, Union[int, float, str, bool, None, np.ndarray]] = {
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

            if ema is not None:
                ema.update(self.prior)

            loss_val = float(loss.detach().cpu().item())

            if verbose_every and (ep % verbose_every == 0 or ep == 1 or ep == epochs):
                print(f"[prior:{self.prior_type}] epoch {ep}/{epochs} | loss={loss_val:.6f}")

            if save_best and loss_val < float(train_info["best_loss"]):
                train_info["best_loss"] = loss_val
                train_info["best_epoch"] = ep

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

        if save_best and best_state is not None:
            self.prior.load_state_dict(best_state, strict=True)
            train_info["loaded_at_end"] = True

        if return_train_info:
            return train_info
        return None

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
        if self.emission is None:
            raise RuntimeError("Emission parameters not fitted. Call fit_emissions() first.")
        em = self.emission

        h, y_t = self.sample_h(n, y=y)

        if self.n_classes is not None and self.use_class_cond_missingness and em.miss_rate_y is not None and y_t is not None:
            miss_p = em.miss_rate_y[y_t]
        else:
            miss_p = em.miss_rate.unsqueeze(0).expand(n, self.m)

        U_m = torch.rand_like(miss_p)
        R = (U_m > miss_p).long()

        X = torch.empty(n, self.m, device=self.device, dtype=torch.float32)

        ftb_np = np.zeros(self.m, dtype=int)
        for t, idx in enumerate(self.blocks):
            ftb_np[idx.astype(int)] = t
        feat_to_block = torch.from_numpy(ftb_np).to(self.device)

        for j in range(self.m):
            t = int(feat_to_block[j].item())
            hj = h[:, t]
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

        # canonical -> observed
        if apply_perm and (self.perm is not None):
            X_out = X_out[:, self.perm]
            R_out = R_out[:, self.perm]

        return X_out, R_out, y_out


# ============================================================
# End-to-end fit helper with GO-BS
# ============================================================

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
    save_dir: Optional[str] = None,
    save_name: str = "blocksubunit",
    save_best: bool = True,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    return_train_info: bool = False,
    # GO-BS options
    use_gobs: bool = False,
    gobs_num_clusters: int = 7,
    gobs_metric: str = "kl_divergence",
    gobs_bins: int = 32,
    gobs_top_k: Optional[int] = None,
    gobs_refine_order: bool = True,
    gobs_direction_select: bool = True,
    gobs_refine_passes: int = 1,
    gobs_boundary_refine_passes: int = 5,
    gobs_boundary_window: int = 8,
    gobs_lambda_cross: float = 1.0,
    gobs_gamma_balance: float = 1e-3,
    gobs_use_cpu_kmeans: bool = False,
    use_gobs_fc: bool = False,
) -> Union[
    BlockSubunitGenerator,
    Tuple[BlockSubunitGenerator, Dict[str, Union[int, float, str, bool, None, np.ndarray]]],
]:
    set_seed(seed)
    X = np.asarray(X)
    n, m = X.shape
    _validate_feature_specs(feature_specs, m)

    R = np.isfinite(X).astype(np.int64)

    n_classes = None
    if y is not None:
        y = np.asarray(y).astype(int)
        n_classes = int(y.max()) + 1

    learned_perm_obs_to_can = None
    learned_perm_can_to_obs = None
    learned_boundaries = None

    # --------------------------------------------------------
    # GO-BS preprocessing
    # --------------------------------------------------------
    if use_gobs:
        gobs_cls = GOBSFCOrdering if use_gobs_fc else GOBSOrdering

        gobs = gobs_cls(
            n_blocks=M,
            num_clusters=gobs_num_clusters,
            metric=gobs_metric,
            bins=gobs_bins,
            top_k=gobs_top_k,
            refine_order=gobs_refine_order,
            direction_select=gobs_direction_select,
            refine_passes=gobs_refine_passes,
            boundary_refine_passes=gobs_boundary_refine_passes,
            boundary_window=gobs_boundary_window,
            lambda_cross=gobs_lambda_cross,
            gamma_balance=gobs_gamma_balance,
            device=device,
        )

        gobs_res = gobs.fit(
            X_train=X,
            y=y,
            seed=seed,
            deterministic=True,
            use_cpu_kmeans=gobs_use_cpu_kmeans,
        )

        learned_perm_obs_to_can = gobs_res.perm_obs_to_can
        learned_perm_can_to_obs = gobs_res.perm_can_to_obs
        learned_boundaries = gobs_res.boundaries
        blocks = gobs_res.blocks

        X_train = X[:, learned_perm_obs_to_can]
        R_train = R[:, learned_perm_obs_to_can]
        feature_specs_train = reorder_feature_specs(feature_specs, learned_perm_obs_to_can)
    else:
        X_train = X
        R_train = R
        feature_specs_train = feature_specs
        if blocks is None:
            blocks = make_equal_blocks(m=m, M=M)

    # --------------------------------------------------------
    # Create generator in canonical space
    # --------------------------------------------------------
    gen = BlockSubunitGenerator(
        feature_specs=feature_specs_train,
        blocks=blocks,
        n_classes=n_classes,
        prior_type=prior_type,
        device=device,
        use_class_cond_marginals=True,
        use_class_cond_missingness=True,
    )

    # generator expects canonical -> observed permutation for sample output
    if use_gobs and learned_perm_can_to_obs is not None:
        gen.set_permutation(learned_perm_can_to_obs)
    elif permute_features:
        perm = np.random.permutation(m)
        gen.set_permutation(perm)
    else:
        gen.set_permutation(None)

    # --------------------------------------------------------
    # Fit BSTabDiff in canonical space
    # --------------------------------------------------------
    gen.fit_marginals(X=X_train, R=R_train, y=y)
    h_hat = gen.infer_h(X=X_train, R=R_train, y=y)
    gen.fit_emissions(X=X_train, R=R_train, y=y, h_hat=h_hat)

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

    if use_gobs:
        train_info["gobs_used"] = True
        train_info["gobs_variant"] = "GO-BS-FC" if use_gobs_fc else "GO-BS"
        train_info["use_gobs_fc"] = bool(use_gobs_fc)
        train_info["gobs_num_clusters"] = gobs_num_clusters
        train_info["gobs_metric"] = gobs_metric
        train_info["gobs_top_k"] = gobs_top_k
        train_info["gobs_boundaries"] = learned_boundaries
        train_info["gobs_perm_obs_to_can"] = learned_perm_obs_to_can
        train_info["gobs_perm_can_to_obs"] = learned_perm_can_to_obs
    else:
        train_info["gobs_used"] = False
        train_info["gobs_variant"] = None
        train_info["use_gobs_fc"] = False

    if return_train_info:
        return gen, train_info
    return gen


# ============================================================
# Minimal demo
# ============================================================

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
        device="cuda" if torch.cuda.is_available() else "cpu",
        prior_epochs=300,
        verbose_every=100,
        save_dir="checkpoints_demo",
        save_name="demo_M20_diff_gobs",
        save_best=True,
        use_ema=True,
        ema_decay=0.999,
        return_train_info=True,
        use_gobs=True,
        gobs_num_clusters=7,
        gobs_metric="kl_divergence",
        gobs_bins=32,
        gobs_top_k=32,
        gobs_refine_order=True,
        gobs_direction_select=True,
        gobs_refine_passes=1,
        gobs_boundary_refine_passes=5,
        gobs_boundary_window=8,
        gobs_lambda_cross=1.0,
        gobs_gamma_balance=1e-3,
        gobs_use_cpu_kmeans=False,
    )

    print("Train info keys:", list(info.keys()))
    X_syn, R_syn, y_syn = gen.sample(n=50)
    print("Synthetic shapes:", X_syn.shape, R_syn.shape, None if y_syn is None else y_syn.shape)