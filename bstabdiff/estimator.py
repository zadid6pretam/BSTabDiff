"""
Scikit-learn style interface for BSTabDiff.


This module is intentionally a thin wrapper around the existing
fit_block_subunit_generator API. The core BSTabDiff implementation
remains unchanged.
"""


from __future__ import annotations


from typing import List, Optional, Union


import numpy as np
import torch


from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


from .bstabdiff_gobs import (
    FeatureSpec,
    BlockSubunitGenerator,
    fit_block_subunit_generator,
)




class BSTabDiff(BaseEstimator):
    """
    Scikit-learn-style estimator interface for BSTabDiff.


    BSTabDiff is a generative model, so its natural interface is::


        model.fit(X, y)
        X_syn, y_syn = model.sample(n_samples=100)


    rather than ``predict()``, which is conventionally used for
    discriminative prediction in scikit-learn.


    Parameters
    ----------
    feature_specs : list of FeatureSpec, optional
        Schema describing each feature. If None, all features are
        treated as continuous.


    M : int, default=32
        Number of block subunits.


    blocks : list of ndarray, optional
        Predefined feature blocks. If None, blocks are learned or
        constructed automatically by the existing BSTabDiff backend.


    permute_features : bool, default=False
        Whether to randomly permute features when GO-BS is disabled.


    prior_type : {"diffusion", "flow"}, default="diffusion"
        Latent block prior.


    device : str, default="auto"
        Torch device. ``"auto"`` selects CUDA when available,
        otherwise CPU. Explicit devices such as ``"cuda:0"`` are
        also supported.


    random_state : int, default=0
        Random seed.


    prior_epochs : int, default=1500
        Number of prior-training epochs.


    prior_batch : int, default=128
        Prior-training batch size.


    prior_lr : float, default=1e-3
        Prior-training learning rate.


    verbose_every : int, default=200
        Training logging frequency. Use 0 to suppress progress output.


    save_dir : str, optional
        Directory for checkpoints.


    save_name : str, default="blocksubunit"
        Checkpoint base name.


    save_best : bool, default=True
        Whether to retain the best prior state.


    use_ema : bool, default=True
        Whether to use exponential moving average for diffusion training.


    ema_decay : float, default=0.999
        EMA decay.


    use_gobs : bool, default=False
        Whether to use GO-BS feature ordering and block segmentation.


    gobs_num_clusters : int, default=7
        Number of GO-BS clusters.


    gobs_metric : str, default="kl_divergence"
        GO-BS graph metric.


    gobs_bins : int, default=32
        Number of histogram bins for KL-based GO-BS.


    gobs_top_k : int, optional
        Optional graph sparsification parameter.


    gobs_refine_order : bool, default=True
        Whether to locally refine the feature ordering.


    gobs_direction_select : bool, default=True
        Whether to evaluate ordering direction.


    gobs_refine_passes : int, default=1
        Number of ordering refinement passes.


    gobs_boundary_refine_passes : int, default=5
        Number of block-boundary refinement passes.


    gobs_boundary_window : int, default=8
        Boundary-search window.


    gobs_lambda_cross : float, default=1.0
        Cross-block leakage penalty.


    gobs_gamma_balance : float, default=1e-3
        Block-size balancing regularization.


    gobs_use_cpu_kmeans : bool, default=False
        Force CPU KMeans for GO-BS.


    use_gobs_fc : bool, default=False
        Use feature-clustered GO-BS instead of the standard
        sample-clustered GO-BS.
    """


    def __init__(
        self,
        feature_specs: Optional[List[FeatureSpec]] = None,
        M: int = 32,
        blocks=None,
        permute_features: bool = False,
        prior_type: str = "diffusion",
        device: str = "auto",
        random_state: int = 0,
        prior_epochs: int = 1500,
        prior_batch: int = 128,
        prior_lr: float = 1e-3,
        verbose_every: int = 200,
        save_dir: Optional[str] = None,
        save_name: str = "blocksubunit",
        save_best: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
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
    ):
        # IMPORTANT:
        # sklearn estimators should only assign constructor arguments here.
        # Do not perform fitting or parameter modification in __init__.


        self.feature_specs = feature_specs
        self.M = M
        self.blocks = blocks
        self.permute_features = permute_features
        self.prior_type = prior_type
        self.device = device
        self.random_state = random_state
        self.prior_epochs = prior_epochs
        self.prior_batch = prior_batch
        self.prior_lr = prior_lr
        self.verbose_every = verbose_every
        self.save_dir = save_dir
        self.save_name = save_name
        self.save_best = save_best
        self.use_ema = use_ema
        self.ema_decay = ema_decay


        self.use_gobs = use_gobs
        self.gobs_num_clusters = gobs_num_clusters
        self.gobs_metric = gobs_metric
        self.gobs_bins = gobs_bins
        self.gobs_top_k = gobs_top_k
        self.gobs_refine_order = gobs_refine_order
        self.gobs_direction_select = gobs_direction_select
        self.gobs_refine_passes = gobs_refine_passes
        self.gobs_boundary_refine_passes = gobs_boundary_refine_passes
        self.gobs_boundary_window = gobs_boundary_window
        self.gobs_lambda_cross = gobs_lambda_cross
        self.gobs_gamma_balance = gobs_gamma_balance
        self.gobs_use_cpu_kmeans = gobs_use_cpu_kmeans
        self.use_gobs_fc = use_gobs_fc


    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"


        return self.device


    def _make_feature_specs(self, X, n_features: int):
        if self.feature_specs is not None:
            return self.feature_specs


        # Preserve DataFrame column names when possible.
        if hasattr(X, "columns"):
            names = [str(col) for col in X.columns]
        else:
            names = [f"f{j}" for j in range(n_features)]


        return [
            FeatureSpec(
                name=name,
                kind="continuous",
            )
            for name in names
        ]


    def fit(self, X, y=None):
        """
        Fit BSTabDiff.


        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training table. NaN values are supported by the BSTabDiff backend.


        y : array-like of shape (n_samples,), optional
            Optional class-conditioning labels.


        Returns
        -------
        self : BSTabDiff
            Fitted estimator.
        """


        # Save feature names before conversion.
        if hasattr(X, "columns"):
            columns = list(X.columns)


            if all(isinstance(c, str) for c in columns):
                self.feature_names_in_ = np.asarray(
                    columns,
                    dtype=object,
                )


        X_arr = np.asarray(X, dtype=np.float32)


        if X_arr.ndim != 2:
            raise ValueError(
                "X must be a 2D array with shape "
                "(n_samples, n_features)."
            )


        n_samples, n_features = X_arr.shape


        if n_samples < 1:
            raise ValueError("X must contain at least one sample.")


        if n_features < 1:
            raise ValueError("X must contain at least one feature.")


        self.n_features_in_ = n_features


        # ---------------------------------------------------------
        # Feature schema
        # ---------------------------------------------------------


        feature_specs = self._make_feature_specs(
            X,
            n_features,
        )


        self.feature_specs_ = feature_specs


        # ---------------------------------------------------------
        # Class labels
        #
        # The core BSTabDiff implementation expects labels encoded
        # as integers 0, ..., K-1. LabelEncoder allows sklearn-style
        # arbitrary labels such as:
        #
        # ["control", "disease", ...]
        # ---------------------------------------------------------


        y_encoded = None
        self.label_encoder_ = None


        if y is not None:
            y_arr = np.asarray(y)


            if y_arr.ndim != 1:
                y_arr = y_arr.ravel()


            if y_arr.shape[0] != n_samples:
                raise ValueError(
                    "X and y contain different numbers of samples."
                )


            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y_arr)


            self.classes_ = self.label_encoder_.classes_


        # ---------------------------------------------------------
        # Resolve CPU / CUDA
        # ---------------------------------------------------------


        self.device_ = self._resolve_device()


        # ---------------------------------------------------------
        # Delegate ALL actual training to the existing API.
        # ---------------------------------------------------------


        generator, train_info = fit_block_subunit_generator(
            X=X_arr,
            feature_specs=feature_specs,
            y=y_encoded,
            M=self.M,
            blocks=self.blocks,
            permute_features=self.permute_features,
            prior_type=self.prior_type,
            device=self.device_,
            seed=self.random_state,
            prior_epochs=self.prior_epochs,
            prior_batch=self.prior_batch,
            prior_lr=self.prior_lr,
            verbose_every=self.verbose_every,
            save_dir=self.save_dir,
            save_name=self.save_name,
            save_best=self.save_best,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            return_train_info=True,
            use_gobs=self.use_gobs,
            gobs_num_clusters=self.gobs_num_clusters,
            gobs_metric=self.gobs_metric,
            gobs_bins=self.gobs_bins,
            gobs_top_k=self.gobs_top_k,
            gobs_refine_order=self.gobs_refine_order,
            gobs_direction_select=self.gobs_direction_select,
            gobs_refine_passes=self.gobs_refine_passes,
            gobs_boundary_refine_passes=self.gobs_boundary_refine_passes,
            gobs_boundary_window=self.gobs_boundary_window,
            gobs_lambda_cross=self.gobs_lambda_cross,
            gobs_gamma_balance=self.gobs_gamma_balance,
            gobs_use_cpu_kmeans=self.gobs_use_cpu_kmeans,
            use_gobs_fc=self.use_gobs_fc,
        )


        self.generator_: BlockSubunitGenerator = generator
        self.train_info_ = train_info


        return self


    def _encode_sampling_y(
        self,
        n_samples: int,
        y=None,
    ):
        if self.label_encoder_ is None:
            if y is not None:
                raise ValueError(
                    "This estimator was fitted without y, so "
                    "class-conditional sampling is unavailable."
                )


            return None


        if y is None:
            # Existing BSTabDiff behavior:
            # randomly sample a class internally.
            return None


        # Single class requested for every generated sample.
        if np.isscalar(y):
            encoded = self.label_encoder_.transform(
                np.asarray([y])
            )[0]


            return int(encoded)


        y_arr = np.asarray(y)


        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()


        if y_arr.shape[0] != n_samples:
            raise ValueError(
                "When y is an array, it must contain exactly "
                f"{n_samples} labels."
            )


        return self.label_encoder_.transform(y_arr)


    def sample(
        self,
        n_samples: int = 1,
        y=None,
        return_mask: bool = False,
    ):
        """
        Generate synthetic samples from the fitted BSTabDiff model.


        Parameters
        ----------
        n_samples : int, default=1
            Number of synthetic rows.


        y : scalar or array-like, optional
            Optional class-conditioning labels.


            Examples
            --------
            Generate 100 samples from class 1::


                model.sample(100, y=1)


            Generate samples with specified labels::


                model.sample(
                    4,
                    y=[0, 0, 1, 1],
                )


        return_mask : bool, default=False
            If True, also return the BSTabDiff observation mask.


        Returns
        -------
        X_syn, y_syn
            If return_mask=False.


        X_syn, R_syn, y_syn
            If return_mask=True.
        """


        check_is_fitted(
            self,
            attributes=["generator_"],
        )


        n_samples = int(n_samples)


        if n_samples <= 0:
            raise ValueError(
                "n_samples must be a positive integer."
            )


        y_encoded = self._encode_sampling_y(
            n_samples=n_samples,
            y=y,
        )


        X_syn, R_syn, y_syn_encoded = self.generator_.sample(
            n=n_samples,
            y=y_encoded,
        )


        # Convert internal integer classes back to the labels
        # supplied by the user during fit().
        y_syn = None


        if y_syn_encoded is not None:
            if self.label_encoder_ is not None:
                y_syn = self.label_encoder_.inverse_transform(
                    np.asarray(
                        y_syn_encoded,
                        dtype=int,
                    )
                )
            else:
                y_syn = y_syn_encoded


        if return_mask:
            return X_syn, R_syn, y_syn


        return X_syn, y_syn


    def fit_resample(
        self,
        X,
        y=None,
        n_samples: Optional[int] = None,
    ):
        """
        Convenience fit-and-generate interface.


        Parameters
        ----------
        X : array-like
            Training data.


        y : array-like, optional
            Conditioning labels.


        n_samples : int, optional
            Number of generated rows. Defaults to len(X).


        Returns
        -------
        X_syn, y_syn
            Synthetic dataset.
        """


        self.fit(X, y)


        if n_samples is None:
            n_samples = len(X)


        return self.sample(
            n_samples=n_samples,
        )


    @property
    def generator(self):
        """
        Access the underlying fitted BlockSubunitGenerator.
        """


        check_is_fitted(
            self,
            attributes=["generator_"],
        )


        return self.generator_
