"""
BSTabDiff package.

This module exposes:

- The high-level sklearn-style BSTabDiff estimator.
- The original BSTabDiff generator API.
- GO-BS and GO-BS-FC feature ordering variants.
- Prior and emission components.
- Helper utilities used in the NeurIPS 2026 BSTabDiff paper.
"""

# ============================================================
# High-level sklearn-style API
# ============================================================

from .estimator import BSTabDiff


# ============================================================
# Core BSTabDiff implementation
# ============================================================

from .bstabdiff_gobs import (
    # Core API
    fit_block_subunit_generator,
    BlockSubunitGenerator,

    # Feature schema
    FeatureSpec,

    # GO-BS ordering
    GOBSOrdering,
    GOBSFCOrdering,
    GOBSResult,

    # Priors and emission components
    DiffusionPrior,
    FlowPrior,
    EmissionParams,
    EmpiricalMarginals,

    # Utilities
    set_seed,
    make_equal_blocks,
    apply_permutation,
    invert_permutation,
    contiguous_blocks_from_boundaries,
    reorder_feature_specs,
    infer_block_latents_mean_gaussianized,
    fit_emissions_from_inferred_h,
)


# ============================================================
# Package version
# ============================================================

__version__ = "0.2.0"


# ============================================================
# Public API
# ============================================================

__all__ = [
    # --------------------------------------------------------
    # High-level sklearn-style API
    # --------------------------------------------------------
    "BSTabDiff",

    # --------------------------------------------------------
    # Original / research API
    # --------------------------------------------------------
    "fit_block_subunit_generator",
    "BlockSubunitGenerator",

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------
    "FeatureSpec",

    # --------------------------------------------------------
    # GO-BS ordering
    # --------------------------------------------------------
    "GOBSOrdering",
    "GOBSFCOrdering",
    "GOBSResult",

    # --------------------------------------------------------
    # Priors and emission components
    # --------------------------------------------------------
    "DiffusionPrior",
    "FlowPrior",
    "EmissionParams",
    "EmpiricalMarginals",

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------
    "set_seed",
    "make_equal_blocks",
    "apply_permutation",
    "invert_permutation",
    "contiguous_blocks_from_boundaries",
    "reorder_feature_specs",
    "infer_block_latents_mean_gaussianized",
    "fit_emissions_from_inferred_h",
]
