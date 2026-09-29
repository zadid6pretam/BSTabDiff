"""
BSTabDiff package

This module exposes the main BSTabDiff generator, GO-BS ordering variants,
and helper utilities used in the NeurIPS 2026 BSTabDiff paper.
"""

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

__version__ = "0.1.0"

__all__ = [
    # Core API
    "fit_block_subunit_generator",
    "BlockSubunitGenerator",

    # Feature schema
    "FeatureSpec",

    # GO-BS ordering
    "GOBSOrdering",
    "GOBSFCOrdering",
    "GOBSResult",

    # Priors and emission components
    "DiffusionPrior",
    "FlowPrior",
    "EmissionParams",
    "EmpiricalMarginals",

    # Utilities
    "set_seed",
    "make_equal_blocks",
    "apply_permutation",
    "invert_permutation",
    "contiguous_blocks_from_boundaries",
    "reorder_feature_specs",
    "infer_block_latents_mean_gaussianized",
    "fit_emissions_from_inferred_h",
]
