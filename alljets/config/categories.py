# coding: utf-8

"""
Definition of categories for the top mass analysis.

This module defines event categories used to classify events based on:

Each category defined here must have a corresponding categorizer function
in alljets/categorization/default.py that implements the selection logic.

Structure of add_category:
    - name: Internal identifier used in code
    - selection: Name of the categorizer function in categorization/default.py
    - label: Human-readable label for plots (can include LaTeX formatting)
    - tags (optional): Set of tags for grouping related categories
"""

import order as od
from columnflow.config_util import add_category


def add_categories(cfg: od.Config) -> None:
    """
    Add all event categories to the configuration.
    """

    # ========================================================================
    # Inclusive category
    # ========================================================================

    add_category(
        cfg,
        name="incl",
        selection="cat_incl",  # All events
        label="inclusive",
    )

    # ========================================================================
    # Jet multiplicity categories
    # ========================================================================

    add_category(
        cfg,
        name="6j",
        selection="cat_6j",  # Exactly 6 jets with pT >= 40 GeV
        label="6 jets",
    )
    add_category(
        cfg,
        name="7j",
        selection="cat_7j",  # 7 or more jets with pT >= 40 GeV
        label="7+ jets",
    )

    # ========================================================================
    # B-tagging categories
    # ========================================================================

    add_category(
        cfg,
        name="2btj",
        selection="cat_2btj",  # >= 2 b-tagged jets (tight WP)
        label="2 b-tagged jets or more",
    )
    add_category(
        cfg,
        name="0btj",
        selection="cat_0btj",  # 0 b-tagged jets (tight WP)
        label="0 b-tagged jets",
    )

    # ========================================================================
    # Kinematic fit convergence categories
    # ========================================================================

    add_category(
        cfg,
        name="fit_nconv",
        selection="cat_fit_nconv",  # FitChi2 >= 10000 (non-converged)
        label="kinfit not converged",
    )
    add_category(
        cfg,
        name="fit_conv",
        selection="cat_fit_conv",  # FitChi2 < 10000 (converged)
        label="kinfit converged",
    )

    # ========================================================================
    # Kinematic fit quality categories
    # ========================================================================

    add_category(
        cfg,
        name="fitPgof_fail",
        selection="cat_fitPgof_fail",  # Converged with Pgof <= 0.1 (poor quality)
        label="kinfit converged and $P_{gof} < 0.1$",
    )
    add_category(
        cfg,
        name="fitPgof_pass",
        selection="cat_fitPgof_pass",  # Converged with Pgof > 0.1 (good quality)
        label="$P_{gof} > 0.1$ ",
    )
    add_category(
        cfg,
        name="fitPgof_rbb",
        selection="cat_fitPgof_rbb",
        label=r"below $\chi^2$ cut and above $\Delta R_{\text{b}}$ cut",
    )
    add_category(
        cfg,
        name="fit_rbb",
        selection="cat_rbb",
        label=r"$\Delta R_{\text{b}}$ > 2",
    )
    # ========================================================================
    # Signal and background regions
    # ========================================================================

    add_category(
        cfg,
        name="bkg",
        selection="cat_0btj_bkg",  # 0 b-tags + background trigger + fit quality
        label="QCD estimation",
        tags={"0btj"},
    )
    add_category(
        cfg,
        name="sig",
        selection="cat_2btj_sig",  # >= 2 b-tags + signal trigger + fit quality
        label="Signal Region",
        tags={"2btj"},
    )
    # ========================================================================
    # Truth-level matching categories
    # ========================================================================

    add_category(
        cfg,
        name="fit_matched",
        selection="cat_fit_matched",  # fitCombinationType == 2 (correct match)
        label="correctly matched events",
        tags={"matched"},
    )
    add_category(
        cfg,
        name="fit_unmatched",
        selection="cat_fit_unmatched",  # fitCombinationType != 2 (wrong/unmatched)
        label="wrong or umatched events",
        tags={"matched"},
    )
