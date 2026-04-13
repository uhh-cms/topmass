# coding: utf-8

"""
Definition of categories for the top mass analysis.

This module defines event categories used to classify events.

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
        selection="cat_incl",
        label="inclusive",
    )

    # ========================================================================
    # Jet multiplicity categories
    # ========================================================================

    add_category(
        cfg,
        name="6j",
        selection="cat_6j",
        label="6 jets",
    )
    add_category(
        cfg,
        name="7j",
        selection="cat_7j",
        label="7+ jets",
    )

    # ========================================================================
    # B-tagging categories
    # ========================================================================

    add_category(
        cfg,
        name="2btj",
        selection="cat_2btj",
        label="2 b-tagged jets or more",
    )
    add_category(
        cfg,
        name="0btj",
        selection="cat_0btj",
        label="0 b-tagged jets",
    )

    # ========================================================================
    # Kinematic fit convergence categories
    # ========================================================================

    add_category(
        cfg,
        name="fit_nconv",
        selection="cat_fit_nconv",
        label="kinfit not converged",
    )
    add_category(
        cfg,
        name="fit_conv",
        selection="cat_fit_conv",
        label="kinfit converged",
    )

    # ========================================================================
    # Kinematic fit quality categories
    # ========================================================================

    add_category(
        cfg,
        name="fitPgof_fail",
        selection="cat_fitPgof_fail",
        label="kinfit converged and $P_{gof} < 0.1$",
    )
    add_category(
        cfg,
        name="fitPgof_pass",
        selection="cat_fitPgof_pass",
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
    add_category(
        cfg,
        name="rbb_sig",
        selection="cat_Rbb_sig",
        label=r"$\Delta R_{\text{bb}}$ > 2",
    )
    add_category(
        cfg,
        name="pgof_sig",
        selection="cat_FitPgof_sig",
        label=r"$P_{gof} > 0.1$ ",
    )
    # ========================================================================
    # Signal and background regions
    # ========================================================================

    add_category(
        cfg,
        name="bkg",
        selection="cat_0btj_bkg",
        label="QCD estimation",
        tags={"0btj"},
    )
    add_category(
        cfg,
        name="sig",
        selection="cat_2btj_sig",
        label="Signal Region",
        tags={"2btj"},
    )
    # ========================================================================
    # Truth-level matching categories
    # ========================================================================

    add_category(
        cfg,
        name="fit_matched",
        selection="cat_fit_matched",
        label="correctly matched events",
        tags={"matched"},
    )
    add_category(
        cfg,
        name="fit_unmatched",
        selection="cat_fit_unmatched",
        label="wrong or umatched events",
        tags={"matched"},
    )
