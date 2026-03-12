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
        name="6j100pt",
        selection="cat_6j100pt",  # Exactly 6 jets with pT >= 100 GeV
        label=r"6 jets with $p_T > 100\,\mathrm{GeV}$",
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
        name="1btj",
        selection="cat_1btj",  # Exactly 1 b-tagged jet (tight WP)
        label="1 b-tagged jet",
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
        name="fit_conv_leq",
        selection="cat_fit_conv_leq",  # Converged with Pgof <= 0.1 (poor quality)
        label="kinfit converged and $P_{gof} < 0.1$",
    )
    add_category(
        cfg,
        name="fit_conv_big",
        selection="cat_fit_conv_big",  # Converged with Pgof > 0.1 (good quality)
        label="$P_{gof} > 0.1$ ",
    )
    add_category(
        cfg,
        name="fit_conv_leq_rbb",
        selection="cat_fit_conv_leq_rbb",
        label=r"below $\chi^2$ cut and above $\Delta R_{\text{b}}$ cut",
    )
    add_category(
        cfg,
        name="fit_Pgof_02",
        selection="cat_fit_Pgof_02",  # Pgof > 0.2 (very good quality)
        label="kinfit $P_{gof} > 0.2$ ",
    )
    add_category(
        cfg,
        name="fit_rbb",
        selection="cat_rbb",
        label=r"above $\Delta R_{\text{b}}$ cut",
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

    # ==============================================================================
    # Jet selection categories
    # ==============================================================================
    add_category(
        cfg,
        name="sig_correct",
        selection="cat_sig_correct",
        label="Signal Region + \n correctly matched events",
    )
    add_category(
        cfg,
        name="jetid",
        selection="cat_jetid",
        label="Signal Region +\n" r"$\geq 6$ jets passing TightLepVeto Jet ID",
    )
    add_category(
        cfg,
        name="jetid_correct",
        selection="cat_jetid_correct",
        label="Signal Region + \n correctly matched events +\n" r"$\geq 6$ jets passing TightLepVeto Jet ID",
    )
    add_category(
        cfg,
        name="puid",
        selection="cat_jetpuid",
        label="Signal Region + \n" r"$\geq 6$ jets passing Tight Pileup ID",
    )
    add_category(
        cfg,
        name="puid_correct",
        selection="cat_jetpuid_correct",
        label="Signal Region + \n correctly matched events +\n" r"$\geq 6$ jets passing Tight Pileup ID",
    )
    add_category(
        cfg,
        name="veto_map_mask",
        selection="cat_jetvetomap",
        label="Signal Region + \n" r"$\geq 6$ jets passing Jet Veto Map",
    )
    add_category(
        cfg,
        name="veto_map_mask_correct",
        selection="cat_jetvetomap_correct",
        label="Signal Region + \n correctly matched events +\n" r"$\geq 6$ jets passing Jet Veto Map",
    )
    add_category(
        cfg,
        name="jetid_puid",
        selection="cat_jetid_puid",
        label="Signal Region + \n" r"$\geq 6$ jets passing Jet ID + Pileup ID",
    )
    add_category(
        cfg,
        name="jetid_puid_correct",
        selection="cat_jetid_puid_correct",
        label="Signal Region + \n correctly matched events +\n" r"$\geq 6$ jets passing Jet ID + Pileup ID",
    )
    add_category(
        cfg,
        name="jetcleaning",
        selection="cat_jetfullclean",
        label="Signal Region + \n" r"$\geq 6$ jets passing Full Jet Cleaning",
    )
    add_category(
        cfg,
        name="jetcleaning_correct",
        selection="cat_jetfullclean_correct",
        label="Signal Region + \n correctly matched events + \n" r"$\geq 6$ jets passing Full Jet Cleaning",
    )
    # add_category(
    #     cfg,
    #     name="reco_sig",
    #     selection="cat_reco_sig",
    #     label=r"below $\chi^2$ cut and above $\Delta R_{\text{b}}$ cut",
    # )
    # Uncomment to define orthogonal or overlapping category sets:
    #
    # main_categories = {
    #     # number of jets
    #     "njets": CategoryGroup(["incl", "6j", "7j"], is_complete=True, has_overlap=True),
    #     # number of btagged jets
    #     "sigorbkg": CategoryGroup(["sig", "bkg"], is_complete=False, has_overlap=False),
    #     # kinematic fit convergence
    #     "kinfitconv": CategoryGroup(["fit_conv", "fit_nconv"], is_complete=True, has_overlap=False),
    # }
