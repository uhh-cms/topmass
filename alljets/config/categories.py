# coding: utf-8

"""
Definition of categories for the top mass analysis.

This module defines event categories used to classify events based on:

Each category defined here must have a corresponding categorizer function
in alljets/categorization/example.py that implements the selection logic.

Structure of add_category:
    - name: Internal identifier used in code
    - selection: Name of the categorizer function in categorization/example.py
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
        label="kinfit converged and $P_{gof} > 0.1$ ",
    )
    add_category(
        cfg,
        name="fit_Pgof_02",
        selection="cat_fit_Pgof_02",  # Pgof > 0.2 (very good quality)
        label="kinfit $P_{gof} > 0.2$ ",
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
        label=">2 b-tagged jets & signal trigger",
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

# ============================================================================
# analyze jet overlap
# ============================================================================

# ----------------------------------------------------------------------------
# GenParticles Cuts
# ----------------------------------------------------------------------------

    add_category(
        cfg,
        name="gen_eta21_pt60",
        selection="gen_eta21_pt60",
        label="GenParticle eta2.1+pt60 Cut",
    )
    add_category(
        cfg,
        name="gen_eta21_pt60_corr",
        selection="gen_eta21_pt60_corr",
        label="GenParticle eta2.1+pt60 Cut + corr",
    )
    add_category(
        cfg,
        name="gen_eta21_pt60_conv",
        selection="gen_eta21_pt60_conv",
        label="GenParticle eta2.1+pt60 Cut + conv",
    )
    add_category(
        cfg,
        name="gen_eta21_pt60_nconv",
        selection="gen_eta21_pt60_nconv",
        label="GenParticle eta2.1+pt60 Cut + nconv",
    )
    add_category(
        cfg,
        name="gen_eta21_pt60_corrB",
        selection="gen_eta21_pt60_corrB",
        label="GenParticle eta2.1+pt60 Cut + corr (B)",
    )

# ----------------------------------------------------------------------------
# Delta Rmin Cuts t1
# ----------------------------------------------------------------------------

    add_category(
        cfg,
        name="deltaRmin08_t1",
        selection="deltaRmin08_t1",
        label="dRmin_t1 < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin08_t1",
        selection="gen_cut_deltaRmin08_t1",
        label="gen cut + dRmin < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin06_t1",
        selection="gen_cut_deltaRmin06_t1",
        label="gen cut + dRmin < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin05_t1",
        selection="gen_cut_deltaRmin05_t1",
        label="gen cut + dRmin < 0.5",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin04_t1",
        selection="gen_cut_deltaRmin04_t1",
        label="gen cut + dRmin < 0.4",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin06_08_t1",
        selection="gen_cut_deltaRmin06_08_t1",
        label="gen cut + 0.6 < dRmin < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin04_06_t1",
        selection="gen_cut_deltaRmin04_06_t1",
        label="gen cut + 0.4 < dRmin < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRmin08_inf_t1",
        selection="gen_cut_deltaRmin08_inf_t1",
        label="gen cut + 0.8 < dRmin",
    )
    add_category(
        cfg,
        name="gen_cut_corr_deltaRmin08_t1",
        selection="gen_cut_deltaRmin08_inf_t1",
        label="gen cut + corr + dRmin < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_corr_deltaRmin06_t1",
        selection="gen_cut_corr_deltaRmin06_t1",
        label="gen cut + corr + dRmin < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_wrong_deltaRmin06_t1",
        selection="gen_cut_wrong_deltaRmin06_t1",
        label="gen cut + wrong + dRmin < 0.6",
    )
# ----------------------------------------------------------------------------
# Delta R Jet to all other Jets
# ----------------------------------------------------------------------------
    add_category(
        cfg,
        name="gen_cut_corr_deltaR06_q1",
        selection="gen_cut_corr_deltaR06_q1",
        label="gen cut + corr + dR < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaR06_q1",
        selection="gen_cut_deltaR06_q1",
        label="gen cut + dR < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaR04_06_q1",
        selection="gen_cut_deltaR04_06_q1",
        label="gen cut + 0.4 < dR < 0.6",
    )
# ----------------------------------------------------------------------------
# Angular Distance recoJet to recoJet
# ----------------------------------------------------------------------------
    add_category(
        cfg,
        name="gen_cut_deltaRrecoJet08_q1q2",
        selection="gen_cut_deltaRrecoJet08_q1q2",
        label=r"gen cut + $dR^{reco}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRrecoJet06_q1q2",
        selection="gen_cut_deltaRrecoJet06_q1q2",
        label=r"gen cut + $dR^{reco}_{q1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRrecoJet04_q1q2",
        selection="gen_cut_deltaRrecoJet04_q1q2",
        label=r"gen cut + $dR^{reco}_{q1q2}$ < 0.4",
    )
# ----------------------------------------------------------------------------
# Angular Distance GenParton to GenParton
# ----------------------------------------------------------------------------
    add_category(
        cfg,
        name="gen_cut_deltaRgen08_q1q2",
        selection="gen_cut_deltaRgen08_q1q2",
        label=r"gen cut + $dR^{gen}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_q1q2",
        selection="gen_cut_deltaRgen06_q1q2",
        label=r"gen cut + $dR^{gen}_{q1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_q1q2",
        selection="gen_cut_deltaRgen04_q1q2",
        label=r"gen cut + $dR^{gen}_{q1q2}$ < 0.4",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRgen08_q1q2_corr",
        selection="gen_cut_deltaRgen08_q1q2_corr",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_q1q2_corr",
        selection="gen_cut_deltaRgen06_q1q2_corr",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_q1q2_corr",
        selection="gen_cut_deltaRgen04_q1q2_corr",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ < 0.4",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRgen08_inf_q1q2",
        selection="gen_cut_deltaRgen08_inf_q1q2",
        label=r"gen cut + $dR^{gen}_{q1q2}$ > 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_08_q1q2",
        selection="gen_cut_deltaRgen06_08_q1q2",
        label=r"gen cut + 0.6 < $dR^{gen}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_06_q1q2",
        selection="gen_cut_deltaRgen04_06_q1q2",
        label=r"gen cut + 0.4 < $dR^{gen}_{q1q2}$ < 0.6",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRgen08_inf_q1q2_corr",
        selection="gen_cut_deltaRgen04_06_q1q2_corr",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ > 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_08_q1q2_corr",
        selection="gen_cut_deltaRgen06_08_q1q2_corr",
        label=r"gen cut + corr + 0.6 < $dR^{gen}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_06_q1q2_corr",
        selection="gen_cut_deltaRgen04_06_q1q2_corr",
        label=r"gen cut + corr + 0.4 < $dR^{gen}_{q1q2}$ < 0.6",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRgen08_q1q2_corr_pgof",
        selection="gen_cut_deltaRgen08_q1q2_corr_pgof",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ < 0.8 + pgof",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_q1q2_corr_pgof",
        selection="gen_cut_deltaRgen06_q1q2_corr_pgof",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ < 0.6 + pgof",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_q1q2_corr_pgof",
        selection="gen_cut_deltaRgen04_q1q2_corr_pgof",
        label=r"gen cut + corr + $dR^{gen}_{q1q2}$ < 0.4 + pgof",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRgen08_q1q2_corrB",
        selection="gen_cut_deltaRgen08_q1q2_corrB",
        label=r"gen cut + corr (B) + $dR^{gen}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_q1q2_corrB",
        selection="gen_cut_deltaRgen06_q1q2_corrB",
        label=r"gen cut + corr (B) + $dR^{gen}_{q1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_q1q2_corrB",
        selection="gen_cut_deltaRgen04_q1q2_corrB",
        label=r"gen cut + corr (B) + $dR^{gen}_{q1q2}$ < 0.4",
    )
# ----------------------------------------------------------------------------
# Delta R_min  b-Parton to q1q2
# ----------------------------------------------------------------------------
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_08_q1q2",
        selection="gen_cut_deltaRminbq1q2_08_q1q2",
        label=r"gen cut +  $dR^{gen}_{min,bq1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_06_q1q2",
        selection="gen_cut_deltaRminbq1q2_06_q1q2",
        label=r"gen cut +  $dR^{gen}_{min,bq1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_04_q1q2",
        selection="gen_cut_deltaRminbq1q2_04_q1q2",
        label=r"gen cut +  $dR^{gen}_{min,bq1q2}$ < 0.4",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_08_q1q2_corrB",
        selection="gen_cut_deltaRminbq1q2_08_q1q2_corrB",
        label=r"gen cut + corr (B) +  $dR^{gen}_{min,bq1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_06_q1q2_corrB",
        selection="gen_cut_deltaRminbq1q2_06_q1q2_corrB",
        label=r"gen cut + corr (B) +  $dR^{gen}_{min,bq1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_04_q1q2_corrB",
        selection="gen_cut_deltaRminbq1q2_04_q1q2_corrB",
        label=r"gen cut + corr (B) +  $dR^{gen}_{min,bq1q2}$ < 0.4",
    )

    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_08_q1q2_corr",
        selection="gen_cut_deltaRminbq1q2_08_q1q2_corr",
        label=r"gen cut + corr +  $dR^{gen}_{min,bq1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_06_q1q2_corr",
        selection="gen_cut_deltaRminbq1q2_06_q1q2_corr",
        label=r"gen cut + corr +  $dR^{gen}_{min,bq1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRminbq1q2_04_q1q2_corr",
        selection="gen_cut_deltaRminbq1q2_04_q1q2_corr",
        label=r"gen cut + corr +  $dR^{gen}_{min,bq1q2}$ < 0.4",
    )

# ----------------------------------------------------------------------------
# Angular Distance genParton to genParton (closet)
# ----------------------------------------------------------------------------
    add_category(
        cfg,
        name="gen_cut_deltaRgen08_q1q2_closest",
        selection="gen_cut_deltaRgen08_q1q2_closest",
        label=r"gen cut  +  $dR^{gen}_{q1q2}$ < 0.8",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen06_q1q2_closest",
        selection="gen_cut_deltaRgen06_q1q2_closest",
        label=r"gen cut  +  $dR^{gen}_{q1q2}$ < 0.6",
    )
    add_category(
        cfg,
        name="gen_cut_deltaRgen04_q1q2_closest",
        selection="gen_cut_deltaRgen04_q1q2_closest",
        label=r"gen cut  +  $dR^{gen}_{q1q2}$ < 0.4",
    )
# ----------------------------------------------------------------------------
# Multiple matching Jets
# ----------------------------------------------------------------------------
    add_category(
        cfg,
        name="gen_cut_mergingJets_q1q2",
        selection="gen_cut_mergingJets_q1q2",
        label=r"gen cut + multiple matching jet q1q2",
    )
    add_category(
        cfg,
        name="gen_cut_mergingJets_once_q1q2",
        selection="gen_cut_mergingJets_once_q1q2",
        label=r"gen cut + multiple matching jet (q1q2)",
    )
    add_category(
        cfg,
        name="gen_cut_unmatched_q1q2",
        selection="gen_cut_unmatched_q1q2",
        label=r"gen cut + unmatched q1q2",
    )
    add_category(
        cfg,
        name="gen_cut_unmatched_reco_q1q2",
        selection="gen_cut_unmatched_reco_q1q2",
        label=r"gen cut + unmatched (fit) q1q2 ",
    )

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
