# coding: utf-8

"""
Categorizer functions for event classification in the top mass kinematic fit analysis.

This module contains the selection logic for each category defined in
alljets/config/categories.py. Each categorizer function takes an awkward array
of events and returns a tuple of (events, selection_mask) where the mask is a
boolean array indicating which events pass the selection.

The @categorizer decorator registers the function and declares which columns
it uses via the 'uses' parameter, enabling automatic dependency tracking and
column loading optimization.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


# ============================================================================
# Inclusive category
# ============================================================================

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Fully inclusive category - selects all events."""
    return events, ak.ones_like(events.event) == 1


# ============================================================================
# Jet multiplicity categorizers
# ============================================================================

@categorizer(uses={"Jet.pt"})
def cat_6j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Select events with exactly 6 jets (pT >= 40 GeV)."""
    return events, ak.sum((events.Jet.pt >= 40.0), axis=1) == 6


@categorizer(uses={"Jet.pt"})
def cat_6j100pt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Select events with exactly 6 jets (pT >= 100 GeV)."""
    return events, ak.sum((events.Jet.pt >= 100.0), axis=1) == 6


@categorizer(uses={"Jet.pt"})
def cat_7j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Select events with 7 or more jets (pT >= 40 GeV)."""
    return events, ak.sum((events.Jet.pt >= 40.0), axis=1) >= 7


# ============================================================================
# B-tagging categorizers
# ============================================================================

@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta"})
def cat_2btj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events with >= 2 b-tagged jets (tight WP).
    Requires: pT >= 40 GeV, |eta| < 2.4, DeepJet b-tag >= tight WP.
    """
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum(
        (events.Jet.pt >= 40.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) >= 2
    )


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta"})
def cat_1btj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events with exactly 1 b-tagged jet (tight WP).
    Requires: pT >= 40 GeV, |eta| < 2.4, DeepJet b-tag >= tight WP.
    """
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum(
        (events.Jet.pt >= 40.0) &
        abs(events.Jet.eta < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) == 1
    )


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta"})
def cat_0btj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events with 0 b-tagged jets (tight WP).
    Requires: pT >= 40 GeV, |eta| < 2.4, DeepJet b-tag < tight WP.
    """
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum((
        events.Jet.pt >= 40.0) &
        abs(events.Jet.eta < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) == 0
    )


# ============================================================================
# Kinematic fit quality categorizers
# ============================================================================

@categorizer(uses={"FitPgof", "FitChi2"})
def cat_fit_conv_leq(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events where kinematic fit converged with poor quality.
    Requires: FitChi2 < 10000 (converged) and FitPgof <= config threshold.
    """
    pgofcut = self.config_inst.x.fitpgofcut
    return events, (events.FitChi2 < 10000) & (events.FitPgof <= pgofcut)


@categorizer(uses={"FitPgof", "FitChi2"})
def cat_fit_conv_big(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events where kinematic fit converged with good quality.
    Requires: FitChi2 < 10000 (converged) and FitPgof > config threshold.
    """
    pgofcut = self.config_inst.x.fitpgofcut
    return events, (events.FitChi2 < 10000) & (events.FitPgof > pgofcut)


@categorizer(uses={"FitPgof"})
def cat_fit_Pgof_02(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Select events with very good kinematic fit quality (Pgof > 0.2)."""
    pgofcut = 0.2
    return events, events.FitPgof > pgofcut


@categorizer(uses={"FitChi2"})
def cat_fit_nconv(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Select events where kinematic fit did not converge (FitChi2 >= 10000)."""
    return events, (events.FitChi2 >= 10000)


@categorizer(uses={"FitChi2"})
def cat_fit_conv(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """Select events where kinematic fit converged (FitChi2 < 10000)."""
    return events, (events.FitChi2 < 10000)

@categorizer(uses={"FitChi2", "FitRbb"})
def cat_fit_conv_leq_rbb(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged and is below chi2 cut (bad events)
    chi2cut = self.config_inst.x.fitchi2cut
    return events, (events.FitChi2 < 10000) & (events.FitChi2 <= chi2cut) & (events.FitRbb > 2.0)


@categorizer(uses={"FitChi2", "FitRbb"})
def cat_rbb(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged and is below chi2 cut (bad events)
    return events, (events.FitRbb > 2.0)

# ============================================================================
# Signal and background region categorizers
# ============================================================================

@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*"})
def cat_2btj_sig(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Signal region: >= 2 b-tagged jets + signal trigger + good fit quality.
    Requires: signal trigger fired, FitChi2 <= config threshold, >= 2 b-tags (tight WP).
    """
    chi2cut = self.config_inst.x.fitchi2cut
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    signal_trigger = self.config_inst.x.trigger["tt_fh"][0]
    return events, (events.HLT[signal_trigger] &
                    (events.FitChi2 <= chi2cut) &
                    (ak.sum(
                        (events.Jet.pt >= 40.0) &
                        (abs(events.Jet.eta) < 2.4) &
                        (events.Jet.btagDeepFlavB >= wp_tight), axis=1,
                    ) >= 2))


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*", "FitChi2"})
def cat_0btj_bkg(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Background region: 0 b-tagged jets + background trigger + good fit quality.
    Requires: background trigger fired, FitChi2 <= config threshold, 0 b-tags (loose WP veto).
    """
    chi2cut = self.config_inst.x.fitchi2cut
    wp_loose = self.config_inst.x.btag_working_points.deepjet.loose
    bkg_trigger = self.config_inst.x.bkg_trigger["tt_fh"][0]
    return events, (events.HLT[bkg_trigger] &
                    (events.FitChi2 <= chi2cut) &
                    (ak.sum(
                        (events.Jet.pt >= 40.0) &
                        (abs(events.Jet.eta) < 2.4) &
                        (events.Jet.btagDeepFlavB >= wp_loose), axis=1,
                    ) == 0))


# ============================================================================
# Truth-level matching categorizers
# ============================================================================

@categorizer(uses={"fitCombinationType"})
def cat_fit_matched(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events with correct jet-parton matching (fitCombinationType == 2).
    For MC validation: kinematic fit selected the correct jet combination.
    """
    return events, (events.fitCombinationType == 2)


@categorizer(uses={"fitCombinationType"})
def cat_fit_unmatched(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Select events with incorrect or unmatched jet-parton assignment (fitCombinationType != 2).
    For MC validation: kinematic fit selected wrong jet combination or no match found.
    """
    return events, (events.fitCombinationType != 2)

# @categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*", "deltaRb", "chi2"})
# def cat_reco_sig(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     # two or more b-jets
#     chi2cut = self.config_inst.x.fitchi2cut
#     wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
#     signal_trigger = self.config_inst.x.trigger["tt_fh"][0]
#     return events, (events.HLT[signal_trigger] &
#                     (events.deltaRb > 2.0) &
#                     (events.chi2 <= chi2cut) &
#                     (ak.sum(
#                         (events.Jet.pt >= 40.0) &
#                         (abs(events.Jet.eta) < 2.4) &
#                         (events.Jet.btagDeepFlavB >= wp_tight), axis=1,
#                     ) >= 2))
