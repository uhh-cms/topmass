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
    return events, (events.HLT[signal_trigger] & (events.FitChi2 <= chi2cut) & (ak.sum(
        (events.Jet.pt >= 40.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) >= 2))


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*"})
def cat_0btj_bkg(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    """
    Background region: 0 b-tagged jets + background trigger + good fit quality.
    Requires: background trigger fired, FitChi2 <= config threshold, 0 b-tags (loose WP veto).
    """
    chi2cut = self.config_inst.x.fitchi2cut
    wp_loose = self.config_inst.x.btag_working_points.deepjet.loose
    # wp_loose = 0.01
    bkg_trigger = self.config_inst.x.bkg_trigger["tt_fh"][0]
    return events, (events.HLT[bkg_trigger] & (events.FitChi2 <= chi2cut) & (ak.sum(
        (events.Jet.pt >= 40.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_loose), axis=1) == 0))


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

# ============================================================================
# Cuts on gen-level
# ============================================================================


@categorizer(uses={"gen_top.*"})
def gen_eta21_pt60(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1)], axis=0)
    return events, ak.all([eta, pt], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_corr(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    matching = events.fitCombinationType == 2
    return events, ak.all([eta, pt, matching], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_deltaRmin08_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = events.dRmin_gen_t1 < 0.8
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_deltaRmin06_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = events.dRmin_gen_t1 < 0.6
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_deltaRmin05_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = events.dRmin_gen_t1 < 0.5
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_deltaRmin06_08_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = (events.dRmin_gen_t1 > 0.6) & (events.dRmin_gen_t1 < 0.8)
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_deltaRmin04_06_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = (events.dRmin_gen_t1 > 0.4) & (events.dRmin_gen_t1 < 0.6)
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   })
def gen_eta21_pt60_deltaRmin08_inf_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = (events.dRmin_gen_t1 > 0.8)
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   "dRmin_gen_t1",
                   })
def gen_cut_deltaRmin08_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    matching = events.fitCombinationType == 2
    dR = events.dRmin_gen_t1 < 0.8
    return events, ak.all([eta, pt, matching, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   "dRmin_gen_t1",
                   })
def gen_cut_deltaRmin06_t1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    matching = events.fitCombinationType == 2
    dR = events.dRmin_gen_t1 < 0.6
    return events, ak.all([eta, pt, matching, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   "dRmin_gen_t1",
                   })
def gen_cut_deltaRmin06_t1_false(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    matching = events.fitCombinationType > 0
    dR = events.dRmin_gen_t1 < 0.6
    return events, ak.all([eta, pt, matching, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   "dRmin_gen_t1",
                   "n_deltaR06_reco_q1",
                   })
def gen_cut_deltaRmin06_q1(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    matching = events.fitCombinationType == 2
    dR = events.n_deltaR06_reco_q1 >= 2
    return events, ak.all([eta, pt, matching, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   "dRmin_gen_t1",
                   "n_deltaR06_reco_q1",
                   })
def gen_cut_deltaR06_q1_without_matching(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = events.n_deltaR06_reco_q1 >= 2
    return events, ak.all([eta, pt, dR], axis=0)


@categorizer(uses={"gen_top.*",
                   "fitCombinationType",
                   "dRmin_gen_t1",
                   "n_deltaR06_reco_q1",
                   })
def gen_cut_deltaR0406_q1_without_matching(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    pt_cut = 60
    eta = ak.all([
        ak.all(abs(events.gen_top.b.eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 0].eta) < 2.1, axis=1),
        ak.all(abs(events.gen_top.w_children[:, :, 1].eta) < 2.1, axis=1)], axis=0)
    pt = ak.all([
        ak.all(events.gen_top.b.pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 0].pt > pt_cut, axis=1),
        ak.all(events.gen_top.w_children[:, :, 1].pt > pt_cut, axis=1),
    ], axis=0)
    dR = events.n_deltaR04_06_reco_q1 >= 1
    return events, ak.all([eta, pt, dR], axis=0)
