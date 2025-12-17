# coding: utf-8

"""
Exemplary selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")

#
# categorizer functions used by categories definitions
#


@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"Jet.pt"})
def cat_6j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # six jets
    return events, ak.num((events.Jet.pt >= 40.0), axis=1) == 6


@categorizer(uses={"Jet.pt"})
def cat_7j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # seven or more jets
    return events, ak.num((events.Jet.pt >= 40.0), axis=1) >= 7


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta"})
def cat_2btj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more b-jets
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum(
        (events.Jet.pt >= 40.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) >= 2
    )


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta"})
def cat_1btj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # one b-jet
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum(
        (events.Jet.pt >= 40.0) &
        abs(events.Jet.eta < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) == 1
    )


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta"})
def cat_0btj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # zero b-jets, rejection with very loose working point
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum((
        events.Jet.pt >= 40.0) &
        abs(events.Jet.eta < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) == 0
    )


@categorizer(uses={"FitPgof","FitChi2"})
def cat_fit_conv_leq(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged and is below pgof cut (bad events)
    pgofcut = self.config_inst.x.fitpgofcut
    return events, (events.FitChi2 < 10000) & (events.FitPgof <= pgofcut)


@categorizer(uses={"FitPgof","FitChi2"})
def cat_fit_conv_big(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged and is above pgof cut (good events)
    pgofcut = self.config_inst.x.fitpgofcut
    return events, (events.FitChi2 < 10000) & (events.FitPgof > pgofcut)


# @categorizer(uses={"FitChi2", "FitRbb"})
# def cat_fit_conv_leq_rbb(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     # kinematic fit has converged and is below chi2 cut (bad events)
#     chi2cut = self.config_inst.x.fitchi2cut
#     return events, (events.FitChi2 < 10000) & (events.FitChi2 <= chi2cut) & (events.FitRbb > 2.0)


# @categorizer(uses={"FitChi2", "FitRbb"})
# def cat_rbb(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     # kinematic fit has converged and is below chi2 cut (bad events)
#     return events, (events.FitRbb > 2.0)


@categorizer(uses={"FitChi2"})
def cat_fit_nconv(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has not converged
    return events, (events.FitChi2 >= 10000)


@categorizer(uses={"FitChi2"})
def cat_fit_conv(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged
    return events, (events.FitChi2 < 10000)


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*", "FitChi2"})
def cat_2btj_sig(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more b-jets
    chi2cut = self.config_inst.x.fitchi2cut
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    signal_trigger = self.config_inst.x.trigger["tt_fh"][0]
    return events, (events.HLT[signal_trigger] &
                    # (events.FitRbb > 2.0) &
                    (events.FitChi2 <= chi2cut) &
                    (ak.sum(
                        (events.Jet.pt >= 40.0) &
                        (abs(events.Jet.eta) < 2.4) &
                        (events.Jet.btagDeepFlavB >= wp_tight), axis=1,
                    ) >= 2))


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*", "FitChi2"})
def cat_0btj_bkg(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # zero b-jets, rejection with very loose working point
    chi2cut = self.config_inst.x.fitchi2cut
    wp_loose = self.config_inst.x.btag_working_points.deepjet.loose
    # wp_loose = 0.01
    bkg_trigger = self.config_inst.x.bkg_trigger["tt_fh"][0]
    return events, (events.HLT[bkg_trigger] &
                    # (events.FitRbb > 2.0) &
                    (events.FitChi2 <= chi2cut) &
                    (ak.sum(
                        (events.Jet.pt >= 40.0) &
                        (abs(events.Jet.eta) < 2.4) &
                        (events.Jet.btagDeepFlavB >= wp_loose), axis=1,
                    ) == 0))

@categorizer(uses={"fitCombinationType"})
def cat_fit_matched(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has not converged
    return events, (events.fitCombinationType == 2)

@categorizer(uses={"fitCombinationType"})
def cat_fit_unmatched(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has not converged
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
