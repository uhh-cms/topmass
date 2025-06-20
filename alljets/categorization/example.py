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
    # zero b-jets
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    return events, (ak.sum((
        events.Jet.pt >= 40.0) &
        abs(events.Jet.eta < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) == 0
    )


@categorizer(uses={"FitChi2"})
def cat_fit_conv_big(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged and is above chi2 cut (bad events)
    chi2cut = self.config_inst.x.fitchi2cut
    return events, (events.FitChi2 < 10000) & (events.FitChi2 > chi2cut)


@categorizer(uses={"FitChi2"})
def cat_fit_conv_leq(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has converged and is below chi2 cut (bad events)
    chi2cut = self.config_inst.x.fitchi2cut
    return events, (events.FitChi2 < 10000) & (events.FitChi2 <= chi2cut)


@categorizer(uses={"FitChi2"})
def cat_fit_nconv(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # kinematic fit has not converged
    return events, (events.FitChi2 >= 10000)
