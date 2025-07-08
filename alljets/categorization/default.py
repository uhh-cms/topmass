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


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*"})
def cat_2btj_sig(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more b-jets
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    signal_trigger = self.config_inst.x.trigger["tt_fh"][0]
    return events, (events.HLT[signal_trigger] & (ak.sum(
        (events.Jet.pt >= 40.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_tight), axis=1) >= 2))


@categorizer(uses={"Jet.pt", "Jet.btagDeepFlavB", "Jet.eta", "HLT.*"})
def cat_0btj_bkg(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # zero b-jets, rejection with very loose working point
    wp_loose = self.config_inst.x.btag_working_points.deepjet.loose
    # wp_loose = 0.01
    bkg_trigger = self.config_inst.x.bkg_trigger["tt_fh"][0]
    return events, (events.HLT[bkg_trigger] & (ak.sum(
        (events.Jet.pt >= 40.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.btagDeepFlavB >= wp_loose), axis=1) == 0))
