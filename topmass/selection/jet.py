# coding: utf-8

"""
Jet selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "nJet",
        "Jet.pt",
        "Jet.phi",
        "Jet.eta",
        "Jet.jetId",
        "Jet.puId",
        "Jet.btagDeepFlavB",
    },
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Dummy jet selection.
    """
    is_2016 = self.config_inst.campaign.x.year == 2016

    # common ak4 jet mask for normal and vbf jets
    default_mask = (
        (events.Jet.pt > 30.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.jetId == 6) &
        (  # tight plus lepton veto
            (events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4))
        ) &
        ak.all(events.Jet.metric_table(lepton_results.x.lepton_pair) > 0.5, axis=2)
    )

    # pt sorted indices to convert mask
    indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    jet_indices = indices[default_mask]
    jet_sel = ak.sum(default_mask, axis=1) >= 2

    # b-tagged jets, tight working point
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (default_mask) & (events.Jet.btagDeepFlavB >= wp_tight)
    bjet_indices = indices[bjet_mask][:, :2]
    bjet_sel = (ak.sum(bjet_mask, axis=1) >= 2) & (
        ak.sum(default_mask[:, :2], axis=1) == ak.sum(bjet_mask[:, :2], axis=1)
    )

    veto_mask = (default_mask) & (events.Jet.btagDeepFlavB < wp_tight)
    veto_indices = indices[veto_mask]
    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"jet": jet_sel, "bjet": bjet_sel},
        objects={
            "Jet": {"Jet": jet_indices, "Bjet": bjet_indices, "VetoBjet": veto_indices},
        },
        aux={
            # jet mask that lead to the jet_indices
            # "jet_mask": default_mask,  TODO: needed?
            # used to determine sum of weights in increment_stats
            "n_central_jets": ak.num(jet_indices, axis=1),
            "jet_mask": default_mask,
        },
    )
