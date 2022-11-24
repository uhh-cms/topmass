# coding: utf-8

"""
Jet selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]

@selector(
    uses={
        "nJet", "Jet.pt", "Jet.eta", "Jet.jetId", "Jet.puId","Jet.btagDeepFlavB"
    },
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Dummy jet selection.
    """
    is_2016 = self.config_inst.campaign.x.year == 2016

    # common ak4 jet mask for normal and vbf jets
    default_mask = (
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.jetId == 6) &  # tight plus lepton veto
        ((events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4)))  # flipped in 2016
    )

    # pt sorted indices to convert mask
    #sorted_indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    jet_indices = masked_sorted_indices(default_mask,events.Jet.pt)

    # final event selection, just pick events with 2 or mor ejets
    jet_sel = ak.sum(default_mask, axis=1) >= 1

    # b-tagged jets, medium working point
    wp_tight = self.config_inst.x.btag_working_points.deepcsv.tight
    bjet_mask = (default_mask) & (events.Jet.btagDeepFlavB >= wp_tight)
    bjet_sel = ak.sum(bjet_mask, axis=1) >= 1

    # sort jets after b-score and define b-jets as the two b-score leading jets
    bjet_indices = masked_sorted_indices(bjet_mask, events.Jet.btagDeepFlavB)
    bjet_indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"jet": jet_sel, "Bjet": bjet_sel},
        
        objects={"Jet": {"Jet": jet_indices, "Bjet": bjet_indices},},
        
        aux={
            # jet mask that lead to the jet_indices
            # "jet_mask": default_mask,  TODO: needed?
            # used to determine sum of weights in increment_stats
            "n_central_jets": ak.num(jet_indices, axis=1),
        },
    )