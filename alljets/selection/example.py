# coding: utf-8

"""
Exemplary selection methods.
"""

from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import

from alljets.production.example import cutflow_features


np = maybe_import("numpy")
ak = maybe_import("awkward")


#
# other unexposed selectors
# (not selectable from the command line but used by other, exposed selectors)
#


@selector(
    uses={"Muon.pt", "Muon.eta"},
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # example muon selection: exactly one muon
    muon_mask = (events.Muon.pt >= 20.0) & (abs(events.Muon.eta) < 2.1)
    muon_sel = ak.sum(muon_mask, axis=1) >= 0

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Muon": {"MySelectedMuon": indices_applied_to_Muon}}
    return events, SelectionResult(
        steps={
            "muon": muon_sel,
        },
        objects={
            "Muon": {
                "Muon": muon_mask,
            },
        },
    )


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB", "Jet.jetId", "Jet.puId"},
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # example jet selection: at least six jets, lowest jet at least 40 GeV and H_T > 450 GeV
    jet_mask = ((events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4) & (ak.sum(events.Jet.pt, axis=1) >= 450))
    jet_sel = ak.sum(jet_mask, axis=1) >= 6

    # pt sorted indices
    # indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    # jet_indices = indices[jet_mask]
    # b-tagged jets (tight wp)
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_tight)
    # bjet_indices = indices[bjet_mask][:, :2]
    bjet_sel = (ak.sum(bjet_mask, axis=1) >= 2) & (ak.sum(jet_mask[:, :2], axis=1) == ak.sum(bjet_mask[:, :2], axis=1))
    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Jet": {"MyCustomJetCollection": indices_applied_to_Jet}}
    return events, SelectionResult(
        steps={
            "jet": jet_sel,
            "bjet": bjet_sel,
        },
        objects={
            "Jet": {
                "Jet": sorted_indices_from_mask(jet_mask, events.Jet.pt, ascending=False),
                "Bjet": sorted_indices_from_mask(bjet_mask, events.Jet.pt, ascending=False),
            },
        },
        aux={
            "n_jets": ak.sum(jet_mask, axis=1),
            "n_bjets": ak.sum(bjet_mask, axis=1),
        },
    )


#
# exposed selectors
# (those that can be invoked from the command line)
#

@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight, cutflow_features, process_ids, muon_selection, jet_selection,
        increment_stats,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids,
    },
    exposed=True,
)
def example(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # muon selection
    events, muon_results = self[muon_selection](events, **kwargs)
    results += muon_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.main["event"] = results.steps.muon & results.steps.jet

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": results.main.event,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.main.event),
        }
        group_map = {
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
            # per jet multiplicity
            "njet": {
                "values": results.x.n_jets,
                "mask_fn": (lambda v: results.x.n_jets == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events, results
