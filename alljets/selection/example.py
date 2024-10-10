# coding: utf-8

"""
Exemplary selection methods.
"""

from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.btag import btag_weights
from columnflow.util import maybe_import
from alljets.selection.jet import jet_selection

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


#
# exposed selectors
# (those that can be invoked from the command line)
#


@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight, cutflow_features, process_ids,
        muon_selection,
        jet_selection,
        increment_stats,
        kinFit,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids, kinFit,
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
    results.event = results.steps.muon & results.steps.jet

    # create process ids
    events = self[process_ids](events, **kwargs)
    
    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    #     # create process ids
    #     events = self[process_ids](events, **kwargs)

    #     # pdf weights
    #     events = self[pdf_weights](events, **kwargs)

    #     # renormalization/factorization scale weights
    #     events = self[murmuf_weights](events, **kwargs)

    #     # pileup weights
    #     events = self[pu_weight](events, **kwargs)

    #     # btag weights
    #     events = self[btag_weights](events, results.x.jet_mask, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": results.event,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.event),
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
