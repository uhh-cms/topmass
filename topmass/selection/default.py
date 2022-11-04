# coding: utf-8

"""
Selection methods.
"""

from operator import and_
from functools import reduce
from collections import defaultdict, OrderedDict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.mc_weight import mc_weight
from columnflow.production.processes import process_ids
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import

from topmass.selection.met import met_filter_selection
from topmass.selection.jet import jet_selection
from topmass.production.features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector()
def increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # get event masks
    event_mask = results.main.event

    # increment plain counts
    stats["n_events"] += len(events)
    stats["n_events_selected"] += ak.sum(event_mask, axis=0)

    # get a list of unique jet multiplicities present in the chunk
    unique_process_ids = np.unique(events.process_id)
    unique_n_jets = []
    if results.has_aux("n_central_jets"):
        unique_n_jets = np.unique(results.x.n_central_jets)

    # create a map of entry names to (weight, mask) pairs that will be written to stats
    weight_map = OrderedDict()
    if self.dataset_inst.is_mc:
        # mc weight for all events
        weight_map["mc_weight"] = (events.mc_weight, Ellipsis)

        # mc weight for selected events
        weight_map["mc_weight_selected"] = (events.mc_weight, event_mask)

    # get and store the weights
    for name, (weights, mask) in weight_map.items():
        joinable_mask = True if mask is Ellipsis else mask

        # sum for all processes
        stats[f"sum_{name}"] += ak.sum(weights[mask])

        # sums per process id and again per jet multiplicity
        stats.setdefault(f"sum_{name}_per_process", defaultdict(float))
        stats.setdefault(f"sum_{name}_per_process_and_njet", defaultdict(lambda: defaultdict(float)))
        for p in unique_process_ids:
            stats[f"sum_{name}_per_process"][int(p)] += ak.sum(
                weights[(events.process_id == p) & joinable_mask],
            )
            for n in unique_n_jets:
                stats[f"sum_{name}_per_process_and_njet"][int(p)][int(n)] += ak.sum(
                    weights[
                        (events.process_id == p) &
                        (results.x.n_central_jets == n) &
                        joinable_mask
                    ],
                )

    return events


@selector(
    uses={
        attach_coffea_behavior, mc_weight, met_filter_selection, jet_selection, process_ids,
        cutflow_features, increment_stats,
    },
    produces={
        mc_weight, met_filter_selection, jet_selection, process_ids, cutflow_features,
        increment_stats,
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # add corrected mc weights
    events = self[mc_weight](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # met filter selection
    events, met_filter_results = self[met_filter_selection](events, **kwargs)
    results += met_filter_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    # create process ids
    events = self[process_ids](events, **kwargs)

    # increment stats
    events = self[increment_stats](events, results, stats, **kwargs)

    # some cutflow features
    events = self[cutflow_features](events, **kwargs)

    return events, results
