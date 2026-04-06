# coding: utf-8

"""
Selection methods for the top mass analysis.

The core concept is the `Selector` class, which represents a configurable
selection pipeline. Each selector function is decorated with `@selector`,
which registers its dependencies and outputs. Selectors can be composed,
allowing modular and reusable analysis code.

Selectors in this file:
- muon_selection: Selects events with muons passing basic kinematic cuts.
- default_trig_weight: Like `example`, but also applies trigger weights.

Each selector returns a tuple of (events, SelectionResult), where
SelectionResult contains selection masks and object indices for downstream use.
"""

from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats
from columnflow.production.processes import process_ids
from columnflow.production.categories import category_ids
from columnflow.production.util import attach_coffea_behavior
from columnflow.selection import SelectionResult, Selector, selector

from columnflow.production.cms.pdf import pdf_weights
from columnflow.selection.cms.jets import jet_veto_map
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.btag import btag_weights
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.parton_shower import ps_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.gen_particles import gen_top_lookup

from alljets.selection.jet import jet_selection
from alljets.production.default import cutflow_features
from alljets.production.trig_cor_weight import trig_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")


@selector(
    uses={"Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass"},
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Selects muons passing basic kinematic cuts.

    This selector applies a simple muon selection: at least one muon with
    pt >= 20 GeV and |eta| < 2.1. It returns a mask for selected muons and
    a step mask for events passing the selection.

    Returns:
        events: The unchanged events array.
        SelectionResult: Contains the muon selection mask and step.
    """

    # example muon selection: at least one muon with pt and eta cuts
    muon_mask = (events.Muon.pt >= 20.0) & (abs(events.Muon.eta) < 2.1)
    muon_sel = ak.sum(muon_mask, axis=1) >= 0

    # build and return selection results
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


# Extract the category ids for the inclusve category, used for CutFlow
incl_category_ids = category_ids.derive("incl_category_ids",
                                        cls_dict={"skip_category": lambda self, cat: cat.name != "incl"},
                                        )


@selector(
    uses={
        cutflow_features,
        muon_selection,
        jet_veto_map,
        jet_selection,
        attach_coffea_behavior,
        gen_top_lookup,
        process_ids,
        increment_stats,
        deterministic_seeds,
        incl_category_ids,
        mc_weight,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        trig_weights,
        ps_weights,
    },
    produces={
        cutflow_features,
        jet_veto_map,
        jet_selection,
        gen_top_lookup,
        process_ids,
        deterministic_seeds,
        incl_category_ids,
        mc_weight,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        trig_weights,
        ps_weights,
        "gen_top.*.{eta,phi,pt,mass,pdgId}",
        "gen_top",
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
    },
    exposed=True,
)
def default_trig_weight(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Event selection pipeline with trigger weights.

    This selector is the main for the analysis and applies trigger weights
    for MC events. It ensures all relevant columns exist, applies muon and
    jet selections, and computes weights for MC, including trigger weights.

    Returns:
        events: The events array with new columns added.
        SelectionResult: Contains selection masks and object indices.

    Important Note: DO NOT use the returned selector step from the jet_veto_map selector
              for the jet selection. This selector removes the event if one jet lies in the veto region.
              Instead, we use the Jet.veto_map_mask in the jet selection to remove individual jets.
    """

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # Produce gen_top_decay if available
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_lookup](events, **kwargs)
    else:
        events = set_ak_column(events, "gen_top", False)
        events = set_ak_column(events, "GenPart.eta", False)

    # ensure trigger columns exist
    if "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2" not in ak.fields(events.HLT):
        events = set_ak_column(
            events, "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", False,
        )

    # muon selection
    events, muon_results = self[muon_selection](events, **kwargs)
    results += muon_results

    # jet veto map
    events, veto_result = self[jet_veto_map](events, **kwargs)
    results += veto_result

    # jet selection, using the jet veto map mask and jet Id criteria
    events, jet_results = self[jet_selection](events, mode="analysis", **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.event = (
        results.steps.Trigger &
        results.steps.HT &
        results.steps.jet &
        results.steps.BTag &
        results.steps.LeadingSix2BTag
    )

    # create process ids, deterministic seeds, and inclusive category ids for cutflow
    events = self[process_ids](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[incl_category_ids](events, **kwargs)

    # add the mc weight and other weights for MC datasets
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[process_ids](events, **kwargs)
        events = self[pdf_weights](events, **kwargs)
        events = self[murmuf_weights](events, **kwargs)
        events = self[pu_weight](events, **kwargs)
        jet_mask = (events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4)
        events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)
        events = self[ps_weights](events, **kwargs)
        events = self[trig_weights](events, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": results.event,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        for route in sorted(self[btag_weights].produced_columns):
            weight_name = str(route)
            if not weight_name.startswith(btag_weights.weight_name):
                continue

            if self.dataset_inst.is_mc:
                weight_map = {
                    **weight_map,
                    # mc weight for all events
                    "sum_mc_weight": (events.mc_weight, Ellipsis),
                    "sum_mc_weight_selected": (events.mc_weight, results.event),
                    # TODO: Add variations for shifts
                    "sum_mc_weight_pu_weight": (events.mc_weight * events.pu_weight, Ellipsis),
                    "sum_btag_weight": (events[weight_name], Ellipsis),
                    "sum_btag_weight_selected": (events[weight_name], results.event),
                    "sum_trig_weight": (events.trig_weight, Ellipsis),
                    "sum_trig_weight_selected": (events.trig_weight, results.event),
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


@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight,
        cutflow_features,
        process_ids,
        jet_selection,
        increment_stats,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        attach_coffea_behavior,
        gen_top_lookup,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight,
        cutflow_features,
        process_ids,
        jet_selection,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        gen_top_lookup,
        "gen_top.*.{eta,phi,pt,mass,pdgId}",
        "gen_top",
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
    },
    exposed=True,
)
def no_btag(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Selector for the b-tagging efficiency in the simulation using the pre-selection.
    Essentially the same as default selector but without b-tagging step.

    Returns:
        events: The events array with new columns added.
        SelectionResult: Contains selection masks and object indices.

    """

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # Produce gen_top_decay if available
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_lookup](events, **kwargs)
    else:
        events = set_ak_column(events, "gen_top", False)
        events = set_ak_column(events, "GenPart.eta", False)

    # ensure trigger columns exist
    if "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2" not in ak.fields(events.HLT):
        events = set_ak_column(
            events, "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", False,
        )

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.event = (
        results.steps.jet &
        results.steps.Trigger &
        results.steps.HT
    )

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add the mc weight and other weights for MC datasets
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[process_ids](events, **kwargs)
        events = self[pdf_weights](events, **kwargs)
        events = self[murmuf_weights](events, **kwargs)
        events = self[pu_weight](events, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)

    # increment stats for bookkeeping and monitoring
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": results.event,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
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
