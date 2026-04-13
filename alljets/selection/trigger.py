"""
Selection methods for the top mass analysis, specifically for trigger studies.

Selectors in this file:
- trigger_eff: Selector for computing trigger efficiency using trigger objects.
- trigger_eval: Selector for evaluating trigger efficiency on selected events.

Each selector returns a tuple of (events, SelectionResult), where
SelectionResult contains selection masks and object indices for downstream use.
"""

from collections import defaultdict

from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats
from columnflow.production.processes import process_ids
from columnflow.production.util import attach_coffea_behavior
from columnflow.selection import SelectionResult, Selector, selector

from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.pileup import pu_weight
from columnflow.selection.cms.jets import jet_veto_map
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.parton_shower import ps_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.gen_particles import gen_top_lookup
from columnflow.selection.cms.btag import fill_btag_wp_count_hists

from alljets.selection.jet import jet_selection
from alljets.selection.default import muon_selection
from alljets.production.default import cutflow_features
from alljets.production.trig_cor_weight import trig_weights


np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
hist = maybe_import("hist")


@selector(
    uses={
        attach_coffea_behavior,
        cutflow_features,
        muon_selection,
        jet_selection,
        jet_veto_map,
        process_ids,
        increment_stats,
        deterministic_seeds,
        fill_btag_wp_count_hists,
        gen_top_lookup,
        mc_weight,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        ps_weights,
        "TrigObj*",
    },
    produces={
        cutflow_features,
        jet_selection,
        jet_veto_map,
        process_ids,
        gen_top_lookup,
        fill_btag_wp_count_hists,
        mc_weight,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        ps_weights,
        "trig_weight",
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
        "trig_ht",
    },
    exposed=True,
)
def trigger_eff(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: DotDict[str, hist.Hist],
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Computes trigger efficiency using trigger objects.

    This selector calculates the HT of trigger objects, ensures trigger columns,
    applies jet selection, and builds a combined event selection mask. It is
    intended for studies of trigger efficiency.

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

    # Produce gen_top_decay
    if self.dataset_inst.has_tag("has_top"):
        events = self[gen_top_lookup](events, **kwargs)
    else:
        events = set_ak_column(events, "gen_top", False)
        events = set_ak_column(events, "GenPart.eta", False)

    # Calculate HT from trigger objects (jets with pt >= 32 GeV, |eta| <= 2.6, id == 1)
    trig_ht = ak.sum(events.TrigObj.pt[(events.TrigObj.pt >= 32) & (abs(events.TrigObj.eta) <= 2.6) &
                                       (events.TrigObj.id == 1)], axis=1)
    events = set_ak_column(events, "trig_ht", trig_ht)

    # ensure trigger columns
    if "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2" not in ak.fields(events.HLT):
        events = set_ak_column(
            events, "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", False,
        )

    # jet veto map
    events, veto_result = self[jet_veto_map](events, **kwargs)
    results += veto_result

    # jet selection
    events, jet_results = self[jet_selection](events, mode="trigger", **kwargs)
    results += jet_results

    # combined event selection after all steps: Choose one of the trigger efficiency selector steps
    results.event = (
        results.steps.All &
        results.steps.BaseTrigger &
        results.steps.BTag &
        results.steps.HT
    )

    # Combined event selection for efficiency calculation, without b-tagging requirements
    results.event_eff = (
        results.steps.All &
        results.steps.BaseTrigger &
        results.steps.HT
    )

    # create process ids and deterministic seeds
    events = self[process_ids](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)

    # Set default trigger weight to 1 (for data or MC without trigger weights)
    events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)

    # add the mc weight and other weights for MC datasets
    if self.dataset_inst.is_mc:
        events = set_ak_column(events, "mc_weight", np.ones(len(events)), value_type=np.float32)
        events = self[pdf_weights](events, **kwargs)
        events = self[murmuf_weights](events, **kwargs)
        events = self[pu_weight](events, **kwargs)
        events = self[ps_weights](events, **kwargs)
        jet_mask = (events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4)
        self[fill_btag_wp_count_hists](events, results.event_eff, jet_mask, hists, **kwargs)

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
        attach_coffea_behavior,
        cutflow_features,
        muon_selection,
        jet_selection,
        jet_veto_map,
        process_ids,
        increment_stats,
        deterministic_seeds,
        fill_btag_wp_count_hists,
        gen_top_lookup,
        mc_weight,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        ps_weights,
        trig_weights,
    },
    produces={
        cutflow_features,
        muon_selection,
        jet_selection,
        jet_veto_map,
        process_ids,
        gen_top_lookup,
        fill_btag_wp_count_hists,
        mc_weight,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        ps_weights,
        trig_weights,
        "gen_top.*.{eta,phi,pt,mass,pdgId}",
        "gen_top",
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
    },
    exposed=True,
)
def trigger_eval(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: DotDict[str, hist.Hist],
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Event selection pipeline with trigger weights.

    This selector is similar to `default_trig_weight`
    It ensures all relevant columns exist, applies muon and
    jet selections, and computes weights for MC, including trigger weights.

    This should be used for evaluating trigger efficiency on selected events.
    Specifically, on the pt of the Jet used for the trigger SF and the HT of the event.

    As we want to evaluate the trigger efficiency, we want to have Jets close to the trigger.

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
    events, jet_results = self[jet_selection](events, mode="trigger", **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.event = (
        results.steps.jet &
        results.steps.Trigger &
        results.steps.BTag &
        results.steps.HT
    )

    # Combined event selection for efficiency calculation, without b-tagging requirements
    results.event_eff = (
        results.steps.All &
        results.steps.BaseTrigger &
        results.steps.HT
    )

    # create process ids and deterministic seeds
    events = self[process_ids](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)

    # add the mc weight and other weights for MC datasets
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[process_ids](events, **kwargs)
        events = self[pdf_weights](events, **kwargs)
        events = self[murmuf_weights](events, **kwargs)
        events = self[pu_weight](events, **kwargs)
        events = self[ps_weights](events, **kwargs)
        events = self[trig_weights](events, **kwargs)
        jet_mask = (events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4)
        self[fill_btag_wp_count_hists](events, results.event_eff, jet_mask, hists, **kwargs)

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
            # TODO: Add variations for shifts
            "sum_mc_weight_pu_weight": (events.mc_weight * events.pu_weight, Ellipsis),
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
