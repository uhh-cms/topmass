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
from columnflow.production.cms.gen_top_decay import gen_top_decay_products
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from alljets.selection.jet import jet_selection
from alljets.selection.jet import kinFit

from alljets.production.example import cutflow_features


np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")

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
<<<<<<< HEAD
        kinFit,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids, kinFit,
=======
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        attach_coffea_behavior,
        gen_top_decay_products,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids,
        jet_selection,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        gen_top_decay_products,
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
>>>>>>> refs/remotes/origin/master
    },
    exposed=True,
)
def example(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # ensure trigger columns
    if "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2" not in ak.fields(events.HLT):
        events = set_ak_column(events, "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", False)
        results += SelectionResult(steps={"missing_whatever": events.HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2})
    else:
        results += SelectionResult(steps={"missing_whatever": np.ones(len(events), dtype=bool)})

    # muon selection
    events, muon_results = self[muon_selection](events, **kwargs)
    results += muon_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.event = (results.steps.muon & results.steps.jet &
                    results.steps.Trigger & results.steps.BTag &
                    results.steps.HT & results.steps.Chi2 &
                    results.steps.SixJets &
                    results.steps.BaseTrigger)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # Produce gen_top_decay
    if self.dataset_inst.name.startswith("tt_"):
        events = self[gen_top_decay_products](events, **kwargs)
    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

        # create process ids
        events = self[process_ids](events, **kwargs)

        # pdf weights
        events = self[pdf_weights](events, **kwargs)

        # renormalization/factorization scale weights
        events = self[murmuf_weights](events, **kwargs)

        # pileup weights
        events = self[pu_weight](events, **kwargs)

        # btag weights
        jet_mask = ((events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4))
        events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)

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


@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight, cutflow_features, process_ids,
        muon_selection,
        jet_selection,
        increment_stats,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        attach_coffea_behavior,
        gen_top_decay_products,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids,
        jet_selection,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        gen_top_decay_products,
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
    },
    exposed=True,
)
def trigger_eff(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # ensure trigger column
    if "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2" not in ak.fields(events.HLT):
        events = set_ak_column(events, "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", False)
        results += SelectionResult(steps={"missing_whatever": events.HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2})
    else:
        results += SelectionResult(steps={"missing_whatever": np.ones(len(events), dtype=bool)})

    # muon selection
    events, muon_results = self[muon_selection](events, **kwargs)
    results += muon_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.event = (
        results.steps.All &
        results.steps.BaseTrigger &
        results.steps.BTag
    )
    # results.steps.SixJets &
    # results.steps.jet &
    # results.steps.HT

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

        # create process ids
        events = self[process_ids](events, **kwargs)

        # pdf weights
        events = self[pdf_weights](events, **kwargs)

        # renormalization/factorization scale weights
        events = self[murmuf_weights](events, **kwargs)

        # pileup weights
        events = self[pu_weight](events, **kwargs)

        # btag weights
        jet_mask = ((events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4))
        events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)

        if self.dataset_inst.has_tag("has_top"):
            events = self[gen_top_decay_products](events, **kwargs)

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


@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight, cutflow_features, process_ids,
        muon_selection,
        jet_selection,
        increment_stats,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        attach_coffea_behavior,
        gen_top_decay_products,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids,
        jet_selection,
        pdf_weights,
        murmuf_weights,
        pu_weight,
        btag_weights,
        gen_top_decay_products,
        "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
    },
    exposed=True,
)
def trigger_eff2(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # ensure trigger column
    if "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2" not in ak.fields(events.HLT):
        events = set_ak_column(events, "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", False)
        results += SelectionResult(steps={"missing_whatever": events.HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2})
    else:
        results += SelectionResult(steps={"missing_whatever": np.ones(len(events), dtype=bool)})

    # muon selection
    events, muon_results = self[muon_selection](events, **kwargs)
    results += muon_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # combined event selection after all steps
    results.event = (
        # results.steps.SixJets &
        # results.steps.jet
        results.steps.BTag &
        results.steps.HT)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

        # create process ids
        events = self[process_ids](events, **kwargs)

        # pdf weights
        events = self[pdf_weights](events, **kwargs)

        # renormalization/factorization scale weights
        events = self[murmuf_weights](events, **kwargs)

        # pileup weights
        events = self[pu_weight](events, **kwargs)

        # btag weights
        jet_mask = ((events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4))
        events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)

        if self.dataset_inst.has_tag("has_top"):
            events = self[gen_top_decay_products](events, **kwargs)

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
