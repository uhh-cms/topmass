# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.production.cms.pdf import pdf_weights
from columnflow.production.cms.btag import btag_weights
from columnflow.production.cms.scale import murmuf_weights
from columnflow.production.cms.pileup import pu_weight
from columnflow.util import maybe_import, safe_div
from columnflow.columnar_util import set_ak_column

import functools

ak = maybe_import("awkward")
np = maybe_import("numpy")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        pu_weight.PRODUCES,
        # custom columns created upstream, probably by a producer
        "process_id",
    },
    # only run on mc
    mc_only=True,
)
def normalized_pu_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self[pu_weight].produces:
        if not weight_name.startswith("pu_weight"):
            continue

        # create a weight vector starting with ones
        norm_weight_per_pid = np.ones(len(events), dtype=np.float32)

        # fill weights with a new mask per unique process id (mostly just one)
        for pid in self.unique_process_ids:
            pid_mask = events.process_id == pid
            norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]

        # multiply with actual weight
        norm_weight_per_pid = norm_weight_per_pid * events[weight_name]

        # store it
        norm_weight_per_pid = ak.values_astype(norm_weight_per_pid, np.float32)
        events = set_ak_column(events, f"normalized_{weight_name}", norm_weight_per_pid, value_type=np.float32)

    return events


@normalized_pu_weight.init
def normalized_pu_weight_init(self: Producer) -> None:
    self.produces |= {
        f"normalized_{weight_name}"
        for weight_name in self[pu_weight].produces
        if weight_name.startswith("pu_weight")
    }


@normalized_pu_weight.requires
def normalized_pu_weight_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_pu_weight.setup
def normalized_pu_weight_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    # load the selection stats
    stats = inputs["selection_stats"]["collection"][0].load(formatter="json")

    # get the unique process ids in that dataset
    key = "sum_mc_weight_pu_weight_per_process"
    self.unique_process_ids = list(map(int, stats[key].keys()))

    # helper to get numerators and denominators
    def numerator_per_pid(pid):
        key = "sum_mc_weight_per_process"
        return stats[key].get(str(pid), 0.0)

    def denominator_per_pid(weight_name, pid):
        key = f"sum_mc_weight_{weight_name}_per_process"
        return stats[key].get(str(pid), 0.0)

    # extract the ratio per weight and pid
    self.ratio_per_pid = {
        weight_name: {
            pid: safe_div(numerator_per_pid(pid), denominator_per_pid(weight_name, pid))
            for pid in self.unique_process_ids
        }
        for weight_name in self[pu_weight].produces
        if weight_name.startswith("pu_weight")
    }


@producer(
    uses={
        "pdf_weight", "pdf_weight_up", "pdf_weight_down",
    },
    produces={
        "normalized_pdf_weight", "normalized_pdf_weight_up", "normalized_pdf_weight_down",
    },
    # only run on mc
    mc_only=True,
)
def normalized_pdf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for postfix in ["", "_up", "_down"]:
        # create the normalized weight
        avg = self.average_pdf_weights[postfix]

        normalized_weight = events[f"pdf_weight{postfix}"] / avg

        # store it
        events = set_ak_column(events, f"normalized_pdf_weight{postfix}", normalized_weight, value_type=np.float32)

    return events


@normalized_pdf_weight.requires
def normalized_pdf_weight_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_pdf_weight.setup
def normalized_pdf_weight_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    # load the selection stats
    stats = inputs["selection_stats"]["collection"][0].load(formatter="json")

    # save average weights
    self.average_pdf_weights = {
        postfix: safe_div(stats[f"sum_pdf_weight{postfix}"], stats["n_events"])
        for postfix in ["", "_up", "_down"]
    }


@producer(
    uses={
        "murmuf_weight", "murmuf_weight_up", "murmuf_weight_down",
    },
    produces={
        "normalized_murmuf_weight", "normalized_murmuf_weight_up", "normalized_murmuf_weight_down",
    },
    # only run on mc
    mc_only=True,
)
def normalized_murmuf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for postfix in ["", "_up", "_down"]:
        # create the normalized weight
        avg = self.average_murmuf_weights[postfix]
        normalized_weight = events[f"murmuf_weight{postfix}"] / avg

        # store it
        events = set_ak_column(events, f"normalized_murmuf_weight{postfix}", normalized_weight, value_type=np.float32)

    return events


@normalized_murmuf_weight.requires
def normalized_murmuf_weight_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_murmuf_weight.setup
def normalized_murmuf_weight_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    # load the selection stats
    stats = inputs["selection_stats"]["collection"][0].load(formatter="json")

    # save average weights
    self.average_murmuf_weights = {
        postfix: safe_div(stats[f"sum_murmuf_weight{postfix}"], stats["n_events"])
        for postfix in ["", "_up", "_down"]
    }


@producer(
    uses={
        btag_weights.PRODUCES,
        # custom columns created upstream, probably by a producer
        "process_id",
        # nano columns
        "Jet.pt",
    },
    # only run on mc
    mc_only=True,
)
def normalized_btag_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for weight_name in self[btag_weights].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        # create a weight vectors starting with ones for both weight variations, i.e.,
        # nomalization per pid and normalization per pid and jet multiplicity
        norm_weight_per_pid = np.ones(len(events), dtype=np.float32)
        norm_weight_per_pid_njet = np.ones(len(events), dtype=np.float32)

        # fill weights with a new mask per unique process id (mostly just one)
        for pid in self.unique_process_ids:
            pid_mask = events.process_id == pid
            # single value
            norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]
            # lookup table
            n_jets = ak.num(events[pid_mask].Jet.pt, axis=1)
            norm_weight_per_pid_njet[pid_mask] = self.ratio_per_pid_njet[weight_name][pid][n_jets]

        # multiply with actual weight
        norm_weight_per_pid = norm_weight_per_pid * events[weight_name]
        norm_weight_per_pid_njet = norm_weight_per_pid_njet * events[weight_name]

        # store them
        events = set_ak_column_f32(events, f"normalized_{weight_name}", norm_weight_per_pid)
        events = set_ak_column_f32(events, f"normalized_njet_{weight_name}", norm_weight_per_pid_njet)

    # import IPython
    # IPython.embed()

    return events


@normalized_btag_weights.init
def normalized_btag_weights_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return

    for weight_name in self[btag_weights].produces:
        if not weight_name.startswith("btag_weight"):
            continue

        self.produces |= {f"normalized_{weight_name}", f"normalized_njet_{weight_name}"}


@normalized_btag_weights.requires
def normalized_btag_weights_requires(self: Producer, reqs: dict) -> None:
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        self.task,
        tree_index=0,
        branch=-1,
        _exclude=MergeSelectionStats.exclude_params_forest_merge,
    )


@normalized_btag_weights.setup
def normalized_btag_weights_setup(self: Producer, reqs: dict, inputs: dict) -> None:
    # load the selection stats
    stats = inputs["selection_stats"]["collection"][0].load(formatter="json")

    # get the unique process ids in that dataset
    key = "sum_mc_weight_selected_no_bjet_per_process_and_njet"
    self.unique_process_ids = list(map(int, stats[key].keys()))

    # get the maximum numbers of jets
    max_n_jets = max(map(int, sum((list(d.keys()) for d in stats[key].values()), [])))

    # helper to get numerators and denominators
    def numerator_per_pid(pid):
        key = "sum_mc_weight_selected_no_bjet_per_process"
        return stats[key].get(str(pid), 0.0)

    def denominator_per_pid(weight_name, pid):
        key = f"sum_mc_weight_{weight_name}_selected_no_bjet_per_process"
        return stats[key].get(str(pid), 0.0)

    def numerator_per_pid_njet(pid, n_jets):
        key = "sum_mc_weight_selected_no_bjet_per_process_and_njet"
        d = stats[key].get(str(pid), {})
        return d.get(str(n_jets), 0.0)

    def denominator_per_pid_njet(weight_name, pid, n_jets):
        key = f"sum_mc_weight_{weight_name}_selected_no_bjet_per_process_and_njet"
        d = stats[key].get(str(pid), {})
        return d.get(str(n_jets), 0.0)

    # extract the ratio per weight and pid
    self.ratio_per_pid = {
        weight_name: {
            pid: safe_div(numerator_per_pid(pid), denominator_per_pid(weight_name, pid))
            for pid in self.unique_process_ids
        }
        for weight_name in self[btag_weights].produces
        if weight_name.startswith("btag_weight")
    }

    # extract the ratio per weight, pid and also the jet multiplicity, using the latter as in index
    # for a lookup table (since it naturally starts at 0)
    self.ratio_per_pid_njet = {
        weight_name: {
            pid: np.array([
                safe_div(numerator_per_pid_njet(pid, n_jets), denominator_per_pid_njet(weight_name, pid, n_jets))
                for n_jets in range(max_n_jets + 1)
            ])
            for pid in self.unique_process_ids
        }
        for weight_name in self[btag_weights].produces
        if weight_name.startswith("btag_weight")
    }


@producer(
    uses={
        normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        electron_weights, muon_weights,
    },
    produces={
        normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        electron_weights, muon_weights,
    },
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers.
    """
    # normalization weights
    events = self[normalization_weights](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalized pdf weight
        events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # btag weights
        events = self[normalized_btag_weights](events, **kwargs)

        # electron weights
        events = self[electron_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

    return events
