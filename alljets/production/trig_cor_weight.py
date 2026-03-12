# coding: utf-8

"""
Trigger related event weights.
"""

from __future__ import annotations

import law
from law.config import get
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from law.util import InsertableDict

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Jet.pt", "Jet.eta",
    },
    produces={
        "trig_weight",
        "trig_weight_up",
        "trig_weight_down",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    # get_trig_file=(lambda self, external_files: external_files.trig_sf),
    # function to determine the trigger weight config
    # get_trig_config=(lambda self: self.config_inst.x.trig_sf_names),
)
def trig_weights(
    self: Producer,
    events: ak.Array,
    # trig_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Compute event-level trigger scale-factor weights for MC events.

    The trigger weight is evaluated using a correctionlib ``CorrectionSet``
    based on the configured trigger scale-factor variable
    (e.g. ``jet6_pt`` or ``ht``).

    The following columns are produced:
    - ``trig_weight``: nominal trigger scale-factor weight
    - ``trig_weight_up``: upward systematic variation
    - ``trig_weight_down``: downward systematic variation

    The uncertainty is modeled as a symmetric variation around the nominal
    weight using ±0.5 · |1 − weight|.

    For datasets without top quarks, all trigger weights are set to unity.
    """

    if self.dataset_inst.has_tag("has_top"):
        # Determine the pT of the 6th leading jet within |eta| < 2.6. If fewer than 6 jets are present, set to 0.
        jet6_pt = ak.where(
            ak.num(events.Jet[(abs(events.Jet.eta) < 2.6)], axis=1) > 5,
            ak.sort(events.Jet[(abs(events.Jet.eta) < 2.6)].pt[:], ascending=False, axis=1),
            np.zeros((len(events), 6)))[:, 5]

        # Compute HT: scalar sum of jet pT for jets passing the trigger-like selection
        ht = ak.sum(events.Jet.pt[(events.Jet.pt > 32) & (abs(events.Jet.eta) < 2.6)], axis=1)

        # Evaluate trigger scale factor using the configured variable
        if self.config_inst.x.trigger_sf_variable.startswith("jet6_pt"):
            # Apply the correction as a function of the 6th jet pT. Events with fewer than 6 jets receive weight = 0.
            weight = ak.where(jet6_pt == 0, np.zeros((len(events))), self.trig_sf_corrector(jet6_pt))

        if self.config_inst.x.trigger_sf_variable.startswith("ht"):
            # Apply the correction as a function of HT.
            weight = self.trig_sf_corrector(ht)

        # Define systematic variations around the nominal weight, corresponds to ±50% of the deviation from unity.
        weight_up = weight + abs(1 - weight) * 0.5
        weight_down = ak.where((weight - abs(1 - weight) * 0.5) > 0, (weight - abs(1 - weight) * 0.5), 0)

        # Store the nominal and varied weights as event-level columns
        events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
        events = set_ak_column(events, "trig_weight_up", weight_up, value_type=np.float32)
        events = set_ak_column(events, "trig_weight_down", weight_down, value_type=np.float32)
    else:
        # For datasets without top quarks, no trigger correction is applied and all weights default to unity.
        events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)
        events = set_ak_column(events, "trig_weight_up", np.ones(len(events)), value_type=np.float32)
        events = set_ak_column(events, "trig_weight_down", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights.requires
def trig_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if ("external_files") in reqs:
        return

    from alljets.tasks.ProduceTriggerWeights import ProduceTriggerWeight
    pinned_version = get("versions", "cfg_2017_v9__task_cf.ProduceTriggerWeight")
    reqs["external_files"] = ProduceTriggerWeight(
        version=pinned_version,
        datasets="tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,data*",
        configs=task.config,
        selector="trigger_eff",
        producers="no_norm,trigger_prod",
        variables=self.config_inst.x.trigger_sf_variable + "-trig_bits",
        hist_producer="trig_all_weights",
        processes="data,tt",
        selector_steps=self.config_inst.x.selector_step_groups[self.config_inst.x.trigger_sf_variable],
        general_settings="bin_sel=1,unweighted=0,cut_vis=vspan",
        categories="incl",
    )


@trig_weights.setup
def trig_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    # bundle = reqs["external_files"]
    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    # correction_set = correctionlib.CorrectionSet.from_string(
    #     self.get_trig_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    # )
    correction_set = correctionlib.CorrectionSet.from_string(
        inputs["external_files"]["collection"].targets[0]["weights"][0].load(
            formatter="gzip").decode("utf-8"),
    )
    # Add distinction for year and working point later. For now only one
    # corrector_name, self.year, self.wp = self.get_trig_config()
    # self.trig_sf_corrector = correction_set[corrector_name]
    self.trig_sf_corrector = correction_set["trig_cor"]
    # self.trig_sf_up_corrector = correction_set["trig_cor_up"]
    # self.trig_sf_down_corrector = correction_set["trig_cor_down"]
