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
        "trig_weight_full_up",
        "trig_weight_full_down",
    },
    mc_only=True,
)
def trig_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Compute event-level trigger scale-factor weights for MC events.

    The trigger weight is evaluated using a correctionlib CorrectionSet
    built from the ProduceTriggerWeights task. The underlying trigger efficiency
    is parameterized as a function of a jet-based variable (trigjet6_pt, or ht),
    originally derived from trigger-like jets (TrigJets).

    In this function, the same variable is reconstructed from the NanoAOD
    Jet collection, and the corresponding scale factor is evaluated.
    The resulting weight is applied at the event level.

    The following event-level columns are produced:
    - trig_weight: nominal trigger scale-factor weight
    - trig_weight_up/down: variations corresponding to ±50% of the nominal deviation from unity
    - trig_weight_full_up/down: variations corresponding to ±100% of the nominal deviation from unity

    Systematic variations are defined as fractions of the nominal weight's
    deviation from unity. The standard variations correspond to ±50% of
    this deviation, while the full variations correspond to ±100%.
    """

    jet6_pt = ak.where(
        ak.num(events.Jet[(abs(events.Jet.eta) < 2.6)], axis=1) > 5,
        ak.sort(events.Jet[(abs(events.Jet.eta) < 2.6)].pt[:], ascending=False, axis=1),
        np.zeros((len(events), 6)))[:, 5]

    # Compute HT: scalar sum of jet pT for jets passing the trigger-like selection
    ht = ak.sum(events.Jet.pt[(events.Jet.pt > 32) & (abs(events.Jet.eta) < 2.6)], axis=1)

    # Evaluate trigger scale factor using the configured variable
    if self.config_inst.x.trigger_sf_variable.startswith("trigjet6_pt"):
        # Apply the correction as a function of the 6th jet pT. Events with fewer than 6 jets receive weight = 0.
        weight = ak.where(jet6_pt == 0, np.zeros((len(events))), self.trig_sf_corrector(jet6_pt))

    elif self.config_inst.x.trigger_sf_variable.startswith("ht"):
        # Apply the correction as a function of HT.
        weight = self.trig_sf_corrector(ht)
    else:
        raise ValueError(
            f"Unsupported trigger SF variable: " f"{self.config_inst.x.trigger_sf_variable}",
        )

    # Define systematic variations
    # Deviation of the nominal trigger SF from unity
    deviation = np.abs(1.0 - weight)

    # Variations corresponding to 50% and 100% of the deviation
    weight_up = weight + 0.5 * deviation
    weight_down = np.maximum(weight - 0.5 * deviation, 0.0)

    weight_full_up = weight + deviation
    weight_full_down = np.maximum(weight - deviation, 0.0)

    # Store the nominal and varied weights as event-level columns
    events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
    events = set_ak_column(events, "trig_weight_up", weight_up, value_type=np.float32)
    events = set_ak_column(events, "trig_weight_down", weight_down, value_type=np.float32)
    events = set_ak_column(events, "trig_weight_full_up", weight_full_up, value_type=np.float32)
    events = set_ak_column(events, "trig_weight_full_down", weight_full_down, value_type=np.float32)

    return events


@trig_weights.requires
def trig_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if ("external_files") in reqs:
        return

    year = self.config_inst.campaign.x.year

    from alljets.tasks.ProduceTriggerWeights import ProduceTriggerWeight
    pinned_version = get("versions", f"cfg_{year}_v9__task_cf.ProduceTriggerWeight")
    reqs["external_files"] = ProduceTriggerWeight(
        version=pinned_version,
        datasets="tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,data*",
        configs=task.config,
        selector="trigger",
        producers="trigSF_prod,trigger_prod",
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

    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        inputs["external_files"]["collection"].targets[0]["weights"][0].load(
            formatter="gzip").decode("utf-8"),
    )
    self.trig_sf_corrector = correction_set["trig_cor"]
