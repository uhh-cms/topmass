# coding: utf-8

"""
Trigger related event weights.
"""

from __future__ import annotations

import law
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from law.util import InsertableDict
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Jet.pt", "Jet.eta",
    },
    produces={
        "trig_weight",
        # "trig_weight_up",
        # "trig_weight_down",
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
    Creates trigger weights using the correctionlib. Requires an external file in the config under
    ``trig_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "trig_sf": "/afs/desy.de/user/d/davidsto/public/mirrors/trigger_correction_HT350_CSV.json.gz",  # noqa
        })
    """
    if self.dataset_inst.has_tag("has_top"):
        jet6_pt = ak.where(
            ak.num(events.Jet[(abs(events.Jet.eta) < 2.6)], axis=1) > 5,
            ak.sort(events.Jet[(abs(events.Jet.eta) < 2.6)].pt[:], ascending=False, axis=1),
            np.zeros((len(events), 6)),
        )[:, 5]
        ht = ak.sum(events.Jet.pt[(events.Jet.pt > 32) & (abs(events.Jet.eta) < 2.6)], axis=1)
        if self.config_inst.x.trigger_sf_variable.startswith("jet6_pt"):
            weight = self.trig_sf_corrector(jet6_pt)
        if self.config_inst.x.trigger_sf_variable.startswith("ht"):
            weight = self.trig_sf_corrector(ht)

        # store it
        events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
        # events = set_ak_column(events, "trig_weight_up", weight_up, value_type=np.float32)
        # events = set_ak_column(events, "trig_weight_down", weight_down, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)
        # events = set_ak_column(events, "trig_weight_up", np.ones(len(events)), value_type=np.float32)
        # events = set_ak_column(events, "trig_weight_down", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights.requires
def trig_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    if ("external_files") in reqs:
        return

    # from columnflow.tasks.external import BundleExternalFiles
    # reqs["external_files"] = BundleExternalFiles.req(self.task)
    from alljets.tasks.ProduceTriggerWeights import ProduceTriggerWeight
    reqs["external_files"] = ProduceTriggerWeight(
        version=task.version,
        datasets="tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,data*",
        configs=task.config,
        selector="trigger_eff",
        producers="no_norm,trigger_prod",
        variables=self.config_inst.x.trigger_sf_variable + "-trig_bits",
        hist_producer="trig_all_weights",
        selector_steps=self.config_inst.x.selector_step_groups[self.config_inst.x.trigger_sf_variable],
        general_settings="bin_sel=1,unweighted=1",
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
        inputs["external_files"]["collection"].targets[0]["weights"][0].load(formatter="gzip").decode("utf-8"),
    )
    # Add distinction for year and working point later. For now only one
    # corrector_name, self.year, self.wp = self.get_trig_config()
    # self.trig_sf_corrector = correction_set[corrector_name]
    self.trig_sf_corrector = correction_set["trig_cor"]
    # self.trig_sf_up_corrector = correction_set["trig_cor_up"]
    # self.trig_sf_down_corrector = correction_set["trig_cor_down"]
