# coding: utf-8

"""
Trigger related event weights.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Jet.pt", "Jet.eta",
    },
    produces={
        "trig_weight",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_trig_file=(lambda self, external_files: external_files.trig_sf),
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
    # flat super cluster eta and pt views
    # sc_eta = flat_np_view((
    #     events.Electron.eta[electron_mask] +
    #     events.Electron.deltaEtaSC[electron_mask]
    # ), axis=1)
    # pt = flat_np_view(events.Jet.pt, axis=1)
    if self.dataset_inst.has_tag("has_top"):
        jet6_pt = ak.where(
            ak.num(events.Jet, axis=1) > 5,
            ak.sort(events.Jet.pt[:], ascending=False, axis=1),
            np.zeros((len(events), 6)),
        )[:, 5]
        ht = ak.sum(events.Jet.pt[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.6)], axis=1)
        weight = self.trig_sf_corrector(jet6_pt, ht)
        # store it
        events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights.requires
def trig_weights_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trig_weights.setup
def trig_weights_setup(
    self: Producer,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    bundle = reqs["external_files"]

    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_trig_file(bundle.files).load(formatter="gzip").decode("utf-8"),
    )
    # Add distinction for year and working point later. For now only one
    # corrector_name, self.year, self.wp = self.get_trig_config()
    # self.trig_sf_corrector = correction_set[corrector_name]
    self.trig_sf_corrector = correction_set["proto_trig_cor"]
