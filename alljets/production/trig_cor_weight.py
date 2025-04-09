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
        "trig_weight", "trig_weight_up", "trig_weight_down",
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
    if self.dataset_inst.has_tag("has_top"):
        jet6_pt = ak.where(
            ak.num(events.Jet[(abs(events.Jet.eta) < 2.6)], axis=1) > 5,
            ak.sort(events.Jet[(abs(events.Jet.eta) < 2.6)].pt[:], ascending=False, axis=1),
            np.zeros((len(events), 6)),
        )[:, 5]
        ht = ak.sum(events.Jet.pt[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.6)], axis=1)
        weight = self.trig_sf_corrector(jet6_pt, ht)
        weight_up = self.trig_sf_up_corrector(jet6_pt, ht)
        weight_down = self.trig_sf_down_corrector(jet6_pt, ht)
        # store it
        events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
        events = set_ak_column(events, "trig_weight_up", weight_up, value_type=np.float32)
        events = set_ak_column(events, "trig_weight_down", weight_down, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)
        events = set_ak_column(events, "trig_weight_up", np.ones(len(events)), value_type=np.float32)
        events = set_ak_column(events, "trig_weight_down", np.ones(len(events)), value_type=np.float32)
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
    self.trig_sf_corrector = correction_set["trig_cor"]
    self.trig_sf_up_corrector = correction_set["trig_cor_up"]
    self.trig_sf_down_corrector = correction_set["trig_cor_down"]


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
    get_trig_file=(lambda self, external_files: external_files.trig_sf_pt),
    # function to determine the trigger weight config
    # get_trig_config=(lambda self: self.config_inst.x.trig_sf_names),
)
def trig_weights_pt(
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
        weight = self.trig_sf_corrector(jet6_pt, 400.0)
        # store it
        events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights_pt.requires
def trig_weights_pt_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trig_weights_pt.setup
def trig_weights_pt_setup(
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
    self.trig_sf_corrector = correction_set["trig_cor"]


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
    get_trig_file=(lambda self, external_files: external_files.trig_sf_ht),
    # function to determine the trigger weight config
    # get_trig_config=(lambda self: self.config_inst.x.trig_sf_names),
)
def trig_weights_ht(
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
        ht = ak.sum(events.Jet.pt[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.6)], axis=1)
        weight = self.trig_sf_corrector(ht, 40.0)
        # store it
        events = set_ak_column(events, "trig_weight", weight, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights_ht.requires
def trig_weights_ht_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trig_weights_ht.setup
def trig_weights_ht_setup(
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
    self.trig_sf_corrector = correction_set["trig_cor"]


@producer(
    uses={
        "Jet.pt", "Jet.eta",
    },
    produces={
        "trig_weight_2",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_trig_file=(lambda self, external_files: external_files.trig_sf_pt_after_ht),
    # function to determine the trigger weight config
    # get_trig_config=(lambda self: self.config_inst.x.trig_sf_names),
)
def trig_weights_pt_after_ht(
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
        weight = self.trig_sf_corrector(jet6_pt, 400.0)
        # store it
        events = set_ak_column(events, "trig_weight_2", weight, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight_2", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights_pt_after_ht.requires
def trig_weights_pt_after_ht_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trig_weights_pt_after_ht.setup
def trig_weights_pt_after_ht_setup(
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
    self.trig_sf_corrector = correction_set["second_trig_cor"]


@producer(
    uses={
        "Jet.pt", "Jet.eta",
    },
    produces={
        "trig_weight_2",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_trig_file=(lambda self, external_files: external_files.trig_sf_ht_after_pt),
    # function to determine the trigger weight config
    # get_trig_config=(lambda self: self.config_inst.x.trig_sf_names),
)
def trig_weights_ht_after_pt(
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
        ht = ak.sum(events.Jet.pt[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.6)], axis=1)
        weight = self.trig_sf_corrector(ht, 40.0)
        events = set_ak_column(events, "trig_weight_2", weight, value_type=np.float32)
    else:
        events = set_ak_column(events, "trig_weight_2", np.ones(len(events)), value_type=np.float32)
    return events


@trig_weights_ht_after_pt.requires
def trig_weights_ht_after_pt_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@trig_weights_ht_after_pt.setup
def trig_weights_ht_after_pt_setup(
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
    self.trig_sf_corrector = correction_set["second_trig_cor"]
