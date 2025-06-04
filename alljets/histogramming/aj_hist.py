# coding: utf-8

"""
Custom trigger histogram producer.
"""

from __future__ import annotations

import law
import order as od

from columnflow.histogramming import HistProducer, hist_producer
from columnflow.util import maybe_import
from columnflow.hist_util import translate_hist_intcat_to_strcat, add_hist_axis
from columnflow.columnar_util import has_ak_column, Route
from columnflow.types import Any
from columnflow.columnar_util import flat_np_view

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


@hist_producer()
def aj_trighist(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.Array(np.ones(len(events), dtype=np.float32))


@aj_trighist.create_hist
def aj_trighist_create_hist(
    self: HistProducer,
    variables: list[od.Variable],
    task: law.Task,
    **kwargs,
) -> hist.Histogram:
    """
    Define the histogram structure for the default histogram producer.
    Defines additional weight category to store unweighted events for trigger efficiency validation.
    """
    return create_hist_from_variables_with_mean(
        *variables,
        categorical_axes=(
            ("weightcategory", "intcat"),
            ("category", "intcat"),
            ("process", "intcat"),
            ("shift", "intcat"),
        ),
        storage="mean",
    )


@aj_trighist.fill_hist
def aj_trighist_fill_hist(self: HistProducer, h: hist.Histogram, data: dict[str, Any], task: law.Task) -> None:
    """
    Fill the histogram with the data.
    """
    fill_kwargs = {}
    # determine the axis names, figure out which which axes the last bin correction should be done
    axis_names = []
    correct_last_bin_axes = []
    for ax in h.axes:
        axis_names.append(ax.name)
        # include values hitting last edge?
        if not len(ax.widths) or not isinstance(ax, hist.axis.Variable):
            continue

    # check data
    if not isinstance(data, dict):
        if len(axis_names) != 1:
            raise ValueError("got multi-dimensional hist but only one-dimensional data")
        data = {axis_names[0]: data}
    else:
        for name in axis_names:
            if name not in data and name not in fill_kwargs and not name == "weightcategory":
                raise ValueError(f"missing data for histogram axis '{name}'")

    # correct last bin values
    for ax in correct_last_bin_axes:
        right_egde_mask = ak.flatten(data[ax.name], axis=None) == ax.edges[-1]
        if np.any(right_egde_mask):
            data[ax.name] = ak.copy(data[ax.name])
            flat_np_view(data[ax.name])[right_egde_mask] -= ax.widths[-1] * 1e-5

    # check if conversion to records is needed
    arr_types = (ak.Array, np.ndarray)
    vals = list(data.values())
    convert = (
        # values is a mixture of singular and array types
        (any(isinstance(v, arr_types) for v in vals) and not all(isinstance(v, arr_types) for v in vals)) or
        # values contain at least one array with more than one dimension
        any(isinstance(v, arr_types) and v.ndim != 1 for v in vals)
    )

    # actual conversion
    if convert:
        data2 = data.copy()
        data["weightcategory"] = 0
        data2["weightcategory"] = 1
        data2["weight"] = ak.Array(np.ones(len(data["weight"])))
        arrays = ak.flatten(ak.cartesian(data))
        arrays2 = ak.flatten(ak.cartesian(data2))
        data = {field: arrays[field] for field in arrays.fields}
        data2 = {field: arrays2[field] for field in arrays2.fields}
        del arrays
        del arrays2

    names = [ax.name for ax in h.axes if ax.__class__.__name__ not in ("StrCategory", "IntCategory")]
    h.fill(**data, sample=data[names[0]])
    h.fill(**data2, sample=data2[names[0]])


@aj_trighist.post_process_hist
def aj_trighist_post_process_hist(self: HistProducer, h: hist.Histogram, task: law.Task) -> hist.Histogram:
    """
    Post-process the histogram, converting integer to string axis for consistent lookup across configs where ids might
    be different.
    """
    axis_names = {ax.name for ax in h.axes}
    # translate axes
    if "category" in axis_names:
        category_map = {cat.id: cat.name for cat in self.config_inst.get_leaf_categories()}
        h = translate_hist_intcat_to_strcat(h, "category", category_map)
    if "process" in axis_names:
        process_map = {proc_id: self.config_inst.get_process(proc_id).name for proc_id in h.axes["process"]}
        h = translate_hist_intcat_to_strcat(h, "process", process_map)
    if "shift" in axis_names:
        shift_map = {task.global_shift_inst.id: task.global_shift_inst.name}
        h = translate_hist_intcat_to_strcat(h, "shift", shift_map)
    if "weightcategory" in axis_names:
        wcat_map = {0: "weighted", 1: "unweighted"}
        h = translate_hist_intcat_to_strcat(h, "weightcategory", wcat_map)
    return h


@aj_trighist.hist_producer()
def trig_all_weights(self: HistProducer, events: ak.Array, **kwargs) -> ak.Array:
    """
    HistProducer that combines all event weights from the *event_weights* aux entry from either the config or the
    dataset. The weights are multiplied together to form the full event weight.

    The expected structure of the *event_weights* aux entry is a dictionary with the weight column name as key and a
    list of shift sources as values. The shift sources are used to declare the shifts that the produced event weight
    depends on. Example:

    .. code-block:: python

        from columnflow.config_util import get_shifts_from_sources
        # add weights and their corresponding shifts for all datasets
        cfg.x.event_weights = {
            "normalization_weight": [],
            "muon_weight": get_shifts_from_sources(config, "mu_sf"),
            "btag_weight": get_shifts_from_sources(config, "btag_hf", "btag_lf"),
        }
        for dataset_inst in cfg.datasets:
            # add dataset-specific weights and their corresponding shifts
            dataset.x.event_weights = {}
            if not dataset_inst.has_tag("skip_pdf"):
                dataset_inst.x.event_weights["pdf_weight"] = get_shifts_from_sources(config, "pdf")
    """
    weight = ak.Array(np.ones(len(events)))

    # build the full event weight
    if self.dataset_inst.is_mc and len(events):
        # multiply weights from global config `event_weights` aux entry
        for column in self.config_inst.x.event_weights:
            weight = weight * Route(column).apply(events)

        # multiply weights from dataset-specific `event_weights` aux entry
        for column in self.dataset_inst.x("event_weights", []):
            if has_ak_column(events, column):
                weight = weight * Route(column).apply(events)
            else:
                self.logger.warning_once(
                    f"missing_dataset_weight_{column}",
                    f"weight '{column}' for dataset {self.dataset_inst.name} not found",
                )

    return events, weight


@trig_all_weights.init
def trig_all_weights_init(self: HistProducer) -> None:
    weight_columns = set()

    if self.dataset_inst.is_data:
        return

    # add used weight columns and declare shifts that the produced event weight depends on
    if self.config_inst.has_aux("event_weights"):
        weight_columns |= {Route(column) for column in self.config_inst.x.event_weights}
        for shift_insts in self.config_inst.x.event_weights.values():
            self.shifts |= {shift_inst.name for shift_inst in shift_insts}

    # optionally also for weights defined by a dataset
    if self.dataset_inst.has_aux("event_weights"):
        weight_columns |= {Route(column) for column in self.dataset_inst.x("event_weights", [])}
        for shift_insts in self.dataset_inst.x.event_weights.values():
            self.shifts |= {shift_inst.name for shift_inst in shift_insts}

    # add weight columns to uses
    self.uses |= weight_columns


def create_hist_from_variables_with_mean(
    *variable_insts,
    categorical_axes: tuple[tuple[str, str]] | None = None,
    weight: bool = True,
    mean: bool = True,
    storage: str | None = None,
) -> hist.Hist:
    histogram = hist.Hist.new

    # additional category axes
    if categorical_axes:
        for name, axis_type in categorical_axes:
            if axis_type in ("intcategory", "intcat"):
                histogram = histogram.IntCat([], name=name, growth=True)
            elif axis_type in ("strcategory", "strcat"):
                histogram = histogram.StrCat([], name=name, growth=True)
            else:
                raise ValueError(f"unknown axis type '{axis_type}' in argument 'categorical_axes'")

    # requested axes from variables
    for variable_inst in variable_insts:
        histogram = add_hist_axis(histogram, variable_inst)

    # add the storage
    if storage is None:
        # use weight value for backwards compatibility
        storage = "weight" if weight else "double"
    else:
        storage = storage.lower()
    if storage == "weight":
        histogram = histogram.Weight()
    elif storage == "double":
        histogram = histogram.Double()
    elif storage == "mean":
        histogram = histogram.WeightedMean()
    else:
        raise ValueError(f"unknown storage type '{storage}'")
    return histogram
