# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from alljets.plotting.aj_plot_all import aj_plot_all, convert_weightedmean_to_weight
from columnflow.plotting.plot_util import (
    prepare_style_config,
    remove_residual_axis,
    apply_variable_settings,
    apply_density,
)

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)

"""
law run cf.PlotVariables1D --version v1
--processes tt --variables jet6_pt-trig_bits
--datasets tt_fh_powheg --selector trigger_sel
--producers example,trigger_prod
--plot-function alljets.plotting.trigger_eff_closure.plot_efficiencies
"""


def plot_efficiencies(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    TODO.
    """
    hist_list = list(hists.values())
    if isinstance(hist_list[0]._storage_type(), hist.storage.WeightedMean):
        # hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    import IPython
    IPython.embed()
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(list(hists[0].keys())) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )
    color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    num_bins = int(len((hist_list[0])[1, 0, :, 0].values()))

    for i in range(num_bins):
        myhist_0 = hist_list[0]
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i}"] = {
            "method": "draw_efficiency",
            "ratio_method": "draw_hist",
            "hist": myhist_0[0, :, i, eff_bin],
            "kwargs": {
                "fillstyle": "none",
                "linestyle": "dotted",
                "color": color_list[i],
                "norm": myhist_0[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
            "ratio_kwargs": {
                "color": color_list[i],
                "norm": (myhist_1[0, :, i, eff_bin].values() * myhist_0[1, :, i, 0].values()) /
                (myhist_1[1, :, i, 0].values()),
                "histtype": "errorbar",
                "capsize": 3,
            },
        }

    for i in range(num_bins):
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i + 1}"] = {
            "method": "draw_efficiency",
            "hist": myhist_1[0, :, i, eff_bin],
            "kwargs": {
                "color": color_list[i],
                "norm": myhist_1[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
        }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 15
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    return aj_plot_all(plot_config, style_config, **kwargs)


def plot_efficiencies_no_weight(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    TODO.
    """
    hist_list = list(hists.values())
    if isinstance(hist_list[0]._storage_type(), hist.storage.WeightedMean):
        # hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(list(hists[0].keys())) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )

    color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    num_bins = int(len((hist_list[0])[1, 0, :, 0].values()))

    for i in range(num_bins):
        myhist_0 = hist_list[0]
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i}"] = {
            "method": "draw_efficiency",
            "ratio_method": "draw_hist",
            "hist": myhist_0[1, :, i, eff_bin],
            "kwargs": {
                "fillstyle": "none",
                "linestyle": "dotted",
                "color": color_list[i],
                "norm": myhist_0[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
            "ratio_kwargs": {
                "color": color_list[i],
                "norm": (myhist_1[1, :, i, eff_bin].values() * myhist_0[1, :, i, 0].values()) /
                (myhist_1[1, :, i, 0].values()),
                "histtype": "errorbar",
                "capsize": 3,
            },
        }

    for i in range(num_bins):
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i + 1}"] = {
            "method": "draw_efficiency",
            "hist": myhist_1[1, :, i, eff_bin],
            "kwargs": {
                "color": color_list[i],
                "norm": myhist_1[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
        }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 15
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)


def produce_weight(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    TODO.
    """
    hists = apply_density(hists, density)
    hist_list = list(hists.values())
    if isinstance(hist_list[0]._storage_type(), hist.storage.WeightedMean):
        # hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])

    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(list(hists[0].keys())) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )

    color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    num_bins = int(len((hist_list[0])[1, 0, :, 0].values()))
    num_0 = np.ones_like((hist_list[0])[1, :, :, 0].values())
    num_1 = np.ones_like((hist_list[0])[1, :, :, 0].values())
    den_0 = np.ones_like((hist_list[0])[1, :, :, 0].values())
    den_1 = np.ones_like((hist_list[0])[1, :, :, 0].values())

    for i in range(num_bins):
        myhist_0 = hist_list[0]
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        norm = ((myhist_1[1, :, i, eff_bin].values() * myhist_0[1, :, i, 0].values()) /
                (myhist_1[1, :, i, 0].values()))
        plot_config[f"hist_{2*i}"] = {
            "method": "draw_efficiency",
            "ratio_method": "draw_hist",
            "hist": myhist_0[1, :, i, eff_bin],
            "kwargs": {
                "fillstyle": "none",
                "linestyle": "dotted",
                "color": color_list[i],
                "norm": myhist_0[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
            "ratio_kwargs": {
                "color": color_list[i],
                "norm": norm,
                "histtype": "errorbar",
                "capsize": 3,
            },
        }
        num_0[:, i] = myhist_0[1, :, i, eff_bin].values()
        den_0[:, i] = myhist_0[1, :, i, 0].values()
        num_1[:, i] = myhist_1[1, :, i, eff_bin].values()
        den_1[:, i] = myhist_1[1, :, i, 0].values()

    for i in range(num_bins):
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i + 1}"] = {
            "method": "draw_efficiency",
            "hist": myhist_1[1, :, i, eff_bin],
            "kwargs": {
                "color": color_list[i],
                "norm": myhist_1[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
        }

    # create weight
    import correctionlib.convert
    import correctionlib.schemav2
    from scipy.stats import beta
    weight = (num_0 * den_1) / (num_1 * den_0)
    weight = np.nan_to_num(weight, nan=1.0)
    confint = 0.68
    quant = (1 - confint) / 2.
    low_0 = beta.ppf(quant, num_0, den_0 - num_0 + 1)
    high_0 = beta.ppf(1 - quant, num_0 + 1, den_0 - num_0)
    low_0 = np.nan_to_num(low_0)
    high_0 = np.where(np.isnan(high_0), 1, high_0)
    low_1 = beta.ppf(quant, num_1, den_1 - num_1 + 1)
    high_1 = beta.ppf(1 - quant, num_1 + 1, den_1 - num_1)
    low_1 = np.nan_to_num(low_1)
    high_1 = np.where(np.isnan(high_1), 1, high_1)
    weight_down = low_0 / high_1
    weight_up = np.nan_to_num(high_0 / low_1, posinf=99999)
    weight_hist = hist.Hist(*(hist_list[0])[1, :, :, 0].axes, data=weight)
    weight_up_hist = hist.Hist(*(hist_list[0])[1, :, :, 0].axes, data=weight_up)
    weight_down_hist = hist.Hist(*(hist_list[0])[1, :, :, 0].axes, data=weight_down)
    weight_name = kwargs.get("name", "trig_cor")
    weight_hist.name = weight_name
    weight_up_hist.name = weight_name + "_up"
    weight_down_hist.name = weight_name + "_down"
    weight_hist.label = "out"
    weight_down_hist.label = "out"
    weight_up_hist.label = "out"
    trig_cor = correctionlib.convert.from_histogram(weight_hist)
    trig_cor_up = correctionlib.convert.from_histogram(weight_up_hist)
    trig_cor_down = correctionlib.convert.from_histogram(weight_down_hist)
    trig_cor.description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger"
    )
    trig_cor_up.description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger, up variation"
    )
    trig_cor_down.description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger, down variation"
    )
    trig_cor.data.flow = "clamp"
    trig_cor_up.data.flow = "clamp"
    trig_cor_down.data.flow = "clamp"
    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="Custom trigger corrections with up and down variations",
        corrections=[
            trig_cor,
            trig_cor_up,
            trig_cor_down,
        ],
    )
    axes = (hist_list[0])[1, :, :, 0].axes.name
    file_name = f"{weight_name}_{trigger_names[eff_bin]}_{trigger_names[0]}_{axes[0]}_{axes[1]}.json"
    with open(file_name, "w") as fout:
        fout.write(cset.json(exclude_unset=True))
    import gzip
    with gzip.open(file_name + ".gz", "wt") as fout:
        fout.write(cset.json(exclude_unset=True))

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 15
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)


def produce_sec_weight(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    TODO.
    """
    hist_list = list(hists.values())
    if isinstance(hist_list[0]._storage_type(), hist.storage.WeightedMean):
        # hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(list(hists[0].keys())) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )
    color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    num_bins = int(len((hist_list[0])[0, 0, :, 0].values()))
    num_0 = np.ones_like((hist_list[0])[0, :, :, 0].values())
    num_1 = np.ones_like((hist_list[0])[0, :, :, 0].values())
    den_0 = np.ones_like((hist_list[0])[0, :, :, 0].values())
    den_1 = np.ones_like((hist_list[0])[0, :, :, 0].values())

    for i in range(num_bins):
        myhist_0 = hist_list[0]
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[0].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i}"] = {
            "method": "draw_efficiency",
            "ratio_method": "draw_hist",
            "hist": myhist_0[1, :, i, eff_bin],
            "kwargs": {
                "fillstyle": "none",
                "linestyle": "dotted",
                "color": color_list[i],
                "norm": myhist_0[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
            "ratio_kwargs": {
                "color": color_list[i],
                "norm": (myhist_1[0, :, i, eff_bin].values() * myhist_0[1, :, i, 0].values()) /
                (myhist_1[1, :, i, 0].values()),
                "histtype": "errorbar",
                "capsize": 3,
            },
        }
        num_0[:, i] = myhist_0[0, :, i, eff_bin].values()
        den_0[:, i] = myhist_0[1, :, i, 0].values()
        num_1[:, i] = myhist_1[0, :, i, eff_bin].values()
        den_1[:, i] = myhist_1[1, :, i, 0].values()

    for i in range(num_bins):
        myhist_1 = hist_list[1]
        if (float("inf") == myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1]):
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to inf"
            )
        else:
            text = (
                f"{list(hists[0].keys())[1].name}, "
                f"{myhist_0[0, 0, :, eff_bin].axes[0].label}:"
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i])} to "
                f"{int(myhist_0[0, 0, :, eff_bin].axes.edges[0][i + 1])}"
            )
        plot_config[f"hist_{2*i + 1}"] = {
            "method": "draw_efficiency",
            "hist": myhist_1[0, :, i, eff_bin],
            "kwargs": {
                "color": color_list[i],
                "norm": myhist_1[1, :, i, 0].values(),
                "label": text,
                "capsize": 3,
            },
        }

    # create weight
    import correctionlib.convert
    import correctionlib.schemav2
    from scipy.stats import beta
    weight = (num_0 * den_1) / (num_1 * den_0)
    weight = np.nan_to_num(weight, nan=1.0)
    confint = 0.68
    quant = (1 - confint) / 2.
    low_0 = beta.ppf(quant, num_0, den_0 - num_0 + 1)
    high_0 = beta.ppf(1 - quant, num_0 + 1, den_0 - num_0)
    low_0 = np.nan_to_num(low_0)
    high_0 = np.where(np.isnan(high_0), 1, high_0)
    low_1 = beta.ppf(quant, num_1, den_1 - num_1 + 1)
    high_1 = beta.ppf(1 - quant, num_1 + 1, den_1 - num_1)
    low_1 = np.nan_to_num(low_1)
    high_1 = np.where(np.isnan(high_1), 1, high_1)
    weight_down = low_0 / high_1
    weight_up = np.nan_to_num(high_0 / low_1, posinf=99999)
    weight_hist = hist.Hist(*(hist_list[0])[1, :, :, 0].axes, data=weight)
    weight_up_hist = hist.Hist(*(hist_list[0])[1, :, :, 0].axes, data=weight_up)
    weight_down_hist = hist.Hist(*(hist_list[0])[1, :, :, 0].axes, data=weight_down)
    weight_name = kwargs.get("name", "second_trig_cor")
    weight_hist.name = weight_name
    weight_up_hist.name = weight_name + "_up"
    weight_down_hist.name = weight_name + "_down"
    weight_hist.label = "out"
    weight_down_hist.label = "out"
    weight_up_hist.label = "out"
    trig_cor = correctionlib.convert.from_histogram(weight_hist)
    trig_cor_up = correctionlib.convert.from_histogram(weight_up_hist)
    trig_cor_down = correctionlib.convert.from_histogram(weight_down_hist)
    trig_cor.description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger"
    )
    trig_cor_up.description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger, up variation"
    )
    trig_cor_down.description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger, down variation"
    )
    trig_cor.data.flow = "clamp"
    trig_cor_up.data.flow = "clamp"
    trig_cor_down.data.flow = "clamp"
    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="Custom trigger corrections with up and down variations",
        corrections=[
            trig_cor,
            trig_cor_up,
            trig_cor_down,
        ],
    )
    axes = (hist_list[0])[1, :, :, 0].axes.name
    file_name = f"{weight_name}_{trigger_names[eff_bin]}_{trigger_names[0]}_{axes[0]}_{axes[1]}.json"
    with open(file_name, "w") as fout:
        fout.write(cset.json(exclude_unset=True))
    import gzip
    with gzip.open(file_name + ".gz", "wt") as fout:
        fout.write(cset.json(exclude_unset=True))

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 15
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)
