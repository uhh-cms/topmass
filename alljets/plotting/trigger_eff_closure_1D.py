# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from inspect import signature

import law

from columnflow.util import maybe_import
from alljets.plotting.aj_plot_all import aj_plot_all
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
--producers default,trigger_prod
--plot-function alljets.plotting.trigger_eff_plot.plot_efficiencies
"""


def convert_weightedmean_to_weight(h_mean: hist.Hist, include_flow: bool = True) -> hist.Hist:
    if not isinstance(h_mean._storage_type(), hist.storage.WeightedMean):
        logger.warning(
            "Storage type is not WeightedMean.",
        )
        return h_mean

    # Reconstruct axes with full category sets and metadata
    axes = []
    for ax in h_mean.axes:
        name = ax.name
        label = ax.label

        if isinstance(ax, hist.axis.Regular):
            axes.append(hist.axis.Regular(ax.size, ax.start, ax.stop, name=name, label=label, flow=ax.options.flow))
        elif isinstance(ax, hist.axis.Integer):
            axes.append(hist.axis.Integer(ax.start, ax.stop, name=name, label=label, flow=ax.options.flow))
        elif isinstance(ax, hist.axis.IntCategory):
            axes.append(hist.axis.IntCategory(list(ax), name=name, label=label))
        elif isinstance(ax, hist.axis.StrCategory):
            axes.append(hist.axis.StrCategory(list(ax), name=name, label=label))
        elif isinstance(ax, hist.axis.Variable):
            axes.append(hist.axis.Variable(ax.edges, name=name, label=label))
        else:
            raise TypeError(f"Unsupported axis type: {type(ax)}")

    # Create the new histogram with Weight storage
    h_weight = hist.Hist(*axes, storage=hist.storage.Weight())

    # View with/without flow
    mean_view = h_mean.view(flow=include_flow)
    weight_view = h_weight.view(flow=include_flow)

    # Copy bin contents
    for idx in np.ndindex(mean_view.shape):
        bin_content = mean_view[idx]
        weight_view[idx] = (
            bin_content.sum_of_weights,
            bin_content.sum_of_weights_squared,
        )

    return h_weight


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def arctan(x, L, x0, k, b):
    y = L * (np.arctan(k * (x - x0) + np.pi * 0.5)) / np.pi + b
    return y


def binom_int(num, den, confint=0.68):
    from scipy.stats import beta
    quant = (1 - confint) / 2.
    low = beta.ppf(quant, num, den - num + 1)
    high = beta.ppf(1 - quant, num + 1, den - num)
    return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))


def eff_fit(
    h: hist.Hist,
    n: hist.Hist,
    fit_function: Callable,
):
    means = h.values()
    values = convert_weightedmean_to_weight(h).values()
    norm = convert_weightedmean_to_weight(n).values()
    efficiency = np.nan_to_num(values / norm, nan=0)
    band_low, band_high = binom_int(values, norm)
    error_low = np.asarray(efficiency - band_low)
    error_high = np.asarray(band_high - efficiency)
    error_low[error_low == 1] = 0
    error_high[error_high == 1] = 0
    err_max = np.where(abs(error_low) < abs(error_high), abs(error_high), abs(error_low))
    err_max = np.where(err_max == 0, 1, err_max)
    L0 = np.max(efficiency)
    x0_0 = means[np.argmin(abs(efficiency - (L0 / 2)))]
    k0 = 4 / (means[np.argmin(abs(efficiency - (L0 * 9 / 10)))] - means[np.argmin(abs(efficiency - (L0 / 10)))])
    from scipy.optimize import curve_fit
    fit = curve_fit(fit_function, means, efficiency, [L0, x0_0, k0, 0], sigma=err_max, absolute_sigma=True)
    dof = (len(means) - len(signature(fit_function).parameters))
    if dof > 0:
        chi2 = np.sum(
            (((fit_function(means, *fit[0]) - efficiency) ** 2) / (err_max ** 2)) / dof,
        )
    else:
        chi2 = -9999.0
    return [fit, chi2]


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
        hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))
    weighted = int(kwargs.get("unweighted", False))
    fit_func = kwargs.get("fit", "sigmoid")
    func_dict = {
        "sigmoid": sigmoid,
        "arctan": arctan,
    }

    if fit_func not in func_dict:
        TypeError("Unsupported function type")

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(hist_list) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )

    # color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    # for updating labels of individual selector steps
    myhist_0 = hist_list[0]
    myhist_1 = hist_list[1]

    norm_hist_0 = np.array(myhist_0[1, 0, :, 0].values())
    norm_hist_1 = np.array(myhist_1[1, 0, :, 0].values())

    # Fitting sigmoid or other function to efficiencies
    fit_result = np.zeros((len(hist_list_mean), 4))
    chi2 = np.zeros((len(hist_list_mean)))
    variances = np.zeros((len(hist_list_mean), 4, 4))
    for i in range(len(hist_list_mean)):
        fit = eff_fit(
            hist_list_mean[i][weighted, 0, :, eff_bin],
            hist_list_mean[i][weighted, 0, :, 0],
            fit_function=func_dict[fit_func],
        )
        fit_result[i] = fit[0][0]
        variances[i] = fit[0][1]
        chi2[i] = fit[1]

    if "hist_list_mean" not in locals():
        values = convert_weightedmean_to_weight(myhist_0[weighted, 0, :, eff_bin]).values()
        norm = norm_hist_0
        efficiency = np.nan_to_num(values / norm, nan=0)
        band_low, band_high = binom_int(values, norm)
        yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_low[yerror_low == 1] = 0
        yerror_high[yerror_high == 1] = 0
        yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                             yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
        yerrors = yerrors.T
        plot_config["hist_0"] = {
            "method": "draw_efficiency",
            "hist": myhist_0[weighted, 0, :, eff_bin],
            "fit_result": fit_result[0],
            "kwargs": {
                "color": "b",
                "norm": norm_hist_0,
                "label": f"{list(hists[0].keys())[0].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[0],3)})",
                "capsize": 3,
            },
            "ratio_method": "draw_ratio_of_fit",
            "ratio_kwargs": {
                "color": "b",
                "capsize": 3,
                "linestyle": "none",
                "norm": (myhist_1[weighted, 0, :, eff_bin].values() * norm_hist_0) / norm_hist_1,
                "yerr": yerrors,
                "fit_norm": fit_result[1],
            },
        }

        plot_config["hist_1"] = {
            "method": "draw_efficiency",
            "hist": myhist_1[weighted, 0, :, eff_bin],
            "fit_result": fit_result[1],
            "kwargs": {
                "color": "g",
                "norm": norm_hist_1,
                "label": f"{list(hists[0].keys())[1].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[1],3)})",
                "capsize": 3,
            },
        }
    else:
        values = convert_weightedmean_to_weight(hist_list_mean[0][weighted, 0, :, eff_bin]).values()
        norm = norm_hist_0
        efficiency = np.nan_to_num(values / norm, nan=0)
        band_low, band_high = binom_int(values, norm)
        yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_low[yerror_low == 1] = 0
        yerror_high[yerror_high == 1] = 0
        yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                             yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
        yerrors = yerrors.T
        plot_config["fit_0"] = {
            "method": "draw_efficiency",
            "hist": convert_weightedmean_to_weight(hist_list_mean[0][weighted, 0, :, eff_bin]),
            "kwargs": {
                "x": hist_list_mean[0][weighted, 0, :, eff_bin].values(),
                "color": "b",
                "norm": norm_hist_0,
                "label": f"{list(hists[0].keys())[0].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[0],3)})",
                "capsize": 3,
            },
            "ratio_method": "draw_errorbars",
            "ratio_kwargs": {
                "x": hist_list_mean[0][weighted, 0, :, eff_bin].values(),
                "color": "b",
                "capsize": 3,
                "linestyle": "none",
                "norm": (myhist_1[weighted, 0, :, eff_bin].values() * norm_hist_0) / norm_hist_1,
                "yerr": yerrors,
            },
        }
        plot_config["fit_1"] = {
            "method": "draw_efficiency",
            "hist": convert_weightedmean_to_weight(hist_list_mean[1][weighted, 0, :, eff_bin]),
            "kwargs": {
                "x": hist_list_mean[1][weighted, 0, :, eff_bin].values(),
                "color": "r",
                "norm": norm_hist_1,
                "label": f"{list(hists[0].keys())[1].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[1],3)})",
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
    return aj_plot_all(plot_config, style_config, fit_function=func_dict[fit_func], **kwargs)


def plot_efficiencies_with_fit(
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
        hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))
    weighted = int(kwargs.get("unweighted", False))
    fit_func = kwargs.get("fit", "sigmoid")
    func_dict = {
        "sigmoid": sigmoid,
        "arctan": arctan,
    }

    if fit_func not in func_dict:
        TypeError("Unsupported function type")

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(hist_list) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )

    # color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    # for updating labels of individual selector steps
    myhist_0 = hist_list[0]
    myhist_1 = hist_list[1]

    norm_hist_0 = np.array(myhist_0[1, 0, :, 0].values())
    norm_hist_1 = np.array(myhist_1[1, 0, :, 0].values())

    # Fitting sigmoid or other function to efficiencies
    fit_result = np.zeros((len(hist_list_mean), 4))
    chi2 = np.zeros((len(hist_list_mean)))
    variances = np.zeros((len(hist_list_mean), 4, 4))
    for i in range(len(hist_list_mean)):
        fit = eff_fit(
            hist_list_mean[i][weighted, 0, :, eff_bin],
            hist_list_mean[i][weighted, 0, :, 0],
            fit_function=func_dict[fit_func],
        )
        fit_result[i] = fit[0][0]
        variances[i] = fit[0][1]
        chi2[i] = fit[1]

    if "hist_list_mean" not in locals():
        values = convert_weightedmean_to_weight(myhist_0[weighted, 0, :, eff_bin]).values()
        norm = norm_hist_0
        efficiency = np.nan_to_num(values / norm, nan=0)
        band_low, band_high = binom_int(values, norm)
        yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_low[yerror_low == 1] = 0
        yerror_high[yerror_high == 1] = 0
        yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                             yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
        yerrors = yerrors.T
        plot_config["hist_0"] = {
            "method": "draw_efficiency",
            "hist": myhist_0[weighted, 0, :, eff_bin],
            "fit_result": fit_result[0],
            "kwargs": {
                "color": "b",
                "norm": norm_hist_0,
                "label": f"{list(hists[0].keys())[0].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[0],3)})",
                "capsize": 3,
            },
            "ratio_method": "draw_ratio",
            "ratio_kwargs": {
                "color": "b",
                "capsize": 3,
                "linestyle": "none",
                "norm": (myhist_1[weighted, 0, :, eff_bin].values() * norm_hist_0) / norm_hist_1,
                "yerr": yerrors,
                "fit_norm": fit_result[1],
            },
        }

        plot_config["hist_1"] = {
            "method": "draw_efficiency",
            "hist": myhist_1[weighted, 0, :, eff_bin],
            "fit_result": fit_result[1],
            "kwargs": {
                "color": "g",
                "norm": norm_hist_1,
                "label": f"{list(hists[0].keys())[1].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[1],3)})",
                "capsize": 3,
            },
        }
    else:
        values = convert_weightedmean_to_weight(hist_list_mean[0][weighted, 0, :, eff_bin]).values()
        norm = norm_hist_0
        efficiency = np.nan_to_num(values / norm, nan=0)
        band_low, band_high = binom_int(values, norm)
        yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
        yerror_low[yerror_low == 1] = 0
        yerror_high[yerror_high == 1] = 0
        yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                             yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
        yerrors = yerrors.T
        plot_config["fit_0"] = {
            "method": "draw_efficiency_with_fit",
            "hist": convert_weightedmean_to_weight(hist_list_mean[0][weighted, 0, :, eff_bin]),
            "fit_result": fit_result[0],
            "kwargs": {
                "x": hist_list_mean[0][weighted, 0, :, eff_bin].values(),
                "color": "b",
                "norm": norm_hist_0,
                "label": f"{list(hists[0].keys())[0].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[0],3)})",
                "capsize": 3,
            },
            "ratio_method": "draw_ratio_of_fit",
            "ratio_kwargs": {
                "x": hist_list_mean[0][weighted, 0, :, eff_bin].values(),
                "color": "b",
                "capsize": 3,
                "linestyle": "none",
                "norm": (myhist_1[weighted, 0, :, eff_bin].values() * norm_hist_0) / norm_hist_1,
                "yerr": yerrors,
                "fit_norm": fit_result[1],
            },
        }
        plot_config["fit_1"] = {
            "method": "draw_efficiency_with_fit",
            "hist": convert_weightedmean_to_weight(hist_list_mean[1][weighted, 0, :, eff_bin]),
            "fit_result": fit_result[1],
            "kwargs": {
                "x": hist_list_mean[1][weighted, 0, :, eff_bin].values(),
                "color": "r",
                "norm": norm_hist_1,
                "label": f"{list(hists[0].keys())[1].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[1],3)})",
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
    return aj_plot_all(plot_config, style_config, fit_function=func_dict[fit_func], **kwargs)


def produce_trig_weight(
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
        hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    else:
        TypeError("Unsupported hist storage type (not WeightedMean)")
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))
    weighted = int(kwargs.get("unweighted", False))
    fit_func = kwargs.get("fit", "sigmoid")
    func_dict = {
        "sigmoid": sigmoid,
        "arctan": arctan,
    }

    if fit_func not in func_dict:
        TypeError("Unsupported function type")

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(hist_list) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )

    # color_list = ["b", "g", "r", "c", "m", "y"]
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    # for updating labels of individual selector steps
    myhist_0 = hist_list[0]
    myhist_1 = hist_list[1]

    norm_hist_0 = np.array(myhist_0[1, 0, :, 0].values())
    norm_hist_1 = np.array(myhist_1[1, 0, :, 0].values())

    # Fitting sigmoid or other function to efficiencies
    fit_result = np.zeros((len(hist_list_mean), 4))
    chi2 = np.zeros((len(hist_list_mean)))
    variances = np.zeros((len(hist_list_mean), 4, 4))
    for i in range(len(hist_list_mean)):
        fit = eff_fit(
            hist_list_mean[i][weighted, 0, :, eff_bin],
            hist_list_mean[i][weighted, 0, :, 0],
            fit_function=func_dict[fit_func],
        )
        fit_result[i] = fit[0][0]
        variances[i] = fit[0][1]
        chi2[i] = fit[1]

    values = convert_weightedmean_to_weight(hist_list_mean[0][weighted, 0, :, eff_bin]).values()
    norm = norm_hist_0
    efficiency = np.nan_to_num(values / norm, nan=0)
    band_low, band_high = binom_int(values, norm)
    yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
    yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / myhist_1[weighted, 0, :, eff_bin].values()
    yerror_low[yerror_low == 1] = 0
    yerror_high[yerror_high == 1] = 0
    yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                            yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
    yerrors = yerrors.T
    plot_config["fit_0"] = {
        "method": "draw_efficiency_with_fit",
        "hist": convert_weightedmean_to_weight(hist_list_mean[0][weighted, 0, :, eff_bin]),
        "fit_result": fit_result[0],
        "kwargs": {
            "x": hist_list_mean[0][weighted, 0, :, eff_bin].values(),
            "color": "b",
            "norm": norm_hist_0,
            "label": f"{list(hists[0].keys())[0].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[0],3)})",
            "capsize": 3,
        },
        "ratio_method": "draw_ratio_of_fit",
        "ratio_kwargs": {
            "x": hist_list_mean[0][weighted, 0, :, eff_bin].values(),
            "color": "b",
            "capsize": 3,
            "linestyle": "none",
            "norm": (myhist_1[weighted, 0, :, eff_bin].values() * norm_hist_0) / norm_hist_1,
            "yerr": yerrors,
            "fit_norm": fit_result[1],
        },
    }
    plot_config["fit_1"] = {
        "method": "draw_efficiency_with_fit",
        "hist": convert_weightedmean_to_weight(hist_list_mean[1][weighted, 0, :, eff_bin]),
        "fit_result": fit_result[1],
        "kwargs": {
            "x": hist_list_mean[1][weighted, 0, :, eff_bin].values(),
            "color": "r",
            "norm": norm_hist_1,
            "label": f"{list(hists[0].keys())[1].name} " + r"($\chi^2$/d.o.f. $= \,$" + f"{round(chi2[1],3)})",
            "capsize": 3,
        },
    }
    # num_0 = myhist_0[weighted, 0, :, eff_bin].values()
    # den_0 = norm_hist_0
    # num_1 = myhist_1[weighted, 0, :, eff_bin].values()
    # den_1 = norm_hist_1
    # import correctionlib.convert
    import correctionlib.schemav2
    # from scipy.stats import beta
    # weight = (num_0 * den_1) / (num_1 * den_0)
    # weight = np.nan_to_num(weight, nan=1.0)
    # confint = 0.68
    # quant = (1 - confint) / 2.
    # low_0 = beta.ppf(quant, num_0, den_0 - num_0 + 1)
    # high_0 = beta.ppf(1 - quant, num_0 + 1, den_0 - num_0)
    # low_0 = np.nan_to_num(low_0)
    # high_0 = np.where(np.isnan(high_0), 1, high_0)
    # low_1 = beta.ppf(quant, num_1, den_1 - num_1 + 1)
    # high_1 = beta.ppf(1 - quant, num_1 + 1, den_1 - num_1)
    # low_1 = np.nan_to_num(low_1)
    # high_1 = np.where(np.isnan(high_1), 1, high_1)
    # weight_down = low_0 / high_1
    # weight_up = np.nan_to_num(high_0 / low_1, posinf=99999)
    # weight_hist = hist.Hist(*(hist_list[0])[weighted, 0, :, eff_bin].axes, data=weight)
    # weight_up_hist = hist.Hist(*(hist_list[0])[weighted, 0, :, eff_bin].axes, data=weight_up)
    # weight_down_hist = hist.Hist(*(hist_list[0])[weighted, 0, :, eff_bin].axes, data=weight_down)
    weight_name = kwargs.get("name", "trig_cor")
    # weight_hist.name = weight_name
    # weight_up_hist.name = weight_name + "_up"
    # weight_down_hist.name = weight_name + "_down"
    # weight_hist.label = "out"
    # weight_down_hist.label = "out"
    # weight_up_hist.label = "out"
    # trig_cor = correctionlib.convert.from_histogram(weight_hist)
    # trig_cor_up = correctionlib.convert.from_histogram(weight_up_hist)
    # trig_cor_down = correctionlib.convert.from_histogram(weight_down_hist)
    description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger"
    )
    # trig_cor_up.description = (
    #     f"Trigger correction using {trigger_names[0]}" +
    #     f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger, up variation"
    # )
    # trig_cor_down.description = (
    #     f"Trigger correction using {trigger_names[0]}" +
    #     f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger, down variation"
    # )
    # trig_cor.data.flow = "clamp"
    # trig_cor_up.data.flow = "clamp"
    # trig_cor_down.data.flow = "clamp"
    if fit_func == "sigmoid":
        trig_cor = correctionlib.schemav2.Correction(
            name=weight_name,
            description=description,
            version=1,
            inputs=[
                correctionlib.schemav2.Variable(name=variable_insts[0].name, type="real"),
            ],
            output=correctionlib.schemav2.Variable(
                name="weight",
                type="real",
                description="Multiplicative event weight"),
            data=correctionlib.schemav2.Formula(
                nodetype="formula",
                variables=[variable_insts[0].name],
                parser="TFormula",
                expression=f"({fit_result[0, 0]} / (1 + exp(-{fit_result[0, 2]} * (x - {fit_result[0, 1]}))) +" +
                f" {fit_result[0, 3]}) / ({fit_result[1, 0]} / (1 + exp(-{fit_result[1, 2]} " +
                f" * (x - {fit_result[1, 1]}))) + {fit_result[1, 3]})",
            ),
        )

    # elif fit_func == "arctan":
    #     trig_cor = cs.Correction(
    #     name=weight_name,
    #     description=description,
    #     version=1,
    #     inputs=[
    #         cs.Variable(name=variable_insts[0].name, type="real"),
    #     ],
    #     output=cs.Variable(name="weight", type="real", description="Multiplicative event weight"),
    #     data=cs.Category(
    #         nodetype="category",
    #         input=1,
    #         content=[
    #             cs.CategoryItem(
    #                 key=1,
    #                 value=cs.Formula(
    #                     nodetype="formula",
    #                     variables=[variable_insts[0].name],
    #                     parser="TFormula",
    #                     expression="(1+0.1*sin(x+0.3))/(1+0.07*sin(x+0.4))",
    #                 ),
    #             ),
    #         ]
    #     ),
    # )
    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="Custom trigger correction",
        corrections=[
            trig_cor,
            # trig_cor_up,
            # trig_cor_down,
        ],
    )

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
    return aj_plot_all(plot_config, style_config, fit_function=func_dict[fit_func], **kwargs), cset
