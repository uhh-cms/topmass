# coding: utf-8

"""
Utilities and plotting helpers for trigger-efficiency studies.

This module provides several plotting functions used by the analysis
pipeline to compute and visualise trigger efficiencies and closures.

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


def convert_weightedmean_to_weight(h_mean: hist.Hist, include_flow: bool = True) -> hist.Hist:
    """
    Convert a `WeightedMean` storage histogram into `Weight` storage.

    Parameters
    - h_mean: histogram with `WeightedMean` storage
    - include_flow: whether to include under/overflow bins in the view

    Returns
    - h_weight: new histogram with `hist.storage.Weight` storage
    """

    # If the provided histogram is not WeightedMean, do nothing but warn.
    if not isinstance(h_mean._storage_type(), hist.storage.WeightedMean):
        logger.warning("Storage type is not WeightedMean.")
        return h_mean

    # Reconstruct axes preserving labels and flow options so the new
    # histogram has the same binning and metadata as the input.
    axes = []
    for ax in h_mean.axes:
        name = ax.name
        label = ax.label

        if isinstance(ax, hist.axis.Regular):
            # Regular axis: size, start, stop
            axes.append(
                hist.axis.Regular(
                    ax.size, ax.start, ax.stop, name=name, label=label, flow=ax.options.flow,
                ),
            )
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

    # Build the new histogram using Weight storage
    h_weight = hist.Hist(*axes, storage=hist.storage.Weight())

    # Use .view(flow=...) to include/exclude under/overflow bins
    mean_view = h_mean.view(flow=include_flow)
    weight_view = h_weight.view(flow=include_flow)

    # Iterate over all bin indices and copy the numeric sums
    # from the WeightedMean representation into the (sum,w2) pair
    for idx in np.ndindex(mean_view.shape):
        bin_content = mean_view[idx]
        weight_view[idx] = (bin_content.sum_of_weights, bin_content.sum_of_weights_squared)

    return h_weight


def sigmoid(x, L, x0, k, b):
    # Classic logistic-style turn-on curve used for trigger efficiencies.
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def arctan(x, L, x0, k, b):
    # Alternative smooth turn-on using a scaled arctan to resemble S-shape.
    y = L * (np.arctan(k * (x - x0) + np.pi * 0.5)) / np.pi + b
    return y


def binom_int(num, den, confint=0.68):
    # Compute Clopper-Pearson binomial confidence interval using
    # the Beta distribution quantiles. Returns (low, high) arrays for
    # the given successes `num` and trials `den`.
    from scipy.stats import beta

    quant = (1 - confint) / 2.0
    low = beta.ppf(quant, num, den - num + 1)
    high = beta.ppf(1 - quant, num + 1, den - num)

    # Replace NaNs caused by edge cases and ensure sensible bounds
    return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))


def eff_fit(
    h: np.Array,
    n: np.Array,
    x: np.Array,
    fit_function: Callable,
):
    """
    Fit a parametric function to binned efficiency points.

    Parameters
    - h: numerator counts (or sum-of-weights) per bin
    - n: denominator counts (or sum-of-weights) per bin
    - x: bin centres / mean x-values for the bins
    - fit_function: callable f(x, L, x0, k, b) returning model values

    Returns
    - [fit, chi2]: `fit` is the tuple returned by `curve_fit` and `chi2`
      is the reduced chi2.
    """

    # Prepare arrays: means (x), values (numerator), norm (denominator)
    means = x
    values = h
    norm = n

    # Efficiency and binomial uncertainties
    efficiency = np.nan_to_num(values / norm, nan=0)
    band_low, band_high = binom_int(values, norm)

    # Convert intervals into symmetric-ish errors for fitting
    error_low = np.asarray(efficiency - band_low)
    error_high = np.asarray(band_high - efficiency)

    # Handle degenerate edge-cases
    error_low[error_low == 1] = 0
    error_high[error_high == 1] = 0

    # Use the larger side of the asymmetric error as the fit sigma
    err_max = np.where(abs(error_low) < abs(error_high), abs(error_high), abs(error_low))
    err_max = np.where(err_max == 0, 1, err_max)

    # Initial parameter guesses for the fit
    L0 = np.max(efficiency)
    x0_0 = means[np.argmin(abs(efficiency - (L0 / 2)))]
    k0 = 4 / (means[np.argmin(abs(efficiency - (L0 * 9 / 10)))] - means[np.argmin(abs(efficiency - (L0 / 10)))])

    # Perform weighted non-linear least squares fit
    from scipy.optimize import curve_fit
    fit = curve_fit(fit_function, means, efficiency, [L0, x0_0, k0, 0], sigma=err_max, absolute_sigma=True)

    # Compute reduced chi2 when possible
    dof = (len(means) - len(signature(fit_function).parameters))
    if dof > 0:
        chi2 = np.sum((((fit_function(means, *fit[0]) - efficiency) ** 2) / (err_max ** 2)) / dof)
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
    Plot 1D trigger efficiency curves as a function of a single variable.

    This plot function relies on a specific setup
    - the hist_producer trig_all_weights
    - the corresponding producers no_norm and trigger_prod

    These are needed to ensure the correct histogram structure for the
    efficiency and ratio calculation.

    Example:
    ---------
    law run cf.PlotVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \
    --selector-steps All,BaseTrigger,BTag,HT --selector trigger_eff
    --producers no_norm,trigger_prod \
    --variables trigjet6_pt-trig_bits --hist-producer trig_all_weights \
    --processes data,tt --categories incl \
    --plot-function alljets.plotting.trigger_eff_closure_1D.plot_efficiencies --general-settings "bin_sel=1"

    For the ht_trigger variable, replace --variables and --selector-steps accordingly:
    --variables ht_trigger-trig_bits \
    --selector-steps All,BaseTrigger,SixJets,BTag,jet
    """

    hist_list = list(hists.values())
    if isinstance(hist_list[0]._storage_type(), hist.storage.WeightedMean):
        # Keep a copy of the original storage type for later bin-content access.
        hist_list_mean = hist_list.copy()
        for i in range(len(hist_list)):
            # Convert each histogram to Weight storage for efficiency arithmetic.
            hist_list[i] = convert_weightedmean_to_weight(hist_list[i])
    remove_residual_axis(hists, "shift")

    # Only the first variable is plotted here
    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    # Plot items are collected in an ordered dictionary to preserve draw order.
    plot_config = OrderedDict()

    # Efficiency bin selection for the trigger under study. The selected bin is used
    # In 2018, one MC trigger bin is treated differently
    # When using the HT380 Trigger in data (eff_bin 2) -> plot the HT400 trigger for MC (eff bin 1)
    eff_bin = int(kwargs.get("bin_sel", 0))
    eff_bin_data = eff_bin
    eff_bin_mc = 1 if (eff_bin == 2) and (config_inst.campaign.x.year == 2018) else eff_bin
    weighted = int(kwargs.get("unweighted", False))

    # Select the fitting shape used for the turn-on curve.
    fit_func = kwargs.get("fit", "sigmoid")
    func_dict = {
        "sigmoid": sigmoid,
        "arctan": arctan,
    }
    cut_vis = kwargs.get("cut_vis", None)

    if fit_func not in func_dict:
        TypeError("Unsupported function type")

    if eff_bin == 0:
        logger.warning("No bin selected, bin zero is used for efficiency calculation")

    if (len(hist_list) > 2):
        logger.warning("More than two input processes, only two are considered")

    # Build the trigger label list once so the selected bin can be annotated
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_bkg = np.array(config_inst.x.bkg_trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers, trigger_bkg))

    trig_alias = kwargs.get("alias", "None")

    # Allow a custom label for the selected trigger bin.
    if not trig_alias == "None":
        trigger_names[eff_bin_data] = trig_alias

    # Check if the user requested to combine two trigger bins into one for plotting..
    combine_triggers = kwargs.get("combine_triggers", None)

    if combine_triggers is not None:

        # Trigger combinations are only supported in the 2018 inclusive setup.
        if config_inst.campaign.x.year != 2018:
            raise ValueError("Combination of triggers should be only considered for 2018")

        if category_inst.name != "incl":
            raise ValueError("Combination of triggers should be only considered for inclusive category")

        # Accept several user-facing input formats for the pair of triggers.
        if isinstance(combine_triggers, str):
            combine_triggers = [int(x) for x in combine_triggers.split(";")]

        elif isinstance(combine_triggers, (list, tuple)):
            combine_triggers = list(combine_triggers)

        elif isinstance(combine_triggers, (int, float)):
            raise TypeError("combine_triggers must be two indices like 1;2")

        i, j = map(int, combine_triggers)

        # Create explicit histogram views
        data_hist = hist_list_mean[0].copy()   # only data is modified
        mc_hist = hist_list_mean[1]            # MC unchanged

        # Merge only the numerator trigger in data
        data_hist[..., i] = data_hist[..., i] + data_hist[..., j]

        eff_bin = i
    else:
        data_hist = hist_list_mean[0]
        mc_hist = hist_list_mean[1]

    # MC histogram used for scaling the ratio uncertainties in the ratio panel.
    myhist_1 = convert_weightedmean_to_weight(mc_hist[weighted, 0, :, eff_bin_mc])

    norm_hist_0 = np.array(convert_weightedmean_to_weight(data_hist[1, 0, :, 0]).values())
    norm_hist_1 = np.array(convert_weightedmean_to_weight(mc_hist[1, 0, :, 0]).values())

    # Build the efficiency and asymmetric binomial uncertainty for the data histogram.
    values = convert_weightedmean_to_weight(data_hist[weighted, 0, :, eff_bin_data]).values()
    norm = norm_hist_0
    efficiency = np.nan_to_num(values / norm, nan=0)
    band_low, band_high = binom_int(values, norm)
    yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / np.array(myhist_1.values())
    yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / np.array(myhist_1.values())
    yerror_low[yerror_low == 1] = 0
    yerror_low[yerror_low < 0] = 0
    yerror_high[yerror_high == 1] = 0
    yerror_high[yerror_high < 0] = 0

    yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                            yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
    yerrors = yerrors.T

    # plot config for ttbar MC
    plot_config["fit_1"] = {
        "method": "draw_efficiency_x",
        "hist": convert_weightedmean_to_weight(mc_hist[weighted, 0, :, eff_bin_mc]),
        "kwargs": {
            "x": mc_hist[weighted, 0, :, eff_bin_mc].values(),
            "color": "r",
            "linestyle": "none",
            "norm": norm_hist_1,
            "label": r"$t\bar{t}$",
            "capsize": 3,
        },
    }

    # plot config for data
    plot_config["fit_0"] = {
        "method": "draw_efficiency_x",
        "hist": convert_weightedmean_to_weight(data_hist[weighted, 0, :, eff_bin_data]),
        "kwargs": {
            "x": data_hist[weighted, 0, :, eff_bin_data].values(),
            "color": "b",
            "norm": norm_hist_0,
            "linestyle": "none",
            "label": f"{list(hists[0].keys())[0].name}",
            "capsize": 3,
        },
        "ratio_method": "draw_errorbars",
        "ratio_kwargs": {
            "error_type": "variance",
            "x": data_hist[weighted, 0, :, eff_bin_data].values(),
            "color": "b",
            "capsize": 3,
            "linestyle": "none",
            "norm": (myhist_1.values() * norm_hist_0) / norm_hist_1,
            "yerr": yerrors,
        },
    }

    # Optionally highlight region with a shaded vertical band.
    if cut_vis == "vspan":
        plot_config["cut_region"] = {
            "method": "draw_vspan",
            "kwargs": {
                "x_start": 30 if variable_inst == "trigjet6_pt" else 250,
                "x_end": 40 if variable_inst == "trigjet6_pt" else 450,
                "ymin": 0.0,
                "ymax": 0.7,
                "relative": True,
                "color": "grey",
                "alpha": 0.25,
                "zorder": 0,
            },
        }

    # setup style config
    # Start from the shared style template and then apply trigger-efficiency specific tweaks.
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"

    if variable_inst == "trigjet6_pt":
        default_style_config["ax_cfg"]["xlim"] = (30, 100)

    # Show a stacked legend title that reflects the merged trigger pair.
    if combine_triggers is not None:
        A = str(trigger_names[j])
        B = str(trigger_names[i])
        width = max(len(A), len(B)) + 4
        shift = (width - len(r"$\vee$")) // 4
        vee_line = ("     " * shift) + r"$\vee$"
        legend_title = "\n".join([A.center(width), vee_line.center(width), B.center(width)])
        default_style_config["legend_cfg"]["title"] = legend_title
    else:
        default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin_data]

    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 20
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["annotate_cfg"]["text"] = ""
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    return aj_plot_all(plot_config, style_config, fit_function=func_dict[fit_func], **kwargs)


def plot_efficiencies_with_uncert(
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
    Plot 1D trigger efficiency curves including trigger systematic uncertainties.

    This plot function computes and displays trigger efficiencies as a function
    of a single variable, together with their ratio and an uncertainty band
    derived from trigger up/down shifts.

    Important requirements:
    - Histograms must be produced with the hist_producer trig_all_weights
    - The corresponding producers must be used: no_norm and trigger_prod
    - Trigger shift sources trig_up/trig_down must be available
    - The selector must be the default selector e.g. default_trig_weight, not trigger_eff

    This plot is intended to be run with shifted histograms.

    Example command using the jet6_pt_trigger variable:
    ---------
    law run cf.PlotShiftedVariables1D --version v1 --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*' \
    --producers no_norm,trigger_prod \
    --variables trigjet6_pt-trig_bits \
    --hist-producer trig_all_weights \
    --processes data,tt \
    --categories incl \
    --plot-function alljets.plotting.trigger_eff_closure_1D.plot_efficiencies_with_uncert \
    --general-settings "bin_sel=1" \
    --shift-sources trig

    For the ht_trigger variable, replace --variables and --selector-steps accordingly:
    --variables ht_trigger-trig_bits \
    --selector-steps All,BaseTrigger,SixJets,BTag,jet
    """

    # Separate data from simulation by key so each can be handled explicitly.
    keys = list(hists.keys())
    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)

    # Cache the keys for the data and process histograms.
    for key in keys:
        if (key.name == "data"):
            data_key = key
        else:
            proc_key = key

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))

    if eff_bin == 0:
        logger.warning("No bin selected, bin zero is used for efficiency calculation")

    cut_vis = kwargs.get("cut_vis", None)

    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    eff_bin_data = eff_bin
    eff_bin_mc = (
        1 if (eff_bin == 2) and (config_inst.campaign.x.year == 2018)
        else eff_bin
    )

    combine_triggers = kwargs.get("combine_triggers", None)

    # setup plotting configs
    plot_config = {}

    # Histogram extraction for the data and process histograms, selecting the nominal shift.
    myhist_data_all_shifts = (hists[0][data_key])
    myhist_data = myhist_data_all_shifts[{"shift": "nominal"}]

    myhist_all_shifts = (hists[0][proc_key])
    myhist = myhist_all_shifts[{"shift": "nominal"}]

    data_hist = myhist_data.copy()   # only data is modified
    mc_hist = myhist

    if combine_triggers is not None:

        # Trigger combinations are only supported in the 2018 inclusive setup.
        if config_inst.campaign.x.year != 2018:
            raise ValueError("Combination of triggers should be only considered for 2018")

        if category_inst.name != "incl":
            raise ValueError("Combination of triggers should be only considered for inclusive category")

        # Accept several user-facing input formats for the pair of triggers.
        if isinstance(combine_triggers, str):
            combine_triggers = [int(x) for x in combine_triggers.split(";")]

        elif isinstance(combine_triggers, (list, tuple)):
            combine_triggers = list(combine_triggers)

        elif isinstance(combine_triggers, (int, float)):
            raise TypeError("combine_triggers must be two indices like 1;2")

        i, j = map(int, combine_triggers)

        # Merge only the numerator trigger in data
        data_hist[..., i] = data_hist[..., i] + data_hist[..., j]

        eff_bin_data = i

    norm_hist_data = np.array(convert_weightedmean_to_weight((data_hist)[1, :, 0]).values())
    norm_hist = np.array(convert_weightedmean_to_weight((mc_hist)[1, :, 0]).values())

    # errors for ratio
    # The uncertainty band is derived from the data binomial interval.
    values = convert_weightedmean_to_weight(data_hist[0, :, eff_bin]).values()
    norm = norm_hist_data

    efficiency = np.nan_to_num(values / norm, nan=0)
    band_low, band_high = binom_int(values, norm)
    yerror_low = np.asarray(efficiency - band_low) / efficiency
    yerror_high = np.asarray(band_high - efficiency) / efficiency
    yerror_low[yerror_low == 1] = 0
    yerror_high[yerror_high == 1] = 0
    yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                            yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
    yerrors = yerrors.T

    for i in range(len(list(hists[0].keys()))):
        if not keys[i].name == "data":
            plot_config["fit_1"] = {
                "method": "draw_efficiency_x",
                "hist": convert_weightedmean_to_weight((data_hist)[0, :, eff_bin_data]),
                "kwargs": {
                    "linestyle": "none",
                    "x": (data_hist)[0, :, eff_bin_data].values(),
                    "color": "b",
                    "norm": norm_hist_data,
                    "label": f"{data_key.name} ",
                    "capsize": 3,
                },
                "ratio_method": "draw_errorbars",
                "ratio_kwargs": {
                    "yerr": yerrors,
                    "x": (data_hist)[0, :, eff_bin_data].values(),
                    "color": "b",
                    "capsize": 3,
                    "linestyle": "none",
                    "norm": (
                        convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]).values() *
                        norm_hist_data) / norm_hist,
                },
            }
        else:
            plot_config["hist1"] = {
                "method": "draw_efficiency_x",
                "hist": convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]),
                "kwargs": {
                    "x": mc_hist[0, :, eff_bin_mc].values(),
                    "color": "r",
                    "norm": norm_hist,
                    "label": r"$t\bar{t}$",
                    "capsize": 3,
                },
            }

    # Compute the trigger systematic uncertainty band using the up/down shifted histograms.
    low = convert_weightedmean_to_weight(myhist_all_shifts[{"shift": "trig_down"}])[0, :, eff_bin_mc].values()
    high = convert_weightedmean_to_weight(myhist_all_shifts[{"shift": "trig_up"}])[0, :, eff_bin_mc].values()

    # Build a simple up/down envelope around the nominal MC efficiency.
    errors_low = abs((low / norm_hist) -
                     (convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]).values() / norm_hist))
    errors_high = abs((high / norm_hist) -
                      (convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]).values() / norm_hist))
    eff = convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]).values() / norm_hist

    # Add the systematic uncertainty band to the plot configuration.
    plot_config["syst"] = {
        "method": "draw_error_bands",
        "ratio_method": "draw_error_bands",
        "hist": convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]),
        "kwargs": {
            "bottom": (eff - errors_low),
            "height": errors_low + errors_high,
            "norm": norm_hist,
        },
        "ratio_kwargs": {
            "height": ((errors_low + errors_high) / eff),
            "bottom": (eff - errors_low) / eff,
            "norm": convert_weightedmean_to_weight(mc_hist[0, :, eff_bin_mc]).values(),
        },
    }

    # Shade the threshold interval
    if cut_vis == "vspan":
        plot_config["cut_region"] = {
            "method": "draw_vspan",
            "kwargs": {
                "x_start": 30 if variable_inst == "trigjet6_pt" else 250,
                "x_end": 40 if variable_inst == "trigjet6_pt" else 450,
                "ymin": 0.0,
                "ymax": 0.7,
                "relative": True,
                "color": "grey",
                "alpha": 0.25,
                "zorder": 0,
            },
        }

    # setup style config
    # Keep the visual language aligned with the other efficiency plots.
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    if variable_inst == "trigjet6_pt":
        default_style_config["ax_cfg"]["xlim"] = (30, 100)

    # Show a stacked legend title that reflects the merged trigger pair.
    if combine_triggers is not None:
        A = str(trigger_names[j])
        B = str(trigger_names[i])
        width = max(len(A), len(B)) + 4
        shift = (width - len(r"$\vee$")) // 4
        vee_line = ("     " * shift) + r"$\vee$"
        legend_title = "\n".join([A.center(width), vee_line.center(width), B.center(width)])
        default_style_config["legend_cfg"]["title"] = legend_title
    else:
        default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin_data]

    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 20
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["annotate_cfg"]["text"] = ""
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    return aj_plot_all(plot_config, style_config, **kwargs)


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
    Produce trigger efficiency curves and derive a trigger correction weight.

    This function is used by the ProduceTriggerWeight task to compute a 1D trigger
    efficiency curve as a function of a single variable for tt̄ MC and data.
    The efficiencies are fitted, and a trigger correction weight is derived from
    the ratio of the data and MC fits and stored as a correctionlib object.

    The function returns the efficiency plot together with the corresponding
    correctionlib CorrectionSet.
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

    # Extract input parameters from kwargs
    eff_bin = int(kwargs.get("bin_sel", 0))
    eff_bin_data = eff_bin
    eff_bin_mc = 1 if (eff_bin == 2) and (config_inst.campaign.x.year == 2018) else eff_bin
    weighted = int(kwargs.get("unweighted", False))
    fit_func = kwargs.get("fit", "sigmoid")
    func_dict = {
        "sigmoid": sigmoid,
        "arctan": arctan,
    }
    cut_vis = kwargs.get("cut_vis", None)

    if fit_func not in func_dict:
        TypeError("Unsupported function type")

    if eff_bin == 0:
        logger.warning("No bin selected, bin zero is used for efficiency calculation")

    if (len(hist_list) > 2):
        logger.warning("More than two input processes, only two are considered")

    # Build the trigger label list once so the selected bin can be annotated
    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    # Allow a custom label for the selected trigger bin
    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    # Check if the user requested to combine two trigger bins into one for plotting..
    combine_triggers = kwargs.get("combine_triggers", None)

    if combine_triggers is not None:

        # safety checks
        if config_inst.campaign.x.year != 2018:
            raise ValueError("Combination only supported for 2018")

        if category_inst.name != "incl":
            raise ValueError("Combination only supported for inclusive category")

        # parse input
        if isinstance(combine_triggers, str):
            combine_triggers = [int(x) for x in combine_triggers.split(";")]
        elif isinstance(combine_triggers, (list, tuple)):
            combine_triggers = list(combine_triggers)
        elif isinstance(combine_triggers, (int, float)):
            raise TypeError("combine_triggers must be two indices like 1;2")

        i, j = map(int, combine_triggers)

        data_hist = hist_list_mean[0].copy()
        mc_hist = hist_list_mean[1]

        # merge ONLY numerator bin in data
        data_hist[..., i] = data_hist[..., i] + data_hist[..., j]

        eff_bin = i
    else:
        # default case: no combination
        data_hist = hist_list_mean[0]
        mc_hist = hist_list_mean[1]

    # MC histogram used for scaling the ratio uncertainties in the ratio panel
    myhist_1 = convert_weightedmean_to_weight(mc_hist[weighted, 0, :, eff_bin_mc])

    # Convert the WeightedMean histograms to Weight storage for efficiency calculations
    norm_hist_0 = np.array(convert_weightedmean_to_weight(data_hist[1, 0, :, 0]).values())
    norm_hist_1 = np.array(convert_weightedmean_to_weight(mc_hist[1, 0, :, 0]).values())

    # Fitting sigmoid or other function to efficiencies
    # Use eff_bin_data for data (j=0) and eff_bin_mc for MC (j=1) to respect the 2018 special case
    fit_result = np.zeros((len(hist_list_mean), 4))
    chi2 = np.zeros((len(hist_list_mean)))
    variances = np.zeros((len(hist_list_mean), 4, 4))
    for j in range(len(hist_list_mean)):
        current_hist = data_hist if j == 0 else mc_hist
        bin_to_use = eff_bin_data if j == 0 else eff_bin_mc
        num = convert_weightedmean_to_weight(current_hist[weighted, 0, :, bin_to_use]).values()
        den = convert_weightedmean_to_weight(current_hist[weighted, 0, :, 0]).values()
        raw = current_hist[weighted, 0, :, bin_to_use].values()
        fit = eff_fit(num, den, raw, fit_function=func_dict[fit_func])
        fit_result[j] = fit[0][0]
        variances[j] = fit[0][1]
        chi2[j] = fit[1]

    # Compute the trigger efficiency and asymmetric binomial uncertainty for the data histogram
    values = convert_weightedmean_to_weight(data_hist[weighted, 0, :, eff_bin_data]).values()
    norm = norm_hist_0
    efficiency = np.nan_to_num(values / norm, nan=0)
    band_low, band_high = binom_int(values, norm)
    yerror_low = np.asarray(efficiency - band_low) * norm_hist_1 / np.array(myhist_1.values())
    yerror_high = np.asarray(band_high - efficiency) * norm_hist_1 / np.array(myhist_1.values())
    yerror_low[yerror_low == 1] = 0
    yerror_high[yerror_high == 1] = 0
    yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                            yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
    yerrors = yerrors.T

    # ttbar plot config (uses eff_bin_mc)
    plot_config["fit_1"] = {
        "method": "draw_efficiency_with_fit",
        "hist": convert_weightedmean_to_weight(mc_hist[weighted, 0, :, eff_bin_mc]),
        "fit_result": fit_result[1],
        "kwargs": {
            "x": mc_hist[weighted, 0, :, eff_bin_mc].values(),
            "color": "r",
            "norm": norm_hist_1,
            "label": r"$t\bar{t}$",
            "capsize": 3,
        },
    }

    # data plot config (uses eff_bin_data)
    plot_config["fit_0"] = {
        "method": "draw_efficiency_with_fit",
        "hist": convert_weightedmean_to_weight(data_hist[weighted, 0, :, eff_bin_data]),
        "fit_result": fit_result[0],
        "kwargs": {
            "x": data_hist[weighted, 0, :, eff_bin_data].values(),
            "color": "b",
            "norm": norm_hist_0,
            "label": "data",
            "capsize": 3,
        },
        "ratio_method": "draw_ratio_of_fit",
        "ratio_kwargs": {
            "x": data_hist[weighted, 0, :, eff_bin_data].values(),
            "color": "b",
            "capsize": 3,
            "linestyle": "none",
            "norm": (myhist_1.values() * norm_hist_0) / norm_hist_1,
            "yerr": yerrors,
            "fit_norm": fit_result[1],
        },
    }

    # Optionally highlight region with a shaded vertical band.
    if cut_vis == "vspan":
        plot_config["cut_region"] = {
            "method": "draw_vspan",
            "kwargs": {
                "x_start": 30 if variable_inst == "trigjet6_pt" else 250,
                "x_end": 40 if variable_inst == "trigjet6_pt" else 450,
                "ymin": 0.0,
                "ymax": 0.7,
                "relative": True,
                "color": "grey",
                "alpha": 0.25,
                "zorder": 0,
            },
        }

    # Generate a correctionlib CorrectionSet for the trigger efficiency weight
    import correctionlib.schemav2
    weight_name = kwargs.get("name", "trig_cor")
    description = (
        f"Trigger correction using {trigger_names[0]}" +
        f" as the base trigger and {trigger_names[eff_bin]} as the signal trigger"
    )

    # Create a correctionlib Correction object for the trigger efficiency weight
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

    # Build the correctionlib CorrectionSet
    cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        description="Custom trigger correction",
        corrections=[
            trig_cor,
        ],
    )

    # setup style config
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    if variable_inst == "trigjet6_pt":
        default_style_config["ax_cfg"]["xlim"] = (30, 100)

    # Show a stacked legend title that reflects the merged trigger pair.
    if combine_triggers is not None:
        A = str(trigger_names[j])
        B = str(trigger_names[i])
        width = max(len(A), len(B)) + 4
        shift = (width - len(r"$\vee$")) // 4
        vee_line = ("     " * shift) + r"$\vee$"
        legend_title = "\n".join([A.center(width), vee_line.center(width), B.center(width)])
        default_style_config["legend_cfg"]["title"] = legend_title
    else:
        default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin_data]

    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 20
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["annotate_cfg"]["text"] = ""
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)
    return aj_plot_all(plot_config, style_config, fit_function=func_dict[fit_func], **kwargs), cset
