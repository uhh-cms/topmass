# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

import scinum as sn

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_style_config,
    remove_residual_axis,
    get_position,
    apply_variable_settings,
    apply_density_to_hists,
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
--plot-function alljets.plotting.trigger_eff_plot.plot_efficiencies
"""
color_list = ["#5790fc", "#f89c20", "#e42536", "#964a8b"]

def binom_int(num, den, confint=0.68):
    from scipy.stats import beta
    quant = (1 - confint) / 2.
    low = beta.ppf(quant, num, den - num + 1)
    high = beta.ppf(1 - quant, num + 1, den - num)
    return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))


def calculate_errors(values, norm):
    # getting error bars
    band_low, band_high = binom_int(values, norm)
    error_low = np.asarray((values / norm) - band_low)
    error_high = np.asarray(band_high - (values / norm))

    # removing large errors in empty bins
    error_low[error_low == 1] = 0
    error_high[error_high == 1] = 0

    # stacking errors
    errors = np.concatenate((error_low.reshape(error_low.shape[0], 1),
                             error_high.reshape(error_high.shape[0], 1)), axis=1,
                            )
    return abs(errors.T)


def hist_to_num(h: hist.Histogram, unc_name=str(sn.DEFAULT)) -> sn.Number:
    return sn.Number(h.values(flow=True), {unc_name: h.variances(flow=True)**0.5})


def plot_2Dto1D(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    num_bins = int(len((list(hists.values())[0])[0, :].values()))
    color_list = ["b", "r", "y", "g"]
    for i in range(num_bins):
        myhist_0 = hists[list(hists.keys())[0]]
        text = (
            f"{list(hists.keys())[0].name}, "
            f"{myhist_0[0, :].axes[0].label}:"
            f"{(myhist_0[0, :].axes.edges[0][i])} to "
            f"{(myhist_0[0, :].axes.edges[0][i+1])}"
        )
        plot_config[f"hist_{i}"] = {
            "method": "draw_errorbars",
            "ratio_method": "draw_hist",
            "hist": myhist_0[:, i],
            "kwargs": {
                "color": color_list[i],
                "label": text,
            },
        }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Counts"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)


def plot_2Dto1D_ratio(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    color_list = ["b", "r", "y", "g"]

    myhist_0 = hists[list(hists.keys())[0]]

    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0[i:, sum].values()))
        int_hist[i] = np.sum((myhist_0[i:, 1].values()))

    hist_copy = myhist_0[:, 0].copy()
    hist_copy.view().value = int_hist
    # import IPython; IPython.embed()
    # hist_copy.view().variance = int_hist(sn.UP, sn.ALL, unc=True)**2
    label = myhist_0.axes[1].label
    plot_config["hist"] = {
        "method": "draw_errorbars",
        "hist": hist_copy,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist, int_norm),
            "label": label,
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Ratio"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)


def plot_2Dto1D_ratio_3plots(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    color_list = ["b", "r", "y", "g"]

    myhist_0 = hists[list(hists.keys())[0]]

    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_x = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0[i:, :].values()))
        int_hist_x[i] = np.sum((myhist_0[i, :].values())) / np.sum((myhist_0.values()))
        int_hist_0[i] = np.sum((myhist_0[i:, 0].values()))
        int_hist_1[i] = np.sum((myhist_0[i:, 1].values())) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0[i:, 2].values())) + int_hist_1[i]

    # import IPython; IPython.embed()
    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2
    hist_copy_x = myhist_0[:, 0].copy()
    hist_copy_x.view().value = int_hist_x

    label = myhist_0.axes[1].label
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_0, int_norm),
            "label": label + r"$< 0.4$",
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_1, int_norm),
            "label": label + r"$ < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_2, int_norm),
            "label": label + r"$< 1$",
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    x_label = myhist_0.axes[0].label
    default_style_config["ax_cfg"]["ylabel"] = "Ratio"
    default_style_config["ax_cfg"]["xlabel"] = x_label + " Cut"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    ax1 = ax.twinx()
    plot_kwargs = {
        "ax": ax1,
        "color": color_list[3],
        "label": r"$p_{T,t}$",
        "histtype": "fill",
        "alpha": 0.25,
    }

    hist_copy_x.plot1d(**plot_kwargs)
    ax1_ymin = 0.0000001
    ax1_ymax = get_position(
        ax1_ymin,
        ax1.get_ylim()[1],
        factor=1 / (1 - kwargs.get("whitespace_fraction", 0.3)),
        logscale=False,
    )
    ax1.set(
        ylim=(ax1_ymin, ax1_ymax),
        ylabel=r"$\Delta N/N$",
        yscale="linear",
    )
    plt.tight_layout()
    return fig, (ax, ax1)


def plot_2Dto1D_BinRatio_3plots(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    color_list = ["b", "r", "y", "g"]

    myhist_0 = hists[list(hists.keys())[0]]

    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_x = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0.view().value[i:, :]))
        int_hist_x[i] = np.sum((myhist_0[i, :].values())) / np.sum((myhist_0.values()))
        int_hist_0[i] = np.sum((myhist_0.view().value[i, 0]))
        int_hist_1[i] = np.sum((myhist_0.view().value[i, 1])) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0.view().value[i, 2])) + int_hist_1[i]

    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2
    hist_copy_x = myhist_0[:, 0].copy()
    hist_copy_x.view().value = int_hist_x

    # hist_copy.view().variance = int_hist(sn.UP, sn.ALL, unc=True)**2
    # import IPython; IPython.embed()
    label = myhist_0.axes[1].label
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_0, int_norm),
            "label": label + r"$< 0.4$",
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_1, int_norm),
            "label": label + r"$ < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_2, int_norm),
            "label": label + r"$< 1$",
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Ratio"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    ax1 = ax.twinx()
    plot_kwargs = {
        "ax": ax1,
        "color": color_list[3],
        "histtype": "fill",
        "alpha": 0.25,
    }

    hist_copy_x.plot1d(**plot_kwargs)
    ax1_ymin = 0.0000001
    ax1_ymax = get_position(
        ax1_ymin,
        ax1.get_ylim()[1],
        factor=1 / (1 - kwargs.get("whitespace_fraction", 0.3)),
        logscale=False,
    )
    ax1.set(
        ylim=(ax1_ymin, ax1_ymax),
        ylabel=r"$\Delta N/N$",
        yscale="linear",
    )
    plt.tight_layout()
    return fig, (ax, ax1)


def plot_2Dto1D_ratio_3plots_test(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    color_list = ["b", "r", "y", "g", "c", "m", "k"]

    myhist_0 = hists[list(hists.keys())[0]]

    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 0].values())):
        int_norm[i] = np.sum((myhist_0[i:, :].values()))
        int_hist_0[i] = np.sum((myhist_0[i:, 1].values())) + np.sum((myhist_0[i:, 3].values()))
        int_hist_1[i] = np.sum((myhist_0[i:, 2].values())) + np.sum((myhist_0[i:, 3].values()))
        int_hist_2[i] = np.sum((myhist_0[i:, 3].values()))

    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2

    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_0, int_norm),
            "label": r"$t: \Delta R< 1.0$",

        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_1, int_norm),
            "label": r"$\overline{t}: \Delta R< 1.0$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_2, int_norm),
            "label": r" $t +\overline{t}: \Delta R< 1.0$",
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"Fraction"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)


def plot_bin_ratio(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    color_list = ["b", "r", "y", "g"]

    myhist_0 = hists[list(hists.keys())[0]]

    int_hist = np.zeros(len(myhist_0[:, 1].values()))
    int_norm = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_hist[i] = np.sum((myhist_0[i, 1].value))
        int_norm[i] = np.sum((myhist_0[i, :2].values()))

    hist_copy = myhist_0[:, 0].copy()
    hist_copy.view().value = int_hist

    # hist_copy.view().variance = int_hist(sn.UP, sn.ALL, unc=True)**2
    plot_config["hist"] = {
        "method": "draw_errorbars",
        "hist": hist_copy,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist, int_norm),
            # "label": label,
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = "Ratio"
    # default_style_config["legend_cfg"]["ncol"] = 1
    # default_style_config["legend_cfg"]["title_fontsize"] = 17
    # default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)


def plot_2Dto1D_3plots_integral(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_x = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0[i:, :].values()))
        int_hist_x[i] = np.sum((myhist_0[i:, :].values()))
        int_hist_0[i] = np.sum((myhist_0[i:, 0].values()))
        int_hist_1[i] = np.sum((myhist_0[i:, 1].values())) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0[i:, 2].values())) + int_hist_1[i]

    # import IPython; IPython.embed()
    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2
    hist_copy_x = myhist_0[:, 0].copy()
    hist_copy_x.view().value = int_hist_x

    label = myhist_0.axes[1].label
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0 * 100,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_0, int_norm) * 100,
            "label": label + r"$< 0.4$",
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1 * 100,
        "kwargs": {
            "color": color_list[1],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_1, int_norm) * 100,
            "label": label + r"$ < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2 * 100,
        "kwargs": {
            "color": color_list[2],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_2, int_norm) * 100,
            "label": label + r"$< 1$",
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    x_label = myhist_0.axes[0].label
    default_style_config["ax_cfg"]["ylabel"] = "Proportion in %"
    default_style_config["ax_cfg"]["ylim"] = (0, 100 / (1 - kwargs.get("whitespace_fraction", 0.3)))
    x_parts = x_label.split('/ ')
    default_style_config["ax_cfg"]["xlabel"] = x_parts[0] + " Cut" + " / " + x_parts[1]
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    ax1 = ax.twinx()
    plot_kwargs = {
        "ax": ax1,
        "color": color_list[3],
        "histtype": "fill",
        "alpha": 0.25,
    }

    hist_copy_x.plot1d(**plot_kwargs)
    ax1.set(
        ylim=(0, np.max(int_hist_x) / (1 - kwargs.get("whitespace_fraction", 0.3))),
        ylabel=r"Number of events",
        yscale="linear",
    )

    plt.tight_layout()
    return fig, (ax, ax1)


def plot_2Dto1D_3plots_differential(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_x = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0[i, :].values()))
        int_hist_x[i] = np.sum((myhist_0[i, :].values()))
        int_hist_0[i] = np.sum((myhist_0[i, 0].value))
        int_hist_1[i] = np.sum((myhist_0[i, 1].value)) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0[i, 2].value)) + int_hist_1[i]

    # import IPython; IPython.embed()
    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2
    hist_copy_x = myhist_0[:, 0].copy()
    hist_copy_x.view().value = int_hist_x

    label = myhist_0.axes[1].label
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0 * 100,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_0, int_norm) * 100,
            "label": label + r"$< 0.4$",
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1 * 100,
        "kwargs": {
            "color": color_list[1],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_1, int_norm) * 100,
            "label": label + r"$ < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2 * 100,
        "kwargs": {
            "color": color_list[2],
            "norm": int_norm,
            "yerr": calculate_errors(int_hist_2, int_norm) * 100,
            "label": label + r"$< 1$",
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    x_label = myhist_0.axes[0].label
    default_style_config["ax_cfg"]["ylabel"] = "Proportion in %"
    default_style_config["ax_cfg"]["ylim"] = (0, 100 / (1 - kwargs.get("whitespace_fraction", 0.3)))
    default_style_config["ax_cfg"]["xlabel"] = x_label
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    ax1 = ax.twinx()
    plot_kwargs = {
        "ax": ax1,
        "color": color_list[3],
        "histtype": "fill",
        "alpha": 0.25,
    }

    hist_copy_x.plot1d(**plot_kwargs)

    ax1.set(
        ylim=(0, np.max(int_hist_x) / (1 - kwargs.get("whitespace_fraction", 0.3))),
        ylabel=r"Number of events",
        yscale="linear",
    )

    plt.tight_layout()
    return fig, (ax, ax1)


def plot_2Dto1D_3plots_integral_absolut(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_x = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_hist_x[i] = np.sum((myhist_0[i:, :].values()))
        int_hist_0[i] = np.sum((myhist_0[i:, 0].values()))
        int_hist_1[i] = np.sum((myhist_0[i:, 1].values())) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0[i:, 2].values())) + int_hist_1[i]

    # import IPython; IPython.embed()
    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2
    hist_copy_x = myhist_0[:, 0].copy()
    hist_copy_x.view().value = int_hist_x

    label = myhist_0.axes[1].label
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "yerr": calculate_errors(int_hist_0, int_hist_x),
            "label": label + r"$< 0.4$",
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "yerr": calculate_errors(int_hist_1, int_hist_x),
            "label": label + r"$ < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "yerr": calculate_errors(int_hist_2, int_hist_x),
            "label": label + r"$< 1$",
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    x_label = myhist_0.axes[0].label
    default_style_config["ax_cfg"]["ylabel"] = "Number of events"
    x_parts = x_label.split('/ ')
    default_style_config["ax_cfg"]["xlabel"] = x_parts[0] + " Cut" + " / " + x_parts[1]
    default_style_config["ax_cfg"]["ylim"] = (0, 3 * 10**5)
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    plot_kwargs = {
        "ax": ax,
        "color": color_list[3],
        "histtype": "fill",
        "alpha": 0.25,
    }
    hist_copy_x.plot1d(**plot_kwargs)

    plt.tight_layout()
    return fig, ax


def plot_2Dto1D_3plots_differential_absolut(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    # int_hist_x = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_x = np.array([])
    for i in range(len(myhist_0[:, 1].values())):
        int_hist_x = np.append(int_hist_x, np.sum((myhist_0[i, :].values())))
        int_hist_0[i] = np.sum((myhist_0[i, 0].value))
        int_hist_1[i] = np.sum((myhist_0[i, 1].value)) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0[i, 2].value)) + int_hist_1[i]

    # import IPython; IPython.embed()
    hist_copy_0 = myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 = myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 = myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2
    hist_copy_x = myhist_0[:, 0].copy()
    hist_copy_x.view().value = int_hist_x

    label = myhist_0.axes[1].label
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "yerr": calculate_errors(int_hist_0, int_hist_x),
            "label": label + r"$< 0.4$",
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "yerr": calculate_errors(int_hist_1, int_hist_x),
            "label": label + r"$ < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "yerr": calculate_errors(int_hist_2, int_hist_x),
            "label": label + r"$< 1$",
        },
    }
    plot_config["hist_x"] = {
        "method": "draw_hist",
        "hist": hist_copy_x,
        "kwargs": {
            "color": color_list[3],
            "histtype": "fill",
            "alpha": 0.25,
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    x_label = myhist_0.axes[0].label
    default_style_config["ax_cfg"]["ylabel"] = "Number of events"
    default_style_config["ax_cfg"]["xlabel"] = x_label
    default_style_config["ax_cfg"]["ylim"] = (0, np.max(int_hist_x) / (1 - kwargs.get("whitespace_fraction", 0.3)))
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    plt.tight_layout()
    return fig, (ax, )


def plot_2Dto1D_3plots_integral_4(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    hist_0 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_1 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_2 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_all = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_norm = np.zeros(len(myhist_0[:, 0, 0, 0].values()))

    for i in range(len(myhist_0[:, 0, 0, 0].values())):
        hist_norm[i] = np.sum(myhist_0[i:, :, :, :].values())
        hist_0[i] = np.sum(myhist_0[i:, 1, :, :].values())
        hist_1[i] = np.sum(myhist_0[i:, :, 1, :].values())
        hist_2[i] = np.sum(myhist_0[i:, :, :, 1].values())
        hist_all[i] = np.sum(myhist_0[i:, :, :, :].values())

    hist_copy_0 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_0.view().value = hist_0

    hist_copy_1 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_1.view().value = hist_1

    hist_copy_2 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_2.view().value = hist_2

    hist_copy_all = myhist_0[:, 1, 1, 1].copy()
    hist_copy_all.view().value = hist_all

    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0 * 100,
        "kwargs": {
            "color": color_list[0],
            "norm": hist_norm,
            "yerr": calculate_errors(hist_1, hist_norm) * 100,
            "label": myhist_0.axes[1].label,
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1 * 100,
        "kwargs": {
            "color": color_list[1],
            "norm": hist_norm,
            "yerr": calculate_errors(hist_1, hist_norm) * 100,
            "label": myhist_0.axes[2].label,
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2 * 100,
        "kwargs": {
            "color": color_list[2],
            "norm": hist_norm,
            "yerr": calculate_errors(hist_2, hist_norm) * 100,
            "label": myhist_0.axes[3].label,
        },
    }

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"Proportion in %"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["ax_cfg"]["ylim"] = (0, 100 / (1 - kwargs.get("whitespace_fraction", 0.3)))
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    x_label = myhist_0.axes[0].label
    x_parts = x_label.split('/ ')
    default_style_config["ax_cfg"]["xlabel"] = x_parts[0] + " Cut" + " / " + x_parts[1]
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)
    ax1 = ax.twinx()
    plot_kwargs = {
        "ax": ax1,
        "color": color_list[3],
        "histtype": "fill",
        "alpha": 0.25,
    }

    hist_copy_all.plot1d(**plot_kwargs)

    ax1.set(
        ylim=(0, np.max(hist_all) / (1 - kwargs.get("whitespace_fraction", 0.3))),
        ylabel=r"Number of events",
        yscale="linear",
    )

    plt.tight_layout()
    return fig, ax


def plot_2Dto1D_3plots_integral_absolut_4(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    hist_0 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_1 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_2 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_all = np.zeros(len(myhist_0[:, 0, 0, 0].values()))

    for i in range(len(myhist_0[:, 0, 0, 0].values())):
        hist_0[i] = np.sum(myhist_0[i:, 1, :, :].values())
        hist_1[i] = np.sum(myhist_0[i:, :, 1, :].values())
        hist_2[i] = np.sum(myhist_0[i:, :, :, 1].values())
        hist_all[i] = np.sum(myhist_0[i:, :, :, :].values())

    hist_copy_0 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_0.view().value = hist_0

    hist_copy_1 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_1.view().value = hist_1

    hist_copy_2 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_2.view().value = hist_2

    hist_copy_all = myhist_0[:, 1, 1, 1].copy()
    hist_copy_all.view().value = hist_all

    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "label": myhist_0.axes[1].label,
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "label": myhist_0.axes[2].label,
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "label": myhist_0.axes[3].label,
        },
    }
    plot_config["hist_all"] = {
        "method": "draw_hist",
        "hist": hist_copy_all,
        "kwargs": {
            "color": color_list[3],
            "histtype": "fill",
            "alpha": 0.25,
        },
    }

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"Number of events"
    default_style_config["legend_cfg"]["ncol"] = 1
    x_label = myhist_0.axes[0].label
    x_parts = x_label.split('/ ')
    default_style_config["ax_cfg"]["xlabel"] = x_parts[0] + " Cut" + " / " + x_parts[1]
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    plt.tight_layout()
    return fig, ax


def plot_2Dto1D_3plots_differential_4(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    hist_0 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_1 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_2 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_all = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_norm = np.zeros(len(myhist_0[:, 0, 0, 0].values()))

    for i in range(len(myhist_0[:, 0, 0, 0].values())):
        hist_norm[i] = np.sum(myhist_0[i, :, :, :].values())
        hist_0[i] = np.sum(myhist_0[i, 1, :, :].values())
        hist_1[i] = np.sum(myhist_0[i, :, 1, :].values())
        hist_2[i] = np.sum(myhist_0[i, :, :, 1].values())
        hist_all[i] = np.sum(myhist_0[i, :, :, :].values())

    hist_copy_0 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_0.view().value = hist_0

    hist_copy_1 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_1.view().value = hist_1

    hist_copy_2 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_2.view().value = hist_2

    hist_copy_all = myhist_0[:, 1, 1, 1].copy()
    hist_copy_all.view().value = hist_all

    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0 * 100,
        "kwargs": {
            "color": color_list[0],
            "norm": hist_norm,
            "yerr": calculate_errors(hist_1, hist_norm) * 100,
            "label": myhist_0.axes[1].label,
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1 * 100,
        "kwargs": {
            "color": color_list[1],
            "norm": hist_norm,
            "yerr": calculate_errors(hist_1, hist_norm) * 100,
            "label": myhist_0.axes[2].label,
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2 * 100,
        "kwargs": {
            "color": color_list[2],
            "norm": hist_norm,
            "yerr": calculate_errors(hist_2, hist_norm) * 100,
            "label": myhist_0.axes[3].label,
        },
    }

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"Proportion in %"
    default_style_config["ax_cfg"]["ylim"] = (0, 100 / (1 - kwargs.get("whitespace_fraction", 0.3)))
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True
    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)
    ax1 = ax.twinx()
    plot_kwargs = {
        "ax": ax1,
        "color": color_list[3],
        "histtype": "fill",
        "alpha": 0.25,
    }

    hist_copy_all.plot1d(**plot_kwargs)

    ax1.set(
        ylim=(0, np.max(hist_all) / (1 - kwargs.get("whitespace_fraction", 0.3))),
        ylabel=r"Number of events",
        yscale="linear",
    )

    plt.tight_layout()
    return fig, ax


def plot_2Dto1D_3plots_differential_absolut_4(
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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    myhist_0 = hists[list(hists.keys())[0]]

    hist_0 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_1 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_2 = np.zeros(len(myhist_0[:, 0, 0, 0].values()))
    hist_all = np.zeros(len(myhist_0[:, 0, 0, 0].values()))

    for i in range(len(myhist_0[:, 0, 0, 0].values())):
        hist_0[i] = np.sum(myhist_0[i, 1, :, :].values())
        hist_1[i] = np.sum(myhist_0[i, :, 1, :].values())
        hist_2[i] = np.sum(myhist_0[i, :, :, 1].values())
        hist_all[i] = np.sum(myhist_0[i, :, :, :].values())

    hist_copy_0 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_0.view().value = hist_0

    hist_copy_1 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_1.view().value = hist_1

    hist_copy_2 = myhist_0[:, 1, 1, 1].copy()
    hist_copy_2.view().value = hist_2

    hist_copy_all = myhist_0[:, 1, 1, 1].copy()
    hist_copy_all.view().value = hist_all

    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "label": myhist_0.axes[1].label,
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "label": myhist_0.axes[2].label,
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "label": myhist_0.axes[3].label,
        },
    }
    plot_config["hist_all"] = {
        "method": "draw_hist",
        "hist": hist_copy_all,
        "kwargs": {
            "color": color_list[3],
            "histtype": "fill",
            "alpha": 0.25,
        },
    }

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"Number of events"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, (ax,) = plot_all(plot_config, style_config, **kwargs)

    plt.tight_layout()
    return fig, ax