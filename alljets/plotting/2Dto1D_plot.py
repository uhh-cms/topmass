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

    num_bins = int(len((list(hists.values())[0])[0, :].values()))
    color_list = ["b", "r", "y", "g"]

    myhist_0 = hists[list(hists.keys())[0]]
    
    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist = np.zeros(len(myhist_0[:, 1].values()))
    int_variances = np.zeros(len(myhist_0[:, 1].values()))
    norm_variances = np.zeros(len(myhist_0[:, 1].values()))
    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0[i:, sum].values()))
        int_hist[i] = np.sum((myhist_0[i:, 0].values()))
        # int_variances[i] = (myhist_0[i:, 0])[sum].variance
        int_variances[i] = np.var((myhist_0[i:, 0].values()))
        norm_variances[i] = np.var((myhist_0[i:, sum].values()))

    hist_copy =  myhist_0[:, 0].copy()
    hist_copy.view().value = int_hist
    error = np.sqrt(int_variances*(1/int_norm**2) + norm_variances*((int_hist)/(int_norm**2))**2)

    #hist_copy.view().variance = int_hist(sn.UP, sn.ALL, unc=True)**2
    plot_config["hist"] = {
        "method": "draw_errorbars",
        "hist": hist_copy,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": error
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

    num_bins = int(len((list(hists.values())[0])[0, :].values()))
    color_list = ["b", "r", "y", "g"]

    myhist_0 = hists[list(hists.keys())[0]]
    
    int_norm = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_0 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_hist_2 = np.zeros(len(myhist_0[:, 1].values()))
    int_variances = np.zeros(len(myhist_0[:, 1].values()))
    int_variances_1 = np.zeros(len(myhist_0[:, 1].values()))
    int_variances_2 = np.zeros(len(myhist_0[:, 1].values()))
    norm_variances = np.zeros(len(myhist_0[:, 1].values()))

    for i in range(len(myhist_0[:, 1].values())):
        int_norm[i] = np.sum((myhist_0[i:, sum].values()))
        int_hist_0[i] = np.sum((myhist_0[i:, 0].values()))
        int_hist_1[i] = np.sum((myhist_0[i:, 1].values())) + int_hist_0[i]
        int_hist_2[i] = np.sum((myhist_0[i:, 2].values())) + int_hist_1[i]
        # int_variances[i] = (myhist_0[i:, 0])[sum].variance
        int_variances[i] = np.var((myhist_0[i:, 0].values()))

        int_variances_1[i] = np.var(np.concatenate((myhist_0[i:, 0].values(),myhist_0[i:, 1].values())))
        int_variances_2[i] = np.var(np.concatenate((myhist_0[i:, 0].values(),myhist_0[i:, 1].values(),myhist_0[i:, 2].values())))
        
        norm_variances[i] = np.var((myhist_0[i:, sum].values()))

    hist_copy_0 =  myhist_0[:, 0].copy()
    hist_copy_0.view().value = int_hist_0
    hist_copy_1 =  myhist_0[:, 1].copy()
    hist_copy_1.view().value = int_hist_1
    hist_copy_2 =  myhist_0[:, 2].copy()
    hist_copy_2.view().value = int_hist_2

    error = np.sqrt(int_variances*(1/int_norm)**2 + norm_variances*((int_hist_0)/(int_norm**2))**2)
    error_1 = np.sqrt(int_variances_1*(1/int_norm)**2 + norm_variances*((int_hist_1)/(int_norm**2))**2)
    error_2 = np.sqrt(int_variances_2*(1/int_norm)**2 + norm_variances*((int_hist_2)/(int_norm**2))**2)
    #hist_copy.view().variance = int_hist(sn.UP, sn.ALL, unc=True)**2
    #import IPython; IPython.embed()
    plot_config["hist_0"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "norm": int_norm,
            "yerr": error,
            "label": r"$\Delta R_{max} < 0.4$",
            
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "norm": int_norm,
            "yerr": error_1,
            "label": r"$\Delta R_{max} < 0.8$",
        },
    }
    plot_config["hist_2"] = {
        "method": "draw_errorbars",
        "hist": hist_copy_2,
        "kwargs": {
            "color": color_list[2],
            "norm": int_norm,
            "yerr": error_2,
            "label": r"$\Delta R_{max} < 1$",
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