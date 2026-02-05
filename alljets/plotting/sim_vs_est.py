# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from modules.columnflow.columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import (
    prepare_style_config,
    apply_variable_settings,
    apply_density,
)
from modules.columnflow.columnflow.plotting.plot_util import get_cms_label

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)


def qcd_mc_vs_est(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot function to compare QCD estimation vs. simulation

    Example usage:

    law run cf.PlotVariables1D --version v1 \
    --configs 2017_v9 \
    --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*','qcd*' \
    --variables fit_Top1_mass_coarse \
    --selector-steps All,SignalOrBkgTrigger,BTag20,jet,HT \
    --processes tt,data,qcd,qcd_est
    --categories sig \
    --plot-function alljets.plotting.sim_vs_est.qcd_mc_vs_est \
    --hist-hook qcd

    """
    keys = hists.keys()
    for i in range(len(list(keys))):
        if ((list(keys)[i] == "qcd_est")):
            est_index = i
        elif ((list(keys)[i] == "qcd")):
            qcd_index = i

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    # hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)
    plot_config = OrderedDict()
    # for updating labels of individual selector steps
    plot_config["hist_mc"] = {
        "method": "draw_hist",
        "hist": hists[0][list(keys)[qcd_index]][0, :],
        "kwargs": {
            "histtype": "fill",
            "norm": hists[0][list(keys)[qcd_index]][0, sum].value,
            "label": "QCD multijet",
            "color": "#ffff00",
        },
        "ratio_method": "draw_stat_error_bands",
        "ratio_kwargs": {
            "norm": hists[0][list(keys)[qcd_index]][0, :].values(),
        },
    }

    plot_config["hist_data"] = {
        "method": "draw_errorbars",
        "ratio_method": "draw_errorbars",
        "hist": hists[0][list(keys)[est_index]][0, :],
        "kwargs": {
            "norm": hists[0][list(keys)[est_index]][0, sum].value,
            "label": "Bkg. estimation",
        },
        "ratio_kwargs": {
            # "linestyle": "none",
            "norm": hists[0][list(keys)[qcd_index]][0, :].values() *
            (hists[0][list(keys)[est_index]][0, sum].value / hists[0][list(keys)[qcd_index]][0, sum].value),
        },
    }

    plot_config["hist_uncert"] = {
        "method": "draw_stat_error_bands",
        "hist": hists[0][list(keys)[qcd_index]][0, :],
        "kwargs": {
            "norm": hists[0][list(keys)[qcd_index]][0, sum].value,
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"$\Delta N / N$"
    # default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)


def qcd_sig_vs_bkg_sel(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot function to compare the QCQ Multijet MC in the signal vs. background selection

    Example usage:
    law run cf.PlotVariables1D --version v1 \
    --configs 2017_v9 --datasets '*qcd*' \
    --selector-steps All,SignalOrBkgTrigger,BTag20,jet,HT \
    --producers default,kinFitMatch,trigger_prod \
    --variables fit_Top1_mass_coarse-secmaxbtag_type-trig_bits \
    --processes qcd \
    --categories fit_conv_leq_rbb \
    --plot-function alljets.plotting.sim_vs_est.qcd_sig_vs_bkg_sel \
    """

    keys = hists.keys()
    for i in range(len(list(keys))):
        if ((list(keys)[i] == "qcd")):
            qcd_index = i

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    # hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)
    plot_config = OrderedDict()
    # for updating labels of individual selector steps

    plot_config["hist_bkg"] = {
        "method": "draw_errorbars",
        "hist": hists[0][list(keys)[qcd_index]][0, :, 0, 3],
        "kwargs": {
            "marker": "^",
            "error_type": "variance",
            "norm": hists[0][list(keys)[qcd_index]][0, sum, 0, 3].value,
            "label": "Bkg. sel.",
        },
        "ratio_method": "draw_errorbars",
        "ratio_kwargs": {
            # "linestyle": "none",
            "error_type": "variance",
            "marker": "^",
            "norm": hists[0][list(keys)[qcd_index]][0, :, 1, 1].values() *
            (hists[0][list(keys)[qcd_index]][0, sum, 0, 3].value / hists[0][list(keys)[qcd_index]][0, sum, 1, 1].value),
        },
    }

    plot_config["hist_sig"] = {
        "method": "draw_hist",
        "hist": hists[0][list(keys)[qcd_index]][0, :, 1, 1],
        "kwargs": {
            "histtype": "fill",
            "norm": hists[0][list(keys)[qcd_index]][0, sum, 1, 1].value,
            "label": "Sig. sel.",
            "color": "#ffff00",
        },
        "ratio_method": "draw_stat_error_bands",
        "ratio_kwargs": {
            "norm": hists[0][list(keys)[qcd_index]][0, :, 1, 1].values(),
        },
    }

    plot_config["hist_uncert"] = {
        "method": "draw_stat_error_bands",
        "hist": hists[0][list(keys)[qcd_index]][0, :, 1, 1],
        "kwargs": {
            "norm": hists[0][list(keys)[qcd_index]][0, sum, 1, 1].value,
        },
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"$\Delta N / N$"
    # default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.5, 2.0)
    default_style_config["rax_cfg"]["ylabel"] = "Bkg. sel./Sig. sel."
    default_style_config["cms_label_cfg"]["llabel"] = get_cms_label(None, "simpw")["llabel"]
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)
