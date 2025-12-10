# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from alljets.plotting.aj_plot_all import aj_plot_all
from columnflow.plotting.plot_util import (
    prepare_style_config,
    remove_residual_axis,
    apply_process_settings,
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
--plot-function alljets.plotting.trigger_eff_plot.plot_efficiencies
"""


def plot_hist_matching(
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
    TODO.
    
    """
    n_processes = len(list(hists.keys()))
    keys = hists.keys()
    for i in range(len(list(keys))):
        if list(keys)[i] == "tt":
            tt_index = i
        elif ((list(keys)[i] == "qcd")):
            qcd_index = i
            label = f"QCD"
        elif ((list(keys)[i] == "qcd_est")):
            qcd_index = i
            label = f"Bkg. estimation"
        else:
            data_index = i

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    # hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)
    plot_config = OrderedDict()
    norm = np.ones_like(hists[0][list(hists[0].keys())[2]][0, :, 3].values())
    # for updating labels of individual selector steps

    plot_config[f"hist_qcd"] = {
        "method": "draw_hist",
        "ratio_method": "draw_stat_error_bands",
        "hist": hists[0][list(hists[0].keys())[qcd_index]][0, :, sum] + hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
        "kwargs": {
            "color": "#5790fc",
            "histtype": "fill",
            "label": label,
        },
        "ratio_kwargs": {
            "norm": (hists[0][list(hists[0].keys())[qcd_index]][0, :, sum] + hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1]).values(),
        },
    }
    

    plot_config["hist_wrong"] = {
        "method": "draw_hist",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
        "kwargs": {
            "color": "#f08181",
            "histtype": "fill",
            "label": f"{list(hists[0].keys())[tt_index].name}, wrong",
        },
    }

    plot_config["hist_unmatched"] = {
        "method": "draw_hist",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
        "kwargs": {
            "color": "#a33333",
            "histtype": "fill",
            "label": f"{list(hists[0].keys())[tt_index].name}, unmatched",
        },
    }
    
    plot_config["hist_correct"] = {
        "method": "draw_hist",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3],
        "kwargs": {
            "color": "#380000",
            "histtype": "fill",
            "label": f"{list(hists[0].keys())[tt_index].name}, correct",
        },
    }
    plot_config[f"hist_data"] = {
        "method": "draw_errorbars",
        "ratio_method": "draw_errorbars",
        "hist": hists[0][list(hists[0].keys())[data_index]][0, :, sum],
        "kwargs": {
            "label": f"{list(hists[0].keys())[data_index].name}",
        },
        "ratio_kwargs": {
            # "linestyle": "none",
            "norm": (hists[0][list(hists[0].keys())[qcd_index]][0, :, 1] + hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1]).values(),
        },
    }

    plot_config[f"hist_total_uncert"] = {
        "method": "draw_stat_error_bands",
        "hist": hists[0][list(hists[0].keys())[qcd_index]][0, :, sum] + hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    # default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    # default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)


def plot_hist_matching_MC(
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
    Matching for MC only
    
    """
    n_processes = len(list(hists.keys()))
    keys = hists.keys()
    for i in range(len(list(keys))):
        if list(keys)[i] == "tt":
            tt_index = i


    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    # hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)
    plot_config = OrderedDict()
    # for updating labels of individual selector steps
    

    plot_config["hist_wrong"] = {
        "method": "draw_hist",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
        "kwargs": {
            "color": "#f08181",
            "histtype": "fill",
            "label": f"{list(hists[0].keys())[tt_index].name}, wrong",
        },
    }

    plot_config["hist_unmatched"] = {
        "method": "draw_hist",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
        "kwargs": {
            "color": "#a33333",
            "histtype": "fill",
            "label": f"{list(hists[0].keys())[tt_index].name}, unmatched",
        },
    }
    
    plot_config["hist_correct"] = {
        "method": "draw_hist",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3],
        "kwargs": {
            "color": "#380000",
            "histtype": "fill",
            "label": f"{list(hists[0].keys())[tt_index].name}, correct",
        },
    }

    plot_config[f"hist_total_uncert"] = {
        "method": "draw_stat_error_bands",
        "hist": hists[0][list(hists[0].keys())[tt_index]][0, :, 3] + hists[0][list(hists[0].keys())[tt_index]][0, :, 2] + hists[0][list(hists[0].keys())[tt_index]][0, :, 1],
    }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    # default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    # default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)

def plot_hist_chi2cuts(
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
    Matching for MC only
    
    """
    n_processes = len(list(hists.keys()))
    keys = hists.keys()
    for i in range(len(list(keys))):
        if list(keys)[i] == "tt":
            tt_index = i


    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    # hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)
    plot_config = OrderedDict()
    # for updating labels of individual selector steps

    # Define cumulative hists:
    if (hists[0][list(hists[0].keys())[tt_index]].ndim == 4):
        hist_cor = hists[0][list(hists[0].keys())[tt_index]][0, :, 3, 1]
        hist_tot = hists[0][list(hists[0].keys())[tt_index]][0, :, sum, 1]
    else:
        hist_cor = hists[0][list(hists[0].keys())[tt_index]][0, :, 3]
        hist_tot = hists[0][list(hists[0].keys())[tt_index]][0, :, sum]

    # Make copies of the original histograms
    cumulative_hist_cor = hist_cor.copy()
    cumulative_hist_tot = hist_tot.copy()

    # Get views including flow bins
    orig_view_cor = hist_cor.view(flow=True)
    orig_view_tot = hist_tot.view(flow=True)
    cumul_view_cor = cumulative_hist_cor.view(flow=True)
    cumul_view_tot = cumulative_hist_tot.view(flow=True)

    # Fill cumulative histograms by reverse summing original values
    for i in range(len(orig_view_cor)):
        cumul_view_cor[i] = orig_view_cor[:(i + 1)].sum()
        cumul_view_tot[i] = orig_view_tot[:(i + 1)].sum()

    plot_config["hist_frac"] = {
        "method": "draw_errorbars",
        "hist": cumulative_hist_cor,
        "kwargs": {
            "error_type": "variance",
            "norm": cumulative_hist_tot.values(),
            "color": "#380000",
            "label": f"Fraction of correct assignments",
        },
    }
    plot_config["hist_twin"] = {
        "method": "draw_hist_twin",
        "hist": cumulative_hist_tot,
        "kwargs": {
            "label": f"Number of events",
        },
    }
    import IPython
    IPython.embed()
    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"$N_{corr}/N_{tot}$"
    # default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["gridspec_cfg"] = {"left": 0.10, "right": 0.85, "top": 0.95, "bottom": 0.1}
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 18
    default_style_config["annotate_cfg"]["text"] = ""
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)
