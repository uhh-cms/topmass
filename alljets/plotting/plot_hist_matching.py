# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law
from columnflow.plotting.plot_util import (apply_density, apply_variable_settings, prepare_style_config)
from columnflow.util import maybe_import
from alljets.plotting.aj_plot_all import aj_plot_all
from modules.columnflow.columnflow.plotting.plot_util import get_cms_label


hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)


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
    Plot matching types for data and MC, including QCD and data.

    This function visualizes the matching status of events (QCD, correct, wrong, unmatched, data)
    using cumulative stacking. The color scheme is based on previous analyses, e.g.
    https://arxiv.org/pdf/2302.01967v2.

    Example command to run the plot function using the QCD MC samples

    law run cf.PlotVariables1D --version v1 --configs 2017_v9\
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg,'data*','qcd*' \
    --variables reco_Top1_mass-fit_combination_type, --processes tt,data,qcd \
    --categories incl --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching

    To plot the data driven QCD estimation:
    Use the 'qcd_est' process instead of 'qcd' in the command above AND use --hist-hook qcd

    law run cf.PlotVariables1D --version v1 --configs 2017_v9 --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg,'data*'\
    --variables reco_Top1_mass-fit_combination_type --processes tt,qcd_est,data --categories sig \
    --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching --hist-hook qcd
    """
    # Identify indices for each process in the histogram keys
    keys = hists.keys()
    for i in range(len(list(keys))):
        if list(keys)[i] == "tt":
            tt_index = i
            tt_label = r"$t\bar{t}$"
        elif ((list(keys)[i] == "qcd")):
            qcd_index = i
            label = "QCD multijet"
        elif ((list(keys)[i] == "qcd_est")):
            qcd_index = i
            label = " Multijet est."
        else:
            data_index = i

    # Use the first variable instance for plotting
    variable_inst = variable_insts[0]

    # Apply variable and density settings to histograms
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    # hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)

    # Prepare the plot configuration dictionary
    plot_config = OrderedDict()

    # Extract base histograms for each matching type
    qcd_hist = hists[0][list(hists[0].keys())[qcd_index]][0, :, sum]
    correct_hist = hists[0][list(hists[0].keys())[tt_index]][0, :, 3]
    wrong_hist = hists[0][list(hists[0].keys())[tt_index]][0, :, 2]
    unmatched_hist = hists[0][list(hists[0].keys())[tt_index]][0, :, 1]

    # Build cumulative stacks for plotting
    stack_correct = correct_hist
    stack_wrong = stack_correct + wrong_hist
    stack_unmatched = stack_wrong + unmatched_hist
    stack_qcd = stack_unmatched + qcd_hist

    # Add each matching type to the plot config with appropriate color and stacking order
    plot_config["hist_correct"] = {
        "method": "draw_hist",
        "hist": stack_correct,
        "kwargs": {
            "color": "#cc0000",
            "histtype": "fill",
            "label": f"{tt_label} correct",
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 4,
        },
    }
    plot_config["hist_wrong"] = {
        "method": "draw_hist",
        "hist": stack_wrong,
        "kwargs": {
            "color": "#ff6666",
            "histtype": "fill",
            "label": f"{tt_label} wrong",
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 3,
        },
    }
    plot_config["hist_unmatched"] = {
        "method": "draw_hist",
        "hist": stack_unmatched,
        "kwargs": {
            "color": "#ffcccc",
            "histtype": "fill",
            "label": f"{tt_label} unmatched",
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 2,
        },
    }
    plot_config["hist_qcd"] = {
        "method": "draw_hist",
        "ratio_method": "draw_stat_error_bands",
        "hist": stack_qcd,
        "kwargs": {
            "color": "#ffff00" if "est." not in label else "#4da6ff",
            "histtype": "fill",
            "label": label,
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 1,
        },
        "ratio_kwargs": {
            "norm": (qcd_hist + correct_hist + wrong_hist + unmatched_hist).values(),
        },
    }
    # Add statistical uncertainty band for the total prediction
    plot_config["hist_total_uncert"] = {
        "method": "draw_stat_error_bands",
        "hist": (qcd_hist + correct_hist + wrong_hist + unmatched_hist),
        "kwargs": {
            "label": "MC stat. unc.",
            "zorder": 5,
        },
    }
    HIDE_DATA_MARKERS = {
        "fit_Top1_mass": (140, 195),
        "reco_Top1_mass": (140, 210),
        "reco_Top2_mass": (140, 210),
    }

    data_hist = hists[0][list(hists[0].keys())[data_index]][0, :, sum]

    hide_range = HIDE_DATA_MARKERS.get(variable_inst.name)

    if hide_range is not None:
        low, high = hide_range

        # Bin edges & centers
        edges = data_hist.axes[0].edges
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Mask bins inside the hide window
        mask = (centers >= low) & (centers <= high)

        # Copy to avoid modifying original histogram
        data_hist = data_hist.copy()

        # Zero values & variances → markers disappear
        data_hist.values()[mask] = 0.0
        data_hist.variances()[mask] = 0.0

    # Add data points with error bars
    plot_config["hist_data"] = {
        "method": "draw_errorbars",
        "ratio_method": "draw_errorbars",
        "hist": data_hist,
        "kwargs": {
            "label": f"{list(hists[0].keys())[data_index].name}",
            "zorder": 6,
        },
        "ratio_kwargs": {
            "norm": (
                hists[0][list(hists[0].keys())[qcd_index]][0, :, 1] +
                correct_hist + wrong_hist + unmatched_hist).values(),
        },
    }

    # Prepare and merge style configuration
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # Set legend and font sizes for clarity
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = False

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # Draw the plot using the aj_plot_all utility
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
    This function visualizes the matching status of MC events (correct, wrong, unmatched)
    using cumulative stacking. The color scheme is based on previous analyses, e.g.
    https://arxiv.org/pdf/2302.01967v2.

    law run cf.PlotVariables1D --version v1 --configs 2017_v9\
    --datasets tt_fh_powheg,tt_dl_powheg,tt_sl_powheg \
    --variables reco_Top1_mass-fit_combination_type \
    --categories incl --plot-function alljets.plotting.plot_hist_matching.plot_hist_matching_MC

    Example command to run the plot function. The matching information is stored in
    'fit_combination_type' column. Here, a 2D histogram is created with the
    information of the matching type on an additional axis.
    """

    # Find the index for the ttbar process in the histogram keys
    keys = hists.keys()
    for i in range(len(list(keys))):
        if list(keys)[i] == "tt":
            tt_index = i
            tt_label = r"$t\bar{t}$"

    divide_by_width = bool(kwargs.get("divide_by_width", False))

    # Use the first variable instance for plotting
    variable_inst = variable_insts[0]

    # Apply variable and density settings to histograms
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    if not divide_by_width:
        hists = apply_density(hists, density)

    # Prepare the plot configuration dictionary
    plot_config = OrderedDict()

    # Extract base histograms for each matching type
    correct_hist = hists[0][list(hists[0].keys())[tt_index]][0, :, 3]
    wrong_hist = hists[0][list(hists[0].keys())[tt_index]][0, :, 2]
    unmatched_hist = hists[0][list(hists[0].keys())[tt_index]][0, :, 1]

    if divide_by_width:
        bin_edges = hists[0][list(hists[0].keys())[tt_index]].axes[1].edges
        bin_widths = np.diff(bin_edges)

        correct_hist = correct_hist / bin_widths
        wrong_hist = wrong_hist / bin_widths
        unmatched_hist = unmatched_hist / bin_widths

    # Build cumulative stacks for plotting
    stack_correct = correct_hist
    stack_wrong = stack_correct + wrong_hist
    stack_unmatched = stack_wrong + unmatched_hist

    # Add each matching type to the plot config with appropriate color and stacking order
    plot_config["hist_correct"] = {
        "method": "draw_hist",
        "hist": stack_correct,
        "kwargs": {
            "color": "#cc0000",
            "histtype": "fill",
            "label": f"{tt_label} correct",
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 3,
        },
    }
    plot_config["hist_wrong"] = {
        "method": "draw_hist",
        "hist": stack_wrong,
        "kwargs": {
            "color": "#ff6666",
            "histtype": "fill",
            "label": f"{tt_label} wrong",
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 2,
        },
    }
    plot_config["hist_unmatched"] = {
        "method": "draw_hist",
        "hist": stack_unmatched,
        "kwargs": {
            "color": "#ffcccc",
            "histtype": "fill",
            "label": f"{tt_label} unmatched",
            "edgecolor": "black",
            "linewidth": 1,
            "zorder": 1,
        },
    }
    # Add statistical uncertainty band for the total MC prediction
    plot_config["hist_total_uncert"] = {
        "method": "draw_stat_error_bands",
        "hist": stack_unmatched,
    }

    # Prepare and merge style configuration
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # Set legend and font sizes for clarity
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["cms_label_cfg"]["llabel"] = get_cms_label(None, "simpw")["llabel"]
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # Draw the plot using the aj_plot_all utility
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
            "label": "Fraction of correct assignments",
        },
    }
    plot_config["hist_twin"] = {
        "method": "draw_hist_twin",
        "hist": cumulative_hist_tot,
        "kwargs": {
            "label": "Number of events",
        },
    }

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
