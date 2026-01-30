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
--plot-function alljets.plotting.multi_hist_in_1plot
"""

# CMS-colors
color_list = ["#5790fc", "#f89c20", "#e42536", "#964a8b"]


def hist_2D(
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
    Plots a 2D histogram in one plot. With one histogram for each axis in the 2D_Histogram
    """
    # Use the first variable instance for plotting
    variable_inst = variable_insts[0]

    # Apply variable and density settings to histograms
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)[0]

    # Prepare the plot configuration dictionary
    plot_config = OrderedDict()
    # Extract base histograms
    hists_base = hists[list(hists.keys())[0]]

    # Add each hist to the plot config
    plot_config["hist2"] = {
        "method": "draw_hist",
        "hist": hists_base.project(1),
        "kwargs": {
            "color": color_list[0],
            "histtype": "step",
            "label": hists_base.project(1).axes[0].label,
        },
    }

    plot_config["hist1"] = {
        "method": "draw_hist",
        "hist": hists_base.project(2),
        "kwargs": {
            "color": color_list[1],
            "histtype": "step",
            "label": hists_base.project(2).axes[0].label,
        },
    }

    # Prepare and merge style configuration
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # Set legend and font sizes for clarity
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["rax_cfg"]["xlabel"] = r"$ \lambda$"
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # Draw the plot using the aj_plot_all utility
    return aj_plot_all(plot_config, style_config, **kwargs)


def hist_3D(
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
    Plots a 2D histogram in one plot. With one histogram for each axis in the 2D_Histogram
    """
    # Use the first variable instance for plotting
    variable_inst = variable_insts[0]

    # Apply variable and density settings to histograms
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density(hists, density)[0]

    # Prepare the plot configuration dictionary
    plot_config = OrderedDict()
    # Extract base histograms
    hists_base = hists[list(hists.keys())[0]]

    # Add each hist to the plot config
    plot_config["hist0"] = {
        "method": "draw_hist",
        "hist": hists_base.project(1),
        "kwargs": {
            "color": color_list[1],
            "histtype": "step",
            "label": hists_base.project(1).axes[0].label,
        },
    }

    plot_config["hist1"] = {
        "method": "draw_hist",
        "hist": hists_base.project(2),
        "kwargs": {
            "color": color_list[0],
            "histtype": "step",
            "label": hists_base.project(2).axes[0].label,
        },
    }

    plot_config["hist2"] = {
        "method": "draw_hist",
        "hist": hists_base.project(3),
        "kwargs": {
            "color": color_list[2],
            "histtype": "step",
            "label": hists_base.project(3).axes[0].label,
        },
    }

    # Prepare and merge style configuration
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # Set legend and font sizes for clarity
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["ax_cfg"]["xlabel"] = r"$ \lambda$"
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    # Draw the plot using the aj_plot_all utility
    return aj_plot_all(plot_config, style_config, **kwargs)
