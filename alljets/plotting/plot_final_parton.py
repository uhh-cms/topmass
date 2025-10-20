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


def plot_final_parton(
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

    hist_0 = np.zeros(len(myhist_0[:, 0, 0].values()))
    hist_1 = np.zeros(len(myhist_0[:, 0, 0].values()))
    hist_all = np.zeros(len(myhist_0[:, 0, 0].values()))

    for i in range(len(myhist_0[:, 0, 0].values())):
        hist_0[i] = np.sum(myhist_0[i, 1, :].values())
        hist_1[i] = np.sum(myhist_0[i, :, 1].values())
        hist_all[i] = np.sum(myhist_0[i, :, :].values())

    hist_copy_0 = myhist_0[:, 1, 1].copy()
    hist_copy_0.view().value = hist_0

    hist_copy_1 = myhist_0[:, 1, 1].copy()
    hist_copy_1.view().value = hist_1

    hist_copy_all = myhist_0[:, 1, 1].copy()
    hist_copy_all.view().value = hist_all

    plot_config["hist_0"] = {
        "method": "draw_hist",
        "hist": hist_copy_0,
        "kwargs": {
            "color": color_list[0],
            "label": myhist_0.axes[1].label,
        },
    }
    plot_config["hist_1"] = {
        "method": "draw_hist",
        "hist": hist_copy_1,
        "kwargs": {
            "color": color_list[1],
            "label": myhist_0.axes[2].label,
        },
    }
    plot_config["hist_all"] = {
        "method": "draw_hist",
        "hist": hist_copy_all,
        "kwargs": {
            "color": color_list[2],
            "label": "all events",
        },
    }

    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    default_style_config["ax_cfg"]["ylabel"] = r"number of events"
    default_style_config["legend_cfg"]["ncol"] = 1
    default_style_config["legend_cfg"]["title_fontsize"] = 17
    default_style_config["legend_cfg"]["fontsize"] = 20
    default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)
