# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.util import maybe_import
from columnflow.plotting.plot_all import plot_all
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
    import IPython
    IPython.embed()
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_settings(hists, process_settings)
    hists = apply_density(hists, density)

    plot_config = OrderedDict()

    # for updating labels of individual selector steps
    norm = (np.full((1, len(hists[list(hists.keys())[0]][:, 1].values())), 1))[0]

    for i in range(len(list(hists.keys()))):
        if list(hists.keys())[i] == "tt":
            plot_config["hist_correct"] = {
                "method": "draw_hist",
                "hist": hists[list(hists.keys())[i]][:, 3],
                "kwargs": {
                    "stack": True,
                    "norm": norm,
                    "label": f"{list(hists.keys())[i].name}, correct",
                },
            }

            plot_config["hist_wrong"] = {
                "method": "draw_hist",
                "hist": hists[list(hists.keys())[i]][:, 2],
                "kwargs": {
                    "stack": True,
                    "norm": norm,
                    "label": f"{list(hists.keys())[i].name}, wrong",
                },
            }

            plot_config["hist_unmatched"] = {
                "method": "draw_hist",
                "hist": hists[list(hists.keys())[i]][:, 1],
                "kwargs": {
                    "stack": True,
                    "norm": norm,
                    "label": f"{list(hists.keys())[i].name}, unmatched",
                },
            }
        else:
            plot_config[f"hist_{i}"] = {
                "method": "draw_hist",
                # "ratio_method": "draw_hist",
                "hist": hists[list(hists.keys())[i]][:, 0],
                "kwargs": {
                    "norm": norm,
                    "label": f"{list(hists.keys())[i].name}",
                },
                # "ratio_kwargs": {
                #     "color": "#5790fc",
                #     # "linestyle": "none",
                #     "norm": (myhist_2[:, 0] + myhist_2[:, 1] + myhist_2[:, 2] + myhist_1[:, 0]),
                #     "histtype": "errorbar",
                # },
            }

    # setup style config
    default_style_config = prepare_style_config(
        config_inst, category_inst, variable_inst, density, shape_norm, yscale,
    )
    # plot-function specific changes
    # default_style_config["ax_cfg"]["ylabel"] = "Efficiency"
    # default_style_config["legend_cfg"]["title"] = trigger_names[eff_bin]
    default_style_config["legend_cfg"]["ncol"] = 2
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20
    # default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
    kwargs["skip_ratio"] = True

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return plot_all(plot_config, style_config, **kwargs)
