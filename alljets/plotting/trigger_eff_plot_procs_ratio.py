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
--producers default,trigger_prod
--plot-function alljets.plotting.trigger_eff_plot.plot_efficiencies
"""


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
    remove_residual_axis(hists, "shift")

    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_density_to_hists(hists, density)

    plot_config = OrderedDict()

    # calculate efficiencies
    eff_bin = int(kwargs.get("bin_sel", 0))

    if eff_bin == 0:
        logger.warning(
            "No bin selected, bin zero is used for efficiency calculation",
        )

    if (len(list(hists.keys())) > 2):
        logger.warning(
            "More than two input processes, only two are considered",
        )

    trigger_ref = np.array(config_inst.x.ref_trigger["tt_fh"])
    triggers = np.array(config_inst.x.trigger["tt_fh"])
    trigger_names = np.hstack((trigger_ref, triggers))

    # for i in range(len(trigger_names)):
    #     trigger_names[i] = trigger_names[i].replace("PF","")
    #     trigger_names[i] = trigger_names[i].replace("SixJet","SixJ")
    #     trigger_names[i] = trigger_names[i].replace("Double","2x")
    #     trigger_names[i] = trigger_names[i].replace("_2p2","")
    #     trigger_names[i] = trigger_names[i].replace("_2p94","")
    #     trigger_names[i] = trigger_names[i].replace("_p056","")

    trig_alias = kwargs.get("alias", "None")

    if not trig_alias == "None":
        trigger_names[eff_bin] = trig_alias

    # for updating labels of individual selector steps
    myhist_0 = hists[list(hists.keys())[0]]
    myhist_1 = hists[list(hists.keys())[1]]

    norm_hist_0 = np.array(myhist_0[:, 0].values())
    norm_hist_1 = np.array(myhist_1[:, 0].values())

    plot_config["hist_0"] = {
        "method": "draw_efficiency",
        "ratio_method": "draw_hist",
        "hist": myhist_0[:, eff_bin],
        "kwargs": {
            "color": "b",
            "norm": norm_hist_0,
            "label": f"{list(hists.keys())[0].name}",
            "histtype": "errorbar",
            "capsize": 3,
        },
        "ratio_kwargs": {
            "color": "b",
            "capsize": 3,
            "linestyle": "none",
            "norm": (myhist_1[:, eff_bin].values() * norm_hist_0) / norm_hist_1,
            "histtype": "errorbar",
        },
    }

    plot_config["hist_1"] = {
        "method": "draw_efficiency",
        "hist": myhist_1[:, eff_bin],
        "kwargs": {
            "norm": norm_hist_1,
            "label": f"{list(hists.keys())[1].name}",
            "histtype": "errorbar",
            "capsize": 3,
        },
        # "ratio_kwargs": {
        #     "color": "orange",
        #     # "linestyle": "none",
        #     "norm": (myhist_0[:, eff_bin].values()*norm_hist_1)/norm_hist_0,
        #     "histtype":"errorbar",
        # },
        # "rax_cfg": {
        #     "ylim": (0.6, 1.4),
        #     "ylabel": "MC / Data",
        # }
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

    return plot_all(plot_config, style_config, **kwargs)
