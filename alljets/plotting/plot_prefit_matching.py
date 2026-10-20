# coding: utf-8

"""
Prefit plot with tt split into correct / wrong / unmatched.

The matching decomposition is cosmetic.

The stack is drawn here, but the uncertainty bands and the data entry come
from columnflow's prepare_stack_plot_config, fed with the original shifted MC
histograms after collapsing the matching axis. They are therefore identical
to what PlotShiftedVariablesFromModel1D draws (incl. --merge-stat-errors).
"""

from __future__ import annotations

from collections import OrderedDict

import law

from columnflow.plotting.plot_util import (
    apply_density,
    apply_variable_settings,
    prepare_stack_plot_config,
    prepare_style_config,
)
from columnflow.util import maybe_import

from alljets.plotting.aj_plot_all import aj_plot_all
from modules.columnflow.columnflow.plotting.plot_util import get_cms_label


hist = maybe_import("hist")
plt = maybe_import("matplotlib.pyplot")
od = maybe_import("order")


logger = law.logger.get_logger(__name__)


HIDE_DATA_MARKERS = {
    "fit_Top1_mass": (140, 195),
    "reco_Top1_mass": (140, 210),
    "reco_Top2_mass": (140, 210),
    "reco_Top_mass_avg": (140, 210),
}

BLINDED_CATEGORIES = (
    "sig",
    "rbb_sig",
    "pgof_sig",
)

MATCHING_PROCESS_NAMES = ("tt",)

MATCHING_COLORS = {
    "correct": "#cc0000",
    "wrong": "#ff6666",
    "unmatched": "#ffcccc",
}

MATCHING_INDEX = {
    "correct": 3,
    "wrong": 2,
    "unmatched": 1,
}

BLOCK_PROCESS_NAMES = (
    "st",
    "qcd",
    "qcd_est",
)

_FALLBACK_COLOR = "#999999"


def _nominal(h):
    return h[{"shift": hist.loc("nominal")}]


def _hide_data_in_window(data_hist, variable_inst):

    hide_range = HIDE_DATA_MARKERS.get(variable_inst.name)

    if hide_range is None:
        return data_hist

    low, high = hide_range

    edges = data_hist.axes[0].edges
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = (centers >= low) & (centers <= high)

    data_hist = data_hist.copy()
    data_hist.view()["value"][mask] = -999.0
    data_hist.view()["variance"][mask] = 0.0

    return data_hist


def _collapse_matching_axis(h):
    """
    Return the histogram with the matching axis summed out.

    This is the histogram whose uncertainties correspond to the total
    MC prediction, independent of how tt is visually decomposed.
    """

    if "fit_combination_type" in [a.name for a in h.axes]:
        return h[{"fit_combination_type": sum}]

    return h


def plot_prefit_matching(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift] | None = None,
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot a prefit histogram with tt split into correct / wrong / unmatched.

    Per default the stat and syst bands are plotted seperately, but can be merged with --merge-stat-errors.

    Example commands:
    law run cf.PlotShiftedVariablesFromModelMatching1D --version v2_TopMass\
        --configs 2017_v9 --inference-model peak_3D \
        --variables fit_Top1_mass-fit_combination_type  --categories sig:"SR_3D_peak" \
        --nuisances 'CMS_scale_j_Flavor*'  --merge-processes tt,data,qcd_est \
        --hist-hooks qcd
    """
    variable_inst = variable_insts[0]

    hists = apply_variable_settings(
        hists,
        variable_insts[:1],
        variable_settings,
    )

    h = hists[0]
    h = apply_density(h, density)

    tt_entry = None

    block_hists = OrderedDict()
    extra_hists = OrderedDict()
    cfg_hists = OrderedDict()

    # Collect histograms
    for process_inst, proc_hist in h.items():
        name = process_inst.name
        label = (getattr(process_inst, "label", None) or name)
        color = (getattr(process_inst, "color1", None) or _FALLBACK_COLOR)

        if getattr(process_inst, "is_data", False):
            cfg_hists[process_inst] = proc_hist
            continue

        # Keep the matching axis for the visual decomposition, BUT give
        # the COLLAPSED histogram to the standard config for the bands.
        if name in MATCHING_PROCESS_NAMES:
            tt_entry = (process_inst, proc_hist)
            cfg_hists[process_inst] = _collapse_matching_axis(proc_hist)
            continue

        # all other MC:
        # Shifted histograms unchanged for the standard config.
        cfg_hists[process_inst] = proc_hist

        entry = {
            "hist": _nominal(proc_hist),
            "label": label,
            "color": color,
        }

        if name in BLOCK_PROCESS_NAMES:
            block_hists[name] = entry
        else:
            extra_hists[name] = entry

    block_hists = OrderedDict((n, block_hists[n]) for n in BLOCK_PROCESS_NAMES if n in block_hists)

    if (tt_entry is None and not block_hists and not extra_hists):
        raise ValueError(
            "No recognized MC process found in hists "
            f"(got: {[p.name for p in h.keys()]})",
        )

    # Build visual stack
    layers = []
    running_stack = None
    tt_label = None

    if tt_entry is not None:
        tt_inst, tt_hist = tt_entry
        tt_label = (getattr(tt_inst, "label", None) or tt_inst.name)
        tt_nom = _nominal(tt_hist)

        running = None

        for comp in ("correct", "wrong", "unmatched"):
            part = tt_nom[:, MATCHING_INDEX[comp]]
            running = (part if running is None else running + part)
            layers.append((f"hist_{comp}", running, comp))

        running_stack = running

    for name, info in block_hists.items():
        running_stack = (info["hist"] if running_stack is None else running_stack + info["hist"])
        layers.append((f"hist_{name}", running_stack, ("block", name)))

    for name, info in extra_hists.items():
        running_stack = (info["hist"] if running_stack is None else running_stack + info["hist"])
        layers.append((f"hist_{name}", running_stack, ("extra", name)))

    n_layers = len(layers)

    # Plot stack
    plot_config = OrderedDict()

    for i, (key, layer_hist, comp_type) in enumerate(layers):
        if comp_type in MATCHING_COLORS:
            fill_color = MATCHING_COLORS[comp_type]
            label = f"{tt_label} {comp_type}"
        else:
            info = (block_hists if comp_type[0] == "block" else extra_hists)[comp_type[1]]
            fill_color = info["color"]
            label = info["label"]

        plot_config[key] = {
            "method": "draw_hist",
            "hist": layer_hist,
            "kwargs": {
                "color": fill_color,
                "histtype": "fill",
                "label": label,
                "edgecolor": "black",
                "linewidth": 1,
                "zorder": n_layers - i,
            },
        }

    # Uncertainty bands and data
    # Taken from the standard columnflow config (stat band, syst band,
    # --merge-stat-errors, data), only the stack is replaced by the
    # layers above.
    base_config = prepare_stack_plot_config(
        hists=cfg_hists,
        shift_insts=shift_insts,
        shape_norm=shape_norm,
        density=density,
        **kwargs,
    )

    for key, entry in base_config.items():

        if key == "mc_stack":
            continue

        if key in ("mc_stat_unc", "mc_syst_unc"):
            entry["kwargs"].setdefault("zorder", n_layers + 1)

        elif key == "data":
            entry["kwargs"].setdefault("zorder", n_layers + 2)

            if category_inst.name in BLINDED_CATEGORIES:
                entry["hist"] = _hide_data_in_window(entry["hist"], variable_inst)

        plot_config[key] = entry

    has_data = "data" in plot_config

    # Style
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    default_style_config["ax_cfg"]["ylabel"] = (
        variable_inst.get_full_y_title(
            bin_width=None,
            unit=None,
            unit_format="{title} / {unit}",
        )
    )

    n_tt = 3 if tt_entry is not None else 0
    n_other = len(block_hists) + len(extra_hists)

    default_style_config["legend_cfg"]["ncols"] = (2 if (n_tt and n_other) else 1)
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20

    if has_data:
        kwargs.setdefault("skip_ratio", False)
    else:
        default_style_config["cms_label_cfg"]["llabel"] = (get_cms_label(None, "simpw")["llabel"])
        kwargs.setdefault("skip_ratio", True)

    style_config = law.util.merge_dicts(
        default_style_config,
        style_config,
        deep=True,
    )

    return aj_plot_all(plot_config, style_config, **kwargs)
