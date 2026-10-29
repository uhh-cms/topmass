# coding: utf-8

"""
Unified, process-adaptive version of plot_hist_matching / plot_hist_matching_ttbar_data /
plot_hist_matching_MC.

Instead of three near-duplicate functions, this single function inspects which processes are
actually present in `hists` and adapts:

  - "tt", if present, is split into a correct/wrong/unmatched cumulative stack using fixed
    hardcoded colors (MATCHING_COLORS below) -- tt's own `color1` is not used for the split.
  - "st", "qcd" or "qcd_est", if present, are each stacked on top as a single filled block
  - A data process (detected via `process_inst.is_data`), if present, is drawn with error
    bars, the ratio panel is enabled, and its normalization is computed from whatever MC
    ended up in the stack above
  - If no data is present, the ratio panel is skipped and the CMS label switches to the
    simulation-only ("simpw") variant

Only the tt correct/wrong/unmatched colors are hardcoded here (MATCHING_COLORS) -- everything
else (st, qcd, qcd_est, and any other process) comes from your `stylize_processes` config.
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


# bins along the reco/fit variable to hide data markers in
HIDE_DATA_MARKERS = {
    "fit_Top1_mass": (140, 195),
    "reco_Top1_mass": (140, 210),
    "reco_Top2_mass": (140, 210),
    "reco_Top_mass_avg": (140, 210),
}

# processes that carry a "fit_combination_type"-style axis
MATCHING_PROCESS_NAMES = ("tt",)

# fixed colors for tt's correct/wrong/unmatched split
MATCHING_COLORS = {"correct": "#cc0000", "wrong": "#ff6666", "unmatched": "#ffcccc"}

# processes stacked as a single filled block
BLOCK_PROCESS_NAMES = ("st", "qcd", "qcd_est")

_FALLBACK_COLOR = "#999999"


def _hide_data_in_window(data_hist, variable_inst):
    hide_range = HIDE_DATA_MARKERS.get(variable_inst.name)
    if hide_range is None:
        return data_hist
    low, high = hide_range
    edges = data_hist.axes[0].edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= low) & (centers <= high)
    data_hist = data_hist.copy()
    data_hist.values()[mask] = -999.0
    data_hist.variances()[mask] = 0.0
    return data_hist


def plot_hist_matching_combined(
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
    Process-adaptive matching-type plot. Handles any combination of tt, st, qcd/qcd_est
    and data that is passed in via `--processes`,
    """
    cut_vis = kwargs.get("cut_vis", None)
    variable_inst = variable_insts[0]
    hists = apply_variable_settings(hists, variable_insts, variable_settings)
    h = hists[0]
    h = apply_density(h, density)

    tt_entry = None
    block_hists = OrderedDict()
    data_hist = None
    data_label = None
    extra_hists = OrderedDict()

    for process_inst, proc_hist in h.items():
        name = process_inst.name
        if getattr(process_inst, "is_data", False):
            data_hist = proc_hist[0, :, sum]
            data_label = getattr(process_inst, "label", None) or name
            continue
        if name in MATCHING_PROCESS_NAMES:
            tt_entry = (process_inst, proc_hist)
            continue
        if name in BLOCK_PROCESS_NAMES:
            block_hists[name] = {
                "hist": proc_hist[0, :, sum],
                "label": getattr(process_inst, "label", None) or name,
                "color": getattr(process_inst, "color1", None) or _FALLBACK_COLOR,
            }
            continue
        # unrecognized MC process: sum over the whole (matching) axis, stack as a plain block
        extra_hists[name] = {
            "hist": proc_hist[0, :, sum],
            "label": getattr(process_inst, "label", None) or name,
            "color": getattr(process_inst, "color1", None) or _FALLBACK_COLOR,
        }

    # enforce a stable stacking order (st below qcd/qcd_est) regardless of dict iteration order
    block_hists = OrderedDict(
        (name, block_hists[name]) for name in BLOCK_PROCESS_NAMES if name in block_hists
    )

    if tt_entry is None and not block_hists and not extra_hists:
        raise ValueError(
            "No recognized MC process found in hists "
            f"(got: {[p.name for p in h.keys()]})",
        )

    # --- build the stack bottom-to-top: correct, wrong, unmatched (tt), st, qcd ---
    layers = []  # (key, cumulative_hist, comp_type) in stacking order
    running_stack = None

    if tt_entry is not None:
        tt_inst, tt_hist = tt_entry
        tt_label = getattr(tt_inst, "label", None) or tt_inst.name

        correct_hist = tt_hist[0, :, 3]
        wrong_hist = tt_hist[0, :, 2]
        unmatched_hist = tt_hist[0, :, 1]

        running = correct_hist
        layers.append(("hist_correct", running, "correct"))
        running = running + wrong_hist
        layers.append(("hist_wrong", running, "wrong"))
        running = running + unmatched_hist
        layers.append(("hist_unmatched", running, "unmatched"))

        running_stack = running

    for name, info in block_hists.items():
        running_stack = info["hist"] if running_stack is None else running_stack + info["hist"]
        layers.append((f"hist_{name}", running_stack, ("block", name)))

    for name, info in extra_hists.items():
        running_stack = info["hist"] if running_stack is None else running_stack + info["hist"]
        layers.append((f"hist_{name}", running_stack, ("extra", name)))

    total_mc = running_stack

    n_layers = len(layers)
    plot_config = OrderedDict()
    for i, (key, layer_hist, comp_type) in enumerate(layers):
        zorder = n_layers - i
        if comp_type in ("correct", "wrong", "unmatched"):
            fill_color = MATCHING_COLORS[comp_type]
            label = f"{tt_label} {comp_type}"
        elif isinstance(comp_type, tuple) and comp_type[0] == "block":
            name = comp_type[1]
            fill_color = block_hists[name]["color"]
            label = block_hists[name]["label"]
        else:
            name = comp_type[1]
            fill_color = extra_hists[name]["color"]
            label = extra_hists[name]["label"]
        plot_config[key] = {
            "method": "draw_hist",
            "hist": layer_hist,
            "kwargs": {
                "color": fill_color,
                "histtype": "fill",
                "label": label,
                "edgecolor": "black",
                "linewidth": 1,
                "zorder": zorder,
            },
        }

    if total_mc is not None:
        plot_config["hist_total_uncert"] = {
            "method": "draw_stat_error_bands",
            "ratio_method": "draw_stat_error_bands",
            "hist": total_mc,
            "kwargs": {"label": "MC stat. unc.", "zorder": n_layers + 1},
            "ratio_kwargs": {"norm": total_mc.values()},
        }

    has_data = data_hist is not None
    if has_data:
        data_hist = _hide_data_in_window(data_hist, variable_inst)
        plot_config["hist_data"] = {
            "method": "draw_errorbars",
            "ratio_method": "draw_errorbars",
            "hist": data_hist,
            "kwargs": {"label": data_label, "zorder": n_layers + 2},
            "ratio_kwargs": {"norm": total_mc.values() if total_mc is not None else None},
        }

    if cut_vis == "vline":
        plot_config["cut_region"] = {
            "method": "draw_vline",
            "kwargs": {
                "x": 6.3 if variable_inst == "fitchi2" else 2,
                "ymin": 0.0,
                "ymax": 0.7,
                "zorder": 10,
                "color": "black",
                "linestyle": "--",
            },
        }
    # --- style config ---
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    default_style_config["ax_cfg"]["ylabel"] = variable_inst.get_full_y_title(
        bin_width=None,
        unit=None,
        unit_format="{title} / {unit}",
    )

    tt_split_labels = [f"{tt_label} {c}" for c in ("correct", "wrong", "unmatched")] if tt_entry is not None else []
    block_extra_labels = [info["label"] for info in list(block_hists.values()) + list(extra_hists.values())]
    use_two_col = bool(tt_split_labels) and bool(block_extra_labels)

    default_style_config["legend_cfg"]["ncols"] = 2 if use_two_col else 1
    default_style_config["legend_cfg"]["title_fontsize"] = 24
    default_style_config["legend_cfg"]["fontsize"] = 20

    if has_data:
        default_style_config["rax_cfg"]["ylim"] = (0.61, 1.39)
        kwargs.setdefault("skip_ratio", False)
    else:
        default_style_config["cms_label_cfg"]["llabel"] = get_cms_label(None, "simpw")["llabel"]
        kwargs.setdefault("skip_ratio", True)

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    return aj_plot_all(plot_config, style_config, **kwargs)
