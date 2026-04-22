# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

from collections import OrderedDict

import law
from columnflow.plotting.plot_util import (apply_density, apply_variable_settings, prepare_style_config)
from columnflow.util import maybe_import
from modules.columnflow.columnflow.plotting.plot_all import plot_all
from modules.columnflow.columnflow.plotting.plot_util import (
    apply_process_settings,
    apply_process_scaling,
    remove_negative_contributions,
)
from columnflow.hist_util import add_missing_shifts, sum_hists

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)


def plot_shifted_variable(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift] | None,
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    hide_stat_errors: bool | None = None,
    legend_title: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    law run cf.PlotShiftedVariablesPerShift1D --configs 2017_v9 --version v1_Analysis\
        --datasets tt_fh_powheg,tt_sl_powheg,tt_dl_powheg\
        --selector-steps All,SignalOrBkgTrigger,HT,jet,BTag20,LeadingSix20BTag\
        --variables fit_Top1_mass_percentile\
        --categories sig\
        --shift-sources mtop1,hdamp,jer\
        --custom-style-config "shift_plots_mtop"\
        --plot-function alljets.plotting.plot_shifts.plot_shifted_variable\
        --general-settings "density"
    """
    import hist

    pretty_labels = kwargs.get("pretty_labels", False)
    show_shift_percent = kwargs.get("show_shift_percent", False)
    variable_inst = variable_insts[0]

    hists, process_style_config = apply_process_settings(hists, process_settings)
    hists, variable_style_config = apply_variable_settings(hists, variable_insts, variable_settings)
    if kwargs.get("remove_negative", None):
        hists = remove_negative_contributions(hists)
    hists = apply_process_scaling(hists)
    if density:
        hists = apply_density(hists, density)

    # add missing shifts to all histograms
    all_shifts = set.union(*[set(h.axes["shift"]) for h in hists.values()])
    for h in hists.values():
        add_missing_shifts(h, all_shifts, str_axis="shift", nominal_bin="nominal")

    # create the sum of histograms over all processes
    h_sum = sum_hists(hists.values())

    # setup plotting configs
    plot_config = {}
    colors = {
        "nominal": "black",
        "up": "red",
        "down": "blue",
    }
    shift_order = {"up": 0, "nominal": 1, "down": 2}

    sorted_shifts = sorted(
        h_sum.axes["shift"],
        key=lambda s: shift_order.get(config_inst.get_shift(s).direction, 99),
    )
    has_mtop_shifts = any("mtop" in shift_name for shift_name in h_sum.axes["shift"])
    has_hdamp_shifts = any("hdamp" in s for s in h_sum.axes["shift"])

    for shift_name in sorted_shifts:
        shift_inst = config_inst.get_shift(shift_name)

        h = h_sum[{"shift": hist.loc(shift_name)}]
        # assuming `nominal` always has shift id 0
        ratio_norm = h_sum[{"shift": hist.loc("nominal")}].values()

        diff = sum(h.values()) / sum(ratio_norm) - 1

        mass = get_mtop_mass(shift_name)
        factor = get_hdamp_factor(shift_name)

        if has_mtop_shifts and mass is not None:
            label = rf"$m_t^{{gen}} = {mass:.1f}\,\mathrm{{GeV}}$"

        elif has_hdamp_shifts:
            if shift_inst.name == "nominal":
                factor = 1.379

            if factor is not None:
                label = rf"$h_{{damp}} = {factor:.4f} \cdot m_t$"

        elif pretty_labels:
            label = format_shift_label(shift_inst)
        else:
            label = shift_inst.label

        if show_shift_percent and not shift_inst.is_nominal:
            label += f" ({diff:+.2%})"

        plot_config[shift_inst.name] = plot_cfg = {
            "method": "draw_hist",
            "hist": h,
            "kwargs": {
                "norm": sum(h.values()) if shape_norm else 1,
                "label": label,
                "color": colors[shift_inst.direction],
            },
            "ratio_kwargs": {
                "norm": ratio_norm,
                "color": colors[shift_inst.direction],
            },
        }
        if hide_stat_errors:
            for key in ("kwargs", "ratio_kwargs"):
                if key in plot_cfg:
                    plot_cfg[key]["yerr"] = None

    # legend title setting
    if not legend_title and len(hists) == 1:
        # use process label as default if 1 process
        process_inst = list(hists.keys())[0]
        legend_title = process_inst.label

    if not yscale:
        yscale = "log" if variable_inst.log_y else "linear"

    default_style_config = prepare_style_config(
        config_inst,
        category_inst,
        variable_inst,
        density,
        shape_norm,
        yscale,
    )
    nominal_hist = h_sum[{"shift": hist.loc("nominal")}]
    n = nominal_hist.values()
    n_var = nominal_hist.variances()

    ratio_min = np.inf
    ratio_max = -np.inf

    for shift_name in h_sum.axes["shift"]:
        h = h_sum[{"shift": hist.loc(shift_name)}]
        v = h.values()
        v_var = h.variances()

        mask = n > 0

        # central ratio
        r = np.ones_like(v)
        r[mask] = v[mask] / n[mask]

        sig_v = np.sqrt(v_var)
        sig_n = np.sqrt(n_var)

        r_up = np.ones_like(r)
        r_dn = np.ones_like(r)

        r_up[mask] = (v[mask] + sig_v[mask]) / (n[mask] - sig_n[mask] + 1e-12)
        r_dn[mask] = (v[mask] - sig_v[mask]) / (n[mask] + sig_n[mask] + 1e-12)

        r_low = np.minimum(r_up, r_dn)
        r_high = np.maximum(r_up, r_dn)

        ratio_min = min(ratio_min, np.min(r_low[mask]))
        ratio_max = max(ratio_max, np.max(r_high[mask]))

    default_style_config["rax_cfg"]["ylim"] = (ratio_min, ratio_max)
    default_style_config["rax_cfg"]["ylabel"] = "Ratio"
    if legend_title:
        default_style_config["legend_cfg"]["title"] = legend_title
    if shape_norm:
        default_style_config["ax_cfg"]["ylabel"] = "Normalized entries"
    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        variable_style_config[variable_inst],
        style_config,
        deep=True,
    )

    return plot_all(plot_config, style_config, **kwargs)


import re


def get_mtop_mass(shift_name, nominal_mass=172.5):
    if shift_name == "nominal":
        return nominal_mass

    match = re.search(r"mtop(\d+)_(up|down)", shift_name)
    if not match:
        return None

    shift_val = float(match.group(1))
    direction = match.group(2)

    if direction == "up":
        return nominal_mass + shift_val
    elif direction == "down":
        return nominal_mass - shift_val

    return nominal_mass


def get_hdamp_factor(shift_name):
    if shift_name == "hdamp_up":
        return 2.305
    elif shift_name == "hdamp_down":
        return 0.8738
    return None


def format_shift_label(shift_inst):
    name = shift_inst.name

    if name == "nominal":
        return "Nominal"

    parts = name.split("_")

    if len(parts) >= 2:
        source = parts[0].upper()
        direction = parts[-1].capitalize()
        middle = " ".join(parts[1:-1])  # handles complex cases

        if middle:
            return f"{source} {middle} {direction}"
        else:
            return f"{source} {direction}"

    return shift_inst.label
