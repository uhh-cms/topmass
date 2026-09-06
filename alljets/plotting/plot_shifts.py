# coding: utf-8

"""
Custom plot functions for shifted variables with support for unrolled 2D/3D histograms.
"""

from __future__ import annotations

import re
from collections import OrderedDict

import law
from columnflow.plotting.plot_util import (
    apply_density,
    apply_variable_settings,
    prepare_style_config,
)
from columnflow.util import maybe_import
from modules.columnflow.columnflow.plotting.plot_all import plot_all
from modules.columnflow.columnflow.plotting.plot_util import (
    apply_process_settings,
    apply_process_scaling,
    get_cms_label,
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


# Shift Label Formatting Helpers
def get_mtop_mass(shift_name, nominal_mass=172.5):
    """Extract top quark mass from shift name (e.g., 'mtop171_up' -> 173.5 GeV)."""
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
    """Get h_damp scaling factor from shift name."""
    if shift_name == "hdamp_up":
        return 2.305
    elif shift_name == "hdamp_down":
        return 0.8738
    return None


def format_shift_label(shift_inst):
    """Format shift instance name into a readable label."""
    name = shift_inst.name

    if name == "nominal":
        return "Nominal"

    parts = name.split("_")

    if len(parts) >= 2:
        source = parts[0].upper()
        direction = parts[-1].capitalize()
        middle = " ".join(parts[1:-1])

        if middle:
            return f"{source} {middle} {direction}"
        return f"{source} {direction}"

    return shift_inst.label


# Unrolled Histogram Layout Helpers
def _compute_unrolled_layout(variable_insts, first_hist):
    """
    Compute layout for unrolled histograms (2D/3D flattened into 1D).

    Returns:
        - n_total_bins: Total number of bins after unrolling
        - block_sizes: Number of bins per dimension [outer, middle, inner]
        - sep_positions: Dict with 'outer' and 'middle' separator positions
    """
    n_dims = len(variable_insts)
    n_total_bins = first_hist.axes["unrolled"].size
    sep_positions = {"outer": [], "middle": [], "inner": []}
    block_sizes = None

    if n_dims < 2:
        return n_total_bins, block_sizes, sep_positions

    block_sizes = [vi.n_bins for vi in variable_insts]

    if n_dims == 2:
        # 2D: [outer, inner] - outer boundaries separate different outer bins
        for i in range(1, block_sizes[0]):
            sep_positions["outer"].append(i * block_sizes[1])

    elif n_dims == 3:
        # 3D: [outer, middle, inner] - both outer and middle separators
        outer_size, middle_size, inner_size = block_sizes

        # Middle separators: between different middle bins within same outer bin
        for outer_idx in range(outer_size):
            for middle_idx in range(1, middle_size):
                pos = (outer_idx * middle_size * inner_size) + (middle_idx * inner_size)
                sep_positions["middle"].append(pos)

        # Outer separators: between different outer bins
        for i in range(1, outer_size):
            sep_positions["outer"].append(i * middle_size * inner_size)

    return n_total_bins, block_sizes, sep_positions


def _style_tick_positions(block_sizes, variable_insts, n_total_bins):
    """
    Pre-plotting tick positions for style config.
    Used to set axis limits and default ticks before plotting.
    """
    if block_sizes and len(variable_insts) == 2:
        return np.arange(0, n_total_bins + 1, block_sizes[1])

    if block_sizes and len(variable_insts) == 3:
        outer_size, middle_size, inner_size = block_sizes
        tick_positions = []

        for outer_idx in range(outer_size):
            tick_positions.append(outer_idx * middle_size * inner_size)
            # Only show middle ticks if there aren't too many
            if middle_size <= 5:
                for middle_idx in range(middle_size):
                    pos = (outer_idx * middle_size * inner_size) + (middle_idx * inner_size)
                    if pos not in tick_positions:
                        tick_positions.append(pos)

        # Limit number of ticks if too many
        if len(tick_positions) > 30:
            step = max(1, len(tick_positions) // 20)
            tick_positions = tick_positions[::step]

        return tick_positions

    # Default: evenly spaced ticks
    n_ticks = min(20, n_total_bins)
    return np.linspace(0, n_total_bins, n_ticks + 1, dtype=int)


def _final_tick_positions(block_sizes, variable_insts, n_total_bins):
    """
    Post-plotting tick positions applied directly to axes.
    Similar to _style_tick_positions but with different inclusion rules.
    """
    if block_sizes and len(variable_insts) == 2:
        return np.arange(0, n_total_bins + 1, block_sizes[1])

    if block_sizes and len(variable_insts) == 3:
        outer_size, middle_size, inner_size = block_sizes
        tick_positions = [i * middle_size * inner_size for i in range(outer_size)]

        # Include middle ticks if total number is manageable
        if outer_size * middle_size <= 30:
            for outer_idx in range(outer_size):
                for middle_idx in range(middle_size):
                    pos = (outer_idx * middle_size * inner_size) + (middle_idx * inner_size)
                    if pos not in tick_positions:
                        tick_positions.append(pos)

        return sorted(tick_positions)

    return None


def _draw_block_separators(main_ax, rax, sep_positions):
    """Draw vertical separator lines between blocks in unrolled histograms."""
    # Outer separators: dashed black lines
    for pos in sep_positions["outer"]:
        main_ax.axvline(pos, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=10)
        if rax is not None:
            rax.axvline(pos, color="black", linestyle="--", linewidth=1.0, alpha=0.7, zorder=10)

    # Middle separators: dotted gray lines
    for pos in sep_positions["middle"]:
        main_ax.axvline(pos, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, zorder=9)
        if rax is not None:
            rax.axvline(pos, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, zorder=9)


# Shift Label and Configuration Helpers
def _get_shift_label(
    shift_inst,
    shift_name,
    diff,
    has_mtop_shifts,
    has_hdamp_shifts,
    pretty_labels,
    show_shift_percent,
):
    """Generate formatted label for a shift variation."""
    mass = get_mtop_mass(shift_name)
    factor = get_hdamp_factor(shift_name)
    label = shift_inst.label

    # Special formatting for mtop shifts
    if has_mtop_shifts and mass is not None:
        label = rf"$m_t^{{gen}} = {mass:.1f}\,\mathrm{{GeV}}$"

    # Special formatting for hdamp shifts
    elif has_hdamp_shifts:
        if shift_inst.name == "nominal":
            factor = 1.379
        if factor is not None:
            label = rf"$h_{{damp}} = {factor:.4f} \cdot m_t$"

    # General pretty formatting
    elif pretty_labels:
        label = format_shift_label(shift_inst)

    # Optionally show percent difference from nominal
    if show_shift_percent and not shift_inst.is_nominal:
        label += f" ({diff:+.2%})"

    return label


def _update_cfg(container, key, **updates):
    """Update a nested configuration dictionary."""
    sub = container.setdefault(key, {})
    sub.update(updates)
    return sub


# Uncertainty Calculation for Weight Variations
def _weight_variation_yerr(h, h_nom):
    """
    Compute systematic uncertainty for correlated weight variations.
    """
    var_diff = h.variances() - h_nom.variances()
    return np.sqrt(np.abs(var_diff))


# Main Plotting Function
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
    Plot shifted variables with support for unrolled 2D and 3D histograms.

    For correlated shifts (weight variations), uses variance difference method
    to isolate systematic uncertainty. For disjoint shifts, standard statistical
    uncertainties are used.
    """
    import hist

    pretty_labels = kwargs.get("pretty_labels", False)
    show_shift_percent = kwargs.get("show_shift_percent", False)
    variable_inst = variable_insts[0]

    # Prepare histograms: apply settings, scaling, density
    hists, process_style_config = apply_process_settings(hists, process_settings)

    first_hist = list(hists.values())[0]
    is_unrolled = "unrolled" in first_hist.axes.name

    n_total_bins = None
    block_sizes = None
    sep_positions = {"outer": [], "middle": [], "inner": []}

    # Handle unrolled histograms (2D/3D flattened to 1D)
    if is_unrolled:
        n_total_bins, block_sizes, sep_positions = _compute_unrolled_layout(variable_insts, first_hist)
    else:
        has_variable_axes = all(vi.name in first_hist.axes.name for vi in variable_insts)
        if has_variable_axes:
            hists, _ = apply_variable_settings(hists, variable_insts, variable_settings)

    # Remove negative contributions if requested
    if kwargs.get("remove_negative", None):
        hists = remove_negative_contributions(hists)

    # Apply process scaling and density
    hists = apply_process_scaling(hists)

    if density:
        hists = apply_density(hists, density)

    # Ensure all histograms have all shifts (add missing as nominal)
    all_shifts = set.union(*[set(h.axes["shift"]) for h in hists.values()])
    for h in hists.values():
        add_missing_shifts(h, all_shifts, str_axis="shift", nominal_bin="nominal")

    # Sum histograms over all processes
    h_sum = sum_hists(hists.values())

    # Build plot configuration
    plot_config = {}
    colors = {"nominal": "black", "up": "red", "down": "blue"}
    shift_order = {"up": 0, "nominal": 1, "down": 2}

    # Sort shifts: up first, then nominal, then down
    sorted_shifts = sorted(
        h_sum.axes["shift"],
        key=lambda s: shift_order.get(config_inst.get_shift(s).direction, 99),
    )

    # Check for special shift types for label formatting
    has_mtop_shifts = any("mtop" in shift_name for shift_name in h_sum.axes["shift"])
    has_hdamp_shifts = any("hdamp" in s for s in h_sum.axes["shift"])

    # Extract nominal histogram
    nominal_hist = h_sum[{"shift": hist.loc("nominal")}]
    nominal_values = nominal_hist.values()
    nominal_sum = sum(nominal_values)

    ratio_lows, ratio_highs = [], []

    # Loop over each shift and create plot config entries
    for shift_name in sorted_shifts:
        shift_inst = config_inst.get_shift(shift_name)
        h = h_sum[{"shift": hist.loc(shift_name)}]
        diff = sum(h.values()) / nominal_sum - 1

        # Generate shift label
        label = _get_shift_label(
            shift_inst,
            shift_name,
            diff,
            has_mtop_shifts,
            has_hdamp_shifts,
            pretty_labels,
            show_shift_percent,
        )

        # Normalization factors
        norm_factor = (sum(h.values()) / nominal_sum) if shape_norm else 1
        ratio_norm = nominal_values * norm_factor

        # Main plot config
        plot_config[shift_inst.name] = plot_cfg = {
            "method": "draw_hist",
            "hist": h,
            "kwargs": {
                "norm": norm_factor,
                "label": label,
                "color": colors[shift_inst.direction],
            },
            "ratio_kwargs": {
                "norm": ratio_norm,
                "color": colors[shift_inst.direction],
            },
        }

        if shift_inst.is_nominal:
            # Nominal ratio is exactly 1.0 with zero uncertainty
            plot_cfg["ratio_kwargs"]["yerr"] = False

        elif not shift_inst.has_tag("disjoint_from_nominal"):
            # Correlated shift: compute systematic uncertainty
            syst_err = _weight_variation_yerr(h, nominal_hist)

            # Main panel: absolute systematic error on the shifted histogram
            plot_cfg["kwargs"]["yerr"] = syst_err / norm_factor

            # Ratio panel: relative systematic error (shift / nominal)
            # Uses ratio_norm to properly account for shape normalization
            plot_cfg["ratio_kwargs"]["yerr"] = syst_err / ratio_norm

        # Override errors if requested
        if hide_stat_errors:
            for key in ("kwargs", "ratio_kwargs"):
                if key in plot_cfg:
                    plot_cfg[key]["yerr"] = False

        # Compute ratio values and uncertainties for axis range
        ratio_values = h.values() / ratio_norm
        ratio_yerr = plot_cfg["ratio_kwargs"].get("yerr", None)

        if ratio_yerr is False:
            r_low = r_high = ratio_values
        elif ratio_yerr is not None:
            r_low = ratio_values - ratio_yerr
            r_high = ratio_values + ratio_yerr
        else:
            auto_yerr = np.sqrt(h.variances()) / ratio_norm
            r_low = ratio_values - auto_yerr
            r_high = ratio_values + auto_yerr

        ratio_lows.append(r_low)
        ratio_highs.append(r_high)

    # Style configuration
    if not legend_title and len(hists) == 1:
        legend_title = list(hists.keys())[0].label

    # Determine y-scale
    if not yscale:
        yscale = "log" if variable_inst.log_y else "linear"

    # Build default style config
    default_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    # Set up tick positions for unrolled histograms
    style_tick_positions = None
    if is_unrolled:
        style_tick_positions = _style_tick_positions(block_sizes, variable_insts, n_total_bins)
        if "variable_cfg" in default_style_config:
            default_style_config["variable_cfg"]["xlabel"] = "Bin index"

    # Compute ratio axis limits based on actual data range
    ratio_min = float(np.min(np.concatenate(ratio_lows)))
    ratio_max = float(np.max(np.concatenate(ratio_highs)))

    max_dev = max(abs(1.0 - ratio_min), abs(ratio_max - 1.0))
    max_dev *= 1.1
    ratio_ylim = (1.0 - max_dev, 1.0 + max_dev)

    default_style_config["rax_cfg"]["ylim"] = ratio_ylim
    default_style_config["rax_cfg"]["ylabel"] = "Ratio"

    if legend_title:
        default_style_config["legend_cfg"]["title"] = legend_title

    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        style_config,
        deep=True,
    )

    # Additional styling for unrolled histograms
    if is_unrolled:
        ax_cfg = style_config.setdefault("ax_cfg", {})
        ax_cfg["xlim"] = (0, n_total_bins)
        ax_cfg["xlabel"] = "Bin index"
        ax_cfg["ylabel"] = "Events"
        ax_cfg["xticks"] = style_tick_positions

    # Add CMS label
    _update_cfg(style_config, "legend_cfg", ncols=1, loc="upper right", title="")
    _update_cfg(style_config, "cms_label_cfg", llabel=get_cms_label(None, "simpw")["llabel"])

    fig, ax = plot_all(plot_config, style_config, **kwargs)

    # Post-processing for unrolled histograms

    if is_unrolled and sep_positions:
        if isinstance(ax, (list, tuple, np.ndarray)):
            main_ax = ax[0]
            rax = ax[1] if len(ax) > 1 else None
        else:
            main_ax = ax
            rax = None

        # Set axis limits and labels
        main_ax.set_xlim(0, n_total_bins)
        main_ax.set_xlabel("Bin index")

        # Draw block separator lines
        _draw_block_separators(main_ax, rax, sep_positions)

        # Apply final tick positions
        final_tick_positions = _final_tick_positions(block_sizes, variable_insts, n_total_bins)
        if final_tick_positions is not None:
            main_ax.set_xticks(final_tick_positions)
            if rax is not None:
                rax.set_xticks(final_tick_positions)

        if rax is not None:
            rax.set_xlim(0, n_total_bins)
            rax.set_xlabel("Bin index")

    return fig, ax
