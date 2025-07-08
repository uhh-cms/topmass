# coding: utf-8

"""
Example plot function.
"""

from __future__ import annotations

__all__ = []

from collections.abc import Callable
from typing import Optional
from columnflow.types import Sequence
from columnflow.util import maybe_import
from columnflow.plotting.plot_util import (
    get_position,
    apply_ax_kwargs,
    get_cms_label,
    remove_label_placeholders,
    apply_label_placeholders,
    calculate_stat_error,
)
from columnflow.plotting.plot_all import (
    draw_stat_error_bands, draw_syst_error_bands, draw_stack, draw_hist, draw_profile, draw_errorbars,
)
import law.logger

logger = law.logger.get_logger(__name__)
hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


def binom_int(num, den, confint=0.68):
    from scipy.stats import beta
    quant = (1 - confint) / 2.
    low = beta.ppf(quant, num, den - num + 1)
    high = beta.ppf(1 - quant, num + 1, den - num)
    return (np.nan_to_num(low), np.where(np.isnan(high), 1, high))


def draw_efficiency(
    ax: plt.Axes,
    h: hist.Hist,
    norm: float | Sequence | np.ndarray = 1.0,
    **kwargs,
) -> None:

    # Extract histogram data
    values = h.values()
    bins = h.axes[0].edges

    efficiency = np.nan_to_num(values / norm)
    efficiency = np.where(efficiency > 1, 1, efficiency)  # beware negative event weights
    efficiency = np.where(efficiency < 0, 0, efficiency)  # beware negative event weights

    # getting error bars
    band_low, band_high = binom_int(values, norm)
    error_low = np.asarray(efficiency - band_low)
    error_high = np.asarray(band_high - efficiency)

    # removing large errors in empty bins
    error_low[error_low == 1] = 0
    error_high[error_high == 1] = 0

    # stacking errors
    errors = np.concatenate((error_low.reshape(error_low.shape[0], 1),
                             error_high.reshape(error_high.shape[0], 1)), axis=1,
                            )
    errors = errors.T

    defaults = {
        "x": (bins[:-1] + bins[1:]) / 2,
        "y": efficiency,
        "yerr": errors,
        "color": "k",
        "marker": "o",
        "elinewidth": 1,
    }
    defaults.update(kwargs)
    ax.errorbar(**defaults)

    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_ylim(0, 1)
    ax.legend()


def draw_efficiency_with_fit(
    ax: plt.Axes,
    h: hist.Hist,
    fit_function: Callable,
    fit_result: float | np.ndarray,
    norm: float | Sequence | np.ndarray = 1.0,
    **kwargs,
) -> None:
    values = h.values() / norm
    values = np.nan_to_num(values, nan=0)

    band_low, band_high = binom_int(h.values(), norm)
    yerror_low = np.asarray(values - band_low)
    yerror_high = np.asarray(band_high - values)
    yerror_low[yerror_low == 1] = 0
    yerror_high[yerror_high == 1] = 0

    # removing large errors in empty bins

    # stacking y errors
    yerrors = np.concatenate((yerror_low.reshape(yerror_low.shape[0], 1),
                             yerror_high.reshape(yerror_high.shape[0], 1)), axis=1)
    yerrors = yerrors.T

    defaults = {
        "x": h.axes[0].centers,
        "y": values,
        "yerr": yerrors,
        "color": "k",
        "linestyle": "none",
        "marker": "o",
        "elinewidth": 1,
    }
    defaults.update(kwargs)

    # calculate x error
    xerror_low = h.axes[0].edges
    xerror_low = xerror_low[:(len(xerror_low) - 1)]
    x_filled = np.where(defaults["x"] == 0, h.axes[0].centers, defaults["x"])
    xerror_low = x_filled - xerror_low
    xerror_high = h.axes[0].edges[1:] - x_filled
    xerrors = np.concatenate((xerror_low.reshape(xerror_low.shape[0], 1),
                             xerror_high.reshape(xerror_high.shape[0], 1)), axis=1)
    xerrors = xerrors.T

    defaults.update({"xerr": xerrors})

    ax.errorbar(**defaults)
    x = np.linspace(np.min(h.axes[0].edges), np.max(h.axes[0].edges), num=100)
    ax.plot(x, fit_function(x, *fit_result), color=defaults["color"])
    ax.set_xlabel("Bin")
    ax.set_ylabel("Counts")
    ax.set_ylim(0, 1)
    ax.legend()


def draw_ratio_of_fit(
    ax: plt.Axes,
    h: hist.Hist,
    fit_function: Callable,
    fit_result: float | np.ndarray,
    norm: float | Sequence | np.ndarray = 1.0,
    error_type: str = "poisson_unweighted",
    **kwargs,
) -> None:
    error_bar_kwargs = kwargs.copy()
    error_bar_kwargs.pop("fit_norm")
    error_bar_kwargs.pop("linestyle")
    draw_errorbars(ax, h, norm, error_type, **error_bar_kwargs)

    fit_defaults = {
        "color": "k",
        "linewidth": 1,
        "fit_norm": fit_result,
    }
    fit_defaults.update(kwargs)

    x = np.linspace(np.min(h.axes[0].edges), np.max(h.axes[0].edges), num=100)
    ax.plot(
        x, fit_function(x, *fit_result) / fit_function(x, *fit_defaults["fit_norm"]),
        color=fit_defaults["color"], linewidth=fit_defaults["linewidth"], linestyle="solid",
    )


def draw_ratio_hist_fit(
    ax: plt.Axes,
    h: hist.Hist,
    fit_function: Callable,
    fit_result: float | np.ndarray,
    norm: float | Sequence | np.ndarray = 1.0,
    error_type: str = "poisson_unweighted",
    **kwargs,
) -> None:
    assert error_type in {"variance", "poisson_unweighted", "poisson_weighted"}
    values = h.values() / fit_function(h.axes[0].centers, *fit_result)

    defaults = {
        "x": h.axes[0].centers,
        "y": values,
        "color": "k",
        "marker": "o",
        "elinewidth": 1,
    }
    defaults.update(kwargs)
    defaults.pop("fit_norm")
    defaults["linestyle"] = "none"

    if "yerr" not in defaults:
        if h.storage_type.accumulator is not hist.accumulators.WeightedSum:
            raise TypeError(
                "Error bars calculation only implemented for histograms with storage type WeightedSum "
                "either change the Histogram storage_type or set yerr manually",
            )
        yerr = calculate_stat_error(h, error_type)
        # normalize yerr to the histogram = error propagation on standard deviation
        yerr = abs(yerr / fit_function(h.axes[0].centers, *fit_result))
        # replace inf with nan for any bin where norm = 0 and calculate_stat_error returns a non zero value
        if np.any(np.isinf(yerr)):
            yerr[np.isinf(yerr)] = np.nan
        defaults["yerr"] = yerr

    ax.errorbar(**defaults)


def draw_errorbars_with_fit(
    ax: plt.Axes,
    h: hist.Hist,
    fit_function: Callable,
    fit_result: float | np.ndarray,
    norm: float | Sequence | np.ndarray = 1.0,
    error_type: str = "poisson_unweighted",
    **kwargs,
) -> None:
    error_bar_kwargs = kwargs.copy()
    error_bar_kwargs.pop("linestyle")
    draw_errorbars(ax, h, norm, error_type, **error_bar_kwargs)

    fit_defaults = {
        "color": "k",
        "linewidth": 1,
        "fit_norm": fit_result,
    }
    fit_defaults.update(kwargs)

    x = np.linspace(np.min(h.axes[0].edges), np.max(h.axes[0].edges), num=100)
    ax.plot(
        x, fit_function(x, *fit_result) / fit_function(x, *fit_defaults["fit_norm"]),
        color=fit_defaults["color"], linewidth=fit_defaults["linewidth"], linestyle="solid",
    )


def aj_plot_all(
    plot_config: dict,
    style_config: dict,
    skip_ratio: bool = False,
    skip_legend: bool = False,
    cms_label: str = "wip",
    whitespace_fraction: float = 0.3,
    magnitudes: float = 4,
    fit_function: Optional[Callable] = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Function that calls multiple plotting methods based on two configuration dictionaries, *plot_config* and
    *style_config*.

    The *plot_config* expects dictionaries with fields:

        - "method": str, identical to the name of a function defined above
        - "hist": hist.Hist or hist.Stack
        - "fit_result": Array of fit parameters (optional)
        - "kwargs": dict (optional)
        - "ratio_kwargs": dict (optional)

    The *style_config* expects fields (all optional):

        - "gridspec_cfg": dict
        - "ax_cfg": dict
        - "rax_cfg": dict
        - "legend_cfg": dict
        - "cms_label_cfg": dict

    :param plot_config: Dictionary that defines which plot methods will be called with which key word arguments.
    :param style_config: Dictionary that defines arguments on how to style the overall plot.
    :param skip_ratio: Optional bool parameter to not display the ratio plot.
    :param skip_legend: Optional bool parameter to not display the legend.
    :param cms_label: Optional string parameter to set the CMS label text.
    :param whitespace_fraction: Optional float parameter that defines the ratio of which the plot will consist of
        whitespace for the legend and labels
    :param magnitudes: Optional float parameter that defines the displayed ymin when plotting with a logarithmic scale.
    :return: tuple of plot figure and axes
    """
    # general mplhep style
    plt.style.use(mplhep.style.CMS)

    # setup figure and axes
    rax = None
    grid_spec = {"left": 0.15, "right": 0.95, "top": 0.95, "bottom": 0.1}
    grid_spec |= style_config.get("gridspec_cfg", {})
    if not skip_ratio:
        grid_spec = {"height_ratios": [3, 1], "hspace": 0, **grid_spec}
        fig, axs = plt.subplots(2, 1, gridspec_kw=grid_spec, sharex=True)
        (ax, rax) = axs
    else:
        grid_spec.pop("height_ratios", None)
        fig, ax = plt.subplots(gridspec_kw=grid_spec)
        axs = (ax,)

    # invoke all plots methods
    plot_methods = {
        func.__name__: func
        for func in [
            draw_stat_error_bands, draw_syst_error_bands, draw_stack, draw_hist, draw_profile,
            draw_errorbars, draw_efficiency, draw_efficiency_with_fit, draw_ratio_of_fit, draw_ratio_hist_fit,
            draw_errorbars_with_fit,
        ]
    }

    # keep track of number of iterations
    for key, cfg in plot_config.items():
        # check if required fields are present
        if "method" not in cfg:
            raise ValueError(f"no method given in plot_cfg entry {key}")
        if "hist" not in cfg:
            raise ValueError(f"no histogram(s) given in plot_cfg entry {key}")

        # invoke the method
        method = cfg["method"]
        h = cfg["hist"]
        fit_result = cfg.get("fit_result", None)
        if method.endswith("fit"):
            plot_methods[method](ax, h, fit_function, fit_result, **cfg.get("kwargs", {}))
        else:
            plot_methods[method](ax, h, **cfg.get("kwargs", {}))

        # repeat for ratio axes if configured
        if not skip_ratio and "ratio_kwargs" in cfg:
            # take ratio_method if the ratio plot requires a different plotting method
            method = cfg.get("ratio_method", method)
            if method.endswith("fit"):
                plot_methods[method](
                    rax, h, fit_function, fit_result, **cfg.get("ratio_kwargs", {}),
                )
            else:
                plot_methods[method](rax, h, **cfg.get("ratio_kwargs", {}))

    # axis styling
    ax_kwargs = {
        "ylabel": "Counts",
        "xlabel": "variable",
        "yscale": "linear",
    }

    # some default ylim settings based on yscale
    log_y = style_config.get("ax_cfg", {}).get("yscale", "linear") == "log"

    ax_ymin = ax.get_ylim()[1] / 10**magnitudes if log_y else 0.0000001
    ax_ymax = get_position(ax_ymin, ax.get_ylim()[1], factor=1 / (1 - whitespace_fraction), logscale=log_y)
    ax_kwargs.update({"ylim": (ax_ymin, ax_ymax)})

    # prioritize style_config ax settings
    ax_kwargs.update(style_config.get("ax_cfg", {}))

    # apply axis kwargs
    apply_ax_kwargs(ax, ax_kwargs)

    # ratio plot
    if not skip_ratio:
        # hard-coded line at 1
        rax.axhline(y=1.0, linestyle="dashed", color="gray")
        rax_kwargs = {
            "ylim": (0.72, 1.28),
            "ylabel": "Ratio",
            "xlabel": "Variable",
            "yscale": "linear",
        }
        rax_kwargs.update(style_config.get("rax_cfg", {}))

        # apply axis kwargs
        apply_ax_kwargs(rax, rax_kwargs)

        # remove x-label from main axis
        if "xlabel" in rax_kwargs:
            ax.set_xlabel("")

    # label alignment
    fig.align_labels()

    # legend
    if not skip_legend:
        # resolve legend kwargs
        legend_kwargs = {
            "ncols": 1,
            "loc": "upper right",
        }
        legend_kwargs.update(style_config.get("legend_cfg", {}))

        if "title" in legend_kwargs:
            legend_kwargs["title"] = remove_label_placeholders(legend_kwargs["title"])

        # retrieve the legend handles and their labels
        handles, labels = ax.get_legend_handles_labels()

        # custom argument: entries_per_column
        n_cols = legend_kwargs.get("ncols", 1)
        entries_per_col = legend_kwargs.pop("cf_entries_per_column", None)
        if callable(entries_per_col):
            entries_per_col = entries_per_col(ax, handles, labels, n_cols)
        if entries_per_col and n_cols > 1:
            if isinstance(entries_per_col, (list, tuple)):
                assert len(entries_per_col) == n_cols
            else:
                entries_per_col = [entries_per_col] * n_cols
            # fill handles and labels with empty entries
            max_entries = max(entries_per_col)
            empty_handle = ax.plot([], label="", linestyle="None")[0]
            for i, n in enumerate(entries_per_col):
                for _ in range(max_entries - min(n, len(handles) - sum(entries_per_col[:i]))):
                    handles.insert(i * max_entries + n, empty_handle)
                    labels.insert(i * max_entries + n, "")

        # custom hook to adjust handles and labels
        update_handles_labels = legend_kwargs.pop("cf_update_handles_labels", None)
        if callable(update_handles_labels):
            update_handles_labels(ax, handles, labels, n_cols)

        # interpret placeholders
        apply = []
        if legend_kwargs.pop("cf_short_labels", False):
            apply.append("SHORT")
        if legend_kwargs.pop("cf_line_breaks", False):
            apply.append("BREAK")
        labels = [apply_label_placeholders(label, apply=apply) for label in labels]

        # drop remaining placeholders
        labels = list(map(remove_label_placeholders, labels))

        # make legend using ordered handles/labels
        ax.legend(handles, labels, **legend_kwargs)

    # custom annotation
    log_x = style_config.get("ax_cfg", {}).get("xscale", "linear") == "log"
    annotate_kwargs = {
        "text": "",
        "xy": (
            get_position(*ax.get_xlim(), factor=0.05, logscale=log_x),
            get_position(*ax.get_ylim(), factor=0.95, logscale=log_y),
        ),
        "xycoords": "data",
        "color": "black",
        "fontsize": 22,
        "horizontalalignment": "left",
        "verticalalignment": "top",
    }
    annotate_kwargs.update(style_config.get("annotate_cfg", {}))
    ax.annotate(**annotate_kwargs)

    # cms label
    if cms_label != "skip":
        cms_label_kwargs = get_cms_label(ax, cms_label)

        cms_label_kwargs.update(style_config.get("cms_label_cfg", {}))
        mplhep.cms.label(**cms_label_kwargs)

    # finalization
    fig.tight_layout()

    return fig, axs