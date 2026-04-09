
# coding: utf-8

"""
Histogram hook for QCD data-driven estimation.
"""

from __future__ import annotations


import law
import order as od

from columnflow.util import maybe_import
from columnflow.types import Any

np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)


def normalize_single_variable_axis(h):
    # Find the axis to normalize (exclude bookkeeping axes)
    exclude = {"category", "shift"}
    var_axes = [ax for ax in h.axes if ax.name not in exclude]

    if len(var_axes) != 1:
        raise ValueError(
            f"Expected exactly 1 variable axis, got {len(var_axes)}")

    axis = var_axes[0]
    widths = axis.widths

    values = h.values()
    variances = h.variances() if h.variances() is not None else None

    # Build broadcast shape
    shape = [1] * values.ndim
    axis_index = h.axes.index(axis.name)
    shape[axis_index] = len(widths)

    widths = widths.reshape(shape)

    # Apply normalization
    h.values()[...] = values / widths

    if variances is not None:
        h.variances()[...] = variances / (widths ** 2)

    return h


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def divide_by_bin_width(
        task: law.Task,
        hists: dict[od.Process, Any],
        **kwargs,
    ) -> dict[od.Process, Any, Any]:
        return hists

    # add the hook
    analysis_inst.x.hist_hooks["divide_by_bin_width"] = divide_by_bin_width
