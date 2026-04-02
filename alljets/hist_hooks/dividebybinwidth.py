
# coding: utf-8

"""
Histogram hook for QCD data-driven estimation.
"""

from __future__ import annotations

from collections import defaultdict

import law
import order as od
import scinum as sn

from columnflow.util import maybe_import, DotDict
from columnflow.types import Any

np = maybe_import("numpy")
hist = maybe_import("hist")


logger = law.logger.get_logger(__name__)

def normalize_single_hist(h):
    # import IPython
    # IPython.embed()
# Find Variable axis
    
    from hist.axis import Variable

    # Find the index of the Variable axis dynamically
    var_axis_index = next(i for i, ax in enumerate(h.axes) if isinstance(ax, Variable))

    # Get bin widths
    bin_widths = h.axes[var_axis_index].widths  # property, no ()

    # Reshape for broadcasting
    shape = [1] * len(h.axes)
    shape[var_axis_index] = -1
    bin_widths_reshaped = bin_widths.reshape(shape)

    # Normalize histogram values in place
    h.view()[:] /= bin_widths_reshaped

    return h


    

def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def divide_by_bin_width(
        task: law.Task,
        hists: dict[od.Process, Any],
        **kwargs
    ) -> dict[od.Process, Any, Any]:

        for config, proc_dict in hists.items():
            for process, h in proc_dict.items():
                normalize_single_hist(h)
               

        return hists

    # add the hook
    analysis_inst.x.hist_hooks["divide_by_bin_width"] = divide_by_bin_width
