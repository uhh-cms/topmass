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


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def unrolling(
        task: law.Task,
        hists: dict[od.Process, Any],
        **kwargs,
    ) -> dict[od.Process, Any, Any]:

        for config, proc_dict in hists.items():
            for process, h in proc_dict.items():
                cat_axis = h.axes["category"]
                shift_axis = h.axes["shift"]
                values = h.values(flow=False)
                variances = h.variances(flow=False)

                # Flatten data axes while keeping category and shift
                flat_values = values.reshape(len(cat_axis), len(shift_axis), -1)
                flat_variances = variances.reshape(len(cat_axis), len(shift_axis), -1)

                # Create new unrolled axis
                n_flat = flat_values.shape[2]
                unrolled_axis = hist.axis.Regular(
                    n_flat,
                    0,
                    n_flat,
                    name="unrolled",
                )

                new_hist = hist.Hist(
                    cat_axis,
                    shift_axis,
                    unrolled_axis,
                    storage=hist.storage.Weight(),
                )

                new_hist[...] = np.stack([flat_values, flat_variances], axis=-1)
                proc_dict[process] = new_hist

        return hists

    # add the hook
    analysis_inst.x.hist_hooks["unrolling"] = unrolling