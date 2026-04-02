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


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def unrolling_2D(
        task: law.Task,
        hists: dict[od.Process, Any],
        **kwargs
    ) -> dict[od.Process, Any, Any]:

        for config, proc_dict in hists.items():
            for process, h in proc_dict.items():

                cat_axis = h.axes["category"]
                shift_axis = h.axes["shift"]
                top_axis = h.axes["fit_Top1_mass_percentile"]
                w_axis = h.axes["reco_W_mass_avg_percentile"]

                # ---- Define rebin groups ----
                # 8 → 6 (top)
                top_groups = [
                    [0], [1], [2], [3], [4], [5, 6, 7]
                ]

                # 8 → 3 (W)
                w_groups = [
                    [0, 1, 2],
                    [3, 4],
                    [5, 6, 7],
                ]

                n_unrolled = len(top_groups) * len(w_groups)

                # Create new unrolled axis
                unrolled_axis = hist.axis.Regular(
                    n_unrolled,
                    0,
                    n_unrolled,
                    name="unrolled"
                )

                new_hist = hist.Hist(
                    cat_axis,
                    shift_axis,
                    unrolled_axis,
                    storage=hist.storage.Weight(),
                )

                values = h.values(flow=False)
                variances = h.variances(flow=False)

                for i_cat in range(len(cat_axis)):
                    for i_shift in range(len(shift_axis)):

                        unroll_index = 0

                        for top_bins in top_groups:
                            for w_bins in w_groups:

                                val = 0.0
                                var = 0.0

                                for t in top_bins:
                                    for w in w_bins:
                                        val += values[i_cat, i_shift, t, w]
                                        var += variances[i_cat, i_shift, t, w]

                                new_hist.view(flow=False)[
                                    i_cat, i_shift, unroll_index
                                ] = (val, var)

                                unroll_index += 1

                proc_dict[process] = new_hist

        return hists

    # add the hook
    analysis_inst.x.hist_hooks["unrolling_2D"] = unrolling_2D
