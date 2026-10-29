# coding: utf-8

"""
Prefit plots from an inference model, with the tt contribution split by matching type.

The matching axis is purely a visual decomposition:
the parent PlotShiftedVariablesFromModel1D performs all transformations,
nuisance handling and process merging on the total tt histogram first.

Afterwards, the transformed tt histogram is re-expanded using the nominal
matching fractions. Summing the matching components therefore reproduces
the parent tt histogram exactly in value, while the parent histogram remains
the authoritative source for uncertainties.
"""

from __future__ import annotations

from columnflow.tasks.plotting import PlotShiftedVariablesFromModel1D
from columnflow.util import maybe_import

hist = maybe_import("hist")
np = maybe_import("numpy")


class PlotShiftedVariablesFromModelMatching1D(PlotShiftedVariablesFromModel1D):

    task_namespace = "cf"

    matching_axis = "fit_combination_type"
    matching_processes = ("tt",)

    plot_function = PlotShiftedVariablesFromModel1D.plot_function.copy(
        default="alljets.plotting.plot_prefit_matching.plot_prefit_matching",
        add_default_to_description=True,
    )

    def _matching_key(self, config_inst, proc_inst) -> str | None:
        """
        Return the process name under which the matching process appears
        after process merging.
        """
        if self.merge_processes:
            for name in self.merge_processes:
                target = config_inst.get_process(name)

                if proc_inst == target or target.has_process(proc_inst, deep=True):
                    return name if name in self.matching_processes else None

            return None

        for name in self.matching_processes:
            target = config_inst.get_process(name)

            if proc_inst == target or target.has_process(proc_inst, deep=True):
                return proc_inst.name

        return None

    def update_hists_before_config_merging(
        self,
        hists,
        category_name,
        variable_name,
    ):
        ax_name = self.matching_axis

        # 1. Store the NOMINAL matching decomposition.
        # At this point the histogram is still the original input
        # histogram, before ShapeTransformer / process merging.

        nominal_split = {}
        match_axes = {}

        for config_inst, proc_hists in hists.items():
            for proc_inst, h in list(proc_hists.items()):

                if ax_name not in [a.name for a in h.axes]:
                    continue

                key_name = self._matching_key(config_inst, proc_inst)

                if key_name is not None:

                    h_nom = h[{"shift": hist.loc("nominal")}]

                    other_axes = [
                        a.name for a in h_nom.axes
                        if a.name != ax_name
                    ]

                    split = h_nom.project(
                        *other_axes,
                        ax_name,
                    ).values()

                    key = (config_inst.name, key_name)

                    if key in nominal_split:
                        nominal_split[key] += split
                    else:
                        nominal_split[key] = split.copy()

                    match_axes[key] = h.axes[ax_name]

                # Collapse matching axis BEFORE calling the parent.
                # This guarantees that ShapeTransformer sees exactly the same
                # 1D histogram as in PlotShiftedVariablesFromModel1D.

                proc_hists[proc_inst] = h[{ax_name: sum}]

        # 2. Let the parent do EVERYTHING physics-related:
        # - nuisance transformations
        # - creation of shift bins
        # - process merging
        # - statistical variances

        hists = super().update_hists_before_config_merging(
            hists,
            category_name,
            variable_name,
        )

        # 3. Re-expand ONLY the matching decomposition.
        for config_inst, proc_hists in hists.items():

            for proc_inst, h1d in list(proc_hists.items()):

                key = (config_inst.name, proc_inst.name)

                if key not in nominal_split:
                    continue

                split = nominal_split[key]

                total = split.sum(axis=-1, keepdims=True)

                frac = np.divide(
                    split,
                    total,
                    out=np.zeros_like(split, dtype=float),
                    where=total != 0,
                )

                new_h = hist.Hist(
                    *h1d.axes,
                    match_axes[key],
                    storage=hist.storage.Weight(),
                )

                # h1d axes: (shift, variable)
                # frac axes: (variable, matching)
                new_h.view()["value"] = (
                    h1d.values()[..., None] * frac[None]
                )

                new_h.view()["variance"] = (
                    h1d.variances()[..., None] * frac[None] ** 2
                )

                proc_hists[proc_inst] = new_h

        return hists
