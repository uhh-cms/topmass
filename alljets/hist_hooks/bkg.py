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


# helper to convert a histogram to a number object containing bin values and uncertainties
# from variances stored in an array of values
def hist_to_num(h: hist.Histogram, unc_name=str(sn.DEFAULT)) -> sn.Number:
    return sn.Number(h.values(flow=True), {unc_name: h.variances(flow=True)**0.5})


# helper to integrate values stored in an array based number object
def integrate_num(num: sn.Number, axis=None) -> sn.Number:
    return sn.Number(
        nominal=num.nominal.sum(axis=axis),
        uncertainties={
            unc_name: (
                (unc_values_up**2).sum(axis=axis)**0.5,
                (unc_values_down**2).sum(axis=axis)**0.5,
            )
            for unc_name, (unc_values_up, unc_values_down) in num.uncertainties.items()
        },
    )


# helper to ensure that a specific category exists on the "category" axis of a histogram
def ensure_category(h: hist.Histogram, category_name: str) -> hist.Histogram:
    cat_axis = h.axes["category"]
    if category_name in cat_axis:
        return h
    dummy_fill = {ax.name: ax[0] for ax in h.axes if ax.name != "category"}
    h.fill(**dummy_fill, category=category_name, weight=0.0)
    return h


def add_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to a analysis.
    """
    def qcd_estimation_per_config(
        task: law.Task,
        config_inst: od.Config,
        hists: dict[od.Process, Any],
    ) -> dict[od.Process, Any]:
        # get the qcd process
        qcd_proc = config_inst.get_process("qcd_est", default=None)

        if not qcd_proc:
            return hists

        # extract all unique category names and verify that the axis order is exactly
        # "category -> shift -> variable" which is needed to insert values at the end
        CAT_AXIS, SHIFT_AXIS, VAR_AXIS = range(3)
        category_names = set()
        for proc, h in hists.items():
            # validate axes
            assert len(h.axes) == 3
            assert h.axes[CAT_AXIS].name == "category"
            assert h.axes[SHIFT_AXIS].name == "shift"
            # get the category axis
            cat_ax = h.axes["category"]
            category_names.update(list(cat_ax))

        # define QCD groups
        qcd_groups: dict[str, dict[str, od.Category]] = defaultdict(DotDict)
        for cat_name in category_names:
            cat_inst = config_inst.get_category(cat_name)

            if cat_inst.has_tag({"0btj"}, mode=all):
                qcd_groups["0btj_bkg"].no_jets = cat_inst
                # qcd_groups[cat_inst.x.qcd_group].no_jets = cat_inst
            elif cat_inst.has_tag({"2btj"}, mode=all):
                qcd_groups["2btj_sig"].two_jets = cat_inst
                # qcd_groups[cat_inst.x.qcd_group].two_jets = cat_inst

        complete_groups = [name for name, cats in qcd_groups.items() if len(cats) > 0]

        if not complete_groups:
            return hists

        # Read data histogram
        data_hists = [h for p, h in hists.items() if p.is_data]
        if not data_hists:
            return hists
        data_hist = sum(data_hists[1:], data_hists[0].copy())

        # Initialize empty QCD histogram, to be filled later on
        hists[qcd_proc] = qcd_hist = data_hist.copy().reset()
        group = qcd_groups["0btj_bkg"]

        # get the corresponding histograms and convert them to number objects, each one storing an array of values
        def get_hist(h: hist.Histogram, region_name: str) -> hist.Histogram:
            h = ensure_category(h, group[region_name].name)
            return h[{"category": hist.loc(group[region_name].name)}]
        no_jets_data = hist_to_num(get_hist(data_hist, "no_jets"), "no_jets")

        # define signal category
        sig_category = qcd_groups["2btj_sig"].get("two_jets")
        cat_axis = qcd_hist.axes["category"]

        # shape TODO: this scaling factor only works, if over/underflow bins are present
        tt_hist = hists[config_inst.get_process("tt", default=None)]
        for i in range(len(tt_hist.axes["category"])):
            if tt_hist.axes["category"][i] == "2btj_sig":
                tt_index = i

        for cat_index in range(cat_axis.size):
            if cat_axis.value(cat_index) == sig_category.name:
                dif = data_hist[cat_index, ...].sum(flow=True).value - tt_hist[tt_index, ...].sum(flow=True).value
                factor = dif / get_hist(data_hist, "no_jets").sum(flow=True).value

        bkg_qcd = factor * no_jets_data  # here you can multiply your shape by a constant factor

        # combine uncertainties and store values in bare arrays
        bkg_qcd_values = bkg_qcd()
        bkg_qcd_variances = bkg_qcd(sn.UP, sn.ALL, unc=True)**2

        # residual zero filling
        zero_mask = bkg_qcd_values <= 0
        bkg_qcd_values[zero_mask] = 1e-5
        bkg_qcd_variances[zero_mask] = 0

        # insert QCD shape into signal category

        stop = 0
        for cat_index in range(cat_axis.size):
            if cat_axis.value(cat_index) == sig_category.name:
                qcd_hist.view(flow=True).value[cat_index, ...] = bkg_qcd_values
                qcd_hist.view(flow=True).variance[cat_index, ...] = bkg_qcd_variances
                stop += 1
            if cat_axis.value(cat_index) == "incl":
                qcd_hist.view(flow=True).value[cat_index, ...] = bkg_qcd_values
                qcd_hist.view(flow=True).variance[cat_index, ...] = bkg_qcd_variances
                stop += 1
            if stop == 2:
                break
        else:
            raise RuntimeError(
                f"could not find index of bin on 'category' axis of qcd histogram {qcd_hist} for category "
                f"{sig_category}",
            )

        hists[qcd_proc] = qcd_hist
        return hists

    def qcd_estimation(
        task: law.Task,
        hists: dict[od.Config, dict[od.Process, Any]],
    ) -> dict[od.Config, dict[od.Process, Any]]:
        return {
            config_inst: qcd_estimation_per_config(task, config_inst, hists[config_inst])
            for config_inst in hists.keys()
        }

    # add the hook
    analysis_inst.x.hist_hooks["qcd"] = qcd_estimation
