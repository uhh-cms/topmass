# coding: utf-8

"""
Examples for custom plot functions.
"""

from __future__ import annotations

import law
from columnflow.plotting.plot_util import (apply_density, remove_residual_axis)
from columnflow.util import maybe_import
from modules.columnflow.columnflow.plotting.plot_functions_2d import plot_2d

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
od = maybe_import("order")

logger = law.logger.get_logger(__name__)


def btag_efficiency(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable] | dict[str, list[od.Variable]],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    yscale: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    """
    This function visualizes the matching status of MC events (correct, wrong, unmatched)
    using cumulative stacking. The color scheme is based on previous analyses, e.g.
    https://arxiv.org/pdf/2302.01967v2.

    law run cf.PlotVariables2D --version v1 --config 2017_v9 \
    --dataset tt_fh_powheg --selector no_btag \
    --selector-steps Trigger,HT,jet --producer btag_eff \
    --variables bflav_jet_pt-bflav_jet_eta,bflav_bjet_pt-bflav_bjet_eta \
    --processes tt --multi-variable \
    --plot-function alljets.plotting.btag_eff.btag_efficiency

    Example command to run the plot function. The matching information is stored in
    'fit_combination_type' column. Here, a 2D histogram is created with the
    information of the matching type on an additional axis.
    """

    # Find the index of the two 2D histograms (one for jets, one for b-jets)
    keys = list(hists.keys())
    jet_index = None
    bjet_index = None

    for i, k in enumerate(keys):
        if "bjet" in k:
            bjet_index = i
            bjet_flavour = k.replace("_bjet", "")
        elif "jet" in k:
            jet_index = i
            jet_flavour = k.replace("_jet", "")

    # Check that both were found
    if jet_index is None or bjet_index is None:
        raise ValueError(
            f"Expected one 'jet' and one 'bjet' histogram in hists, "
            f"found jet_index={jet_index}, bjet_index={bjet_index}",
        )

    # Check flavour consistency
    if jet_flavour != bjet_flavour:
        raise ValueError(
            f"Flavour mismatch between jet and bjet histograms: "
            f"jet flavour='{jet_flavour}', bjet flavour='{bjet_flavour}'.",
        )

    # extract the variable instances for jets and b-jets
    vars_jet = variable_insts[keys[jet_index]]

    # Extract per-observable dicts
    hists_jet = hists[keys[jet_index]]
    hists_jet = remove_residual_axis(hists_jet, "shift")

    hists_bjet = hists[keys[bjet_index]]
    hists_bjet = remove_residual_axis(hists_bjet, "shift")

    # Apply variable and density settings to histograms
    hists_jet = apply_density(hists_jet, density)
    hists_bjet = apply_density(hists_bjet, density)

    import numpy as np

    ratio_hists = {}

    for proc in hists_jet:
        # compute efficiency
        h_eff = hists_bjet[proc] / hists_jet[proc].values()

        # mask zero-denominator bins
        mask = hists_jet[proc].values() == 0
        h_view = h_eff.view()
        h_view.value[mask] = np.nan
        h_view.variance[mask] = 0.0

        def reaxis_efficiency(eff_hist, template_hist):
            new_hist = template_hist.copy()
            new_hist.reset()
            new_hist.view().value[:] = eff_hist.view().value
            new_hist.view().variance[:] = eff_hist.view().variance
            return new_hist

        # re-map efficiency onto jet histogram axes
        h_eff_mapped = reaxis_efficiency(h_eff, hists_jet[proc])

        # store in ratio_hists
        ratio_hists[proc] = h_eff_mapped

    vmin, vmax = np.nanmin(h_eff.values()), np.nanmax(h_eff.values())

    print("Minimum efficiency:", vmin)
    print("Maximum efficiency:", vmax)

    fig, (ax,) = plot_2d(
        hists=ratio_hists,
        config_inst=config_inst,
        category_inst=category_inst,
        variable_insts=vars_jet,
        style_config=style_config,
        density=False,
        shape_norm=shape_norm,
        zscale=yscale or "linear",
        process_settings=process_settings,
        variable_settings=variable_settings,
        **kwargs,
    )

    return fig, (ax,)
