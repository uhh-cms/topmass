# coding: utf-8

"""
Plotting helpers for b-tag efficiency maps.
"""

from __future__ import annotations

from collections.abc import Mapping

import law
from modules.columnflow.columnflow.plotting.plot_util import prepare_style_config
from modules.columnflow.columnflow.plotting.plot_functions_2d import plot_2d
import order as od

from modules.columnflow.columnflow.plotting.plot_util import get_cms_label
from columnflow.util import maybe_import

hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")

logger = law.logger.get_logger(__name__)


FLAVOR_LABELS = {
    0: r"light",
    4: r"c",
    5: r"b",
}

PT_LABEL = r"$p_{\mathrm{T}}$ [GeV]"
ETA_LABEL = r"$|\eta|$"


def _pick_axis_name(h: hist.Hist, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in h.axes.name:
            return name
    return None


def _set_hist_values(h: hist.Hist, values: np.ndarray) -> None:
    """
    Write `values` into a histogram's view, handling both weighted
    (structured, with .value/.variance) and plain storage.
    """
    view = h.view()
    if hasattr(view, "value"):
        view.value[...] = values
        view.variance[...] = 0.0
    else:
        view[...] = values


def _get_total_and_wp_categories(
    h: hist.Hist,
    wp_axis: str,
    wp_label: str = "tight",
) -> tuple[str, str]:
    """
    Get the total and working point category names from the wp axis.
    Default wp_label is 'tight'.
    """
    categories = list(h.axes[wp_axis])

    total_cat = None
    wp_cat = None

    for cat in categories:
        cat_lower = cat.lower()
        if cat_lower == "total":
            total_cat = cat
        if cat_lower == wp_label.lower():
            wp_cat = cat

    if total_cat is None and len(categories) > 0:
        total_cat = categories[0]
        logger.warning(f"Could not find 'total' category, using '{total_cat}' instead")

    if wp_cat is None:
        for cat in categories:
            cat_lower = cat.lower()
            if cat_lower in ["medium", "loose"]:
                wp_cat = cat
                logger.warning(f"Could not find '{wp_label}', using '{wp_cat}' instead")
                break

        if wp_cat is None and len(categories) > 1:
            wp_cat = categories[1]
            logger.warning(f"Could not find '{wp_label}', using '{wp_cat}' instead")
        elif wp_cat is None and len(categories) == 1:
            wp_cat = categories[0]
            logger.warning(f"Only one category found, using '{wp_cat}' for both total and wp")

    return total_cat, wp_cat


def _efficiency_hist(
    h: hist.Hist,
    wp_axis: str,
    wp_label: str = "tight",
) -> tuple[hist.Hist, str, str]:
    """
    Compute the wp / total efficiency as a hist.Hist carrying the same remaining
    axes as the input (e.g. pt x eta), so it can be fed straight into plot_2d.
    plot_2d requires Weight() storage (it reads h_view.value/.variance directly),
    so we rebuild explicitly with that storage rather than copying wp_h's storage.
    """
    total_cat, wp_cat = _get_total_and_wp_categories(h, wp_axis, wp_label)

    if total_cat == wp_cat:
        raise ValueError(f"Cannot compute efficiency: total and wp categories are the same ('{total_cat}')")

    wp_h = h[{wp_axis: hist.loc(wp_cat)}]
    total_h = h[{wp_axis: hist.loc(total_cat)}]

    wp_vals = wp_h.values()
    total_vals = total_h.values()

    eff = np.divide(
        wp_vals,
        total_vals,
        out=np.full_like(wp_vals, np.nan, dtype=float),
        where=total_vals > 0,
    )

    # plot_2d expects Weight() storage; build eff_h with that explicitly rather
    # than inheriting wp_h's storage, which may be plain (unweighted) Double()
    eff_h = hist.Hist(*wp_h.axes, storage=hist.storage.Weight())
    eff_h.view().value[...] = eff
    # nonzero variance where we have a real number, so plot_2d's
    # `variance == 0` -> nan masking doesn't wipe out valid efficiency bins
    eff_h.view().variance[...] = np.where(np.isnan(eff), 0.0, 1e-12)

    return eff_h, total_cat, wp_cat


def btag_efficiency(
    hists: Mapping[str, hist.Hist],
    config_inst: od.Config,
    category_inst: od.Category,
    shift_insts: list[od.Shift],
    flavor: int,
    wp_label: str = "tight",
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool = False,
    zlim: tuple | None = None,
    zscale: str | None = None,
    cms_label: str | None = None,
    yscale: str | None = None,
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    if not hists:
        raise ValueError("no histograms were provided for btag efficiency plotting")

    first_hist = next(iter(hists.values()))

    pt_axis_name = _pick_axis_name(first_hist, ("pt", "x"))
    eta_axis_name = _pick_axis_name(first_hist, ("abs_eta", "eta", "y"))
    wp_axis = _pick_axis_name(first_hist, ("wp",))
    flavor_axis = _pick_axis_name(first_hist, ("flavor",))

    if wp_axis is None:
        raise ValueError("expected a 'wp' axis with categories")
    if pt_axis_name is None or eta_axis_name is None:
        raise ValueError("expected pt ('pt'/'x') and eta ('abs_eta'/'eta'/'y') axes")

    # drop the first (lowest) pt bin everywhere
    pt_slice = {pt_axis_name: slice(1, None)}
    first_hist = first_hist[pt_slice]

    pt_variable_inst = od.Variable(
        name="pt",
        expression="pt",
        binning=list(first_hist.axes[pt_axis_name].edges),
        x_title=PT_LABEL,
        log_x=True,
    )
    eta_variable_inst = od.Variable(
        name="abs_eta",
        expression="abs_eta",
        binning=list(first_hist.axes[eta_axis_name].edges),
        x_title=ETA_LABEL,
    )

    combined_h = None
    for group_name, hist_obj in hists.items():
        h = hist_obj
        if flavor_axis is not None:
            h = h[{flavor_axis: hist.loc(flavor)}]
        combined_h = h if combined_h is None else combined_h + h

    eff_h, total_cat, wp_cat = _efficiency_hist(combined_h, wp_axis, wp_label)

    resolved_procs = [
        config_inst.get_process(name, default=None) or name
        for name in hists.keys()
    ]
    combined_label = ", ".join(
        p.label if isinstance(p, od.Process) else str(p)
        for p in resolved_procs
    )
    is_data = any(getattr(p, "is_data", False) for p in resolved_procs)

    combined_proc = od.Process(
        name="btag_eff_combined",
        id=int(1e7) + flavor,
        label=combined_label,
        is_data=is_data,
    )
    ratio_hists: dict[od.Process, hist.Hist] = {combined_proc: eff_h}

    pt_edges = pt_variable_inst.binning
    flavor_label = FLAVOR_LABELS.get(flavor, str(flavor))

    process_labels = [
        p.label if isinstance(p, od.Process) else str(p)
        for p in resolved_procs
    ]
    process_labels_joined = ", ".join(process_labels)

    base_style_config = prepare_style_config(
        config_inst=config_inst,
        category_inst=category_inst,
        variable_inst=pt_variable_inst,
        density=density,
        shape_norm=shape_norm,
        yscale=yscale,
    )

    base_style_config = {"cms_label_cfg": base_style_config.get("cms_label_cfg", {})}

    if cms_label is None:
        # plot_2d's own label_options only has "pw"/"simpw", not "datapw"
        cms_label = "pw" if is_data else "simpw"

    title = f"{flavor_label}-flavor jets, {wp_cat} WP"

    default_style_config = law.util.merge_dicts(
        base_style_config,
        {
            "legend_cfg": {
                "title": title,
                "title_fontsize": 24,
                "handles": [mpl.lines.Line2D([0], [0], lw=0)],
                "fontsize": 24,
                "labels": [process_labels_joined],
                "ncol": 1,
                "loc": "upper left",
                "handlelength": 0,
                "handletextpad": 0.2,
                "borderpad": 0.3,
                "alignment": "left",
            },
            "ax_cfg": {
                "xlim": (pt_edges[0], pt_edges[-1]),
            },
        },
        deep=True,
    )

    cms_kwargs = get_cms_label(None, cms_label)
    default_style_config["cms_label_cfg"]["llabel"] = cms_kwargs["llabel"]
    if "exp" in cms_kwargs:
        default_style_config["cms_label_cfg"]["exp"] = cms_kwargs["exp"]

    levels = np.arange(0.0, 1.1, 0.1)

    cmap = plt.get_cmap("viridis", len(levels) - 1)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    style_config = law.util.merge_dicts(
        default_style_config,
        {
            "plot2d_cfg": {
                "cmap": cmap,
                "norm": norm,
            },
        },
        deep=True,
    )

    style_config = law.util.merge_dicts(default_style_config, style_config, deep=True)

    fig, axes = plot_2d(
        hists=ratio_hists,
        config_inst=config_inst,
        category_inst=category_inst,
        shift_insts=shift_insts,
        variable_insts=[pt_variable_inst, eta_variable_inst],
        style_config=style_config,
        density=False,
        shape_norm=shape_norm,
        zscale=zscale or "linear",
        zlim=(0.0, 1.0),
        cms_label=cms_label,
        process_settings=process_settings,
        variable_settings=variable_settings,
        **kwargs,
    )

    for ax in axes:
        ax.set_xlim(pt_edges[0], pt_edges[-1])

    return fig, axes
