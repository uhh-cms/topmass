# coding: utf-8

"""
Lepton selection methods.
"""

from __future__ import annotations

import law
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import sorted_indices_from_mask
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


logger = law.logger.get_logger(__name__)


@selector(
    uses={
        "Electron.{pt,eta,phi,mass,cutBased}",
    },
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Selector for veto electrons as defined in the Run 2 lepton+jets analysis note [1].

    In that analysis, the veto region is defined inclusively, such that signal leptons also satisfy the veto selection.

    For the all-jets analysis, we select only events with no electron passing the veto selection.

    Reference:
    [1] https://cms.cern.ch/iCMS/user/noteinfo?cmsnoteid=CMS%20AN-2024/119
    """

    # year-dependent eta cut
    if self.config_inst.campaign.x.year == 2016:
        eta_max = 2.4
    else:
        eta_max = 2.5

    veto_mask = (
        (events.Electron.pt > 15.0) &
        (abs(events.Electron.eta) < eta_max) &
        (events.Electron.cutBased >= 2)
    )

    return veto_mask


@selector(
    uses={"Muon.{pt,eta,phi,mass,looseId,pfIsoId}"},
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Selector for veto muons as defined in the Run 2 lepton+jets analysis note [1].

    In that analysis, the veto region is defined inclusively, such that signal leptons also satisfy the veto selection.

    For the all-jets analysis, we select only events with no muon passing the veto selection.

    Reference:
    [1] https://cms.cern.ch/iCMS/user/noteinfo?cmsnoteid=CMS%20AN-2024/119
    """

    veto_mask = (
        (events.Muon.pt > 15.0) &
        (abs(events.Muon.eta) < 2.4) &
        (events.Muon.looseId == 1) &
        (events.Muon.pfIsoId >= 2)
    )

    return veto_mask


@selector(
    uses={electron_selection, muon_selection},
)
def lepton_selection(self, events: ak.Array, **kwargs):
    """
    Event-level lepton veto selection combining electron and muon veto definitions.

    The electron and muon veto definitions follow the Run 2 lepton+jets analysis note [1],
    where the veto regions are defined inclusively (i.e. signal leptons satisfy the veto criteria).

    In this analysis, we enforce orthogonality by requiring that events contain no electrons or muons
    satisfying the veto-lepton definitions.

    Reference:
    [1] https://cms.cern.ch/iCMS/user/noteinfo?cmsnoteid=CMS%20AN-2024/119
    """
    electron_mask = self[electron_selection](events, **kwargs)
    muon_mask = self[muon_selection](events, **kwargs)

    lepton_veto = (
        (ak.sum(electron_mask, axis=1) == 0) &
        (ak.sum(muon_mask, axis=1) == 0)
    )

    return events, SelectionResult(
        steps={"Lepton_Veto": lepton_veto},
        objects={
            "Electron": {"Electron": sorted_indices_from_mask(electron_mask, events.Electron.pt)},
            "Muon": {"Muon": sorted_indices_from_mask(muon_mask, events.Muon.pt)},
        },
    )
