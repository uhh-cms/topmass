# coding: utf-8

"""
Lepton selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge"
    },
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    electron_min_pt = 23,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    electron_mask = (events.Electron.pt > electron_min_pt) & (abs(events.Electron.eta) < 2.4) & (events.Electron.charge == 1)
    
    # pt sorted indices to convert mask
    e_indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    electron_indices = e_indices[electron_mask][:,:2]
    electron_sel = ak.num(electron_indices, axis=-1) >= 2
    
    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"electron": electron_sel},
        
        objects={"Electron": {"Electron": electron_indices},},
    )

@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass"
    },
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    muon_min_pt: int,
    muon_min_eta,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    muon_mask = (events.Muon.pt > muon_min_pt) & (abs(events.Muon.eta) < muon_min_eta)
    
    # pt sorted indices to convert mask
    mu_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    muon_indices = mu_indices[muon_mask]
    muon_sel = ak.sum(muon_mask, axis=1) >= 2

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"muon": muon_sel},
        
        objects={"Muon": {"muon": muon_indices},},
    )