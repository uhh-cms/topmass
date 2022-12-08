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

    default_mask = (events.Electron.pt > electron_min_pt) & (abs(events.Electron.eta) < 2.4) 
    positron_mask = (events.Electron.charge == 1) & default_mask
    electron_mask = (events.Electron.charge == -1) & default_mask
    
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    
    positron_indices = indices[positron_mask][:,:1]
    positron_sel = ak.num(positron_indices, axis=-1) >= 1
    
    electron_indices = indices[electron_mask][:,:1]
    electron_sel = ak.num(electron_indices, axis=-1) >= 1
    
    
    print(positron_indices[:30])
    print(electron_indices[:30])
    print(ak.concatenate([positron_indices, electron_indices], axis=1)[:30])
    
    print(ak.max(ak.num(ak.concatenate([positron_indices, electron_indices]),axis=1)))
    print(ak.size(ak.zip([positron_indices, electron_indices], depth_limit=1), axis=0))
    
    e_pair_indices = indices[ak.concatenate([positron_indices, electron_indices], axis=1)] 
    e_pair_sel = ak.num(e_pair_indices, axis=-1) >= 2 
    #e_pair_indices = e_pair_indices[e_pair_sel]
    
    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"electron": e_pair_sel},
        
        objects={"Electron": {"Electron": e_pair_indices},},
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