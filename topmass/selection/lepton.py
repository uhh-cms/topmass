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
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge",
    },
)

def e_e_selection(
    self: Selector,
    events: ak.Array,
    electron_min_pt = 23,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    mask = (events.Electron.pt > electron_min_pt) & (abs(events.Electron.eta) < 2.4) 
   
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    e_e_indices = indices[mask][:,:2]
    e_e_sel = (ak.num(e_e_indices, axis=-1) >= 2) & (ak.sum(events.Electron.charge,axis=1) == 0)

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_e":  e_e_sel},
        
        objects={"Electron": {"E_E": e_e_indices},}
    )

@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass","Muon.charge",
    },
)

def mu_mu_selection(
    self: Selector,
    events: ak.Array,
    muon_min_pt = 23,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    mask = (events.Muon.pt > muon_min_pt) & (abs(events.Muon.eta) < 2.4) 
   
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    mu_mu_indices = indices[mask][:,:2]
    mu_mu_sel = (ak.num(mu_mu_indices, axis=-1) >= 2) & (ak.sum(events.Muon.charge,axis=1) == 0)

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"mu_mu":  mu_mu_sel},
        
        objects={"Muon": {"Mu_Mu": mu_mu_indices},}
    )

@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass","Muon.charge",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge",
    },
)

def e_mu_selection(
    self: Selector,
    events: ak.Array,
    muon_min_pt = 23,
    electron_min_pt = 23
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    mu_mask = (events.Muon.pt > muon_min_pt) & (abs(events.Muon.eta) < 2.4)
    e_mask = (events.Electron.pt > electron_min_pt) & (abs(events.Electron.eta) < 2.4) 
   
    # pt sorted indices to convert mask
    mu_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    mu_indices = mu_indices[mu_mask][:,:1]
    e_indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    e_indices = e_indices[e_mask][:,:1]
    
    e_mu_sel = (ak.num(mu_indices, axis=-1) >= 1) & (ak.num(e_indices, axis=-1) >= 1) & ((events.Electron.charge+events.Muon.charge) == 0)

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_mu":  e_mu_sel},
        
        objects={"Muon": {"E_Mu": mu_indices},"Electron": {"E_Mu": e_indices}}
    )
"""
@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge",
    },
)

def e_e_selection(
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
    electron_indices = indices[electron_mask][:,:1]
    
    e_e_indices = ak.concatenate([positron_indices, electron_indices], axis=1)
    e_e_sel = ak.num(e_e_indices, axis=-1) >= 2 

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_e":  e_e_sel},
        
        objects={"Electron": {"E_E": e_e_indices},}
    )


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass","Muon.charge"
    },
)
def e_mu_selection(
    self: Selector,
    events: ak.Array,
    electron_min_pt = 23,
    muon_min_pt = 23,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    default_electron_mask = (events.Electron.pt > electron_min_pt) & (abs(events.Electron.eta) < 2.4)
    default_muon_mask = (events.Muon.pt > electron_min_pt) & (abs(events.Muon.eta) < 2.4) 
    positron_mask = (events.Electron.charge == 1) & default_electron_mask
    electron_mask = (events.Electron.charge == -1) & default_electron_mask
    anti_muon_mask = (events.Muon.charge == 1) & default_muon_mask
    muon_mask = (events.Muon.charge == -1) & default_muon_mask

    # pt sorted indices to convert mask
    indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    positron_indices = indices[positron_mask][:,:1]
    electron_indices = indices[electron_mask][:,:1]
    
    indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    muon_indices = indices[muon_mask][:,:1]
    anti_muon_indices = indices[anti_muon_mask][:,:1]
    
    
    e_mu_indices = ak.concatenate([positron_indices, muon_indices], axis=1)
    mu_e_indices = ak.concatenate([electron_indices, anti_muon_indices], axis=1)
    
    e_mu_sel = (ak.num(mu_e_indices, axis=-1) >= 2) | (ak.num(e_mu_indices, axis=-1) >= 2)
    
    
    e_indices = ak.concatenate([electron_indices, positron_indices], axis=1)
    
    mu_indices = ak.concatenate([muon_indices,anti_muon_indices],axis=1)
    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_mu": e_mu_sel},
        
        objects={"Electron": {"E_Mu": e_indices},"Muon": {"Mu_E": mu_indices}}
    )


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
    
    e_pair_indices = indices[ak.concatenate([positron_indices, electron_indices], axis=1)]
    e_pair_sel = ak.num(e_pair_indices, axis=-1) >= 2 

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_pair": e_pair_sel, "electron": electron_sel, "positron": positron_sel},
        
        objects={"Electron": {"E_pair": e_pair_indices, "Electron": electron_indices, "Positron": positron_indices},}
    )

@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge"
    },
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    muon_min_pt = 23,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    default_mask = (events.Muon.pt > muon_min_pt) & (abs(events.Muon.eta) < 2.4) 
    muon_pos_mask = (events.Muon.charge == 1) & default_mask
    muon_neg_mask = (events.Muon.charge == -1) & default_mask
    
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    
    muon_pos_indices = indices[muon_pos_mask][:,:1]
    muon_pos_sel = ak.num(muon_pos_indices, axis=-1) >= 1
    
    muon_neg_indices = indices[muon_neg_mask][:,:1]
    muon_neg_sel = ak.num(muon_neg_indices, axis=-1) >= 1
    
    e_pair_indices = indices[ak.concatenate([positron_indices, electron_indices], axis=1)]
    e_pair_sel = ak.num(e_pair_indices, axis=-1) >= 2 

    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_pair": e_pair_sel, "electron": electron_sel, "positron": positron_sel},
        
        objects={"Electron": {"E_pair": e_pair_indices, "Electron": electron_indices, "Positron": positron_indices},}
    )"""