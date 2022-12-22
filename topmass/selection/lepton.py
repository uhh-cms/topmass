# coding: utf-8

"""
Lepton selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
import IPython

np = maybe_import("numpy")
ak = maybe_import("awkward")

coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

def invariant_mass(events: ak.Array):
    
    empty_events = ak.zeros_like(events, dtype=np.uint16)[:,0:0]
    where = (ak.num(events,axis=1) == 2)
    events_2 = ak.where(where, events, empty_events)
    mass = ak.fill_none(ak.firsts((1*events_2[:,:1]+1*events_2[:,1:2]).mass), 0)
    return mass

@selector(uses={"Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge"})
def electron_selection(self: Selector, events: ak.Array, **kwargs):

    mask = ((events.Electron.pt > 20) &
            (abs(events.Electron.eta) < 2.4) &
            ((abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.5660))
           )
    
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    indices = indices[mask]
    # build and return selection results plus new columns (src -> dst -> indices)
    return indices



@selector(uses={"Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass","Muon.charge", "Muon.tightId",})
def muon_selection(self: Selector, events: ak.Array, **kwargs):

    mask = ((events.Muon.pt > 20) &
            (abs(events.Muon.eta) < 2.4) &
            (events.Muon.tightId == 1))
    
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    indices = indices[mask]
    # build and return selection results plus new columns (src -> dst -> indices)
    return indices



@selector(uses={muon_selection,electron_selection,
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass","Electron.charge",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass","Muon.charge"},
         produces={
        "channel_id"})
def l_l_selection(self: Selector, events: ak.Array, **kwargs)-> tuple[ak.Array, SelectionResult]:
    
    # get channels from the config
    ch_ee = self.config_inst.get_channel("ee")
    ch_mumu = self.config_inst.get_channel("mumu")
    ch_emu = self.config_inst.get_channel("emu")
    
    #get the preselected muons and electrons
    muon_indices = self[muon_selection](events, **kwargs)
    electron_indices = self[electron_selection](events, **kwargs)
    
    # prepare output vectors
    empty_events = ak.zeros_like(1 * events.event, dtype=np.uint16)
    empty_indicies = empty_events[..., None][..., :0]
    channel_id = empty_events
    sel_muon_indices = empty_indicies
    sel_electron_indices = empty_indicies
    
    # exact two electrons of oppsosite charge and no muons
    inv_ee_mass = invariant_mass(events.Electron[:,:2])
    

    where_ee = ((ak.num(electron_indices, axis=1) == 2) &
                (ak.num(muon_indices, axis=1) == 0) &
                (ak.sum(events.Electron.charge, axis=1) == 0) & 
                (inv_ee_mass > 20) &
                ((inv_ee_mass > 106) | 
                 (inv_ee_mass < 76))
                )
                
    #IPython.embed()
    #channel_id = ak.where(where_ee, ch_ee.id, channel_id)
    sel_electron_indices = ak.where(where_ee, electron_indices, sel_electron_indices)
    
    # exact two muons of oppsosite charge and no electrons
    inv_mumu_mass = invariant_mass(events.Muon[:,:2])
    where_mumu = ((ak.num(muon_indices, axis=1) == 2) &
                  (ak.num(electron_indices, axis=1) == 0) &
                  (ak.sum(events.Muon.charge, axis=1) == 0) &
                  (inv_mumu_mass > 20) &
                  ((inv_mumu_mass > 106) | 
                   (inv_mumu_mass < 76))
                 )

    #channel_id = ak.where(where_mumu, ch_mumu.id, channel_id)
    sel_muon_indices = ak.where(where_mumu, muon_indices, sel_muon_indices)
    
    # exact on electron and one muon of total charge 0
    leptons = ak.concatenate((events.Electron[:,:1],events.Muon[:,:1]),axis=1)
    inv_emu_mass = invariant_mass(leptons)
    where_emu = ((ak.num(electron_indices, axis=1) == 1) &
                 (ak.num(muon_indices, axis=1) == 1) &
                 (ak.sum(leptons.charge,axis=1) == 0) &
                 (inv_emu_mass > 20)
                )

    channel_id = ak.where(where_emu, ch_emu.id, channel_id)
    sel_electron_indices = ak.where(where_emu, electron_indices, sel_electron_indices)
    sel_muon_indices = ak.where(where_emu, muon_indices, sel_muon_indices)

    events = set_ak_column(events, "channel_id", channel_id)

    return events, SelectionResult(
        steps={"leptons":  channel_id != 0 },
        
        objects={"Muon": {"Muon" : sel_muon_indices} ,"Electron": {"Electron": sel_electron_indices}})

"""
def l_l_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    
    mu_mask = (events.Muon.pt > 17) & (abs(events.Muon.eta) < 2.4)
    e_mask = (events.Electron.pt > 17) & (abs(events.Electron.eta) < 2.4) 
    
    IPython.embed()
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    mu_mu_indices = indices[mask]
    indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    e_e_indices = indices[mask]
    
    mu_mu_sel = ((ak.num(mu_mu_indices, axis=-1)+ak.num(e_e_indices, axis=-1)) == 2) & (ak.sum(events.Muon.charge,axis=1) == 0)

    e_e_sel = (ak.num(e_e_indices, axis=-1) >= 2) & (ak.sum(events.Electron.charge,axis=1) == 0)
    
    e_mu_sel = (ak.num(mu_indices, axis=-1) >= 1) & (ak.num(e_indices, axis=-1) >= 1) & ((events.Electron.charge+events.Muon.charge) == 0)
    
    
    events = set_ak_column(events, "Lepton_Pair", 0)
    # build and return selection results plus new columns (src -> dst -> indices)
    return events, SelectionResult(
        steps={"e_mu":  e_mu_sel,"mu_mu":  mu_mu_sel,"e_e":  e_e_sel},
        
        objects={"Muon": {"E_Mu": mu_indices,"E_E": e_e_indices} ,"Electron": {"E_Mu": e_indices}}
    )

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