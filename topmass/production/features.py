# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

@producer(
    uses={
        "Jet.pt","Bjet.pt", "E_Mu.pt",
    },
    produces={
        "ht", "n_jet", "n_bjet", "n_e_e", "n_e_mu",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_bjet", ak.num(events.Bjet.pt, axis=1))
    events = set_ak_column(events, "n_e_e", ak.num(events.E_Mu.pt, axis=1))
    events = set_ak_column(events, "n_e_mu", ak.num(events.E_Mu.pt, axis=1))
    return events

@producer(
    uses={
        "Bjet.pt", "Electron.pt","Bjet.mass","Electron.mass"
    },
    produces={"2"
    },
)
def lb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    return events
"""
@producer(
    uses={
        "Jet.pt","Bjet.pt", "E_Mu.pt","E_E.pt",
    },
    produces={
        "ht", "n_jet", "n_bjet", "n_e_e", "n_e_mu",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_bjet", ak.num(events.Bjet.pt, axis=1))
    events = set_ak_column(events, "n_e_e", ak.num(events.E_E.pt, axis=1))
    events = set_ak_column(events, "n_e_mu", ak.num(events.E_Mu.pt, axis=1))
    return events
    
@producer(
    uses={
        "Bjet.pt", "Electron.pt","Bjet.mass","Electron.mass"
    },
    produces={
        "m_min_lb",
    },
)
def lb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Bjet"] = ak.with_name(events.Bjet, "PtEtaPhiMLorentzVector")
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["E_Mu"] = ak.with_name(events.E_Mu, "PtEtaPhiMLorentzVector")
    
    if ak.any(ak.num(events.Bjet, axis=-1) != 2):
        raise Exception("In features.py: there should be exactly 2 bjets in each B_jet")
        
        
    if ak.any(ak.num(events.E_Mu, axis=-1) != 2):
        raise Exception("In features.py: there should be exactly 2 leptons in each E_pair")
    
    m_bjet_e=[events.Bjet, events.E_Mu]
    mleft, mright = ak.unzip(ak.cartesian(m_bjet_e, axis=1))
    m_min_lb = ak.min((mleft+mright).mass, axis=1)

    #m=(events.Bjet[:, 0] + events.Electron[:, 1]).mass
    
    events = set_ak_column(events, "m_min_lb", m_min_lb)
    
    return events
"""

@producer(
    uses={
        mc_weight, category_ids, "Jet.pt",
    },
    produces={
        mc_weight, category_ids, "cutflow.n_jet", "cutflow.ht", "cutflow.jet1_pt",
    },
)
def cutflow_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[mc_weight](events, **kwargs)
    events = self[category_ids](events, **kwargs)

    events = set_ak_column(events, "cutflow.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "cutflow.jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))

    return events

