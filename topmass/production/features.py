# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        "Jet.pt",
        "Bjet.pt",
        "Electron.pt",
        "Muon.pt",
        "Electron.eta",
        "Muon.eta",
        "PV.npvs",
        "PV.npvsGood",

    },
    produces={
        "ht",
        "n_jet",
        "n_bjet",
        "n_electron",
        "n_muon",
        "lepton_pt",
        "lepton_eta",
        "nPV",
        "nPVGood",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_bjet", ak.num(events.Bjet.pt, axis=1))
    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=1))
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=1))

    events = set_ak_column(events, "nPV", events.PV.npvs)
    events = set_ak_column(events, "nPVGood", events.PV.npvsGood)

    lepton_pt = ak.concatenate((events.Muon.pt, events.Electron.pt), axis=1)
    lepton_eta = ak.concatenate((events.Muon.eta, events.Electron.eta), axis=1)

    sort = ak.argsort(lepton_pt, axis=-1, ascending=False)

    events = set_ak_column(events, "lepton_pt", lepton_pt[sort])
    events = set_ak_column(events, "lepton_eta", lepton_eta[sort])

    return events


@producer(
    uses={
        "Bjet.pt",
        "Bjet.mass",
        "Bjet.eta",
        "Bjet.phi",
        "channel_id",
        "Electron.pt",
        "Electron.mass",
        "Electron.eta",
        "Electron.phi",
        "Muon.pt",
        "Muon.mass",
        "Muon.eta",
        "Muon.phi",
    },
    produces={
        "m_min_lb",
    },
)
def lb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Bjet"] = ak.with_name(events.Bjet, "PtEtaPhiMLorentzVector")
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Electron"] = ak.with_name(events.Electron, "PtEtaPhiMLorentzVector")
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Muon"] = ak.with_name(events.Muon, "PtEtaPhiMLorentzVector")

    leptons = ak.concatenate((1 * events.Electron, 1 * events.Muon), axis=1)

    if ak.any(ak.num(events.Bjet, axis=-1) != 2):
        raise Exception("In features.py: there should be exactly 2 bjets in each B_jet")

    if ak.any(ak.num(leptons, axis=-1) != 2):
        raise Exception(
            "In features.py: there should be exactly 2 leptons in each lepton pair",
        )

    bjet_l = [1 * events.Bjet, leptons]

    mleft, mright = ak.unzip(ak.cartesian(bjet_l, axis=1))
    m_min_lb = ak.min((mleft + mright).mass, axis=1)

    # m=(events.Bjet[:, 0] + events.Electron[:, 1]).mass

    events = set_ak_column(events, "m_min_lb", m_min_lb)

    return events


@producer(
    uses={
        "VetoBjet.pt",
    },
    produces={
        "trailing_pt",
    },
)
def jet_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "trailing_pt", ak.sum(events.VetoBjet.pt, axis=1))

    return events


@producer(
    uses={
        mc_weight,
        category_ids,
        "Jet.pt",
    },
    produces={
        mc_weight,
        category_ids,
        "cutflow.n_jet",
        "cutflow.ht",
        "cutflow.jet1_pt",
    },
)
def cutflow_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[mc_weight](events, **kwargs)
    events = self[category_ids](events, **kwargs)

    events = set_ak_column(events, "cutflow.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(
        events, "cutflow.jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT),
    )

    return events
