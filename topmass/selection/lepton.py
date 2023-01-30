# coding: utf-8

"""
Lepton selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")

coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


def invariant_mass(events: ak.Array):

    empty_events = ak.zeros_like(events, dtype=np.uint16)[:, 0:0]
    where = ak.num(events, axis=1) == 2
    events_2 = ak.where(where, events, empty_events)
    mass = ak.fill_none(ak.firsts((1 * events_2[:, :1] + 1 * events_2[:, 1:2]).mass), 0)
    return mass


@selector(
    uses={
        "Electron.pt",
        "Electron.eta",
        "Electron.phi",
        "Electron.mass",
        "Electron.charge",
        "Electron.mvaFall17V2Iso_WP80",
    }
)
def electron_selection(self: Selector, events: ak.Array, **kwargs):

    mask = (
        (events.Electron.pt > 20) &
        (abs(events.Electron.eta) < 2.4) &
        ((abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.5660)) &
        (events.Electron.mvaFall17V2Iso_WP80 == 1)
    )

    # pt sorted indices to convert mask
    indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)
    indices = indices[mask]
    # build and return selection results plus new columns (src -> dst -> indices)
    return indices


@selector(
    uses={
        "Muon.pt",
        "Muon.eta",
        "Muon.phi",
        "Muon.mass",
        "Muon.charge",
        "Muon.tightId",
        "Muon.pfRelIso04_all",
    }
)
def muon_selection(self: Selector, events: ak.Array, **kwargs):

    mask = (
        (events.Muon.pt > 20) &
        (abs(events.Muon.eta) < 2.4) &
        (events.Muon.tightId == 1) &
        (events.Muon.pfRelIso04_all < 0.15)
    )

    # pt sorted indices to convert mask
    indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)
    indices = indices[mask]
    # build and return selection results plus new columns (src -> dst -> indices)
    return indices


@selector(
    uses={
        muon_selection,
        electron_selection,
        "Electron.pt",
        "Electron.eta",
        "Electron.phi",
        "Electron.mass",
        "Electron.charge",
        "Muon.pt",
        "Muon.eta",
        "Muon.phi",
        "Muon.mass",
        "Muon.charge",
    },
    produces={"m_ll","channel_id"},
)
def l_l_selection(
    self: Selector, events: ak.Array, **kwargs
) -> tuple[ak.Array, SelectionResult]:

    # get channels from the config
    ch_ee = self.config_inst.get_channel("ee")
    ch_mumu = self.config_inst.get_channel("mumu")
    ch_emu = self.config_inst.get_channel("emu")

    # get the preselected muons and electrons
    muon_indices = self[muon_selection](events, **kwargs)
    electron_indices = self[electron_selection](events, **kwargs)

    # prepare output vectors
    empty_events = ak.zeros_like(1 * events.event, dtype=np.uint16)
    empty_indicies = empty_events[..., None][..., :0]
    channel_id = empty_events
    m_ll = empty_events
    sel_muon_indices = empty_indicies
    sel_electron_indices = empty_indicies

    # exact two electrons of oppsosite charge and no muons
    inv_ee_mass = invariant_mass(events.Electron[:, :2])

    where_ee = (
        (ak.num(electron_indices, axis=1) == 2) &
        (ak.num(muon_indices, axis=1) == 0) &
        (ak.sum(events.Electron.charge, axis=1) == 0) &
        (inv_ee_mass > 20) &
        ((inv_ee_mass > 106) | (inv_ee_mass < 76))
    )

    # IPython.embed()
    channel_id = ak.where(where_ee, ch_ee.id, channel_id)
    m_ll = ak.where(where_ee, inv_ee_mass, m_ll)
    sel_electron_indices = ak.where(where_ee, electron_indices, sel_electron_indices)

    # exact two muons of oppsosite charge and no electrons
    inv_mumu_mass = invariant_mass(events.Muon[:, :2])
    where_mumu = (
        (ak.num(muon_indices, axis=1) == 2) &
        (ak.num(electron_indices, axis=1) == 0) &
        (ak.sum(events.Muon.charge, axis=1) == 0) &
        (inv_mumu_mass > 20) &
        ((inv_mumu_mass > 106) | (inv_mumu_mass < 76))
    )

    channel_id = ak.where(where_mumu, ch_mumu.id, channel_id)
    m_ll = ak.where(where_mumu, inv_mumu_mass, m_ll)
    sel_muon_indices = ak.where(where_mumu, muon_indices, sel_muon_indices)

    # exact on electron and one muon of total charge 0
    leptons = ak.concatenate((events.Electron[:, :1], events.Muon[:, :1]), axis=1)
    inv_emu_mass = invariant_mass(leptons)
    where_emu = (
        (ak.num(electron_indices, axis=1) == 1) &
        (ak.num(muon_indices, axis=1) == 1) &
        (ak.sum(leptons.charge, axis=1) == 0) &
        (inv_emu_mass > 20)
    )

    channel_id = ak.where(where_emu, ch_emu.id, channel_id)
    m_ll = ak.where(where_emu, inv_emu_mass, m_ll)
    sel_electron_indices = ak.where(where_emu, electron_indices, sel_electron_indices)
    sel_muon_indices = ak.where(where_emu, muon_indices, sel_muon_indices)

    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "m_ll", m_ll)
    
    return events, SelectionResult(
        steps={"leptons": channel_id != 0},
        objects={
            "Muon": {"Muon": sel_muon_indices},
            "Electron": {"Electron": sel_electron_indices},
        },
    )
