# coding: utf-8

import pyKinFit as pyKinFit
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column, flat_np_view
from columnflow.production import Producer, producer

# from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB"},
    produces={
        "FitJet.pt",
        "FitJet.eta",
        "FitJet.phi",
        "FitJet.mass",
        "FitChi2",
    },
    jet_pt=None,
    jet_trigger=None,
    sandbox="bash::$CF_REPO_BASE/sandboxes/cmsswtest.sh",
)
def kinFit(
    self: Producer, events: ak.Array, sel_jet_mask: ak.Array, eventmask: ak.Array,
    **kwargs
) -> ak.Array:
    import IPython

    sel_events = events[eventmask]
    sorted_indices = ak.argsort(sel_events.Jet.btagDeepFlavB, ascending=False)
    sorted_jets = sel_events.Jet[sorted_indices]
    fitPt, fitEta, fitPhi, fitMass, indexlist, fitChi2 = pyKinFit.setBestCombi(
        ak.to_list(sorted_jets.pt[sel_jet_mask]),
        ak.to_list(sorted_jets.eta[sel_jet_mask]),
        ak.to_list(sorted_jets.phi[sel_jet_mask]),
        ak.to_list(sorted_jets.mass[sel_jet_mask]),
    )

    # function to insert append indices not found in a list yet to a target length
    def appendindices(initial_array, target_lengths):
        for i in range(len(initial_array)):
            inner_list = initial_array[i]
            target_length = target_lengths[i]
            available_numbers = list(range(target_length))
            for num in available_numbers:
                if num not in inner_list:
                    inner_list.append(num)
                if len(inner_list) >= target_length:
                    break
        return initial_array

    # function to insert values of one awkward array into another at a list of given indices
    def insert_at_index(to_insert, where, indices_to_replace):
        full_true = ak.full_like(where, True, dtype=bool)
        mask = full_true & indices_to_replace
        flat = flat_np_view(where[mask])
        flat = flat_np_view(to_insert)
        cut_orig = ak.num(where[mask])
        cut_replaced = ak.unflatten(flat, cut_orig)
        original = where[~mask]
        combined = ak.concatenate((original, cut_replaced), axis=1)
        return combined

    lok_ind = ak.local_index(events.Jet)
    indexmask = appendindices(indexlist, ak.num(lok_ind[eventmask], axis=1))
    combined_indices = insert_at_index(indexmask, lok_ind, eventmask)

    sorted_jet = events.Jet[combined_indices]

    # Take only the first 6 jets per event
    sorted_jets_top6 = sorted_jet[:, :6]
    # Convert your Python lists to awkward arrays
    fitPt_ak = ak.Array(fitPt)
    fitPt_full = insert_at_index(
        fitPt_ak[:, :6], sorted_jets_top6.pt, eventmask)
    fitEta_ak = ak.Array(fitEta)
    fitEta_full = insert_at_index(
        fitEta_ak[:, :6], sorted_jets_top6.eta, eventmask)
    fitPhi_ak = ak.Array(fitPhi)
    fitPhi_full = insert_at_index(
        fitPhi_ak[:, :6], sorted_jets_top6.phi, eventmask)
    fitMass_ak = ak.Array(fitMass)
    fitMass_full = insert_at_index(
        fitMass_ak[:, :6], sorted_jets_top6.mass, eventmask)
    # Create FitJet collection for the selected events with fit values
    fitJet_record = ak.Array({"reco": sorted_jets_top6})
    fitJet_record = ak.with_field(fitJet_record, fitPt_full[:, :6], "pt")
    fitJet_record = ak.with_field(fitJet_record, fitEta_full[:, :6], "eta")
    fitJet_record = ak.with_field(fitJet_record, fitPhi_full[:, :6], "phi")
    fitJet_record = ak.with_field(fitJet_record, fitMass_full[:, :6], "mass")
    events = set_ak_column(events, "FitJet", fitJet_record)
    total_chi2 = np.full(len(events), EMPTY_FLOAT)
    total_chi2[eventmask] = ak.Array(fitChi2)
    events = set_ak_column(events, "FitChi2", total_chi2)
    return events
