# coding: utf-8

"""
KinFit producer utilities.

This module provides the `kinFit` producer that wraps an external
kinematic fitter (via `pyKinFit`) to compute best jet combinations
and fitted four-vectors. The producer exposes `FitJet` collection
fields and basic fit quality columns such as `FitChi2` and `FitPgof`.

How this producer is used:
- It accepts masks selecting jets and events to run the fit on and
    returns the original `events` array with added `FitJet`, `FitChi2`,
    and `FitPgof` columns.
"""

from columnflow.columnar_util import EMPTY_FLOAT, flat_np_view, set_ak_column
from columnflow.production import Producer, producer
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        "EventJet.pt",
        "EventJet.eta",
        "EventJet.phi",
        "EventJet.mass",
        "EventJet.btagDeepFlavB",
        "EventJet.jetId",
        "EventJet.puId",
        "event",
        "run",
        "luminosityBlock",
    },
    produces={
        "FitJet.pt",
        "FitJet.eta",
        "FitJet.phi",
        "FitJet.mass",
        "FitJet.reco.*",
        "FitChi2",
        "FitPgof",
    },
    jet_pt=None,
    jet_trigger=None,
    sandbox="bash::$CF_REPO_BASE/sandboxes/cmsswtest.sh",
)
def kinFit(
    self: Producer,
    events: ak.Array,
    sel_jet_mask: ak.Array,
    eventmask: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Run the external kinematic fitter and produce `FitJet` + quality columns.

    Parameters
    ----------
    sel_jet_mask : ak.Array
        Boolean mask selecting jets to be considered by the fitter.
    eventmask : ak.Array
        Boolean mask selecting which events should run the fit.

    Returns
    -------
    ak.Array
        Original events array with added columns:
        - `FitJet` : fitted jet collection (with `reco` subfield pointing to original jets)
        - `FitChi2` : chi2 of the fit (EMPTY_FLOAT for events not fitted)
        - `FitPgof` : p-value of the fit (EMPTY_FLOAT for events not fitted)

    Notes
    -----
    The fitter (`pyKinFit.setBestCombi`) requires a choice of b-jet candidates:

    1. If the event has at least 2 b-tagged jets (using the tight deepjet working point),
    the two highest-pt b-tagged jets are chosen as b-jet candidates.

    2. If fewer than 2 b-tagged jets are available, the six leading jets are randomly permuted
    and the first two are used as b-jet candidates.

    The `FitJet` collection stores the fitted four-vector components in the following order:
    (B1, B2, W1Prod1, W1Prod2, W2Prod1, W2Prod2).
    """

    import pyKinFit

    # Slice to events that will be fitted
    sel_events = events[eventmask]
    sel_Jets = sel_events.EventJet[sel_jet_mask[eventmask]]

    # Generate per-event random seeds based on event identifiers for reproducibility
    # Use these seeds to generate random indices for each event to shuffle the six leading jets
    seeds = (events.run + events.luminosityBlock + events.event) % (2**32)
    random_indices = ak.Array([list(np.random.default_rng(int(seed)).permutation(min(len(j), 6))) +
                               list(range(6, len(j))) for seed, j in zip(seeds, events.EventJet)])

    # Sorting logic for the b-jet candidates for the kinFit input:
    # The first 2 jets are used as b-jet candidates,
    # If an event has at least 2 b-tagged jets, sort b-tagged jets by pt and use highest two as b-jet candidates.
    # If there are < 2 b-tagged jets, randomly sort six leading jets for b-jet candidates.

    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    sorted_indices = ak.where(
        ak.sum(sel_Jets.btagDeepFlavB >= wp_tight, axis=1) >= 2,
        ak.argsort(ak.where(sel_Jets.btagDeepFlavB >= wp_tight, sel_Jets.pt, -999), ascending=False),
        random_indices,
    )

    # Apply ordering and call fitter (returns lists per-event)
    sorted_jets = sel_Jets[sorted_indices]

    fitPt, fitEta, fitPhi, fitMass, indexlist, fitChi2, fitPgof = pyKinFit.setBestCombi(
        ak.to_list(sorted_jets.pt),
        ak.to_list(sorted_jets.eta),
        ak.to_list(sorted_jets.phi),
        ak.to_list(sorted_jets.mass),
    )

    # Helper: ensure index lists are padded to the expected length
    def appendindices(initial_array, target_lengths):
        """Pad inner index-lists so each reaches its event-specific length."""
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

    # Helper: insert per-event values (flattened) back into awkward layout
    def insert_at_index(to_insert, where, indices_to_replace):
        full_true = ak.full_like(where, True, dtype=bool)
        mask = full_true & indices_to_replace
        flat = flat_np_view(to_insert)
        cut_orig = ak.num(where[mask])
        cut_replaced = ak.unflatten(flat, cut_orig)
        original = where[~mask]
        combined = ak.concatenate((original, cut_replaced), axis=1)
        return combined

    # Build index mappings from the original jet layout to the fitted order
    lok_ind = ak.local_index(events.EventJet[sel_jet_mask])
    indexmask = appendindices(indexlist, ak.num(lok_ind[eventmask], axis=1))
    combined_indices = insert_at_index(indexmask, lok_ind, eventmask)

    # Repeat the ordering logic on the full events to map reco jets
    sorted_reco_indices = ak.where(
        ak.sum(events.EventJet.btagDeepFlavB >= wp_tight, axis=1) >= 2,
        ak.argsort(ak.where(events.EventJet.btagDeepFlavB >= wp_tight, events.EventJet.pt, -999), ascending=False),
        random_indices,
    )
    sorted_reco = (events.EventJet[sel_jet_mask])[sorted_reco_indices]
    sorted_jet = sorted_reco[combined_indices]

    # Only keep the first 6 jets per event (fitter returns up to 6)
    sorted_jets_top6 = sorted_jet[:, :6]

    # Convert fitter outputs (Python lists) to awkward arrays and merge
    fitPt_ak = ak.Array(fitPt)
    fitPt_full = insert_at_index(fitPt_ak[:, :6], sorted_jets_top6.pt, eventmask)
    fitEta_ak = ak.Array(fitEta)
    fitEta_full = insert_at_index(fitEta_ak[:, :6], sorted_jets_top6.eta, eventmask)
    fitPhi_ak = ak.Array(fitPhi)
    fitPhi_full = insert_at_index(fitPhi_ak[:, :6], sorted_jets_top6.phi, eventmask)
    fitMass_ak = ak.Array(fitMass)
    fitMass_full = insert_at_index(fitMass_ak[:, :6], sorted_jets_top6.mass, eventmask)

    # Create FitJet collection for the selected events with fit values
    fitJet_record = ak.Array({"reco": sorted_jets_top6})
    fitJet_record = ak.with_field(fitJet_record, fitPt_full[:, :6], "pt")
    fitJet_record = ak.with_field(fitJet_record, fitEta_full[:, :6], "eta")
    fitJet_record = ak.with_field(fitJet_record, fitPhi_full[:, :6], "phi")
    fitJet_record = ak.with_field(fitJet_record, fitMass_full[:, :6], "mass")
    events = set_ak_column(events, "FitJet", fitJet_record)

    # FitJets are in Order (B1,B2,W1Prod1,W1Prod2,W2Prod1,W2Prod2)
    total_chi2 = np.full(len(events), EMPTY_FLOAT)
    total_chi2[eventmask] = ak.Array(fitChi2)
    total_pgof = np.full(len(events), EMPTY_FLOAT)
    total_pgof[eventmask] = ak.Array(fitPgof)
    events = set_ak_column(events, "FitChi2", total_chi2)
    events = set_ak_column(events, "FitPgof", total_pgof)
    return events
