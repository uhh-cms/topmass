# coding: utf-8

"""
KinFit producer utilities.

This module provides the `kinFit` producer that wraps an external
kinematic fitter (via `pyKinFit`) to compute best jet combinations
and fitted four-vectors. The producer exposes `FitJet` collection
fields and basic fit quality columns such as `FitChi2` and `FitPgof`.

How this producer is used:
- Run the fit on and returns the original `events` array with added `FitJet`, `FitChi2`,
  and `FitPgof` columns.
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer
from alljets.production.kinfit_utils import appendindices, insert_at_index

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        "KinFitJets.pt",
        "KinFitJets.eta",
        "KinFitJets.phi",
        "KinFitJets.mass",
        "KinFitJets.btagDeepFlavB",
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
    **kwargs,
) -> ak.Array:
    """
    Run the external kinematic fitter and produce `FitJet` + quality columns.

    Parameters
    ----------
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

    1. If the event has 2 b-tagged jets (using the tight deepjet working point),
       they are chosen as b-jet candidates.

    2. If fewer than 2 b-tagged jets are available, the six leading jets are randomly permuted
       and the first two are used as b-jet candidates.

    The `FitJet` collection stores the fitted four-vector components in the following order:
    (B1, B2, W1Prod1, W1Prod2, W2Prod1, W2Prod2).
    """

    import pyKinFit

    # Generate per-event random seeds based on event identifiers for reproducibility
    # Use these seeds to generate random indices for each event to shuffle the six leading jets

    seeds = (events.run + events.luminosityBlock + events.event) % (2**32)
    random_indices = ak.Array([list(np.random.default_rng(int(seed)).permutation(min(len(j), 6)))
                               for seed, j in zip(seeds, events.KinFitJets)])

    # Sorting logic needed for the kinematic fit.
    # The fitter expects the first two jets to be the b-jet candidates.

    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    sorted_indices = ak.where(
        ak.sum(events.KinFitJets.btagDeepFlavB >= wp_tight, axis=1) == 2,
        ak.argsort(ak.where(events.KinFitJets.btagDeepFlavB >= wp_tight, events.KinFitJets.pt, -999), ascending=False),
        random_indices,
    )

    # Apply ordering and call fitter (returns lists per-event)
    sorted_jets = events.KinFitJets[sorted_indices]

    fitPt, fitEta, fitPhi, fitMass, indexlist, fitChi2, fitPgof = pyKinFit.setBestCombi(
        ak.to_list(sorted_jets.pt),
        ak.to_list(sorted_jets.eta),
        ak.to_list(sorted_jets.phi),
        ak.to_list(sorted_jets.mass),
    )

    # Build index mappings from the original jet layout to the fitted order
    lok_ind = ak.local_index(events.KinFitJets)
    indexmask = appendindices(indexlist, ak.num(lok_ind, axis=1))
    combined_indices = insert_at_index(indexmask, lok_ind)

    # sorted_reco_indices = sorted_indices
    sorted_reco = events.KinFitJets[sorted_indices]
    sorted_jet = sorted_reco[combined_indices]

    fitPt_full = ak.Array(fitPt)
    fitEta_full = ak.Array(fitEta)
    fitPhi_full = ak.Array(fitPhi)
    fitMass_full = ak.Array(fitMass)

    # Create FitJet collection for the selected events with fit values
    fitJet_record = ak.Array({"reco": sorted_jet})
    fitJet_record = ak.with_field(fitJet_record, fitPt_full, "pt")
    fitJet_record = ak.with_field(fitJet_record, fitEta_full, "eta")
    fitJet_record = ak.with_field(fitJet_record, fitPhi_full, "phi")
    fitJet_record = ak.with_field(fitJet_record, fitMass_full, "mass")
    events = set_ak_column(events, "FitJet", fitJet_record)

    # FitJets are in Order (B1,B2,W1Prod1,W1Prod2,W2Prod1,W2Prod2)
    total_chi2 = ak.Array(fitChi2)
    total_pgof = ak.Array(fitPgof)
    events = set_ak_column(events, "FitChi2", total_chi2)
    events = set_ak_column(events, "FitPgof", total_pgof)
    return events
