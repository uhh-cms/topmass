# coding: utf-8
"""
Jet selection and reconstruction methods for top quark mass analysis.

This module defines the main jet selection logic, including trigger selection,
jet kinematic cuts, b-tagging, combinatorics for top/W reconstruction, and
event categorization for signal and background. It is designed for use with
columnar data (awkward arrays) in the context of the columnflow framework.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import sorted_indices_from_mask

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB", "Jet.jetId", "Jet.puId",
          "Jet.phi", "Jet.mass", attach_coffea_behavior, "HLT.*",
          "gen_top", "GenPart.*", "Jet.veto_map_mask",
          },
    jet_pt=None, jet_trigger=None, jet_base_trigger=None, alt_jet_trigger=None,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    mode: str = "analysis",
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Perform jet selection for top mass analysis.

    This function applies a series of kinematic and trigger-based selections
    to jets in each event, identifies b-tagged and light jets.

    The function returns SelectionResult object containing selection masks and object indices.

    The objects returned in the SelectionResult include the following jet collections:

    - Jet: All Jets within the |eta| < 2.6 acceptance, interesting for trigger studies
    - EventJet: Jets passing the main selection (pT >= 40 GeV, |eta| < 2.4),
    - Bjet: Subset of EventJet that are b-tagged (passing the tight working point)
    - VetoJet: Jets that fail the main selection (not passing pT or eta cuts)
    - LightJet: Subset of EventJet that are not b-tagged (failing the tight working point)

    Args:
        self (Selector): The selector instance.
        events (ak.Array): The awkward array of event data.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[ak.Array, SelectionResult]: The updated events and selection results.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID?rev=107#nanoAOD_Flags
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL?rev=15#Recommendations_for_the_13_T_AN1
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL?rev=17
    https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD?rev=100#Jets
    """

    # Ensure that the Jets we use are passing the tight + tightLepVeto jet Id
    # and are not vetoed by the jet veto map
    if mode == "analysis":
        ak4_mask = (events.Jet.jetId >= 6) & (events.Jet.veto_map_mask)

        # Require tight pileup Id for jets with pt < 50 GeV
        if self.config_inst.campaign.x.run == 2:
            ak4_mask = (ak4_mask & ((events.Jet.pt >= 50.0) | (events.Jet.puId == 7)))

    elif mode == "trigger":
        ak4_mask = ak.ones_like(events.Jet.pt, dtype=bool)
    else:
        raise ValueError(f"Unknown jet_selection mode: {mode}")

    # Step 1: Basic event and jet selection
    ht1_sel = (ak.sum(events.Jet.pt[ak4_mask], axis=1) >= 1)              # At least one jet with pt
    jet_mask0 = ak4_mask & (abs(events.Jet.eta) < 2.6)                    # Jets within eta acceptance
    jet_mask = (jet_mask0 & (events.Jet.pt >= 32.0))                      # Jets passing pt and eta
    ht_sel = (ak.sum(events.Jet.pt[jet_mask], axis=1) >= 450)             # HT > 450 GeV

    # Step 2: Tight selection for jets, pT > 40 GeV and |eta| < 2.4, require at least 6 jets
    jet_mask2 = ak4_mask & ((abs(events.Jet.eta) < 2.4) & (events.Jet.pt >= 40.0))
    jet_sel = ak.sum(jet_mask2, axis=1) >= 6

    # Step 3: Identify veto jets (not passing main selection)
    veto_jet = ~jet_mask

    # Step 4: Identify b-tagged and light jets
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    light_jet = (jet_mask2) & (events.Jet.btagDeepFlavB < wp_tight)
    bjet_mask = (jet_mask2) & (events.Jet.btagDeepFlavB >= wp_tight)

    # Step 5: Event selection based on b-jet and light jet multiplicity
    bjet_sel = ((ak.sum(bjet_mask, axis=1) >= 2))
    sixjets_sel = (bjet_sel & (ak.sum(light_jet, axis=1) >= 4))

    # Step 6: Background estimation (b-jet veto)
    wp_loose = self.config_inst.x.btag_working_points.deepjet.loose
    loose_bjet_mask = (events.Jet.btagDeepFlavB >= wp_loose)
    bjet_rej = (ak.sum(((jet_mask2) & loose_bjet_mask), axis=1) == 0)
    sel_bjet_2or0 = ak.any([bjet_sel, bjet_rej], axis=0)

    # Step 7: Trigger selection (skip for QCD MC)
    if not self.dataset_inst.name.startswith("qcd"):
        ones = ak.ones_like(jet_sel)
        jet_trigger_sel = ones if not self.jet_trigger else events.HLT[self.jet_trigger]
        alt_jet_trigger_sel = ones if not self.jet_trigger else events.HLT[self.alt_jet_trigger]
        jet_base_trigger_sel = ones if not self.jet_base_trigger else events.HLT[self.jet_base_trigger]
    else:
        jet_trigger_sel = [True] * len(events)
        alt_jet_trigger_sel = [True] * len(events)
        jet_base_trigger_sel = [True] * len(events)
    signal_or_bkg_trigger = ak.any([jet_trigger_sel, alt_jet_trigger_sel], axis=0)

    # Step 8: Build and return selection results
    # The SelectionResult object contains masks for each selection step,
    # indices for jet collections, and auxiliary variables.
    return events, SelectionResult(
        steps={
            "All": ht1_sel,
            "BaseTrigger": jet_base_trigger_sel,
            "SignalOrBkgTrigger": signal_or_bkg_trigger,
            "BkgTrigger": alt_jet_trigger_sel,
            "Trigger": jet_trigger_sel,
            "HT": ht_sel,
            "jet": jet_sel,
            "BTag": bjet_sel,
            "BTag20": sel_bjet_2or0,
            "SixJets": sixjets_sel,
        },
        objects={
            "Jet": {
                "Jet": sorted_indices_from_mask(jet_mask0, events.Jet.pt, ascending=False),
                "EventJet": sorted_indices_from_mask(jet_mask2, events.Jet.pt, ascending=False),
                "Bjet": sorted_indices_from_mask(bjet_mask, events.Jet.pt, ascending=False),
                "VetoJet": sorted_indices_from_mask(veto_jet, events.Jet.pt, ascending=False),
                "LightJet": sorted_indices_from_mask(light_jet, events.Jet.pt, ascending=False),
                "JetsByBTag": sorted_indices_from_mask(jet_mask, events.Jet.btagDeepFlavB, ascending=False),
            },
        },
        aux={
            "n_jets": ak.sum(jet_mask2, axis=1),
            "n_bjets": ak.sum(bjet_mask, axis=1),
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    """
    Initialization for jet selection: set up triggers and pt thresholds based on year.
    """
    year = self.config_inst.campaign.x.year
    # register shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer", "tune", "hdamp", "trig"))
    }
    # NOTE: the none will not be overwritten later when doing this...
    # self.jet_trigger = None

    # Jet pt thresholds (if not set manually) based on year (1 pt above trigger threshold)
    # When jet pt thresholds are set manually, don't use any trigger
    if not self.jet_pt:
        self.jet_pt = {2016: 31, 2017: 33, 2018: 33}[year]

        # Trigger choice based on year of data-taking (for now: only single trigger)
        self.jet_trigger = {
            2016: "PFHT400_SixJet30_DoubleBTagCSV_p056",
            # or "HLT_PFHT450_SixJet40_BTagCSV_p056")
            2017: "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2",
            # "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2" or "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2"
            # or "PFHT430_SixPFJet40_PFBTagCSV_1p5"
            # Base Trigger: "PFHT370"
            2018: "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
            # or "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59")
        }[year]
        self.uses.add(f"HLT.{self.jet_trigger}")

        # Trigger choice based on year of data-taking (for now: only single trigger)
        self.jet_base_trigger = {
            2016: "PFHT400_SixJet30_DoubleBTagCSV_p056",
            # or "HLT_PFHT450_SixJet40_BTagCSV_p056")
            2017: "PFHT350",
            # Base Trigger: "PFHT370", "PFHT350", "IsoMu24", "Physics"
            2018: "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
            # or "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59")
        }[year]
        self.uses.add(f"HLT.{self.jet_base_trigger}")

        self.alt_jet_trigger = {
            2016: "PFHT400_SixJet30_DoubleBTagCSV_p056",
            # or "HLT_PFHT450_SixJet40_BTagCSV_p056")
            2017: "PFHT380_SixPFJet32",
            # "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2" or "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2"
            # or "PFHT430_SixPFJet40_PFBTagCSV_1p5"
            # Base Trigger: "PFHT370"
            2018: "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
            # or "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59")
        }[year]
        self.uses.add(f"HLT.{self.alt_jet_trigger}")
