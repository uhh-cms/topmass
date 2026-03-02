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
from columnflow.columnar_util import set_ak_column, sorted_indices_from_mask

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB", "Jet.jetId", "Jet.puId",
          "Jet.phi", "Jet.mass", attach_coffea_behavior, "HLT.*",
          "gen_top", "GenPart.*", "Jet.veto_map_mask",
          },
    produces={"MW1", "MW2", "Mt1", "Mt2", "chi2",
              "deltaRb", "combination_type", "R2b4q",
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
    Perform jet selection and event categorization for top mass analysis.

    This function applies a series of kinematic and trigger-based selections
    to jets in each event, identifies b-tagged and light jets, and reconstructs
    top quark and W boson candidates using combinatorics. It also handles
    background estimation by pseudo-reconstruction. The function returns
    the modified events array with new columns and a SelectionResult object
    containing selection masks and object indices.

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
            ak4_mask = (
                ak4_mask &
                ((events.Jet.pt >= 50.0) | (events.Jet.puId == 7))
            )
    elif mode == "trigger":
        ak4_mask = ak.ones_like(events.Jet.pt, dtype=bool)
    else:
        raise ValueError(f"Unknown jet_selection mode: {mode}")

    # Define a placeholder for empty float values
    EF = -99999.0

    # Step 1: Basic event and jet selection
    ht1_sel = (ak.sum(events.Jet.pt[ak4_mask], axis=1) >= 1)              # At least one jet with pt
    jet_mask0 = ak4_mask & (abs(events.Jet.eta) < 2.6)                    # Jets within eta acceptance
    jet_mask = (jet_mask0 & (events.Jet.pt >= 32.0))                      # Jets passing pt and eta
    ht_sel = (ak.sum(events.Jet.pt[jet_mask], axis=1) >= 450)             # HT > 450 GeV

    # Step 2: Tight selection for leading jets, pT > 40 GeV and |eta| < 2.4, require at least 6 jets
    jet_mask2 = ak4_mask & ((abs(events.Jet.eta) < 2.4) & (events.Jet.pt >= 40.0))
    jet_sel = ak.sum(jet_mask2, axis=1) >= 6
    # jet_sel = ((ak.num(events.Jet) >= 6) & ak.all(jet_mask2[:, :6],axis=1))

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

    # Step 8: Prepare for combinatorial reconstruction
    # Reference values and resolutions for W and top mass
    mwref = 80.36
    mwsig = 12  # Jette: 11.01
    mtsig = 15  # Jette: 27.07
    mu_tt = 0  # Jette: 2.07
    mu_w = 0  # Jette: 0.88

    # Helper functions for mass and deltaR calculations
    m = lambda j1, j2: (j1.add(j2)).mass
    m3 = lambda j1, j2, j3: (j1.add(j2.add(j3))).mass
    dr = lambda j1, j2: j1.delta_r(j2)

    # Step 9: Build jet combinations for signal and background
    bjet_after_jet_mask = (events.Jet[jet_mask2].btagDeepFlavB >= wp_tight)
    ljet_after_jet_mask = (events.Jet[jet_mask2].btagDeepFlavB < wp_tight)
    leading_six_sel = ((ak.num((events.Jet.pt[jet_mask2][:, :6])[bjet_after_jet_mask[:, :6]], axis=1) == 2) &
                       (ak.num((events.Jet.pt[jet_mask2][:, :6])[ljet_after_jet_mask[:, :6]], axis=1) == 4))

    leading_six_or_bkg = ak.any([leading_six_sel, alt_jet_trigger_sel])
    # ljets = ak.combinations((events.Jet[light_jet])[sixjets_sel], 4, axis=1)
    # bjets = ak.combinations((events.Jet[bjet_mask])[sixjets_sel], 2, axis=1)
    ljets = ak.combinations((events.Jet[light_jet][:, :4])[sixjets_sel], 4, axis=1)
    bjets = ak.combinations((events.Jet[bjet_mask][:, :2])[sixjets_sel], 2, axis=1)

    # Pseudo recontruction for background
    rej_jets = ((events.Jet[~loose_bjet_mask])[bjet_rej & alt_jet_trigger_sel & jet_sel & ht_sel])
    rej_jets = rej_jets[ak.argsort((rej_jets.pt), axis=1, ascending=False)][:, :6]

    rng = np.random.default_rng()
    rng_index = rng.permuted(np.full((len(rej_jets), 6), [0, 1, 2, 3, 4, 5]), axis=1)

    rej_jets = rej_jets[ak.Array(rng_index).to_list()]
    # rej_bjets = ak.combinations((rej_jets[ak.argsort(
    #     (rej_jets.btagDeepFlavB),
    #     axis=1,
    #     ascending=False)])[:, :2], 2, axis=1)
    # rej_ljets = ak.combinations((rej_jets[ak.argsort(
    #     (rej_jets.btagDeepFlavB),
    #     axis=1,
    #     ascending=False)])[:, 2:], 4, axis=1)
    rej_bjets = ak.combinations(rej_jets[:, :2], 2, axis=1)
    rej_ljets = ak.combinations(rej_jets[:, 2:], 4, axis=1)

    # Step 10: Helper functions for permutations and combinations
    def lpermutations(ljets):
        """
        Generate permutations of four light jets for W reconstruction.
        """
        j1, j2, j3, j4 = ljets
        return ak.concatenate([ak.zip([j1, j2, j3, j4]), ak.zip([j1, j3, j2, j4]), ak.zip([j1, j4, j2, j3])], axis=1)

    def bpermutations(bjets):
        """
        Generate permutations of two b-jets for top reconstruction.
        """
        j1, j2 = bjets
        return ak.concatenate([ak.zip([j1, j2]), ak.zip([j2, j1])], axis=1)

    def sixjetcombinations(bjets, ljets):
        """
        Combine b-jet and light-jet permutations into six-jet combinations.
        """
        return ak.cartesian([bjets, ljets], axis=1)

    # Step 11: Top quark mass reconstruction
    def mt(sixjets):
        """
        Calculate reconstructed masses and chi2 for all six-jet combinations.
        Returns best combination per event.
        """
        b1, b2 = ak.unzip(ak.unzip(sixjets)[0])
        j1, j2, j3, j4 = ak.unzip(ak.unzip(sixjets)[1])
        mt1 = ak.where((b1.pt > b2.pt), m3(b1, j1, j2), m3(b2, j3, j4))
        mt2 = ak.where((b1.pt > b2.pt), m3(b2, j3, j4), m3(b1, j1, j2))
        mw1 = ak.where((b1.pt > b2.pt), m(j1, j2), m(j3, j4))
        mw2 = ak.where((b1.pt > b2.pt), m(j3, j4), m(j1, j2))
        drbb = dr(b1, b2)
        chi2 = ak.sum([
            ((mw1 - mwref - mu_w) ** 2) / (mwsig ** 2),
            ((mw2 - mwref - mu_w) ** 2) / (mwsig ** 2),
            ((mt1 - mt2 - mu_tt) ** 2) / (mtsig ** 2)],
            axis=0,
        )
        if len(chi2) > 0:
            bestc2 = ak.argmin(chi2, axis=1, keepdims=True)
            return mt1[bestc2], mt2[bestc2], mw1[bestc2], mw2[bestc2], drbb[bestc2], chi2[bestc2], bestc2, sixjets
        else:
            return [[EF]], [[EF]], [[EF]], [[EF]], [[EF]], [[EF]], [[EF]], [[EF]]

    mt_result = mt(sixjetcombinations(bpermutations(ak.unzip(bjets)), lpermutations(ak.unzip(ljets))))
    mt_bkg_result = mt(sixjetcombinations(bpermutations(ak.unzip(rej_bjets)), lpermutations(ak.unzip(rej_ljets))))
    chi2_cut = 50
    mt_result_filled = np.full((6, ak.num(events, axis=0)), EF)
    for i in range(6):
        (mt_result_filled[i])[sixjets_sel] = ak.flatten(mt_result[i])
        (mt_result_filled[i])[bjet_rej & alt_jet_trigger_sel & jet_sel & ht_sel] = ak.flatten(mt_bkg_result[i])

    chi2_sel = ak.Array((mt_result_filled[5] < chi2_cut) & (mt_result_filled[5] > -1))
    chi2_sel1 = ak.Array((mt_result_filled[5] < 25) & (mt_result_filled[5] > -1))
    chi2_sel2 = ak.Array((mt_result_filled[5] < 10) & (mt_result_filled[5] > -1))
    chi2_sel3 = ak.Array((mt_result_filled[5] < 5) & (mt_result_filled[5] > -1))
    Rbb_sel = ak.Array(mt_result_filled[4] > 2)

    events = set_ak_column(events, "Mt1", mt_result_filled[0])
    events = set_ak_column(events, "Mt2", mt_result_filled[1])
    events = set_ak_column(events, "MW1", mt_result_filled[2])
    events = set_ak_column(events, "MW2", mt_result_filled[3])
    events = set_ak_column(events, "deltaRb", mt_result_filled[4])
    events = set_ak_column(events, "chi2", mt_result_filled[5])

    # Step 12: Combination type matching (for MC truth)
    def combinationtype(bestcomb, correctcomb):
        """
        Determine if the selected jet combination matches the MC truth.
        Returns a type code (1: matched, 0: not matched, -1: not applicable).
        """
        b1, b2 = ak.unzip(ak.unzip(bestcomb)[0])
        j1, j2, j3, j4 = ak.unzip(ak.unzip(bestcomb)[1])

        b1cor = correctcomb.b[:, 0]
        q1cor = correctcomb.w_children[:, 0, 0]
        q2cor = correctcomb.w_children[:, 0, 1]
        b2cor = correctcomb.b[:, 1]
        q3cor = correctcomb.w_children[:, 1, 0]
        q4cor = correctcomb.w_children[:, 1, 1]
        drmax = 0.4
        drb11, drb22, drq11 = (dr(b1, b1cor) < drmax), (dr(b2, b2cor) < drmax), (dr(j1, q1cor) < drmax)
        drq22, drq33, drq44 = (dr(j2, q2cor) < drmax), (dr(j3, q3cor) < drmax), (dr(j4, q4cor) < drmax)
        drq21, drq12 = (dr(j2, q1cor) < drmax), (dr(j1, q2cor) < drmax)
        drq43, drq34 = (dr(j4, q3cor) < drmax), (dr(j3, q4cor) < drmax)
        drb21, drb12, drq31 = (dr(b2, b1cor) < drmax), (dr(b1, b2cor) < drmax), (dr(j3, q1cor) < drmax)
        drq42, drq13, drq24 = (dr(j4, q2cor) < drmax), (dr(j1, q3cor) < drmax), (dr(j2, q4cor) < drmax)
        drq41, drq32 = (dr(j4, q1cor) < drmax), (dr(j3, q2cor) < drmax)
        drq23, drq14 = (dr(j2, q3cor) < drmax), (dr(j1, q4cor) < drmax)
        # b1b2: 1234 2134 1243 2143, b2b1: 3412 4312 3421 4321
        drlist = [(drb11 & drb22 & drq11 & drq22 & drq33 & drq44),
                  (drb11 & drb22 & drq21 & drq12 & drq33 & drq44),
                  (drb11 & drb22 & drq11 & drq22 & drq43 & drq34),
                  (drb11 & drb22 & drq21 & drq12 & drq43 & drq34),
                  (drb21 & drb12 & drq31 & drq42 & drq13 & drq24),
                  (drb21 & drb12 & drq41 & drq32 & drq13 & drq24),
                  (drb21 & drb12 & drq31 & drq42 & drq23 & drq14),
                  (drb21 & drb12 & drq41 & drq32 & drq23 & drq14),
                  ]
        # test if all jets are matched
        matched = ak.all([
            ak.any([drb11, drb12], axis=0),
            ak.any([drb22, drb21], axis=0),
            ak.any([drq11, drq12, drq13, drq14], axis=0),
            ak.any([drq21, drq22, drq23, drq24], axis=0),
            ak.any([drq31, drq32, drq33, drq34], axis=0),
            ak.any([drq41, drq42, drq43, drq44], axis=0)], axis=0)
        type = ak.flatten(matched) * 1 + ak.flatten(ak.any(drlist, axis=0))
        return type

    if self.dataset_inst.has_tag("has_top"):
        type = np.full((1, ak.num(events, axis=0)), -1)
        type_unfilled = combinationtype((mt_result[7])[mt_result[6]], events.gen_top[sixjets_sel])
        type[0][sixjets_sel] = type_unfilled
        type = ak.flatten(type)
    else:
        type = -1
    events = set_ak_column(events, "combination_type", type)

    if (len(ak.unzip(mt_result[6][:])) > 1):
        if ((len(ak.unzip(ak.unzip(mt_result[6][:])[0])) > 1) & (len(ak.unzip(ak.unzip(mt_result[6][:])[1])) > 3)):
            R2b4q = (
                ak.unzip(ak.unzip(mt_result[6][:])[0])[0].pt + ak.unzip(ak.unzip(mt_result[6][:])[0])[1].pt
            ) / (
                ak.unzip(ak.unzip(mt_result[6][:])[1])[0].pt +
                ak.unzip(ak.unzip(mt_result[6][:])[1])[1].pt +
                ak.unzip(ak.unzip(mt_result[6][:])[1])[2].pt +
                ak.unzip(ak.unzip(mt_result[6][:])[1])[3].pt
            )
            R2b4q_filled = np.full((1, ak.num(events, axis=0)), EF)
            R2b4q_filled[0][sixjets_sel] = ak.flatten(R2b4q)

            events = set_ak_column(events, "R2b4q", R2b4q_filled[0])
        else:
            events = set_ak_column(events, "R2b4q", EF)
    else:
        events = set_ak_column(events, "R2b4q", EF)

    # TODO: test mt cuts
    mt_sel = (events.Mt1 < 150)
    mt_sel2 = (events.Mt1 < 175)
    mt_sel3 = (events.Mt1 < 125)

    # Step 13: Build and return selection results
    # The SelectionResult object contains masks for each selection step,
    # indices for jet collections, and auxiliary variables.
    return events, SelectionResult(
        steps={
            "All": ht1_sel,
            "Mt": mt_sel,
            "Mt1": mt_sel3,
            "Mt2": mt_sel2,
            "BaseTrigger": jet_base_trigger_sel,
            "SignalOrBkgTrigger": signal_or_bkg_trigger,
            "LeadingSix": leading_six_or_bkg,
            "BkgTrigger": alt_jet_trigger_sel,
            "Trigger": jet_trigger_sel,
            "HT": ht_sel,
            "jet": jet_sel,
            "BTag": bjet_sel,
            "BTag20": sel_bjet_2or0,
            "Rbb": Rbb_sel,
            "SixJets": sixjets_sel,
            "Chi2": chi2_sel,
            "n25Chi2": chi2_sel1,
            "n10Chi2": chi2_sel2,
            "n5Chi2": chi2_sel3,
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
