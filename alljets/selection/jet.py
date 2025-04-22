# coding: utf-8
"""
Jet selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import set_ak_column, flat_np_view, mask_from_indices
import pyKinFitTest as pyKinFit

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
          "Jet.btagDeepFlavB", "Jet.jetId", "Jet.puId"},
    produces={""},
    jet_pt=None, jet_trigger=None,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # example jet selection: at least six jets, lowest jet at least 40 GeV and H_T > 450 GeV
    ht_sel = (ak.sum(events.Jet.pt, axis=1) >= 450)
    jet_mask = ((events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4))
    jet_sel = ak.sum(jet_mask, axis=1) >= 6
    veto_jet = ((events.Jet.pt < 40.0) | (abs(events.Jet.eta) > 2.4))
    # pt sorted indices
    # indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    # jet_indices = indices[jet_mask]
    # b-tagged jets (tight wp)
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_tight)
    # bjet_indices = indices[bjet_mask][:, :2]
    bjet_sel = (ak.sum(bjet_mask, axis=1) >= 2) & (
        ak.sum(jet_mask[:, :2], axis=1) == ak.sum(bjet_mask[:, :2], axis=1))
    # Trigger selection step is skipped for QCD MC, which has no Trigger columns
    if not self.dataset_inst.name.startswith("qcd"):
        ones = ak.ones_like(jet_sel)
    # trigger
        jet_trigger_sel = ones if not self.jet_trigger else events.HLT[self.jet_trigger]
    else:
        jet_trigger_sel = True

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Jet": {"MyCustomJetCollection": indices_applied_to_Jet}}
    return events, SelectionResult(
        steps={
            "BTag": bjet_sel,
            "jet": jet_sel,
            "HT": ht_sel,
            "Trigger": jet_trigger_sel,
        },
        objects={
            "Jet": {
                "Jet": sorted_indices_from_mask(jet_mask, events.Jet.pt, ascending=False),
                "Bjet": sorted_indices_from_mask(bjet_mask, events.Jet.pt, ascending=False),
                "VetoJet": sorted_indices_from_mask(veto_jet, events.Jet.pt, ascending=False),
            },
        },
        aux={
            "n_jets": ak.sum(jet_mask, axis=1),
            "n_bjets": ak.sum(bjet_mask, axis=1),
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    year = self.config_inst.campaign.x.year

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
            # or "HLT_PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2", "HLT_PFHT430_SixPFJet40_PFBTagCSV_1p5")
            2018: "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
            # or "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59")
        }[year]
        self.uses.add(f"HLT.{self.jet_trigger}")


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB"},
    produces={
        "FitJet.pt", "FitJet.eta", "FitJet.phi", "FitJet.mass", "FitChi2"
    },
    jet_pt=None, jet_trigger=None, sandbox="bash::$CF_REPO_BASE/sandboxes/cmsswtest.sh"
)
def kinFit(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    EMPTY_FLOAT = -99999.0
    import IPython
    # PRELIMINARY SELECTIONS
    ht_sel = (ak.sum(events.Jet.pt, axis=1) >= 450)
    jet_mask = ((events.Jet.pt >= 40.0) & (abs(events.Jet.eta) < 2.4))
    jet_sel = ak.sum(jet_mask, axis=1) >= 6
    # b-tagged jets (tight wp)
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_tight)
    # bjet_indices = indices[bjet_mask][:, :2]
    bjet_sel = (ak.sum(bjet_mask, axis=1) >= 2) & (
        ak.sum(jet_mask[:, :2], axis=1) == ak.sum(bjet_mask[:, :2], axis=1))
    eventmask = ht_sel & jet_sel & bjet_sel
    sel_events = events[eventmask]
    sorted_indices = ak.argsort(sel_events.Jet.btagDeepFlavB, ascending=False)
    sorted_jets = sel_events.Jet[sorted_indices]
    sel_jet_mask = ((sorted_jets.pt >= 40.0) & (abs(sorted_jets.eta) < 2.4))
    fitPt, fitEta, fitPhi, fitMass, indexlist, fitChi2 = pyKinFit.setBestCombi(ak.to_list(sorted_jets.pt[sel_jet_mask]), ak.to_list(
        sorted_jets.eta[sel_jet_mask]), ak.to_list(sorted_jets.phi[sel_jet_mask]), ak.to_list(sorted_jets.mass[sel_jet_mask]))
    def mask_from_indices(indices: np.array | ak.Array, layout_array: ak.Array) -> ak.Array:
        """
        Takes an array of *indices* and a *layout_array* and returns a boolean mask with the same shape
        as *layout_array* where values at *indices* are set to *True*. Example:
    
        .. code-block:: python
    
            indices = [[2, 0], ...]
            layout_array = [[x, y, z], ...]
    
            mask_from_indices(indices, layout_array)
            # -> [[True, False, True], ...]
    
        :param indices: The indices to set to *True*.
        :param layout_array: The layout array to use as a template.
        :return: A boolean mask with the same shape as *layout_array*.
        """
        # create a flat mask starting with false
        flat_mask = flat_np_view(ak.full_like(layout_array, False, dtype=bool))
        # use offsets of the layout to create increasing indices
        flat_indices = flat_np_view((indices + layout_array.layout.offsets.data[:-1, ..., None]))
        # set the indices to true
        flat_mask[flat_indices] = True
        # layout the mask
        return layout_ak_array(flat_mask, layout_array)

    def appendindices(initial_array, target_lengths):
        # Check that initial_array and target_lengths have the same number of outer entries
        if len(initial_array) != len(target_lengths):
            raise ValueError(
                "initial_array and target_lengths must have the same number of outer entries.")

        # Process each pair of inner lists
        for i in range(len(initial_array)):
            inner_list = initial_array[i]
            # this should be a list of integers
            target_length = target_lengths[i]
            # Create a list of all integers in the range from 0 to target_length - 1
            available_numbers = list(range(target_length))

            # Iterate through the available numbers and fill the inner list
            for num in available_numbers:
                # Only add the number if it's not already in the inner list
                if num not in inner_list:
                    inner_list.append(num)
                # Stop if we reach the target length
                if len(inner_list) >= target_length:
                    break

        return initial_array
    # Convert data to arrays with explicit record structure
    indexmask = appendindices(indexlist, ak.num(sorted_jets.pt, axis=1))
    lok_ind = ak.local_index(events.Jet)
    IPython.embed()
    lok_ind[eventmask] = indexmask
    IPython.embed()
    #mask_from_indices
    # Convert your Python lists to awkward arrays
    fitPt_ak = ak.Array(fitPt)
    fitEta_ak = ak.Array(fitEta)
    fitPhi_ak = ak.Array(fitPhi)
    fitMass_ak = ak.Array(fitMass)
    # First, get the original jets that pass your selection and sort them by indexmask
    sorted_jet = events.Jet[lok_ind]

    # Take only the first 6 jets per event
    sorted_jets_top6 = sorted_jet[:, :6]

    # Create FitJet collection for the selected events with fit values
    selected_fitJet = sorted_jets_top6  # Start with the sorted structure
    selected_fitJet = ak.with_field(selected_fitJet, fitPt_ak[:, :6], "pt")
    selected_fitJet = ak.with_field(selected_fitJet, fitEta_ak[:, :6], "eta")
    selected_fitJet = ak.with_field(selected_fitJet, fitPhi_ak[:, :6], "phi")
    selected_fitJet = ak.with_field(selected_fitJet, fitMass_ak[:, :6], "mass")
    print(selected_fitJet)
    # Now we need to expand this back to match the full events structure

    # Create a full-sized array filled with empty lists
    full_fitJet = ak.zeros_like(events.Jet, dtype=float)



#    # Add the fit jet fields
#    for field in ak.fields(selected_fitJet):
#        # Start with empty values
#        field_values = ak.zeros_like(events.Jet)
#        new_field_values = ak.to_list(field_values)
#        # Fill in the values for selected events
#        fit_values = selected_fitJet[field]
#        event_indices = ak.flatten(ak.where(eventmask))  # Get the indices of the events from the mask
#        for idx, val in zip(event_indices, fit_values):
#            new_field_values[idx] = val  # Fill only for the masked entries
#        field_values = ak.Array(new_field_values)
#    # Add field_values back to full_fitJet
#        full_fitJet = ak.with_field(full_fitJet, field_values, field)
    IPython.embed()
    # Now add this full-sized array to events
    events = set_ak_column(events, "FitJet", full_fitJet)
    total_chi2 = np.full(len(events), EMPTY_FLOAT)
    total_chi2[eventmask] = ak.Array(fitChi2).flatten()
    events = set_ak_column(events, "FitChi2", fitChi2)
    return events, SelectionResult()


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
          "Jet.phi", "Jet.mass", attach_coffea_behavior, "HLT.*",
          "gen_top_decay", "GenPart.*",
          },
    # "Jet.jetId", "Jet.puId", "Jet.genJetIdx", "GenJet.*",
    produces={"MW1", "MW2", "Mt1", "Mt2", "chi2",
              "deltaRb", "combination_type", "R2b4q",
              },
    jet_pt=None, jet_trigger=None, jet_base_trigger=None, alt_jet_trigger=None,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # example jet selection: at least six jets, lowest jet at least 40 GeV and H_T > 450 GeV
    EF = -99999.0  # define EMPTY_FLOAT
    ht1_sel = (ak.sum(events.Jet.pt, axis=1) >= 1)
    ht_sel = (ak.sum(events.Jet.pt, axis=1) >= 450)
    jet_mask = ((abs(events.Jet.eta) < 2.6))
    jet_mask2 = ((abs(events.Jet.eta) < 2.4) & (events.Jet.pt >= 40.0))
    jet_sel = ak.sum(jet_mask2, axis=1) >= 6
    veto_jet = ~jet_mask
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    light_jet = (jet_mask2) & (events.Jet.btagDeepFlavB < wp_tight)
    # b-tagged jets (tight wp)
    bjet_mask = (jet_mask2) & (events.Jet.btagDeepFlavB >= wp_tight)
    # bjet_indices = indices[bjet_mask][:, :2]
    bjet_sel = ((ak.sum(bjet_mask, axis=1) >= 2) &
                (ak.sum(jet_mask2[:, :2], axis=1) ==
                 ak.sum(bjet_mask[:, :2], axis=1))
                )
    sixjets_sel = (bjet_sel & (ak.sum(light_jet, axis=1) >= 4))
    # Trigger selection step is skipped for QCD MC, which has no Trigger columns
    if not self.dataset_inst.name.startswith("qcd"):
        ones = ak.ones_like(jet_sel)
    # trigger
        jet_trigger_sel = ones if not self.jet_trigger else events.HLT[self.jet_trigger]
        # alt_jet_trigger_sel = ones if not self.jet_trigger else events.HLT[self.alt_jet_trigger]
        jet_base_trigger_sel = ones if not self.jet_base_trigger else events.HLT[
            self.jet_base_trigger]
    else:
        jet_base_trigger_sel = True
    mwref = 80.4
    mwsig = 11.01  # 12
    mtsig = 27.07  # 15
    mu_tt = 2.07
    mu_w = 0.88
    def m(j1, j2): return (j1.add(j2)).mass
    def m3(j1, j2, j3): return (j1.add(j2.add(j3))).mass
    def dr(j1, j2): return j1.delta_r(j2)
    ljets = ak.combinations((events.Jet[light_jet])[sixjets_sel], 4, axis=1)
    bjets = ak.combinations((events.Jet[bjet_mask])[sixjets_sel], 2, axis=1)
    # Building combination light jet mass functions

    def lpermutations(ljets):
        j1, j2, j3, j4 = ljets
        return ak.concatenate([ak.zip([j1, j2, j3, j4]), ak.zip([j1, j3, j2, j4]), ak.zip([j1, j4, j2, j3])], axis=1)

    def bpermutations(bjets):
        j1, j2 = bjets
        return ak.concatenate([ak.zip([j1, j2]), ak.zip([j2, j1])], axis=1)

    def sixjetcombinations(bjets, ljets):
        return ak.cartesian([bjets, ljets], axis=1)

    # def mt(sixjets):
    #     b1, b2 = ak.unzip(ak.unzip(sixjets)[0])
    #     j1, j2, j3, j4 = ak.unzip(ak.unzip(sixjets)[1])
    #     mt1 = m3(b1, j1, j2)
    #     mt2 = m3(b2, j3, j4)
    #     mw1 = m(j1, j2)
    #     mw2 = m(j3, j4)
    #     drbb = dr(b1, b2)
    #     chi2 = ak.sum([
    #         ((mw1 - mwref) ** 2) / mwsig ** 2,
    #         ((mw2 - mwref) ** 2) / mwsig ** 2,
    #         ((mt1 - mt2) ** 2) / (2 * (mtsig ** 2))],
    #         axis=0,
    #     )
    #     if len(chi2) > 0:
    #         bestc2 = ak.argmin(chi2, axis=1, keepdims=True)
    #         return mt1[bestc2], mt2[bestc2], mw1[bestc2], mw2[bestc2], drbb[bestc2], chi2[bestc2], bestc2, sixjets
    #     else:
    #         return [[EF]], [[EF]], [[EF]], [[EF]], [[EF]], [[EF]], [[EF]], [[EF]]

    def mt(sixjets):
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

    mt_result = mt(sixjetcombinations(bpermutations(
        ak.unzip(bjets)), lpermutations(ak.unzip(ljets))))
    chi2_cut = 50
    mt_result_filled = np.full((6, ak.num(events, axis=0)), EF)
    for i in range(6):
        (mt_result_filled[i])[sixjets_sel] = ak.flatten(mt_result[i])

    chi2_sel = ak.Array(
        (mt_result_filled[5] < chi2_cut) & (mt_result_filled[5] > -1))
    chi2_sel1 = ak.Array(
        (mt_result_filled[5] < 25) & (mt_result_filled[5] > -1))
    chi2_sel2 = ak.Array(
        (mt_result_filled[5] < 10) & (mt_result_filled[5] > -1))
    chi2_sel3 = ak.Array(
        (mt_result_filled[5] < 5) & (mt_result_filled[5] > -1))
    # verify_jet_trigger = (check_trigger == True)

    events = set_ak_column(events, "Mt1", mt_result_filled[0])
    events = set_ak_column(events, "Mt2", mt_result_filled[1])
    events = set_ak_column(events, "MW1", mt_result_filled[2])
    events = set_ak_column(events, "MW2", mt_result_filled[3])
    events = set_ak_column(events, "deltaRb", mt_result_filled[4])
    events = set_ak_column(events, "chi2", mt_result_filled[5])

    def combinationtype(bestcomb, correctcomb):
        b1, b2 = ak.unzip(ak.unzip(bestcomb)[0])
        j1, j2, j3, j4 = ak.unzip(ak.unzip(bestcomb)[1])

        b1cor = correctcomb[:, 0, 1]
        q1cor = correctcomb[:, 0, 3]
        q2cor = correctcomb[:, 0, 4]
        b2cor = correctcomb[:, 1, 1]
        q3cor = correctcomb[:, 1, 3]
        q4cor = correctcomb[:, 1, 4]
        drmax = 0.4
        drb11, drb22, drq11 = (dr(b1, b1cor) < drmax), (dr(
            b2, b2cor) < drmax), (dr(j1, q1cor) < drmax)
        drq22, drq33, drq44 = (dr(j2, q2cor) < drmax), (dr(
            j3, q3cor) < drmax), (dr(j4, q4cor) < drmax)
        drq21, drq12 = (dr(j2, q1cor) < drmax), (dr(j1, q2cor) < drmax)
        drq43, drq34 = (dr(j4, q3cor) < drmax), (dr(j3, q4cor) < drmax)
        drb21, drb12, drq31 = (dr(b2, b1cor) < drmax), (dr(
            b1, b2cor) < drmax), (dr(j3, q1cor) < drmax)
        drq42, drq13, drq24 = (dr(j4, q2cor) < drmax), (dr(
            j1, q3cor) < drmax), (dr(j2, q4cor) < drmax)
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
        type_unfilled = combinationtype(
            (mt_result[7])[mt_result[6]], events.gen_top_decay[sixjets_sel])
        type[0][sixjets_sel] = type_unfilled
        type = ak.flatten(type)
    else:
        type = -1
    events = set_ak_column(events, "combination_type", type)

    if (len(ak.unzip(mt_result[6][:])) > 1):
        if ((len(ak.unzip(ak.unzip(mt_result[6][:])[0])) > 1) & (len(ak.unzip(ak.unzip(mt_result[6][:])[1])) > 3)):
            R2b4q = (
                ak.unzip(ak.unzip(mt_result[6][:])[0])[
                    0].pt + ak.unzip(ak.unzip(mt_result[6][:])[0])[1].pt
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

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Jet": {"MyCustomJetCollection": indices_applied_to_Jet}}
    return events, SelectionResult(
        steps={
            "All": ht1_sel,
            "BaseTrigger": jet_base_trigger_sel,
            # "AltTrigger": alt_jet_trigger_sel,
            "Trigger": jet_trigger_sel,
            "HT": ht_sel,
            "jet": jet_sel,
            "BTag": bjet_sel,
            "SixJets": sixjets_sel,
            "Chi2": chi2_sel,
            "n25Chi2": chi2_sel1,
            "n10Chi2": chi2_sel2,
            "n5Chi2": chi2_sel3,
        },
        objects={
            "Jet": {
                "Jet": sorted_indices_from_mask(jet_mask, events.Jet.pt, ascending=False),
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
    year = self.config_inst.campaign.x.year

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
            2017: "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
            # "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2" or "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2"
            # or "PFHT430_SixPFJet40_PFBTagCSV_1p5"
            # Base Trigger: "PFHT370"
            2018: "PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94",
            # or "HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59")
        }[year]
        self.uses.add(f"HLT.{self.alt_jet_trigger}")
