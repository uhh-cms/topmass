
"""
Jet selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt", "Jet.eta","Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.jetId", "Jet.puId"},
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
    bjet_sel = (ak.sum(bjet_mask, axis=1) >= 2) & (ak.sum(jet_mask[:, :2], axis=1) == ak.sum(bjet_mask[:, :2], axis=1))
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
  uses={"Jet.pt", "Jet.eta","Jet.phi", "Jet.mass", "Jet.btagDeepFlavB"},
    produces={"FitJet.pt", "FitJet.eta", "FitJet.phi", "FitJet.mass", "FitChi2"},
    jet_pt=None, jet_trigger=None,sandbox="bash::$CF_REPO_BASE/sandboxes/cmsswtest.sh"
)

def kinFit(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    import pyKinFitTest as pyKinFit

    fitData = pyKinFit.setBestCombi(events.Jet.pt, events.Jet.eta, events.Jet.phi, events.Jet.mass)
    events = set_ak_column(events, "FitJet.pt", fitData[0])
    events = set_ak_column(events, "FitJet.eta", fitData[1])
    events = set_ak_column(events, "FitJet.phi", fitData[2])
    events = set_ak_column(events, "FitJet.mass", fitData[3])
    events = set_ak_column(events, "FitChi2", fitData[4])

    return events,SelectionResult(
    steps={
        },
        objects={
            "Jet": {
                "Jet": sorted_indices_from_mask(jet_mask, events.Jet.pt, ascending=False),
                            },
        },
        aux={
                   },
    )


