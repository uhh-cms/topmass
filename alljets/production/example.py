# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.production.util import attach_coffea_behavior
# from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        # nano columns
        "Jet.pt", "Bjet.pt", "LightJet.pt", "Jet.phi", "Bjet.phi",
        "LightJet.phi", "Jet.eta", "Bjet.eta", "LightJet.eta",
        "Jet.mass", "VetoJet.pt", "Bjet.mass", "LightJet.mass",
        "event", attach_coffea_behavior,
    },
    produces={
        # new columns
        "ht", "n_jet", "n_bjet",
        # "deltaR",
        "deltaRb",
        "MW1",
        "MW2",
        "Mt1",
        "Mt2",
        "chi2",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    jetcollections = {
        "Bjet": {
            "type_name": "Jet",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G",
        },
        "LightJet": {
            "type_name": "Jet",
            "check_attr": "metric_table",
            "skip_fields": "*Idx*G",
        },
    }
    events = self[attach_coffea_behavior](events, jetcollections, **kwargs)
    events = set_ak_column(events, "ht", (ak.sum(events.Jet.pt, axis=1) + ak.sum(events.VetoJet.pt, axis=1)))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1), value_type=np.int32)
    mwref = 80.4
    mwsig = 12
    mtref = 172.5
    mtsig = 15
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    dr = lambda j1, j2: j1.delta_r(j2)
    m = lambda j1, j2: (j1.add(j2)).mass
    m3 = lambda j1, j2, j3: (j1.add(j2.add(j3))).mass
    ljets = ak.combinations(events.LightJet, 4, axis=1)
    bjets = ak.combinations(events.Bjet, 2, axis=1)
    # Building combination light jet mass functions
    def lpermutations(ljets):
        j1, j2, j3, j4 = ljets
        return ak.concatenate([ak.zip([j1, j2, j3, j4]), ak.zip([j1, j3, j2, j4]), ak.zip([j1, j4, j2, j3])], axis=1)

    def bpermutations(bjets):
        j1, j2 = bjets
        return ak.concatenate([ak.zip([j1, j2]), ak.zip([j2, j1])], axis=1)

    def sixjetcombinations(bjets, ljets):
        return ak.cartesian([bjets, ljets], axis=1)

    # def mw(j1, j2, j3, j4):
    #     mw1 = m(j1, j2)
    #     mw2 = m(j3, j4)
    #     chi2 = ak.sum([(mw1 - mwref) ** 2, (mw2 - mwref) ** 2], axis=0)
    #     bestc2 = ak.argmin(chi2, axis=1, keepdims=True)
    #     return mw1[bestc2], mw2[bestc2]

    def mt(sixjets):
        b1, b2 = ak.unzip(ak.unzip(sixjets)[0])
        j1, j2, j3, j4 = ak.unzip(ak.unzip(sixjets)[1])
        mt1 = m3(b1, j1, j2)
        mt2 = m3(b2, j3, j4)
        mw1 = m(j1, j2)
        mw2 = m(j3, j4)
        chi2 = ak.sum([
            ((mw1 - mwref) ** 2) / mwsig,
            ((mw2 - mwref) ** 2) / mwsig,
            ((mt1 - mtref) ** 2) / mtsig,
            ((mt2 - mtref) ** 2) / mtsig],
            axis=0,
        )
        bestc2 = ak.argmin(chi2, axis=1, keepdims=True)
        return mt1[bestc2], mt2[bestc2], mw1[bestc2], mw2[bestc2], chi2[bestc2]

    # events = set_ak_column(events, "deltaR", ak.min(dr(*ljets), axis=1))
    events = set_ak_column(events, "deltaRb", ak.min(dr(*ak.unzip(bjets)), axis=1))
    # Mass of W1
    # import IPython
    # IPython.embed()
    events = set_ak_column(events, "Mt1", mt(sixjetcombinations(bpermutations(ak.unzip(bjets)),
                                                                lpermutations(ak.unzip(ljets)),
                                                                ))[0])
    events = set_ak_column(events, "Mt2", mt(sixjetcombinations(bpermutations(ak.unzip(bjets)),
                                                                lpermutations(ak.unzip(ljets)),
                                                                ))[1])
    events = set_ak_column(events, "MW1", mt(sixjetcombinations(bpermutations(ak.unzip(bjets)),
                                                                lpermutations(ak.unzip(ljets)),
                                                                ))[2])
    events = set_ak_column(events, "MW2", mt(sixjetcombinations(bpermutations(ak.unzip(bjets)),
                                                                lpermutations(ak.unzip(ljets)),
                                                                ))[3])
    events = set_ak_column(events, "chi2", mt(sixjetcombinations(bpermutations(ak.unzip(bjets)),
                                                                 lpermutations(ak.unzip(ljets)),
                                                                 ))[4])
    events = set_ak_column(
        events, "n_bjet",
        ak.sum((events.Jet.btagDeepFlavB >= wp_tight), axis=1),
        value_type=np.int32,
    )

    return events


@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt",
        "Jet.btagDeepFlavB",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.jet6_pt",
        "cutflow.ht",
        "cutflow.jet1_pt",
        "cutflow.n_jet",
        "cutflow.n_bjet",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],
    **kwargs,
) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # apply object masks and create new collections
    # reduced_events = create_collections_from_masks(events, object_masks)

    # create category ids per event and add categories back to the
    events = self[category_ids](
        # reduced_events,
        # target_events=events,
        events,
        **kwargs,
    )

    # add cutflow columns
    events = set_ak_column(events, "cutflow.jet6_pt", Route("Jet.pt[:,5]").apply(events, EMPTY_FLOAT))
    events = set_ak_column(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "cutflow.jet1_pt", Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column(events, "cutflow.n_jet", ak.num(events.Jet.pt, axis=1))
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    events = set_ak_column(events, "cutflow.n_bjet", ak.sum((events.Jet.btagDeepFlavB >= wp_tight), axis=1))
    return events


@producer(
    uses={
        features, category_ids, normalization_weights, muon_weights, deterministic_seeds,
    },
    produces={
        features, category_ids, normalization_weights, muon_weights, deterministic_seeds,
    },
)
def example(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

    return events
