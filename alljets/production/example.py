# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.normalization import normalization_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.production.cms.gen_top_decay import gen_top_decay_products

# from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import

from alljets.production.KinFit import kinFit

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        # nano columns
        "Jet.pt",
        "Bjet.pt",
        "LightJet*.pt",
        "Jet.phi",
        "Bjet.phi",
        "LightJet.phi",
        "Jet.eta",
        "Bjet.eta",
        "LightJet.eta",
        "Jet.mass",
        "VetoJet.pt",
        "Bjet.mass",
        "LightJet.mass",
        "event",
        attach_coffea_behavior,
        "HLT.*",
        "Jet.btagDeepFlavB",
        "Mt1",
        "Mt2",
    },
    produces={
        # new columns
        "ht",
        "ht_old",
        "n_jet",
        "n_bjet",
        "maxbtag",
        "secmaxbtag",
        "deltaMt",
        "Jet.pt",
        "Bjet.pt",
        "LightJet*.pt",
        "Jet.phi",
        "Bjet.phi",
        "LightJet.phi",
        "Jet.eta",
        "Bjet.eta",
        "LightJet.eta",
        "Jet.mass",
        "VetoJet.pt",
        "Bjet.mass",
        "LightJet.mass",
        "event",
        attach_coffea_behavior,
        "HLT.*",
        "Jet.btagDeepFlavB",
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
    # events = set_ak_column(events, "ht", (ak.sum(events.Jet.pt, axis=1) + ak.sum(events.VetoJet.pt, axis=1)))
    events = set_ak_column(
        events,
        "ht_old",
        (ak.sum(events.Jet[(abs(events.Jet.eta) < 2.4)].pt, axis=1)),
    )
    events = set_ak_column(
        events,
        "ht",
        (ak.sum(events.Jet[(events.Jet.pt >= 30.0)].pt, axis=1)),
    )
    events = set_ak_column(
        events,
        "n_jet",
        ak.num(events.Jet.pt, axis=1),
        value_type=np.int32,
    )
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    events = set_ak_column(
        events,
        "n_bjet",
        ak.sum((events.Jet.btagDeepFlavB >= wp_tight), axis=1),
        value_type=np.int32,
    )
    events = set_ak_column(
        events,
        "maxbtag",
        (ak.max(events.Jet.btagDeepFlavB, axis=1)),
    )
    # Insert dummy value for one jet events
    secmax = ak.sort(events.Jet.btagDeepFlavB, axis=1, ascending=False)
    empty = ak.singletons(np.full(len(events), EMPTY_FLOAT))
    events = set_ak_column(events, "deltaMt", (events.Mt1 - events.Mt2))
    events = set_ak_column(
        events,
        "secmaxbtag",
        (ak.concatenate([secmax, empty, empty], axis=1)[:, 1]),
    )
    return events


@producer(
    uses={
        # nano columns
        "Jet.pt",
        "Jet.phi",
        "Jet.eta",
        "Jet.mass",
        "event",
        attach_coffea_behavior,
        "Jet.btagDeepFlavB",
        kinFit,
        "gen_top_decay",
    },
    produces={
        # new columns
        kinFit,
        "fitCombinationType",
        "FitW1.*",
        "FitW2.*",
        "FitTop1.*",
        "FitTop2.*",
        "FitRbb",
        # "Mt1", "Mt2", "MW1", "MW2", "chi2", "deltaRb",
    },
)
def kinFitMatch(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    from alljets.scripts.default import combinationtype

    EF = -99999.0
    kinFit_eventmask = len(events) * [True]
    kinFit_jetmask = (events[kinFit_eventmask].Jet.pt >= 40.0) & (abs(events[kinFit_eventmask].Jet.eta) < 2.4)

    events = self[kinFit](events, kinFit_jetmask, kinFit_eventmask, **kwargs)

    if events.gen_top_decay.ndim > 1:
        jetcollections = {
            "FitJet": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "",
            },
            "FitJet.reco": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "",
            },
            "gen_top_decay": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "",
            },
        }
        events = self[attach_coffea_behavior](events, jetcollections, **kwargs)
        fitcomb = combinationtype(
            events.FitJet.reco[kinFit_eventmask][:, 0],
            events.FitJet.reco[kinFit_eventmask][:, 1],
            events.FitJet.reco[kinFit_eventmask][:, 2],
            events.FitJet.reco[kinFit_eventmask][:, 3],
            events.FitJet.reco[kinFit_eventmask][:, 4],
            events.FitJet.reco[kinFit_eventmask][:, 5],
            events.gen_top_decay[kinFit_eventmask],
        )
        full_fitcomb = np.full(len(events), EF)
        full_fitcomb[kinFit_eventmask] = fitcomb
        events = set_ak_column(events, "fitCombinationType", full_fitcomb)
    else:
        events = set_ak_column(events, "fitCombinationType", 0)
        jetcollections = {
            "FitJet": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "",
            },
            "FitJet.reco": {
                "type_name": "Jet",
                "check_attr": "metric_table",
                "skip_fields": "",
            },
        }
        events = self[attach_coffea_behavior](events, jetcollections, **kwargs)

    B1 = events.FitJet[:, 0]
    B2 = events.FitJet[:, 1]
    W1 = events.FitJet[:, 2].add(events.FitJet[:, 3])
    W2 = events.FitJet[:, 4].add(events.FitJet[:, 5])
    Top1 = events.FitJet[:, 0].add(W1)
    Top2 = events.FitJet[:, 1].add(W2)
    events = set_ak_column(events, "FitRbb", B1.delta_r(B2))
    events = set_ak_column(events, "FitB1", B1)
    events = set_ak_column(events, "FitB2", B2)
    events = set_ak_column(events, "FitW1", W1)
    events = set_ak_column(events, "FitW2", W2)
    events = set_ak_column(events, "FitTop1", Top1)
    events = set_ak_column(events, "FitTop2", Top2)

    return events


@producer(
    uses={
        mc_weight,
        category_ids,
        # nano columns
        "Jet.pt",
        "Jet.eta",
        "Jet.phi",
        "Jet.btagDeepFlavB",
    },
    produces={
        mc_weight,
        category_ids,
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

    # add cutflow columns
    events = set_ak_column(
        events,
        "cutflow.jet6_pt",
        Route("Jet.pt[:,5]").apply(events, EMPTY_FLOAT),
    )
    events = set_ak_column(events, "cutflow.ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(
        events,
        "cutflow.jet1_pt",
        Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT),
    )
    events = set_ak_column(events, "cutflow.n_jet", ak.num(events.Jet.pt, axis=1))
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    events = set_ak_column(
        events,
        "cutflow.n_bjet",
        ak.sum((events.Jet.btagDeepFlavB >= wp_tight), axis=1),
    )
    return events


@producer(
    uses={
        features,
        category_ids,
        normalization_weights,
        muon_weights,
        deterministic_seeds,
        kinFitMatch,
        gen_top_decay_products,
        attach_coffea_behavior,
    },
    produces={
        features,
        category_ids,
        normalization_weights,
        muon_weights,
        deterministic_seeds,
        kinFitMatch,
        gen_top_decay_products,
        "gen_top_decay",
        attach_coffea_behavior,
    },
)
def example(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # attach coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)
    # features
    if not self.dataset_inst.has_tag("has_top"):
        events = set_ak_column(events, "gen_top_decay", False)

    events = self[features](events, **kwargs)
    # apply kinematic fit
    events = self[kinFitMatch](events, **kwargs)
    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # muon weights
        # events = self[muon_weights](events, **kwargs)

    return events


@producer(
    uses={
        normalization_weights,
        features,
        category_ids,
        muon_weights,
        deterministic_seeds,
        kinFitMatch,
    },
    produces={
        normalization_weights,
        features,
        category_ids,
        muon_weights,
        deterministic_seeds,
        kinFitMatch,
    },
)
def no_norm(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    if not self.dataset_inst.has_tag("has_top"):
        events = set_ak_column(events, "gen_top_decay", False)
    events = self[features](events, **kwargs)

    # fake kinfit for trig weights creation
    events = set_ak_column(events, "FitChi2", 0)
    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)
        events = set_ak_column(
            events,
            "normalization_weight",
            np.ones(len(events)),
            value_type=np.float32,
        )
        events = set_ak_column(
            events,
            "mc_weight",
            np.ones(len(events)),
            value_type=np.float32,
        )
        # muon weights
        # events = self[muon_weights](events, **kwargs)

    return events


@producer(
    produces={"trig_bits", "trig_bits_orth"},
    channel=["tt_fh"],
)
def trigger_prod(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces column where each bin corresponds to a certain trigger
    """

    arr = ak.singletons(np.zeros(len(events)))
    arr_orth = ak.singletons(np.zeros(len(events)))

    id = 1

    for channel in self.channel:
        ref_trig = self.config_inst.x.ref_trigger[channel]
        for trigger in self.config_inst.x.trigger[channel]:
            trig_passed = ak.singletons(
                ak.flatten(
                    ak.nan_to_none(
                        ak.unzip(ak.where(events.HLT[trigger], id, np.float64(np.nan))),
                    ),
                ),
            )
            trig_passed_orth = ak.flatten(
                ak.singletons(
                    ak.nan_to_none(
                        ak.where(
                            ak.singletons(ak.flatten(ak.unzip(events.HLT[ref_trig]))) &
                            ak.singletons(ak.flatten(ak.unzip(events.HLT[trigger]))),
                            id,
                            np.float64(np.nan),
                        ),
                    ),
                ),
                axis=1,
            )
            # trig_passed_orth = ak.singletons(ak.nan_to_none(
            #     ak.where((events.HLT[ref_trig] & events.HLT[trigger]), id, np.float64(np.nan))
            # ))
            arr = ak.concatenate([arr, trig_passed], axis=1)
            arr_orth = ak.concatenate([arr_orth, trig_passed_orth], axis=1)
            id += 1

    """ for channel, trig_cols in self.config_inst.x.trigger.items():
        for trig_col in trig_cols:
            trig_passed = ak.singletons(ak.nan_to_none(
                ak.where(events.HLT[trig_col], id, np.float64(np.nan))
            ))
            trig_passed_orth = ak.singletons(ak.nan_to_none(
                ak.where((events.HLT[ref_trig] & events.HLT[trig_col]), id, np.float64(np.nan))
            ))
            arr = ak.concatenate([arr, trig_passed], axis=1)
            arr_orth = ak.concatenate([arr_orth, trig_passed_orth], axis=1)
            id += 1 """

    events = set_ak_column(events, "trig_bits", arr)
    events = set_ak_column(events, "trig_bits_orth", arr_orth)

    return events


@trigger_prod.init
def trigger_prod_init(self: Producer) -> None:

    for channel in self.channel:
        for trigger in self.config_inst.x.trigger[channel]:
            self.uses.add(f"HLT.{trigger}")
        self.uses.add(f"HLT{self.config_inst.x.ref_trigger[channel]}")


# producers for single channels
tt_fh_trigger_prod = trigger_prod.derive(
    "tt_fh_trigger_prod",
    cls_dict={"channel": ["tt_fh"]},
)

# Trigger categories
#
# @producer(
#     uses=category_ids,
#     produces=category_ids,
#     version=1,
# )
# def trig_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
#     """
#     Reproduces the category ids to include the trigger categories
#     """

#     events = self[category_ids](events, **kwargs)

#     return events


# @trig_cats.init
# def trig_cats_init(self: Producer) -> None:

#     add_trigger_categories(self.config_inst)
