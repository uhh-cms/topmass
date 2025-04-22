# coding: utf-8

"""
Configuration of the topmass_alljets analysis.
"""

import os
import functools

import law
import order as od
from scinum import Number

from columnflow.util import DotDict, maybe_import
from columnflow.config_util import (
    get_root_processes_from_campaign, add_shift_aliases, get_shifts_from_sources,
    verify_config_processes,
)

ak = maybe_import("awkward")


#
# the main analysis object
#

analysis_aj = ana = od.Analysis(
    name="analysis_aj",
    id=1,
)

# analysis-global versions
# (see cfg.x.versions below for more info)
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    law.config.get("analysis", "default_columnar_sandbox"),
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("AJ_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}


#
# setup configs
#

# an example config is setup below, based on cms NanoAOD v9 for Run2 2017, focussing on
# ttbar and single top MCs, plus single muon data
# update this config or add additional ones to accomodate the needs of your analysis

from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9


# copy the campaign
# (creates copies of all linked datasets, processes, etc. to allow for encapsulated customization)
campaign = campaign_run2_2017_nano_v9.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign)

# create a config by passing the campaign, so id and name will be identical
cfg = ana.add_config(campaign)

# gather campaign data
year = campaign.x.year

# add processes we are interested in
process_names = [
    "data",
    "qcd",
    "tt",
    "st",
]
for process_name in process_names:
    # add the process
    proc = cfg.add_process(procs.get(process_name))

    # configuration of colors, labels, etc. can happen here
    if proc.is_mc:
        proc.color1 = (244, 182, 66) if proc.name == "tt" else (244, 93, 66)

# add datasets we need to study
dataset_names = [
    # data
    # Trigger does not work for set B
    # "data_jetht_b",
    "data_jetht_c",
    "data_jetht_d",
    "data_jetht_e",
    "data_jetht_f",
    # backgrounds
    # "tt_sl_powheg",
    # "tt_dl_powheg",
    # "st_tchannel_t_powheg",
    # "st_tchannel_tbar_powheg",
    # "st_twchannel_t_powheg",
    # "st_twchannel_tbar_powheg",
    # "qcd_ht50to100_madgraph",
    # "qcd_ht100to200_madgraph",
    # "qcd_ht200to300_madgraph",
    "qcd_ht300to500_madgraph",
    "qcd_ht500to700_madgraph",
    "qcd_ht700to1000_madgraph",
    "qcd_ht1000to1500_madgraph",
    "qcd_ht1500to2000_madgraph",
  #  "qcd_ht2000_madgraph",
    # These are not in 2017 nano v9:
    # "st_schannel_lep_amcatnlo",
    # "st_schannel_had_amcatnlo",
    # "dy_lep_pt50To100_amcatnlo",
    # "dy_lep_pt100To250_amcatnlo",
    # "dy_lep_pt250To400_amcatnlo",
    # "dy_lep_pt400To650_amcatnlo",
    # "dy_lep_pt650_amcatnlo",
    "qcd_ht2000toinf_madgraph",
    # signals
    "tt_fh_powheg",
]
for dataset_name in dataset_names:
    # add the dataset
    dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

    # for testing purposes, limit the number of files to 2
    for info in dataset.info.values():
        info.n_files = min(info.n_files, 1)
    # for info in dataset.info.values():
    #     info.n_files = min(info.n_files, 2)
    # # Add has_top tag to tt events
    if dataset_name.startswith("tt_"):
        dataset.add_tag("has_top")

# verify that the root process of all datasets is part of any of the registered processes
verify_config_processes(cfg, warn=True)

# default objects, such as calibrator, selector, producer, ml model, inference model, etc
cfg.x.default_calibrator = "example"
cfg.x.default_selector = "example"
cfg.x.default_producer = "example"
cfg.x.default_ml_model = None
cfg.x.default_inference_model = "example"
cfg.x.default_categories = ("incl",)
cfg.x.default_variables = ("n_jet", "jet1_pt")
cfg.x.default_weight_producer = "all_weights"

# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
cfg.x.process_groups = {}

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
cfg.x.dataset_groups = {}

# category groups for conveniently looping over certain categories
# (used during plotting)
cfg.x.category_groups = {}

# variable groups for conveniently looping over certain variables
# (used during plotting)
cfg.x.variable_groups = {}

# shift groups for conveniently looping over certain shifts
# (used during plotting)
cfg.x.shift_groups = {}

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
cfg.x.selector_step_groups = {
    "default": ["muon", "jet"],
}

# custom method and sandbox for determining dataset lfns
cfg.x.get_dataset_lfns = None
cfg.x.get_dataset_lfns_sandbox = None

# whether to validate the number of obtained LFNs in GetDatasetLFNs
# (currently set to false because the number of files per dataset is truncated to 2)
cfg.x.validate_dataset_lfns = False

# lumi values in inverse pb
# https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations: 41480
# Luminosity by trigger: PFHT380_SixPFJet32_DoublePFBTagCSV_2p2: 36674
# PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2: 27121
cfg.x.luminosity = Number(36674, {
    "lumi_13TeV_2017": 0.02j,
    "lumi_13TeV_1718": 0.006j,
    "lumi_13TeV_correlated": 0.009j,
})

# b-tag working points
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
cfg.x.btag_working_points = DotDict.wrap(
    {
        "deepjet": {
            "loose": 0.0532,
            "medium": 0.3040,
            "tight": 0.7476,
        },
        "deepcsv": {
            "loose": 0.1355,
            "medium": 0.4506,
            "tight": 0.7738,
        },
    },
)

# JEC uncertainty sources propagated to btag scale factors
# (names derived from contents in BTV correctionlib file)
cfg.x.btag_sf_jec_sources = [
    "",  # same as "Total"
    "Absolute",
    "AbsoluteMPFBias",
    "AbsoluteScale",
    "AbsoluteStat",
    f"Absolute_{year}",
    "BBEC1",
    f"BBEC1_{year}",
    "EC2",
    f"EC2_{year}",
    "FlavorQCD",
    "Fragmentation",
    "HF",
    f"HF_{year}",
    "PileUpDataMC",
    "PileUpPtBB",
    "PileUpPtEC1",
    "PileUpPtEC2",
    "PileUpPtHF",
    "PileUpPtRef",
    "RelativeBal",
    "RelativeFSR",
    "RelativeJEREC1",
    "RelativeJEREC2",
    "RelativeJERHF",
    "RelativePtBB",
    "RelativePtEC1",
    "RelativePtEC2",
    "RelativePtHF",
    "RelativeSample",
    f"RelativeSample_{year}",
    "RelativeStatEC",
    "RelativeStatFSR",
    "RelativeStatHF",
    "SinglePionECAL",
    "SinglePionHCAL",
    "TimePtEta",
]

# name of the btag_sf correction set and jec uncertainties to propagate through
cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources, "btagDeepFlavB")

# names of muon correction sets and working points
# (used in the muon producer)
cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}_UL")

# register shifts
cfg.add_shift(name="nominal", id=0)

# tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
# (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

# fake jet energy correction shift, with aliases flaged as "selection_dependent", i.e. the aliases
# affect columns that might change the output of the event selection
cfg.add_shift(name="jec_up", id=20, type="shape")
cfg.add_shift(name="jec_down", id=21, type="shape")
add_shift_aliases(
    cfg,
    "jec",
    {
        "Jet.pt": "Jet.pt_{name}",
        "Jet.mass": "Jet.mass_{name}",
        "MET.pt": "MET.pt_{name}",
        "MET.phi": "MET.phi_{name}",
    },
)

# event weights due to muon scale factors
cfg.add_shift(name="mu_up", id=10, type="shape")
cfg.add_shift(name="mu_down", id=11, type="shape")
add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})

# Renormalization and scale factor shifts
cfg.add_shift(name="murmuf_up", id=140, type="shape")
cfg.add_shift(name="murmuf_down", id=141, type="shape")
add_shift_aliases(
    cfg,
    "murmuf",
    {
        "murmuf_weight": "murmuf_weight_{direction}",
        "normalized_murmuf_weight": "normalized_murmuf_weight_{direction}",
    },
)

# Pdf shifts
cfg.add_shift(name="pdf_up", id=130, type="shape")
cfg.add_shift(name="pdf_down", id=131, type="shape")
add_shift_aliases(
    cfg,
    "pdf",
    {
        "pdf_weight": "pdf_weight_{direction}",
        "normalized_pdf_weight": "normalized_pdf_weight_{direction}",
    },
)

# external files
json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-377439e8"
year = "2017"
corr_postfix = ""
cfg.x.external_files = DotDict.wrap({
    # lumi files
    "lumi": {
        "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },

    # muon scale factors
    "muon_sf": (f"{json_mirror}/POG/MUO/{year}_UL/muon_Z.json.gz", "v1"),

    # btag scale factor
    "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

    # pileup weight corrections
    "pu_sf": (f"{json_mirror}/POG/LUM/{year}{corr_postfix}_UL/puWeights.json.gz", "v1"),

    # jet energy correction
    "jet_jerc": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

    # electron scale factors
    "electron_sf": (f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

    # prototype trigger weight corrections
    "trig_sf": ("/afs/desy.de/user/d/davidsto/public/mirrors/trigger_correction_HT350_CSV.json.gz", "v1"),
})

cfg.x.trigger = {
    "tt_fh": ["PFHT380_SixPFJet32_DoublePFBTagCSV_2p2", "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2"],
}

cfg.x.ref_trigger = {
    "tt_fh": ["Physics"],
}

# IsoMu24, Mu50, PFHT350 for MC with all events: Physics
#
# if year == 2017:
#     cfg.x.trigger = {
#         "tt_fh": ["PFHT380_SixPFJet32_DoublePFBTagCSV_2p2"]
#     }
#     cfg.x.ref_trigger= {
#         "tt_fh": ["PFHT370"]
#     }
# elif year == 2018:
#     cfg.x.trigger = {
#         "tt_fh": ["PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94"]
#     }
#     cfg.x.ref_trigger= {
#         "tt_fh": ["PFHT370"]
#     }
# elif year == 2016:
#     cfg.x.trigger = {
#         "tt_fh": ["PFHT400_SixJet30_DoubleBTagCSV_p056"]
#     }
#     cfg.x.ref_trigger= {
#         "tt_fh": ["PFHT370"]
#     }

# target file size after MergeReducedEvents in MB
cfg.x.reduced_file_size = 512.0

# columns to keep after certain steps
cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        # general event info
        "run", "luminosityBlock", "event",
        # object info
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Bjet.*", "VetoJet.*", "LightJet.*", "JetsByBTag.*",
        # "EventJet.*",
        "Jet.btagDeepFlavB", "Jet.hadronFlavour",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.pfRelIso04_all",
        "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
        "PV.npvs",
        "FitJet.*",  "FitChi2",
        "PV.npvs", "PV.npvsGood", "DeltaR", "GenPart.*",
        "MW1", "MW2", "Mt1", "Mt2", "chi2", "deltaRb", "HLT.Mu50", "HLT.PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
        "HLT.PFHT380_SixPFJet32_DoublePFBTagCSV_2p2", "HLT.IsoMu24", "HLT.PFHT370", "HLT.PFHT350", "HLT.Physics",
        "HLT.PFHT1050", "HLT.PFHT890",
        # columns added during selection
        "deterministic_seed", "process_id", "mc_weight", "cutflow.*", "pdf_weight", "trig_weight",
        "murmuf_weight", "pu_weight", "btag_weight", "combination_type", "R2b4q", "trig_ht",
    },
    "cf.MergeSelectionMasks": {
        "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
    "cf.UniteColumns": {
        "*_weight", "Jet.*", "combination_type", "ht", "Mt*", "MW*", "trig_bits",
    },
})

# event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
# TODO: Add BTag weight shifts
get_shifts = functools.partial(get_shifts_from_sources, cfg)
cfg.x.event_weights = DotDict({
     "normalization_weight": [],
    # "btag_weight": [],
    # "trig_weight": [],
    # "muon_weight": get_shifts("mu"),
    # "pdf_weight": get_shifts("pdf"),
    # "murmuf_weight": get_shifts("murmuf"),
})

# versions per task family, either referring to strings or to callables receving the invoking
# task instance and parameters to be passed to the task family
cfg.x.versions = {
    # "cf.CalibrateEvents": "prod1",
    # "cf.SelectEvents": (lambda cls, inst, params: "prod1" if params.get("selector") == "default" else "dev1"),
    # ...
}

# channels
# (just one for now)
cfg.add_channel(name="mutau", id=1)

# add categories using the "add_category" tool which adds auto-generated ids
from alljets.config.categories import add_categories
add_categories(cfg)
# add variables
from alljets.config.variables import add_variables
add_variables(cfg)
