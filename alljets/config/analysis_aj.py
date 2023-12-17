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
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.config_util import (
    get_root_processes_from_campaign, add_shift_aliases, get_shifts_from_sources, add_category,
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
    # "data_e_b",
    # "data_e_c",
    # "data_e_d",
    # "data_e_e",
    # "data_e_f",
    # "data_mu_b",
    # "data_mu_c",
    # "data_mu_d",
    # "data_mu_e",
    # "data_mu_f",
    # backgrounds
    "tt_sl_powheg",
    "tt_dl_powheg",
    "st_tchannel_t_powheg",
    "st_tchannel_tbar_powheg",
    "st_twchannel_t_powheg",
    "st_twchannel_tbar_powheg",
    # These are not in 2017 nano v9:
    # "st_schannel_lep_amcatnlo",
    # "st_schannel_had_amcatnlo",
    # "dy_lep_pt50To100_amcatnlo",
    # "dy_lep_pt100To250_amcatnlo",
    # "dy_lep_pt250To400_amcatnlo",
    # "dy_lep_pt400To650_amcatnlo",
    # "dy_lep_pt650_amcatnlo",
    # signals
    "tt_fh_powheg",
]
for dataset_name in dataset_names:
    # add the dataset
    dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

    # for testing purposes, limit the number of files to 2
    for info in dataset.info.values():
        info.n_files = min(info.n_files, 2)

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
# https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
cfg.x.luminosity = Number(41480, {
    "lumi_13TeV_2017": 0.02j,
    "lumi_13TeV_1718": 0.006j,
    "lumi_13TeV_correlated": 0.009j,
})

# b-tag working points (NEW)
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

# external files
json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-849c6a6e"
cfg.x.external_files = DotDict.wrap({
    # lumi files
    "lumi": {
        "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },

    # muon scale factors
    "muon_sf": (f"{json_mirror}/POG/MUO/{year}_UL/muon_Z.json.gz", "v1"),
})

# target file size after MergeReducedEvents in MB
cfg.x.reduced_file_size = 512.0

# columns to keep after certain steps
cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        # general event info
        "run", "luminosityBlock", "event",
        # object info
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Bjet.*", "Jet.btagDeepFlavB", "Jet.hadronFlavour",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.pfRelIso04_all",
        "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",
        "PV.npvs",
        # columns added during selection
        "deterministic_seed", "process_id", "mc_weight", "cutflow.*",
    },
    "cf.MergeSelectionMasks": {
        "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
    "cf.UniteColumns": {
        "*",
    },
})

# event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
get_shifts = functools.partial(get_shifts_from_sources, cfg)
cfg.x.event_weights = DotDict({
    "normalization_weight": [],
    "muon_weight": get_shifts("mu"),
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
# the "selection" entries refer to names of selectors, e.g. in selection/example.py
add_category(
    cfg,
    name="incl",
    selection="cat_incl",
    label="inclusive",
)
add_category(
    cfg,
    name="2j",
    selection="cat_2j",
    label="2 jets",
)

# add variables
# (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
# and also correspond to the minimal set of columns that coffea's nano scheme requires)
cfg.add_variable(
    name="event",
    expression="event",
    binning=(1, 0.0, 1.0e9),
    x_title="Event number",
    discrete_x=True,
)
cfg.add_variable(
    name="run",
    expression="run",
    binning=(1, 100000.0, 500000.0),
    x_title="Run number",
    discrete_x=True,
)
cfg.add_variable(
    name="lumi",
    expression="luminosityBlock",
    binning=(1, 0.0, 5000.0),
    x_title="Luminosity block",
    discrete_x=True,
)
cfg.add_variable(
    name="n_jet",
    expression="n_jet",
    binning=(11, -0.5, 10.5),
    x_title="Number of jets",
    discrete_x=True,
)
cfg.add_variable(
    name="jets_pt",
    expression="Jet.pt",
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"$p_{T}$ of all jets",
)
cfg.add_variable(
    name="jets_eta",
    expression="Jet.eta",
    binning=(30, -3.0, 3.0),
    unit="GeV",
    x_title=r"$\eta$ of all jets",
)
cfg.add_variable(
    name="jets_phi",
    expression="Jet.phi",
    binning=(40, -3.2, 3.2),
    unit="GeV",
    x_title=r"$\phi$ of all jets",
)
cfg.add_variable(
    name="jet1_pt",
    expression="Jet.pt[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Jet 1 $p_{T}$",
)
cfg.add_variable(
    name="jet1_eta",
    expression="Jet.eta[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(30, -3.0, 3.0),
    x_title=r"Jet 1 $\eta$",
)
cfg.add_variable(
    name="jet1_phi",
    expression="Jet.phi[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(40, -3.2, 3.2),
    x_title=r"Jet 1 $\phi$",
)
cfg.add_variable(
    name="jet2_pt",
    expression="Jet.pt[:,1]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Jet 2 $p_{T}$",
)
cfg.add_variable(
    name="jet2_eta",
    expression="Jet.eta[:,1]",
    null_value=EMPTY_FLOAT,
    binning=(30, -3.0, 3.0),
    x_title=r"Jet 2 $\eta$",
)
cfg.add_variable(
    name="jet2_phi",
    expression="Jet.phi[:,1]",
    null_value=EMPTY_FLOAT,
    binning=(40, -3.2, 3.2),
    x_title=r"Jet 2 $\phi$",
)
cfg.add_variable(
    name="ht",
    expression=lambda events: ak.sum(events.Jet.pt, axis=1),
    binning=(40, 0.0, 800.0),
    unit="GeV",
    x_title="$H_T$",
)
# weights
cfg.add_variable(
    name="mc_weight",
    expression="mc_weight",
    binning=(200, -10, 10),
    x_title="MC weight",
)
# cutflow variables
cfg.add_variable(
    name="cf_jet1_pt",
    expression="cutflow.jet1_pt",
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Jet 1 $p_{T}$",
)
# New variables
cfg.add_variable(
    name="n_bjet",
    expression="n_bjet",
    binning=(11, -0.5, 10.5),
    x_title="Number of Bjets",
)
cfg.add_variable(
    name="bjet1_pt",
    expression="Bjet.pt[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(15, 0.0, 300.0),
    unit="GeV",
    x_title=r"BJet 1 $p_{T}$",
)
cfg.add_variable(
    name="bjet2_pt",
    expression="Bjet.pt[:,1]",
    null_value=EMPTY_FLOAT,
    binning=(15, 0.0, 300.0),
    unit="GeV",
    x_title=r"BJet 2 $p_{T}$",
)
cfg.add_variable(
    name="jet1_btag",
    expression="Jet.btagDeepFlavB[:,0]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0, 1),
    x_title=r"Jet 1 bTag",
)
cfg.add_variable(
    name="jet2_btag",
    expression="Jet.btagDeepFlavB[:,1]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0, 1),
    x_title=r"Jet 2 bTag",
)
cfg.add_variable(
    name="jet3_btag",
    expression="Jet.btagDeepFlavB[:,2]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0, 1),
    x_title=r"Jet 3 bTag",
)
cfg.add_variable(
    name="jet4_btag",
    expression="Jet.btagDeepFlavB[:,3]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0, 1),
    x_title=r"Jet 4 bTag",
)
cfg.add_variable(
    name="jet5_btag",
    expression="Jet.btagDeepFlavB[:,4]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0, 1),
    x_title=r"Jet 5 bTag",
)
cfg.add_variable(
    name="jet6_btag",
    expression="Jet.btagDeepFlavB[:,5]",
    null_value=EMPTY_FLOAT,
    binning=(40, 0, 1),
    x_title=r"Jet 6 bTag",
)
