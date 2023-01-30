# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

import os
import re

from scinum import Number
import law
import order as od
import cmsdb
import cmsdb.campaigns.run2_2017_nano_v9

from columnflow.util import DotDict, get_root_processes_from_campaign

from topmass.config.styles import stylize_processes
from topmass.config.categories import add_categories
from topmass.config.variables import add_variables
from topmass.config.met_filters import add_met_filters


thisdir = os.path.dirname(os.path.abspath(__file__))


#
# the main analysis object
#

analysis_topmass = ana = od.Analysis(
    name="analysis_topmass",
    id=1,
)

# analysis-global versions
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf_prod.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("TM_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}


#
# 2017 standard config
#

# copy the campaign, which in turn copies datasets and processes
campaign_run2_2017_nano_v9 = (
    cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9.copy()
)

# get all root processes
procs = get_root_processes_from_campaign(campaign_run2_2017_nano_v9)

# create a config by passing the campaign, so id and name will be identical
config_2017 = cfg = ana.add_config(campaign_run2_2017_nano_v9)

# add processes we are interested in
cfg.add_process(procs.n.data)
cfg.add_process(procs.n.tt)
cfg.add_process(procs.n.st)
cfg.add_process(procs.n.dy)
cfg.add_process(procs.n.w_lnu)

# configure colors, labels, etc
stylize_processes(cfg)

# add datasets we need to study
dataset_names = [
    # signal
    "tt_sl_powheg",
    "tt_dl_powheg",
    "tt_fh_powheg",
    # backgrounds
    "st_tchannel_t_powheg",
    "st_tchannel_tbar_powheg",
    "st_twchannel_t_powheg",
    "st_twchannel_tbar_powheg",
    "st_schannel_lep_amcatnlo",
    "st_schannel_had_amcatnlo",
    "w_lnu_madgraph",
    "dy_lep_m50_ht200to400_madgraph",
    "dy_lep_m50_ht400to600_madgraph",
    "dy_lep_m50_ht600to800_madgraph",
    "dy_lep_m50_ht800to1200_madgraph",
    "dy_lep_m50_ht1200to2500_madgraph",
    "dy_lep_m50_ht2500_madgraph",
]
"""
# data
"data_e_b",
"data_e_c",
"data_e_d",
"data_e_e",
"data_e_f",
"data_mu_b",
"data_mu_c",
"data_mu_d",
"data_mu_e",
"data_mu_f",
# backgrounds
"tt_sl_powheg",
"tt_dl_powheg",
"tt_fh_powheg",
"st_tchannel_t_powheg",
"st_tchannel_tbar_powheg",
"st_twchannel_t_powheg",
"st_twchannel_tbar_powheg",
"st_schannel_lep_amcatnlo",
"st_schannel_had_amcatnlo",
"dy_lep_pt50To100_amcatnlo",
"dy_lep_pt100To250_amcatnlo",
"dy_lep_pt250To400_amcatnlo",
"dy_lep_pt400To650_amcatnlo",
"dy_lep_pt650_amcatnlo",
"""
for dataset_name in dataset_names:
    dataset = cfg.add_dataset(campaign_run2_2017_nano_v9.get_dataset(dataset_name))

    # add aux info to datasets
    if dataset.name.startswith(("st", "tt")):
        dataset.x.has_top = True
    if dataset.name.startswith("tt"):
        dataset.x.is_ttbar = True

    # temporary adjustment: set the number of files per dataset to 2 for faster prototyping
    for info in dataset.info.values():
        info.n_files = 2

# default objects, such as calibrator, selector, producer, ml model, inference model, etc
cfg.x.default_calibrator = "default"
cfg.x.default_selector = "default"
cfg.x.default_producer = "default"
cfg.x.default_ml_model = None
cfg.x.default_inference_model = None
cfg.x.default_categories = ("incl",)
cfg.x.default_variables = "n_l"

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
    "test": ["met_filter", "jet"],
}

# 2017 luminosity with values in inverse pb and uncertainties taken from
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=176#LumiComb
cfg.x.luminosity = Number(
    41480,
    {
        "lumi_13TeV_2017": 0.02j,
        "lumi_13TeV_1718": 0.006j,
        "lumi_13TeV_correlated": 0.009j,
    },
)

# 2018 minimum bias cross section in mb (milli) for creating PU weights, values from
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
cfg.x.minbias_xs = Number(69.2, 0.046j)

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


# helper to add column aliases for both shifts of a source
def add_aliases(
    shift_source: str,
    aliases: dict,
    selection_dependent: bool = False,
):
    aux_key = "column_aliases" + ("_selection_dependent" if selection_dependent else "")
    for direction in ["up", "down"]:
        shift = cfg.get_shift(od.Shift.join_name(shift_source, direction))
        _aliases = shift.x(aux_key, {})
        # format keys and values
        inject_shift = lambda s: re.sub(r"\{([^_])", r"{_\1", s).format(
            **shift.__dict__
        )
        _aliases.update(
            {inject_shift(key): inject_shift(value) for key, value in aliases.items()}
        )
        # extend existing or register new column aliases
        shift.set_aux(aux_key, _aliases)


# register shifts
cfg.add_shift(name="nominal", id=0)
cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})


# external files
cfg.x.external_files = DotDict.wrap(
    {
        # files from TODO
        "lumi": {
            "golden": (
                "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
                "v1",
            ),  # noqa
            "normtag": (
                "/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json",
                "v1",
            ),
        },
        # files from
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
        "pu": {
            "json": (
                "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt",
                "v1",
            ),  # noqa
            "mc_profile": (
                "https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py",
                "v1",
            ),  # noqa
            "data_profile": {
                "nominal": (
                    "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root",
                    "v1",
                ),  # noqa
                "minbias_xs_up": (
                    "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root",
                    "v1",
                ),  # noqa
                "minbias_xs_down": (
                    "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root",
                    "v1",
                ),  # noqa
            },
        },
    }
)

# target file size after MergeReducedEvents in MB
cfg.x.reduced_file_size = 512.0

# columns to keep after certain steps
cfg.x.keep_columns = DotDict.wrap(
    {
        "cf.ReduceEvents": {
            # general event info
            "run",
            "luminosityBlock",
            "event",
            # object info
            "Jet.pt",
            "Jet.eta",
            "Jet.phi",
            "Jet.mass",
            "Jet.btagDeepFlavB",
            "Jet.hadronFlavour",
            "Bjet.pt",
            "Bjet.eta",
            "Bjet.phi",
            "Bjet.mass",
            "Bjet.btagDeepFlavB",
            "n_bjet",
            "Electron.pt",
            "Electron.eta",
            "Electron.phi",
            "Electron.mass",
            "Electron.charge",
            "Muon.pt",
            "Muon.eta",
            "Muon.phi",
            "Muon.mass",
            "m_min_lb",
            "MET.pt",
            "MET.phi",
            "PV.npvs",
            "normalization_weight",
            "Muon.pfRelIso04_all",
            "Electron.mvaFall17V2Iso_WP80"
            # columns added during selection
            "channel",
            "process_id",
            "category_ids",
            "mc_weight",
            "channel_id",
            "deterministic_seed",
            "cutflow.*",
        },
        "cf.MergeSelectionMasks": {
            "mc_weight",
            "normalization_weight",
            "process_id",
            "category_ids",
            "cutflow.*",
            "channel_id",
        },
    }
)

# event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
get_shifts = lambda *names: sum(
    ([cfg.get_shift(f"{name}_up"), cfg.get_shift(f"{name}_down")] for name in names), []
)
cfg.x.event_weights = DotDict()
cfg.x.event_weights["normalization_weight"] = []


# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
cfg.x.versions = {}

# cannels
cfg.add_channel(name="ee", id=1)
cfg.add_channel(name="mumu", id=2)
cfg.add_channel(name="emu", id=3)

# 2017 b-tag working points
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
    }
)

# add categories
add_categories(cfg)

# add variables
add_variables(cfg)

# add met filters
add_met_filters(cfg)
