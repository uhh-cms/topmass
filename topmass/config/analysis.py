# coding: utf-8

"""
Configuration of the HH → bb𝜏𝜏 analysis.
"""

import os
import functools

from scinum import Number
import law
import order as od
import cmsdb
import cmsdb.campaigns.run2_2017_nano_v9

from columnflow.util import DotDict
from columnflow.config_util import get_root_processes_from_campaign, get_shifts_from_sources, add_shift_aliases

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
    "$TM_BASE/sandboxes/venv_columnar_tf.sh",
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
    # backgrounds
    "tt_sl_powheg",
    "tt_dl_powheg",
    "tt_fh_powheg",
    "st_tchannel_t_powheg",
    "st_tchannel_tbar_powheg",
    "st_twchannel_t_powheg",
    "st_twchannel_tbar_powheg",
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

# register shifts
cfg.add_shift(name="nominal", id=0)

cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})

# external files
cfg.x.external_files = DotDict.wrap(
    {# files from TODO
        "lumi": {
            "golden": (
                "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017"\
                "/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt",
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
                "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp"\
                "/UltraLegacy/pileup_latest.txt",
                "v1",
            ),  # noqa
            "mc_profile": (
                "https://raw.githubusercontent.com/cms-sw/cmssw"\
                "/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python"\
                "/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py",
                "v1",
            ),  # noqa
            "data_profile": {
                "nominal": (
                    "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp"\
                    "/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root",
                    "v1",
                ),  # noqa
                "minbias_xs_up": (
                    "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp"\
                    "/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root",
                    "v1",
                ),  # noqa
                "minbias_xs_down": (
                    "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp"\
                    "/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root",
                    "v1",
                ),  # noqa
            },
        },
    },
)

json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-878881a8"
year = "2017"
corr_postfix=""

cfg.x.external_files.update(DotDict.wrap({
        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{year}{corr_postfix}_UL/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz", "v1"),

    }))

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

cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", "2017_UL",)
cfg.x.btag_sf = ("deepJet_shape", ["Absolute", "FlavorQCD",],)
cfg.x.electron_sf_names = ("UL-Electron-ID-SF", "2017", "wp80iso")
cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources)

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
            "VetoBjet.pt",
            "VetoBjet.eta",
            "VetoBjet.phi",
            "VetoBjet.mass",
            "Bjet.btagDeepFlavB",
            "n_bjet",
            "Electron.pt",
            "Electron.eta",
            "Electron.phi",
            "Electron.mass",
            "Electron.charge",
            "Electron.deltaEtaSC",
            "Muon.pt",
            "Muon.eta",
            "Muon.phi",
            "Muon.mass",
            "MET.pt",
            "MET.phi",
            "PV.npvs",
            "PV.npvsGood",
            "normalization_weight",
            "Muon.pfRelIso04_all",
            "Electron.mvaFall17V2Iso_WP80"
            # columns added during selection
            "channel",
            "process_id",
            "m_min_lb",
            "category_ids",
            "mc_weight",
            "pdf_weight*",
            "murmuf_weight*",
            "pu_weight*",
            "btag_weight*",
            "channel_id",
            "m_ll",
            "lepton_pt",
            "trailing_pt",
            "lepton_eta",
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
    },
)

get_shifts = functools.partial(get_shifts_from_sources, cfg)
cfg.x.event_weights = DotDict()
cfg.x.event_weights["normalization_weight"] = []


# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
cfg.x.versions = {}

# cannels
cfg.add_channel(name="ee", id=1)
cfg.add_channel(name="mumu", id=2)
cfg.add_channel(name="emu", id=3)


# add categories
add_categories(cfg)

# add variables
add_variables(cfg)

# add met filters
add_met_filters(cfg)

cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
add_shift_aliases(
    cfg,
    "minbias_xs",
    {
        "pu_weight": "pu_weight_{name}",
        "normalized_pu_weight": "normalized_pu_weight_{name}",
    },
)

cfg.add_shift(name="top_pt_up", id=9, type="shape")
cfg.add_shift(name="top_pt_down", id=10, type="shape")
add_shift_aliases(cfg, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})


cfg.add_shift(name="e_up", id=90, type="shape")
cfg.add_shift(name="e_down", id=91, type="shape")
add_shift_aliases(cfg, "e", {"electron_weight": "electron_weight_{direction}"})

cfg.add_shift(name="mu_up", id=100, type="shape")
cfg.add_shift(name="mu_down", id=101, type="shape")
add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})

btag_uncs = [
    "hf", "lf",
    f"hfstats1_{year}", f"hfstats2_{year}",
    f"lfstats1_{year}", f"lfstats2_{year}",
    "cferr1", "cferr2",
]
for i, unc in enumerate(btag_uncs):
    cfg.add_shift(name=f"btag_{unc}_up", id=110 + 2 * i, type="shape")
    cfg.add_shift(name=f"btag_{unc}_down", id=111 + 2 * i, type="shape")
    add_shift_aliases(
        cfg,
        f"btag_{unc}",
        {
            "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
            "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
        },
    )

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

cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pdf_weight": get_shifts("pdf"),
        "murmuf_weight": get_shifts("murmuf"),
        "normalized_pu_weight": get_shifts("minbias_xs"),
        "normalized_njet_btag_weight": get_shifts(*(f"btag_{unc}" for unc in btag_uncs)),
        "electron_weight": get_shifts("e"),
        "muon_weight": get_shifts("mu"),
    })
