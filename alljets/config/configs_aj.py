# coding: utf-8

"""
Configuration of the ttbar analysis.
"""

from __future__ import annotations

import functools
import itertools
import os
import re

import law
import order as od
import yaml
from columnflow.columnar_util import ColumnCollection, skip_column
from columnflow.config_util import (add_shift_aliases,
                                    get_root_processes_from_campaign,
                                    get_shifts_from_sources,
                                    verify_config_processes)
from columnflow.tasks.external import ExternalFile as Ext
from columnflow.util import DotDict, dev_sandbox
from scinum import Number

thisdir = os.path.dirname(os.path.abspath(__file__))

logger = law.logger.get_logger(__name__)


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
    sync_mode: bool = False,
) -> od.Config:
    # gather campaign data
    run = campaign.x.run
    year = campaign.x.year
    year2 = year % 100

    # some validations
    assert run in {2}
    assert year in {2016, 2017, 2018}

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = od.Config(
        name=config_name,
        id=config_id,
        campaign=campaign,
        aux={
            "sync": sync_mode,
        },
    )

    ################################################################################################
    # helpers
    ################################################################################################

    # helper to enable processes / datasets only for a specific era
    def _match_era(
        *,
        run: int | set[int] | None = None,
        year: int | set[int] | None = None,
        postfix: str | set[int] | None = None,
        tag: str | set[str] | None = None,
        nano: int | set[int] | None = None,
        sync: bool = False,
    ) -> bool:
        return (
            (run is None or campaign.x.run in law.util.make_set(run)) and
            (year is None or campaign.x.year in law.util.make_set(year)) and
            (postfix is None or campaign.x.postfix in law.util.make_set(postfix)) and
            (tag is None or campaign.has_tag(tag, mode=any)) and
            (nano is None or campaign.x.version in law.util.make_set(nano)) and
            (sync is sync_mode)
        )

    def if_era(*, values: list[str | None] | None = None, **kwargs) -> list[str]:
        return list(filter(bool, values or [])) if _match_era(**kwargs) else []

    def if_not_era(*, values: list[str | None] | None = None, **kwargs) -> list[str]:
        return list(filter(bool, values or [])) if not _match_era(**kwargs) else []

    ################################################################################################
    # processes
    ################################################################################################
    # processes we are interested in
    process_names = [
        "data",
        "tt",
        "st",
        "qcd",
        "qcd_est",
    ]

    for process_name in process_names:
        # add the process
        if process_name == "qcd_est":
            proc = cfg.add_process(name="qcd_est", id=30002)
        else:
            proc = cfg.add_process(procs.get(process_name))

        # configuration of colors, labels, etc. can happen here
        if proc.is_mc:
            if proc.name == "qcd_est":
                proc.color1 = (244, 93, 244)
            elif proc.name == "tt":
                proc.color1 = (244, 182, 66)
            else:
                (244, 93, 66)

    # configure colors, labels, etc
    # from aj.config.styles import stylize_processes

    # stylize_processes(cfg)

    ################################################################################################
    # datasets
    ################################################################################################

    # add datasets we need to study
    dataset_names = [
        *if_era(
            year=2016,
            tag="HIPM",
            values=[
                "data_jetht_b",
                "data_jetht_c",
                "data_jetht_d",
                "data_jetht_e",
                "data_jetht_f",
            ],
        ),
        *if_era(
            year=2016,
            tag="notHIPM",
            values=[
                "data_jetht_f",
                "data_jetht_g",
                "data_jetht_h",
            ],
        ),
        *if_era(
            year=2017,
            values=[
                "data_jetht_c",
                "data_jetht_d",
                "data_jetht_e",
                "data_jetht_f",
            ],
        ),
        *if_era(
            year=2018,
            values=[
                "data_jetht_a",
                "data_jetht_b",
                "data_jetht_c",
                "data_jetht_d",
            ],
        ),
        # ttbar
        # "tt_sl_powheg",
        # "tt_dl_powheg",
        # single top
        # "st_tchannel_t_4f_powheg",
        # "st_tchannel_tbar_4f_powheg",
        # "st_twchannel_t_sl_powheg",
        # "st_twchannel_tbar_sl_powheg",
        # "st_twchannel_t_dl_powheg",
        # "st_twchannel_tbar_dl_powheg",
        # "st_twchannel_t_fh_powheg",
        # "st_twchannel_tbar_fh_powheg",
        # "st_schannel_t_lep_4f_amcatnlo",
        # "st_schannel_tbar_lep_4f_amcatnlo",
        # qcd datasets
        # "qcd_ht50to100_madgraph",
        # "qcd_ht100to200_madgraph",
        # "qcd_ht200to300_madgraph",
        "qcd_ht300to500_madgraph",
        "qcd_ht500to700_madgraph",
        "qcd_ht700to1000_madgraph",
        "qcd_ht1000to1500_madgraph",
        "qcd_ht1500to2000_madgraph",
        "qcd_ht2000toinf_madgraph",
        # signals
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        "tt_fh_mt166p5_powheg",
        "tt_fh_mt169p5_powheg",
        "tt_fh_mt171p5_powheg",
        "tt_fh_mt173p5_powheg",
        "tt_fh_mt175p5_powheg",
        "tt_fh_mt178p5_powheg",
    ]
    for dataset_name in dataset_names:
        # skip when in sync mode and not exiting
        if sync_mode and not campaign.has_dataset(dataset_name):
            continue

        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))
        if dataset.name.startswith("tt_"):
            dataset.add_tag({"has_top", "ttbar", "tt"})
        if dataset.name.startswith("st_"):
            dataset.add_tag({"has_top", "single_top", "st"})
        # apply an optional limit on the number of files
        if limit_dataset_files:
            for info in dataset.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)

        # apply synchronization settings
        if sync_mode:
            # only first file per
            for info in dataset.info.values():
                info.n_files = 1

    # verify that the root process of each dataset is part of any of the registered processes
    if not sync_mode:
        verify_config_processes(cfg, warn=True)

    ################################################################################################
    # task defaults and groups
    ################################################################################################

    # default objects
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "example"
    cfg.x.default_reducer = "cf_default"
    cfg.x.default_producer = "example"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "default_no_shifts"
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = ("njet", "jet1_pt")
    cfg.x.default_hist_producer = "all_weights"

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
        "default_Mt": [
            "All",
            "SignalOrBkgTrigger",
            "BTag20",
            "jet",
            "HT",
            "Rbb",
            "LeadingSix",
            "n5Chi2",
            "Mt",
        ],
        "default_Rbb": [
            "Rbb",
            "n5Chi2",
            "All",
            "SignalOrBkgTrigger",
            "BTag20",
            "jet",
            "HT",
        ],
        "default_LS": [
            "All",
            "SignalOrBkgTrigger",
            "BTag20",
            "jet",
            "HT",
            "Rbb",
            "LeadingSix",
            "n5Chi2",
        ],
        "default_bkg_n5Chi2": [
            "All",
            "n5Chi2",
            "SignalOrBkgTrigger",
            "BTag20",
            "jet",
            "HT",
        ],
        "default_bkg_n10Chi2": [
            "All",
            "n10Chi2",
            "SignalOrBkgTrigger",
            "BTag20",
            "jet",
            "HT",
        ],
        "default_bkg_chi2": [
            "All",
            "Chi2",
            "SignalOrBkgTrigger",
            "BTag20",
            "jet",
            "HT",
        ],
        "default_bkg": ["All", "SignalOrBkgTrigger", "BTag20", "jet", "HT"],
        "trig_eff_ht": ["All", "BaseTrigger", "SixJets", "BTag", "jet"],
        "ht7": ["All", "BaseTrigger", "SixJets", "BTag", "jet"],
        "trig_eff_ht2": ["All", "BaseTrigger", "BTag", "jet"],
        "trig_eff_pt": ["All", "BaseTrigger", "BTag", "HT"],
        "jet6_pt_5": ["All", "BaseTrigger", "BTag", "HT"],
        "trig_eff_bjet": ["All", "BaseTrigger", "jet", "HT"],
        "trig_eff_ht_pt": ["All", "BaseTrigger", "BTag"],
    }
    cfg.x.default_selector_steps = "default"

    # plotting overwrites
    # from hbt.config.styles import setup_plot_styles

    # setup_plot_styles(cfg)

    ################################################################################################
    # luminosity and normalization
    ################################################################################################

    # lumi values in 1/pb (= 1000/fb)
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=7
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3?rev=25
    # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis
    # difference pre-post VFP: https://cds.cern.ch/record/2854610/files/DP2023_006.pdf
    # Lumis for Run3 within the Twiki are outdated as stated here:
    # https://cms-talk.web.cern.ch/t/luminosity-in-run2023c/116859/2
    # Run3 Lumis can be calculated with brilcalc tool https://twiki.cern.ch/twiki/bin/view/CMS/BrilcalcQuickStart?rev=15
    # CClub computed this already: https://gitlab.cern.ch/cclubbtautau/AnalysisCore/-/issues/49
    if year == 2016 and campaign.has_tag("HIPM"):
        cfg.x.luminosity = Number(
            19_500,
            {
                "lumi_13TeV_2016": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            },
        )
    elif year == 2016 and campaign.has_tag("notHIPM"):
        cfg.x.luminosity = Number(
            16_800,
            {
                "lumi_13TeV_2016": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            },
        )
    elif year == 2017:
        cfg.x.luminosity = Number(
            36674,
            {
                "lumi_13TeV_2017": 0.02j,
                "lumi_13TeV_1718": 0.006j,
                "lumi_13TeV_correlated": 0.009j,
            },
        )
    elif year == 2018:
        cfg.x.luminosity = Number(
            59_830,
            {
                "lumi_13TeV_2017": 0.015j,
                "lumi_13TeV_1718": 0.002j,
                "lumi_13TeV_correlated": 0.02j,
            },
        )
    # minimum bias cross section in mb (milli) for creating PU weights, values from
    # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=52#Recommended_cross_section
    cfg.x.minbias_xs = Number(69.2, 0.046j)

    # jet settings
    # TODO: keep a single table somewhere that configures all settings: btag correlation, year
    #       dependence, usage in calibrator, etc
    ################################################################################################

    # common jec/jer settings configuration
    if run == 2:
        # https://cms-jerc.web.cern.ch/Recommendations/#run-2
        # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=204
        # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=109
        jec_campaign = f"Summer19UL{year2}{campaign.x.postfix}"
        jec_version = {2016: "V7", 2017: "V5", 2018: "V5"}[year]
        jer_campaign = (
            f"Summer{'20' if year == 2016 else '19'}UL{year2}{campaign.x.postfix}"
        )
        jer_version = "JR" + {2016: "V3", 2017: "V2", 2018: "V2"}[year]
        jet_type = "AK4PFchs"

    cfg.x.jec = DotDict.wrap(
        {
            "Jet": {
                "campaign": jec_campaign,
                "version": jec_version,
                "jet_type": jet_type,
                "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
                "levels_for_type1_met": ["L1FastJet"],
                "uncertainty_sources": list(
                    filter(
                        bool,
                        [
                            # "AbsoluteStat",
                            # "AbsoluteScale",
                            # "AbsoluteSample",
                            # "AbsoluteFlavMap",
                            # "AbsoluteMPFBias",
                            # "Fragmentation",
                            # "SinglePionECAL",
                            # "SinglePionHCAL",
                            # "FlavorQCD",
                            # "TimePtEta",
                            # "RelativeJEREC1",
                            # "RelativeJEREC2",
                            # "RelativeJERHF",
                            # "RelativePtBB",
                            # "RelativePtEC1",
                            # "RelativePtEC2",
                            # "RelativePtHF",
                            # "RelativeBal",
                            # "RelativeSample",
                            # "RelativeFSR",
                            # "RelativeStatFSR",
                            # "RelativeStatEC",
                            # "RelativeStatHF",
                            # "PileUpDataMC",
                            # "PileUpPtRef",
                            # "PileUpPtBB",
                            # "PileUpPtEC1",
                            # "PileUpPtEC2",
                            # "PileUpPtHF",
                            # "PileUpMuZero",
                            # "PileUpEnvelope",
                            # "SubTotalPileUp",
                            # "SubTotalRelative",
                            # "SubTotalPt",
                            # "SubTotalScale",
                            # "SubTotalAbsolute",
                            # "SubTotalMC",
                            "Total",
                            # "TotalNoFlavor",
                            # "TotalNoTime",
                            # "TotalNoFlavorNoTime",
                            # "FlavorZJet",
                            # "FlavorPhotonJet",
                            # "FlavorPureGluon",
                            # "FlavorPureQuark",
                            # "FlavorPureCharm",
                            # "FlavorPureBottom",
                            "CorrelationGroupMPFInSitu",
                            "CorrelationGroupIntercalibration",
                            "CorrelationGroupbJES",
                            "CorrelationGroupFlavor",
                            "CorrelationGroupUncorrelated",
                        ],
                    )
                ),
            },
        }
    )

    # JER
    cfg.x.jer = DotDict.wrap(
        {
            "Jet": {
                "campaign": jer_campaign,
                "version": jer_version,
                "jet_type": jet_type,
            },
        }
    )

    # updated jet id
    from columnflow.production.cms.jet import JetIdConfig

    cfg.x.jet_id = JetIdConfig(
        corrections={"AK4PUPPI_Tight": 2, "AK4PUPPI_TightLeptonVeto": 3}
    )
    cfg.x.fatjet_id = JetIdConfig(
        corrections={"AK8PUPPI_Tight": 2, "AK8PUPPI_TightLeptonVeto": 3}
    )

    # trigger sf corrector
    cfg.x.jet_trigger_corrector = "jetlegSFs"

    ################################################################################################
    # b tagging
    ################################################################################################
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
    cfg.x.btag_sf = (
        "deepJet_shape", cfg.x.btag_sf_jec_sources, "btagDeepFlavB")

    ################################################################################################
    # dataset / process specific methods
    ################################################################################################
    cfg.x.fitchi2cut = 10

    ################################################################################################
    # shifts
    ################################################################################################
    # register shifts
    cfg.add_shift(name="nominal", id=0)

    # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
    # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
    cfg.add_shift(name="tune_up", id=1, type="shape",
                  tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape",
                  tags={"disjoint_from_nominal"})

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

    # Trigger shifts
    cfg.add_shift(name="trig_up", id=120, type="shape")
    cfg.add_shift(name="trig_down", id=121, type="shape")
    add_shift_aliases(
        cfg,
        "trig",
        {
            "trig_weight": "trig_weight_{direction}",
        },
    )

    # Pile-up shifts
    cfg.add_shift(name="pu_weight_minbias_xs_up", id=150, type="shape")
    cfg.add_shift(name="pu_weight_minbias_xs_down", id=151, type="shape")
    add_shift_aliases(
        cfg,
        "pu_weight_minbias_xs",
        {
            "pu_weight": "pu_weight_minbias_xs_{direction}",
        },
    )

    ################################################################################################
    # external files
    ################################################################################################

    cfg.x.external_files = DotDict()
    json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-377439e8"
    year = 2017
    corr_postfix = ""

    # helper
    def add_external(name, value):
        if isinstance(value, dict):
            value = DotDict.wrap(value)
        cfg.x.external_files[name] = value

    # common files
    # (versions in the end are for hashing in cases where file contents changed but paths did not)
    add_external(
          # lumi files
        "lumi", {
            "golden": (
                "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collision"
                "s17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_"
                "Collisions17_GoldenJSON.txt",
                "v1",
            ),  # noqa
            "normtag": (
                "/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json",
                "v1",
            ),
        },
    )
    # pileup weight corrections
    add_external(
        "pu_sf", (
            f"{json_mirror}/POG/LUM/{year}{corr_postfix}_UL/puWeights.json.gz",
            "v1",
        ),

    )
    # muon scale factors
    add_external(
        "muon_sf", (f"{json_mirror}/POG/MUO/{year}_UL/muon_Z.json.gz", "v1"),
    )
    # jet energy correction
    add_external(
        "jet_jerc", (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1")
    )
    # btag scale factor
    add_external(
        "btag_sf_corr", (
            f"{json_mirror}/POG/BTV/{year}{corr_postfix}_UL/btagging.json.gz",
            "v1",
        ),

    )
    # electron scale factors
    add_external(
        "electron_sf", (
            f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz",
            "v1",
        ),
    )

    ################################################################################################
    # reductions
    ################################################################################################

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
                "Jet.{pt,eta,phi,mass,btagDeepFlavB,hadronFlavour}",
                "Bjet.*",
                "VetoJet.*",
                "LightJet.*",
                "JetsByBTag.*",
                # "EventJet.*",
                "Muon.{pt,eta,phi,mass,pfRelIso04_all}",
                "MET.{pt,phi,significance,covXX,covXY,covYY}",
                "PV.{npvs,npvsGood}",
                "FitJet.*",
                "FitChi2",
                "fitCombinationType",
                "reco_combination_type",
                "DeltaR",
                "GenPart.*",
                "MW1",
                "MW2",
                "Mt1",
                "Mt2",
                "chi2",
                "deltaRb",
                "HLT.{Mu50,PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2, PFHT380_SixPFJet32_DoublePFBTagCSV_2p2,PFHT380_SixPFJet32,IsoMu24,PFHT370,PFHT350,Physics,PFHT1050,PFHT890}",
                # columns added during selection
                "deterministic_seed",
                "process_id",
                "mc_weight",
                "cutflow.*",
                "pdf_weight",
                "trig_weight",
                "trig_weight_up",
                "trig_weight_down",
                "murmuf_weight",
                "pu_weight",
                "btag_weight",
                "combination_type",
                "R2b4q",
                "trig_ht",
                "gen_top_decay",
                "gen_top_decay.{eta,phi,pt,mass,genPartIdxMother,pdgId,status,statusFlags}",
            },
            "cf.MergeSelectionMasks": {
                "normalization_weight",
                "process_id",
                "category_ids",
                "cutflow.*",
            },
            "cf.UniteColumns": {
                "*_weight",
                "Jet.*",
                "combination_type",
                "ht",
                "Mt*",
                "MW*",
                "trig_bits",
            },
        },
    )

 ################################################################################################
 # weights
 ################################################################################################

    # configurations for all possible event weight columns as keys in an OrderedDict,
    # mapped to shift instances they depend on
    # (this info is used by weight producers)
    get_shifts = functools.partial(get_shifts_from_sources, cfg)

    cfg.x.event_weights = DotDict(
        {
            "normalization_weight": [],
            "btag_weight": [],
            # "trig_weight": [],
            # "trig_weight": get_shifts("trig"),
            # "muon_weight": get_shifts("mu"),
            "pdf_weight": get_shifts("pdf"),
            "murmuf_weight": get_shifts("murmuf"),
            "pu_weight": get_shifts("pu_weight_minbias_xs")
        },
    )
    # define per-dataset event weights
    cfg.x.shift_groups = {}

################################################################################################
    # external configs: channels, categories, met filters, triggers, variables
################################################################################################
    cfg.x.met_name = "MET"
    cfg.x.raw_met_name = "RawMET"

    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}_UL")
    cfg.x.trigger = {
        "tt_fh": [
            "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2",
            "PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2",
        ],
    }

    cfg.x.ref_trigger = {
        "tt_fh": ["PFHT350"],
    }

    cfg.x.bkg_trigger = {
        "tt_fh": ["PFHT380_SixPFJet32"],
    }

    # channels
    cfg.add_channel(name="mutau", id=1, label=r"$\mu\tau_{h}$")

    # add categories
    from alljets.config.categories import add_categories

    add_categories(cfg)

    # add variables
    from alljets.config.variables import add_variables

    add_variables(cfg)

    ################################################################################################
    # LFN settings
    ################################################################################################

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = False
    return cfg
