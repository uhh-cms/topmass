# coding: utf-8

"""
Configuration of the ttbar analysis.
"""

from __future__ import annotations

import functools
import os

import law
import order as od
import yaml
from columnflow.columnar_util import ColumnCollection, skip_column
from columnflow.config_util import (add_shift_aliases,
                                    get_root_processes_from_campaign,
                                    get_shifts_from_sources,
                                    verify_config_processes)
from columnflow.cms_util import CATInfo, CATSnapshot
from columnflow.util import DotDict
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
        # single top
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_powheg",
        "st_twchannel_tbar_powheg",
        "st_schannel_lep_4f_amcatnlo",
        "st_schannel_had_4f_amcatnlo",
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

    # ---------------------------------------------------------
    # DATASET GROUPS (for efficiency calculation)
    # ---------------------------------------------------------
    cfg.x.btag_wp_eff_groups = [
        ["tt_*", "st_*"],
        ["qcd_*"],
    ]

    # assign dataset tags based on these groups
    for dataset in cfg.datasets:
        group_matched = False
        for i, dataset_pattern in enumerate(cfg.x.btag_wp_eff_groups):
            if law.util.multi_match(dataset.name, dataset_pattern):
                if group_matched:
                    raise ValueError(
                        f"dataset '{dataset.name}' already has a btag WP group assigned!",
                    )
                group_matched = True
                dataset.add_tag(f"btag_wp_eff_group_{i}")
        if not group_matched and dataset.is_mc:
            raise ValueError(f"no btag_wp_eff_group_* assigned to dataset '{dataset.name}'")
        if group_matched and dataset.is_data:
            raise ValueError(f"must not assign btag_wp_eff_group_* to dataset '{dataset.name}'")

    ################################################################################################
    # task defaults and groups
    ################################################################################################

    # default objects
    cfg.x.default_calibrator = "default"
    cfg.x.default_selector = "default_trig_weight"
    cfg.x.default_reducer = "cf_default"
    cfg.x.default_producer = ["default", "kinFitMatch"]
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
        "default": [],
        "default_bkg": ["All", "SignalOrBkgTrigger", "BTag20", "jet", "HT"],
        "ht_trigger": ["All", "BaseTrigger", "SixJets", "BTag", "jet"],
        "trigjet6_pt": ["All", "BaseTrigger", "BTag", "HT"],
        "trig_eff_bjet": ["All", "BaseTrigger", "jet", "HT"],
        "trig_eff_ht_pt": ["All", "BaseTrigger", "BTag"],
    }
    cfg.x.default_selector_steps = "default"

    cfg.x.custom_style_config_groups = {
        "default": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 16,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "annotate_cfg": {
                "xy": (0.05, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 16,
            },
        },
        "default_rax10": {
            # "legend_cfg": {
            #     "ncols": 2,
            #     "fontsize": 16,
            #     "bbox_to_anchor": (0., 0., 1., 1.),
            # },
            # "ax_cfg": {
            #     "ylim": (-10, 10),
            # },
            "rax_cfg": {
                "ylim": (0.9, 1.1),
            },
            # "annotate_cfg": {
            #     "xy": (0.05, 0.95),
            #     "xycoords": "axes fraction",
            #     "fontsize": 16,
            # },
        },
        "shift_plots_mtop": {
            "ax_cfg": {
                "xlim": (50, 400),
                "ylabel": "Entries/BinWidth",
            },
        },
        "shift_plots_mwreco": {
            "ax_cfg": {
                "ylabel": "Entries/BinWidth",
            },
        },
        "shift_plots_rbq": {
            "ax_cfg": {
                "xlim": (0, 2),
                "ylabel": "Entries/BinWidth",
            },
        },
    }

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
        # Updated the lumi value for 2017 based on the latest brilcalc results for the relevant trigger, see
        # /afs/cern.ch/user/l/lgriesin/eos/BrilCal/2017/ValuesFb/HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2.csv
        cfg.x.luminosity = Number(
            37_186,
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

    # Met names for Run2, mainly needed for alias in JEC and JER shifts
    cfg.x.met_name = "MET"
    cfg.x.raw_met_name = "RawMET"

    # common jec/jer settings configuration
    if run == 2:
        # https://cms-jerc.web.cern.ch/Recommendations/#run-2
        # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=204
        # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=109
        # See https://twiki.cern.ch/twiki/bin/view/CMS/JECUncertaintySources#Main_uncertainties_2017_94X
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
                "data_per_era": True,
                "jet_type": jet_type,
                "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
                "levels_for_type1_met": ["L1FastJet"],
                "uncertainty_sources": list(
                    filter(
                        bool,
                        [
                            "AbsoluteStat",
                            "AbsoluteScale",
                            # "AbsoluteSample",
                            # "AbsoluteFlavMap",
                            "AbsoluteMPFBias",
                            "Fragmentation",
                            "SinglePionECAL",
                            "SinglePionHCAL",
                            # "FlavorQCD",
                            "TimePtEta",
                            "RelativeJEREC1",
                            "RelativeJEREC2",
                            "RelativeJERHF",
                            "RelativePtBB",
                            "RelativePtEC1",
                            "RelativePtEC2",
                            "RelativePtHF",
                            "RelativeBal",
                            "RelativeSample",
                            "RelativeFSR",
                            "RelativeStatFSR",
                            "RelativeStatEC",
                            "RelativeStatHF",
                            "PileUpDataMC",
                            "PileUpPtRef",
                            "PileUpPtBB",
                            "PileUpPtEC1",
                            "PileUpPtEC2",
                            "PileUpPtHF",
                            "PileUpMuZero",
                            "PileUpEnvelope",
                            # "SubTotalPileUp",
                            # "SubTotalRelative",
                            # "SubTotalPt",
                            # "SubTotalScale",
                            # "SubTotalAbsolute",
                            # "SubTotalMC",
                            # "Total",
                            # "TotalNoFlavor",
                            # "TotalNoTime",
                            # "TotalNoFlavorNoTime",
                            # "FlavorZJet",
                            # "FlavorPhotonJet",
                            "FlavorPureGluon",
                            "FlavorPureQuark",
                            "FlavorPureCharm",
                            "FlavorPureBottom",
                            # "CorrelationGroupMPFInSitu",
                            # "CorrelationGroupIntercalibration",
                            # "CorrelationGroupbJES",
                            # "CorrelationGroupFlavor",
                            # "CorrelationGroupUncorrelated",
                        ],
                    ),
                ),
            },
        },
    )

    # JER
    cfg.x.jer = DotDict.wrap(
        {
            "Jet": {
                "campaign": jer_campaign,
                "version": jer_version,
                "jet_type": jet_type,
            },
        },
    )

    ################################################################################################
    # b tagging
    ################################################################################################
    # b-tag working points
    btag_key = f"{year}{campaign.x.postfix}"
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
    # https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18?rev=18
    cfg.x.btag_working_points = DotDict.wrap({
        "deepjet": {
            "loose": {"2016APV": 0.0508, "2016": 0.0480, "2017": 0.0532, "2018": 0.0490}[btag_key],
            "medium": {"2016APV": 0.2598, "2016": 0.2489, "2017": 0.3040, "2018": 0.2783}[btag_key],
            "tight": {"2016APV": 0.6502, "2016": 0.6377, "2017": 0.7476, "2018": 0.7100}[btag_key],
        },
        "deepcsv": {
            "loose": {"2016APV": 0.2027, "2016": 0.1918, "2017": 0.1355, "2018": 0.1208}[btag_key],
            "medium": {"2016APV": 0.6001, "2016": 0.5847, "2017": 0.4506, "2018": 0.4168}[btag_key],
            "tight": {"2016APV": 0.8819, "2016": 0.8767, "2017": 0.7738, "2018": 0.7665}[btag_key],
        },
    })
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

    # https://btv-wiki.docs.cern.ch/PerformanceCalibration/SFUncertaintiesAndCorrelations/#ak4-working-point-based-sfs-fixedwp-sfs
    # name of the btag_sf correction set and jec uncertainties to propagate through
    # For the b/c jets consider deepJets_comb (QCD + ttbar enriched)
    # For the light jets SF, should use deepJet_incl

    cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources, "btagDeepFlavB")
    # ---------------------------------------------------------
    # BTag WP COUNT CONFIG
    # ---------------------------------------------------------
    from columnflow.selection.cms.btag import BTagWPCountConfig

    cfg.x.btag_wp_count_config = BTagWPCountConfig(
        jet_name="Jet",
        btag_column="btagDeepFlavB",
        btag_wps={"tight": cfg.x.btag_working_points.deepjet.tight},
        pt_edges=(20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000),
        abs_eta_edges=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5),
    )

    # ---------------------------------------------------------
    # BTag WP SCALE FACTOR CONFIG
    # ---------------------------------------------------------
    from columnflow.production.cms.btag import BTagWPSFConfig

    def dataset_groups(dataset_inst: od.Dataset) -> list[od.Dataset]:
        for group_index in range(len(cfg.x.btag_wp_eff_groups)):
            group_tag = f"btag_wp_eff_group_{group_index}"
            if dataset_inst.has_tag(group_tag):
                return [
                    _dataset_inst
                    for _dataset_inst in cfg.datasets
                    if _dataset_inst.has_tag(group_tag)
                ]
        raise NotImplementedError(f"btag WP efficiency group not implemented for dataset {dataset_inst.name}")

    cfg.x.btag_wp_sf_config = BTagWPSFConfig(
        jet_name="Jet",
        btag_column="btagDeepFlavB",
        correction_set="deepJet_merged",
        btag_wps={"tight": cfg.x.btag_working_points.deepjet.tight},
        dataset_groups=dataset_groups,
        pt_edges=(20, 30, 50, 70, 100, 140, 200, 300, 10_000),
        abs_eta_edges=(0.0, 0.4, 0.8, 1.2, 1.6, 2.5),
        wp_merging={},
    )

    ################################################################################################
    # dataset / process specific methods
    ################################################################################################
    cfg.x.fitchi2cut = 6.3
    cfg.x.fitpgofcut = 0.1
    cfg.x.trigger_sf_variable = "trigjet6_pt"

    # top pt reweighting
    # https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting?rev=31

    # theory-based method preferred
    from columnflow.production.cms.top_pt_weight import TopPtWeightFromTheoryConfig
    cfg.x.top_pt_weight = TopPtWeightFromTheoryConfig(params={
        "a": 0.103,
        "b": -0.0118,
        "c": -0.000134,
        "d": 0.973,
    })

    # data-based method preferred
    # from columnflow.production.cms.top_pt_weight import TopPtWeightFromDataConfig
    # cfg.x.top_pt_weight = TopPtWeightFromDataConfig(
    #     params={
    #         "a": 0.0615,
    #         "a_up": 0.0615 * 1.5,
    #         "a_down": 0.0615 * 0.5,
    #         "b": -0.0005,
    #         "b_up": -0.0005 * 1.5,
    #         "b_down": -0.0005 * 0.5,
    #     },
    #     pt_max=500.0,
    # )

    ################################################################################################
    # shifts
    ################################################################################################
    # register shifts
    cfg.add_shift(name="nominal", id=0)

    # top mass shifts of 1GeV
    cfg.add_shift(name="mtop1_up", id=5, type="shape", tags={"disjoint_from_nominal", "mtop1"})
    cfg.add_shift(name="mtop1_down", id=6, type="shape", tags={"disjoint_from_nominal", "mtop1"})

    # top mass shifts of 3 GeV
    cfg.add_shift(name="mtop3_up", id=12, type="shape", tags={"disjoint_from_nominal", "mtop3"})
    cfg.add_shift(name="mtop3_down", id=13, type="shape", tags={"disjoint_from_nominal", "mtop3"})

    # top mass shifts of 6 GeV
    cfg.add_shift(name="mtop6_up", id=14, type="shape", tags={"disjoint_from_nominal", "mtop6"})
    cfg.add_shift(name="mtop6_down", id=15, type="shape", tags={"disjoint_from_nominal", "mtop6"})

    # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
    # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal", "tune"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal", "tune"})
    add_shift_aliases(cfg, "tune", {"tune": "tune_{direction}"})

    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal", "hdamp"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal", "hdamp"})

    # fake jet energy correction shift, with aliases flaged as "selection_dependent", i.e. the aliases
    # affect columns that might change the output of the event selection
    # load jec sources
    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]
    for jec_source in cfg.x.jec.Jet.uncertainty_sources:
        idx = all_jec_sources.index(jec_source)
        cfg.add_shift(
            name=f"jec_{jec_source}_up",
            id=5000 + 2 * idx,
            type="shape",
            tags={"jec"},
            aux={"jec_source": jec_source},
        )
        cfg.add_shift(
            name=f"jec_{jec_source}_down",
            id=5001 + 2 * idx,
            type="shape",
            tags={"jec"},
            aux={"jec_source": jec_source},
        )
        add_shift_aliases(
            cfg,
            f"jec_{jec_source}",
            {
                "Jet.pt": "Jet.pt_{name}",
                "Jet.mass": "Jet.mass_{name}",
                f"{cfg.x.met_name}.pt": f"{cfg.x.met_name}.pt_{{name}}",
                f"{cfg.x.met_name}.phi": f"{cfg.x.met_name}.phi_{{name}}",
            },
        )
        # TODO: check the JEC de/correlation across years and the interplay with btag weights
        if ("" if jec_source == "Total" else jec_source) in cfg.x.btag_sf_jec_sources:
            add_shift_aliases(
                cfg,
                f"jec_{jec_source}",
                {
                    "normalized_btag_deepjet_weight": "normalized_btag_deepjet_weight_{name}",
                    "normalized_njet_btag_deepjet_weight": "normalized_njet_btag_deepjet_weight_{name}",
                    "normalized_btag_pnet_weight": "normalized_btag_pnet_weight_{name}",
                    "normalized_njet_btag_pnet_weight": "normalized_njet_btag_pnet_weight_{name}",
                },
            )

    # JER shift
    cfg.add_shift(name="jer_up", id=6000, type="shape", tags={"jer"})
    cfg.add_shift(name="jer_down", id=6001, type="shape", tags={"jer"})
    add_shift_aliases(
        cfg,
        "jer",
        {
            "Jet.pt": "Jet.pt_{name}",
            "Jet.mass": "Jet.mass_{name}",
            f"{cfg.x.met_name}.pt": f"{cfg.x.met_name}.pt_{{name}}",
            f"{cfg.x.met_name}.phi": f"{cfg.x.met_name}.phi_{{name}}",
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

    # Pdf shifts (up/down variations from CF)
    cfg.add_shift(name="pdf_up", id=130, type="shape", tags="pdf")
    cfg.add_shift(name="pdf_down", id=131, type="shape", tags="pdf")
    add_shift_aliases(
        cfg,
        "pdf",
        {
            "pdf_weight": "pdf_weight_{direction}",
            "normalized_pdf_weight": "normalized_pdf_weight_{direction}",
        },
    )

    # Trigger shifts
    cfg.add_shift(name="trig_up", id=120, type="shape", tags="trig")
    cfg.add_shift(name="trig_down", id=121, type="shape", tags="trig")
    add_shift_aliases(
        cfg,
        "trig",
        {
            "trig_weight": "trig_weight_{direction}",
        },
    )

    # Pile-up shifts
    cfg.add_shift(name="pu_weight_minbias_xs_up", id=150, type="shape", tags="pu_weight")
    cfg.add_shift(name="pu_weight_minbias_xs_down", id=151, type="shape", tags="pu_weight")
    add_shift_aliases(
        cfg,
        "pu_weight_minbias_xs",
        {
            "pu_weight": "pu_weight_minbias_xs_{direction}",
        },
    )

    # FSR shifts
    cfg.add_shift(name="fsr_up", id=152, type="shape", tags="fsr")
    cfg.add_shift(name="fsr_down", id=153, type="shape", tags="fsr")
    add_shift_aliases(
        cfg,
        "fsr",
        {
            "fsr_weight": "fsr_weight_{direction}",
        },
    )

    # ISR shifts
    cfg.add_shift(name="isr_up", id=154, type="shape", tags="isr")
    cfg.add_shift(name="isr_down", id=155, type="shape", tags="isr")
    add_shift_aliases(
        cfg,
        "isr",
        {
            "isr_weight": "isr_weight_{direction}",
        },
    )

    # Top pt reweighting shifts
    cfg.add_shift(name="top_pt_up", id=156, type="shape", tags="top_pt")
    cfg.add_shift(name="top_pt_down", id=157, type="shape", tags="top_pt")
    add_shift_aliases(
        cfg,
        "top_pt",
        {
            "top_pt_weight": "top_pt_weight_{direction}",
        },
    )

    # PDF shifts based on alpha_s variations
    cfg.add_shift(name="alphas_up", id=158, type="shape", tags="alphas")
    cfg.add_shift(name="alphas_down", id=159, type="shape", tags="alphas")
    add_shift_aliases(
        cfg,
        "alphas",
        {
            "pdf_weight": "pdf_alphas_weight_{direction}",
        },
    )

    # PDF shifts based on hessian variations, up to 100 variations
    for i in range(100):
        idx = i + 1
        name = f"hessian_{idx:03d}"
        cfg.add_shift(name=f"{name}_up", id=2000 + 2 * i, type="shape", tags="hessian")
        cfg.add_shift(name=f"{name}_down", id=2001 + 2 * i, type="shape", tags="hessian")
        add_shift_aliases(
            cfg,
            name,
            {
                "pdf_weight": f"pdf_hessian_{idx:03d}_weight_{{direction}}",
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

    if run == 2:
        cat_info = CATInfo(
            run=2,
            era=f"{year}{corr_postfix}-UL",
            vnano=9,
            snapshot=CATSnapshot(btv="latest", egm="latest", jme="latest", lum="latest", muo="latest", tau="latest"),
        )

    # common files
    # (versions in the end are for hashing in cases where file contents changed but paths did not)
    # lumi files
    add_external("lumi", {
        "golden": {
            2016: ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt", "v1"),  # noqa: E501
            2017: ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa: E501
            2018: ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt", "v1"),  # noqa: E501
        }[year],
        "normtag": {
            2016: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json", "v1"),
            2017: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json", "v1"),
            2018: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json", "v1"),
        }[year],
    })
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
        "jet_jerc", (f"{json_mirror}/POG/JME/{year}{corr_postfix}_UL/jet_jerc.json.gz", "v1"),
    )
    # WP based btag SF
    add_external(
        "btag_wp_sf_corr",
        (f"/afs/cern.ch/user/l/lgriesin/public/mTop/BTV_files/deepJet_{year}{corr_postfix}_merged.json.gz", "v1"),
    )

    # electron scale factors
    add_external(
        "electron_sf", (
            f"{json_mirror}/POG/EGM/{year}{corr_postfix}_UL/electron.json.gz",
            "v1",
        ),
    )
    # jet veto map
    add_external("jet_veto_map", (cat_info.get_file("jme", "jetvetomaps.json.gz"), "v2"))
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
                "Jet.{pt,eta,phi,mass,btagDeepFlavB,partonFlavour,hadronFlavour,veto_map_mask}",
                "TrigJets.{pt,eta,phi,mass,btagDeepFlavB,partonFlavour,hadronFlavour,veto_map_mask}",
                "SelectedJets.{pt,eta,phi,mass,btagDeepFlavB,partonFlavour,hadronFlavour,jetId,puId,veto_map_mask}",
                "KinFitJets.{pt,eta,phi,mass,btagDeepFlavB,partonFlavour,hadronFlavour,jetId,puId,veto_map_mask}",
                "Electron.{pt,eta,cutBased}",
                "Muon.{pt,eta,looseId,pfIsoId}",
                "PV.{npvs,npvsGood}",
                "GenPart.*",
                (
                    "HLT.{Mu50,PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2,"
                    "PFHT380_SixPFJet32_DoublePFBTagCSV_2p2,PFHT380_SixPFJet32,"
                    "IsoMu24,PFHT370,PFHT350,Physics,PFHT1050,PFHT890}"
                ),
                # columns added during selection
                "deterministic_seed",
                "process_id",
                "mc_weight",
                "pdf_weight",
                "trig_weight",
                "trig_weight_up",
                "trig_weight_down",
                "murmuf_weight",
                "pu_weight",
                "btag_weight",
                "trig_ht",
                "gen_top",
                "gen_top.{eta,phi,pt,mass,genPartIdxMother,pdgId,status,statusFlags}",
                ColumnCollection.ALL_FROM_SELECTOR,
                skip_column("pdf_weights_alphas*"),
                skip_column("pdf_weights_hessian*"),
                skip_column("cutflow.*"),
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
                "ht",
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
            "trig_weight": get_shifts("trig"),
            "pdf_weight": get_shifts("pdf", "alphas", "hessian_*"),
            "murmuf_weight": get_shifts("murmuf"),
            "pu_weight": get_shifts("pu_weight_minbias_xs"),
            "fsr_weight": get_shifts("fsr"),
            "isr_weight": get_shifts("isr"),
        },
    )

    # # define per-dataset event weights
    for dataset in cfg.datasets:
        if dataset.has_tag("ttbar"):
            dataset.x.event_weights = {"top_pt_weight": get_shifts("top_pt")}

    # define per-dataset event weights
    cfg.x.shift_groups = {}

    ################################################################################################
    # external configs: channels, categories, met filters, triggers, variables
    ################################################################################################

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
