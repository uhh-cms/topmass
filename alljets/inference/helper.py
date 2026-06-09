# coding: utf-8

"""
Helper functions for inference models.
"""

from __future__ import annotations


from columnflow.inference import InferenceModel, ParameterType, ParameterTransformation


def add_processes(im: InferenceModel) -> None:
    im.add_process(
        "TT",
        config_data={
            config_inst.name: im.process_config_spec(
                process="tt",
                mc_datasets=["tt_fh_powheg",
                             "tt_sl_powheg",
                             "tt_dl_powheg"],
            )
            for config_inst in im.config_insts
        },
        is_signal=True,
    )
    im.add_process(
        "BKG",
        config_data={
            config_inst.name: im.process_config_spec(
                process="qcd_est",
                # mc_datasets=["data*"],
            )
            for config_inst in im.config_insts
        },
        is_signal=False,
        is_dynamic=True,
    )


def add_parameters(im: InferenceModel) -> None:
    # groups
    im.add_parameter_group("experimental")
    im.add_parameter_group("modelling")

    im.add_parameter(
        "mtop",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: im.parameter_config_spec(
                shift_source="mtop1",
            )
            for config_inst in im.config_insts
        },
    )

    experimental = {
        "CMS_res_j_13TeV": "jer",

        # JES flavor
        "CMS_scale_j_FlavorPureBottom": "jec_FlavorPureBottom",
        "CMS_scale_j_FlavorPureGluon": "jec_FlavorPureGluon",
        "CMS_scale_j_FlavorPureCharm": "jec_FlavorPureCharm",
        "CMS_scale_j_FlavorPureQuark": "jec_FlavorPureQuark",

        # JES absolute
        "CMS_scale_j_AbsoluteStat": "jec_AbsoluteStat",
        "CMS_scale_j_AbsoluteScale": "jec_AbsoluteScale",
        "CMS_scale_j_AbsoluteMPFBias": "jec_AbsoluteMPFBias",

        # JES modelling
        "CMS_scale_j_Fragmentation": "jec_Fragmentation",
        "CMS_scale_j_SinglePionECAL": "jec_SinglePionECAL",
        "CMS_scale_j_SinglePionHCAL": "jec_SinglePionHCAL",

        # Relative
        "CMS_scale_j_RelativeJEREC1": "jec_RelativeJEREC1",
        "CMS_scale_j_RelativeJEREC2": "jec_RelativeJEREC2",
        "CMS_scale_j_RelativeJERHF": "jec_RelativeJERHF",

        "CMS_scale_j_RelativePtBB": "jec_RelativePtBB",
        "CMS_scale_j_RelativePtEC1": "jec_RelativePtEC1",
        "CMS_scale_j_RelativePtEC2": "jec_RelativePtEC2",
        "CMS_scale_j_RelativePtHF": "jec_RelativePtHF",

        "CMS_scale_j_RelativeBal": "jec_RelativeBal",
        "CMS_scale_j_RelativeSample": "jec_RelativeSample",

        "CMS_scale_j_RelativeFSR": "jec_RelativeFSR",
        "CMS_scale_j_RelativeStatFSR": "jec_RelativeStatFSR",
        "CMS_scale_j_RelativeStatEC": "jec_RelativeStatEC",
        "CMS_scale_j_RelativeStatHF": "jec_RelativeStatHF",

        # Pileup
        "CMS_scale_j_PileUpDataMC": "jec_PileUpDataMC",
        "CMS_scale_j_PileUpPtRef": "jec_PileUpPtRef",
        "CMS_scale_j_PileUpPtBB": "jec_PileUpPtBB",
        "CMS_scale_j_PileUpPtEC1": "jec_PileUpPtEC1",
        "CMS_scale_j_PileUpPtEC2": "jec_PileUpPtEC2",
        "CMS_scale_j_PileUpPtHF": "jec_PileUpPtHF",

        "CMS_pileup": "pu_weight_minbias_xs",
        "CMS_trig_htsixjets2btag": "trig",

        "CMS_btag_fixedWP_bc_correlated": "btag_heavy_cor",
        "CMS_btag_fixedWP_bc_uncorrelated": "btag_heavy_uncor",

        "CMS_btag_fixedWP_light_correlated": "btag_light_cor",
        "CMS_btag_fixedWP_light_uncorrelated": "btag_light_uncor",

    }

    modelling = {
        "ps_hdamp": "hdamp",
        "ps_hdamp_dctr": "hdamp_dctr",
        "pdf_alphas": "alphas",
        "underlying_event": "tune",
        "QCD_scale_ttbar": "murmuf",
        "top_pt_reweighting": "top_pt",
        "fragmentation_dctr": "rb_dctr",
    }

    modelling_envelope = {
        "ps_CR1": "tune_cr1",
        "ps_CR2": "tune_cr2",
        "ps_ERD": "tune_erdON",
        "ps_Recoil": "tune_rtt",
    }

    splittings = ("G2GG", "G2QQ", "Q2QG", "X2XG")
    for var in [f"{a}_{b}_{c}" for a in ["isr", "fsr"] for b in splittings for c in ["muR", "cNS"]]:
        modelling["ps_" + var] = var
    for i in range(100):
        modelling[f"pdf_{i:02}"] = f"hessian_{i + 1:03d}"

    def add_source(nuisance_name, shift_name, group, transformations=()):
        im.add_parameter(
            nuisance_name,
            process=["TT"],
            type=ParameterType.shape,
            transformations=transformations,
            config_data={
                config_inst.name: im.parameter_config_spec(
                    shift_source=shift_name,
                )
                for config_inst in im.config_insts
            },
            group=group,
        )

    for nuis_name, shift_name in experimental.items():
        add_source(nuis_name, shift_name, "experimental")

    for nuis_name, shift_name in modelling.items():
        add_source(nuis_name, shift_name, "modelling")
    for nuis_name, shift_name in modelling_envelope.items():
        add_source(nuis_name, shift_name, "modelling", (ParameterTransformation.envelope,))
