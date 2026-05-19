# coding: utf-8

"""
Helper functions for inference models.
"""

from __future__ import annotations

import re
import abc
import functools
import dataclasses

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
                #mc_datasets=["data*"],
            )
            for config_inst in im.config_insts
        },
        is_signal=False,
        is_dynamic= True,
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
        "CMS_scale_j_FlavorPureBottom": "jec_FlavorPureBottom",
        "CMS_scale_j_FlavorPureGluon": "jec_FlavorPureGluon",
        "CMS_scale_j_FlavorPureCharm": "jec_FlavorPureCharm",
        "CMS_scale_j_FlavorPureQuark": "jec_FlavorPureQuark",
        "CMS_pileup": "pu_weight_minbias_xs",
        "CMS_trig_htsixjets2btag": "trig"
    }

    modelling = {
        "ps_isr": "isr",
        "ps_fsr": "fsr",
        "ps_hdamp": "hdamp",
        "pdf_alphas": "pdf",
        "underlying_event": "tune",
        "ps_CR1": "tune_cr1",
        "QCD_scale_ttbar": "murmuf"
    }


    def add_source(nuisance_name, shift_name, group):
        im.add_parameter(
            nuisance_name,
            process=["TT"],
            type=ParameterType.shape,
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
    # # # lumi
    # lumi = im.config_inst.x.luminosity
    # for unc_name in lumi.uncertainties:
    #     im.add_parameter(
    #         unc_name,
    #         type=ParameterType.rate_gauss,
    #         effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
    #         transformations=[ParameterTransformation.symmetrize],
    #     )

    # im.add_parameter(
    #     "jec_Total",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_Total",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # jet energy correction uncertainties

    # im.add_parameter(
    #     "jec_AbsoluteStat",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_AbsoluteStat",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_AbsoluteScale",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_AbsoluteScale",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_AbsoluteMPFBias",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_AbsoluteMPFBias",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_Fragmentation",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_Fragmentation",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_SinglePionECAL",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_SinglePionECAL",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_SinglePionHCAL",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_SinglePionHCAL",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeJEREC1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeJEREC1",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeJEREC2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeJEREC2",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeJERHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeJERHF",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativePtBB",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativePtBB",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativePtEC1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativePtEC1",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativePtEC2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativePtEC2",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativePtHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativePtHF",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeBal",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeBal",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeSample",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeSample",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeFSR",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeFSR",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeStatFSR",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeStatFSR",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeStatEC",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeStatEC",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_RelativeStatHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_RelativeStatHF",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpDataMC",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpDataMC",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpPtRef",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpPtRef",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpPtBB",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpPtBB",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpPtEC1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpPtEC1",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpPtEC2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpPtEC2",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpPtHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpPtHF",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpMuZero",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpMuZero",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # im.add_parameter(
    #     "jec_PileUpEnvelope",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="jec_PileUpEnvelope",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )




    # tune shift

    # im.add_parameter(
    #     "tune_cr2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="tune_cr2",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )
    # tune shift

    # im.add_parameter(
    #     "tune_rtt",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="tune_rtt",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )

    # im.add_parameter(
    #     "tune_erdON",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: im.parameter_config_spec(
    #             shift_source="tune_erdON",
    #         )
    #         for config_inst in im.config_insts
    #     },
    # )

