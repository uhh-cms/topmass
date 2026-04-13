# coding: utf-8

"""
Example inference model.
"""

from columnflow.inference import InferenceModel, ParameterType, inference_model, ParameterTransformation


@inference_model
def default_1D(self: InferenceModel) -> None:

    #
    # categories
    #

    self.add_category(
        "SR_top_mass",
        config_data={
            config_inst.name: self.category_config_spec(
                category="sig",
                variable="fit_Top1_mass_percentile",
                data_datasets=["data_jetht*"],
            )
            for config_inst in self.config_insts
        },
    )

    self.add_category(
        "SR_reco_W_avg",
        config_data={
            config_inst.name: self.category_config_spec(
                category="sig",
                variable="reco_W_mass_avg_percentile",
                data_datasets=["data_jetht*"],
            )
            for config_inst in self.config_insts
        },
    )

    self.add_category(
        "SR_reco_R_bq",
        config_data={
            config_inst.name: self.category_config_spec(
                category="sig",
                variable="reco_R_bq_percentile",
                data_datasets=["data_jetht*"],
            )
            for config_inst in self.config_insts
        },
    )
    #
    # processes
    #

    self.add_process(
        "TT",
        config_data={
            config_inst.name: self.process_config_spec(
                process="tt",
                mc_datasets=["tt_fh_powheg",
                             "tt_sl_powheg",
                             "tt_dl_powheg"],
            )

            for config_inst in self.config_insts
        },
        is_signal=True,
    )
    #
    # parameters
    #

    # groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # # # lumi
    # lumi = self.config_inst.x.luminosity
    # for unc_name in lumi.uncertainties:
    #     self.add_parameter(
    #         unc_name,
    #         type=ParameterType.rate_gauss,
    #         effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
    #         transformations=[ParameterTransformation.symmetrize],
    #     )

    # self.add_parameter(
    #     "jec_Total",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_Total",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # jet energy resolution
    # self.add_parameter(
    #     "jer",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jer",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # jet energy correction uncertainties

    # self.add_parameter(
    #     "jec_AbsoluteStat",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_AbsoluteStat",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_AbsoluteScale",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_AbsoluteScale",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_AbsoluteMPFBias",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_AbsoluteMPFBias",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_Fragmentation",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_Fragmentation",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_SinglePionECAL",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_SinglePionECAL",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_SinglePionHCAL",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_SinglePionHCAL",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeJEREC1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeJEREC1",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeJEREC2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeJEREC2",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeJERHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeJERHF",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativePtBB",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativePtBB",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativePtEC1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativePtEC1",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativePtEC2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativePtEC2",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativePtHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativePtHF",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeBal",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeBal",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeSample",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeSample",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeFSR",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeFSR",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeStatFSR",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeStatFSR",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeStatEC",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeStatEC",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_RelativeStatHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_RelativeStatHF",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpDataMC",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpDataMC",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpPtRef",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpPtRef",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpPtBB",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpPtBB",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpPtEC1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpPtEC1",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpPtEC2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpPtEC2",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpPtHF",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpPtHF",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpMuZero",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpMuZero",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PileUpEnvelope",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_PileUpEnvelope",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    self.add_parameter(
        "jec_BottomFlavour",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="jec_FlavorPureBottom",
            )
            for config_inst in self.config_insts
        },
    )
    self.add_parameter(
        "jec_GluonFlavour",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="jec_FlavorPureGluon",
            )
            for config_inst in self.config_insts
        },
    )
    self.add_parameter(
        "jec_QuarkFlavour",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="jec_FlavorPureQuark",
            )
            for config_inst in self.config_insts
        },
    )
    self.add_parameter(
        "jec_CharmFlavour",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="jec_FlavorPureCharm",
            )
            for config_inst in self.config_insts
        },
    )
    # self.add_parameter(
    #     "jec_ZJetFlavour",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_FlavorZJet",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # self.add_parameter(
    #     "jec_PhotonJetFlavour",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="jec_FlavorPhotonJet",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
# parton shower shifts(isr/fsr)
    self.add_parameter(
        "isr",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="isr",
            )
            for config_inst in self.config_insts
        },
    )
    self.add_parameter(
        "fsr",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="fsr",
            )
            for config_inst in self.config_insts
        },
    )
# Hdamp
    self.add_parameter(
        "hdamp",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="hdamp",
            )
            for config_inst in self.config_insts
        },
    )
    # pile-up weights
    self.add_parameter(
        "pu_weight",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="pu_weight_minbias_xs",
            )
            for config_inst in self.config_insts
        },
    )
    # pdf shift
    self.add_parameter(
        "pdf",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="pdf",
            )
            for config_inst in self.config_insts
        },
    )
    # trigger
    self.add_parameter(
        "trigger",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="trig",
            )
            for config_inst in self.config_insts
        },
    )
    # tune shift
    self.add_parameter(
        "tune",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="tune",
            )
            for config_inst in self.config_insts
        },
    )
    # tune shift

    # self.add_parameter(
    #     "tune_cr1",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="tune_cr1",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # tune shift

    # self.add_parameter(
    #     "tune_cr2",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="tune_cr2",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    # tune shift

    # self.add_parameter(
    #     "tune_rtt",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="tune_rtt",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )

    # self.add_parameter(
    #     "tune_erdON",
    #     process=["TT"],
    #     type=ParameterType.shape,
    #     transformations={ParameterTransformation.envelope,
    #                      ParameterTransformation.normalize},
    #     config_data={
    #         config_inst.name: self.parameter_config_spec(
    #             shift_source="tune_erdON",
    #         )
    #         for config_inst in self.config_insts
    #     },
    # )
    self.add_parameter(
        "mtop",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="mtop3",
            )
            for config_inst in self.config_insts
        },
    )
    # murmuf
    self.add_parameter(
        "murmuf",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="murmuf",
            )
            for config_inst in self.config_insts
        },
    )
#  # rate uncertainty
# self.add_parameter(
#     "rate",
#     process=["TT"],
#     type=ParameterType.rate_gauss,
#     effect=(0.9,1.1),
# )
# # a custom asymmetric uncertainty that is converted from rate to shape
# self.add_parameter(
#     "QCDscale_ttbar",
#     process="TT_FH",
#     type=ParameterType.shape,
#     transformations=[ParameterTransformation.effect_from_rate],
#     effect=(0.5, 1.1),
# )
