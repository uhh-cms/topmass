# coding: utf-8

"""
Example inference model.
"""

from columnflow.inference import (ParameterTransformation, ParameterType,
                                  inference_model, InferenceModel)


@inference_model
def example(self: InferenceModel) -> None:

    #
    # categories
    #

    self.add_category(
        "fit_conv_big_top_mass",
        config_data={
            config_inst.name: self.category_config_spec(
                category="fit_conv_big",
                variable="fit_Top1_mass",
                data_datasets=["data_jetht*"],
            )
            for config_inst in self.config_insts
        },
    )

    self.add_category(
        "fit_conv_big_reco_W1",
        config_data={
            config_inst.name: self.category_config_spec(
                category="fit_conv_big",
                variable="reco_W1_mass",
                data_datasets=["data_jetht*"],
            )
            for config_inst in self.config_insts
        },
    )

    self.add_category(
        "fit_conv_big_reco_W2",
        config_data={
            config_inst.name: self.category_config_spec(
                category="fit_conv_big",
                variable="reco_W2_mass",
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

    # # lumi
    # lumi = self.config_inst.x.luminosity
    # for unc_name in lumi.uncertainties:
    #     self.add_parameter(
    #         unc_name,
    #         type=ParameterType.rate_gauss,
    #         effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
    #         transformations=[ParameterTransformation.symmetrize],
    #     )

    # jet energy correction uncertainty
    self.add_parameter(
        "jec",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="jec_Total",
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
    self.add_parameter(
        "mtop",
        process=["TT"],
        type=ParameterType.shape,
        config_data={
            config_inst.name: self.parameter_config_spec(
                shift_source="mtop",
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
    # # a custom asymmetric uncertainty that is converted from rate to shape
    # self.add_parameter(
    #     "QCDscale_ttbar",
    #     process="TT_FH",
    #     type=ParameterType.shape,
    #     transformations=[ParameterTransformation.effect_from_rate],
    #     effect=(0.5, 1.1),
    # )


@inference_model
def example_no_shapes(self):
    # same initialization as "example" above
    example.init_func.__get__(self, self.__class__)()

    #
    # remove all shape parameters
    #

    for category_name, process_name, parameter in self.iter_parameters():
        if parameter.type.is_shape or any(trafo.from_shape for trafo in parameter.transformations):
            self.remove_parameter(
                parameter.name, process=process_name, category=category_name)
