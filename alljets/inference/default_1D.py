# coding: utf-8

"""
Example inference model.
"""

from columnflow.inference import InferenceModel, ParameterType, inference_model, ParameterTransformation, FlowStrategy
from alljets.inference.helper import add_processes, add_parameters

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
        mc_stats=[100, 1, 1],
        flow_strategy=FlowStrategy.remove,
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
        mc_stats=[100, 1, 1],
        flow_strategy=FlowStrategy.remove,
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
        mc_stats=[100, 1, 1],
        flow_strategy=FlowStrategy.remove,
    )
    #
    # processes
    #
    add_processes(self)
    #
    # parameters
    #
    add_parameters(self)
