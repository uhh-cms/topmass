# coding: utf-8

"""
Datacard on the fine grid of mtfit, m_W,avg and R_bq, input of alljets/scripts/optimize_binning.py.
Create it for one config at a time (e.g. --configs 2017_v9) with the unrolling hist hook.
"""

from columnflow.inference import InferenceModel, FlowStrategy, inference_model

from alljets.config.binning_tree import FINE_AXES
from alljets.inference.helper import add_processes, add_parameters


@inference_model
def fine_3D(self: InferenceModel) -> None:
    self.add_category(
        "SR_fine_3D",
        config_data={
            config_inst.name: self.category_config_spec(
                category="sig",
                variable="-".join(f"{axis}_fine" for axis in FINE_AXES),
                data_datasets=["data_jetht*"],
            )
            for config_inst in self.config_insts
        },
        mc_stats=[100, 1, 1],
        flow_strategy=FlowStrategy.remove,
    )
    add_processes(self)
    add_parameters(self)
