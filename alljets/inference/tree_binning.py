# coding: utf-8

"""
Measurement in the tree binnings of mtfit x m_W,avg (2D) and mtfit x m_W,avg x R_bq (3D) found by
alljets/scripts/optimize_binning.py.
"""

from columnflow.inference import InferenceModel, FlowStrategy, inference_model

from alljets.config.binning_tree import load_tree
from alljets.inference.helper import add_processes, add_parameters


@inference_model
def tree_binning(self: InferenceModel) -> None:
    for tree_name in ("2D", "3D"):
        if load_tree(tree_name) is None:
            continue
        self.add_category(
            f"SR_tree_{tree_name}",
            config_data={
                config_inst.name: self.category_config_spec(
                    category="sig",
                    variable=f"tree_bin_{tree_name}",
                    data_datasets=["data_jetht*"],
                )
                for config_inst in self.config_insts
            },
            mc_stats=[100, 1, 1],
            flow_strategy=FlowStrategy.remove,
        )
    add_processes(self)
    add_parameters(self)
