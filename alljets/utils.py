# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations

__all__ = []

from columnflow.columnar_util import (  # noqa: F401
    IF_DATA, IF_MC, IF_DATASET_HAS_TAG, EMPTY_FLOAT, ArrayFunction, deferred_column,
    ak_concatenate_safe,
)
from columnflow.util import maybe_import
from columnflow.types import Any


np = maybe_import("numpy")
ak = maybe_import("awkward")


@deferred_column
def IF_RUN_2(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 2:
        return self.get()
    return None


@deferred_column
def IF_RUN_2_2016(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 2 and func.config_inst.campaign.x.year == 2016:
        return self.get()
    return None


@deferred_column
def IF_RUN_2_2017(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 2 and func.config_inst.campaign.x.year == 2017:
        return self.get()
    return None


@deferred_column
def IF_RUN_2_2018(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    if func.config_inst.campaign.x.run == 2 and func.config_inst.campaign.x.year == 2018:
        return self.get()
    return None
