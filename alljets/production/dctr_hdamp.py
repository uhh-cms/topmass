# coding: utf-8

"""
Producers for storing parton shower weights.
"""

from __future__ import annotations
import law

from columnflow.util import maybe_import
from columnflow.production import producer, Producer
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


@producer(
    mc_only=True,
)
def dctr_hdamp(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer that reads out DCTR weights for hdamp reweighting.
    Falls back gracefully if weights are missing.
    """

    n = len(events)
    ones = np.ones(n, dtype=np.float32)

    # always produce nominal
    events = set_ak_column(events, "hdamp_weight", ones)

    # safely access nested weights
    weight = getattr(events, "weight", None)

    up = None
    down = None

    if weight is not None:
        up = getattr(weight, "mlhdamp_up", None)
        down = getattr(weight, "mlhdamp_down", None)

    has_variations = (up is not None) and (down is not None)

    if has_variations:
        events = set_ak_column(events, "hdamp_weight_up", up)
        events = set_ak_column(events, "hdamp_weight_down", down)
    else:
        logger.warning(
            f"[{self.dataset_inst.name}] Missing mlhdamp weights → only nominal produced",
        )

    return events


@dctr_hdamp.init
def dctr_hdamp_init(self: Producer, **kwargs) -> None:

    self.produces.add("hdamp_weight")

    task = kwargs.get("task", None)

    shift = getattr(task, "global_shift_inst", None) if task else None
    is_nominal = (shift is None) or (shift.name == "nominal")

    if is_nominal:
        self.uses.add("weight.mlhdamp_up")
        self.uses.add("weight.mlhdamp_down")

        self.produces.add("hdamp_weight_up")
        self.produces.add("hdamp_weight_down")
