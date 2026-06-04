# coding: utf-8

"""
Producers for storing fragmentation weights from DCTR.
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
def dctr_rb(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer that reads out DCTR weights for rb reweighting.
    """

    n = len(events)
    ones = np.ones(n, dtype=np.float32)

    # always produce nominal
    events = set_ak_column(events, "rb_weight", ones)

    # safely access nested weights
    weight = getattr(events, "weight", None)

    up = None
    nom = None

    if weight is not None:
        up = getattr(weight, "rB_up", None)
        nom = getattr(weight, "rB_nominal", None)

    # Conditionally produce up/down variations if both nominal and up weights are available
    has_variations = (up is not None) and (nom is not None)

    if has_variations:
        # Symmetrize around 1.0: up variation is the provided weight, down variation is 2 - up
        events = set_ak_column(events, "rb_weight_up", nom)
        events = set_ak_column(events, "rb_weight_down", 2 - nom)
    else:
        logger.warning(
            f"[{self.dataset_inst.name}] Missing rb weights → only nominal produced",
        )

    return events


@dctr_rb.post_init
def dctr_rb_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    self.produces.add("rb_weight")

    shift = task.global_shift_inst
    is_nominal = ((shift.name == "nominal") and self.dataset_inst.has_tag("tt"))

    if is_nominal:
        self.uses.add("weight.rB_nominal")
        self.uses.add("weight.rB_up")

        self.produces.add("rb_weight_up")
        self.produces.add("rb_weight_down")
