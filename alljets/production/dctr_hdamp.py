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
    uses={"weight.mlhdamp{_up,_down}"},
    produces={"hdamp_weight{,_up,_down}"},
    # only run on mc
    mc_only=True,
)
def dctr_hdamp(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer that reads out the DCTR weights for reweighting hdamp on an event-by-event basis.
    """

    # setup nominal weights
    ones = np.ones(len(events), dtype=np.float32)
    events = set_ak_column(events, "hdamp_weight", ones)

    up = getattr(events.weight, "mlhdamp_up", None)
    down = getattr(events.weight, "mlhdamp_down", None)

    if up is None or down is None:
        logger.warning(
            f"Missing mlhdamp weights in {self.dataset_inst.name}, filling with 1",
        )
        up = np.ones(len(events), dtype=np.float32)
        down = np.ones(len(events), dtype=np.float32)

    events = set_ak_column(events, "hdamp_weight_up", up)
    events = set_ak_column(events, "hdamp_weight_down", down)

    return events
