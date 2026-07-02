# coding: utf-8

"""
Producers for storing the bfrag weights.
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
def bfrag_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer that reads out bfrag weights.
    """

    n = len(events)
    ones = np.ones(n, dtype=np.float32)

    # always produce nominal
    events = set_ak_column(events, "bfrag_weight", ones)

    # safely access nested weights
    bfragweight = getattr(events, "bfragweight", None)

    up = None
    down = None
    peterson = None
    nom = None

    if bfragweight is not None:
        up = getattr(bfragweight, "up", None)
        down = getattr(bfragweight, "down", None)
        peterson = getattr(bfragweight, "peterson", None)
        nom = getattr(bfragweight, "nominal", None)

    # Conditionally produce up/down variations if both nominal and up weights are available
    has_variations = (up is not None) and (down is not None) and (peterson is not None) and (nom is not None)

    if has_variations:
        # Symmetrize around 1.0: up variation is the provided weight, down variation is 2 - up
        events = set_ak_column(events, "bfrag_weight_up", nom)
        events = set_ak_column(events, "bfrag_weight_down", ones)

        events = set_ak_column(events, "bfrag_peterson_weight_up", peterson)
        events = set_ak_column(events, "bfrag_peterson_weight_down", ones)

        rel_up = up / nom
        rel_down = down / nom
        events = set_ak_column(events, "bfrag_rel_weight_up", rel_up)
        events = set_ak_column(events, "bfrag_rel_weight_down", rel_down)

        events = set_ak_column(events, "bfrag_lund_weight_up", up)
        events = set_ak_column(events, "bfrag_lund_weight_down", down)
    else:
        logger.warning(
            f"[{self.dataset_inst.name}] Missing bfrag weights → only nominal produced",
        )

    return events


@bfrag_weights.post_init
def bfrag_weights_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    self.produces.add("bfrag_weight")

    shift = task.global_shift_inst
    is_nominal = ((shift.name == "nominal") and self.dataset_inst.has_tag("tt"))

    if is_nominal:
        self.uses.add("bfragweight.nominal")
        self.uses.add("bfragweight.down")
        self.uses.add("bfragweight.up")
        self.uses.add("bfragweight.peterson")

        # Columns where nominal is applied and symmetrized
        self.produces.add("bfrag_weight_up")
        self.produces.add("bfrag_weight_down")

        # Columns for Peterson
        self.produces.add("bfrag_peterson_weight_up")
        self.produces.add("bfrag_peterson_weight_down")

        # Columns for relative weights
        self.produces.add("bfrag_rel_weight_up")
        self.produces.add("bfrag_rel_weight_down")

        # Columns for lund weights
        self.produces.add("bfrag_lund_weight_up")
        self.produces.add("bfrag_lund_weight_down")
