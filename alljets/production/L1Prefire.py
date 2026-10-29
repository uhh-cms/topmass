# coding: utf-8

"""
Producer for storing L1 prefiring weights.
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
def l1_prefiring(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Producer that reads out the L1 prefiring weights, and writes them out
    as flat columns 'l1_prefiring_weight', 'l1_prefiring_weight_up' and
    'l1_prefiring_weight_down'.

    Resources:

       - https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe?rev=3
    """

    events = set_ak_column(events, "l1_prefiring_weight", events.L1PreFiringWeight.Nom, value_type=np.float32)
    events = set_ak_column(events, "l1_prefiring_weight_up", events.L1PreFiringWeight.Up, value_type=np.float32)
    events = set_ak_column(events, "l1_prefiring_weight_down", events.L1PreFiringWeight.Dn, value_type=np.float32)

    return events


@l1_prefiring.post_init
def l1_prefiring_post_init(self: Producer, task: law.Task, **kwargs) -> None:

    self.uses |= {
        "L1PreFiringWeight.Nom",
        "L1PreFiringWeight.Up",
        "L1PreFiringWeight.Dn",
    }

    self.produces |= {
        "l1_prefiring_weight",
        "l1_prefiring_weight_up",
        "l1_prefiring_weight_down",
    }
