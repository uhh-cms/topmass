# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.util import maybe_import

ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    uses={normalization_weights},
    produces={normalization_weights},
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Opinionated wrapper of several event weight producers. It declares dependence all shifts that
    might possibly change any of the weights.
    """
    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    return events
