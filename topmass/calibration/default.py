# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.production.mc_weight import mc_weight
from columnflow.production.seeds import deterministic_seeds
from columnflow.util import maybe_import

ak = maybe_import("awkward")


@calibrator(
    uses={mc_weight, deterministic_seeds},
    produces={mc_weight, deterministic_seeds},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)

    return events
