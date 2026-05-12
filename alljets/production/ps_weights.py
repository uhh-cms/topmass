# coding: utf-8

"""
Producers for storing parton shower weights.
"""

from __future__ import annotations

from columnflow.production import producer, Producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

ak = maybe_import("awkward")
np = maybe_import("numpy")

# Index map for the 44 PSWeight variations in samples
# This mapping assumes the exact ordering of UncertaintyBands:List
# as defined in the generator configuration
# https://github.com/cms-sw/cmssw/blob/1d517cc3ed9bb410dc82614f9be4a20c9dea3f37/Configuration/Generator/python/PSweightsPythia/PythiaPSweightsSettings_cfi.py
index_map = {
    # --- reduced ---
    "isr_weight_red_up": 0,
    "fsr_weight_red_up": 1,
    "isr_weight_red_down": 2,
    "fsr_weight_red_down": 3,

    # --- default ---
    "isr_weight_up": 4,
    "fsr_weight_up": 5,
    "isr_weight_down": 6,
    "fsr_weight_down": 7,

    # --- conservative ---
    "isr_weight_con_up": 8,
    "fsr_weight_con_up": 9,
    "isr_weight_con_down": 10,
    "fsr_weight_con_down": 11,

    # --- FSR decorrelated (12–27) ---
    "fsr_weight_G2GG_muR_down": 12,
    "fsr_weight_G2GG_muR_up": 13,
    "fsr_weight_G2QQ_muR_down": 14,
    "fsr_weight_G2QQ_muR_up": 15,
    "fsr_weight_Q2QG_muR_down": 16,
    "fsr_weight_Q2QG_muR_up": 17,
    "fsr_weight_X2XG_muR_down": 18,
    "fsr_weight_X2XG_muR_up": 19,

    "fsr_weight_G2GG_cNS_down": 20,
    "fsr_weight_G2GG_cNS_up": 21,
    "fsr_weight_G2QQ_cNS_down": 22,
    "fsr_weight_G2QQ_cNS_up": 23,
    "fsr_weight_Q2QG_cNS_down": 24,
    "fsr_weight_Q2QG_cNS_up": 25,
    "fsr_weight_X2XG_cNS_down": 26,
    "fsr_weight_X2XG_cNS_up": 27,

    # --- ISR decorrelated (28–43) ---
    "isr_weight_G2GG_muR_down": 28,
    "isr_weight_G2GG_muR_up": 29,
    "isr_weight_G2QQ_muR_down": 30,
    "isr_weight_G2QQ_muR_up": 31,
    "isr_weight_Q2QG_muR_down": 32,
    "isr_weight_Q2QG_muR_up": 33,
    "isr_weight_X2XG_muR_down": 34,
    "isr_weight_X2XG_muR_up": 35,

    "isr_weight_G2GG_cNS_down": 36,
    "isr_weight_G2GG_cNS_up": 37,
    "isr_weight_G2QQ_cNS_down": 38,
    "isr_weight_G2QQ_cNS_up": 39,
    "isr_weight_Q2QG_cNS_down": 40,
    "isr_weight_Q2QG_cNS_up": 41,
    "isr_weight_X2XG_cNS_down": 42,
    "isr_weight_X2XG_cNS_up": 43,
}


SETS = {
    "reduced": [
        "isr_weight_red_up", "fsr_weight_red_up",
        "isr_weight_red_down", "fsr_weight_red_down",
    ],
    "default": [
        "isr_weight_up", "fsr_weight_up",
        "isr_weight_down", "fsr_weight_down",
    ],
    "conservative": [
        "isr_weight_con_up", "fsr_weight_con_up",
        "isr_weight_con_down", "fsr_weight_con_down",
    ],
    "decorrelated": [
        key for key, idx in index_map.items() if idx >= 12
    ],
}


@producer(
    # only run on mc
    mc_only=True,
    # Which set of variations to store, see SETS dict for available options
    mode="decorrelated",
)
def ps_weights(
    self: Producer,
    events: ak.Array,
    invalid_weights_action: str = "raise",
    **kwargs,
) -> ak.Array:
    """
    Producer that reads out parton shower uncertainties on an event-by-event basis.
    """
    known_actions = {"raise", "ignore_one", "ignore"}
    if invalid_weights_action not in known_actions:
        raise ValueError(
            f"unknown invalid_weights_action '{invalid_weights_action}', known values are {','.join(known_actions)}",
        )

    # setup nominal weights
    ones = np.ones(len(events), dtype=np.float32)
    events = set_ak_column(events, "fsr_weight", ones)
    events = set_ak_column(events, "isr_weight", ones)

    # check if weight variations are missing and if needed, pad them
    selected = self.ps_columns
    ps_weights = events.PSWeight

    n_weights = ak.num(events.PSWeight, axis=1)

    # Case: reduced PS weights (only 4 entries)
    if ak.all(n_weights == 4):
        return events

    missing = [c for c in selected if c not in index_map]
    if missing:
        raise ValueError(f"Missing index_map entries: {missing}")

    for column in selected:
        index = index_map.get(column)

        if index is None:
            raise KeyError(f"Missing index for {column}")

        events = set_ak_column(events, column, ps_weights[:, index])

    return events


@ps_weights.init
def ps_weights_init(self: Producer, **kwargs) -> None:
    mode = self.mode

    if mode not in SETS:
        raise ValueError(f"Unknown mode '{mode}', available: {list(SETS.keys())}")

    task = kwargs.get("task", None)

    shift = getattr(task, "global_shift_inst", None) if task else None
    is_nominal = (shift is None) or (shift.name == "nominal")

    # use the PSWeight
    self.uses.add("PSWeight")

    # always produce nominal weights
    self.produces.add("isr_weight")
    self.produces.add("fsr_weight")

    if is_nominal:
        self.ps_columns = SETS[mode]
        for name in self.ps_columns:
            self.produces.add(name)
    else:
        self.ps_columns = []
