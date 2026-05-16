# coding: utf-8

"""
Producers for storing parton shower weights.
"""

from __future__ import annotations
import law

from columnflow.production import producer, Producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

ak = maybe_import("awkward")
np = maybe_import("numpy")

# Index map for the 44 PSWeight variations in samples
# The order of the weights has been extracted from the MiniAOD
index_map = {

    # --- FSR reduced ---
    "fsr_weight_red_up": 0,
    "fsr_weight_red_down": 1,

    # --- FSR default ---
    "fsr_weight_up": 2,
    "fsr_weight_down": 3,

    # --- FSR conservative ---
    "fsr_weight_con_up": 4,
    "fsr_weight_con_down": 5,

    # --- FSR decorrelated ---
    "fsr_weight_G2GG_muR_down": 6,
    "fsr_weight_G2GG_muR_up": 7,
    "fsr_weight_G2QQ_muR_down": 8,
    "fsr_weight_G2QQ_muR_up": 9,
    "fsr_weight_Q2QG_muR_down": 10,
    "fsr_weight_Q2QG_muR_up": 11,
    "fsr_weight_X2XG_muR_down": 12,
    "fsr_weight_X2XG_muR_up": 13,

    "fsr_weight_G2GG_cNS_down": 14,
    "fsr_weight_G2GG_cNS_up": 15,
    "fsr_weight_G2QQ_cNS_down": 16,
    "fsr_weight_G2QQ_cNS_up": 17,
    "fsr_weight_Q2QG_cNS_down": 18,
    "fsr_weight_Q2QG_cNS_up": 19,
    "fsr_weight_X2XG_cNS_down": 20,
    "fsr_weight_X2XG_cNS_up": 21,

    # --- ISR reduced ---
    "isr_weight_red_up": 22,
    "isr_weight_red_down": 23,

    # --- ISR default ---
    "isr_weight_up": 24,
    "isr_weight_down": 25,

    # --- ISR conservative ---
    "isr_weight_con_up": 26,
    "isr_weight_con_down": 27,

    # --- ISR decorrelated ---
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
        key for key in index_map
        if (
            key.startswith("fsr_weight_") or key.startswith("isr_weight_")
        ) and ("muR" in key or "cNS" in key)
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


@ps_weights.post_init
def ps_weights_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    mode = self.mode

    if mode not in SETS:
        raise ValueError(f"Unknown mode '{mode}', available: {list(SETS.keys())}")

    shift = task.global_shift_inst
    is_nominal = ((shift.name == "nominal") and self.dataset_inst.has_tag("tt"))

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
