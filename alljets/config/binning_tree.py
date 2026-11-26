# coding: utf-8

"""
Fine grid and adaptive (tree) binning of the top-mass measurement in mtfit, m_W,avg^reco and R_bq.

The fine grid is used for one datacard with all processes and systematics (inference model fine_3D), from which
alljets/scripts/optimize_binning.py builds a binning of rectangular boxes (leaves of a greedy tree of splits at fine
grid edges) that minimizes the expected uncertainty on the top mass. The leaves are stored in
binning_tree_{name}.json next to this module and used by the tree_bin_{name} variables.

Only numpy and the standard library are needed, so that the optimizer script can import this module.
"""

from __future__ import annotations

import json
import os

import numpy as np

# observables in the order of the fine-grid axes
FINE_AXES = ("fit_Top1_mass", "reco_W_mass_avg", "reco_R_bq")

# fine-grid edges, the outermost cells reach far into the tails so that (almost) no event is outside
FINE_EDGES = {
    "fit_Top1_mass": [0.0] + [float(m) for m in range(120, 271, 5)] + [300.0, 5000.0],
    "reco_W_mass_avg": [0.0] + [float(m) for m in range(64, 107, 3)] + [1000.0],
    "reco_R_bq": [round(0.08 * i, 2) for i in range(21)] + [2.0, 100.0],
}

TREE_DIR = os.path.dirname(os.path.abspath(__file__))


def tree_path(name: str) -> str:
    return os.path.join(TREE_DIR, f"binning_tree_{name}.json")


def load_tree(name: str) -> dict | None:
    # leaves of a tree binning, None if it was not created yet
    path = tree_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def tree_bin_index(values: list[np.ndarray], tree: dict) -> np.ndarray:
    # index of the leaf containing each event, -1 for events outside all leaves (e.g. nan values); the outer edges
    # of the fine grid are open, so that events beyond them are in the outermost leaves
    values = [np.asarray(v, dtype=np.float64) for v in values]
    if len(values) != len(tree["axes"]):
        raise ValueError(f"tree over {tree['axes']} needs {len(tree['axes'])} observables, got {len(values)}")
    out = np.full(len(values[0]), -1, dtype=np.int64)
    for i, box in enumerate(tree["leaves"]):
        mask = np.ones(len(out), dtype=bool)
        for v, (lo, hi), edges in zip(values, box, tree["edges"]):
            if lo > edges[0]:
                mask &= v >= lo
            else:
                mask &= ~np.isnan(v)
            if hi < edges[-1]:
                mask &= v < hi
        out[mask & (out < 0)] = i
    return out


def build_tree_bin(events, builders, tree):
    # variable expression: leaf index from the observables computed by the builders (in the order of tree axes)
    import awkward as ak

    values = [ak.to_numpy(ak.fill_none(build(events), np.nan)) for build in builders]
    return tree_bin_index(values, tree).astype(np.float32)
