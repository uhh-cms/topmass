"""
Utility functions for the kinematic fit, used in `KinFit.py`.
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import flat_np_view

ak = maybe_import("awkward")


def appendindices(initial_array, target_lengths):
    """Pad inner index-lists so each reaches its event-specific length."""
    for i in range(len(initial_array)):
        inner_list = initial_array[i]
        target_length = target_lengths[i]
        available_numbers = list(range(target_length))
        for num in available_numbers:
            if num not in inner_list:
                inner_list.append(num)
            if len(inner_list) >= target_length:
                break
    return initial_array


def insert_at_index(to_insert, where):
    full_true = ak.full_like(where, True, dtype=bool)
    mask = full_true
    flat = flat_np_view(to_insert)
    cut_orig = ak.num(where[mask])
    cut_replaced = ak.unflatten(flat, cut_orig)
    original = where[~mask]
    combined = ak.concatenate((original, cut_replaced), axis=1)
    return combined
