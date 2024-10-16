# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category


# add categories using the "add_category" tool which adds auto-generated ids
# the "selection" entries refer to names of selectors, e.g. in selection/example.py
def add_categories(cfg: od.Config) -> None:
    add_category(
        cfg,
        name="incl",
        selection="cat_incl",
        label="inclusive",
    )
    add_category(
        cfg,
        name="6j",
        selection="cat_6j",
        label="6 jets",
    )
    add_category(
        cfg,
        name="7j",
        selection="cat_7j",
        label="7+ jets",
    )
    add_category(
        cfg,
        name="2btj",
        selection="cat_2btj",
        label="2 b-tagged jets or more",
    )
    add_category(
        cfg,
        name="1btj",
        selection="cat_1btj",
        label="1 b-tagged jet",
    )
    add_category(
        cfg,
        name="0btj",
        selection="cat_0btj",
        label="0 b-tagged jets",
    )
