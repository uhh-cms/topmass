# coding: utf-8

"""
Definition of categories.
"""

import order as od


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    config.add_category(
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )

    lep_channels = ["ee", "mumu", "emu"]
    number_jets = ["2j", "3j", "4j", "5j"]

    for jet_id, n_jet in enumerate(number_jets, start=1):
        config.add_category(
            name=f"{n_jet}",
            id=100 * jet_id,
            selection=f"sel_{n_jet}",
            label=f"{n_jet} jets",
        )
    for lep_id, lep_ch in enumerate(lep_channels, start=1):
        config.add_category(
            name=f"{lep_ch}",
            id=10 * lep_id,
            selection=f"sel_{lep_ch}",
            label=f"events in the {lep_ch} channel",
        )

    for lep_id, lep_ch in enumerate(lep_channels, start=1):
        ch = config.get_category(lep_ch)
        ch.selection = "sel_ee"
        for jet_id, n_jet in enumerate(number_jets, start=1):
            n_jet = ch.add_category(
                name=f"{lep_ch}_{n_jet}",
                id=100 * jet_id + lep_id * 10,
                selection=f"sel_{lep_ch}_{n_jet}",
                label=f"{n_jet} jets in the {lep_ch} channel",
            )
