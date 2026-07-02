# coding: utf-8

"""
Style definitions.
"""

from __future__ import annotations

import order as od

from columnflow.util import DotDict


def stylize_processes(config: od.Config) -> None:
    """
    Adds process colors and adjust labels.
    """
    cfg = config

    # recommended cms colors
    # see https://cms-analysis.docs.cern.ch/guidelines/plotting/colors
    cfg.x.colors = DotDict(
        bright_blue="#3f90da",
        dark_blue="#011c87",
        purple="#832db6",
        aubergine="#964a8b",
        yellow="#f7c331",
        bright_orange="#ffa90e",
        dark_orange="#e76300",
        red="#bd1f01",
        teal="#92dadd",
        grey="#94a4a2",
        brown="#a96b59",
        green="#30c300",
        dark_green="#269c00",
    )

    cfg.x.color_names = [
        "dark_orange", "bright_blue", "dark_green", "red", "purple", "bright_orange", "dark_blue", "teal", "grey",
        "brown", "green",
    ]
    cfg.x.get_color_from_sequence = lambda i: cfg.x.colors[cfg.x.color_names[i % len(cfg.x.color_names)]]

    if not cfg.has_process("qcd_est"):
        cfg.add_process(name="qcd_est", id=30002)

    if (p := config.get_process("tt", default=None)):
        p.color1 = cfg.x.colors.red
        p.label = r"$t\bar{t}$"

    if (p := config.get_process("st", default=None)):
        p.color1 = cfg.x.colors.dark_orange

    if (p := config.get_process("qcd", default=None)):
        p.color1 = cfg.x.colors.yellow

    if (p := config.get_process("qcd_est", default=None)):
        p.color1 = cfg.x.colors.bright_blue
        p.label = r"Multijet est." 
