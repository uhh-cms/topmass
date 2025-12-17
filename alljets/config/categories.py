# coding: utf-8

"""
Definition of categories.
"""

import order as od
# , create_category_combinations, CategoryGroup
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
    add_category(
        cfg,
        name="fit_nconv",
        selection="cat_fit_nconv",
        label="kinfit not converged",
    )
    add_category(
        cfg,
        name="fit_conv",
        selection="cat_fit_conv",
        label="kinfit converged",
    )
    add_category(
        cfg,
        name="fit_conv_leq",
        selection="cat_fit_conv_leq",
        label=r"below $\chi^2$ cut",
    )
    add_category(
        cfg,
        name="fit_conv_big",
        selection="cat_fit_conv_big",
        label="kinfit converged and $P_{gof} > 0.1$ ",
    )
    add_category(
        cfg,
        name="0btj_bkg",
        selection="cat_0btj_bkg",
        label="QCD estimation",
        tags={"0btj"},
    )
    add_category(
        cfg,
        name="2btj_sig",
        selection="cat_2btj_sig",
        label=">2 b-tagged jets & signal trigger",
        tags={"2btj"},
    )
    # add_category(
    #     cfg,
    #     name="fit_conv_leq_rbb",
    #     selection="cat_fit_conv_leq_rbb",
    #     label=r"below $\chi^2$ cut and above $\Delta R_{\text{b}}$ cut",
    # )
    # add_category(
    #     cfg,
    #     name="fit_rbb",
    #     selection="cat_rbb",
    #     label=r"above $\Delta R_{\text{b}}$ cut",
    # )
    # add_category(
    #     cfg,
    #     name="reco_sig",
    #     selection="cat_reco_sig",
    #     label=r"below $\chi^2$ cut and above $\Delta R_{\text{b}}$ cut",
    # )

    add_category(
        cfg,
        name="fit_matched",
        selection="cat_fit_matched",
        label="correctly matched events",
        tags={"matched"},
    )
    add_category(
        cfg,
        name="fit_unmatched",
        selection="cat_fit_unmatched",
        label="wrong or umatched events",
        tags={"matched"},
    )
#    main_categories = {
#        # number of jets
#        "njets": CategoryGroup(["incl", "6j", "7j"], is_complete=True, has_overlap=True),
#        # number of btagged jets
#        "sigorbkg": CategoryGroup(["sig", "bkg"], is_complete=False, has_overlap=False),
#        # kinematic fit convergence
#        "kinfitconv": CategoryGroup(["fit_conv", "fit_nconv"], is_complete=True, has_overlap=False),
#    }
