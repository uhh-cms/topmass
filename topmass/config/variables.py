# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="n_bjet",
        expression="n_bjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of Bjets",
    )
    config.add_variable(
        name="n_electron",
        expression="n_electron",
        binning=(11, -0.5, 10.5),
        x_title="Number of electrons",
    )
    config.add_variable(
        name="n_muon",
        expression="n_muon",
        binning=(11, -0.5, 10.5),
        x_title="Number of muons",
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0., 300.),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0., 300.),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="jet3_pt",
        expression="Jet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0., 300.),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    config.add_variable(
        name="m_min_lb",
        expression="m_min_lb",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 200.),
        unit="GeV",
        x_title="Minimal Lepton + B-quark mass",
    )
    config.add_variable(
        name="m_ll",
        expression="m_ll",
        null_value=EMPTY_FLOAT,
        binning=(15, 0., 300.),
        unit="GeV",
        x_title="Minimal Lepton + Lepton mass",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    config.add_variable(
        name="bjet1_pt",
        expression="Bjet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0., 300.),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    config.add_variable(
        name="bjet2_pt",
        expression="Bjet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0., 300.),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    config.add_variable(
        name="lepton1_pt",
        expression="lepton_pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(10, 0., 200.),
        unit="GeV",
        x_title=r"Lepton 1 $p_{T}$",
    )
    config.add_variable(
        name="lepton2_pt",
        expression="lepton_pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(10, 0., 200.),
        unit="GeV",
        x_title=r"Lepton 2 $p_{T}$",
    )
    config.add_variable(
        name="lepton1_eta",
        expression="lepton_eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(20, -2.5, 2.5),
        x_title=r"Lepton 1 $eta$",
    )
    config.add_variable(
        name="lepton2_eta",
        expression="lepton_eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(20, -2.5, 2.5),
        x_title=r"Lepton 2 $eta$",
    )
    config.add_variable(
        name="trailing_pt",
        expression="trailing_pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 0., 200.),
        x_title=r"trailing jet $p_{T}$",
    )

    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )

    # cutflow variables
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
