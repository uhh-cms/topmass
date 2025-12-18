# coding: utf-8

"""
Definition of variables.

This module defines all observable variables used in the analysis. Variables are used for:
- Histogramming: Creating distributions of physics quantities
- Plotting: Visualizing data and MC comparisons
- Analysis cuts: Defining selection criteria

Each variable includes:
- name: Unique identifier for the variable
- expression: How to compute/extract the variable from events (can be a column name or callable)
- binning: Histogram binning specification (can be tuple or list)
- x_title: Axis label for plots (supports LaTeX via raw strings)
- unit: Physical unit (e.g., "GeV")
- null_value: Default value for missing/invalid entries (typically EMPTY_FLOAT)

The add_variable helper function provides sensible defaults for overflow/underflow handling.
"""
from functools import partial

import order as od
from columnflow.columnar_util import (
    EMPTY_FLOAT,
)
from .jet_builder import (
    build_b1jet,
    build_b2jet,
    build_top1jet,
    build_top1recojet,
    build_w1jet,
    build_w1recojet,
    build_w2recojet,
)


def add_variables(cfg: od.Config) -> None:
    """
    Register all analysis variables to the configuration.

    This function adds variables for:
    - Basic event info (run, lumi, event number)
    - Jet kinematics (pt, eta, phi for individual jets and collections)
    - B-jet properties and b-tagging scores
    - Kinematic fit results (W masses, top masses, chi2)
    - Weights (MC, btag, pileup, trigger, PDF, scale)
    - Cutflow variables for monitoring selections

    Args:
        cfg: The analysis configuration object to add variables to
    """
    ###############################################################################
    #                            Basic Event Information                          #
    ###############################################################################
    add_variable(
        cfg,
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
    )
    add_variable(
        cfg,
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="nPV",
        expression="PV.npvs",
        null_value=EMPTY_FLOAT,
        binning=(60, -0.5, 59.5),
        x_title="Number of primary Vertices",
    )
    add_variable(
        cfg,
        name="nPVGood",
        expression="PV.npvsGood",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 60),
        x_title="Number of good primary Vertices",
    )
    add_variable(
        cfg,
        name="trig_bits",
        expression="trig_bits",
        binning=(3, -0.5, 2.5),
        x_title=r"trig bits",
    )
    ###############################################################################
    #                            Jet Kinematics                                   #
    ###############################################################################
    add_variable(
        cfg,
        name="n_jet",
        expression="n_jet",
        binning=(16, -0.5, 15.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="n_jet_in_acc",
        expression="n_jet_in_event",
        binning=(16, -0.5, 15.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="n_bjet_in_acc",
        expression="n_bjet_in_event",
        binning=(6, -0.5, 5.5),
        x_title="Number of b tagged jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="jets_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T}$ of all jets",
    )
    add_variable(
        cfg,
        name="jets_eta",
        expression="Jet.eta",
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$ of all jets",
    )
    add_variable(
        cfg,
        name="jets_phi",
        expression="Jet.phi",
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"$\phi$ of all jets",
    )
    add_variable(
        cfg,
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(20, 80.0, 500.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(20, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        cfg,
        name="jet1_phi",
        expression="Jet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    add_variable(
        cfg,
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(20, 50.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(20, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$",
    )
    add_variable(
        cfg,
        name="jet2_phi",
        expression="Jet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 2 $\phi$",
    )
    add_variable(
        cfg,
        name="jet3_pt",
        expression="Jet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(20, 40.0, 200.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet3_eta",
        expression="Jet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(20, -3.0, 3.0),
        x_title=r"Jet 3 $\eta$",
    )
    add_variable(
        cfg,
        name="jet3_phi",
        expression="Jet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 3 $\phi$",
    )
    add_variable(
        cfg,
        name="jet4_pt",
        expression="Jet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(20, 45.0, 150.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet4_eta",
        expression="Jet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(20, -3.0, 3.0),
        x_title=r"Jet 4 $\eta$",
    )
    add_variable(
        cfg,
        name="jet4_phi",
        expression="Jet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 4 $\phi$",
    )
    add_variable(
        cfg,
        name="jet5_pt",
        expression="Jet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(20, 40.0, 140.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet5_eta",
        expression="Jet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(20, -3.0, 3.0),
        x_title=r"Jet 5 $\eta$",
    )
    add_variable(
        cfg,
        name="jet5_phi",
        expression="Jet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 5 $\phi$",
    )
    add_variable(
        cfg,
        name="jet6_pt",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_eta",
        expression="Jet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(20, -3.0, 3.0),
        x_title=r"Jet 6 $\eta$",
    )
    add_variable(
        cfg,
        name="jet6_phi",
        expression="Jet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 6 $\phi$",
    )
    add_variable(
        cfg,
        name="jet6_pt_1",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 60, 1000],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_pt_2",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 40, 60, float("inf")],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_pt_3",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 40, 60, 1000],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_pt_4",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 23, 32, 38, 44, 50, 60, 80, 100, 150],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_pt_5",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 100.0),
        aux={"overflow": False, "underflow": False},
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_pt_6",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 23, 26, 29, 32, 36, 40, 44, 50, 60, 100],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_pt_7",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 23, 26, 29, 32, 36, 40, 44, 50, 60, 150],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="jet6_ptdummy",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 9999],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    # Event Jets
    add_variable(
        cfg,
        name="event_jet1_pt",
        expression="EventJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="event_jet1_eta",
        expression="EventJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        cfg,
        name="event_jet1_phi",
        expression="EventJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    add_variable(
        cfg,
        name="event_jet2_pt",
        expression="EventJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="event_jet2_eta",
        expression="EventJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$",
    )
    add_variable(
        cfg,
        name="event_jet2_phi",
        expression="EventJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 2 $\phi$",
    )
    add_variable(
        cfg,
        name="event_jet3_pt",
        expression="EventJet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    add_variable(
        cfg,
        name="event_jet3_eta",
        expression="EventJet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 3 $\eta$",
    )
    add_variable(
        cfg,
        name="event_jet3_phi",
        expression="EventJet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 3 $\phi$",
    )
    add_variable(
        cfg,
        name="event_jet4_pt",
        expression="EventJet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    add_variable(
        cfg,
        name="event_jet4_eta",
        expression="EventJet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 4 $\eta$",
    )
    add_variable(
        cfg,
        name="event_jet4_phi",
        expression="EventJet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 4 $\phi$",
    )
    add_variable(
        cfg,
        name="event_jet5_pt",
        expression="EventJet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    add_variable(
        cfg,
        name="event_jet5_eta",
        expression="EventJet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 5 $\eta$",
    )
    add_variable(
        cfg,
        name="event_jet5_phi",
        expression="EventJet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 5 $\phi$",
    )
    add_variable(
        cfg,
        name="event_jet6_pt",
        expression="EventJet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 100.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="event_jet6_eta",
        expression="EventJet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 6 $\eta$",
    )
    add_variable(
        cfg,
        name="event_jet6_phi",
        expression="EventJet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 6 $\phi$",
    )
    ###############################################################################
    #                            HT                                               #
    ###############################################################################
    add_variable(
        cfg,
        name="coarse_event_jet1_pt",
        expression="EventJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 50.0, 500.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet1_eta",
        expression="EventJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet1_phi",
        expression="EventJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet2_pt",
        expression="EventJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 40.0, 340.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet2_eta",
        expression="EventJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 2 $\eta$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet2_phi",
        expression="EventJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 2 $\phi$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet3_pt",
        expression="EventJet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(15, 40.0, 190),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet3_eta",
        expression="EventJet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 3 $\eta$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet3_phi",
        expression="EventJet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 3 $\phi$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet4_pt",
        expression="EventJet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(15, 40.0, 140.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet4_eta",
        expression="EventJet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 4 $\eta$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet4_phi",
        expression="EventJet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 4 $\phi$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet5_pt",
        expression="EventJet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(16, 35.0, 115.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet5_eta",
        expression="EventJet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 5 $\eta$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet5_phi",
        expression="EventJet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 5 $\phi$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet6_pt",
        expression="EventJet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(17, 37.5, 80.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet6_eta",
        expression="EventJet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 6 $\eta$",
    )
    add_variable(
        cfg,
        name="coarse_event_jet6_phi",
        expression="EventJet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 6 $\phi$",
    )
    # Other Variables
    add_variable(
        cfg,
        name="ht",
        expression="ht",
        binning=(200, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht1",
        expression="ht",
        binning=(70, 250, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht9",
        expression="ht",
        binning=[250, 300, 350, 400, 450, 500,
                 550, 600, 700, 800, 1000, 1250, 1500],
        aux={"overflow": False, "underflow": False},
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht5",
        expression="ht",
        binning=[0, 240, 300, 350, 400, 450, 500, 550,
                 600, 700, 800, 900, 1000, 1250, 1750, 2500],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="htcoarse",
        expression="ht",
        binning=(32, 400.0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht_old",
        expression="ht_old",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht3",
        expression="ht",
        binning=[380, 600, 1000, float("inf")],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht2",
        expression="ht",
        binning=[380, 600, 9999],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht7",
        expression="ht",
        binning=[0, 200, 300, 340, 380, 415, 450, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht8",
        expression="ht",
        binning=[0, 100, 150, 200, 300, 340,
                 380, 415, 450, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht6",
        expression="ht",
        binning=[0, 200, 300, 340, 380, 420, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht_dummy",
        expression="ht",
        binning=[380, 99999],
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="trig_ht",
        expression="trig_ht",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    ###############################################################################
    #                           B-Jet Kinematics & Tagging                        #
    ###############################################################################
    add_variable(
        cfg,
        name="n_bjet",
        expression="n_bjet",
        binning=(6, -0.5, 5.5),
        x_title="Number of Bjets",
    )
    add_variable(
        cfg,
        name="bjet1_pt",
        expression="Bjet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="bjet2_pt",
        expression="Bjet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="bjetbytag1_pt",
        expression="JetsByBTag.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="bjetbytag2_pt",
        expression="JetsByBTag.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="bjet1_phi",
        expression="Bjet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"BJet 1 $\phi$",
    )
    add_variable(
        cfg,
        name="bjet2_phi",
        expression="Bjet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"BJet 2 $\phi$",
    )
    add_variable(
        cfg,
        name="bjetbytag1_phi",
        expression="JetsByBTag.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"Highest b tag Jet $\phi$",
    )
    add_variable(
        cfg,
        name="bjetbytag2_phi",
        expression="JetsByBTag.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"Second highest b tag Jet $\phi$",
    )
    add_variable(
        cfg,
        name="bjet1_eta",
        expression="Bjet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="bjet2_eta",
        expression="Bjet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="bjetbytag1_eta",
        expression="JetsByBTag.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Highest b tag Jet $\eta$",
    )
    add_variable(
        cfg,
        name="bjetbytag2_eta",
        expression="JetsByBTag.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Second highest b tag Jet $\eta$",
    )
    add_variable(
        cfg,
        name="jets_btag",
        expression="Jet.btagDeepFlavB",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"btag scores",
    )
    add_variable(
        cfg,
        name="maxbtag",
        expression="maxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag",
        expression="secmaxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag5",
        expression="secmaxbtag_alt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag2",
        expression="secmaxbtag_alt",
        null_value=EMPTY_FLOAT,
        binning=[0, 0.05, 0.1, 0.15, 0.2, 0.25,
                 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.74, 0.75,
                 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
                 0.91, 0.92, 0.93, 0.94, 0.95,
                 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992, 0.994, 0.996, 0.998, 1],
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag3",
        expression="secmaxbtag_alt",
        null_value=EMPTY_FLOAT,
        aux={"underflow": False},
        binning=[0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                 0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
                 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992, 0.994, 0.996, 0.998, 1],
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag4",
        expression="secmaxbtag_alt",
        null_value=EMPTY_FLOAT,
        aux={"underflow": False},
        binning=[0.7476, 1],
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag6",
        expression="secmaxbtag_alt",
        null_value=EMPTY_FLOAT,
        aux={"underflow": False},
        binning=(52, 0.74, 1),
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag_type",
        expression="secmaxbtag_alt",
        null_value=EMPTY_FLOAT,
        binning=[0, 0.7476, 1],
        x_title=r"Second highest b tag score",
    )
    add_variable(
        cfg,
        name="jet1_btag",
        expression="Jet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 1 bTag",
    )
    add_variable(
        cfg,
        name="jet2_btag",
        expression="Jet.btagDeepFlavB[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 2 bTag",
    )
    add_variable(
        cfg,
        name="jet3_btag",
        expression="Jet.btagDeepFlavB[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 3 bTag",
    )
    add_variable(
        cfg,
        name="jet4_btag",
        expression="Jet.btagDeepFlavB[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 4 bTag",
    )
    add_variable(
        cfg,
        name="jet5_btag",
        expression="Jet.btagDeepFlavB[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 5 bTag",
    )
    add_variable(
        cfg,
        name="jet6_btag",
        expression="Jet.btagDeepFlavB[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 6 bTag",
    )
    ###############################################################################
    #                     Selection / Matching Features                           #
    ###############################################################################
    add_variable(
        cfg,
        name="MW1",
        expression="MW1",
        null_value=EMPTY_FLOAT,
        binning=(100, 40, 140),
        unit="GeV",
        x_title=r"$M_{W1}$",
    )
    add_variable(
        cfg,
        name="MW2",
        expression="MW2",
        null_value=EMPTY_FLOAT,
        binning=(100, 40, 140),
        unit="GeV",
        x_title=r"$M_{W2}$",
    )
    add_variable(
        cfg,
        name="Mt1",
        expression="Mt1",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$M_{t1}$",
    )
    add_variable(
        cfg,
        name="Mt1_1",
        expression="Mt1",
        null_value=EMPTY_FLOAT,
        binning=(50, 100, 400),
        unit="GeV",
        x_title=r"$M_{t1}$",
    )
    add_variable(
        cfg,
        name="Mt2",
        expression="Mt2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$M_{t2}$",
    )
    add_variable(
        cfg,
        name="deltaMt",
        expression="deltaMt",
        null_value=EMPTY_FLOAT,
        binning=(100, -40, 60),
        unit="GeV",
        x_title=r"$M_{t1} - M_{t2}$",
    )
    add_variable(
        cfg,
        name="chi2",
        expression="chi2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 200),
        x_title=r"$\chi^2$",
    )
    add_variable(
        cfg,
        name="chi2_0",
        expression="chi2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 10),
        x_title=r"$\chi^2$",
    )
    add_variable(
        cfg,
        name="deltaR",
        expression="deltaR",
        null_value=EMPTY_FLOAT,
        binning=(300, -0.005, 2.995),
        x_title=r"min $\Delta R$ of light jets",
    )
    add_variable(
        cfg,
        name="deltaRb",
        expression="deltaRb",
        null_value=EMPTY_FLOAT,
        binning=(70, 0, 7),
        x_title=r"min $\Delta R$ of b-jets",
    )
    cfg.add_variable(
        name="reco_combination_type",
        expression="combination_type",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct",
    )
    cfg.add_variable(
        name="reco_combination_type_for_plot",
        expression="combination_type",
        null_value=EMPTY_FLOAT,
        binning=(3, -0.5, 2.5),
        x_title=r"0: unmatched, 1: wrong, 2: correct",
    )
    add_variable(
        cfg,
        name="R2b4q",
        expression="R2b4q",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 3),
        x_title=r"$R_{2b4q}$",
    )
    ###############################################################################
    #                                Weights                                      #
    ###############################################################################
    add_variable(
        cfg,
        name="mc_weight",
        expression="mc_weight",
        binning=(200, 0, 500),
        x_title="MC weight",
    )
    add_variable(
        cfg,
        name="btag_weight",
        expression="btag_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title="btag weight",
    )
    add_variable(
        cfg,
        name="pu_weight",
        expression="pu_weight",
        null_value=EMPTY_FLOAT,
        binning=(60, 0, 1.5),
        x_title="pu weight",
    )
    add_variable(
        cfg,
        name="murmuf_weight",
        expression="murmuf_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title=r"$\mu_{r}\mu_{f}$ weight",
    )
    add_variable(
        cfg,
        name="pdf_weight",
        expression="pdf_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title="pdf weight",
    )
    add_variable(
        cfg,
        name="trig_weight",
        expression="trig_weight",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.7, 1.1),
        x_title="trigger weight",
    )
    ###############################################################################
    #                            Cutflow Features                                 #
    ###############################################################################
    add_variable(
        cfg,
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="cf_ht",
        expression="cutflow.ht",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    add_variable(
        cfg,
        name="cf_jet6_pt",
        expression="cutflow.jet6_pt",
        binning=(25, 0.0, 100.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="cf_n_bjet",
        expression="cutflow.n_bjet",
        binning=(6, -0.5, 5.5),
        unit="GeV",
        x_title=r"Number of Bjets",
    )
    add_variable(
        cfg,
        name="cf_n_jet",
        expression="cutflow.n_jet",
        binning=(6, -0.5, 5.5),
        unit="GeV",
        x_title=r"Number of Jets",
    )
    add_variable(
        cfg,
        name="cf_turnon",
        expression="cutflow.turnon",
        binning=(2, -0.5, 1.5),
        x_title=r"0: only in base trigger, 1: In both",
    )
    add_variable(
        cfg,
        name="cf_combination_type",
        expression="cutflow.combination_type",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct",
    )
    ###############################################################################
    #                            Features from kinematic fit                      #
    ###############################################################################
    add_variable(
        cfg,
        name="fitchi2",
        expression="FitChi2",
        # aux={"overflow": False},
        binning=(100, 0, 210),
        unit="",
        x_title=r"$\chi^{2}_{\text{fit}}$",
    )
    add_variable(
        cfg,
        name="fitPgof",
        expression="FitPgof",
        binning=(100, 0, 1),
        unit="",
        x_title=r"$P_{gof}$ from kinfit",
    )

    build_w1jet.inputs = ["FitW1.{x,y,z,t}"]
    add_variable(
        cfg,
        name="fit_W1_mass",
        expression=partial(build_w1jet, which="mass"),
        aux={"inputs": build_w1jet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{W}^{fit}$",
    )

    build_w1recojet.inputs = ["RecoW1.{x,y,z,t}"]

    add_variable(
        cfg,
        name="reco_W1_mass",
        expression=partial(build_w1recojet, which="mass"),
        aux={"inputs": build_w1recojet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{W_{1}}^{reco}$",
    )
    build_w2recojet.inputs = ["RecoW2.{x,y,z,t}"]

    add_variable(
        cfg,
        name="reco_W2_mass",
        expression=partial(build_w2recojet, which="mass"),
        aux={"inputs": build_w2recojet.inputs},
        binning=(100, 0, 200),
        unit="GeV",
        x_title=r"$m_{W_{2}}^{reco}$",
    )

    build_top1recojet.inputs = ["RecoTop1.{x,y,z,t}"]

    add_variable(
        cfg,
        name="reco_Top1_mass",
        expression=partial(build_top1recojet, which="mass"),
        aux={"inputs": build_top1recojet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{t}^{reco}$",
    )
    build_top1jet.inputs = ["FitTop1.{x,y,z,t}"]

    add_variable(
        cfg,
        name="fit_Top1_mass",
        expression=partial(build_top1jet, which="mass"),
        aux={"inputs": build_top1jet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{\text{t}}^{\text{fit}}$",
    )
    add_variable(
        cfg,
        name="fit_Top1_mass_130_500",
        expression=partial(build_top1jet, which="mass"),
        aux={"inputs": build_top1jet.inputs},
        binning=(100, 130, 500),
        unit="GeV",
        x_title=r"$m_{t}^{fit}$",
    )
    add_variable(
        cfg,
        name="fit_Top1_mass_2",
        expression=partial(build_top1jet, which="mass"),
        aux={"inputs": build_top1jet.inputs, "overflow": False},
        binning=(60, 100, 700),
        unit="GeV",
        x_title=r"fitted Top mass",
    )
    add_variable(
        cfg,
        name="fit_combination_type",
        expression="fitCombinationType",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct",
    )
    cfg.add_variable(
        name="fit_combination_type_for_plot",
        expression="fitCombinationType",
        null_value=EMPTY_FLOAT,
        binning=(3, -0.5, 2.5),
        x_title=r"0: unmatched, 1: wrong, 2: correct",
    )
    build_b1jet.inputs = ["FitB1.{pt,eta,phi,mass}"]
    add_variable(
        cfg,
        name="fit_B1_mass_",
        expression=partial(build_b1jet, which="mass"),
        aux={"inputs": build_b1jet.inputs},
        binning=(40, 0, 300),
        unit="GeV",
        x_title=r"fitted B1 mass",
    )

    build_b2jet.inputs = ["FitB2.{x,y,z,t}"]


# helper to add a variable to the config with some defaults
def add_variable(config: od.Config, *args, **kwargs) -> od.Variable:
    kwargs.setdefault("null_value", EMPTY_FLOAT)

    # create the variable
    variable = config.add_variable(*args, **kwargs)

    # defaults
    if not variable.has_aux("underflow"):
        variable.x.underflow = True
    if not variable.has_aux("overflow"):
        variable.x.overflow = True

    return variable
