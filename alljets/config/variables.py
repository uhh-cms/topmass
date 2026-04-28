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

from modules.columnflow.columnflow.util import maybe_import
import order as od
from columnflow.columnar_util import (
    EMPTY_FLOAT,
)
from .jet_builder import (
    build_top1jet,
    build_top1recojet,
    build_top2recojet,
    build_w1jet,
    build_w1recojet,
    build_w2recojet,
    build_ttbar,
    build_avg_w_mass,
    build_avg_reco_Top_mass,
    build_reco_R_bq,
)

ak = maybe_import("awkward")


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
        binning=(4, -0.5, 3.5),
        x_title=r"trig bits",
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
    add_variable(
        cfg,
        name="top_pt_weight",
        expression="top_pt_weight",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.7, 1.4),
        x_title="top pt weight",
    )
    ###############################################################################
    #                            TriJet Kinematics                                #
    #                       Using Jets within |eta| < 2.6                         #
    ###############################################################################
    add_variable(
        cfg,
        name="trigjet1_pt",
        expression="TrigJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="trigjet1_eta",
        expression="TrigJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        cfg,
        name="trigjet1_phi",
        expression="TrigJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 1 $\phi$",
    )
    add_variable(
        cfg,
        name="trigjet2_pt",
        expression="TrigJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="trigjet2_eta",
        expression="TrigJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$",
    )
    add_variable(
        cfg,
        name="trigjet2_phi",
        expression="TrigJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 2 $\phi$",
    )
    add_variable(
        cfg,
        name="trigjet3_pt",
        expression="TrigJets.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    add_variable(
        cfg,
        name="trigjet3_eta",
        expression="TrigJets.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 3 $\eta$",
    )
    add_variable(
        cfg,
        name="trigjet3_phi",
        expression="TrigJets.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 3 $\phi$",
    )
    add_variable(
        cfg,
        name="trigjet4_pt",
        expression="TrigJets.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    add_variable(
        cfg,
        name="trigjet4_eta",
        expression="TrigJets.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 4 $\eta$",
    )
    add_variable(
        cfg,
        name="trigjet4_phi",
        expression="TrigJets.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 4 $\phi$",
    )
    add_variable(
        cfg,
        name="trigjet5_pt",
        expression="TrigJets.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    add_variable(
        cfg,
        name="trigjet5_eta",
        expression="TrigJets.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 5 $\eta$",
    )
    add_variable(
        cfg,
        name="trigjet5_phi",
        expression="TrigJets.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 5 $\phi$",
    )
    add_variable(
        cfg,
        name="trigjet6_pt",
        expression="TrigJets.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 100.0),
        aux={"overflow": False, "underflow": False},
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="trigjet6_eta",
        expression="TrigJets.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 6 $\eta$",
    )
    add_variable(
        cfg,
        name="trigjet6_phi",
        expression="TrigJets.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 6 $\phi$",
    )
    add_variable(
        cfg,
        name="trigjets_pt",
        expression="TrigJets.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T}$ of all jets",
    )
    add_variable(
        cfg,
        name="trigjets_eta",
        expression="TrigJets.eta",
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta$ of all jets",
    )
    add_variable(
        cfg,
        name="trigjets_phi",
        expression="TrigJets.phi",
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi$ of all jets",
    )
    ###############################################################################
    #                      HT (Scalar sum of pT of the Jets)                      #
    #                     Mostly relevant for trigger studies                     #
    ###############################################################################
    add_variable(
        cfg,
        name="ht",
        expression="ht",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="ht_trigger",
        expression="ht",
        binning=[250, 300, 350, 400, 450, 500,
                 550, 600, 700, 800, 1000, 1250, 1500],
        aux={"overflow": False, "underflow": False},
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
    #                                 B-Jet Tagging                               #
    ###############################################################################
    add_variable(
        cfg,
        name="maxbtag",
        expression="maxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Highest B-Tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag",
        expression="secmaxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Second highest B-Tag score",
    )
    add_variable(
        cfg,
        name="secmaxbtag_type",
        expression="secmaxbtag",
        null_value=EMPTY_FLOAT,
        binning=[0, 0.7476, 1],
        x_title=r"Second highest B-Tag score",
    )
    ###############################################################################
    #                           SelectedJets Kinematics                           #
    #                   Jets within |eta| < 2.4 & pt >= 40 GeV                    #
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
        name="n_bjet",
        expression="n_bjet",
        binning=(6, -0.5, 5.5),
        x_title="Number of Bjets",
    )
    add_variable(
        cfg,
        name="seljet1_pt",
        expression="SelectedJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet1_eta",
        expression="SelectedJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet1_phi",
        expression="SelectedJets.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 1 $\phi$",
    )
    add_variable(
        cfg,
        name="seljet1_jetid",
        expression="SelectedJets.jetId[:,0]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet Id of Jet 1",
    )
    add_variable(
        cfg,
        name="seljet1_puId",
        expression="SelectedJets.puId[:,0]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet 1 PU Id",
    )
    add_variable(
        cfg,
        name="seljet2_pt",
        expression="SelectedJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet2_eta",
        expression="SelectedJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet2_phi",
        expression="SelectedJets.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 2 $\phi$",
    )
    add_variable(
        cfg,
        name="seljet2_jetid",
        expression="SelectedJets.jetId[:,1]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet Id of Jet 2",
    )
    add_variable(
        cfg,
        name="seljet2_puId",
        expression="SelectedJets.puId[:,1]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet 2 PU Id",
    )
    add_variable(
        cfg,
        name="seljet3_pt",
        expression="SelectedJets.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet3_eta",
        expression="SelectedJets.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 3 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet3_phi",
        expression="SelectedJets.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 3 $\phi$",
    )
    add_variable(
        cfg,
        name="seljet3_jetid",
        expression="SelectedJets.jetId[:,2]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet Id of Jet 3",
    )
    add_variable(
        cfg,
        name="seljet3_puId",
        expression="SelectedJets.puId[:,2]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet 3 PU Id",
    )
    add_variable(
        cfg,
        name="seljet4_pt",
        expression="SelectedJets.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet4_eta",
        expression="SelectedJets.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 4 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet4_phi",
        expression="SelectedJets.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 4 $\phi$",
    )
    add_variable(
        cfg,
        name="seljet4_jetid",
        expression="SelectedJets.jetId[:,3]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet Id of Jet 4",
    )
    add_variable(
        cfg,
        name="seljet4_puId",
        expression="SelectedJets.puId[:,3]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet 4 PU Id",
    )
    add_variable(
        cfg,
        name="seljet5_pt",
        expression="SelectedJets.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet5_eta",
        expression="SelectedJets.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 5 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet5_phi",
        expression="SelectedJets.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 5 $\phi$",
    )
    add_variable(
        cfg,
        name="seljet5_jetid",
        expression="SelectedJets.jetId[:,4]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet Id of Jet 5",
    )
    add_variable(
        cfg,
        name="seljet5_puId",
        expression="SelectedJets.puId[:,4]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet 5 PU Id",
    )
    add_variable(
        cfg,
        name="seljet6_pt",
        expression="SelectedJets.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet6_eta",
        expression="SelectedJets.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 6 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet6_phi",
        expression="SelectedJets.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"Jet 6 $\phi$",
    )
    add_variable(
        cfg,
        name="seljet6_jetid",
        expression="SelectedJets.jetId[:,5]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet Id of Jet 6",
    )
    add_variable(
        cfg,
        name="seljet6_puId",
        expression="SelectedJets.puId[:,5]",
        null_value=-1,
        binning=(9, -0.5, 8.5),
        discrete_x=True,
        x_title=r"Jet 6 PU Id",
    )
    add_variable(
        cfg,
        name="seljets_pt",
        expression="SelectedJets.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T}$ of all jets",
    )
    add_variable(
        cfg,
        name="seljets_eta",
        expression="SelectedJets.eta",
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta$ of all jets",
    )
    add_variable(
        cfg,
        name="seljets_phi",
        expression="SelectedJets.phi",
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi$ of all jets",
    )
    add_variable(
        cfg,
        name="seljets_jetId",
        expression="SelectedJets.jetId",
        binning=(8, -0.5, 7.5),
        discrete_x=True,
        x_title=r"Jet ID of all jets",
    )
    add_variable(
        cfg,
        name="seljets_puId",
        expression="SelectedJets.puId",
        binning=(10, -0.5, 9.5),
        discrete_x=False,
        x_title=r"Jet PU ID of all jets",
    )
    add_variable(
        cfg,
        name="seljets_btag",
        expression="SelectedJets.btagDeepFlavB",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"B-Tag scores of all Jets",
    )
    ###############################################################################
    #                           FitJet reco kinematics                            #
    #                Reco values of the six jets that entered the fit             #
    #  First two jets are b-jet candidates, then light quark jets for W1,then W2  #
    ###############################################################################
    add_variable(
        cfg,
        name="fitjetreco1_pt",
        expression="FitJet.reco.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{b_1\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco1_eta",
        expression="FitJet.reco.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{b_1\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco1_phi",
        expression="FitJet.reco.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{b_1\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco2_pt",
        expression="FitJet.reco.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{b_2\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco2_eta",
        expression="FitJet.reco.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{b_2\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco2_phi",
        expression="FitJet.reco.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{b_2\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco3_pt",
        expression="FitJet.reco.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_1\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco3_eta",
        expression="FitJet.reco.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_1\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco3_phi",
        expression="FitJet.reco.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_1\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco4_pt",
        expression="FitJet.reco.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_2\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco4_eta",
        expression="FitJet.reco.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_2\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco4_phi",
        expression="FitJet.reco.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_2\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco5_pt",
        expression="FitJet.reco.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_3\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco5_eta",
        expression="FitJet.reco.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_3\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco5_phi",
        expression="FitJet.reco.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_3\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco6_pt",
        expression="FitJet.reco.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_4\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco6_eta",
        expression="FitJet.reco.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_4\,\mathrm{reco}}$",
    )
    add_variable(
        cfg,
        name="fitjetreco6_phi",
        expression="FitJet.reco.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_4\,\mathrm{reco}}$",
    )
    # Variables for both b-jet canditates
    add_variable(
        cfg,
        name="fitjetreco_B_pt",
        expression="FitJet.reco.pt[:,0:2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{reco}}$ of b-jet candidates",
    )
    add_variable(
        cfg,
        name="fitjetreco_B_eta",
        expression="FitJet.reco.eta[:,0:2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{reco}}$ of b-jet candidates",
    )
    add_variable(
        cfg,
        name="fitjetreco_B_phi",
        expression="FitJet.reco.phi[:,0:2]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{reco}}$ of b-jet candidates",
    )
    # Variables for light quark jet candidates of W1
    add_variable(
        cfg,
        name="fitjetreco_W1_pt",
        expression="FitJet.reco.pt[:,2:4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{reco}}$ of jet candidates for W1",
    )
    add_variable(
        cfg,
        name="fitjetreco_W1_eta",
        expression="FitJet.reco.eta[:,2:4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{reco}}$ of jet candidates for W1",
    )
    add_variable(
        cfg,
        name="fitjetreco_W1_phi",
        expression="FitJet.reco.phi[:,2:4]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{reco}}$ of jet candidates for W1",
    )
    # Variables for light quark jet candidates of W2
    add_variable(
        cfg,
        name="fitjetreco_W2_pt",
        expression="FitJet.reco.pt[:,4:6]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{reco}}$ of jet candidates for W2",
    )
    add_variable(
        cfg,
        name="fitjetreco_W2_eta",
        expression="FitJet.reco.eta[:,4:6]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{reco}}$ of jet candidates for W2",
    )
    add_variable(
        cfg,
        name="fitjetreco_W2_phi",
        expression="FitJet.reco.phi[:,4:6]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{reco}}$ of jet candidates for W2",
    )
    # Variables for all light quark jet candidates
    add_variable(
        cfg,
        name="fitjetreco_light_pt",
        expression="FitJet.reco.pt[:,2:6]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{reco}}$ of light quark jet candidates",
    )
    add_variable(
        cfg,
        name="fitjetreco_light_eta",
        expression="FitJet.reco.eta[:,2:6]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{reco}}$ of light quark jet candidates",
    )
    add_variable(
        cfg,
        name="fitjetreco_light_phi",
        expression="FitJet.reco.phi[:,2:6]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{reco}}$ of light quark jet candidates",
    )
    ###############################################################################
    #                           FitJet kinematics                                 #
    #                Fitted values of the jets that entered the fit               #
    #  First two jets are b-jet candidates, then light quark jets for W1,then W2  #
    ###############################################################################
    add_variable(
        cfg,
        name="fitjet1_pt",
        expression="FitJet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{b_1\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet1_eta",
        expression="FitJet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{b_1\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet1_phi",
        expression="FitJet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{b_1\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet2_pt",
        expression="FitJet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{b_2\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet2_eta",
        expression="FitJet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{b_2\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet2_phi",
        expression="FitJet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{b_2\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet3_pt",
        expression="FitJet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_1\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet3_eta",
        expression="FitJet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_1\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet3_phi",
        expression="FitJet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_1\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet4_pt",
        expression="FitJet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_2\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet4_eta",
        expression="FitJet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_2\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet4_phi",
        expression="FitJet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_2\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet5_pt",
        expression="FitJet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_3\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet5_eta",
        expression="FitJet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_3\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet5_phi",
        expression="FitJet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_3\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet6_pt",
        expression="FitJet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{q_4\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet6_eta",
        expression="FitJet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{q_4\,\mathrm{fit}}$",
    )
    add_variable(
        cfg,
        name="fitjet6_phi",
        expression="FitJet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{q_4\,\mathrm{fit}}$",
    )
    # Variables for both b-jet canditates
    add_variable(
        cfg,
        name="fitjet_B_pt",
        expression="FitJet.pt[:,0:2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{fit}}$ of b-jet candidates",
    )
    add_variable(
        cfg,
        name="fitjet_B_eta",
        expression="FitJet.eta[:,0:2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{fit}}$ of b-jet candidates",
    )
    add_variable(
        cfg,
        name="fitjet_B_phi",
        expression="FitJet.phi[:,0:2]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{fit}}$ of b-jet candidates",
    )
    # Variables for light quark jet candidates of W1
    add_variable(
        cfg,
        name="fitjet_W1_pt",
        expression="FitJet.pt[:,2:4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{fit}}$ of jet candidates for W1",
    )
    add_variable(
        cfg,
        name="fitjet_W1_eta",
        expression="FitJet.eta[:,2:4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{fit}}$ of jet candidates for W1",
    )
    add_variable(
        cfg,
        name="fitjet_W1_phi",
        expression="FitJet.phi[:,2:4]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{fit}}$ of jet candidates for W1",
    )
    # Variables for light quark jet candidates of W2
    add_variable(
        cfg,
        name="fitjet_W2_pt",
        expression="FitJet.pt[:,4:6]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{fit}}$ of jet candidates for W2",
    )
    add_variable(
        cfg,
        name="fitjet_W2_eta",
        expression="FitJet.eta[:,4:6]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{fit}}$ of jet candidates for W2",
    )
    add_variable(
        cfg,
        name="fitjet_W2_phi",
        expression="FitJet.phi[:,4:6]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{fit}}$ of jet candidates for W2",
    )
    # Variables for all light quark jet candidates
    add_variable(
        cfg,
        name="fitjet_light_pt",
        expression="FitJet.pt[:,2:6]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T^{\mathrm{fit}}$ of light quark jet candidates",
    )
    add_variable(
        cfg,
        name="fitjet_light_eta",
        expression="FitJet.eta[:,2:6]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\eta^{\mathrm{fit}}$ of light quark jet candidates",
    )
    add_variable(
        cfg,
        name="fitjet_light_phi",
        expression="FitJet.phi[:,2:6]",
        null_value=EMPTY_FLOAT,
        binning=(66, -3.3, 3.3),
        x_title=r"$\phi^{\mathrm{fit}}$ of light quark jet candidates",
    )
    ###############################################################################
    #                            Features from kinematic fit                      #
    #                           Columns extra produced / stored                   #
    ###############################################################################
    add_variable(
        cfg,
        name="fit_combination_type",
        expression="fitCombinationType",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct",
    )
    add_variable(
        cfg,
        name="fitchi2",
        expression="FitChi2",
        binning=(50, 0, 50),
        aux={"overflow": False, "underflow": False},
        unit="",
        x_title=r"$\chi^{2}$",
    )
    add_variable(
        cfg,
        name="fitPgof",
        expression="FitPgof",
        binning=(100, 0, 1),
        unit="",
        x_title=r"$P_{gof}$",
    )
    add_variable(
        cfg,
        name="fit_deltaRbb",
        expression="FitRbb",
        null_value=EMPTY_FLOAT,
        binning=(59, 0, 5.8),
        x_title=r"$\Delta R_{b\overline{b}}$",
    )
    build_w1recojet.inputs = ["RecoW1.{x,y,z,t}"]
    add_variable(
        cfg,
        name="reco_W1_mass",
        expression=partial(build_w1recojet, which="mass"),
        aux={"inputs": build_w1recojet.inputs},
        binning=(100, 0, 200),
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
    build_avg_w_mass.inputs = (build_w1recojet.inputs + build_w2recojet.inputs)
    add_variable(
        cfg,
        name="reco_W_mass_avg",
        expression=partial(build_avg_w_mass),
        aux={"inputs": build_avg_w_mass.inputs},
        binning=(25, 60, 110),
        unit="GeV",
        x_title=r"$m_{W_{avg}}^{reco}$",
    )
    add_variable(
        cfg,
        name="reco_W_mass_avg_percentile",
        expression=partial(build_avg_w_mass),
        aux={"inputs": build_avg_w_mass.inputs},
        binning=[65.5, 76.3, 79.5, 82, 84.3, 86.6, 89.1, 92.3, 107],
        unit="GeV",
        x_title=r"$m_{W_{avg}}^{reco}$",
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
    build_top1recojet.inputs = ["RecoTop1.{x,y,z,t}"]
    add_variable(
        cfg,
        name="reco_Top1_mass",
        expression=partial(build_top1recojet, which="mass"),
        aux={"inputs": build_top1recojet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{t_{1}}^{reco}$",
    )
    build_top2recojet.inputs = ["RecoTop2.{x,y,z,t}"]
    add_variable(
        cfg,
        name="reco_Top2_mass",
        expression=partial(build_top2recojet, which="mass"),
        aux={"inputs": build_top2recojet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{t_{2}}^{reco}$",
    )
    build_avg_reco_Top_mass.inputs = (build_top1recojet.inputs + build_top2recojet.inputs)
    add_variable(
        cfg,
        name="reco_Top_mass_avg",
        expression=partial(build_avg_reco_Top_mass),
        aux={"inputs": build_avg_reco_Top_mass.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{t_{avg}}^{reco}$",
    )
    build_ttbar.inputs = ["RecoTop1.{x,y,z,t}", "RecoTop2.{x,y,z,t}"]
    add_variable(
        cfg,
        name="reco_ttbar_mass",
        expression=partial(build_ttbar, which="mass"),
        aux={"inputs": build_ttbar.inputs},
        binning=(100, 0, 2000),
        unit="GeV",
        x_title=r"$m_{t\overline{t}}^{reco}$",
    )
    build_top1jet.inputs = ["FitTop1.{x,y,z,t}"]
    add_variable(
        cfg,
        name="fit_Top1_mass",
        expression=partial(build_top1jet, which="mass"),
        aux={"inputs": build_top1jet.inputs},
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$m_{t}^{fit}$",
    )
    add_variable(
        cfg,
        name="fit_Top1_mass_percentile",
        expression=partial(build_top1jet, which="mass"),
        aux={"inputs": build_top1jet.inputs},
        binning=[99.9, 161, 167, 172, 178, 186, 224, 293, 1.38e+03],
        unit="GeV",
        x_title=r"$m_{t}^{fit}$",
    )
    build_reco_R_bq.inputs = ["FitJet.reco.{pt,eta,phi,mass}"]
    add_variable(
        cfg,
        name="reco_R_bq",
        expression=build_reco_R_bq,
        aux={"inputs": build_reco_R_bq.inputs},
        binning=(50, 0, 10),
        unit="",
        x_title=r"$R_{bq}$",
    )
    add_variable(
        cfg,
        name="reco_R_bq_percentile",
        expression=build_reco_R_bq,
        aux={"inputs": build_reco_R_bq.inputs},
        binning=[0.0884, 0.333, 0.42, 0.501, 0.589, 0.69, 0.821, 1.03, 8.17],
        unit="",
        x_title=r"$R_{bq}$",
    )
    ###############################################################################
    #                            Features with coarse binning                     #
    #                                 for validation plots                        #
    ###############################################################################
    add_variable(
        cfg,
        name="reco_Top1_mass_coarse",
        expression=partial(build_top1recojet, which="mass"),
        aux={"inputs": build_top1recojet.inputs},
        binning=(15, 100, 550),
        unit="GeV",
        x_title=r"$m_{t_{1}}^{reco}$",
    )
    add_variable(
        cfg,
        name="reco_Top2_mass_coarse",
        expression=partial(build_top2recojet, which="mass"),
        aux={"inputs": build_top2recojet.inputs},
        binning=(15, 100, 550),
        unit="GeV",
        x_title=r"$m_{t_{2}}^{reco}$",
    )
    add_variable(
        cfg,
        name="reco_Top_mass_avg_coarse",
        expression=partial(build_avg_reco_Top_mass),
        aux={"inputs": build_avg_reco_Top_mass.inputs},
        binning=(15, 100, 550),
        unit="GeV",
        x_title=r"$m_{t_{avg}}^{reco}$",
    )
    add_variable(
        cfg,
        name="fit_Top1_mass_coarse",
        expression=partial(build_top1jet, which="mass"),
        aux={"inputs": build_top1jet.inputs},
        binning=(15, 100, 550),
        unit="GeV",
        x_title=r"$m_{t}^{fit}$",
    )
    add_variable(
        cfg,
        name="reco_W1_mass_coarse",
        expression=partial(build_w1recojet, which="mass"),
        aux={"inputs": build_w1recojet.inputs},
        binning=(15, 60, 120),
        unit="GeV",
        x_title=r"$m_{W_{1}}^{reco}$",
    )
    add_variable(
        cfg,
        name="reco_W2_mass_coarse",
        expression=partial(build_w2recojet, which="mass"),
        aux={"inputs": build_w2recojet.inputs},
        binning=(15, 60, 120),
        unit="GeV",
        x_title=r"$m_{W_{2}}^{reco}$",
    )
    add_variable(
        cfg,
        name="reco_W_mass_avg_coarse",
        expression=partial(build_avg_w_mass),
        aux={"inputs": build_avg_w_mass.inputs},
        binning=(15, 60, 120),
        unit="GeV",
        x_title=r"$m_{W_{avg}}^{reco}$",
    )
    add_variable(
        cfg,
        name="fit_deltaRbb_coarse",
        expression="FitRbb",
        null_value=EMPTY_FLOAT,
        binning=(16, 2, 5.2),
        x_title=r"$\Delta R_{b}$ ",
    )
    add_variable(
        cfg,
        name="ht_coarse",
        expression="ht",
        binning=(32, 400.0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    add_variable(
        cfg,
        name="seljet1_pt_coarse",
        expression="SelectedJets.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 50.0, 340.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet1_eta_coarse",
        expression="SelectedJets.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 1 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet2_pt_coarse",
        expression="SelectedJets.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(20, 50.0, 500.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet2_eta_coarse",
        expression="SelectedJets.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 2 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet3_pt_coarse",
        expression="SelectedJets.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(15, 40.0, 190),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet3_eta_coarse",
        expression="SelectedJets.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 3 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet4_pt_coarse",
        expression="SelectedJets.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(15, 40.0, 140.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet4_eta_coarse",
        expression="SelectedJets.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 4 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet5_pt_coarse",
        expression="SelectedJets.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(16, 35.0, 115.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet5_eta_coarse",
        expression="SelectedJets.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 5 $\eta$",
    )
    add_variable(
        cfg,
        name="seljet6_pt_coarse",
        expression="SelectedJets.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(17, 37.5, 80.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    add_variable(
        cfg,
        name="seljet6_eta_coarse",
        expression="SelectedJets.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(18, -2.7, 2.7),
        x_title=r"Jet 6 $\eta$",
    )
############################################################
#              Observables for b tagging efficiency        #
############################################################
    add_variable(
        cfg,
        name="hadronFlav",
        expression="Jet.hadronFlavour",
        binning=(7, -0.5, 6.5),
        x_title="Hadron flavour",
        discrete_x=True,
    )
    # True flavour bjets
    add_variable(
        cfg,
        name="n_bflav_jet",
        expression="jets_b_num",
        binning=(7, -0.5, 6.5),
        x_title="Number of true b flavoured jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="bflav_jet_pt",
        expression="jets_b.pt",
        binning=[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
        x_title=r"True b flavoured jet $p_{T}$",
        unit="GeV",
    )
    add_variable(
        cfg,
        name="bflav_jet_eta",
        expression="jets_b.eta",
        binning=[0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0],
        x_title=r"True b flavoured jet $\eta$",
    )
    add_variable(
        cfg,
        name="n_btag_bflav_jet",
        expression="bjets_b_num",
        binning=(7, -0.5, 6.5),
        x_title="Number of true b flavoured b-tagged jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="bflav_bjet_pt",
        expression="bjets_b.pt",
        binning=[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
        x_title=r"True b flavoured b-tagged jet $p_{T}$",
        unit="GeV",
    )
    add_variable(
        cfg,
        name="bflav_bjet_eta",
        expression="bjets_b.eta",
        binning=[0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0],
        x_title=r"True b flavoured b-tagged jet $\eta$",
    )
    # True flavour cjets
    add_variable(
        cfg,
        name="n_cflav_jet",
        expression="jets_c_num",
        binning=(7, -0.5, 6.5),
        x_title="Number of true c flavoured jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="cflav_jet_pt",
        expression="jets_c.pt",
        binning=[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
        x_title=r"True c flavoured jet $p_{T}$",
        unit="GeV",
    )
    add_variable(
        cfg,
        name="cflav_jet_eta",
        expression="jets_c.eta",
        binning=[0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0],
        x_title=r"True c flavoured jet $\eta$",
    )
    add_variable(
        cfg,
        name="n_btag_cflav_jet",
        expression="bjets_c_num",
        binning=(7, -0.5, 6.5),
        x_title="Number of true c flavoured b-tagged jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="cflav_bjet_pt",
        expression="bjets_c.pt",
        binning=[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
        x_title=r"True c flavoured b-tagged jet $p_{T}$",
        unit="GeV",
    )
    add_variable(
        cfg,
        name="cflav_bjet_eta",
        expression="bjets_c.eta",
        binning=[0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0],
        x_title=r"True c flavoured b-tagged jet $\eta$",
    )
    # True flavour light jets
    add_variable(
        cfg,
        name="n_light_jet",
        expression="jets_light_num",
        binning=(7, -0.5, 6.5),
        x_title="Number of true light flavoured jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="lightflav_jet_pt",
        expression="jets_light.pt",
        binning=[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
        x_title=r"True light flavoured jet $p_{T}$",
        unit="GeV",
    )
    add_variable(
        cfg,
        name="lightflav_jet_eta",
        expression="jets_light.eta",
        binning=[0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0],
        x_title=r"True light flavoured jet $\eta$",
    )
    add_variable(
        cfg,
        name="n_btag_light_jet",
        expression="bjets_light_num",
        binning=(7, -0.5, 6.5),
        x_title="Number of true light flavoured b-tagged jets",
        discrete_x=True,
    )
    add_variable(
        cfg,
        name="lightflav_bjet_pt",
        expression="bjets_light.pt",
        binning=[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000],
        x_title=r"True light flavoured b-tagged jet $p_{T}$",
        unit="GeV",
    )
    add_variable(
        cfg,
        name="lightflav_bjet_eta",
        expression="bjets_light.eta",
        binning=[0., 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0],
        x_title=r"True light flavoured b-tagged jet $\eta$",
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
