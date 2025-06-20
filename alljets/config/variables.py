# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(cfg: od.Config) -> None:
    # Adds all variables to config
    add_variable(cfg,
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number")
    add_variable(cfg,
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True)
    add_variable(cfg,
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True)
    add_variable(cfg,
        name="n_jet",
        expression="n_jet",
        binning=(16, -0.5, 15.5),
        x_title="Number of jets",
        discrete_x=True)
    add_variable(cfg,
        name="jets_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T}$ of all jets")
    add_variable(cfg,
        name="jets_eta",
        expression="Jet.eta",
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$ of all jets")
    add_variable(cfg,
        name="jets_phi",
        expression="Jet.phi",
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"$\phi$ of all jets")
    add_variable(cfg,
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$")
    add_variable(cfg,
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$")
    add_variable(cfg,
        name="jet1_phi",
        expression="Jet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$")
    add_variable(cfg,
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$")
    add_variable(cfg,
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$")
    add_variable(cfg,
        name="jet2_phi",
        expression="Jet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 2 $\phi$")
    add_variable(cfg,
        name="jet3_pt",
        expression="Jet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$")
    add_variable(cfg,
        name="jet3_eta",
        expression="Jet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 3 $\eta$")
    add_variable(cfg,
        name="jet3_phi",
        expression="Jet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 3 $\phi$")
    add_variable(cfg,
        name="jet4_pt",
        expression="Jet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$")
    add_variable(cfg,
        name="jet4_eta",
        expression="Jet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 4 $\eta$")
    add_variable(cfg,
        name="jet4_phi",
        expression="Jet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 4 $\phi$")
    add_variable(cfg,
        name="jet5_pt",
        expression="Jet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$")
    add_variable(cfg,
        name="jet5_eta",
        expression="Jet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 5 $\eta$")
    add_variable(cfg,
        name="jet5_phi",
        expression="Jet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 5 $\phi$")
    add_variable(cfg,
        name="jet6_pt",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 100.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_1",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 60, 1000],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_2",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 40, 60, float("inf")],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_3",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 40, 60, 1000],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_4",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 26, 32, 38, 44, 50, 60, 80, 100],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_5",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 26, 32, 40, 44, 50, 60, 100],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_6",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 23, 26, 29, 32, 36, 40, 44, 50, 60, 100],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_pt_7",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 23, 26, 29, 32, 36, 40, 44, 50, 60, 150],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_ptdummy",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 9999],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="jet6_eta",
        expression="Jet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 6 $\eta$")
    add_variable(cfg,
        name="jet6_phi",
        expression="Jet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 6 $\phi$")
    add_variable(cfg,
        name="ht",
        expression="ht",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="ht_old",
        expression="ht_old",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="ht3",
        expression="ht",
        binning=[380, 600, 1000, float("inf")],
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="ht2",
        expression="ht",
        binning=[380, 600, 9999],
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="ht7",
        expression="ht",
        binning=[0, 200, 300, 340, 380, 415, 450, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="ht6",
        expression="ht",
        binning=[0, 200, 300, 340, 380, 420, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="ht_dummy",
        expression="ht",
        binning=[380, 99999],
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="trig_ht",
        expression="trig_ht",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$")
    add_variable(cfg,
        name="nPV",
        expression="PV.npvs",
        null_value=EMPTY_FLOAT,
        binning=(60, -0.5, 59.5),
        x_title="Number of primary Vertices")
    add_variable(cfg,
        name="MW1",
        expression="MW1",
        null_value=EMPTY_FLOAT,
        binning=(100, 40, 140),
        unit="GeV",
        x_title=r"$M_{W1}$")
    add_variable(cfg,
        name="MW2",
        expression="MW2",
        null_value=EMPTY_FLOAT,
        binning=(100, 40, 140),
        unit="GeV",
        x_title=r"$M_{W2}$")
    add_variable(cfg,
        name="Mt1",
        expression="Mt1",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$M_{t1}$")
    add_variable(cfg,
        name="Mt1_1",
        expression="Mt1",
        null_value=EMPTY_FLOAT,
        binning=(50, 100, 400),
        unit="GeV",
        x_title=r"$M_{t1}$")
    add_variable(cfg,
        name="Mt2",
        expression="Mt2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$M_{t2}$")
    add_variable(cfg,
        name="deltaMt",
        expression="deltaMt",
        null_value=EMPTY_FLOAT,
        binning=(100, -40, 60),
        unit="GeV",
        x_title=r"$M_{t1} - M_{t2}$")
    add_variable(cfg,
        name="chi2",
        expression="chi2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 200),
        x_title=r"$\chi^2$")
    add_variable(cfg,
        name="chi2_0",
        expression="chi2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 10),
        x_title=r"$\chi^2$")
    add_variable(cfg,
        name="deltaR",
        expression="deltaR",
        null_value=EMPTY_FLOAT,
        binning=(300, -0.005, 2.995),
        x_title=r"min $\Delta R$ of light jets")
    add_variable(cfg,
        name="deltaRb",
        expression="deltaRb",
        null_value=EMPTY_FLOAT,
        binning=(70, 0, 7),
        x_title=r"min $\Delta R$ of b-jets")
    add_variable(cfg,
        name="nPVGood",
        expression="PV.npvsGood",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 60),
        x_title="Number of good primary Vertices")
    add_variable(cfg,
        name="n_bjet",
        expression="n_bjet",
        binning=(6, -0.5, 5.5),
        x_title="Number of Bjets")
    add_variable(cfg,
        name="bjet1_pt",
        expression="Bjet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$")
    add_variable(cfg,
        name="bjet2_pt",
        expression="Bjet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$")
    add_variable(cfg,
        name="bjetbytag1_pt",
        expression="JetsByBTag.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$")
    add_variable(cfg,
        name="bjetbytag2_pt",
        expression="JetsByBTag.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$")
    add_variable(cfg,
        name="bjet1_phi",
        expression="Bjet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"BJet 1 $\phi$")
    add_variable(cfg,
        name="bjet2_phi",
        expression="Bjet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"BJet 2 $\phi$")
    add_variable(cfg,
        name="bjetbytag1_phi",
        expression="JetsByBTag.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"Highest B-Tag Jet $\phi$")
    add_variable(cfg,
        name="bjetbytag2_phi",
        expression="JetsByBTag.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"Second highest B-Tag Jet $\phi$")
    add_variable(cfg,
        name="bjet1_eta",
        expression="Bjet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$")
    add_variable(cfg,
        name="bjet2_eta",
        expression="Bjet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$")
    add_variable(cfg,
        name="bjetbytag1_eta",
        expression="JetsByBTag.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Highest B-Tag Jet $\eta$")
    add_variable(cfg,
        name="bjetbytag2_eta",
        expression="JetsByBTag.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Second highest B-Tag Jet $\eta$")
    add_variable(cfg,
        name="jets_btag",
        expression="Jet.btagDeepFlavB",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"btag scores")
    add_variable(cfg,
        name="maxbtag",
        expression="maxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Highest B-Tag score")
    add_variable(cfg,
        name="secmaxbtag",
        expression="secmaxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Second highest B-Tag score")
    add_variable(cfg,
        name="jet1_btag",
        expression="Jet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 1 bTag")
    add_variable(cfg,
        name="jet2_btag",
        expression="Jet.btagDeepFlavB[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 2 bTag")
    add_variable(cfg,
        name="jet3_btag",
        expression="Jet.btagDeepFlavB[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 3 bTag")
    add_variable(cfg,
        name="jet4_btag",
        expression="Jet.btagDeepFlavB[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 4 bTag")
    add_variable(cfg,
        name="jet5_btag",
        expression="Jet.btagDeepFlavB[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 5 bTag")
    add_variable(cfg,
        name="jet6_btag",
        expression="Jet.btagDeepFlavB[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 6 bTag")
    add_variable(cfg,
        name="combination_type",
        expression="combination_type",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct")
    add_variable(cfg,
        name="R2b4q",
        expression="R2b4q",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 3),
        x_title=r"$R_{2b4q}$")
    # weights
    add_variable(cfg,
        name="mc_weight",
        expression="mc_weight",
        binning=(200, 0, 500),
        x_title="MC weight")
    add_variable(cfg,
        name="btag_weight",
        expression="btag_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title="btag weight")
    add_variable(cfg,
        name="pu_weight",
        expression="pu_weight",
        null_value=EMPTY_FLOAT,
        binning=(60, 0, 1.5),
        x_title="pu weight")
    add_variable(cfg,
        name="murmuf_weight",
        expression="murmuf_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title=r"$\mu_{r}\mu_{f}$ weight")
    add_variable(cfg,
        name="pdf_weight",
        expression="pdf_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title="pdf weight")
    add_variable(cfg,
        name="trig_weight",
        expression="trig_weight",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.7, 1.1),
        x_title="trigger weight")
    # cutflow variables
    add_variable(cfg,
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$")
    add_variable(cfg,
        name="cf_ht",
        expression="cutflow.ht",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$H_{T}$")
    add_variable(cfg,
        name="cf_jet6_pt",
        expression="cutflow.jet6_pt",
        binning=(25, 0.0, 100.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$")
    add_variable(cfg,
        name="cf_n_bjet",
        expression="cutflow.n_bjet",
        binning=(6, -0.5, 5.5),
        unit="GeV",
        x_title=r"Number of Bjets")
    add_variable(cfg,
        name="cf_n_jet",
        expression="cutflow.n_jet",
        binning=(6, -0.5, 5.5),
        unit="GeV",
        x_title=r"Number of Jets")
    add_variable(cfg,
        name="cf_turnon",
        expression="cutflow.turnon",
        binning=(2, -0.5, 1.5),
        x_title=r"0: only in base trigger, 1: In both")
    add_variable(cfg,
        name="cf_combination_type",
        expression="cutflow.combination_type",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct")
    add_variable(cfg,
        name="trig_bits",
        expression="trig_bits",
        binning=(3, -0.5, 2.5),
        x_title=r"trig bits")


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
