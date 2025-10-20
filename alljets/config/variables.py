# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import maybe_import

ak = maybe_import("awkward")


def add_variables(cfg: od.Config) -> None:
    # Adds all variables to config
    cfg.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
    )
    cfg.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    cfg.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    cfg.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    cfg.add_variable(
        name="jets_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T}$ of all jets",
    )
    cfg.add_variable(
        name="jets_eta",
        expression="Jet.eta",
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"$\eta$ of all jets",
    )
    cfg.add_variable(
        name="jets_phi",
        expression="Jet.phi",
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"$\phi$ of all jets",
    )
    cfg.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    cfg.add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    cfg.add_variable(
        name="jet1_phi",
        expression="Jet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    cfg.add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )

    cfg.add_variable(
        name="jet2_eta",
        expression="Jet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 2 $\eta$",
    )
    cfg.add_variable(
        name="jet2_phi",
        expression="Jet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 2 $\phi$",
    )
    cfg.add_variable(
        name="jet3_pt",
        expression="Jet.pt[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 3 $p_{T}$",
    )
    cfg.add_variable(
        name="jet3_eta",
        expression="Jet.eta[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 3 $\eta$",
    )
    cfg.add_variable(
        name="jet3_phi",
        expression="Jet.phi[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 3 $\phi$",
    )
    cfg.add_variable(
        name="jet4_pt",
        expression="Jet.pt[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 4 $p_{T}$",
    )

    cfg.add_variable(
        name="jet4_eta",
        expression="Jet.eta[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 4 $\eta$",
    )
    cfg.add_variable(
        name="jet4_phi",
        expression="Jet.phi[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 4 $\phi$",
    )
    cfg.add_variable(
        name="jet5_pt",
        expression="Jet.pt[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 5 $p_{T}$",
    )

    cfg.add_variable(
        name="jet5_eta",
        expression="Jet.eta[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 5 $\eta$",
    )
    cfg.add_variable(
        name="jet5_phi",
        expression="Jet.phi[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 5 $\phi$",
    )
    cfg.add_variable(
        name="jet6_pt",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 100.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_pt_1",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 60, 1000],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_pt_2",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 40, 60, float("inf")],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_pt_3",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[32, 40, 60, 1000],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_pt_4",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 26, 32, 38, 44, 50, 60, 80, 100],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_pt_5",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 10, 20, 26, 32, 40, 44, 50, 60, 100],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_ptdummy",
        expression="Jet.pt[:,5]",
        null_value=EMPTY_FLOAT,
        binning=[0, 9999],
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="jet6_eta",
        expression="Jet.eta[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 6 $\eta$",
    )
    cfg.add_variable(
        name="jet6_phi",
        expression="Jet.phi[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"Jet 6 $\phi$",
    )
    cfg.add_variable(
        name="ht",
        expression="ht",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="ht3",
        expression="ht",
        binning=[380, 600, 1000, float("inf")],
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="ht2",
        expression="ht",
        binning=[380, 600, 9999],
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="ht7",
        expression="ht",
        binning=[0, 200, 300, 340, 380, 415, 450, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="ht6",
        expression="ht",
        binning=[0, 200, 300, 340, 380, 420, 500, 700, 1000, 1500],
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="ht_dummy",
        expression="ht",
        binning=[380, 99999],
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="trig_ht",
        expression="trig_ht",
        binning=(20, 0, 2000.0),
        unit="GeV",
        x_title="$H_T$",
    )
    cfg.add_variable(
        name="nPV",
        expression="PV.npvs",
        null_value=EMPTY_FLOAT,
        binning=(60, -0.5, 59.5),
        x_title="Number of primary Vertices",
    )
    cfg.add_variable(
        name="MW1",
        expression="MW1",
        null_value=EMPTY_FLOAT,
        binning=(100, 40, 140),
        unit="GeV",
        x_title=r"$M_{W1}$",
    )
    cfg.add_variable(
        name="MW2",
        expression="MW2",
        null_value=EMPTY_FLOAT,
        binning=(100, 40, 140),
        unit="GeV",
        x_title=r"$M_{W2}$",
    )
    cfg.add_variable(
        name="Mt1",
        expression="Mt1",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$M_{t1}$",
    )
    cfg.add_variable(
        name="Mt1_1",
        expression="Mt1",
        null_value=EMPTY_FLOAT,
        binning=(40, 100, 500),
        unit="GeV",
        x_title=r"$M_{t1}$",
    )
    cfg.add_variable(
        name="Mt2",
        expression="Mt2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 500),
        unit="GeV",
        x_title=r"$M_{t2}$",
    )
    cfg.add_variable(
        name="chi2",
        expression="chi2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 200),
        x_title=r"$\chi^2$",
    )
    cfg.add_variable(
        name="chi2_0",
        expression="chi2",
        null_value=EMPTY_FLOAT,
        binning=(100, 0, 10),
        x_title=r"$\chi^2$",
    )
    cfg.add_variable(
        name="deltaR",
        expression="deltaR",
        null_value=EMPTY_FLOAT,
        binning=(300, -0.005, 2.995),
        x_title=r"min $\Delta R$ of light jets",
    )
    cfg.add_variable(
        name="deltaRb",
        expression="deltaRb",
        null_value=EMPTY_FLOAT,
        binning=(70, 0, 7),
        x_title=r"min $\Delta R$ of b-jets",
    )
    cfg.add_variable(
        name="nPVGood",
        expression="PV.npvsGood",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 60),
        x_title="Number of good primary Vertices",
    )
    cfg.add_variable(
        name="n_bjet",
        expression="n_bjet",
        binning=(6, -0.5, 5.5),
        x_title="Number of Bjets",
    )
    cfg.add_variable(
        name="bjet1_pt",
        expression="Bjet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    cfg.add_variable(
        name="bjet2_pt",
        expression="Bjet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    cfg.add_variable(
        name="bjetbytag1_pt",
        expression="JetsByBTag.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    cfg.add_variable(
        name="bjetbytag2_pt",
        expression="JetsByBTag.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(15, 0.0, 300.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    cfg.add_variable(
        name="bjet1_phi",
        expression="Bjet.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"BJet 1 $\phi$",
    )
    cfg.add_variable(
        name="bjet2_phi",
        expression="Bjet.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"BJet 2 $\phi$",
    )
    cfg.add_variable(
        name="bjetbytag1_phi",
        expression="JetsByBTag.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"Highest B-Tag Jet $\phi$",
    )
    cfg.add_variable(
        name="bjetbytag2_phi",
        expression="JetsByBTag.phi[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        unit="GeV",
        x_title=r"Second highest B-Tag Jet $\phi$",
    )
    cfg.add_variable(
        name="bjet1_eta",
        expression="Bjet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"BJet 1 $p_{T}$",
    )
    cfg.add_variable(
        name="bjet2_eta",
        expression="Bjet.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"BJet 2 $p_{T}$",
    )
    cfg.add_variable(
        name="bjetbytag1_eta",
        expression="JetsByBTag.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Highest B-Tag Jet $\eta$",
    )
    cfg.add_variable(
        name="bjetbytag2_eta",
        expression="JetsByBTag.eta[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        unit="GeV",
        x_title=r"Second highest B-Tag Jet $\eta$",
    )
    cfg.add_variable(
        name="jets_btag",
        expression="Jet.btagDeepFlavB",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"btag scores",
    )
    cfg.add_variable(
        name="maxbtag",
        expression="maxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Highest B-Tag score",
    )
    cfg.add_variable(
        name="secmaxbtag",
        expression="secmaxbtag",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Second highest B-Tag score",
    )
    cfg.add_variable(
        name="jet1_btag",
        expression="Jet.btagDeepFlavB[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 1 bTag",
    )
    cfg.add_variable(
        name="jet2_btag",
        expression="Jet.btagDeepFlavB[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 2 bTag",
    )
    cfg.add_variable(
        name="jet3_btag",
        expression="Jet.btagDeepFlavB[:,2]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 3 bTag",
    )
    cfg.add_variable(
        name="jet4_btag",
        expression="Jet.btagDeepFlavB[:,3]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 4 bTag",
    )
    cfg.add_variable(
        name="jet5_btag",
        expression="Jet.btagDeepFlavB[:,4]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 5 bTag",
    )
    cfg.add_variable(
        name="jet6_btag",
        expression="Jet.btagDeepFlavB[:,5]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 1),
        x_title=r"Jet 6 bTag",
    )
    cfg.add_variable(
        name="combination_type",
        expression="combination_type",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct",
    )
    cfg.add_variable(
        name="R2b4q",
        expression="R2b4q",
        null_value=EMPTY_FLOAT,
        binning=(30, 0, 3),
        x_title=r"$R_{2b4q}$",
    )
    # weights
    cfg.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, 0, 500),
        x_title="MC weight",
    )
    cfg.add_variable(
        name="btag_weight",
        expression="btag_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title="btag weight",
    )
    cfg.add_variable(
        name="pu_weight",
        expression="pu_weight",
        null_value=EMPTY_FLOAT,
        binning=(60, 0, 1.5),
        x_title="pu weight",
    )
    cfg.add_variable(
        name="murmuf_weight",
        expression="murmuf_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title=r"$\mu_{r}\mu_{f}$ weight",
    )
    cfg.add_variable(
        name="pdf_weight",
        expression="pdf_weight",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 2),
        x_title="pdf weight",
    )
    cfg.add_variable(
        name="trig_weight",
        expression="trig_weight",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.7, 1.1),
        x_title="trigger weight",
    )
    # cutflow variables
    cfg.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    cfg.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    cfg.add_variable(
        name="cf_jet6_pt",
        expression="cutflow.jet6_pt",
        binning=(25, 0.0, 100.0),
        unit="GeV",
        x_title=r"Jet 6 $p_{T}$",
    )
    cfg.add_variable(
        name="cf_n_bjet",
        expression="cutflow.n_bjet",
        binning=(6, -0.5, 5.5),
        unit="GeV",
        x_title=r"Number of Bjets",
    )
    cfg.add_variable(
        name="cf_n_jet",
        expression="cutflow.n_jet",
        binning=(6, -0.5, 5.5),
        unit="GeV",
        x_title=r"Number of Jets",
    )
    cfg.add_variable(
        name="cf_turnon",
        expression="cutflow.turnon",
        binning=(2, -0.5, 1.5),
        x_title=r"0: only in base trigger, 1: In both",
    )
    cfg.add_variable(
        name="cf_combination_type",
        expression="cutflow.combination_type",
        null_value=EMPTY_FLOAT,
        binning=(4, -1.5, 2.5),
        x_title=r"Combination types: -1: NA 0: unmatched, 1: wrong, 2: correct",
    )
    cfg.add_variable(
        name="trig_bits",
        expression="trig_bits",
        binning=(3, -0.5, 2.5),
        x_title=r"trig bits",
    )
    # GenParticals and GenJets
    cfg.add_variable(
        name="GenPart_eta",
        expression="GenPart.eta",
        binning=(30, -3.0, 3.0),
        unit="",
        x_title=r"$\eta$ of all GenParticals",
    )
    cfg.add_variable(
        name="GenJet_phi",
        expression="GenJet.phi",
        binning=(30, -3.0, 3.0),
        unit="",
        x_title=r"$\phi$ of all GenJets",
    )
    cfg.add_variable(
        name="GenJet_pt",
        expression="GenJet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of all GenJets",
    )
    cfg.add_variable(
        name="GenJet_mass",
        expression="GenJet.mass",
        binning=(30, 0, 100),
        unit="GeV",
        x_title=r"$\phi$ of all GenJets",
    )
    cfg.add_variable(
        name="GenPart_MotherID",
        expression="GenPart.genPartIdxMother",
        binning=(150, 0, 150),
        unit="",
        x_title=r"ID of the Motherpartical",
    )
    cfg.add_variable(
        name="GenPart_pdgID",
        expression="GenPart.pdgId",
        binning=(100, -50, 50),
        unit="",
        x_title=r"PDG ID of the GenParticals",
    )
    cfg.add_variable(
        name="GenPart_MotherpdgID",
        expression="GenPart_pdgIdMother",
        binning=(100, -50, 50),
        unit="",
        x_title=r"PDG ID of the Motherpartical",
    )
    cfg.add_variable(
        name="n_MotherTop10",
        expression="n_MotherTop",
        binning=(15, 0, 15),
        unit="",
        x_title=r"Number of GenPart wiht a t-quark as Mother",
    )
    cfg.add_variable(
        name="n_top",
        expression="n_top",
        binning=(10, 0, 10),
        unit="",
        x_title=r"Number of t-quark in a Event",
    )
    # gen t1
    cfg.add_variable(
        name="gen_top1_mass",
        expression="gen_top_decay.mass[:,0,0]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_t^\text{gen}$",
    )
    cfg.add_variable(
        name="gen_top1_pt",
        expression="gen_top_decay.pt[:,0,0]",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$p_{T,t}$",
    )
    cfg.add_variable(
        name="gen_top1_p",
        expression="gen_top_p[:,0,0]",
        binning=(40, 0.0, 700.0),
        unit="GeV",
        x_title=r"$p_{T,t}$",
    )
    cfg.add_variable(
        name="gen_top1_mass_Q",
        expression="top_family.mass[:,[0],0]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_{T,t}$",
    )
    # gen t2
    cfg.add_variable(
        name="gen_top2_mass",
        expression="gen_top_decay.mass[:,1,0]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_\overline{t}^\text{gen}$",
    )
    cfg.add_variable(
        name="gen_top2_pt",
        expression="gen_top_decay.pt[:,1,0]",
        binning=(50, 0.0, 1000.0),
        unit="GeV",
        x_title=r"$p_{T,\overline{t}}$",
    )
    # gen b
    cfg.add_variable(
        name="gen_b_mass",
        expression="gen_top_decay.mass[:,:,1]",
        binning=(40, 0, 40),
        unit="GeV",
        x_title=r"Mass of the gen-b-Quark",
    )
    cfg.add_variable(
        name="gen_b1_pt",
        expression="gen_top_decay.pt[:,0,1]",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_T$ of the gen-b-Quark",
    )
    cfg.add_variable(
        name="gen_b1_eta",
        expression="gen_top_decay.eta[:,0,1]",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of the gen-b-Quark",
    )
    # gen W
    cfg.add_variable(
        name="gen_W1_mass",
        expression="gen_top_decay.mass[:,0,2]",
        binning=(40, 65, 95),
        unit="GeV",
        x_title=r"$m_{W^+}^\text{gen}$",
    )
    cfg.add_variable(
        name="gen_W2_mass",
        expression="gen_top_decay.mass[:,1,2]",
        binning=(40, 65, 95),
        unit="GeV",
        x_title=r"$m_{W^-}^\text{gen}$",
    )
    cfg.add_variable(
        name="gen_W1_pt",
        expression="gen_top_decay.pt[:,0,2]",
        binning=(40, 0, 500),
        unit="GeV",
        x_title=r"$p_{T,W^+}^\text{gen}$",
    )
    cfg.add_variable(
        name="gen_W2_pt",
        expression="gen_top_decay.pt[:,1,2]",
        binning=(40, 0, 500),
        unit="GeV",
        x_title=r"$p_{T,W^-}^\text{gen}$",
    )
    # gen q1 and q2
    cfg.add_variable(
        name="gen_q1_pt",
        expression="gen_top_decay.pt[:,:,3]",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_T$ of the gen-quark 1",
    )
    cfg.add_variable(
        name="gen_q2_pt",
        expression="gen_top_decay.pt[:,:,4]",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_T$ of the gen-quark 2",
    )
    # Delta R
    cfg.add_variable(
        name="gen_top_deltaR",
        expression="gen_top_deltaR",
        binning=(40, 0, 6),
        unit="",
        x_title=r"$\Delta R_{t\overline{t}}$",
    )
    cfg.add_variable(
        name="gen_b_deltaR",
        expression="gen_b_deltaR",
        binning=(40, 0, 6),
        unit="",
        x_title=r"$\Delta R_{b\overline{b}}$",
    )
    cfg.add_variable(
        name="gen_q1q2_deltaR",
        expression="gen_q1q2_deltaR[:,0]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{q1q2}$",
    )
    cfg.add_variable(
        name="gen_q3q4_deltaR",
        expression="gen_q1q2_deltaR[:,1]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{q3q4}$",
    )
    cfg.add_variable(
        name="gen_b1W1_deltaR",
        expression="gen_bW_deltaR[:,0]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{bW^+}$",
    )
    cfg.add_variable(
        name="gen_b2W2_deltaR",
        expression="gen_bW_deltaR[:,1]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{\overline{b}W^-}$",
    )
    cfg.add_variable(
        name="gen_top1_max_deltaR",
        expression="gen_max_deltaR[:,0]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{\text{max},t}$",
    )
    cfg.add_variable(
        name="gen_top2_max_deltaR",
        expression="gen_max_deltaR[:,1]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{\text{max},\overline{t}}$",
    )
    cfg.add_variable(
        name="gen_W1q1_deltaR",
        expression="gen_Wq1_deltaR[:,0]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{W^+q1}$",
    )
    cfg.add_variable(
        name="gen_top1_min_deltaR",
        expression="gen_min_deltaR[:,0]",
        binning=(40, 0, 4),
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_top2_min_deltaR",
        expression="gen_min_deltaR[:,1]",
        binning=(40, 0, 4),
        unit="",
        x_title=r"$\Delta R_{\text{min},\overline{t}}$",
    )
    # gen m_reco
    cfg.add_variable(
        name="reco_mt1_bW",
        expression="reco_mt_bW[:,0]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_\text{t}^{bW^+}$",
    )
    cfg.add_variable(
        name="reco_mt2_bW",
        expression="reco_mt_bW[:,1]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_{\overline{t}}^{\overline{b}W^-}$",
    )
    cfg.add_variable(
        name="reco_mW1_q1q2",
        expression="reco_mW_q1q2[:,0]",
        binning=(40, 65, 95),
        unit="GeV",
        # x_title=r"$m_{W^+}^{q1q2}$",
        x_title=r"$m_{W^+}$ reconstructed with $q_1$ and $q_2$",
    )
    cfg.add_variable(
        name="reco_mW2_q3q4",
        expression="reco_mW_q1q2[:,1]",
        binning=(40, 65, 95),
        unit="GeV",
        x_title=r"$m_{W^-}^{q3q4}$",
    )
    cfg.add_variable(
        name="reco_mt1_q1q2b",
        expression="reco_mt_q1q2b[:,0]",
        binning=(40, 150, 190),
        unit="GeV",
        # x_title=r"$m_\text{t}^{q1q2b}$",
        x_title=r"$m_t$ reconstructed with $b_1$, $q_1$ and $q_2$",
    )
    cfg.add_variable(
        name="reco_mt2_q3q4b",
        expression="reco_mt_q1q2b[:,1]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_{\overline{t}}^{q3q4\overline{b}}$",
    )
    cfg.add_variable(
        name="reco_mt1_q1q2b_Q",
        expression="reco_mt_q1q2b_Q[:,0,0]",
        binning=(40, 150, 190),
        unit="GeV",
        x_title=r"$m_{\overline{t}}^{q3q4\overline{b}}$",
    )
    # gen reco pt
    cfg.add_variable(
        name="reco_pt_t1_bW",
        expression="reco_pt_t_bW[:,0]",
        binning=(40, 0.0, 700.0),
        unit="GeV",
        x_title=r"$p_\text{T,t}^{bW^+}$",
    )
    cfg.add_variable(
        name="reco_pt_t2_bW",
        expression="reco_pt_t_bW[:,1]",
        binning=(40, 0.0, 700.0),
        unit="GeV",
        x_title=r"$p_{T,\overline{t}}^{\overline{b}W^-}$",
    )
    cfg.add_variable(
        name="reco_pt_W1_q1q2",
        expression="reco_pt_W_q1q2[:,0]",
        binning=(40, 0, 500),
        unit="GeV",
        x_title=r"$p_{T,W^+}^{q1q2}$",
    )
    cfg.add_variable(
        name="reco_pt_W2_q3q4",
        expression="reco_pt_W_q1q2[:,1]",
        binning=(40, 0, 500),
        unit="GeV",
        x_title=r"$p_{T,W^-}^{q3q4}$",
    )
    cfg.add_variable(
        name="reco_pt_t_q1q2b",
        expression="reco_pt_t_q1q2b[:,0]",
        binning=(40, 0.0, 700.0),
        unit="GeV",
        x_title=r"$p_{T,\overline{t}}^{q1q2b}$",
    )
    cfg.add_variable(
        name="reco_pt_t_q3q4b",
        expression="reco_pt_t_q1q2b[:,1]",
        binning=(40, 0.0, 700.0),
        unit="GeV",
        x_title=r"$p_{T,\overline{t}}^{q3q4\overline{b}}$",
    )
    cfg.add_variable(
        name="reco_p_t_q1q2b",
        expression="reco_p_t_q1q2b[:,0]",
        binning=(40, 0.0, 700.0),
        unit="GeV",
        x_title=r"$p_{t}^{q1q2b}$",
    )
    # Gen mass diff
    cfg.add_variable(
        name="diff_mt1_bW",
        expression="diff_mt_bW[:,0]",
        binning=(80, -2.0, 2.0),
        unit="GeV",
        x_title=r"$ m_t^{gen}-m_t^{bW^+}$",
    )
    cfg.add_variable(
        name="diff_mt1_bW_last_copy",
        expression="diff_mt_bW_last_copy[:,0]",
        binning=(80, -2.0, 2.0),
        unit="GeV",
        x_title=r"$ m_t^{gen}-m_t^{bW^+}$",
    )
    cfg.add_variable(
        name="diff_mt1_bW_with_bmass",
        expression="diff_mt_bW_with_b_mass[:,0]",
        binning=(80, -2.0, 2.0),
        unit="GeV",
        x_title=r"$ m_t^{gen}-m_t^{bW^+}$",
    )
    cfg.add_variable(
        name="diff_mt2_bW",
        expression="diff_mt_bW[:,1]",
        binning=(80, -2.0, 2.0),
        unit="GeV",
        x_title=r"$ m_\overline{t}^{gen}-m_\overline{t}^{\overline{b}W^-}$",
    )
    cfg.add_variable(
        name="diff_mW1_q1q2",
        expression="diff_mW_q1q2[:,0]",
        binning=(80, -2.0, 2.0),
        unit="GeV",
        x_title=r"$ m_W^{+, gen}-m_W^{+,q1q2}$",
    )
    cfg.add_variable(
        name="diff_mW2_q1q2",
        expression="diff_mW_q1q2[:,1]",
        binning=(80, -2.0, 2.0),
        unit="GeV",
        x_title=r"$ m_W^{-,gen}-m_W^{-,q3q4}$",
    )
    cfg.add_variable(
        name="diff_mt1_q1q2b",
        expression="diff_mt_q1q2b[:,0]",
        binning=(100, -3.0, 8.0),
        unit="GeV",
        x_title=r"$ m_t^{gen}-m_t^{bq1q2}$",
    )
    cfg.add_variable(
        name="diff_mt1_q1q2b_last_copy",
        expression="diff_mt_q1q2b_last_copy[:,0]",
        binning=(100, -3.0, 8.0),
        unit="GeV",
        x_title=r"$ m_t^{gen}-m_t^{bq1q2}$",
    )
    cfg.add_variable(
        name="diff_mt2_q3q4b",
        expression="diff_mt_q1q2b[:,0]",
        binning=(100, -3.0, 8.0),
        unit="GeV",
        x_title=r"$ m_\overline{t}^{gen}-m_\overline{t}^{\overline{b}q3q4}$",
    )
    # p diff
    cfg.add_variable(
        name="diff_p_t1_bW",
        expression="diff_p_t_bW[:,0]",
        binning=(80, -10.0, 10.0),
        unit="GeV",
        x_title=r"$ p_t^{gen}-p_t^{bW^+}$",
    )
    cfg.add_variable(
        name="diff_p_t2_bW",
        expression="diff_p_t_bW[:,1]",
        binning=(80, -10.0, 10.0),
        unit="GeV",
        x_title=r"$ p_\overline{t}^{gen}-p_\overline{t}^{\overline{b}W^-}$",
    )
    cfg.add_variable(
        name="diff_p_W1_q1q2",
        expression="diff_p_W_q1q2[:,0]",
        binning=(80, -10.0, 10.0),
        unit="GeV",
        x_title=r"$ p_{W^+}^{gen}-p_{W^+}^{q1q2}$",
    )
    cfg.add_variable(
        name="diff_p_W2_q1q2",
        expression="diff_p_W_q1q2[:,1]",
        binning=(80, -10.0, 10.0),
        unit="GeV",
        x_title=r"$ p_{W^-}^{gen}-p_{W^-}^{q3q4}$",
    )
    cfg.add_variable(
        name="diff_p_t1_q1q2b",
        expression="diff_p_t_q1q2b[:,0]",
        binning=(100, -5.0, 10.0),
        unit="GeV",
        x_title=r"$ p_t^{gen}-p_t^{bq1q2}$",
    )
    cfg.add_variable(
        name="diff_p_t2_q3q4b",
        expression="diff_p_t_q1q2b[:,0]",
        binning=(100, -5.0, 10.0),
        unit="GeV",
        x_title=r"$ p_\overline{t}^{gen}-p_\overline{t}^{\overline{b}q3q4}$",
    )
    cfg.add_variable(
        name="b1_mass",
        expression="b_mass[:,0]",
        binning=(100, -15.0, 15.0),
        unit="GeV",
        x_title=r"$m_b$",
    )
    # delta_fractions
    cfg.add_variable(
        name="deltaR_fraction_04",
        expression="deltaR_fraction_04",
        binning=(10, 0.0, 10.0),
        unit="",
        x_title=r"Fraction of objects with ΔR < 0.4",
    )
    cfg.add_variable(
        name="gen_top1_min_deltaR_3bins",
        expression="gen_min_deltaR[:,0]",
        binning=[0, 0.4, 1, 6],
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_top1_min_deltaR_2bins",
        expression="gen_min_deltaR[:,0]",
        binning=[0, 1, 6],
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_top1_min_deltaR_2bins_04",
        expression="gen_min_deltaR[:,0]",
        binning=[0, 0.4, 6],
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_top1_min_deltaR_2bins_08",
        expression="gen_min_deltaR[:,0]",
        binning=[0, 0.8, 6],
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_top1_min_deltaR_4bins",
        expression="gen_min_deltaR[:,0]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_top1_max_deltaR_4bins",
        expression="gen_max_deltaR[:,0]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\text{max},t}$",
    )
    cfg.add_variable(
        name="gen_b1W1_deltaR_4bins",
        expression="gen_bW_deltaR[:,0]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\text{bW^+},t}$",
    )
    cfg.add_variable(
        name="gen_b2W2_deltaR_4bins",
        expression="gen_bW_deltaR[:,1]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\overline{b}W^-,t}$",
    )
    cfg.add_variable(
        name="gen_q1q2_deltaR_4bins",
        expression="gen_q1q2_deltaR[:,0]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{q1q2,t}$",
    )
    cfg.add_variable(
        name="gen_q3q4_deltaR_4bins",
        expression="gen_q1q2_deltaR[:,1]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="diff_min_deltaR",
        expression="diff_min_deltaR[:,0]",
        binning=(100, -1.0, 1.0),
        unit="",
        x_title=r"Diff $\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="diff_top_deltaR",
        expression="diff_top_deltaR[:,0]",
        binning=(100, -1.0, 1.0),
        unit="",
        x_title=r"Diff $\Delta R_{\text{min},t}$",
    )
    cfg.add_variable(
        name="gen_q1_q3_deltaR",
        expression="gen_q1_q3_deltaR",
        binning=(40, 0, 6),
        unit="",
        x_title=r"$\Delta R_{q1q3}$",
    )
    cfg.add_variable(
        name="gen_q1_q3_deltaR_4bins",
        expression="gen_q1_q3_deltaR",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{q1q3}$",
    )
    cfg.add_variable(
        name="gen_min_deltaR_combinations",
        expression="gen_min_deltaR_combinations",
        binning=[0, 0.9, 9, 99, 999, 9999, 99999, 100001],
        unit="",
        x_title=r"",
    )
    cfg.add_variable(
        name="gen_min_deltaR1_combinations",
        expression="gen_min_deltaR_combinations",
        binning=[0, 0.9, 9, 10.1, 99],
        unit="",
        x_title=r"",
    )
    cfg.add_variable(
        name="gen_min_deltaR1_one_t",
        expression="gen_min_deltaR1_one_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 1.0$",
    )
    cfg.add_variable(
        name="gen_min_deltaR1_tt",
        expression="gen_min_deltaR1_tt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 1.0$",
    )
    cfg.add_variable(
        name="gen_min_deltaR08_one_t",
        expression="gen_min_deltaR08_one_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 0.8$",
    )
    cfg.add_variable(
        name="gen_min_deltaR08_tt",
        expression="gen_min_deltaR08_tt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 0.8$",
    )
    cfg.add_variable(
        name="gen_min_deltaR04_one_t",
        expression="gen_min_deltaR04_one_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 0.4$",
    )
    cfg.add_variable(
        name="gen_max_deltaR04_one_t",
        expression="gen_max_deltaR04_one_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"Only one top: $\Delta R_{\text{max}} < 0.4$",
    )
    cfg.add_variable(
        name="gen_max_deltaR08_one_t",
        expression="gen_max_deltaR08_one_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"Only one top: $\Delta R_{\text{max}} < 0.8$",
    )
    cfg.add_variable(
        name="gen_max_deltaR1_one_t",
        expression="gen_max_deltaR1_one_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"Only one top: $\Delta R_{\text{max}} < 1.0$",
    )
    cfg.add_variable(
        name="gen_max_deltaR04_tt",
        expression="gen_max_deltaR04_tt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{max},t\overline{t}} < 0.4$",
    )
    cfg.add_variable(
        name="gen_max_deltaR08_tt",
        expression="gen_max_deltaR08_tt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{max},t\overline{t}} < 0.8$",
    )
    cfg.add_variable(
        name="gen_max_deltaR1_tt",
        expression="gen_max_deltaR1_tt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{max},t\overline{t}} < 1.0$",
    )
    cfg.add_variable(
        name="gen_min_deltaR04_tt",
        expression="gen_min_deltaR04_tt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 0.4$",
    )
    cfg.add_variable(
        name="gen_top2_max_deltaR_4bins",
        expression="gen_max_deltaR[:,1]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\text{max},\overline{t}}$",
    )
    cfg.add_variable(
        name="gen_top2_min_deltaR_4bins",
        expression="gen_min_deltaR[:,1]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\text{min},\overline{t}}$",
    )
    cfg.add_variable(
        name="gen_b1q1_deltaR_4bins",
        expression="gen_bq1_deltaR[:,0]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{bq1}$",
    )
    cfg.add_variable(
        name="gen_b2q3_deltaR_4bins",
        expression="gen_bq1_deltaR[:,1]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\overline{b}q3}$",
    )
    cfg.add_variable(
        name="gen_b1q2_deltaR_4bins",
        expression="gen_bq2_deltaR[:,0]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{bq2}$",
    )
    cfg.add_variable(
        name="gen_b2q34_deltaR_4bins",
        expression="gen_bq2_deltaR[:,1]",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{\overline{b}q4}$",
    )
    cfg.add_variable(
        name="gen_b1q_min_deltaR_4bins",
        expression="gen_b1q_min_deltaR",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{min,bq}$",
    )
    cfg.add_variable(
        name="gen_tt_min_deltaR_4bins",
        expression="gen_tt_min_deltaR",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{min,tt}$",
    )
    cfg.add_variable(
        name="gen_tt_min_deltaR",
        expression="gen_tt_min_deltaR",
        binning=(100, 0, 6),
        unit="",
        x_title=r"$\Delta R_{min,tt}$",
    )
    cfg.add_variable(
        name="gen_max_deltaR_tt_4bins",
        expression="gen_max_deltaR_tt",
        binning=[0, 0.5, 1.1, 2.2, 4],
        unit="",
        x_title=r"$\Delta R_{max}$",
    )
    cfg.add_variable(
        name="gen_max_deltaR_t_4bins",
        expression="gen_max_deltaR_t",
        binning=[0, 0.5, 1.1, 2.2, 4],
        unit="",
        x_title=r"$\Delta R_{max}$",
    )
    cfg.add_variable(
        name="gen_min_deltaR1_t",
        expression="gen_min_deltaR1_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 1.0$",
    )
    cfg.add_variable(
        name="gen_min_deltaR08_t",
        expression="gen_min_deltaR08_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 0.8$",
    )
    cfg.add_variable(
        name="gen_min_deltaR04_t",
        expression="gen_min_deltaR04_t",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$\Delta R_{\text{min}} < 0.4$",
    )
    cfg.add_variable(
        name="gen_min_deltaR_tt_4bins",
        expression="gen_min_deltaR_tt",
        binning=[0, 0.5, 1.1, 2.2, 4],
        unit="",
        x_title=r"$\Delta R_{min,tt}$",
    )
    cfg.add_variable(
        name="gen_min_deltaR_t_4bins",
        expression="gen_min_deltaR_t",
        binning=[0, 0.5, 1.1, 2.2, 4],
        unit="",
        x_title=r"$\Delta R_{min,t}$",
    )
    cfg.add_variable(
        name="number_boosted_tops_with_dR1",
        expression="number_boosted_tops_with_dR1",
        binning=[0, 0.5, 1.1, 2.2],
        unit="",
        x_title=r"Number boosted tops with $\Delta R_{max} < 1$",
    )
    cfg.add_variable(
        name="number_boosted_tops_with_dR08",
        expression="number_boosted_tops_with_dR08",
        binning=[0, 0.5, 1.1, 2.2],
        unit="",
        x_title=r"Number boosted tops with $\Delta R_{max} < 0.8$",
    )
    cfg.add_variable(
        name="number_boosted_tops_with_dR04",
        expression="number_boosted_tops_with_dR04",
        binning=[0, 0.5, 1.1, 2.2],
        unit="",
        x_title=r"Number boosted tops with $\Delta R_{max} < 0.4$",
    )
    cfg.add_variable(
        name="genJet_top1_min_deltaR_4bins",
        expression="genJet_min_deltaR_top1",
        binning=[0, 0.4, 0.8, 1, 20],
        unit="",
        x_title=r"$\Delta R_{min,t}$",
    )
    cfg.add_variable(
        name="genJet_min_deltaR_t_4bins",
        expression="genJet_min_deltaR_t",
        binning=[0, 0.5, 1.1, 2.2, 4],
        unit="",
        x_title=r"$\Delta R_{min,t}$",
    )
    cfg.add_variable(
        name="genJet_matched",
        expression="genJet_matched",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"matched events",
    )
    cfg.add_variable(
        name="genJet_matched_wrong",
        expression="genJet_matched_wrong",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"False assigned events",
    )
    cfg.add_variable(
        name="genJet_unmatched",
        expression="genJet_unmatched",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"unmatched events",
    )
    cfg.add_variable(
        name="genJet_overlap_b1q12",
        expression="genJet_overlap_b1q12",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"Number of matched events",
    )
    cfg.add_variable(
        name="genJet_overlap_b2q34",
        expression="genJet_overlap_b2q34",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"Number of matched events",
    )
    cfg.add_variable(
        name="genJet_overlap_q12",
        expression="genJet_overlap_q12",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"Number of matched events",
    )
    cfg.add_variable(
        name="genJet_overlap_q34",
        expression="genJet_overlap_q34",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"Number of matched events",
    )
    cfg.add_variable(
        name="genJet_b1_min_deltaR",
        expression="genJet_b1_min_deltaR",
        binning=(40, 0, 0.5),
        unit="",
        x_title=r"$\Delta R_{min}$",
    )
    cfg.add_variable(
        name="genJet_b1_eta",
        expression="genJet_b1.eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of the genJet-b-Quark",
    )
    cfg.add_variable(
        name="genJet_min_pt",
        expression="genJet_min_pt",
        binning=(40, 0.0, 50),
        unit="GeV",
        x_title=r"$p_{min,T}$ of all Gen-jets",
    )
    cfg.add_variable(
        name="genJet_max_pt",
        expression="genJet_max_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{max,T}$ of all Gen-jets",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_b1_pt",
        expression="gen_top_decay_unmatched_b1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of unmatched b1",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_b2_pt",
        expression="gen_top_decay_unmatched_b2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of unmatched b2",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q1_pt",
        expression="gen_top_decay_unmatched_q1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of unmatched q1",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q2_pt",
        expression="gen_top_decay_unmatched_q2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of unmatched q2",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q3_pt",
        expression="gen_top_decay_unmatched_q3_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of unmatched q3",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q4_pt",
        expression="gen_top_decay_unmatched_q4_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_T$ of unmatched q4",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_b1_eta",
        expression="gen_top_decay_unmatched_b1_eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of unmatched b1",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_b2_eta",
        expression="gen_top_decay_unmatched_b2_eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of unmatched b2",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q1_eta",
        expression="gen_top_decay_unmatched_q1_eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of unmatched q1",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q2_eta",
        expression="gen_top_decay_unmatched_q2_eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of unmatched q2",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q3_eta",
        expression="gen_top_decay_unmatched_q3_eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of unmatched q3",
    )
    cfg.add_variable(
        name="gen_top_decay_unmatched_q4_eta",
        expression="gen_top_decay_unmatched_q4_eta",
        binning=(40, -5, 5),
        unit="",
        x_title=r"$\eta$ of unmatched q4",
    )
    cfg.add_variable(
        name="genJet_unmatched_b1",
        expression="genJet_unmatched_b1",
        binning=[0, 0.9, 1.8],
        unit="",
        x_title=r"unmatched b1",
    )
    cfg.add_variable(
        name="gen_b1q1_deltaR",
        expression="gen_b1q1_deltaR[:,0]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{b1q1}$",
    )
    cfg.add_variable(
        name="gen_b2q3_deltaR",
        expression="gen_b1q1_deltaR[:,1]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{b2q3}$",
    )
    cfg.add_variable(
        name="gen_b1q2_deltaR",
        expression="gen_b1q2_deltaR[:,0]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{b1q2}$",
    )
    cfg.add_variable(
        name="gen_b2q4_deltaR",
        expression="gen_b1q2_deltaR[:,1]",
        binning=(40, 0, 5),
        unit="",
        x_title=r"$\Delta R_{b2q4}$",
    )
    cfg.add_variable(
        name="gen_min_deltaR_one_t_4bins",
        expression="gen_min_deltaR_one_t",
        binning=[0, 0.5, 1.1, 2.2, 4],
        unit="",
        x_title=r"$\Delta R_{min,t}$",
    )
    cfg.add_variable(
        name="pt_diff",
        expression="pt_diff",
        binning=(50, -0.5, 0.5),
        unit="",
        x_title=r"$\frac{p_{T, t} - p_{T, \overline{t}}}{p_{T}}$",
    )
    cfg.add_variable(
        name="pt_UE_s01",
        expression="pt_UE_s01",
        binning=(100, 0, 250),
        unit="",
        x_title=r"$p_{T,max}$ of the particles with motherId -1 $|\frac{\Delta p_T}{p_{T,t}}| < 0.1$",
    )
    cfg.add_variable(
        name="pt_UE_s09",
        expression="pt_UE_s09",
        binning=(100, 0, 250),
        unit="",
        x_title=r"$p_{T,max}$ of the particles with motherId -1 $|\frac{\Delta p_T}{p_{T,t}}| > 0.9$",
    )
    cfg.add_variable(
        name="pt_UE",
        expression="pt_UE",
        binning=(100, 0, 1000),
        unit="",
        x_title=r"$p_{T,max}$ of the particles with motherId -1 ",
    )
    cfg.add_variable(
        name="gen_ttbar_pt",
        expression="gen_ttbar_pt",
        binning=(100, 0, 1000),
        unit="",
        x_title=r"$p_{T}$ of the t$\overline{t}$ system",
    )
    cfg.add_variable(
        name="gen_ttbar_phi",
        expression="gen_ttbar_phi",
        binning=(100, 0, 3.5),
        unit="",
        x_title=r"$\Delta \phi$ between t and $\overline{t}$",
    )
    cfg.add_variable(
        name="ISPS_pt_vektorial",
        expression="ISPS_pt_vektorial",
        binning=(100, 0, 300),
        unit="GeV",
        x_title=r"vectorial sum of the $p_{T}$ of the particles with motherId -1",
    )
    cfg.add_variable(
        name="partially_overlap",
        expression="partially_overlap",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"events with partial overlapping jets",
    )
    cfg.add_variable(
        name="resolved",
        expression="resolved",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"resolved events",
    )
    cfg.add_variable(
        name="gen_ttbar_otherPart_pt",
        expression="gen_ttbar_otherPart_pt",
        binning=[0, 0.5, 1.1],
        unit="",
        x_title=r"$p_{T}$ of the t$\overline{t}$ system + other particle",
    )
