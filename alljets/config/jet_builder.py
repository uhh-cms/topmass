"""Helper builders for fitted and reconstructed jet collections."""

from columnflow.columnar_util import (
    attach_coffea_behavior,
    default_coffea_collections,
)


def build_w1jet(events, which=None):
    """Access fitted W1 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"FitW1": default_coffea_collections["Jet"]},
    )
    W1jets = events.FitW1
    if which is None:
        return W1jets * 1
    if which == "mass":
        return W1jets.mass
    if which == "pt":
        return W1jets.pt
    if which == "eta":
        return W1jets.eta
    if which == "abs_eta":
        return abs(W1jets.eta)
    if which == "phi":
        return W1jets.phi
    if which == "energy":
        return W1jets.energy
    raise ValueError(f"Unknown which: {which}")


def build_w1recojet(events, which=None):
    """Access reconstructed W1 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"RecoW1": default_coffea_collections["Jet"]},
    )
    W1recojets = events.RecoW1
    if which is None:
        return W1recojets * 1
    if which == "mass":
        return W1recojets.mass
    if which == "pt":
        return W1recojets.pt
    if which == "eta":
        return W1recojets.eta
    if which == "abs_eta":
        return abs(W1recojets.eta)
    if which == "phi":
        return W1recojets.phi
    if which == "energy":
        return W1recojets.energy
    raise ValueError(f"Unknown which: {which}")


def build_w2recojet(events, which=None):
    """Access reconstructed W2 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"RecoW2": default_coffea_collections["Jet"]},
    )
    W2recojets = events.RecoW2
    if which is None:
        return W2recojets * 1
    if which == "mass":
        return W2recojets.mass
    if which == "pt":
        return W2recojets.pt
    if which == "eta":
        return W2recojets.eta
    if which == "abs_eta":
        return abs(W2recojets.eta)
    if which == "phi":
        return W2recojets.phi
    if which == "energy":
        return W2recojets.energy
    raise ValueError(f"Unknown which: {which}")


def build_top1recojet(events, which=None):
    """Access reconstructed Top1 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"RecoTop1": default_coffea_collections["Jet"]},
    )
    Top1recojets = events.RecoTop1
    if which is None:
        return Top1recojets * 1
    if which == "mass":
        return Top1recojets.mass
    if which == "pt":
        return Top1recojets.pt
    if which == "eta":
        return Top1recojets.eta
    if which == "abs_eta":
        return abs(Top1recojets.eta)
    if which == "phi":
        return Top1recojets.phi
    if which == "energy":
        return Top1recojets.energy
    raise ValueError(f"Unknown which: {which}")


def build_top2recojet(events, which=None):
    """Access reconstructed Top2 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"RecoTop2": default_coffea_collections["Jet"]},
    )
    Top2recojets = events.RecoTop2
    if which is None:
        return Top2recojets * 1
    if which == "mass":
        return Top2recojets.mass
    if which == "pt":
        return Top2recojets.pt
    if which == "eta":
        return Top2recojets.eta
    if which == "abs_eta":
        return abs(Top2recojets.eta)
    if which == "phi":
        return Top2recojets.phi
    if which == "energy":
        return Top2recojets.energy
    raise ValueError(f"Unknown which: {which}")


def build_top1jet(events, which=None):
    """Access fitted Top1 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"FitTop1": default_coffea_collections["Jet"]},
    )
    Top1jets = events.FitTop1
    if which is None:
        return Top1jets * 1
    if which == "mass":
        return Top1jets.mass
    if which == "pt":
        return Top1jets.pt
    if which == "eta":
        return Top1jets.eta
    if which == "abs_eta":
        return abs(Top1jets.eta)
    if which == "phi":
        return Top1jets.phi
    if which == "energy":
        return Top1jets.energy
    raise ValueError(f"Unknown which: {which}")


def build_b1jet(events, which=None):
    """Access fitted B1 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"FitB1": default_coffea_collections["Jet"]},
    )
    B1jets = events.FitB1
    if which is None:
        return B1jets * 1
    if which == "mass":
        return B1jets.mass
    if which == "pt":
        return B1jets.pt
    if which == "eta":
        return B1jets.eta
    if which == "abs_eta":
        return abs(B1jets.eta)
    if which == "phi":
        return B1jets.phi
    if which == "energy":
        return B1jets.energy
    raise ValueError(f"Unknown which: {which}")


def build_b2jet(events, which=None):
    """Access fitted B2 jets or a selected component."""
    events = attach_coffea_behavior(
        events, {"FitB2": default_coffea_collections["Jet"]},
    )
    B2jets = events.FitB2
    if which is None:
        return B2jets * 1
    if which == "mass":
        return B2jets.mass
    if which == "pt":
        return B2jets.pt
    if which == "eta":
        return B2jets.eta
    if which == "abs_eta":
        return abs(B2jets.eta)
    if which == "phi":
        return B2jets.phi
    if which == "energy":
        return B2jets.energy
    raise ValueError(f"Unknown which: {which}")


def build_ttbar(events, which=None):
    """Build ttbar system from RecoTop1 and RecoTop2."""
    events = attach_coffea_behavior(
        events,
        {
            "RecoTop1": default_coffea_collections["Jet"],
            "RecoTop2": default_coffea_collections["Jet"],
        },
    )

    top1 = events.RecoTop1
    top2 = events.RecoTop2

    ttbar = top1 + top2

    if which is None:
        return ttbar * 1
    if which == "mass":
        return ttbar.mass
    if which == "pt":
        return ttbar.pt
    if which == "eta":
        return ttbar.eta
    if which == "phi":
        return ttbar.phi
    if which == "energy":
        return ttbar.energy

    raise ValueError(f"Unknown which: {which}")


def build_avg_w_mass(events):
    W1_mass = build_w1recojet(events, which="mass")
    W2_mass = build_w2recojet(events, which="mass")
    return (W1_mass + W2_mass) / 2


def build_avg_reco_Top_mass(events):
    Top1_mass = build_top1recojet(events, which="mass")
    Top2_mass = build_top2recojet(events, which="mass")
    return (Top1_mass + Top2_mass) / 2


def build_reco_R_bq(events):
    events = attach_coffea_behavior(events, {"FitJet.reco": default_coffea_collections["Jet"]})

    reco = events.FitJet.reco
    return (reco[:, 0].pt + reco[:, 1].pt) / (reco[:, 2].pt + reco[:, 3].pt + reco[:, 4].pt + reco[:, 5].pt)


def build_avg_R_bq(events):
    events = attach_coffea_behavior(events, {"FitJet.reco": default_coffea_collections["Jet"]})

    reco = events.FitJet.reco

    R_bq_top1 = reco[:, 0].pt / (reco[:, 2].pt + reco[:, 3].pt)
    R_bq_top2 = reco[:, 1].pt / (reco[:, 4].pt + reco[:, 5].pt)

    return (R_bq_top1 + R_bq_top2) / 2


def build_xb_avg(events):
    xb_top = events.xb.top
    xb_antitop = events.xb.antitop
    return 0.5 * (xb_top + xb_antitop)


def _get_b_q_pt(events, vectorial=False):
    """Return (pt_b, pt_q) for the b-jet-candidate system and light-jet system,
    either as scalar pt sums or as pt of the vector-summed 4-momenta."""
    fit_jet = attach_coffea_behavior(
        events.FitJet,
        {"reco": default_coffea_collections["Jet"]},
    )
    reco = fit_jet.reco

    if vectorial:
        b_system = reco[:, 0].add(reco[:, 1])
        q_system = reco[:, 2].add(reco[:, 3]).add(reco[:, 4]).add(reco[:, 5])
        return b_system.pt, q_system.pt
    else:
        pt_b = reco[:, 0].pt + reco[:, 1].pt
        pt_q = reco[:, 2].pt + reco[:, 3].pt + reco[:, 4].pt + reco[:, 5].pt
        return pt_b, pt_q


def build_R_bq(events, which="ratio", vectorial=False):
    """
    Build various b-vs-q pt comparison observables.

    which:
      - "ratio":        pt_b / pt_q
      - "diff":         pt_b - pt_q
      - "rel_diff_sum": (pt_b - pt_q) / (pt_b + pt_q)
      - "rel_diff_q":   (pt_b - pt_q) / pt_q
    vectorial:
      - False: pt_b, pt_q are scalar sums of individual jet pt's
      - True:  pt_b, pt_q are pt of the vector-summed 4-momenta
    """
    pt_b, pt_q = _get_b_q_pt(events, vectorial=vectorial)
    if which == "ratio":
        return pt_b / pt_q
    if which == "diff":
        return pt_b - pt_q
    if which == "rel_diff_sum":
        return (pt_b - pt_q) / (pt_b + pt_q)
    if which == "rel_diff_q":
        return (pt_b - pt_q) / pt_q

    raise ValueError(f"Unknown which: {which}")
