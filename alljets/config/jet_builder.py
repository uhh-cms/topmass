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
