# coding: utf-8

"""
Exemplary calibration methods.
"""
from __future__ import annotations

from columnflow.calibration import Calibrator, calibrator
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.calibration.cms.jets import jec, jer
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import (
    deterministic_event_seeds, deterministic_jet_seeds, deterministic_seeds,
)

np = maybe_import("numpy")
ak = maybe_import("awkward")


@calibrator(
    uses={
        deterministic_seeds,
        "Jet.pt", "Jet.mass", "Jet.eta", "Jet.phi",
    },
    produces={
        deterministic_seeds,
        "Jet.pt", "Jet.mass",
        "Jet.pt_jec_up", "Jet.mass_jec_up",
        "Jet.pt_jec_down", "Jet.mass_jec_down",
    },
)
def fake(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    # a) "correct" Jet.pt by scaling four momenta by 1.1 (pt<30) or 0.9 (pt<=30)
    # b) add 4 new columns faking the effect of JEC variations

    # add deterministic seeds that could (e.g.) be used for smearings
    events = self[deterministic_seeds](events, **kwargs)

    # a)
    pt_mask = ak.flatten(events.Jet.pt < 30)
    n_jet_pt = np.asarray(ak.flatten(events.Jet.pt))
    n_jet_mass = np.asarray(ak.flatten(events.Jet.mass))
    n_jet_pt[pt_mask] *= 1.1
    n_jet_pt[~pt_mask] *= 0.9
    n_jet_mass[pt_mask] *= 1.1
    n_jet_mass[~pt_mask] *= 0.9

    # b)
    events = set_ak_column(events, "Jet.pt_jec_up", events.Jet.pt * 1.05)
    events = set_ak_column(events, "Jet.mass_jec_up", events.Jet.mass * 1.05)
    events = set_ak_column(events, "Jet.pt_jec_down", events.Jet.pt * 0.95)
    events = set_ak_column(events, "Jet.mass_jec_down", events.Jet.mass * 0.95)

    return events


@calibrator(
    uses={
        mc_weight, deterministic_event_seeds, deterministic_jet_seeds,
    },
    produces={
        mc_weight, deterministic_event_seeds, deterministic_jet_seeds,
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    task = kwargs["task"]

    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # seed producers
    # !! as this is the first step, the object collections should still be pt-sorted,
    # !! so no manual sorting needed here (but necessary if, e.g., jec is applied before)
    events = self[deterministic_event_seeds](events, **kwargs)
    events = self[deterministic_jet_seeds](events, **kwargs)

    # data/mc specific calibrations
    if self.dataset_inst.is_data:
        # nominal jec
        events = self[self.jec_nominal_cls](events, **kwargs)

    else:
        # for mc, when the nominal shift is requested, apply calibrations with uncertainties (i.e. full), otherwise
        # invoke calibrators configured not to evaluate and save uncertainties
        if task.global_shift_inst.is_nominal:
            # full jec and jer
            events = self[self.jec_full_cls](events, **kwargs)
            events = self[self.deterministic_jer_jec_full_cls](events, **kwargs)
        else:
            # nominal jec and jer
            events = self[self.jec_nominal_cls](events, **kwargs)
            events = self[self.deterministic_jec_jec_nominal_cls](events, **kwargs)

    return events


@default.init
def default_init(self: Calibrator, **kwargs) -> None:
    # set the name of the met collection to use
    met_name = self.config_inst.x.met_name
    raw_met_name = self.config_inst.x.raw_met_name

    # derive calibrators to add settings once
    flag = f"custom_calibs_registered_{self.cls_name}"
    if not self.config_inst.x(flag, False):
        def add_calib_cls(name, base, cls_dict=None):
            self.config_inst.set_aux(f"calib_{name}_cls", base.derive(name, cls_dict=cls_dict or {}))

        # jec calibrators
        add_calib_cls("jec_full", jec, cls_dict={
            "mc_only": True,
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        })
        add_calib_cls("jec_nominal", jec, cls_dict={
            "uncertainty_sources": [],
            "met_name": met_name,
            "raw_met_name": raw_met_name,
        })
        # versions of jer that use the first random number from deterministic_seeds
        add_calib_cls("deterministic_jer_jec_full", jer, cls_dict={
            "deterministic_seed_index": 0,
            "met_name": met_name,
        })
        add_calib_cls("deterministic_jec_jec_nominal", jer, cls_dict={
            "deterministic_seed_index": 0,
            "met_name": met_name,
            "jec_uncertainty_sources": [],
        })

        # change the flag
        self.config_inst.set_aux(flag, True)

    # store references to classes
    self.jec_full_cls = self.config_inst.x.calib_jec_full_cls
    self.jec_nominal_cls = self.config_inst.x.calib_jec_nominal_cls
    self.deterministic_jer_jec_full_cls = self.config_inst.x.calib_deterministic_jer_jec_full_cls
    self.deterministic_jec_jec_nominal_cls = self.config_inst.x.calib_deterministic_jec_jec_nominal_cls

    # collect derived calibrators and add them to the calibrator uses and produces
    derived_calibrators = {
        self.jec_full_cls,
        self.jec_nominal_cls,
        self.deterministic_jer_jec_full_cls,
        self.deterministic_jec_jec_nominal_cls,
    }
    self.uses |= derived_calibrators
    self.produces |= derived_calibrators
