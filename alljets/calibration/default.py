# coding: utf-8

"""
Exemplary calibration methods.
"""
from __future__ import annotations

from columnflow.calibration import Calibrator, calibrator
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, optional_column as optional
from columnflow.calibration.cms.jets import jec, jer
from columnflow.production.cms.mc_weight import mc_weight

np = maybe_import("numpy")
ak = maybe_import("awkward")


@calibrator(
    uses={
        mc_weight,
        optional("Jet.hadronFlavour"), optional("Jet.partonFlavour"),
    },
    produces={
        mc_weight,
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    task = kwargs["task"]

    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

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
            events = self[self.jer_jec_full_cls](events, **kwargs)
        else:
            # nominal jec and jer
            events = self[self.jec_nominal_cls](events, **kwargs)
            events = self[self.jer_jec_nominal_cls](
                events, **kwargs)

    # apply flavour corrections only to the correct columns
    if self.dataset_inst.is_mc and task.global_shift_inst.is_nominal:
        pf = abs(events.Jet.partonFlavour)

        # bottom
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureBottom_up", (ak.where(pf == 5,
                                events.Jet.pt_jec_FlavorPureBottom_up, events.Jet.pt)))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureBottom_up", (ak.where(pf == 5,
                                events.Jet.mass_jec_FlavorPureBottom_up, events.Jet.mass)))
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureBottom_down", ak.where(pf == 5,
                                events.Jet.pt_jec_FlavorPureBottom_down, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureBottom_down", ak.where(pf == 5,
                                events.Jet.mass_jec_FlavorPureBottom_down, events.Jet.mass))

        # charm
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureCharm_up", ak.where(pf == 4,
                                            events.Jet.pt_jec_FlavorPureCharm_up, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureCharm_up", ak.where(pf == 4,
                                            events.Jet.mass_jec_FlavorPureCharm_up, events.Jet.mass))
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureCharm_down", ak.where(pf == 4,
                                            events.Jet.pt_jec_FlavorPureCharm_down, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureCharm_down", ak.where(pf == 4,
                                                events.Jet.mass_jec_FlavorPureCharm_down, events.Jet.mass))
        # gluon
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureGluon_up", ak.where(
            (pf == 0) | (pf == 21), events.Jet.pt_jec_FlavorPureGluon_up, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureGluon_up", ak.where(
            (pf == 0) | (pf == 21), events.Jet.mass_jec_FlavorPureGluon_up, events.Jet.mass))
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureGluon_down", ak.where(
            (pf == 0) | (pf == 21), events.Jet.pt_jec_FlavorPureGluon_down, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureGluon_down", ak.where(
            (pf == 0) | (pf == 21), events.Jet.mass_jec_FlavorPureGluon_down, events.Jet.mass))

        # uds quarks
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureQuark_up", ak.where((pf == 1) | (
            pf == 2) | (pf == 3), events.Jet.pt_jec_FlavorPureQuark_up, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureQuark_up", ak.where((pf == 1) | (
            pf == 2) | (pf == 3), events.Jet.mass_jec_FlavorPureQuark_up, events.Jet.mass))
        events = set_ak_column(events, "Jet.pt_jec_FlavorPureQuark_down", ak.where((pf == 1) | (
            pf == 2) | (pf == 3), events.Jet.pt_jec_FlavorPureQuark_down, events.Jet.pt))
        events = set_ak_column(events, "Jet.mass_jec_FlavorPureQuark_down", ak.where((pf == 1) | (
            pf == 2) | (pf == 3), events.Jet.mass_jec_FlavorPureQuark_down, events.Jet.mass))
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
            self.config_inst.set_aux(
                f"calib_{name}_cls", base.derive(name, cls_dict=cls_dict or {}))

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
        add_calib_cls("jer_jec_full", jer, cls_dict={
            "met_name": met_name,
        })
        add_calib_cls("jer_jec_nominal", jer, cls_dict={
            "met_name": met_name,
            "jec_uncertainty_sources": [],
        })

        # change the flag
        self.config_inst.set_aux(flag, True)

    # store references to classes
    self.jec_full_cls = self.config_inst.x.calib_jec_full_cls
    self.jec_nominal_cls = self.config_inst.x.calib_jec_nominal_cls
    self.jer_jec_full_cls = self.config_inst.x.calib_jer_jec_full_cls
    self.jer_jec_nominal_cls = self.config_inst.x.calib_jer_jec_nominal_cls

    # collect derived calibrators and add them to the calibrator uses and produces
    derived_calibrators = {
        self.jec_full_cls,
        self.jec_nominal_cls,
        self.jer_jec_full_cls,
        self.jer_jec_nominal_cls,
    }
    self.uses |= derived_calibrators
    self.produces |= derived_calibrators
