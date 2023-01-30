# coding: utf-8

"""
Selection methods defining masks for categories.
"""
import functools
from columnflow.util import maybe_import
from columnflow.selection import Selector, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(uses={"event"})
def sel_incl(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.ones_like(events.event)

@selector(uses={"Jet.pt"})
def sel_2j(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.num(events.Jet, axis=-1) == 2


@selector(uses={"Jet.pt"})
def sel_3j(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.num(events.Jet, axis=-1) == 3

@selector(uses={"Jet.pt"})
def sel_4j(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.num(events.Jet, axis=-1) == 4

@selector(uses={"Jet.pt"})
def sel_5j(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.num(events.Jet, axis=-1) == 5


@selector(uses={"channel_id"})
def sel_ee(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    ch = self.config_inst.get_channel("ee")
    return events["channel_id"] == ch.id

@selector(uses={"channel_id"})
def sel_mumu(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    ch = self.config_inst.get_channel("mumu")
    return events["channel_id"] == ch.id

@selector(uses={"channel_id"})
def sel_emu(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    ch = self.config_inst.get_channel("emu")
    return events["channel_id"] == ch.id

import functools
import types


def copy_func(f,name):
    g = types.FunctionType(f.__code__, f.__globals__, name=name)
    #g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

for lep_ch in ["ee", "mumu", "emu"]:
    for n_jet in ["2j", "3j", "4j", "5j"]:

        funcs = {
            "lep_func": locals()[f"sel_{lep_ch}"],
            "jet_func": locals()[f"sel_{n_jet}"],
            # "dnn_func": locals()[f"catid_{dnn_ch}"],
        }
        
        def sel_func(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

            #Selector that call multiple functions and combines their outputs via
            #logical `and`.

            masks = [self[func](events, **kwargs) for func in self.uses]
            leaf_mask = functools.reduce((lambda a, b: a & b), masks)

            return leaf_mask
        
        
        
        name=f"sel_{lep_ch}_{n_jet}"
        uses={"channel_id","Jet.pt"}
        
        globals()[name] = selector(uses=set(funcs.values()),func=copy_func(sel_func,name),cls_name=name)

#import IPython
#IPython.embed()
"""
for lep_ch in ["ee", "mumu", "emu"]:
    for n_jet in ["2j", "3j", "4j"]:

        funcs = {
            "lep_func": locals()[f"sel_{lep_ch}"],
            "jet_func": locals()[f"sel_{n_jet}"],
            # "dnn_func": locals()[f"catid_{dnn_ch}"],
        }

        # @selector(name=f"catid_{lep_ch}_{jet_ch}")
        @selector(cls_name=f"sel_{lep_ch}_{n_jet}",uses=set(funcs.values()),produces=set(funcs.values()),)
        def sel_mask(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

            #Selector that call multiple functions and combines their outputs via
            #logical `and`.

            masks = [self[func](events, **kwargs) for func in self.uses]
            leaf_mask = functools.reduce((lambda a, b: a & b), masks)

            return leaf_mask
"""
#import IPython
#IPython.embed()
