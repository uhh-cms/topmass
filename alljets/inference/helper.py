# coding: utf-8

"""
Helper functions for inference models.
"""

from __future__ import annotations


from columnflow.inference import InferenceModel, ParameterType, ParameterTransformation
from columnflow.inference.cms.datacard import DatacardWriter

# --- workaround for a columnflow bug: plain shape-type parameters (no transformations)
# --- are constructed with effect=None, which DatacardWriter.write() cannot encode.
# --- see: https://github.com/columnflow/columnflow/blob/master/columnflow/inference/cms/datacard.py
_orig_modify_parameter_effect = DatacardWriter.modify_parameter_effect


def _modify_parameter_effect(self, cat_obj, proc_obj, param_obj, effect):
    if effect is None and param_obj.type.is_shape:
        effect = 1.0
    return _orig_modify_parameter_effect(self, cat_obj, proc_obj, param_obj, effect)


DatacardWriter.modify_parameter_effect = _modify_parameter_effect


def add_processes(im: InferenceModel) -> None:
    im.add_process(
        "TT",
        config_data={
            config_inst.name: im.process_config_spec(
                process="tt",
                mc_datasets=["tt_fh_powheg",
                             "tt_sl_powheg",
                             "tt_dl_powheg"],
            )
            for config_inst in im.config_insts
        },
        is_signal=True,
    )
    im.add_process(
        "BKG",
        config_data={
            config_inst.name: im.process_config_spec(
                process="qcd_est",
                # mc_datasets=["data*"],
            )
            for config_inst in im.config_insts
        },
        is_signal=False,
        is_dynamic=True,
    )


def add_parameters(im: InferenceModel) -> None:
    # im.add_parameter(
    #     "r_TT",
    #     process=["TT"],
    #     type=ParameterType.rate_unconstrained,
    # )
    # for config_inst in im.config_insts:
    #     im.add_parameter(
    #         f"r_BKG_{config_inst.campaign.x.year}{config_inst.campaign.x.postfix}",
    #         process=["BKG"],
    #         type=ParameterType.rate_unconstrained,
    #     )
    # groups
    im.add_parameter_group("experimental")
    im.add_parameter_group("modelling")

    for s in ["mtop1", "mtop3", "mtop6"]:
        im.add_parameter(
            s,
            process=["TT"],
            type=ParameterType.shape,
            config_data={
                config_inst.name: im.parameter_config_spec(
                    shift_source=s,
                )
                for config_inst in im.config_insts
            },
        )
    for config_inst in im.config_insts:
        lumi = config_inst.x.luminosity
        for unc_name in lumi.uncertainties:
            im.add_parameter(
                unc_name,
                type=ParameterType.rate_gauss,
                effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                process=["TT"],
                group=["experiment"],
            )

    experimental = {
        "CMS_btag_fixedWP_bc_correlated": "btag_heavy_cor",
        "CMS_btag_fixedWP_light_correlated": "btag_light_cor",
        "CMS_res_j_etaLT1p93_13TeV": "jer_EtaLow",
        "CMS_res_j_etaGE1p93_13TeV": "jer_EtaHigh",
    }

    experimental_uncor = {
        "CMS_pileup": "pu_weight_minbias_xs",
        "CMS_trig_htsixjets2btag": "trig",
        "CMS_btag_fixedWP_bc_uncorrelated": "btag_heavy_uncor",
        "CMS_btag_fixedWP_light_uncorrelated": "btag_light_uncor",
    }

    for jec_unc in im.config_insts[0].x.jec.Jet.uncertainty_sources:
        if jec_unc == "PileUpMuZero" or jec_unc == "PileUpEnvelope":
            continue
        if jec_unc in ["AbsoluteStat", "RelativeJEREC1", "RelativeJEREC2", "RelativePtEC1", "RelativePtEC2",
                       "RelativeSample", "RelativeStatEC", "RelativeStatFSR", "RelativeStatHF", "TimePtEta"]:
            experimental_uncor["CMS_scale_j_" + jec_unc] = "jec_" + jec_unc
        else:
            experimental["CMS_scale_j_" + jec_unc] = "jec_" + jec_unc

    modelling = {
        # "ps_hdamp_sample": "hdamp",
        "ps_hdamp": "hdamp_dctr",
        "pdf_alphas": "alphas",
        # "ps_fragmentation_dctr": "rb_dctr",
        # "ps_fragmentation": "bfrag",
        "ps_fragmentation": "bfrag_rel",
        "ps_fragmentation_bdecay": "bfrag_bdecay",
        "QCDscale_ren_ttbar": "mur",
        "QCDscale_fac_ttbar": "muf",
        "underlying_event": "tune",
    }

    modelling_envelope = {
        "top_pt_reweighting": "top_pt",
        "ps_CR1": "tune_cr1",
        "ps_CR2": "tune_cr2",
        "ps_ERD": "tune_erdON",
        "ps_Recoil": "tune_rtt",
        "ps_fragmentation_lund": "bfrag_lund",
        "ps_fragmentation_peterson": "bfrag_peterson",
    }

    splittings = ("G2GG", "G2QQ", "Q2QG", "X2XG")
    for var in [f"{a}_{b}_{c}" for a in ["isr", "fsr"] for b in splittings for c in ["muR", "cNS"]]:
        modelling["ps_" + var] = var
    for i in range(100):
        modelling_envelope[f"pdf_{i:02}"] = f"hessian_{i + 1:03d}"

    def add_source(nuisance_name, shift_name, group, transformations=()):
        im.add_parameter(
            nuisance_name,
            process=["TT"],
            type=ParameterType.shape,
            transformations=transformations,
            config_data={
                config_inst.name: im.parameter_config_spec(
                    shift_source=shift_name,
                )
                for config_inst in im.config_insts
            },
            group=group,
        )

    def add_source_uncor(nuisance_name, shift_name, group, transformations=()):
        for config_inst in im.config_insts:
            im.add_parameter(
                f"{nuisance_name}_{config_inst.campaign.x.year}{config_inst.campaign.x.postfix}",
                process=["TT"],
                type=ParameterType.shape,
                transformations=transformations,
                config_data={
                    config_inst.name: im.parameter_config_spec(
                        shift_source=shift_name,
                    ),
                },
                group=group,
            )

    for nuis_name, shift_name in experimental.items():
        add_source(nuis_name, shift_name, "experimental")
    for nuis_name, shift_name in experimental_uncor.items():
        add_source_uncor(nuis_name, shift_name, "experimental")

    for nuis_name, shift_name in modelling.items():
        add_source(nuis_name, shift_name, "modelling")
    for nuis_name, shift_name in modelling_envelope.items():
        add_source(nuis_name, shift_name, "modelling", (ParameterTransformation.envelope,))
