# coding: utf-8

"""
Producers for normalized Monte Carlo event weights.

This module defines ColumnFlow producers that normalize event-by-event MC
weights such that their average value over the processed sample is unity.

The normalization factors are derived from selection statistics produced by
``MergeSelectionStats`` and are computed separately for different systematic
weight families, including:

- renormalization/factorization scale weights (muR/muF)
- trigger scale factor variations
- hdamp variations
- parton shower (ISR/FSR) weights
- PDF and alpha_s variations
- pileup weights

For most weights, normalization is performed as:

    normalized_weight = w / <w>

Trigger weights are treated slightly differently in order to preserve the
nominal trigger normalization while re-scaling systematic variations relative
to the nominal average.

The resulting normalized weights are written as additional columns with the
prefix ``normalized_``.
"""

import functools
import law

from columnflow.util import maybe_import, safe_div
from columnflow.production import Producer, producer
from columnflow.production.cms.pileup import pu_weights_from_columnflow
from columnflow.production.cms.scale import murmuf_weights
from columnflow.columnar_util import set_ak_column

from alljets.production.dctr_hdamp import dctr_hdamp
from alljets.production.ps_weights import ps_weights
from alljets.production.trig_cor_weight import trig_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={murmuf_weights.PRODUCES},
    mc_only=True,
)
def normalized_murmuf_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Normalize muR/muF variation weights to unit average.

    The normalization factors are derived from the sample-wide averages
    computed in ``MergeSelectionStats``.
    """

    for weight_name in self.mu_weight_names:

        # Sample-average value of the corresponding scale variation weight.
        avg = self.average_mu_weights.get(weight_name, 1.0)

        # Normalize the event weight such that the mean value becomes unity.
        normalized = events[weight_name] / avg

        events = set_ak_column_f32(events, f"normalized_{weight_name}", normalized)

    return events


@normalized_murmuf_weight.post_init
def normalized_murmuf_weight_post_init(self: Producer, task: law.Task, **kwargs) -> None:
    
    # Collect all produced muR/muF weight columns dynamically.
    # For non-nominal global shifts, only keep the central weights
    # to avoid duplicating systematic variations.

    self.mu_weight_names = {
        str(weight_name)
        for weight_name in self[murmuf_weights].produced_columns
        if (
            str(weight_name).startswith("murmuf_weight") and
            (
                task.global_shift_inst.is_nominal or not str(weight_name).endswith(("_up", "_down"))
            )
        )
    }

    # declare inputs
    self.uses.clear()
    self.uses |= self.mu_weight_names

    # declare outputs
    self.produces |= {
        f"normalized_{w}" for w in self.mu_weight_names
    }


@normalized_murmuf_weight.requires
def normalized_murmuf_weight_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats

    # Access merged selection statistics containing summed MC weights.
    # Workflow tasks use branch=-1 to load fully merged statistics.

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_murmuf_weight.setup
def normalized_murmuf_weight_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:

    # Load cached selection statistics once per task.
    stats = task.cached_value(key="selection_stats",
                              func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"))

    self.average_mu_weights = {}

    # Total number of processed events used to compute averages
    total_events = stats.get("num_events", 1.0)

    for weight_name in self.mu_weight_names:

        # Statistics store the summed weight under:
        sum_key = f"sum_{weight_name}"

        total_weight = stats.get(sum_key)

        # Compute the sample-average weight.
        # Fallback to unity if statistics are unavailable.
        self.average_mu_weights[weight_name] = safe_div(total_weight, total_events) if total_weight is not None else 1.0


@producer(
    uses={trig_weights.PRODUCES},
    mc_only=True,
)
def normalized_trig_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # Trigger-weight variations are normalized relative to the nominal
    # trigger-weight average such that all variations preserve the
    # nominal overall normalization while retaining relative deviations.

    for weight_name in self.trig_weight_names:

        avg_var = self.average_trig_weights.get(weight_name, 1.0)
        avg_nom = self.nominal_trig_average or 1.0

        normalized = events[weight_name] * (avg_nom / avg_var)

        events = set_ak_column_f32(events, f"normalized_{weight_name}", normalized)

    return events


@normalized_trig_weight.post_init
def normalized_trig_weight_post_init(self: Producer, task: law.Task, **kwargs) -> None:

    self.trig_weight_names = {
        str(weight_name)
        for weight_name in self[trig_weights].produced_columns
        if (
            str(weight_name).startswith("trig_weight") and
            (
                task.global_shift_inst.is_nominal or not str(weight_name).endswith(("_up", "_down"))
            )
        )
    }

    self.uses.clear()
    self.uses |= self.trig_weight_names

    self.produces |= {f"normalized_{w}" for w in self.trig_weight_names}


@normalized_trig_weight.requires
def normalized_trig_weight_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_trig_weight.setup
def normalized_trig_weight_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:

    stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    # Store the average nominal trigger weight separately since
    # all trigger variations are normalized relative to it.
    self.nominal_trig_average = None
    self.average_trig_weights = {}

    total_events = stats.get("num_events_selected", 1.0)

    # Trigger-weight averages are computed after event selection,
    # matching the phase space where trigger scale factors are applied.

    for weight_name in self.trig_weight_names:
        sum_key = f"sum_{weight_name}_selected"
        total_weight = stats.get(sum_key)

        avg = safe_div(total_weight, total_events) if total_weight is not None else 1.0
        self.average_trig_weights[weight_name] = avg

        if weight_name == "trig_weight":
            self.nominal_trig_average = avg


@producer(
    uses={dctr_hdamp.PRODUCES},
    mc_only=True,
)
def normalized_hdamp_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    for weight_name in self.hdamp_weight_names:

        avg = self.average_hdamp_weights.get(weight_name, 1.0)

        normalized = events[weight_name] / avg

        events = set_ak_column_f32(
            events,
            f"normalized_{weight_name}",
            normalized,
        )

    return events


@normalized_hdamp_weight.post_init
def normalized_hdamp_weight_post_init(self: Producer, task: law.Task, **kwargs) -> None:

    self.hdamp_weight_names = {
        str(weight_name)
        for weight_name in self[dctr_hdamp].produced_columns
        if (
            str(weight_name).startswith("hdamp_weight") and
            (
                task.global_shift_inst.is_nominal or not
                str(weight_name).endswith(("_up", "_down"))
            )
        )
    }

    self.uses.clear()
    self.uses |= self.hdamp_weight_names

    self.produces |= {f"normalized_{w}" for w in self.hdamp_weight_names}


@normalized_hdamp_weight.requires
def normalized_hdamp_weight_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_hdamp_weight.setup
def normalized_hdamp_weight_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:

    stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    self.average_hdamp_weights = {}

    total_events = stats.get("num_events", 1.0)

    for weight_name in self.hdamp_weight_names:

        sum_key = f"sum_{weight_name}"

        total_weight = stats.get(sum_key)

        self.average_hdamp_weights[weight_name] = (
            safe_div(total_weight, total_events) if total_weight is not None else 1.0
        )


@producer(
    uses={ps_weights.PRODUCES},
    mc_only=True,
)
def normalized_ps_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    for weight_name in self.ps_weight_names:

        avg = self.average_ps_weights.get(weight_name, 1.0)

        normalized = events[weight_name] / avg

        events = set_ak_column_f32(
            events,
            f"normalized_{weight_name}",
            normalized,
        )

    return events


@normalized_ps_weights.post_init
def normalized_ps_weights_post_init(self: Producer, task: law.Task, **kwargs) -> None:

    self.ps_weight_names = {
        str(weight_name)
        for weight_name in self[ps_weights].produced_columns
        if (
            (
                str(weight_name).startswith("isr_weight") or
                str(weight_name).startswith("fsr_weight")
            ) and (
                task.global_shift_inst.is_nominal or not
                str(weight_name).endswith(("_up", "_down"))
            )
        )
    }

    # inputs
    self.uses.clear()
    self.uses |= self.ps_weight_names

    # outputs
    self.produces |= {f"normalized_{w}" for w in self.ps_weight_names}


@normalized_ps_weights.requires
def normalized_ps_weights_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_ps_weights.setup
def normalized_ps_weights_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:

    stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    self.average_ps_weights = {}

    total_events = stats.get("num_events", 1.0)

    for weight_name in self.ps_weight_names:

        sum_key = f"sum_{weight_name}"

        total_weight = stats.get(sum_key)

        self.average_ps_weights[weight_name] = safe_div(total_weight, total_events) if total_weight is not None else 1.0


@producer(
    uses={"pdf_weight"},
    produces={"normalized_pdf_weight"},
    mc_only=True,
)
def normalized_pdf_weight(self, events, **kwargs):

    avg = self.average_pdf_weights.get("pdf_weight", 1.0)

    normalized = events["pdf_weight"] / avg

    events = set_ak_column_f32(
        events,
        "normalized_pdf_weight",
        normalized,
    )

    return events


@normalized_pdf_weight.setup
def normalized_pdf_weight_setup(self, task, inputs, **kwargs):

    stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    total_events = stats.get("num_events", 1.0)
    total_weight = stats.get("sum_pdf_weight")

    self.average_pdf_weights = {
        "pdf_weight": safe_div(total_weight, total_events) if total_weight is not None else 1.0,
    }


@normalized_pdf_weight.requires
def normalized_pdf_weight_requires(self, task, reqs, **kwargs):
    from columnflow.tasks.selection import MergeSelectionStats

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@producer(
    uses={
        "pdf_weight",
        "pdf_hessian_*_weight_{up,down}",
        "pdf_alphas_weight_{up,down}",
    },
    mc_only=True,
)
def normalized_pdf_weights(self, events, **kwargs):

    pdf_weight_names = self.pdf_weight_names

    for weight_name in pdf_weight_names:

        avg = self.average_pdf_weights.get(weight_name, 1.0)

        normalized = events[weight_name] / avg

        events = set_ak_column_f32(
            events,
            f"normalized_{weight_name}",
            normalized,
        )

    return events


@normalized_pdf_weights.post_init
def normalized_pdf_weights_post_init(self, task, **kwargs):

    self.pdf_weight_names = set()

    # nominal weight
    self.pdf_weight_names |= {"pdf_weight"}
    shift = task.global_shift_inst
    is_nominal = ((shift.name == "nominal") and self.dataset_inst.has_tag("tt"))

    if is_nominal:
        # alphas
        self.pdf_weight_names |= {
            "pdf_alphas_weight_up",
            "pdf_alphas_weight_down",
        }

        # 100 Hessian variations (symmetric up/down)
        self.pdf_weight_names |= {
            f"pdf_hessian_{i:03d}_weight_up"
            for i in range(1, 101)
        }

        self.pdf_weight_names |= {
            f"pdf_hessian_{i:03d}_weight_down"
            for i in range(1, 101)
        }

    self.uses.clear()
    self.uses |= self.pdf_weight_names

    self.produces |= {f"normalized_{w}" for w in self.pdf_weight_names}


@normalized_pdf_weights.requires
def normalized_pdf_weights_requires(self: Producer, task: law.Task, reqs: dict, **kwargs) -> None:
    from columnflow.tasks.selection import MergeSelectionStats

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_pdf_weights.setup
def normalized_pdf_weights_setup(self: Producer, task: law.Task, inputs: dict, **kwargs) -> None:

    stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    self.average_pdf_weights = {}

    total_events = stats.get("num_events", 1.0)

    for weight_name in self.pdf_weight_names:

        sum_key = f"sum_{weight_name}"

        total_weight = stats.get(sum_key)

        self.average_pdf_weights[weight_name] = safe_div(
            total_weight,
            total_events,
        ) if total_weight is not None else 1.0


@producer(
    uses={pu_weights_from_columnflow.PRODUCES},
    mc_only=True,
)
def normalized_pu_weights(self, events, **kwargs):

    for weight_name in self.pu_weight_names:

        if weight_name not in events.fields:
            continue

        avg = self.average_pu_weights.get(weight_name, 1.0)

        normalized = events[weight_name] / avg

        events = set_ak_column_f32(events, f"normalized_{weight_name}", normalized)

    return events


@normalized_pu_weights.post_init
def normalized_pu_weights_post_init(self, task, **kwargs):

    self.pu_weight_names = {
        "pu_weight",
        "pu_weight_minbias_xs_up",
        "pu_weight_minbias_xs_down",
    }

    # inputs
    self.uses.clear()
    self.uses |= self.pu_weight_names

    # outputs
    self.produces |= {f"normalized_{w}" for w in self.pu_weight_names}


@normalized_pu_weights.requires
def normalized_pu_weights_requires(self, task, reqs, **kwargs):
    from columnflow.tasks.selection import MergeSelectionStats

    reqs["selection_stats"] = MergeSelectionStats.req_different_branching(
        task,
        branch=-1 if task.is_workflow() else 0,
    )


@normalized_pu_weights.setup
def normalized_pu_weights_setup(self, task, inputs, **kwargs):

    stats = task.cached_value(
        key="selection_stats",
        func=lambda: inputs["selection_stats"]["stats"].load(formatter="json"),
    )

    self.average_pu_weights = {}

    total_events = stats.get("num_events", 1.0)

    for weight_name in self.pu_weight_names:

        sum_key = f"sum_{weight_name}"
        total_weight = stats.get(sum_key)

        self.average_pu_weights[weight_name] = safe_div(
            total_weight,
            total_events,
        ) if total_weight is not None else 1.0
