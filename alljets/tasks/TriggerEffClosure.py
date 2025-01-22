# coding: utf-8

"""
Tasks for trigger efficiency weight closure tests
"""

from __future__ import annotations
from collections import OrderedDict
from abc import abstractmethod

import law
import luigi
import order as od

from columnflow.tasks.framework.base import Requirements, ShiftTask, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, WeightProducerMixin,
    CategoriesMixin, ShiftSourcesMixin, HistHookMixin, VariablesMixin, ChunkedIOMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
)
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, DotDict
from columnflow.tasks.framework.parameters import last_edge_inclusive_inst
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.tasks.histograms import CreateHistograms, MergeShiftedHistograms


class CreateTriggerHistograms(
    CreateHistograms,
    VariablesMixin,
    WeightProducerMixin,
    MLModelsMixin,
    ProducersMixin,
    ReducedEventsUser,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    last_edge_inclusive = last_edge_inclusive_inst

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
        MLEvaluation=MLEvaluation,
    )

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    # names of columns that contain category ids
    # (might become a parameter at some point)
    category_id_columns = {"category_ids"}

    # register sandbox and shifts found in the chosen weight producer to this task
    register_weight_producer_sandbox = True
    register_weight_producer_shifts = True

    @law.util.classproperty
    def mandatory_columns(cls) -> set[str]:
        return set(cls.category_id_columns) | {"process_id"}

    def workflow_requires(self):
        reqs = super().workflow_requires()
        return reqs

    def requires(self):
        reqs = super().requires()
        return reqs

    workflow_condition = ReducedEventsUser.workflow_condition.copy()

    @workflow_condition.output
    def output(self):
        return {"hists": self.target(f"histograms__vars_{self.variables_repr}__{self.branch}.pickle")}

    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import (
            Route, update_ak_array, add_ak_aliases, has_ak_column, fill_hist,
        )

        # prepare inputs
        inputs = self.input()

        # declare output: dict of histograms
        histograms = {}

        # run the weight_producer setup
        producer_reqs = self.weight_producer_inst.run_requires()
        reader_targets = self.weight_producer_inst.run_setup(producer_reqs, luigi.task.getpaths(producer_reqs))

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})

        # define columns that need to be read
        read_columns = {Route("process_id")}
        read_columns |= set(map(Route, self.category_id_columns))
        read_columns |= set(self.weight_producer_inst.used_columns)
        read_columns |= set(map(Route, aliases.values()))
        read_columns |= {
            Route(inp)
            for variable_inst in (
                self.config_inst.get_variable(var_name)
                for var_name in law.util.flatten(self.variable_tuples.values())
            )
            for inp in (
                [variable_inst.expression]
                if isinstance(variable_inst.expression, str)
                # for variable_inst with custom expressions, read columns declared via aux key
                else variable_inst.x("inputs", [])
            )
        }

        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        # iterate over chunks of events and diffs
        file_targets = [inputs["events"]["events"]]
        if self.producer_insts:
            file_targets.extend([inp["columns"] for inp in inputs["producers"]])
        if self.ml_model_insts:
            file_targets.extend([inp["mlcolumns"] for inp in inputs["ml"]])

        # prepare inputs for localization
        with law.localize_file_targets(
            [*file_targets, *reader_targets.values()],
            mode="r",
        ) as inps:
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.abspath for inp in inps],
                source_type=len(file_targets) * ["awkward_parquet"] + [None] * len(reader_targets),
                read_columns=(len(file_targets) + len(reader_targets)) * [read_columns],
                chunk_size=self.weight_producer_inst.get_min_chunk_size(),
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))

                # add additional columns
                events = update_ak_array(events, *columns)

                # add aliases
                events = add_ak_aliases(
                    events,
                    aliases,
                    remove_src=True,
                    missing_strategy=self.missing_column_alias_strategy,
                )

                # build the full event weight
                if hasattr(self.weight_producer_inst, "skip_func") and not self.weight_producer_inst.skip_func():
                    events, weight = self.weight_producer_inst(events)
                else:
                    weight = ak.Array(np.ones(len(events), dtype=np.float32))

                weight_list = [weight, ak.Array(np.ones(len(events), dtype=np.float32))]
                # define and fill histograms, taking into account multiple axes
                for var_key, var_names in self.variable_tuples.items():
                    weight_index = 0
                    for weight in weight_list:
                        # get variable instances
                        variable_insts = [self.config_inst.get_variable(var_name) for var_name in var_names]

                        # create the histogram if not present yet
                        if var_key not in histograms:
                            h = (
                                hist.Hist.new
                                .IntCat([], name="category", growth=True)
                                .IntCat([], name="process", growth=True)
                                .IntCat([], name="shift", growth=True)
                                .IntCat([], name="weightcategory", growth=True)
                            )
                            # add variable axes
                            for variable_inst in variable_insts:
                                h = h.Var(
                                    variable_inst.bin_edges,
                                    name=variable_inst.name,
                                    label=variable_inst.get_full_x_title(),
                                )
                            # enable weights and store it
                            histograms[var_key] = h.Weight()

                        # merge category ids
                        category_ids = ak.concatenate(
                            [Route(c).apply(events) for c in self.category_id_columns],
                            axis=-1,
                        )

                        # broadcast arrays so that each event can be filled for all its categories
                        fill_data = {
                            "category": category_ids,
                            "process": events.process_id,
                            "shift": np.ones(len(events), dtype=np.int32) * self.global_shift_inst.id,
                            "weightcategory": weight_index,
                            "weight": weight,
                        }
                        for variable_inst in variable_insts:
                            # prepare the expression
                            expr = variable_inst.expression
                            if isinstance(expr, str):
                                route = Route(expr)
                                def expr(events, *args, **kwargs):
                                    if len(events) == 0 and not has_ak_column(events, route):
                                        return empty_f32
                                    return route.apply(events, null_value=variable_inst.null_value)
                            # apply it
                            fill_data[variable_inst.name] = expr(events)

                        # fill it
                        fill_hist(
                            histograms[var_key],
                            fill_data,
                            last_edge_inclusive=self.last_edge_inclusive,
                        )
                        weight_index += 1

        # merge output files
        self.output()["hists"].dump(histograms, formatter="pickle")


# overwrite class defaults
check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
CreateTriggerHistograms.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=CreateTriggerHistograms.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


class MergeTriggerHistograms(
    VariablesMixin,
    WeightProducerMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CreateTriggerHistograms=CreateTriggerHistograms,
    )

    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

    def _get_variables(self):
        if self.is_workflow():
            return self.as_branch()._get_variables()

        variables = self.variables

        # optional dynamic behavior: determine not yet created variables and require only those
        if self.only_missing:
            missing = self.output().count(existing=False, keys=True)[1]
            variables = sorted(missing, key=variables.index)

        return variables

    def workflow_requires(self):
        reqs = super().workflow_requires()

        if not self.pilot:
            variables = self._get_variables()
            if variables:
                reqs["hists"] = self.reqs.CreateTriggerHistograms.req_different_branching(
                    self,
                    branch=-1,
                    variables=tuple(variables),
                )
        return reqs

    def requires(self):
        variables = self._get_variables()
        if not variables:
            return []

        return self.reqs.CreateTriggerHistograms.req_different_branching(
            self,
            branch=-1,
            variables=tuple(variables),
            workflow="local",
        )

    def output(self):
        return {"hists": law.SiblingFileCollection({
            variable_name: self.target(f"hist__{variable_name}.pickle")
            for variable_name in self.variables
        })}

    @law.decorator.log
    def run(self):
        # preare inputs and outputs
        inputs = self.input()["collection"]
        outputs = self.output()

        # load input histograms
        hists = [
            inp["hists"].load(formatter="pickle")
            for inp in self.iter_progress(inputs.targets.values(), len(inputs), reach=(0, 50))
        ]

        # create a separate file per output variable
        variable_names = list(hists[0].keys())
        for variable_name in self.iter_progress(variable_names, len(variable_names), reach=(50, 100)):
            self.publish_message(f"merging histograms for '{variable_name}'")

            variable_hists = [h[variable_name] for h in hists]
            merged = sum(variable_hists[1:], variable_hists[0].copy())
            outputs["hists"][variable_name].dump(merged, formatter="pickle")

        # optionally remove inputs
        if self.remove_previous:
            inputs.remove()


MergeTriggerHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeTriggerHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class MergeShiftedTriggerHistograms(
    MergeShiftedHistograms,
    VariablesMixin,
    ShiftSourcesMixin,
    WeightProducerMixin,
    MLModelsMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # disable the shift parameter
    shift = None
    effective_shift = None
    allow_empty_shift = True

    # allow only running on nominal
    allow_empty_shift_sources = True

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeTriggerHistograms=MergeTriggerHistograms,
    )

    def create_branch_map(self):
        # create a dummy branch map so that this task could as a job
        return {0: None}

    def workflow_requires(self):
        reqs = super().workflow_requires()
        return reqs

    def requires(self):
        reqs = super().workflow_requires()
        return reqs

    def store_parts(self):
        parts = super().store_parts()
        return parts

    def output(self):
        return {"hists": law.SiblingFileCollection({
            variable_name: self.target(f"shifted_hist__{variable_name}.pickle")
            for variable_name in self.variables
        })}

    @law.decorator.log
    def run(self):
        # preare inputs and outputs
        inputs = self.input()
        outputs = self.output()["hists"].targets

        for variable_name, outp in self.iter_progress(outputs.items(), len(outputs)):
            self.publish_message(f"merging histograms for '{variable_name}'")

            # load hists
            variable_hists = [
                coll["hists"].targets[variable_name].load(formatter="pickle")
                for coll in inputs.values()
            ]

            # merge and write the output
            merged = sum(variable_hists[1:], variable_hists[0].copy())
            outp.dump(merged, formatter="pickle")


MergeShiftedTriggerHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeShiftedTriggerHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets"],
)


class PlotTriggerVariablesBase(
    HistHookMixin,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    CategoriesMixin,
    MLModelsMixin,
    WeightProducerMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeTriggerHistograms=MergeTriggerHistograms,
    )

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    @abstractmethod
    def get_plot_shifts(self):
        return

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())

        # copy process instances once so that their auxiliary data fields can be used as a storage
        # for process-specific plot parameters later on in plot scripts without affecting the
        # original instances
        fake_root = od.Process(
            name=f"{hex(id(object()))[2:]}",
            id="+",
            processes=list(map(self.config_inst.get_process, self.processes)),
        ).copy()
        process_insts = list(fake_root.processes)
        fake_root.processes.clear()

        # prepare other config objects
        variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        category_inst = self.config_inst.get_category(self.branch_data.category)
        leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
        sub_process_insts = {
            process_inst: [sub for sub, _, _ in process_inst.walk_processes(include_self=True)]
            for process_inst in process_insts
        }

        # histogram data per process copy
        hists = {}

        with self.publish_step(f"plotting {self.branch_data.variable} in {category_inst.name}"):
            for dataset, inp in self.input().items():
                dataset_inst = self.config_inst.get_dataset(dataset)
                h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")

                # loop and extract one histogram per process
                for process_inst in process_insts:
                    # skip when the dataset is already known to not contain any sub process
                    if not any(
                        dataset_inst.has_process(sub_process_inst.name)
                        for sub_process_inst in sub_process_insts[process_inst]
                    ):
                        continue

                    # select processes and reduce axis
                    h = h_in.copy()
                    h = h[{
                        "process": [
                            hist.loc(p.id)
                            for p in sub_process_insts[process_inst]
                            if p.id in h.axes["process"]
                        ],
                    }]
                    h = h[{"process": sum}]

                    # add the histogram
                    if process_inst in hists:
                        hists[process_inst] += h
                    else:
                        hists[process_inst] = h

            # there should be hists to plot
            if not hists:
                raise Exception(
                    "no histograms found to plot; possible reasons:\n"
                    "  - requested variable requires columns that were missing during histogramming\n"
                    "  - selected --processes did not match any value on the process axis of the input histogram",
                )

            # update histograms using custom hooks
            hists = self.invoke_hist_hooks(hists)

            # add new processes to the end of the list
            for process_inst in hists:
                if process_inst not in process_insts:
                    process_insts.append(process_inst)

            # axis selections and reductions, including sorting by process order
            _hists = OrderedDict()
            for process_inst in sorted(hists, key=process_insts.index):
                h = hists[process_inst]
                # selections
                h = h[{
                    "category": [
                        hist.loc(c.id)
                        for c in leaf_category_insts
                        if c.id in h.axes["category"]
                    ],
                    "shift": [
                        hist.loc(s.id)
                        for s in plot_shifts
                        if s.id in h.axes["shift"]
                    ],
                }]
                # reductions
                h = h[{"category": sum}]
                # store
                _hists[process_inst] = h
            hists = _hists

            # call the plot function
            fig, _ = self.call_plot_func(
                self.plot_function,
                hists=hists,
                config_inst=self.config_inst,
                category_inst=category_inst.copy_shallow(),
                variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                **self.get_plot_parameters(),
            )

            # save the plot
            for outp in self.output()["plots"]:
                outp.dump(fig, formatter="mpl")


class PlotTriggerVariablesBaseSingleShift(
    PlotTriggerVariablesBase,
    ShiftTask,
):
    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        PlotTriggerVariablesBase.reqs,
        MergeTriggerHistograms=MergeTriggerHistograms,
    )

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for var_name in sorted(self.variables)
            for cat_name in sorted(self.categories)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # no need to require merged histograms since each branch already requires them as a workflow
        if self.workflow == "local":
            reqs.pop("merged_hists", None)

        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeTriggerHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()

        parts["processes"] = f"proc_{self.processes_repr}"
        parts["category"] = f"cat_{self.branch_data.category}"
        parts["variable"] = f"var_{self.branch_data.variable}"

        hooks_repr = self.hist_hooks_repr
        if hooks_repr:
            parts["hook"] = f"hooks_{hooks_repr}"

        return parts

    def output(self):
        return {
            "plots": [self.target(name) for name in self.get_plot_names("plot")],
        }

    def store_parts(self):
        parts = super().store_parts()
        if "shift" in parts:
            parts.insert_before("datasets", "shift", parts.pop("shift"))
        return parts

    def get_plot_shifts(self):
        return [self.global_shift_inst]


class PlotTriggerVariables1D(
    PlotTriggerVariablesBaseSingleShift,
    PlotBase1D,
):
    plot_function = PlotBase.plot_function.copy(
        default="alljets.plotting.trigger_eff_closure.plot_efficiencies",
        add_default_to_description=True,
    )
