# coding: utf-8

from __future__ import annotations

from typing import Dict, Optional, List, Any
import law

from columnflow.util import maybe_import, dev_sandbox
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.base import ShiftTask
from columnflow.tasks.framework.plotting import PlotBase2D
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin,
    DatasetsProcessesMixin,
    SelectorClassMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.selection import MergeSelectionStats

from alljets.tasks.base import AJTask
from alljets.plotting.btag_eff import btag_efficiency, FLAVOR_LABELS

# Lazy imports for heavy libraries
hist = maybe_import("hist")
np = maybe_import("numpy")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")


class _PlotBtagEfficiencyBase(
    AJTask,
    DatasetsProcessesMixin,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ShiftTask,
    PlotBase2D,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Base class for b-tag efficiency plotting tasks.
    Provides common infrastructure for workflow and remote execution.
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Mark as configurable via command line
    single_config = True

    # Add plot_suffix parameter
    plot_suffix = law.CSVParameter(
        default=[],
        description="suffix to append to plot output filenames; pass multiple values to create multiple variants",
        brace_expand=True,
    )


class PlotBtagEfficiency(_PlotBtagEfficiencyBase):
    """
    Task to merge b-tag counting histograms, derive efficiency maps,
    and create 2D efficiency plots for different jet flavors.

    This task aggregates selection statistics from multiple datasets,
    groups them by physics process, and generates efficiency plots
    showing b-tagging performance as a function of jet kinematics.
    """

    # Task identification
    task_namespace = "cf"

    # Override default plot function to use btag_efficiency from alljets
    plot_function = PlotBase2D.plot_function.copy(
        default="alljets.plotting.btag_eff.btag_efficiency",
        add_default_to_description=True,
    )

    # Upstream task that provides selection statistics
    resolution_task_cls = MergeSelectionStats

    # Task requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeSelectionStats=MergeSelectionStats,
    )

    # Key used to retrieve b-tag working point counts from histograms
    hist_key = "btag_wp_counts"

    # Mapping of individual datasets to physics process groups
    # Each group will be plotted separately to compare efficiency across processes
    process_groups = {
        "tt": ["tt_fh_powheg", "tt_sl_powheg", "tt_dl_powheg"],
        "st": [
            "st_tchannel_t_4f_powheg",
            "st_tchannel_tbar_4f_powheg",
            "st_twchannel_t_powheg",
            "st_twchannel_tbar_powheg",
            "st_schannel_lep_4f_amcatnlo",
            "st_schannel_had_4f_amcatnlo",
        ],
        "qcd": [
            "qcd_ht300to500_madgraph",
            "qcd_ht500to700_madgraph",
            "qcd_ht700to1000_madgraph",
            "qcd_ht1000to1500_madgraph",
            "qcd_ht1500to2000_madgraph",
            "qcd_ht2000toinf_madgraph",
        ],
    }

    # Flavors to plot: 0=light/unknown, 4=charm, 5=bottom
    PLOT_FLAVORS = [0, 4, 5]

    # Output file naming
    OUTPUT_BASE = "btag_efficiency_map"
    EFF_HISTS_FILENAME = "btag_efficiency_hists.pickle"

    def create_branch_map(self) -> Dict[int, None]:
        """
        Create a single branch for this task.
        Returns a mapping with a single branch index.
        """
        return {0: None}

    def workflow_requires(self) -> Dict[str, Any]:
        """
        Define dependencies for the entire workflow.

        Returns:
            Dictionary mapping dataset names to their required MergeSelectionStats tasks.
        """
        reqs = super().workflow_requires()
        reqs["selection_stats"] = self._create_dataset_requirements()
        return reqs

    def requires(self) -> Dict[str, Any]:
        """
        Define immediate task dependencies.

        Returns:
            Dictionary mapping dataset names to their required MergeSelectionStats tasks.
        """
        return self._create_dataset_requirements()

    def _create_dataset_requirements(self) -> Dict[str, Any]:
        """
        Helper method to create MergeSelectionStats requirements for all datasets.

        Returns:
            Dictionary mapping dataset names to task requirements.
        """
        return {
            dataset_name: self.reqs.MergeSelectionStats.req_different_branching(
                self,
                dataset=dataset_name,
                shift=self.shift,
                branch=-1 if self.is_workflow() else 0,
            )
            for dataset_name in self.datasets
        }

    def output(self) -> Dict[str, law.target.BaseTarget]:
        """
        Define output targets for this task.
        """
        # Build suffix string
        suffix_str = ""
        if self.plot_suffix:
            # If plot_suffix is a list, join with underscore
            suffix_str = "_" + "_".join(self.plot_suffix)

        outputs = {
            "eff_hists": self.target(self.EFF_HISTS_FILENAME),
        }

        # Create output targets for each flavor with suffix
        for flavor in self.PLOT_FLAVORS:
            flavor_name = self._get_flavor_name(flavor)
            outputs[f"plots_{flavor}"] = self.target(
                f"{self.OUTPUT_BASE}_{flavor_name}{suffix_str}.pdf",
            )

        return outputs

    def _get_flavor_name(self, flavor: int) -> str:
        """
        Get a safe filename-compatible name for a flavor.

        Args:
            flavor: Flavor integer (0, 4, 5)

        Returns:
            Sanitized flavor name string.
        """
        name = FLAVOR_LABELS.get(flavor, str(flavor))
        # Remove characters that might cause issues in filenames
        return name.replace(" ", "_").replace("(", "").replace(")", "")

    def _group_dataset_name(self, dataset_name: str) -> Optional[str]:
        """
        Determine which physics process group a dataset belongs to.

        Args:
            dataset_name: Name of the dataset to classify

        Returns:
            Group name if matched, None otherwise.
        """
        for group_name, patterns in self.process_groups.items():
            if any(law.util.multi_match(dataset_name, pattern) for pattern in patterns):
                return group_name
        return None

    def _load_and_group_counts(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, hist.Hist]:
        """
        Load and aggregate histograms from multiple datasets.

        Args:
            inputs: Dictionary of input targets from upstream tasks

        Returns:
            Dictionary mapping group names to aggregated histograms.
        """
        grouped_counts = {}

        for dataset_name, inp in inputs.items():
            # Load histograms from pickle
            hists = inp["hists"].load(formatter="pickle")
            counts = hists[self.hist_key]

            # Determine group for this dataset
            group_name = self._group_dataset_name(dataset_name)
            if group_name is None:
                self.logger.warning(
                    f"Dataset {dataset_name} not matched to any group, skipping",
                )
                continue

            # Aggregate histograms by group
            if group_name in grouped_counts:
                grouped_counts[group_name] += counts
            else:
                grouped_counts[group_name] = counts

        return grouped_counts

    def _prepare_plot_parameters(self) -> Dict[str, Any]:
        """
        Prepare parameters for efficiency plotting.

        Returns:
            Dictionary of plot parameters with default values.
        """
        # Get base parameters from parent class
        plot_params = self.get_plot_parameters()
        # Add plot suffix to parameters if it should affect the plot
        if hasattr(self, "plot_suffix") and self.plot_suffix:
            plot_params["plot_suffix"] = "_".join(self.plot_suffix)

        # Override specific parameters for b-tag efficiency plots
        plot_params.update({
            "pt_max": None,  # No pt cut for efficiency plots
            "wp_label": "tight",  # Label for working point
            "n_workers": 1,  # Avoid multiprocessing issues
        })

        return plot_params

    def _generate_flavor_plot(
        self,
        flavor: int,
        grouped_counts: Dict[str, hist.Hist],
        category_inst: Any,
        shift_insts: List[Any],
        plot_params: Dict[str, Any],
    ) -> None:
        """
        Generate and save an efficiency plot for a specific flavor.

        Args:
            flavor: Flavor integer (0, 4, 5)
            grouped_counts: Dictionary of grouped histograms
            category_inst: Category instance for plotting
            shift_insts: List of shift instances
            plot_params: Plotting parameters
        """
        flavor_name = self._get_flavor_name(flavor)
        output_target = self.output()[f"plots_{flavor}"]

        self.logger.info(
            f"Generating efficiency plot for flavor {flavor} ({flavor_name})",
        )

        try:
            # Generate the efficiency plot
            fig, axes = btag_efficiency(
                hists=grouped_counts,
                config_inst=self.config_inst,
                category_inst=category_inst,
                shift_insts=shift_insts,
                flavor=flavor,
                **plot_params,
            )

            # Ensure output directory exists and save
            output_target.parent.touch()
            fig.savefig(output_target.path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            self.logger.info(
                f"Saved efficiency plot for flavor {flavor} to {output_target.path}",
            )

        except Exception as e:
            self.logger.error(f"Error generating plot for flavor {flavor}: {e}")
            self.logger.error("Full traceback:", exc_info=True)
            raise

    @law.decorator.log
    def run(self) -> None:
        """
        Main execution for the task
        """
        try:
            # Load and aggregate histograms
            self.logger.info("Loading and grouping histograms from datasets...")
            inputs = self.input()
            grouped_counts = self._load_and_group_counts(inputs)

            if not grouped_counts:
                raise RuntimeError("No valid histogram groups found for any dataset")

            self.logger.info(f"Found {len(grouped_counts)} process groups: {list(grouped_counts.keys())}")

            # Prepare plotting parameters
            plot_params = self._prepare_plot_parameters()

            # Get category instance (using inclusive category for all events)
            category_inst = self.config_inst.get_category("incl")
            shift_insts = [self.global_shift_inst]

            # Generate plots for each flavor
            for flavor in self.PLOT_FLAVORS:
                self._generate_flavor_plot(
                    flavor=flavor,
                    grouped_counts=grouped_counts,
                    category_inst=category_inst,
                    shift_insts=shift_insts,
                    plot_params=plot_params,
                )

            # Step 4: Save combined histograms
            self.logger.info("Saving combined histograms...")
            self.output()["eff_hists"].dump(grouped_counts, formatter="pickle", cache=False)

            self.logger.info("All flavor efficiency plots generated successfully")

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self.logger.error("Full traceback:", exc_info=True)
            raise
