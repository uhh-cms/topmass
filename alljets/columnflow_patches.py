# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os

import law
from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_bundle_repo_exclude_files():
    from columnflow.tasks.framework.remote import BundleRepo

    # get the relative path to CF_BASE
    cf_rel = os.path.relpath(os.environ["CF_BASE"], os.environ["AJ_BASE"])

    # amend exclude files to start with the relative path to CF_BASE
    exclude_files = [os.path.join(cf_rel, path) for path in BundleRepo.exclude_files]

    # add additional files
    exclude_files.extend([
        "docs", "tests", "data", "assets", ".law", ".setups", ".data", ".github",
    ])

    # overwrite them
    BundleRepo.exclude_files[:] = exclude_files

    logger.debug("patched exclude_files of cf.BundleRepo")


@memoize
def patch_selector_steps_names():
    from columnflow.tasks.framework.mixins import SelectorClassMixin

    SelectorClassMixin.selector_steps_order_sensitive = True


@memoize
def patch_shifted_variables_from_model_dashed_variables():
    """
    PlotVariablesBaseShiftsFromModel.resolve_param_values_post_init re-resolves ``variables`` via a
    literal find_config_objects lookup on the raw CLI string, which fails for dashed
    multi-dimensional names (e.g. "varA-varB") since no single od.Variable is ever registered under
    that combined name -- only the two parts are. The base VariablesMixin call one line above already
    resolves those combos correctly (via split_multi_variable / find_config_objects per-part /
    itertools.product), so this patch just keeps that result instead of overwriting it whenever the
    user passed --variables explicitly. Falls back to the inference model's own variable list exactly
    as before when --variables is empty.
    """
    from columnflow.tasks import plotting as cf_plotting
    import order as od

    base_cls = cf_plotting.PlotVariablesBaseShiftsFromModel

    def resolve_param_values_post_init(cls, params):
        categories_orig = params.get("categories")
        params = super(base_cls, cls).resolve_param_values_post_init(params)
        params["categories"] = categories_orig

        config_insts = params.get("config_insts")
        inference_model_inst = params.get("inference_model_inst")
        if config_insts and inference_model_inst:
            combined_config_data = params[cls._combined_config_data_attr]

            # only fall back to the inference model's own variable list when the user passed
            # nothing on the CLI; otherwise keep what VariablesMixin already resolved above
            if not params.get("variables"):
                variables = sorted(law.util.make_unique(law.util.flatten(
                    config_data["variables"] for config_data in combined_config_data.values()
                )))
                params["variables"] = tuple(variables)

            category_map = cls.resolve_model_category_map(
                inference_model_inst=inference_model_inst,
                config_insts=config_insts,
                categories=params.get("categories"),
            )
            params["categories"] = tuple(sorted(category_map.keys()))
            params["category_map"] = {cat_name: category_map[cat_name] for cat_name in params["categories"]}

            if (merge_processes := params.get("merge_processes")):
                merge_processes = cls.find_config_objects(
                    names=merge_processes,
                    container=config_insts,
                    object_cls=od.Process,
                    groups_str="process_groups",
                    multi_strategy="intersection",
                )
                params["merge_processes"] = tuple(merge_processes)

        return params

    base_cls.resolve_param_values_post_init = classmethod(resolve_param_values_post_init)

    logger.debug("patched resolve_param_values_post_init of cf.PlotVariablesBaseShiftsFromModel")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    patch_selector_steps_names()
    patch_shifted_variables_from_model_dashed_variables()
