# coding: utf-8

"""
Configuration of the topmass_alljets analysis.
"""

from alljets.config.variables import add_variables
from alljets.config.categories import add_categories
from alljets.hist_hooks.bkg import add_hooks as add_qcd_hooks
import functools
import importlib
import os

import law
import order as od
from alljets.config.configs_aj import add_config
from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)
from columnflow.util import DotDict, maybe_import
from scinum import Number
from typing import Optional


ak = maybe_import("awkward")


#
# the main analysis object
#

analysis_aj = ana = od.Analysis(
    name="analysis_aj",
    id=1,
)

# Add hist hooks
analysis_aj.x.hist_hooks = DotDict()
# QCD hist hooks
add_qcd_hooks(analysis_aj)

# analysis-global versions
# (see cfg.x.versions below for more info)
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    law.config.get("analysis", "default_columnar_sandbox"),
    # "$AJ_BASE/sandboxes/example.sh"
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    "$AJ_BASE/sandboxes/cmsswtest.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("AJ_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}
# setup configs


def add_lazy_config(
    *,
    campaign_module: str,
    campaign_attr: str,
    config_name: str,
    config_id: int,
    add_limited: bool = True,
    **kwargs,
):
    def create_factory(
        config_id: int,
        config_name_postfix: str = "",
        limit_dataset_files: Optional[int] = None
    ):
        def factory(configs: od.UniqueObjectIndex):
            # import the campaign
            mod = importlib.import_module(campaign_module)
            campaign = getattr(mod, campaign_attr)

            return add_config(
                analysis_aj,
                campaign.copy(),
                config_name=config_name + config_name_postfix,
                config_id=config_id,
                limit_dataset_files=limit_dataset_files,
                **kwargs,
            )

        return factory
    print(f"added config:{config_name}")

    analysis_aj.configs.add_lazy_factory(
        config_name, create_factory(config_id))
    if add_limited:
        analysis_aj.configs.add_lazy_factory(
            f"{config_name}_limited", create_factory(
                config_id + 200, "_limited", 2)
        )


# 2017,
add_lazy_config(
    campaign_module="cmsdb.campaigns.run2_2017_nano_v9",
    campaign_attr="campaign_run2_2017_nano_v9",
    config_name="2017_v9",
    config_id=2017,
    add_limited=True,
)
