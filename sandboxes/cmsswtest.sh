#!/usr/bin/env bash

# Script that sets up a CMSSW environment in $CF_CMSSW_BASE.
# For more info on functionality and parameters, see the generic setup script _setup_cmssw.sh.

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # get the os version
    local os_version="$( cat /etc/os-release | grep VERSION_ID | sed -E 's/VERSION_ID="([0-9]+)(|\..*)"/\1/' )"

    # set variables and source the generic CMSSW setup
    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    export CF_SCRAM_ARCH="el9_amd64_gcc12"
    export CF_CMSSW_VERSION="CMSSW_13_3_0"
    export CF_CMSSW_ENV_NAME="$( basename "${this_file%.sh}" )"
    export CF_CMSSW_FLAG="1"  # increment when content changed

    # define custom install and setup functions
    cf_cmssw_custom_install() {
        # install a venv into ${CMSSW_BASE}/venvs, which is included by BundleCMSSWSandbox
        CF_VENV_BASE="${CMSSW_BASE}/venvs" cf_create_venv cmsswtest &&
        source "${CMSSW_BASE}/venvs/cmsswtest/bin/activate" "" &&
        pip install -r  "${CF_REPO_BASE}/sandboxes/cmsswtest.txt" --force-reinstall &&
        CF_VENV_BASE="${CMSSW_BASE}/venvs" cf_make_venv_relocatable cmsswtest
    }
    cf_cmssw_custom_setup() {
        source "${CMSSW_BASE}/venvs/cmsswtest/bin/activate" "" &&
        export PYTHONPATH="${PYTHONPATH}:/afs/desy.de/user/s/schaller/HiWi/topmass-alljets-kinfit/standaloneKinFitter/:$( python -c "import os; print(os.path.normpath('$( root-config --libdir )'))" )"
  export LD_LIBRARY_PATH=/afs/desy.de/user/s/schaller/HiWi/topmass-alljets-kinfit/standaloneKinFitter/lib/el9_amd64_gcc12/6.26.11/:/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_13_3_0/external/el9_amd64_gcc12/lib/:$LD_LIBRARY_PATH
        export LD_PRELOAD=/cvmfs/cms.cern.ch/el9_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/lib64/libstdc++.so.6
    }


    source "${CF_BASE}/sandboxes/_setup_cmssw.sh" "$@"
}
action "$@"
