#!/usr/bin/env bash

# Script that sets up a virtual env in $CF_VENV_PATH.
# For more info on functionality and parameters, see the generic setup script _setup_venv.sh.

action() {
    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the generic venv setup
    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    export CF_VENV_NAME="$( basename "${this_file%.sh}" )"
    export CF_VENV_REQUIREMENTS="${this_dir}/spanet.txt"
    source "${CF_BASE}/sandboxes/_setup_venv.sh" "$@"
    local spanet_dir=${CF_SOFTWARE_BASE}/SPANet
    [[ ! -d "$spanet_dir" ]] && git clone -b v2.3-fixes https://github.com/jolange/SPANet.git ${spanet_dir} && python -m pip install -e ${spanet_dir} --no-cache-dir
    return 0
}
action "$@"

