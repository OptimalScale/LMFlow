#!/bin/bash
#
# A useful script when `regression_test/${test_set}/new_${test_set}.sh` is
# provided.

function main() {
  local project_dir=$(cd "$(dirname $0)/.."; pwd)
  local bash_name=$(basename $0)
  local testset_dir=${project_dir}/regression_test
  local tmp_dir=${project_dir}/tmp/regression_test

  mkdir -p ${tmp_dir}

  local RED="\033[0;31m"
  local GREEN="\033[0;32m"
  local END_COLOR="\033[0m"

  for testset in ${testset_dir}/test-finetune-*; do
    if [ ! -d ${testset} ]; then
      continue
    fi

    echo ${testset}
    local test_name=$(basename ${testset})

    # Regression tests scripts are assumed to be "new_${test_name}.sh"
    local test_script=${testset}/new_*.sh       
    local log_file=${tmp_dir}/${test_name}.log

    echo "$(date): running ${test_name}... (see \"${log_file}\" for details)"

    # Runs regression test
    ${test_script} > ${log_file} 2>&1 
    if [ $? -ne 0 ]; then
      echo -e "  ${RED}Failed${END_COLOR}"
      break
    else
      echo -e "  ${GREEN}Passed${END_COLOR}"
    fi
  done
}

main "$@"
