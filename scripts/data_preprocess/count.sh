#!/bin/bash

function main() {
  local bashname=$(basename $0)
  local project_dir="$(cd "$(dirname "$0")"; pwd)/../.."
  local script_dir=${project_dir}/scripts/data_preprocess
  local tmp_dir=/tmp/data_preprocess

  if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
    echo "Usage: ./${bashname} [JSON_FILE_PATH|DIR_PATH]" >&2
    echo "Example: ./${bashname} data/alpaca/train" >&2
  fi

  local dst_file=$1
  
  if [ -d ${dst_file} ]; then
    local dst_dir=${dst_file}
    local num_sample=0
    for file in ${dst_dir}/*.json; do
      local count=$(cat ${file} | python3 ${script_dir}/count.py)
      local num_sample=$((num_sample + count))
      echo "${file}: ${count}"
    done
    echo "total: ${num_sample}"
  elif [ -f ${dst_file} ]; then
    cat ${dst_file} | python3 ${script_dir}/count.py
  else
    echo "ERROR: ${dst_file} not found!" >&2
    return 1
  fi
}

main "$@"
