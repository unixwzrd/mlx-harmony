#!/usr/bin/env bash
set -euo pipefail

VM_PGID=""
VM_KILL_MODE="pid"

abs_path() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
    return
  fi
  if command -v readlink >/dev/null 2>&1; then
    if readlink -f "$1" >/dev/null 2>&1; then
      readlink -f "$1"
      return
    fi
  fi
  local target="$1"
  local dir
  dir="$(cd "$(dirname "$target")" && pwd -P)"
  echo "${dir}/$(basename "$target")"
}

write_run_env() {
  local meta_dir="$1"
  shift
  mkdir -p "$meta_dir"
  {
    for kv in "$@"; do
      echo "$kv"
    done
  } > "${meta_dir}/run.env"
}

start_vmstat() {
  local interval="$1"
  local filter_script="$2"
  local out_tsv="$3"
  local stderr_path="$4"
  shift 4
  local filter_args=("$@")

  : >"$stderr_path"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -c "
      set -euo pipefail
      vm_stat ${interval} | '${filter_script}' -d '${out_tsv}' ${filter_args[*]}
    " >/dev/null 2>>"${stderr_path}" &
    VM_PGID="$!"
    VM_KILL_MODE="pgid"
  else
    bash -c "
      set -euo pipefail
      vm_stat ${interval} | '${filter_script}' -d '${out_tsv}' ${filter_args[*]}
    " >/dev/null 2>>"${stderr_path}" &
    VM_PGID="$!"
    VM_KILL_MODE="pid"
  fi
}

stop_vmstat() {
  if [[ -n "${VM_PGID}" ]]; then
    if [[ "${VM_KILL_MODE}" == "pgid" ]]; then
      kill -- -"${VM_PGID}" >/dev/null 2>&1 || true
    else
      pkill -P "${VM_PGID}" >/dev/null 2>&1 || true
      kill "${VM_PGID}" >/dev/null 2>&1 || true
    fi
    wait "${VM_PGID}" >/dev/null 2>&1 || true
    VM_PGID=""
  fi
}

write_timings_tsv() {
  local debug_log="$1"
  local out_tsv="$2"
  if [[ -f "$debug_log" ]]; then
    grep ^TIMING_ "$debug_log" > "$out_tsv" || true
  fi
}

merge_timings() {
  local merge_script="$1"
  local timings_tsv="$2"
  local vm_stat_tsv="$3"
  local output_tsv="$4"
  if [[ ! -f "$timings_tsv" ]]; then
    return
  fi
  if [[ -x "$merge_script" || -f "$merge_script" ]] && [[ -f "$vm_stat_tsv" ]]; then
    "$merge_script" \
      --timings "$timings_tsv" \
      --vm-stat "$vm_stat_tsv" \
      --output "$output_tsv" \
      || true
  fi
}

plot_tps_vs_wired() {
  local plot_script="$1"
  local merged_tsv="$2"
  local out_png="$3"
  if [[ -f "$merged_tsv" ]] && [[ -x "$plot_script" || -f "$plot_script" ]]; then
    "$plot_script" "$merged_tsv" --out "$out_png" || true
  fi
}
