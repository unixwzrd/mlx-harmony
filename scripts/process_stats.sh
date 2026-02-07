#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "Usage: $0 <debug_log> <metrics_dir> <plots_dir> <vm_stat_tsv> <merge_script> <plot_script> <profile_stats> <profile_svg>" >&2
  exit 1
fi

DEBUG_LOG="$1"
METRICS_DIR="$2"
PLOTS_DIR="$3"
VMSTAT_TSV="$4"
MERGE_SCRIPT="$5"
PLOT_SCRIPT="$6"
PROFILE_STATS="$7"
PROFILE_SVG="$8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/dataset_run_common.sh"

TIMINGS_TSV="${METRICS_DIR}/timings-debug.tsv"
MERGED_TSV="${METRICS_DIR}/merged-timings-vm_stat.tsv"
PLOT_OUT="${PLOTS_DIR}/tps_vs_wired.png"
PROFILE_TEXT="${METRICS_DIR}/profile.stats.txt"
PROFILE_DOT="${METRICS_DIR}/profile.dot"
PROFILE_METRICS_JSON="${METRICS_DIR}/profile.metrics.json"
PROFILE_STATIC_TXT="${METRICS_DIR}/profile.static.txt"

mkdir -p "$METRICS_DIR" "$PLOTS_DIR"

export DOT_FILTER="${DOT_FILTER:-1}"
export DOT_FILTER_SUBSTRING="${DOT_FILTER_SUBSTRING:-mlx_harmony}"
export DOT_FILTER_KEEP_FULL="${DOT_FILTER_KEEP_FULL:-0}"

write_timings_tsv "$DEBUG_LOG" "$TIMINGS_TSV"
merge_timings "$MERGE_SCRIPT" "$TIMINGS_TSV" "$VMSTAT_TSV" "$MERGED_TSV"
plot_tps_vs_wired "$PLOT_SCRIPT" "$MERGED_TSV" "$PLOT_OUT"

if [[ -f "$PROFILE_STATS" ]]; then
  scripts/process_profile_artifacts.py \
    --profile-output "$PROFILE_STATS" \
    --profile-text "$PROFILE_TEXT" \
    --profile-metrics-json "$(abs_path "$PROFILE_METRICS_JSON")" \
    --profile-static-txt "$(abs_path "$PROFILE_STATIC_TXT")" \
    --profile-dot "$PROFILE_DOT" \
    --profile-svg "$PROFILE_SVG" \
    --top 50 || true
fi
