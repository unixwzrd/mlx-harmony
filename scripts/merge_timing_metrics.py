#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class TimingRow:
    timestamp_key: str
    fields: dict[str, str]


def _parse_vm_stat_timestamp(value: str) -> datetime:
    local_tz = datetime.now().astimezone().tzinfo
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=local_tz).astimezone(UTC)


def _parse_timing_timestamp(iso_value: str, unix_value: str | None) -> datetime:
    cleaned = iso_value.strip().replace("Z", "+00:00")
    if cleaned:
        try:
            return datetime.fromisoformat(cleaned).astimezone(UTC)
        except ValueError:
            pass
    if unix_value:
        try:
            return datetime.fromtimestamp(float(unix_value), tz=UTC)
        except ValueError:
            pass
    raise ValueError("Invalid timing timestamp values.")


def _round_to_seconds(dt: datetime, bucket_seconds: int) -> datetime:
    if bucket_seconds <= 1:
        return dt.replace(microsecond=0)
    epoch = int(dt.timestamp())
    rounded = epoch - (epoch % bucket_seconds)
    return datetime.fromtimestamp(rounded, tz=dt.tzinfo or UTC)


def _to_join_timezone(dt: datetime, join_timezone: str) -> datetime:
    if join_timezone == "utc":
        return dt.astimezone(UTC)
    local_tz = datetime.now().astimezone().tzinfo
    return dt.astimezone(local_tz)


def _load_timing_rows(
    path: Path,
    bucket_seconds: int,
    timing_offset_seconds: int,
    join_timezone: str,
) -> tuple[list[str], dict[str, list[TimingRow]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header: list[str] = []
        rows: dict[str, list[TimingRow]] = {}
        for raw in reader:
            if not raw:
                continue
            if raw[0] == "TIMING_STATS_HEADER":
                header = raw[1:]
                continue
            if raw[0] != "TIMING_STATS":
                continue
            if not header:
                raise ValueError("TIMING_STATS_HEADER not found in timings file.")
            values = raw[1:]
            fields = dict(zip(header, values, strict=False))
            timestamp_iso = fields.get("timestamp_iso") or ""
            timestamp_unix = fields.get("timestamp_unix")
            dt = _parse_timing_timestamp(timestamp_iso, timestamp_unix)
            if timing_offset_seconds:
                dt = dt + timedelta(seconds=timing_offset_seconds)
            dt = _to_join_timezone(dt, join_timezone)
            rounded = _round_to_seconds(dt, bucket_seconds)
            key = rounded.strftime("%Y-%m-%d %H:%M:%S")
            entry = TimingRow(timestamp_key=key, fields=fields)
            rows.setdefault(key, []).append(entry)
    return header, rows


def _load_vm_rows(
    path: Path,
    bucket_seconds: int,
    vm_offset_seconds: int,
    join_timezone: str,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header: list[str] = []
        rows: list[dict[str, str]] = []
        for idx, raw in enumerate(reader):
            if not raw:
                continue
            if idx == 0:
                header = raw
                continue
            row = dict(zip(header, raw, strict=False))
            datetime_value = row.get("datetime", "")
            if datetime_value:
                dt = _parse_vm_stat_timestamp(datetime_value)
                if vm_offset_seconds:
                    dt = dt + timedelta(seconds=vm_offset_seconds)
                dt = _to_join_timezone(dt, join_timezone)
                rounded = _round_to_seconds(dt, bucket_seconds)
                row["__join_key__"] = rounded.strftime("%Y-%m-%d %H:%M:%S")
                row["__dt__"] = dt
            rows.append(row)
    if not header or header[0] != "datetime":
        raise ValueError("vm_stat TSV must start with a datetime column.")
    return header, rows


def _flatten_rows(
    vm_rows: Iterable[dict[str, str]],
    timing_map: dict[str, list[TimingRow]],
    timing_header: list[str],
    include_timing_only: bool,
    nearest_seconds: int,
) -> tuple[list[str], list[list[str]]]:
    output_header = ["datetime"] + timing_header
    output_header.extend(
        [
            key
            for key in next(iter(vm_rows)).keys()
            if key not in ("datetime", "__join_key__", "__dt__")
        ]
    )
    output_rows: list[list[str]] = []
    seen_timing_keys: set[str] = set()
    vm_rows_list = list(vm_rows)
    for vm_row in vm_rows_list:
        key = vm_row.get("__join_key__") or vm_row.get("datetime", "")
        timing_entries = timing_map.get(key)
        if not timing_entries:
            blank_timing = ["" for _ in timing_header]
            vm_values = [vm_row.get(col, "") for col in output_header if col not in ("datetime", *timing_header)]
            output_rows.append([key] + blank_timing + vm_values)
            continue
        seen_timing_keys.add(key)
        for entry in timing_entries:
            timing_values = [entry.fields.get(col, "") for col in timing_header]
            vm_values = [vm_row.get(col, "") for col in output_header if col not in ("datetime", *timing_header)]
            output_rows.append([key] + timing_values + vm_values)
    if include_timing_only:
        blank_vm = ["" for _ in output_header if _ not in ("datetime", *timing_header)]
        for key, entries in timing_map.items():
            if key in seen_timing_keys:
                continue
            for entry in entries:
                timing_values = [entry.fields.get(col, "") for col in timing_header]
                vm_values = blank_vm
                if nearest_seconds > 0:
                    target_dt = _parse_timing_timestamp(
                        entry.fields.get("timestamp_iso", ""),
                        entry.fields.get("timestamp_unix"),
                    )
                    best_row = None
                    best_delta = None
                    for vm_row in vm_rows_list:
                        vm_dt = vm_row.get("__dt__")
                        if not isinstance(vm_dt, datetime):
                            continue
                        delta = abs((vm_dt - target_dt).total_seconds())
                        if best_delta is None or delta < best_delta:
                            best_delta = delta
                            best_row = vm_row
                    if best_row is not None and best_delta is not None and best_delta <= nearest_seconds:
                        vm_values = [
                            best_row.get(col, "")
                            for col in output_header
                            if col not in ("datetime", *timing_header)
                        ]
                output_rows.append([key] + timing_values + vm_values)
    return output_header, output_rows


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge timings-debug TSV with vm_stat TSV.")
    parser.add_argument(
        "--timings",
        type=Path,
        default=Path("stats/timings-debug.csv"),
        help="Path to timings-debug TSV.",
    )
    parser.add_argument(
        "--vm-stat",
        type=Path,
        default=Path("stats/vm_stat-timing.csv"),
        help="Path to vm_stat TSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stats/merged-timings-vm_stat.tsv"),
        help="Output TSV path.",
    )
    parser.add_argument(
        "--bucket-seconds",
        type=int,
        default=1,
        help="Round timing timestamps to this many seconds (default: 1).",
    )
    parser.add_argument(
        "--include-timing-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include timing rows that have no matching vm_stat sample (default: true).",
    )
    parser.add_argument(
        "--timing-offset-seconds",
        type=int,
        default=0,
        help="Offset seconds applied to timings-debug timestamps before bucketing.",
    )
    parser.add_argument(
        "--vm-offset-seconds",
        type=int,
        default=0,
        help="Offset seconds applied to vm_stat timestamps before bucketing.",
    )
    parser.add_argument(
        "--join-timezone",
        choices=["local", "utc"],
        default="local",
        help="Timezone used for join keys (default: local).",
    )
    parser.add_argument(
        "--nearest-seconds",
        type=int,
        default=0,
        help="Attach nearest vm_stat sample to timing-only rows within this tolerance.",
    )
    args = parser.parse_args()

    timing_header, timing_rows = _load_timing_rows(
        args.timings, args.bucket_seconds, args.timing_offset_seconds, args.join_timezone
    )
    vm_header, vm_rows = _load_vm_rows(
        args.vm_stat, args.bucket_seconds, args.vm_offset_seconds, args.join_timezone
    )
    if not vm_rows:
        raise ValueError("vm_stat file has no data rows.")
    merged_header, merged_rows = _flatten_rows(
        vm_rows,
        timing_rows,
        timing_header,
        args.include_timing_only,
        args.nearest_seconds,
    )
    try:
        merged_rows.sort(
            key=lambda row: datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        pass
    _write_tsv(args.output, merged_header, merged_rows)


if __name__ == "__main__":
    main()
