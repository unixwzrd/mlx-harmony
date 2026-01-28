#!/usr/bin/env python3

import json
import os
import signal
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import TextIO


# Signal handler to close the file gracefully
def signal_handler(signum: int, frame: object | None) -> None:
    if datafile:
        datafile.close()
    sys.exit(0)


# Initialize argparse
parser = ArgumentParser(description="Filter vm_stat output.")
parser.add_argument("-d", "--datafile", type=str, nargs='?', const=f"{os.getenv('CONDA_DEFAULT_ENV', 'default')}-timing.json", default=None, help="Specify the datafile to write the output to.")
parser.add_argument("-s", "--silent", action="store_true", help="Suppress STDOUT.")
parser.add_argument("-t", "--tsv", action="store_true", help="Output TSV instead of JSON.")
parser.add_argument(
    "-b",
    "--bytes",
    action="store_true",
    help="Convert page counts to bytes using --page-bytes.",
)
parser.add_argument(
    "--page-bytes",
    type=int,
    default=0,
    help="Page size in bytes for --bytes conversion (default: auto-detect).",
)
args = parser.parse_args()

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _parse_metric_value(value: str) -> int:
    cleaned = value.strip()
    if not cleaned:
        return 0
    multiplier = 1
    if cleaned.endswith("K"):
        multiplier = 1_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    return int(cleaned) * multiplier


def filter_vm_stat(datafile: TextIO | None, silent: bool, tsv: bool, as_bytes: bool, page_bytes: int) -> None:
    first_header: bool = True
    lines_skipped: int = 0
    headers: list[str] = []
    wrote_tsv_header: bool = False
    detected_page_bytes: int | None = None
    effective_page_bytes: int | None = None

    for line in sys.stdin:

        if "Mach Virtual Memory Statistics:" in line:
            if "page size of" in line and "bytes" in line:
                try:
                    page_str = line.split("page size of", 1)[1].split("bytes", 1)[0].strip()
                    detected_page_bytes = int(page_str)
                except (IndexError, ValueError):
                    detected_page_bytes = None
            if page_bytes and page_bytes > 0:
                effective_page_bytes = page_bytes
            else:
                effective_page_bytes = detected_page_bytes
            lines_skipped = 0  # Reset the skipped lines counter
            continue  # Skip this line

        lines_skipped += 1  # Increment the skipped lines counter

        if lines_skipped == 1:
            if first_header:
                if not silent:
                    print(line[:-1])
                headers = line.split()
                first_header = False  # Mark that the first header has been printed
                if datafile and tsv and not wrote_tsv_header:
                    output_headers = headers
                    if as_bytes:
                        output_headers = [f"{header}_bytes" for header in headers]
                    datafile.write("datetime\t" + "\t".join(output_headers) + "\n")
                    datafile.flush()
                    wrote_tsv_header = True
            continue  # Skip this line for subsequent headers

        if lines_skipped == 2:
            continue  # Skip this line, it's the third line of any header

        # Write the line to STDOUT if it's not suppressed
        if not silent:
            print(line[:-1])

        if datafile:
            # Combine the current date and time into a single field
            current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data_values = line.strip().split()[2:]  # Skip the first two fields, they are not data

            if as_bytes and not effective_page_bytes:
                raise ValueError(
                    "Page size could not be determined from vm_stat output; "
                    "pass --page-bytes to set it explicitly."
                )

            if tsv:
                values = [current_datetime]
                for value in data_values:
                    numeric_value = _parse_metric_value(value)
                    if as_bytes:
                        numeric_value *= effective_page_bytes
                    values.append(str(numeric_value))
                datafile.write("\t".join(values) + "\n")
                datafile.flush()
                continue

            # Create a JSON object for this line of data
            data_dict: dict[str, int | str] = {"datetime": current_datetime}
            for header, value in zip(headers, data_values):
                numeric_value = _parse_metric_value(value)
                if as_bytes:
                    numeric_value *= effective_page_bytes
                    header = f"{header}_bytes"
                data_dict[header] = numeric_value  # Assuming all data values are integers

            json.dump(data_dict, datafile)
            datafile.write('\n')
            datafile.flush()

# Usage
if __name__ == "__main__":
    datafile: TextIO | None = None
    if args.datafile:
        datafile = open(args.datafile, 'w', encoding="utf-8", buffering=1)

    filter_vm_stat(datafile, args.silent, args.tsv, args.bytes, args.page_bytes)

    if datafile:
        datafile.close()
