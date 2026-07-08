#!/usr/bin/env python3
"""
Integration for the TIME(R)3233 portable stylus surface roughness tester.

Verified facts about this device (Beijing TIME High Technology; see
README.md in this folder for sources): it's a stylus-type tester with a
50mm traversing length and an 800um vertical range, outputs Ra/Rz/Rq/and
~55 other parameters, and transfers readings to a PC over RS232, normally
via the vendor's own "TIMESurf" software, which can also export a session
to Excel.

What's NOT verified, because Beijing TIME High Technology doesn't appear
to publish it and no public source for it was found: the exact byte-level
serial protocol TIMESurf itself speaks to the device (baud rate, framing,
command/response structure). Guessing at that and shipping it as if it
were confirmed would be worse than useless - it would look like it works
right up until it silently doesn't. So this module gives you two paths
instead:

  1. RECOMMENDED: `watch` mode. Point it at the folder TIMESurf exports
     to. It watches for new .xlsx/.csv files and looks for a column whose
     header is recognizably "Ra" (see find_ra_column()). This only
     depends on TIMESurf's export format, not on reverse-engineering an
     undocumented serial protocol - the vendor's own software does the
     actual device communication.

  2. BEST-EFFORT: `serial` mode, using the very common RS232 defaults for
     this class of instrument (9600 baud, 8N1) and a configurable regex
     to pull a number out of whatever text the device sends. This may
     just work - a lot of instruments like this really do just print an
     ASCII line per reading - but verify against your actual device
     before trusting it. `raw-capture` mode is provided specifically to
     help with that: it prints every raw line the port receives so you
     can see the real format and adjust --pattern accordingly, without
     needing to already know the protocol.

Both paths converge on submit_measurement(), which POSTs to this
project's own API (see app/main.py's POST /measurements).
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_BAUD_RATE = 9600
# Common patterns for this class of instrument's ASCII output: "Ra=1.234",
# "Ra : 1.234 um", "Ra  1.234". Verify against your actual device (see
# `raw-capture` mode) and override with --pattern if this doesn't match.
DEFAULT_SERIAL_PATTERN = r"Ra\s*[:=]?\s*([\d.]+)"


def submit_measurement(api_url, ra_measured, job_id=None, device="TIME3233", raw_payload=None, api_key=None, timeout=5):
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = requests.post(
        f"{api_url.rstrip('/')}/measurements",
        json={"Ra_measured": ra_measured, "job_id": job_id, "device": device, "raw_payload": raw_payload},
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def find_ra_column(columns):
    """
    Find whichever column header looks like it holds the Ra value, out of
    an arbitrary/unknown export layout. Matches "Ra" as a standalone token
    (case-insensitive) so it catches "Ra", "Ra (um)", "ra_value",
    "Roughness Ra" etc. without false-matching unrelated headers that
    merely contain the substring "ra" (e.g. "Range", "Parameter",
    "Operator"). Underscores/hyphens are treated as word separators (not
    part of the word, unlike regex's default \\w) so "ra_value" counts as
    a boundary the same way a space would.
    """
    pattern = re.compile(r"\bra\b", re.IGNORECASE)
    for col in columns:
        normalized = re.sub(r"[_-]", " ", str(col))
        if pattern.search(normalized):
            return col
    return None


def parse_timesurf_export(file_path, ra_column=None):
    """
    Read a TIMESurf-exported .xlsx or .csv file and return a list of Ra
    values found in it. If ra_column isn't given, auto-detects via
    find_ra_column() and raises a clear, actionable error (listing the
    columns that were actually found) if nothing matches - this project
    doesn't have a real sample export to verify the exact layout against,
    so failing loudly with enough detail to fix it by hand is the
    responsible default, rather than silently guessing a column.
    """
    import pandas as pd

    file_path = Path(file_path)
    if file_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    column = ra_column or find_ra_column(df.columns)
    if column is None:
        raise ValueError(
            f"Couldn't find an Ra column in {file_path.name}. "
            f"Columns found: {list(df.columns)}. Pass --ra-column to specify it explicitly."
        )

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return values.tolist()


def _scan_folder_once(folder, seen, api_url, job_id=None, ra_column=None, api_key=None, prompt_for_job_id=True):
    """
    Process any not-yet-seen .xlsx/.csv files in `folder` once, submitting
    any Ra readings found. Mutates and relies on `seen` (a set of
    filenames already processed) so repeated calls only handle new files.
    Split out from watch_folder() so the actual scan-and-submit logic is
    testable without needing to deal with an infinite loop.
    """
    folder = Path(folder)
    results = []

    for file_path in sorted(folder.glob("*")):
        if file_path.suffix.lower() not in (".xlsx", ".xls", ".csv") or file_path.name in seen:
            continue
        seen.add(file_path.name)

        this_job_id = job_id
        if this_job_id is None and prompt_for_job_id:
            this_job_id = input(f"New export {file_path.name} - job_id to tag it with (Enter to skip): ").strip() or None

        try:
            values = parse_timesurf_export(file_path, ra_column=ra_column)
        except ValueError as e:
            print(f"  {e}")
            continue

        print(f"  {file_path.name}: found {len(values)} Ra reading(s)")
        for value in values:
            result = submit_measurement(api_url, value, job_id=this_job_id, raw_payload=f"from {file_path.name}", api_key=api_key)
            print(f"    Ra={value} -> {result}")
            results.append((file_path.name, value, result))

    return results


def watch_folder(folder, api_url, job_id=None, ra_column=None, api_key=None, poll_seconds=5):
    """Poll `folder` for new .xlsx/.csv files and submit any Ra readings
    found in them, forever. Each new file is processed once (tracked by
    filename); re-running the script starts that tracking over, so files
    already processed in a previous run will be re-submitted - move or
    delete processed exports if that's not what you want."""

    folder = Path(folder)
    print(f"Watching {folder} for TIMESurf exports (.xlsx/.csv) every {poll_seconds}s. Ctrl+C to stop.")
    seen = set()

    while True:
        _scan_folder_once(folder, seen, api_url, job_id=job_id, ra_column=ra_column, api_key=api_key)
        time.sleep(poll_seconds)


def raw_capture(port, baud_rate):
    """Print every raw line the serial port receives, unparsed. Use this
    first on a device/protocol you haven't verified, to see the actual
    output format before trying to extract values from it."""
    import serial

    print(f"Raw capture on {port} @ {baud_rate} baud. Ctrl+C to stop.")
    with serial.Serial(port, baud_rate, timeout=2) as ser:
        while True:
            line = ser.readline()
            if line:
                print(f"  raw: {line!r}")


def read_serial(port, baud_rate, api_url, pattern, job_id=None, api_key=None):
    """Read lines from the serial port, extract an Ra value from each
    using `pattern`, and submit it. See this module's docstring for why
    this is best-effort rather than a verified protocol implementation."""
    import serial

    regex = re.compile(pattern)
    print(f"Reading {port} @ {baud_rate} baud, pattern={pattern!r}. Ctrl+C to stop.")

    with serial.Serial(port, baud_rate, timeout=2) as ser:
        while True:
            line = ser.readline().decode(errors="replace").strip()
            if not line:
                continue

            match = regex.search(line)
            if not match:
                print(f"  (no match) raw: {line!r}")
                continue

            value = float(match.group(1))
            result = submit_measurement(api_url, value, job_id=job_id, raw_payload=line, api_key=api_key)
            print(f"  Ra={value} (from {line!r}) -> {result}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["watch", "serial", "raw-capture"], help="Ingestion mode - see module docstring")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"Base URL of the prediction API (default: {DEFAULT_API_URL})")
    parser.add_argument("--api-key", default=None, help="X-API-Key, if the API has CNC_API_KEY set")
    parser.add_argument("--job-id", default=None, help="Tag all measurements with this job_id (omit to be prompted per-reading in `watch` mode, or leave untagged in `serial` mode)")

    parser.add_argument("--folder", help="[watch mode] Folder TIMESurf exports to")
    parser.add_argument("--ra-column", default=None, help="[watch mode] Explicit column name for Ra, if auto-detection doesn't find it")
    parser.add_argument("--poll-seconds", type=int, default=5, help="[watch mode] How often to check the folder for new files")

    parser.add_argument("--port", help="[serial/raw-capture mode] Serial port, e.g. /dev/ttyUSB0 or COM3")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD_RATE, help=f"[serial/raw-capture mode] Baud rate (default: {DEFAULT_BAUD_RATE} - common for this instrument class, verify against your device)")
    parser.add_argument("--pattern", default=DEFAULT_SERIAL_PATTERN, help=f"[serial mode] Regex to extract Ra from each line, first capture group (default: {DEFAULT_SERIAL_PATTERN!r})")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "watch":
        if not args.folder:
            print("watch mode needs --folder"); return 1
        watch_folder(args.folder, args.api_url, job_id=args.job_id, ra_column=args.ra_column, api_key=args.api_key, poll_seconds=args.poll_seconds)

    elif args.mode == "raw-capture":
        if not args.port:
            print("raw-capture mode needs --port"); return 1
        raw_capture(args.port, args.baud)

    elif args.mode == "serial":
        if not args.port:
            print("serial mode needs --port"); return 1
        read_serial(args.port, args.baud, args.api_url, args.pattern, job_id=args.job_id, api_key=args.api_key)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
