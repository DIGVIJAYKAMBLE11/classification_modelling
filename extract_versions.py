#!/usr/bin/env python3
"""
Extract timestamped versions of a file from git history.

Usage:
    python extract_versions.py <filename> <start_date> <end_date>

Examples:
    python extract_versions.py train_pipeline.py 2026-02-15 2026-02-16
    python extract_versions.py create_presentation.py 2026-02-12 2026-02-14
"""

import argparse
import os
import subprocess
import sys


def run_git(args):
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Extract timestamped versions of a file from git history.",
    )
    parser.add_argument("filename", help="File path to extract (e.g. train_pipeline.py)")
    parser.add_argument("start_date", help="Start date inclusive (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "-o", "--output-dir", default=".",
        help="Directory to write extracted files (default: current dir)",
    )
    args = parser.parse_args()

    # --after is exclusive so go one day before start;
    # --before is inclusive so go one day after end to cover the full end date
    after = f"{args.start_date}T00:00:00"
    before = f"{args.end_date}T23:59:59"

    log_output = run_git([
        "log", "--all",
        f"--after={after}", f"--before={before}",
        "--format=%H %ai %s",
        "--", args.filename,
    ])

    lines = [l.strip() for l in log_output.strip().splitlines() if l.strip()]
    if not lines:
        print(f"No commits found for '{args.filename}' between {args.start_date} and {args.end_date}.")
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(args.filename))

    print(f"Found {len(lines)} version(s) of '{args.filename}' between {args.start_date} and {args.end_date}:\n")
    print(f"{'#':<4} {'Timestamp':<22} {'File':<50} {'Commit Message'}")
    print("-" * 110)

    for i, line in enumerate(reversed(lines), 1):
        parts = line.split(None, 4)
        commit_hash = parts[0]
        date_str = parts[1]          # YYYY-MM-DD
        time_str = parts[2]          # HH:MM:SS
        message = parts[4] if len(parts) > 4 else ""

        timestamp = f"{date_str}_{time_str.replace(':', '')}"
        out_name = f"{base}_{timestamp}{ext}"
        out_path = os.path.join(args.output_dir, out_name)

        content = run_git(["show", f"{commit_hash}:{args.filename}"])
        with open(out_path, "w") as f:
            f.write(content)

        print(f"{i:<4} {date_str} {time_str:<10} {out_name:<50} {message}")

    print(f"\nDone — {len(lines)} file(s) written to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
