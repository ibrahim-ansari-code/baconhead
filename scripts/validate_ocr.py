#!/usr/bin/env python3
"""
scripts/validate_ocr.py — Phase 1 OCR validation script.

Runs the Capturer live, displays raw OCR output and parsed stage number
in real time on the terminal, and logs every read to logs/ocr_validation.log.

Usage:
    python scripts/validate_ocr.py            # run until Ctrl-C
    python scripts/validate_ocr.py --duration 60   # run for 60 seconds

After the session, the script prints an accuracy summary. To compute
accuracy you must review the log file and mark correct/incorrect reads
against ground truth — this script tracks distinct stage numbers seen
and total reads to help with that process.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from capture.screen import Capturer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR = _PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "ocr_validation.log"

# ---------------------------------------------------------------------------
# Logging setup — dual output: file (CSV-style) + stderr (debug)
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    """Configure root logger for capture module debug output to stderr."""
    root = logging.getLogger("capture")
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    return root


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 OCR validation")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Run for N seconds then stop (0 = run until Ctrl-C)",
    )
    args = parser.parse_args()

    _setup_logging()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cap = Capturer()

    total_reads = 0
    empty_reads = 0
    stages_seen: set[int] = set()
    start = time.monotonic()

    print(f"Logging to {LOG_FILE}")
    print("Press Ctrl-C to stop.\n")
    print(f"{'#':>5}  {'TIME':<12} {'RAW OCR':<30} {'PARSED':>8}  {'DEATH':<6}")
    print("-" * 72)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header if file is new / empty
        if f.tell() == 0:
            writer.writerow(["timestamp", "frame", "raw_ocr", "parsed_stage", "death_event"])

        try:
            while True:
                cap.tick()
                total_reads += 1

                raw = cap.last_raw_ocr
                parsed = cap.current_stage
                death = cap.death_event

                if raw == "":
                    empty_reads += 1

                stages_seen.add(parsed)

                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                writer.writerow([ts, total_reads, raw, parsed, death])
                f.flush()

                # Save debug frame when death is detected
                if death and cap.last_frame is not None:
                    debug_dir = LOG_DIR / "debug_frames"
                    debug_dir.mkdir(exist_ok=True)
                    frame_path = debug_dir / f"death_{total_reads:04d}.png"
                    from PIL import Image
                    Image.fromarray(cap.last_frame[:, :, ::-1]).save(frame_path)
                    print(f"\n  [saved debug frame: {frame_path}]")

                # Live terminal display
                death_marker = "DEATH" if death else ""
                print(
                    f"\r{total_reads:>5}  {ts:<12} {raw:<30} {parsed:>8}  {death_marker:<6}",
                    end="",
                    flush=True,
                )

                if args.duration > 0 and (time.monotonic() - start) >= args.duration:
                    break

        except KeyboardInterrupt:
            pass

    # --- Summary ---
    elapsed = time.monotonic() - start
    print("\n")
    print("=" * 72)
    print("OCR VALIDATION SUMMARY")
    print("=" * 72)
    print(f"  Duration        : {elapsed:.1f}s")
    print(f"  Total reads     : {total_reads}")
    print(f"  Empty reads     : {empty_reads} ({_pct(empty_reads, total_reads)})")
    print(f"  Non-empty reads : {total_reads - empty_reads} ({_pct(total_reads - empty_reads, total_reads)})")
    print(f"  Distinct stages : {len(stages_seen)}  {sorted(stages_seen)}")
    print(f"  Log file        : {LOG_FILE}")
    print()
    print(
        "  To compute accuracy: review the log against ground truth and count\n"
        "  correct vs incorrect reads. Phase 1 gate requires >95% over 20+ stages."
    )
    print("=" * 72)


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * part / total:.1f}%"


if __name__ == "__main__":
    main()
