#!/usr/bin/env python3
"""Analyze distribution of recording lengths in an episode directory."""

import argparse
import numpy as np
from pathlib import Path


TIMEOUT_SECONDS = 240.0
TIMEOUT_TOLERANCE = 2.0  # episodes within this many seconds of timeout are considered timed out


def get_episode_duration(episode_dir: Path) -> float | None:
    ts_file = episode_dir / "camera_top-rgb-timestamp.npy"
    if not ts_file.exists():
        return None
    ts = np.load(ts_file)
    if len(ts) < 2:
        return None
    return float(ts[-1] - ts[0])


def main():
    parser = argparse.ArgumentParser(description="Analyze recording length distribution")
    parser.add_argument("recordings_dir", type=str, help="Path to directory containing episode_* folders")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS, help="Timeout threshold in seconds (default: 240)")
    parser.add_argument("--tolerance", type=float, default=TIMEOUT_TOLERANCE, help="Tolerance for timeout detection (default: 2)")
    args = parser.parse_args()

    recordings_dir = Path(args.recordings_dir)
    episode_dirs = sorted(recordings_dir.glob("episode_*"))

    if not episode_dirs:
        print(f"No episode_* directories found in {recordings_dir}")
        return

    durations = []
    failed = []
    for ep in episode_dirs:
        dur = get_episode_duration(ep)
        if dur is not None:
            durations.append((ep.name, dur))
        else:
            failed.append(ep.name)

    if not durations:
        print("No valid episodes found.")
        return

    all_durs = np.array([d for _, d in durations])
    timed_out_mask = all_durs >= (args.timeout - args.tolerance)
    normal_mask = ~timed_out_mask

    timed_out_durs = all_durs[timed_out_mask]
    normal_durs = all_durs[normal_mask]

    print(f"Directory: {recordings_dir}")
    print(f"{'='*60}")
    print(f"Total episodes:      {len(durations)}")
    if failed:
        print(f"Skipped (no data):   {len(failed)}")
    print(f"Timed out (>={args.timeout - args.tolerance:.0f}s): {len(timed_out_durs)}")
    print(f"Normal:              {len(normal_durs)}")
    print()

    if len(normal_durs) > 0:
        print("--- Normal episodes (did NOT hit timeout) ---")
        print(f"  Count:   {len(normal_durs)}")
        print(f"  Mean:    {np.mean(normal_durs):.1f}s ({np.mean(normal_durs)/60:.1f} min)")
        print(f"  Std:     {np.std(normal_durs):.1f}s")
        print(f"  Median:  {np.median(normal_durs):.1f}s ({np.median(normal_durs)/60:.1f} min)")
        print(f"  Min:     {np.min(normal_durs):.1f}s")
        print(f"  Max:     {np.max(normal_durs):.1f}s")
        percentiles = [25, 75, 90]
        for p in percentiles:
            val = np.percentile(normal_durs, p)
            print(f"  P{p}:     {val:.1f}s")
        print()

    if len(timed_out_durs) > 0:
        print("--- Timed-out episodes ---")
        for name, dur in durations:
            if dur >= (args.timeout - args.tolerance):
                print(f"  {name}: {dur:.1f}s")
        print()

    print("--- All episodes (sorted by duration) ---")
    for name, dur in sorted(durations, key=lambda x: x[1]):
        tag = " [TIMEOUT]" if dur >= (args.timeout - args.tolerance) else ""
        print(f"  {dur:7.1f}s  {name}{tag}")


if __name__ == "__main__":
    main()
