"""Small client for the guarded xArm7 debug server."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


def _value(mapping: dict, key: str) -> Any:
    return mapping.get(key, mapping.get(key.encode()))


def _print_response(response: dict) -> None:
    if not _value(response, "ok"):
        raise SystemExit(f"server rejected command: {_value(response, 'error')}")
    joints = np.asarray(_value(response, "joint_pos"), dtype=float)
    print("joint_pos_rad:", np.array2string(joints, precision=4))
    poses = _value(response, "poses") or {}
    for raw_name, raw_pose in poses.items():
        name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
        position = np.asarray(_value(raw_pose, "position_m"), dtype=float)
        print(f"{name}_xyz_m:", np.array2string(position, precision=4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=50.0)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--target-deg", type=float, nargs=7)
    action.add_argument("--gripper", choices=("open", "close"))
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="number of random joint states when --target-deg is omitted",
    )
    parser.add_argument(
        "--delta-deg",
        type=float,
        default=1.0,
        help="maximum change per joint at each rate step",
    )
    parser.add_argument(
        "--total-delta",
        type=float,
        default=1.0,
        help="maximum random target displacement per joint; capped at 30 degrees",
    )
    parser.add_argument("--seed", type=int, help="random seed for repeatable motion")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be at least 1")
    if args.delta_deg <= 0.0:
        parser.error("--delta-deg must be greater than 0")
    if not 0.0 < args.total_delta <= 30.0:
        parser.error("--total-delta must be greater than 0 and at most 30")

    client = SyncMsgpackNumpyClient(args.host, args.port)
    try:
        status = client.send_request({"op": "status"})
        _print_response(status)
        if args.gripper is not None:
            response = client.send_request(
                {"op": "gripper", "position": 1.0 if args.gripper == "open" else 0.0}
            )
            _print_response(response)
            return

        current = np.asarray(_value(status, "joint_pos"), dtype=float)[:7]
        if args.target_deg is not None:
            response = client.send_request(
                {
                    "op": "move",
                    "target_joint_pos": np.deg2rad(args.target_deg),
                    "duration_s": args.duration,
                    "rate_hz": args.rate,
                    "max_step_deg": args.delta_deg,
                    "max_total_delta_deg": args.total_delta,
                }
            )
            _print_response(response)
            return

        rng = np.random.default_rng(args.seed)
        max_total_delta_rad = np.deg2rad(args.total_delta)
        for index in range(args.n):
            delta = rng.uniform(-max_total_delta_rad, max_total_delta_rad, size=7)
            target = current + delta
            print(
                f"random jog {index + 1}/{args.n}, delta_deg:",
                np.array2string(np.rad2deg(delta), precision=4),
            )
            response = client.send_request(
                {
                    "op": "move",
                    "target_joint_pos": target,
                    "duration_s": args.duration,
                    "rate_hz": args.rate,
                    "max_step_deg": args.delta_deg,
                    "max_total_delta_deg": args.total_delta,
                }
            )
            _print_response(response)
            current = np.asarray(_value(response, "joint_pos"), dtype=float)[:7]
    finally:
        client.close()


if __name__ == "__main__":
    main()
