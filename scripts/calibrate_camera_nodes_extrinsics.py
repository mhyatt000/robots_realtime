#!/usr/bin/env python3
"""Calibrate non-RealSense CameraNode extrinsics by driving the xArm7.

Unlike a single-pose capture, this drives the arm through ``--n`` diverse but
safe joint configurations and grabs a synchronized frame from every camera at
each pose. Every frame is labeled with the *measured* joint angles read back
from the robot (not a nominal constant), which is what DREAM PnP needs.

Motion goes through the guarded xArm7 debug server (port 9000), which validates
joint limits, self-collision, and floor clearance over the whole interpolated
path before issuing any command and hard-caps a single move at 30 deg/joint. To
honor ``--amp-deg`` (default 20 deg around home) targets up to 40 deg apart are
reached by sub-stepping into <=28 deg guarded moves.

Pipeline per camera (same DREAM/SAM/roboreg flow as the RealSense script):

    DREAM PnP -> SAM masks -> roboreg DR refine -> save extrinsics

Approximate OpenCV intrinsics are used (fx = fy = 515, cx = w/2, cy = h/2).

As a diversity diagnostic we report the pairwise SAM-mask IoU across the valid
frames: *lower* mean pairwise IoU means the arm projected into more distinct
image regions, i.e. a more informative calibration set. (Normally high IoU is
"good"; here we invert it as a spread metric.)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robots_realtime.runtime.environment.camera_node import CameraNode
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient
from scripts.calibrate_realsense_extrinsics import (
    Capture,
    Endpoint,
    image_u8,
    mean_extrinsics,
    run_dream_batch,
    run_roboreg_refine,
    run_sam_batch,
    save_extrinsics,
)

ARM_DOFS = 7
# Stay safely under the guard's 30 deg/joint per-move ceiling when sub-stepping.
SUBSTEP_CAP_DEG = 28.0


def _value(mapping: dict, key: str) -> Any:
    """Read a key from a (possibly bytes-keyed) msgpack response."""
    if key in mapping:
        return mapping[key]
    return mapping.get(key.encode())


def load_camera_nodes(config_path: Path, *, include_realsense: bool) -> dict[str, dict[str, Any]]:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cams: dict[str, dict[str, Any]] = {}
    for node in cfg.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") != "CameraNode":
            continue
        if not include_realsense and node.get("driver") == "RealSenseCamera":
            continue
        cams[str(node["name"])] = node
    return cams


def approximate_k(width: int, height: int, fx: float, fy: float | None) -> np.ndarray:
    fy = fx if fy is None else fy
    return np.array(
        [[float(fx), 0.0, width / 2.0], [0.0, float(fy), height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def pairwise_mask_iou(masks: np.ndarray) -> dict[str, Any]:
    """Mean/median pairwise IoU over a stack of boolean masks (N, H, W).

    Pairs whose union is empty are skipped. Returns NaN summaries when fewer
    than two non-empty masks are available.
    """
    bool_masks = [np.asarray(m, dtype=bool) for m in masks if np.asarray(m, dtype=bool).any()]
    if len(bool_masks) < 2:
        return {"mean": float("nan"), "median": float("nan"), "n_masks": len(bool_masks), "n_pairs": 0}
    ious: list[float] = []
    for a, b in combinations(bool_masks, 2):
        union = np.logical_or(a, b).sum()
        if union == 0:
            continue
        ious.append(float(np.logical_and(a, b).sum()) / float(union))
    if not ious:
        return {"mean": float("nan"), "median": float("nan"), "n_masks": len(bool_masks), "n_pairs": 0}
    return {
        "mean": float(np.mean(ious)),
        "median": float(np.median(ious)),
        "n_masks": len(bool_masks),
        "n_pairs": len(ious),
    }


class GuardedArm:
    """Thin client over the guarded xArm7 debug server (port 9000).

    Optionally launches the server as a subprocess and tears it down with SIGINT
    so the server's ``finally: robot.stop()`` runs.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        robot_host: str,
        urdf: str,
        minimum_z_mm: float,
        self_collision_margin_mm: float,
        launch: bool,
        clear_errors_on_startup: bool,
        connect_timeout_s: float,
    ) -> None:
        self.host = host
        self.port = port
        self._proc: subprocess.Popen | None = None
        if launch:
            self._proc = self._launch_server(
                robot_host=robot_host,
                urdf=urdf,
                minimum_z_mm=minimum_z_mm,
                self_collision_margin_mm=self_collision_margin_mm,
                clear_errors_on_startup=clear_errors_on_startup,
            )
        self.client = self._connect(connect_timeout_s)

    def _launch_server(
        self,
        *,
        robot_host: str,
        urdf: str,
        minimum_z_mm: float,
        self_collision_margin_mm: float,
        clear_errors_on_startup: bool,
    ) -> subprocess.Popen:
        cmd = [
            sys.executable,
            "-m",
            "robots_realtime.robots.xarm7_debug_server",
            "--robot-host", robot_host,
            "--bind-host", "127.0.0.1",
            "--port", str(self.port),
            "--urdf", urdf,
            "--minimum-z-mm", str(minimum_z_mm),
            "--self-collision-margin-mm", str(self_collision_margin_mm),
        ]
        if clear_errors_on_startup:
            cmd.append("--clear-errors-on-startup")
        print(f"launching guarded server: {' '.join(cmd)}")
        return subprocess.Popen(cmd)

    def _connect(self, timeout_s: float) -> SyncMsgpackNumpyClient:
        deadline = time.monotonic() + timeout_s
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"guarded server exited early with code {self._proc.returncode} "
                    "before accepting connections"
                )
            try:
                return SyncMsgpackNumpyClient(self.host, self.port)
            except (ConnectionRefusedError, OSError) as exc:
                last_exc = exc
                time.sleep(0.5)
        raise RuntimeError(f"could not connect to guarded server at {self.host}:{self.port}: {last_exc}")

    def _request(self, payload: dict) -> dict:
        resp = self.client.send_request(payload)
        if not _value(resp, "ok"):
            raise RuntimeError(f"guarded server rejected {payload.get('op')!r}: {_value(resp, 'error')}")
        return resp

    def joint_pos(self) -> np.ndarray:
        resp = self._request({"op": "status"})
        return np.asarray(_value(resp, "joint_pos"), dtype=np.float64)

    def arm_joints(self) -> np.ndarray:
        return self.joint_pos()[:ARM_DOFS]

    def move_to(
        self,
        target_arm: np.ndarray,
        *,
        duration_s: float,
        rate_hz: float,
        max_step_deg: float,
    ) -> np.ndarray:
        """Guarded move to a 7-DoF target, sub-stepping under the 30 deg cap.

        Raises RuntimeError if the guard rejects any waypoint along the way.
        """
        target = np.asarray(target_arm, dtype=np.float64)[:ARM_DOFS]
        cap = np.deg2rad(SUBSTEP_CAP_DEG)
        while True:
            current = self.arm_joints()
            delta = target - current
            max_delta = float(np.max(np.abs(delta)))
            if max_delta < 1e-4:
                return current
            frac = min(1.0, cap / max_delta)
            waypoint = current + delta * frac
            self._request(
                {
                    "op": "move",
                    "target_joint_pos": waypoint,
                    "duration_s": duration_s,
                    "rate_hz": rate_hz,
                    "max_step_deg": max_step_deg,
                    "max_total_delta_deg": 30.0,
                }
            )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.send_signal(2)  # SIGINT so the server runs robot.stop()
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()


def read_fresh_frame(driver: Any, *, flush: int, read_timeout_s: float) -> Any:
    """Drop ``flush`` buffered frames, then return the next CameraData."""
    deadline = time.monotonic() + read_timeout_s
    last_exc: Exception | None = None
    for _ in range(max(0, flush)):
        while True:
            try:
                driver.read()
                break
            except Exception as exc:  # transient camera reads
                last_exc = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"camera flush failed within {read_timeout_s:.1f}s: {last_exc}")
                time.sleep(0.02)
    while True:
        try:
            return driver.read()
        except Exception as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                raise RuntimeError(f"camera read failed within {read_timeout_s:.1f}s: {last_exc}")
            time.sleep(0.02)


def warmup_driver(driver: Any, *, warmup: int, read_timeout_s: float) -> None:
    deadline = time.monotonic() + read_timeout_s
    ok = 0
    last_exc: Exception | None = None
    while ok < warmup and time.monotonic() < deadline:
        try:
            driver.read()
            ok += 1
        except Exception as exc:
            last_exc = exc
            time.sleep(0.05)
    if ok < warmup:
        raise RuntimeError(
            f"camera produced only {ok}/{warmup} warmup frames within {read_timeout_s:.1f}s; "
            f"last error: {last_exc}"
        )


def sample_target(home: np.ndarray, amp_rad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return home + rng.uniform(-amp_rad, amp_rad)


def capture_session(
    arm: GuardedArm,
    drivers: dict[str, Any],
    *,
    n: int,
    amp_deg: float,
    settle_s: float,
    pre_move_s: float,
    flush_frames: int,
    read_timeout_s: float,
    move_duration_s: float,
    move_rate_hz: float,
    move_step_deg: float,
    fx: float,
    fy: float | None,
    rng: np.random.Generator,
    max_resamples: int,
    return_home: bool,
) -> dict[str, list[Capture]]:
    """Drive the arm through n diverse poses, capturing every camera at each."""
    home = arm.arm_joints()
    print(f"home joints (deg): {np.array2string(np.rad2deg(home), precision=2)}")
    amp_rad = np.full(ARM_DOFS, np.deg2rad(amp_deg))
    captures: dict[str, list[Capture]] = {name: [] for name in drivers}

    pose = 0
    while pose < n:
        target = sample_target(home, amp_rad, rng)
        try:
            arm.move_to(
                target,
                duration_s=move_duration_s,
                rate_hz=move_rate_hz,
                max_step_deg=move_step_deg,
            )
        except RuntimeError as exc:
            print(f"pose {pose + 1}/{n}: target rejected by guard ({exc}); resampling")
            max_resamples -= 1
            if max_resamples < 0:
                raise RuntimeError("exceeded resample budget while sampling safe poses") from exc
            continue

        time.sleep(settle_s)
        q_rad = arm.arm_joints().astype(np.float32)
        for name, driver in drivers.items():
            data = read_fresh_frame(driver, flush=flush_frames, read_timeout_s=read_timeout_s)
            image = image_u8(data.images["rgb"])
            h, w = image.shape[:2]
            captures[name].append(
                Capture(
                    image=image,
                    q_rad=q_rad,
                    K=approximate_k(w, h, fx=fx, fy=fy),
                    timestamp=float(data.timestamp),
                )
            )
        print(f"pose {pose + 1}/{n}: q_deg={np.array2string(np.rad2deg(q_rad), precision=1)}")
        pose += 1
        time.sleep(pre_move_s)

    if return_home:
        print("returning to home pose")
        arm.move_to(home, duration_s=move_duration_s, rate_hz=move_rate_hz, max_step_deg=move_step_deg)
    return captures


def calibrate_camera(
    camera_name: str,
    spec: dict[str, Any],
    captures: list[Capture],
    args: argparse.Namespace,
) -> dict[str, Any]:
    dream_w2cs, dream_valid, dream_diag = run_dream_batch(
        captures,
        Endpoint(args.dream_host, args.dream_port),
        units=args.dream_units,
        reproj_px=args.max_reproj_px,
    )
    valid_count = int(dream_valid.sum())
    if valid_count == 0:
        print(f"{camera_name}: no valid DREAM calibrations")
        return {"valid": False, "reason": "dream_failed", "diagnostics": dream_diag}

    best_method = "dream_mean"
    best_score = -float(np.nanmedian(np.asarray(dream_diag["pnp_reproj_px"])[dream_valid]))
    best_w2c = mean_extrinsics(dream_w2cs[dream_valid])
    refine_diag: dict[str, Any] = {}
    diversity: dict[str, Any] = {}

    if not args.no_refine:
        masks, mask_valid = run_sam_batch(
            captures,
            dream_valid,
            Endpoint(args.sam_host, args.sam_port),
            prompt=args.sam_prompt,
            confidence=args.sam_confidence,
            close_kernel_size=args.sam_close_kernel,
            min_component_area=args.sam_min_component_area,
        )
        diversity = pairwise_mask_iou(masks[mask_valid] > 0)
        print(
            f"{camera_name}: mask diversity mean_iou={diversity['mean']:.3f} "
            f"median_iou={diversity['median']:.3f} over {diversity['n_masks']} masks "
            f"(lower = more diverse)"
        )
        refined, iou, refine_diag = run_roboreg_refine(
            captures,
            dream_w2cs,
            dream_valid,
            masks,
            mask_valid,
            Endpoint(args.roboreg_host, args.roboreg_port),
            min_iou=args.min_iou,
        )
        if refined is not None:
            best_method = "dream_sam_roboreg"
            best_score = iou
            best_w2c = refined

    out_path = args.out_dir / f"{camera_name}.yaml"
    save_extrinsics(
        out_path,
        camera_name=camera_name,
        serial=str(spec.get("device_path", "")),
        w2c_cv=best_w2c,
        method=best_method,
        score=best_score,
        diagnostics={
            "intrinsics_source": "approximate",
            "fx": args.fx,
            "fy": args.fx if args.fy is None else args.fy,
            "cx": "width/2",
            "cy": "height/2",
            "dream": dream_diag,
            "refine": refine_diag,
            "mask_diversity": diversity,
            "valid_dream_frames": valid_count,
            "total_frames": len(captures),
        },
    )
    print(f"{camera_name}: saved {best_method} extrinsics to {out_path}")
    return {
        "valid": True,
        "path": str(out_path),
        "method": best_method,
        "score": best_score,
        "mask_diversity_mean_iou": diversity.get("mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, default=Path("configs/xarm/xarm_client_no_realsense.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("configs/camera_extrinsics/opencv_calibrated"))
    parser.add_argument("--include-realsense", action="store_true")
    parser.add_argument("--n", type=int, default=24, help="diverse poses to visit")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--read-timeout-s", type=float, default=30.0)
    parser.add_argument("--fx", type=float, default=515.0)
    parser.add_argument("--fy", type=float, default=None)
    # Motion / guarded server.
    parser.add_argument("--motion-host", default="127.0.0.1")
    parser.add_argument("--motion-port", type=int, default=9000)
    parser.add_argument("--robot-host", default="192.168.1.231")
    parser.add_argument("--urdf", default="xarm7_standalone.urdf")
    parser.add_argument("--minimum-z-mm", type=float, default=0.0)
    parser.add_argument("--self-collision-margin-mm", type=float, default=2.0)
    parser.add_argument("--no-launch-server", action="store_true", help="attach to an already-running server")
    parser.add_argument("--clear-errors-on-startup", action="store_true")
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    # Pose sampling / timing.
    parser.add_argument("--amp-deg", type=float, default=20.0, help="+/- per-joint sampling amplitude around home")
    parser.add_argument("--settle-s", type=float, default=0.5, help="wait after move before capturing")
    parser.add_argument("--pre-move-s", type=float, default=0.5, help="wait after capture before next move")
    parser.add_argument("--flush-frames", type=int, default=5, help="stale frames to drop before each capture")
    parser.add_argument("--move-duration-s", type=float, default=2.0)
    parser.add_argument("--move-rate-hz", type=float, default=50.0)
    parser.add_argument("--move-step-deg", type=float, default=1.0)
    parser.add_argument("--max-resamples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-return-home", action="store_true")
    # DREAM / SAM / roboreg.
    parser.add_argument("--dream-host", default="localhost")
    parser.add_argument("--dream-port", type=int, default=8082)
    parser.add_argument("--dream-units", choices=("deg", "rad"), default="deg")
    parser.add_argument("--max-reproj-px", type=float, default=20.0)
    parser.add_argument("--sam-host", default="localhost")
    parser.add_argument("--sam-port", type=int, default=8080)
    parser.add_argument("--sam-prompt", default="robot")
    parser.add_argument("--sam-confidence", type=float, default=0.3)
    parser.add_argument("--sam-close-kernel", type=int, default=3)
    parser.add_argument("--sam-min-component-area", type=int, default=16)
    parser.add_argument("--roboreg-host", default="localhost")
    parser.add_argument("--roboreg-port", type=int, default=8081)
    parser.add_argument("--min-iou", type=float, default=0.75)
    parser.add_argument("--no-refine", action="store_true", help="skip SAM+roboreg and keep DREAM mean")
    args = parser.parse_args()

    camera_specs = load_camera_nodes(args.camera_config, include_realsense=args.include_realsense)
    if not camera_specs:
        raise SystemExit(f"No CameraNode entries found in {args.camera_config}")

    # Open every camera up front so all of them capture at each robot pose.
    nodes: dict[str, CameraNode] = {}
    drivers: dict[str, Any] = {}
    try:
        for camera_name, spec in camera_specs.items():
            print(f"opening {camera_name} driver={spec.get('driver')} device={spec.get('device_path', spec.get('device_id'))}")
            node = CameraNode(**CameraNode.build_kwargs(spec))
            node.setup()
            if node._driver is None:
                raise RuntimeError(f"{camera_name}: CameraNode.setup() did not instantiate a driver")
            print(f"{camera_name}: info={node._driver.get_camera_info()}")
            warmup_driver(node._driver, warmup=args.warmup, read_timeout_s=args.read_timeout_s)
            nodes[camera_name] = node
            drivers[camera_name] = node._driver

        arm = GuardedArm(
            host=args.motion_host,
            port=args.motion_port,
            robot_host=args.robot_host,
            urdf=args.urdf,
            minimum_z_mm=args.minimum_z_mm,
            self_collision_margin_mm=args.self_collision_margin_mm,
            launch=not args.no_launch_server,
            clear_errors_on_startup=args.clear_errors_on_startup,
            connect_timeout_s=args.connect_timeout_s,
        )
        try:
            captures = capture_session(
                arm,
                drivers,
                n=args.n,
                amp_deg=args.amp_deg,
                settle_s=args.settle_s,
                pre_move_s=args.pre_move_s,
                flush_frames=args.flush_frames,
                read_timeout_s=args.read_timeout_s,
                move_duration_s=args.move_duration_s,
                move_rate_hz=args.move_rate_hz,
                move_step_deg=args.move_step_deg,
                fx=args.fx,
                fy=args.fy,
                rng=np.random.default_rng(args.seed),
                max_resamples=args.max_resamples,
                return_home=not args.no_return_home,
            )
        finally:
            arm.close()
    finally:
        for node in nodes.values():
            node.cleanup()

    summary: dict[str, Any] = {}
    for camera_name, spec in camera_specs.items():
        print(f"\n=== {camera_name}: {len(captures[camera_name])} frames ===")
        summary[camera_name] = calibrate_camera(camera_name, spec, captures[camera_name], args)

    print("\nsummary:")
    print(yaml.safe_dump(summary, sort_keys=True))


if __name__ == "__main__":
    main()
