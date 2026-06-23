"""Port-9000 debug server for guarded xArm7 joint jogging."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyroki as pk
from pyroki.collision import RobotCollision
from yourdfpy import URDF

from robots_realtime.robots.xarm7 import XArm7, normalized_gripper_to_urdf
from robots_realtime.utils.server_client_utils import MsgpackNumpyServer


class XArm7FkGuard:
    """Checks joint limits, self-collision, and floor clearance with PyRoKi."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        minimum_z_m: float,
        self_collision_margin_m: float,
        protected_links: tuple[str, ...] = ("link_eef", "link_tcp"),
    ) -> None:
        if minimum_z_m < 0.0:
            raise ValueError("minimum_z_m must be non-negative")
        if self_collision_margin_m < 0.0:
            raise ValueError("self_collision_margin_m must be non-negative")
        urdf = URDF.load(str(urdf_path), load_collision_meshes=True)
        self.robot = pk.Robot.from_urdf(urdf)
        self.minimum_z_m = float(minimum_z_m)
        self.self_collision_margin_m = float(self_collision_margin_m)
        self.robot_collision = RobotCollision.from_urdf(
            urdf,
            user_ignore_pairs=self._build_self_collision_ignore_pairs(urdf),
        )
        self._floor_mesh_vertices: dict[str, np.ndarray] = {}
        for link_name in self.robot.links.names:
            if link_name in {"world", "link_base"}:
                continue
            mesh = RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
            if len(mesh.vertices):
                # A linear projection reaches its minimum on the convex hull.
                self._floor_mesh_vertices[link_name] = np.asarray(
                    mesh.convex_hull.vertices, dtype=np.float64
                )
        self._link_indices = {
            name: self.robot.links.names.index(name) for name in protected_links
        }
        if self.robot.joints.num_actuated_joints < XArm7.ARM_DOFS:
            raise ValueError("xArm URDF has fewer than seven actuated joints")
        self.lower = np.asarray(self.robot.joints.lower_limits[: XArm7.ARM_DOFS])
        self.upper = np.asarray(self.robot.joints.upper_limits[: XArm7.ARM_DOFS])
        self._extra_dofs = self.robot.joints.num_actuated_joints - XArm7.ARM_DOFS

    @staticmethod
    def _build_self_collision_ignore_pairs(
        urdf: URDF,
    ) -> tuple[tuple[str, str], ...]:
        """Ignore structural overlaps while retaining nonlocal collision pairs."""
        adjacency: dict[str, set[str]] = {name: set() for name in urdf.link_map}
        for joint in urdf.joint_map.values():
            adjacency[joint.parent].add(joint.child)
            adjacency[joint.child].add(joint.parent)

        links = tuple(urdf.link_map)
        ignored: set[tuple[str, str]] = set()
        for start in links:
            distances = {start: 0}
            queue = [start]
            for current in queue:
                if distances[current] == 3:
                    continue
                for neighbor in adjacency[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            for end, distance in distances.items():
                if 0 < distance <= 3:
                    ignored.add(tuple(sorted((start, end))))

        gripper_links = {
            name
            for name in links
            if name.startswith(("left_", "right_"))
            or name in {"xarm_gripper_base_link", "link_tcp"}
        }
        sorted_gripper_links = sorted(gripper_links)
        for index, first in enumerate(sorted_gripper_links):
            for second in sorted_gripper_links[index + 1 :]:
                ignored.add((first, second))

        # The one-capsule-per-link approximation overlaps at this mounted pair
        # in the known-safe nominal xArm configuration.
        ignored.add(("link5", "xarm_gripper_base_link"))
        return tuple(sorted(ignored))

    def _configuration(
        self, arm_joints: np.ndarray, gripper_normalized: float = 1.0
    ) -> np.ndarray:
        joints = np.asarray(arm_joints, dtype=np.float64)
        if joints.shape != (XArm7.ARM_DOFS,):
            raise ValueError(f"expected seven arm joints, got {joints.shape}")
        if not 0.0 <= gripper_normalized <= 1.0:
            raise ValueError("normalized gripper position must be in [0, 1]")
        extra = np.zeros(self._extra_dofs)
        if self._extra_dofs:
            extra[0] = normalized_gripper_to_urdf(
                gripper_normalized,
                float(self.robot.joints.upper_limits[XArm7.ARM_DOFS]),
            )
        return np.concatenate([joints, extra])

    def poses(
        self, arm_joints: np.ndarray, gripper_normalized: float = 1.0
    ) -> dict[str, np.ndarray]:
        cfg = self._configuration(arm_joints, gripper_normalized)
        fk = np.asarray(self.robot.forward_kinematics(cfg))
        return {name: fk[index] for name, index in self._link_indices.items()}


    
    def _rotation_matrix(self, wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = wxyz
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def floor_clearances(self, cfg: np.ndarray) -> dict[str, float]:
        """Return exact collision-mesh clearance above the configured floor."""
        fk = np.asarray(self.robot.forward_kinematics(cfg))
        clearances: dict[str, float] = {}
        for link_name, vertices in self._floor_mesh_vertices.items():
            pose = fk[self.robot.links.names.index(link_name)]
            rotation = self._rotation_matrix(pose[:4])
            world_z = vertices @ rotation[2] + pose[6]
            clearances[link_name] = float(np.min(world_z) - self.minimum_z_m)
        return clearances

    def validate(
        self, arm_joints: np.ndarray, gripper_normalized: float = 1.0
    ) -> dict[str, np.ndarray]:
        joints = np.asarray(arm_joints, dtype=np.float64)
        if np.any(joints < self.lower) or np.any(joints > self.upper):
            raise ValueError("target violates URDF joint limits")
        cfg = self._configuration(joints, gripper_normalized)
        poses = self.poses(joints, gripper_normalized)
        for name, pose in poses.items():
            z = float(pose[6])
            if z < self.minimum_z_m:
                raise ValueError(
                    f"{name} z={z:.4f} m is below minimum {self.minimum_z_m:.4f} m"
                )

        self_distances = np.asarray(
            self.robot_collision.compute_self_collision_distance(self.robot, cfg)
        )
        closest_pair = int(np.argmin(self_distances))
        closest_distance = float(self_distances[closest_pair])
        if closest_distance < self.self_collision_margin_m:
            first = self.robot_collision.link_names[
                self.robot_collision.active_idx_i[closest_pair]
            ]
            second = self.robot_collision.link_names[
                self.robot_collision.active_idx_j[closest_pair]
            ]
            raise ValueError(
                f"self-collision clearance between {first} and {second} is "
                f"{closest_distance:.4f} m; minimum is "
                f"{self.self_collision_margin_m:.4f} m"
            )

        floor_clearances = self.floor_clearances(cfg)
        link_name, closest_floor_distance = min(
            floor_clearances.items(), key=lambda item: item[1]
        )
        if closest_floor_distance < 0.0:
            raise ValueError(
                f"{link_name} collision geometry extends "
                f"{-closest_floor_distance:.4f} m below the "
                f"z={self.minimum_z_m:.4f} m floor"
            )
        return poses


class XArm7JogController:
    """State, guarded interpolation, and gripper operations."""

    def __init__(self, robot: XArm7, guard: XArm7FkGuard) -> None:
        self.robot = robot
        self.guard = guard

    def status(self) -> dict[str, Any]:
        state = self.robot.get_joint_pos()
        poses = self.guard.poses(
            state[: XArm7.ARM_DOFS],
            float(state[-1]) if self.robot.enable_gripper else 1.0,
        )
        return {
            "ok": True,
            "joint_pos": state,
            "poses": {
                name: {
                    "wxyz": pose[:4],
                    "position_m": pose[4:],
                }
                for name, pose in poses.items()
            },
            "minimum_z_m": self.guard.minimum_z_m,
        }

    def move(
        self,
        target_arm_joints: np.ndarray,
        *,
        duration_s: float = 2.0,
        rate_hz: float = 50.0,
        max_step_deg: float = 1.0,
        max_total_delta_deg: float = 30.0,
    ) -> dict[str, Any]:
        target = np.asarray(target_arm_joints, dtype=np.float64)
        if target.shape != (XArm7.ARM_DOFS,):
            raise ValueError(f"target_joint_pos must contain 7 values, got {target.shape}")
        if duration_s <= 0.0 or rate_hz <= 0.0:
            raise ValueError("duration_s and rate_hz must be positive")
        if max_step_deg <= 0.0:
            raise ValueError("max_step_deg must be positive")
        if not 0.0 < max_total_delta_deg <= 30.0:
            raise ValueError("max_total_delta_deg must be in (0, 30]")

        current = self.robot.get_joint_pos()
        arm_start = current[: XArm7.ARM_DOFS]
        gripper = current[-1] if self.robot.enable_gripper else None
        joint_delta = np.abs(target - arm_start)
        max_joint_delta = float(np.max(joint_delta))
        max_total_delta = np.deg2rad(max_total_delta_deg)
        if max_joint_delta > max_total_delta + 1e-12:
            raise ValueError(
                f"target changes a joint by {np.rad2deg(max_joint_delta):.3f} deg; "
                f"maximum is {max_total_delta_deg:.3f} deg"
            )

        duration_steps = int(np.ceil(duration_s * rate_hz))
        step_limit = np.deg2rad(max_step_deg)
        delta_steps = int(np.ceil(max_joint_delta / step_limit))
        steps = max(2, duration_steps + 1, delta_steps + 1)
        path = np.linspace(arm_start, target, steps)

        # Validate the complete path before issuing the first command.
        for waypoint in path:
            self.guard.validate(
                waypoint, float(gripper) if gripper is not None else 1.0
            )

        period = 1.0 / rate_hz
        deadline = time.monotonic()
        for waypoint in path[1:]:
            command = (
                np.concatenate([waypoint, [gripper]])
                if gripper is not None
                else waypoint
            )
            self.robot.command_joint_pos(command)
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))

        return self.status()

    def set_gripper(self, normalized: float) -> dict[str, Any]:
        if not self.robot.enable_gripper:
            raise ValueError("gripper is disabled")
        value = float(normalized)
        if not 0.0 <= value <= 1.0:
            raise ValueError("gripper must be in [0, 1]")
        state = self.robot.get_joint_pos()
        state[-1] = value
        self.robot.command_joint_pos(state)
        return self.status()

    def process(self, request: dict) -> dict[str, Any]:
        req = {
            (key.decode() if isinstance(key, bytes) else key): value
            for key, value in request.items()
        }
        operation = req.get("op")
        if isinstance(operation, bytes):
            operation = operation.decode()
        try:
            if operation == "status":
                return self.status()
            if operation == "move":
                return self.move(
                    req["target_joint_pos"],
                    duration_s=float(req.get("duration_s", 2.0)),
                    rate_hz=float(req.get("rate_hz", 50.0)),
                    max_step_deg=float(req.get("max_step_deg", 1.0)),
                    max_total_delta_deg=float(req.get("max_total_delta_deg", 30.0)),
                )
            if operation == "gripper":
                return self.set_gripper(float(req["position"]))
            return {"ok": False, "error": f"unknown operation: {operation!r}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


class XArm7DebugServer(MsgpackNumpyServer):
    def __init__(self, controller: XArm7JogController, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.controller = controller

    def process(self, req: dict) -> dict:
        return self.controller.process(req)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-host", default="192.168.1.231")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--urdf", default="xarm7_standalone.urdf")
    parser.add_argument("--minimum-z-mm", type=float, default=0.0)
    parser.add_argument("--self-collision-margin-mm", type=float, default=0.002)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--clear-errors-on-startup", action="store_true")
    args = parser.parse_args()

    robot = XArm7(
        host_name=args.robot_host,
        enable_gripper=not args.no_gripper,
        clear_errors_on_startup=args.clear_errors_on_startup,
    )
    guard = XArm7FkGuard(
        args.urdf,
        minimum_z_m=args.minimum_z_mm / 1000.0,
        self_collision_margin_m=args.self_collision_margin_mm / 1000.0,
    )
    server = XArm7DebugServer(
        XArm7JogController(robot, guard),
        host=args.bind_host,
        port=args.port,
    )
    try:
        asyncio.run(server.start())
    finally:
        robot.stop()


if __name__ == "__main__":
    main()
