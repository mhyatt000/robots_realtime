"""Interactive xArm7 inverse kinematics backed by PyRoKi and Viser."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pyroki as pk

from robots_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik_vel_cost import (
    solve_ik_vel_cost,
)
from robots_realtime.robots.viser.viser_base import ViserAbstractBase


class XArm7Pyroki(ViserAbstractBase):
    """Single-arm xArm7 IK with a draggable TCP target in Viser."""

    ARM_DOFS = 7
    GRIPPER_TRAVEL_RAD = 0.85
    DEFAULT_TARGET_LINK = "link_tcp"
    DEFAULT_REST_POSE = np.array(
        [-4.48e-4, -7.8567e-1, 2.263e-3, 6.1070e-1, 1.011e-3, 1.1383, 1.5720, 0.0],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        rate: float = 50.0,
        viser_server: object | None = None,
        urdf_path: str | Path | None = None,
        target_link_name: str = DEFAULT_TARGET_LINK,
    ) -> None:
        if urdf_path is None:
            urdf_path = Path(__file__).resolve().parents[3] / "xarm7_standalone.urdf"
        self.urdf_path = str(Path(urdf_path).resolve())
        self.target_link_name = target_link_name
        self.robot: pk.Robot | None = None
        self.rest_pose = self.DEFAULT_REST_POSE.copy()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        super().__init__(
            rate=rate,
            viser_server=viser_server,
            robot_description="xarm7_local",
            urdf_path=self.urdf_path,
            min_distance_from_limits=0.0,
            bimanual=False,
        )
        if self.robot is None:
            raise RuntimeError("failed to initialize xArm7 PyRoKi model")
        if self.robot.joints.num_actuated_joints != self.rest_pose.size:
            raise ValueError(
                "xArm7 URDF must expose 7 arm joints plus one gripper drive joint; "
                f"found {self.robot.joints.num_actuated_joints}"
            )
        self.joints["left"] = self.rest_pose.copy()
        self.home()

        @self.reset_button.on_click
        def _(_event) -> None:
            self.home()

    def _setup_solver_specific(self) -> None:
        self.robot = pk.Robot.from_urdf(self.urdf)

    def _setup_gui(self) -> None:
        super()._setup_gui()
        self.timing_handle_left = self.viser_server.gui.add_number(
            "IK time (ms)", 0.0, disabled=True
        )

    def _initialize_transform_handles(self) -> None:
        self.set_configuration(self.rest_pose, move_target=True)

    def _update_optional_handle_sizes(self) -> None:
        return

    def set_configuration(
        self, configuration: np.ndarray, *, move_target: bool = False
    ) -> None:
        """Set the IK seed and optionally place the target at that configuration's TCP."""
        cfg = np.asarray(configuration, dtype=np.float64)
        if cfg.shape == (self.ARM_DOFS,):
            cfg = np.concatenate([cfg, [0.0]])
        if cfg.shape != (self.ARM_DOFS + 1,):
            raise ValueError(f"expected xArm7 configuration shape (8,), got {cfg.shape}")
        if self.robot is None:
            return

        with self._lock:
            self.joints["left"] = cfg.copy()
            self.urdf_vis_left.update_cfg(cfg)
            if move_target:
                link_index = self.robot.links.names.index(self.target_link_name)
                pose = np.asarray(self.robot.forward_kinematics(cfg))[link_index]
                control = self.transform_handles["left"].control
                if control is not None:
                    control.wxyz = pose[:4]
                    control.position = pose[4:]

    def solve_ik(self) -> None:
        if self.robot is None:
            return
        target = self.get_target_poses().get("left")
        if target is None:
            return

        with self._lock:
            previous = self.joints["left"].copy()
        start = time.perf_counter()
        solution = solve_ik_vel_cost(
            robot=self.robot,
            target_link_name=self.target_link_name,
            target_position=target.translation(),
            target_wxyz=target.rotation().wxyz,
            prev_cfg=previous,
        )
        # link_tcp is upstream of the gripper drive joint. Keep that free DOF fixed.
        solution[-1] = previous[-1]
        with self._lock:
            self.joints["left"] = solution
        self.timing_handle_left.value = (time.perf_counter() - start) * 1e3

    def get_arm_target(self) -> np.ndarray:
        with self._lock:
            return self.joints["left"][: self.ARM_DOFS].copy()

    def update_visualization(self) -> None:
        with self._lock:
            cfg = self.joints["left"].copy()
        self.urdf_vis_left.update_cfg(cfg)

    def home(self) -> None:
        self.set_configuration(self.rest_pose, move_target=True)

    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.perf_counter()
            self.solve_ik()
            self.update_visualization()
            remaining = (1.0 / self.rate) - (time.perf_counter() - start)
            self._stop_event.wait(max(0.0, remaining))

    def close(self) -> None:
        self._stop_event.set()


__all__ = ["XArm7Pyroki"]
