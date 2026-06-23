"""Viser web teleoperation agent for a real UFACTORY xArm7."""

from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import viser
import viser.extras
from dm_env.specs import Array

from robots_realtime.agents.agent import Agent
from robots_realtime.robots.inverse_kinematics.xarm7_pyroki import XArm7Pyroki
from robots_realtime.robots.xarm7 import normalized_gripper_to_urdf
from robots_realtime.robots.xarm7_debug_server import XArm7FkGuard
from robots_realtime.sensors.cameras.camera_utils import (
    obs_get_rgb,
    resize_with_center_crop,
)


class XArm7PyrokiViserAgent(Agent):
    """Drive xArm7 joint targets from a Viser TCP transform control.

    Commands default to disabled and must be enabled explicitly in the sidebar.
    """

    ARM_DOFS = 7

    def __init__(
        self,
        *,
        viser_port: int = 8765,
        ik_rate: float = 50.0,
        urdf_path: str | Path | None = None,
        enable_gripper: bool = True,
        max_joint_step_deg: float | list[float] = 0.5,
        minimum_z_m: float,
        self_collision_margin_m: float,
        collision_guard: bool = True,
    ) -> None:
        max_joint_step = np.asarray(max_joint_step_deg, dtype=np.float64)
        if max_joint_step.ndim == 0:
            max_joint_step = np.full(self.ARM_DOFS, float(max_joint_step))
        if max_joint_step.shape != (self.ARM_DOFS,):
            raise ValueError(
                f"max_joint_step_deg must be a scalar or {self.ARM_DOFS} values, "
                f"got shape {max_joint_step.shape}"
            )
        if not np.all(np.isfinite(max_joint_step)) or np.any(max_joint_step <= 0.0):
            raise ValueError("all max_joint_step_deg values must be finite and positive")
        if urdf_path is None:
            urdf_path = Path(__file__).resolve().parents[3] / "xarm7_standalone.urdf"
        self.urdf_path = str(Path(urdf_path).resolve())
        self.enable_gripper = enable_gripper
        self.max_joint_step_rad = np.deg2rad(max_joint_step)
        self.viser_server = viser.ViserServer(port=viser_port)
        self.ik = XArm7Pyroki(
            rate=ik_rate,
            viser_server=self.viser_server,
            urdf_path=self.urdf_path,
        )
        self.guard = (
            XArm7FkGuard(
                self.urdf_path,
                minimum_z_m=minimum_z_m,
                self_collision_margin_m=self_collision_margin_m,
            )
            if collision_guard
            else None
        )

        self.enable_handle = self.viser_server.gui.add_checkbox(
            "Enable robot commands", initial_value=False
        )
        self.gripper_handle = self.viser_server.gui.add_slider(
            "Gripper", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )
        self._initialized = False
        self._ik_thread: threading.Thread | None = None
        self._last_command: np.ndarray | None = None
        self._last_observation: dict[str, Any] | None = None
        self._setup_visualization()

    def _setup_visualization(self) -> None:
        self.real_urdf = viser.extras.ViserUrdf(
            self.viser_server,
            deepcopy(self.ik.urdf),
            root_node_name="/xarm7_real",
            mesh_color_override=(0.55, 0.75, 0.95),
        )
        for mesh in self.real_urdf._meshes:
            mesh.opacity = 0.3  # type: ignore[attr-defined]
        self.image_handles: dict[str, viser.GuiImageHandle] = {}

    @staticmethod
    def _extract_joint_pos(obs: dict[str, Any]) -> np.ndarray | None:
        arm_obs = obs.get("left")
        if isinstance(arm_obs, dict) and arm_obs.get("joint_pos") is not None:
            return np.asarray(arm_obs["joint_pos"], dtype=np.float64)
        if obs.get("joint_pos") is not None:
            return np.asarray(obs["joint_pos"], dtype=np.float64)
        return None

    def _start_from_observation(self, joint_pos: np.ndarray) -> None:
        arm = joint_pos[: self.ARM_DOFS]
        gripper = float(joint_pos[-1]) if joint_pos.size > self.ARM_DOFS else 1.0
        cfg = np.concatenate(
            [arm, [normalized_gripper_to_urdf(gripper, self.ik.GRIPPER_TRAVEL_RAD)]]
        )
        self.ik.set_configuration(cfg, move_target=True)
        self.gripper_handle.value = gripper
        self._last_command = joint_pos.copy()
        self._initialized = True
        self._ik_thread = threading.Thread(
            target=self.ik.run, name="xarm7_pyroki_ik", daemon=True
        )
        self._ik_thread.start()

    def _safe_command(self, observed: np.ndarray) -> np.ndarray:
        if not self.enable_handle.value:
            self._last_command = observed.copy()
            return observed.copy()

        desired_arm = self.ik.get_arm_target()
        previous = self._last_command if self._last_command is not None else observed
        arm_delta = np.clip(
            desired_arm - previous[: self.ARM_DOFS],
            -self.max_joint_step_rad,
            self.max_joint_step_rad,
        )
        arm = previous[: self.ARM_DOFS] + arm_delta
        gripper = float(self.gripper_handle.value) if self.enable_gripper else None
        if self.guard is not None:
            try:
                self.guard.validate(arm, gripper if gripper is not None else 1.0)
            except ValueError:
                self.enable_handle.value = False
                self._last_command = observed.copy()
                return observed.copy()

        command = np.concatenate([arm, [gripper]]) if gripper is not None else arm
        self._last_command = command
        return command

    def _update_visualization(self, obs: dict[str, Any], joint_pos: np.ndarray) -> None:
        gripper = float(joint_pos[-1]) if joint_pos.size > self.ARM_DOFS else 1.0
        cfg = np.concatenate(
            [
                joint_pos[: self.ARM_DOFS],
                [normalized_gripper_to_urdf(gripper, self.ik.GRIPPER_TRAVEL_RAD)],
            ]
        )
        self.real_urdf.update_cfg(cfg)
        for key, image in obs_get_rgb(obs).items():
            preview = resize_with_center_crop(image, 224, 224)
            if key not in self.image_handles:
                self.image_handles[key] = self.viser_server.gui.add_image(
                    preview, label=key
                )
            else:
                self.image_handles[key].image = preview

    def act(self, obs: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
        self._last_observation = obs
        joint_pos = self._extract_joint_pos(obs)
        if joint_pos is None:
            return {}
        expected = self.ARM_DOFS + int(self.enable_gripper)
        if joint_pos.shape != (expected,):
            raise ValueError(
                f"xArm7 observation must have shape ({expected},), got {joint_pos.shape}"
            )
        if not self._initialized:
            self._start_from_observation(joint_pos)
        self._update_visualization(obs, joint_pos)
        return {"left": {"pos": self._safe_command(joint_pos).astype(np.float32)}}

    def action_spec(self) -> dict[str, dict[str, Array]]:
        size = self.ARM_DOFS + int(self.enable_gripper)
        return {"left": {"pos": Array(shape=(size,), dtype=np.float32)}}

    def close(self) -> None:
        self.enable_handle.value = False
        self.ik.close()
        self.viser_server.stop()


__all__ = ["XArm7PyrokiViserAgent"]
