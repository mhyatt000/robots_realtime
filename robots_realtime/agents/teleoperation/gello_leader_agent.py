"""Agent that reads joint positions from a GELLO-style feetech leader device
and outputs joint-position actions for a simulated (or real) follower robot.

The leader device is accessed through the ``YamActiveLeaderTeleoperator``
class from the lerobot plugin.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from dm_env.specs import Array

import lerobot.robots  # noqa: F401 — resolve circular import in lerobot

from lerobot_teleoperator_yamactiveleader import (
    YamActiveLeaderTeleoperator,
    YamActiveLeaderTeleoperatorConfig,
)

from robots_realtime.agents.agent import Agent

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6


class GelloLeaderAgent(Agent):
    """Teleoperation agent backed by a GELLO feetech leader arm.

    Reads the leader's joint positions (in degrees) and converts them
    to radians for output as follower joint-position commands.

    Args:
        port: Serial port for the feetech bus
            (e.g. ``/dev/tty.usbmodem5AE60805531``).
        robot_name: Key used in the returned action dict
            (must match the robot name in the env config).
        calibrate: Whether to run calibration on connect
            (set to False if already calibrated).
        joint_signs: Per-joint sign multipliers (length 6).
            Use ``-1`` to flip a joint direction.  Defaults to all ``1``.
        joint_offsets_deg: Per-joint offsets in degrees added *after*
            sign flip, *before* conversion to radians.  Defaults to
            all ``0``.
        use_degrees: If True the teleoperator is configured to return
            positions in degrees (default).
        drive_to_zero: If True (default), actively drive all motors to
            the zero-config position on startup, then release the arm
            joints so the operator can move them.
        hold_gripper: If True (default), keep torque enabled on the
            gripper motor so it actively resists changes (holds the
            open position).
        include_gripper: If True, include the gripper value as a 7th
            element in the action output.  Set to False for sim models
            that have no gripper joint.  Defaults to False.
    """

    use_joint_state_as_action: bool = False

    # Reasonable mock pose for DAgger debugging: slight forward reach with elbow bent.
    # Override via dagger_debug_pose_rad in the constructor or YAML config.
    DAGGER_DEBUG_POSE_RAD: List[float] = [0.2, 0.7, 0.9, -0.7, 0.7, 0.2]

    def __init__(
        self,
        port: str = "/dev/tty.usbmodem5AE60805531",
        robot_name: str = "left",
        calibrate: bool = True,
        joint_signs: Optional[List[int]] = None,
        joint_offsets_deg: Optional[List[float]] = None,
        use_degrees: bool = True,
        drive_to_zero: bool = True,
        hold_gripper: bool = True,
        include_gripper: bool = False,
        dagger_debug: bool = False,
        dagger_debug_pose_rad: Optional[List[float]] = False,
    ) -> None:
        self.robot_name = robot_name
        self.joint_signs = np.array(joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.joint_offsets_deg = np.array(
            joint_offsets_deg or [0] * NUM_ARM_JOINTS, dtype=np.float64
        )
        self.include_gripper = include_gripper
        self._held_action: Optional[Dict[str, Any]] = None

        config = YamActiveLeaderTeleoperatorConfig(port=port, use_degrees=use_degrees)
        self.teleop = YamActiveLeaderTeleoperator(config)
        self.teleop.connect(calibrate=calibrate)
        logger.info("GelloLeaderAgent connected to %s", port)

        # ---- Active motor control at startup ---- #
        if drive_to_zero:
            self.teleop.drive_to_zero()

        if hold_gripper:
            self.teleop.start_gripper_spring()

        if dagger_debug:
            pose = np.array(
                dagger_debug_pose_rad or self.DAGGER_DEBUG_POSE_RAD, dtype=np.float64
            )
            # Invert agent transform: output_rad = deg2rad(signs * leader_deg + offsets)
            # → leader_deg = (rad2deg(output_rad) - offsets) * signs  (signs are ±1)
            target_deg = (np.rad2deg(pose) - self.joint_offsets_deg) * self.joint_signs
            target_dict = {f"joint_{i+1}": float(target_deg[i]) for i in range(NUM_ARM_JOINTS)}
            arm_pos = pose[:NUM_ARM_JOINTS].astype(np.float32)
            if include_gripper:
                raw = self.teleop.get_action()
                arm_pos = np.concatenate([arm_pos, [raw["gripper.pos"]]]).astype(np.float32)
            self._held_action = {self.robot_name: {"pos": arm_pos}}
            self.teleop.drive_to_config(target_dict)
            self.teleop.start_arm_hold()
            logger.info("DAgger debug hold armed at pose (rad): %s", pose)

    # ------------------------------------------------------------------ #
    # DAgger intervention support
    # ------------------------------------------------------------------ #

    @property
    def is_intervening(self) -> bool:
        """True once the human has actively pushed against the held pose.

        Useful for a DAgger data-collection loop to label transitions
        between policy rollout and human correction.
        """
        return self.teleop.is_arm_hold_intervening

    def clear_intervention(self) -> None:
        """Reset DAgger hold state to begin a new cycle."""
        self.teleop.clear_arm_hold()

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Read the leader device and return a position action.

        While an arm hold is active (after :pymeth:`drive_to_config` but
        before intervention is detected), returns the held config and polls
        the intervention detector each tick.  Once the human pushes,
        resumes live teleoperation immediately.

        Returns:
            ``{robot_name: {"pos": np.ndarray}}``  —  6 arm-joint angles
            in radians (optionally followed by 1 gripper value if
            *include_gripper* is True).
        """
        # --- Hold mode: return held config and monitor for intervention ---
        if self.teleop.is_arm_hold_active:
            self.teleop.update_arm_hold()
            if not self.teleop.is_arm_hold_intervening:
                return self._held_action

        # --- Normal teleoperation ---
        action = self.teleop.get_action()

        # Adaptive gripper spring — adjusts torque each frame
        # self.teleop.update_gripper_spring(action["gripper.pos"])
        # self.teleop.read_gripper_spring_state()

        # Extract the 6 arm joints (degrees)
        joint_deg = np.array([action[f"joint_{i}.pos"] for i in range(1, NUM_ARM_JOINTS + 1)])

        # Apply sign flips and offsets, then convert to radians
        joint_deg = self.joint_signs * joint_deg + self.joint_offsets_deg
        joint_rad = np.deg2rad(joint_deg)

        if self.include_gripper:
            gripper = action["gripper.pos"]
            # gripper = np.clip(1 - ((gripper - 5) / (85 - 5)), 0, 1)  # normalize gripper position to 0-1 range, w/ some deadzone
            pos = np.concatenate([joint_rad, [gripper]])
        else:
            pos = joint_rad

        return {self.robot_name: {"pos": pos.astype(np.float32)}}

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(n,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self.teleop.disconnect()
        logger.info("GelloLeaderAgent disconnected.")

    def reset(self) -> None:
        pass


class BimanualGelloLeaderAgent(Agent):
    """Teleoperation agent backed by two GELLO feetech leader arms (bimanual).

    Reads both leader arms and returns a combined 14-DOF action
    ``[left_j1..6, left_grip, right_j1..6, right_grip]`` under a single
    robot key.  This matches the layout expected by
    ``XdofSimRobot`` when ``right_arm_only=False``.

    Args:
        left_port: Serial port for the left-arm feetech bus.
        right_port: Serial port for the right-arm feetech bus.
        robot_name: Key used in the returned action dict (must match the
            robot key in the env config, e.g. ``"right"``).
        calibrate: Whether to run calibration on connect.
        left_joint_signs: Per-joint sign multipliers for the left arm.
        right_joint_signs: Per-joint sign multipliers for the right arm.
        left_joint_offsets_deg: Per-joint degree offsets for the left arm.
        right_joint_offsets_deg: Per-joint degree offsets for the right arm.
        use_degrees: Pass-through to ``YamActiveLeaderTeleoperatorConfig``.
        drive_to_zero: Drive both arms to zero on startup.
        hold_gripper: Keep gripper torque enabled on both arms.
        include_gripper: Include the gripper value (7th element per arm).
            Defaults to True for bimanual.
    """

    def __init__(
        self,
        left_port: str,
        right_port: str,
        left_id: str,
        right_id: str,
        robot_name: str = "yam_bimanual",
        calibrate: bool = True,
        left_joint_signs: Optional[List[int]] = None,
        right_joint_signs: Optional[List[int]] = None,
        left_joint_offsets_deg: Optional[List[float]] = None,
        right_joint_offsets_deg: Optional[List[float]] = None,
        use_degrees: bool = True,
        drive_to_zero: bool = True,
        hold_gripper: bool = True,
        include_gripper: bool = True,
    ) -> None:
        self.left_id = left_id
        self.right_id = right_id

        self.robot_name = robot_name
        self.include_gripper = include_gripper

        self.left_joint_signs = np.array(left_joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.right_joint_signs = np.array(right_joint_signs or [1] * NUM_ARM_JOINTS, dtype=np.float64)
        self.left_joint_offsets_deg = np.array(
            left_joint_offsets_deg or [0] * NUM_ARM_JOINTS, dtype=np.float64
        )
        self.right_joint_offsets_deg = np.array(
            right_joint_offsets_deg or [0] * NUM_ARM_JOINTS, dtype=np.float64
        )

        left_cfg = YamActiveLeaderTeleoperatorConfig(port=left_port, use_degrees=use_degrees, id=left_id)
        self.left_teleop = YamActiveLeaderTeleoperator(left_cfg)
        self.left_teleop.connect(calibrate=calibrate)
        logger.info("BimanualGelloLeaderAgent: left arm connected to %s", left_port)

        right_cfg = YamActiveLeaderTeleoperatorConfig(port=right_port, use_degrees=use_degrees, id=right_id)
        self.right_teleop = YamActiveLeaderTeleoperator(right_cfg)
        self.right_teleop.connect(calibrate=calibrate)
        logger.info("BimanualGelloLeaderAgent: right arm connected to %s", right_port)

        if drive_to_zero:
            self.left_teleop.drive_to_zero()
            self.right_teleop.drive_to_zero()

        if hold_gripper:
            self.left_teleop.start_gripper_spring()
            self.right_teleop.start_gripper_spring()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_arm(
        self,
        teleop: YamActiveLeaderTeleoperator,
        signs: np.ndarray,
        offsets_deg: np.ndarray,
    ) -> np.ndarray:
        """Read one arm and return joint_rad [+ gripper] as a 1-D array."""
        action = teleop.get_action()
        joint_deg = np.array([action[f"joint_{i}.pos"] for i in range(1, NUM_ARM_JOINTS + 1)])
        joint_deg = signs * joint_deg + offsets_deg
        joint_rad = np.deg2rad(joint_deg)
        if self.include_gripper:
            return np.concatenate([joint_rad, [action["gripper.pos"]]])
        return joint_rad

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Return combined 14-DOF (or 12-DOF without gripper) action.

        Layout: ``[left_j1..6, (left_grip,) right_j1..6, (right_grip,)]``

        Returns:
            ``{robot_name: {"pos": np.ndarray}}``
        """
        left_pos = self._read_arm(self.left_teleop, self.left_joint_signs, self.left_joint_offsets_deg)
        right_pos = self._read_arm(self.right_teleop, self.right_joint_signs, self.right_joint_offsets_deg)
        combined = np.concatenate([left_pos, right_pos]).astype(np.float32)
        return {self.robot_name: {"pos": combined}}

    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        n_per_arm = NUM_ARM_JOINTS + (1 if self.include_gripper else 0)
        return {self.robot_name: {"pos": Array(shape=(2 * n_per_arm,), dtype=np.float32)}}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self.left_teleop.disconnect()
        self.right_teleop.disconnect()
        logger.info("BimanualGelloLeaderAgent disconnected.")

    def reset(self) -> None:
        pass
