"""xArm7 robot driver backed by UFACTORY's xArm Python SDK.

The robots_realtime boundary always uses radians. When ``enable_gripper`` is
true, the command/state vector is ``[7 arm joints, normalized gripper]`` where
the gripper value is 0.0 for closed and 1.0 for open.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def normalized_gripper_to_urdf(
    normalized: float, travel_rad: float = 0.85
) -> float:
    """Convert hardware convention (0=closed, 1=open) to URDF drive angle."""
    value = float(normalized)
    if not 0.0 <= value <= 1.0:
        raise ValueError("normalized gripper position must be in [0, 1]")
    return (1.0 - value) * float(travel_rad)


class XArm7:
    """Joint-position controller for a UFACTORY xArm7."""

    ARM_DOFS = 7
    SERVO_MODE = 1
    POSITION_MODE = 0
    READY_STATE = 0
    STOP_STATE = 4

    def __init__(
        self,
        host_name: str = "192.168.1.231",
        name: str = "xarm7",
        enable_gripper: bool = True,
        gripper_open_position: int = 850,
        gripper_closed_position: int = 0,
        gripper_speed: int = 5000,
        gripper_command_epsilon: float = 0.01,
        gripper_poll_interval_s: float = 0.1,
        clear_errors_on_startup: bool = False,
        connect_timeout_s: float = 10.0,
        arm: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = name
        self.host_name = host_name
        self.enable_gripper = enable_gripper
        self._gripper_open_position = int(gripper_open_position)
        self._gripper_closed_position = int(gripper_closed_position)
        self._gripper_command_epsilon = float(gripper_command_epsilon)
        self._gripper_poll_interval_s = float(gripper_poll_interval_s)
        self._sleep = sleep_fn
        self._last_gripper_command: float | None = None
        self._last_gripper_state = 1.0
        self._last_gripper_poll_time = -math.inf
        self._stopped = False

        if self._gripper_open_position == self._gripper_closed_position:
            raise ValueError("gripper_open_position and gripper_closed_position must differ")
        if self._gripper_command_epsilon < 0:
            raise ValueError("gripper_command_epsilon must be non-negative")
        if self._gripper_poll_interval_s < 0:
            raise ValueError("gripper_poll_interval_s must be non-negative")

        if arm is None:
            try:
                XArmAPI = importlib.import_module("xarm.wrapper").XArmAPI
            except ImportError as exc:
                raise ImportError(
                    "xArm7 support requires xarm-python-sdk. "
                    "Install the project with the 'xarm7' extra."
                ) from exc
            arm = XArmAPI(
                host_name,
                is_radian=True,
                do_not_open=False,
                timeout=connect_timeout_s,
            )
        self.arm = arm

        if not self.arm.connected:
            raise ConnectionError(f"Could not connect to xArm7 at {host_name}")

        if clear_errors_on_startup:
            self._check_code(self.arm.clean_warn(), "clean_warn")
            self._check_code(self.arm.clean_error(), "clean_error")
        elif int(getattr(self.arm, "error_code", 0)) != 0:
            raise RuntimeError(
                f"xArm7 reports error_code={self.arm.error_code}; clear the fault "
                "on the controller or set clear_errors_on_startup=true"
            )

        self._check_code(self.arm.motion_enable(enable=True), "motion_enable")
        self._enter_servo_mode()

        if self.enable_gripper:
            self._check_code(self.arm.clean_gripper_error(), "clean_gripper_error")
            self._check_code(self.arm.set_gripper_enable(True), "set_gripper_enable")
            self._check_code(self.arm.set_gripper_mode(0), "set_gripper_mode")
            self._check_code(self.arm.set_gripper_speed(gripper_speed), "set_gripper_speed")
            self._refresh_gripper_state(force=True)

        logger.info("Connected to %s at %s", self.name, self.host_name)

    def __repr__(self) -> str:
        return (
            f"XArm7(name={self.name!r}, host_name={self.host_name!r}, "
            f"enable_gripper={self.enable_gripper})"
        )

    def num_dofs(self) -> int:
        return self.ARM_DOFS + int(self.enable_gripper)

    def get_joint_pos(self) -> np.ndarray:
        self._assert_connected()
        joint_pos = np.asarray(self.arm.angles, dtype=np.float64)
        if joint_pos.shape != (self.ARM_DOFS,):
            raise RuntimeError(
                f"xArm7 returned {joint_pos.shape} joint positions; expected ({self.ARM_DOFS},)"
            )
        if not self.enable_gripper:
            return joint_pos.copy()

        self._refresh_gripper_state()
        return np.concatenate([joint_pos, [self._last_gripper_state]])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        target = self._validate_target(joint_pos)
        self._assert_ready()

        code = self.arm.set_servo_angle_j(
            target[: self.ARM_DOFS].tolist(),
            is_radian=True,
        )
        self._check_code(code, "set_servo_angle_j")

        if self.enable_gripper:
            self._command_gripper(float(target[-1]))

    def move_joints(self, joint_pos: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Execute a blocking controller-profiled move, then resume servo mode."""
        target = self._validate_target(joint_pos)
        duration = max(float(time_interval_s), 0.01)
        current = self.get_joint_pos()[: self.ARM_DOFS]
        speed = max(float(np.max(np.abs(target[: self.ARM_DOFS] - current))) / duration, 0.05)

        self._check_code(self.arm.set_mode(self.POSITION_MODE), "set_mode(position)")
        self._check_code(self.arm.set_state(self.READY_STATE), "set_state(ready)")
        self._sleep(0.1)
        try:
            code = self.arm.set_servo_angle(
                angle=target[: self.ARM_DOFS].tolist(),
                speed=speed,
                is_radian=True,
                wait=True,
                timeout=duration + 10.0,
            )
            self._check_code(code, "set_servo_angle")
            if self.enable_gripper:
                self._command_gripper(float(target[-1]), wait=True)
        finally:
            self._enter_servo_mode()

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"joint_pos": self.get_joint_pos()}

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if getattr(self.arm, "connected", False):
                code = self.arm.set_mode(self.POSITION_MODE)
                if code != 0:
                    logger.warning("xArm7 set_mode(position) during stop returned code %s", code)
                code = self.arm.set_state(self.STOP_STATE)
                if code != 0:
                    logger.warning("xArm7 set_state(stop) returned code %s", code)
        finally:
            self.arm.disconnect()

    close = stop

    def _enter_servo_mode(self) -> None:
        self._check_code(self.arm.set_mode(self.SERVO_MODE), "set_mode(servo)")
        self._check_code(self.arm.set_state(self.READY_STATE), "set_state(ready)")
        self._sleep(0.1)

    def _validate_target(self, joint_pos: np.ndarray) -> np.ndarray:
        target = np.asarray(joint_pos, dtype=np.float64)
        expected = (self.num_dofs(),)
        if target.shape != expected:
            raise ValueError(f"xArm7 expected command shape {expected}, got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("xArm7 joint command contains non-finite values")
        if self.enable_gripper and not 0.0 <= target[-1] <= 1.0:
            raise ValueError(
                f"xArm7 normalized gripper command must be in [0, 1], got {target[-1]}"
            )
        return target

    def _command_gripper(self, normalized: float, wait: bool = False) -> None:
        if (
            not wait
            and self._last_gripper_command is not None
            and abs(normalized - self._last_gripper_command) < self._gripper_command_epsilon
        ):
            return
        raw_position = round(
            self._gripper_closed_position
            + normalized * (self._gripper_open_position - self._gripper_closed_position)
        )
        code = self.arm.set_gripper_position(raw_position, wait=wait)
        self._check_code(code, "set_gripper_position")
        self._last_gripper_command = normalized
        if wait:
            self._last_gripper_state = normalized

    def _refresh_gripper_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_gripper_poll_time < self._gripper_poll_interval_s:
            return
        code, raw_position = self.arm.get_gripper_position()
        self._check_code(code, "get_gripper_position")
        span = self._gripper_open_position - self._gripper_closed_position
        normalized = (float(raw_position) - self._gripper_closed_position) / span
        self._last_gripper_state = float(np.clip(normalized, 0.0, 1.0))
        self._last_gripper_poll_time = now

    def _assert_connected(self) -> None:
        if self._stopped or not getattr(self.arm, "connected", False):
            raise ConnectionError(f"xArm7 at {self.host_name} is disconnected")

    def _assert_ready(self) -> None:
        self._assert_connected()
        error_code = int(getattr(self.arm, "error_code", 0))
        if error_code != 0:
            raise RuntimeError(f"xArm7 controller error_code={error_code}")
        state = int(getattr(self.arm, "state", self.READY_STATE))
        if state >= self.STOP_STATE:
            raise RuntimeError(f"xArm7 is not ready for motion (state={state})")

    @staticmethod
    def _check_code(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"xArm7 SDK call {operation} failed with code {code}")
