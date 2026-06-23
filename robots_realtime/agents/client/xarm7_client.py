"""Port-9000 policy client with Viser visualization for a real xArm7."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots_realtime.agents.teleoperation.xarm7_pyroki_viser_agent import (
    XArm7PyrokiViserAgent,
)
from robots_realtime.utils.server_client_utils import SyncMsgpackNumpyClient


def _value(mapping: dict, key: str) -> Any:
    return mapping.get(key, mapping.get(key.encode()))


class XArm7ClientAgent(XArm7PyrokiViserAgent):
    """Send observations to a framed-msgpack server and execute its xArm targets.

    Expected response::

        {"left": {"joint_pos": array(shape=(7,)), "gripper": scalar}}

    Byte-string keys are also accepted because the shared msgpack decoder uses
    ``raw=True``.
    """

    def __init__(
        self,
        *,
        client_host: str = "127.0.0.1",
        client_port: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.client = SyncMsgpackNumpyClient(host=client_host, port=client_port)

    def _parse_response(self, response: dict) -> np.ndarray | None:
        left = _value(response, "left")
        if not isinstance(left, dict):
            return None
        arm = _value(left, "joint_pos")
        if arm is None:
            return None
        arm = np.asarray(arm, dtype=np.float64)
        if arm.shape != (self.ARM_DOFS,):
            raise ValueError(
                f"xArm client response joint_pos must have shape ({self.ARM_DOFS},), "
                f"got {arm.shape}"
            )
        if not np.all(np.isfinite(arm)):
            raise ValueError("xArm client response joint_pos contains non-finite values")

        if not self.enable_gripper:
            return arm
        gripper_raw = _value(left, "gripper")
        if gripper_raw is None:
            raise ValueError("xArm client response is missing left/gripper")
        gripper = float(np.asarray(gripper_raw).reshape(()))
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ValueError("xArm client response gripper must be in [0, 1]")
        return np.concatenate([arm, [gripper]])

    def _safe_client_command(
        self, observed: np.ndarray, desired: np.ndarray
    ) -> np.ndarray:
        if not self.enable_handle.value:
            self._last_command = observed.copy()
            return observed.copy()

        previous = self._last_command if self._last_command is not None else observed
        arm_delta = np.clip(
            desired[: self.ARM_DOFS] - previous[: self.ARM_DOFS],
            -self.max_joint_step_rad,
            self.max_joint_step_rad,
        )
        arm = previous[: self.ARM_DOFS] + arm_delta
        gripper = float(desired[-1]) if self.enable_gripper else None
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

    def act(self, obs: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
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

        desired = self._parse_response(self.client.send_request(obs))
        command = (
            joint_pos.copy()
            if desired is None
            else self._safe_client_command(joint_pos, desired)
        )
        return {"left": {"pos": command.astype(np.float32)}}

    def close(self) -> None:
        self.client.close()
        super().close()


__all__ = ["XArm7ClientAgent"]
