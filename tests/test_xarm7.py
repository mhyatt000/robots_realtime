from __future__ import annotations

import unittest

import numpy as np

from robots_realtime.robots.xarm7 import XArm7


class FakeXArm:
    def __init__(self) -> None:
        self.connected = True
        self.error_code = 0
        self.state = 0
        self.angles = [0.0] * 7
        self.gripper_position = 850
        self.calls: list[tuple] = []
        self.servo_code = 0

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return 0

    def clean_warn(self):
        return self._record("clean_warn")

    def clean_error(self):
        return self._record("clean_error")

    def motion_enable(self, **kwargs):
        return self._record("motion_enable", **kwargs)

    def set_mode(self, mode):
        return self._record("set_mode", mode)

    def set_state(self, state):
        self.state = state
        return self._record("set_state", state)

    def clean_gripper_error(self):
        return self._record("clean_gripper_error")

    def set_gripper_enable(self, enabled):
        return self._record("set_gripper_enable", enabled)

    def set_gripper_mode(self, mode):
        return self._record("set_gripper_mode", mode)

    def set_gripper_speed(self, speed):
        return self._record("set_gripper_speed", speed)

    def get_gripper_position(self):
        self.calls.append(("get_gripper_position", (), {}))
        return 0, self.gripper_position

    def set_servo_angle_j(self, angles, **kwargs):
        self.calls.append(("set_servo_angle_j", (angles,), kwargs))
        self.angles = list(angles)
        return self.servo_code

    def set_gripper_position(self, position, **kwargs):
        self.calls.append(("set_gripper_position", (position,), kwargs))
        self.gripper_position = position
        return 0

    def set_servo_angle(self, **kwargs):
        self.calls.append(("set_servo_angle", (), kwargs))
        self.angles = list(kwargs["angle"])
        return 0

    def disconnect(self):
        self.calls.append(("disconnect", (), {}))
        self.connected = False


class XArm7Test(unittest.TestCase):
    def make_robot(self, arm: FakeXArm, **kwargs) -> XArm7:
        return XArm7(arm=arm, sleep_fn=lambda _: None, **kwargs)

    def test_streams_radian_joint_command_and_normalized_gripper(self) -> None:
        arm = FakeXArm()
        robot = self.make_robot(arm)
        target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.25])

        robot.command_joint_pos(target)

        servo_call = next(call for call in arm.calls if call[0] == "set_servo_angle_j")
        self.assertEqual(servo_call[1][0], target[:7].tolist())
        self.assertTrue(servo_call[2]["is_radian"])
        gripper_call = next(call for call in arm.calls if call[0] == "set_gripper_position")
        self.assertEqual(gripper_call[1][0], round(0.25 * 850))

    def test_observation_contains_seven_joints_and_normalized_gripper(self) -> None:
        arm = FakeXArm()
        arm.angles = [0.1] * 7
        arm.gripper_position = 425
        robot = self.make_robot(arm)

        observation = robot.get_observations()

        np.testing.assert_allclose(observation["joint_pos"], [0.1] * 7 + [0.5])

    def test_profiled_move_returns_to_servo_mode(self) -> None:
        arm = FakeXArm()
        robot = self.make_robot(arm, enable_gripper=False)
        arm.calls.clear()

        robot.move_joints(np.array([0.2] * 7), time_interval_s=2.0)

        modes = [call[1][0] for call in arm.calls if call[0] == "set_mode"]
        self.assertEqual(modes, [XArm7.POSITION_MODE, XArm7.SERVO_MODE])
        move_call = next(call for call in arm.calls if call[0] == "set_servo_angle")
        self.assertTrue(move_call[2]["wait"])
        self.assertTrue(move_call[2]["is_radian"])

    def test_rejects_wrong_shape_and_nonfinite_values(self) -> None:
        robot = self.make_robot(FakeXArm(), enable_gripper=False)

        with self.assertRaises(ValueError):
            robot.command_joint_pos(np.zeros(8))
        with self.assertRaises(ValueError):
            robot.command_joint_pos(np.array([0.0] * 6 + [np.nan]))

    def test_sdk_command_error_is_not_ignored(self) -> None:
        arm = FakeXArm()
        arm.servo_code = -1
        robot = self.make_robot(arm, enable_gripper=False)

        with self.assertRaisesRegex(RuntimeError, "set_servo_angle_j"):
            robot.command_joint_pos(np.zeros(7))


if __name__ == "__main__":
    unittest.main()
