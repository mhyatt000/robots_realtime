from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from robots_realtime.robots.xarm7_debug_server import XArm7FkGuard, XArm7JogController


class FakeRobot:
    ARM_DOFS = 7

    def __init__(self) -> None:
        self.enable_gripper = True
        self.state = np.array([0.0] * 7 + [1.0])
        self.commands: list[np.ndarray] = []

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.state = np.asarray(target, dtype=float).copy()
        self.commands.append(self.state.copy())


class FakeGuard:
    minimum_z_m = 0.02

    def __init__(self, reject_above: float | None = None) -> None:
        self.reject_above = reject_above
        self.checked: list[np.ndarray] = []

    def poses(
        self, joints: np.ndarray, gripper_normalized: float = 1.0
    ) -> dict[str, np.ndarray]:
        pose = np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3])
        return {"link_eef": pose, "link_tcp": pose}

    def validate(
        self, joints: np.ndarray, gripper_normalized: float = 1.0
    ) -> dict[str, np.ndarray]:
        joints = np.asarray(joints, dtype=float)
        self.checked.append(joints.copy())
        if self.reject_above is not None and joints[0] > self.reject_above:
            raise ValueError("TCP floor violation")
        return self.poses(joints)


class XArm7JogControllerTest(unittest.TestCase):
    def test_validates_complete_path_before_motion(self) -> None:
        robot = FakeRobot()
        guard = FakeGuard(reject_above=0.05)
        controller = XArm7JogController(robot, guard)

        with self.assertRaisesRegex(ValueError, "floor"):
            controller.move(np.array([0.1] + [0.0] * 6), duration_s=0.1, rate_hz=20)

        self.assertGreater(len(guard.checked), 1)
        self.assertEqual(robot.commands, [])

    def test_interpolates_arm_and_preserves_gripper(self) -> None:
        robot = FakeRobot()
        guard = FakeGuard()
        controller = XArm7JogController(robot, guard)

        controller.move(np.array([0.1] + [0.0] * 6), duration_s=0.01, rate_hz=200)

        self.assertGreaterEqual(len(robot.commands), 1)
        np.testing.assert_allclose(robot.commands[-1][:7], [0.1] + [0.0] * 6)
        self.assertEqual(robot.commands[-1][-1], 1.0)


    def test_max_step_deg_adds_interpolation_steps(self) -> None:
        robot = FakeRobot()
        guard = FakeGuard()
        controller = XArm7JogController(robot, guard)

        controller.move(
            np.array([np.deg2rad(3.0)] + [0.0] * 6),
            duration_s=0.01,
            rate_hz=50.0,
            max_step_deg=1.0,
        )

        self.assertEqual(len(robot.commands), 3)
        positions = np.array([0.0] + [command[0] for command in robot.commands])
        self.assertTrue(np.all(np.diff(positions) <= np.deg2rad(1.0) + 1e-12))

    def test_rejects_total_delta_above_limit_before_motion(self) -> None:
        robot = FakeRobot()
        guard = FakeGuard()
        controller = XArm7JogController(robot, guard)

        with self.assertRaisesRegex(ValueError, "maximum is 30"):
            controller.move(
                np.array([np.deg2rad(31.0)] + [0.0] * 6),
                max_total_delta_deg=30.0,
            )

        self.assertEqual(robot.commands, [])


class XArm7FkGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        urdf = Path(__file__).resolve().parents[1] / "xarm7_standalone.urdf"
        cls.guard = XArm7FkGuard(
            urdf,
            minimum_z_m=0.020,
            self_collision_margin_m=0.005,
        )

    def test_accepts_validated_nominal_pose(self) -> None:
        joints = np.array(
            [-4.48e-4, -7.8567e-1, 2.263e-3, 6.1070e-1, 1.011e-3, 1.1383, 1.5720]
        )
        self.guard.validate(joints, 0.988)
        cfg = self.guard._configuration(joints, 0.988)
        clearances = self.guard.floor_clearances(cfg)
        self.assertAlmostEqual(clearances["link1"], 0.135, places=3)

    def test_gripper_configuration_reverses_hardware_normalization(self) -> None:
        joints = np.zeros(7)
        open_cfg = self.guard._configuration(joints, 1.0)
        closed_cfg = self.guard._configuration(joints, 0.0)
        self.assertEqual(open_cfg[7], 0.0)
        self.assertAlmostEqual(closed_cfg[7], 0.85)

    def test_rejects_known_self_collision(self) -> None:
        joints = np.array([0.49, -1.588, -4.812, 0.183, -3.117, -0.901, -3.044])
        with self.assertRaisesRegex(ValueError, "self-collision.*link2.*link6"):
            self.guard.validate(joints, 0.988)

    def test_rejects_non_tcp_floor_collision(self) -> None:
        joints = np.array(
            [
                -1.0240263940168717,
                -1.0915585835664192,
                4.128452329757378,
                0.8916939900409797,
                -4.017885967695048,
                0.7678718217491172,
                0.08136421125106352,
            ]
        )
        floor_guard = XArm7FkGuard(
            Path(__file__).resolve().parents[1] / "xarm7_standalone.urdf",
            minimum_z_m=0.050,
            self_collision_margin_m=0.005,
        )
        with self.assertRaisesRegex(ValueError, "collision geometry.*floor"):
            floor_guard.validate(joints, 0.988)


if __name__ == "__main__":
    unittest.main()
