from __future__ import annotations

import unittest

import numpy as np

from robots_realtime.agents.client.xarm7_client import XArm7ClientAgent


class XArm7ClientAgentTest(unittest.TestCase):
    def make_agent(self, enable_gripper: bool = True) -> XArm7ClientAgent:
        agent = XArm7ClientAgent.__new__(XArm7ClientAgent)
        agent.enable_gripper = enable_gripper
        return agent

    def test_parses_string_key_response(self) -> None:
        agent = self.make_agent()
        target = agent._parse_response(
            {"left": {"joint_pos": np.arange(7), "gripper": 0.25}}
        )
        np.testing.assert_allclose(target, [0, 1, 2, 3, 4, 5, 6, 0.25])

    def test_parses_byte_key_response(self) -> None:
        agent = self.make_agent()
        target = agent._parse_response(
            {b"left": {b"joint_pos": np.arange(7), b"gripper": 0.75}}
        )
        np.testing.assert_allclose(target, [0, 1, 2, 3, 4, 5, 6, 0.75])

    def test_rejects_invalid_joint_shape(self) -> None:
        agent = self.make_agent()
        with self.assertRaisesRegex(ValueError, "shape"):
            agent._parse_response(
                {"left": {"joint_pos": np.arange(6), "gripper": 0.5}}
            )

    def test_rejects_invalid_gripper(self) -> None:
        agent = self.make_agent()
        with self.assertRaisesRegex(ValueError, "gripper"):
            agent._parse_response(
                {"left": {"joint_pos": np.arange(7), "gripper": 1.5}}
            )


if __name__ == "__main__":
    unittest.main()
