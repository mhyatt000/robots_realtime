# robots_realtime

A research codebase for real-time robot teleoperation, data collection, and policy deployment.

The collection stack is as modular as the policy itself — agents (GELLO arms, learned policies, interactive IK gizmos) and environments (physical robots, sensors, cameras, MuJoCo sim) are composed at runtime from a YAML config. Swapping a GELLO for a trained policy, or real hardware for sim, is a one-line change. The recording format (MCAP + MP4) is identical regardless — so training pipelines don't need to change when the data source does.

<table>
<tr>
<td><img src="media/yam_realtime.gif" width="360"></td>
<td><img src="media/franka_realtime2.gif" width="360"></td>
</tr>
<tr>
<td><img src="media/yam_active_leader_dagger.gif" width="360"></td>
<td><img src="media/rr_vr_support.gif" width="360"></td>
</tr>
</table>

To build your own YAM active leader arms refer to: [lerobot_teleoperator_yamactiveleader](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)

## Other Documentation
[Architecture & recording format](docs/architecture.md) 

[Extending (new agents, robots, cameras)](docs/extending.md) 

[VR streaming MuJoCo sim to Quest](docs/vr_streaming.md)


---

## Installation

```bash
git clone --recurse-submodules https://github.com/uynitsuj/robots_realtime.git
cd robots_realtime
uv venv --python 3.11 && uv pip install -e .
```

---

## Usage

### Run a session (Teleop Data Collection or Deploy a Policy)

```bash
uv run rr-session configs/yam/yam_sim_gello_teleop.yaml
```
Look under configs for other existing configs

### Replay an episode

```bash
uv run rr-replay recordings/20260323/episode_175805_0473b1bc/
```

Opens a Viser viewer at `http://localhost:8080`. Two modes: **qpos** (exact, restores recorded state) and **physics** (re-simulates from actions).

> Commands run against the project venv via `uv run`. Alternatively: `source .venv/bin/activate` and drop the prefix.
