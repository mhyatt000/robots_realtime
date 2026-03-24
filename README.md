# robots_realtime

A research codebase for real-time robot teleoperation, data collection, and policy deployment.

### Why robots_realtime?
- **Unified Pipeline:** Collect data in MuJoCo sim or on real hardware platforms, and deploy learned policies with the same infrastructure.
- **Modular Stack:** Switch between GELLO leader arms, IK gizmos, Franka or I2RT YAM robot hardware via runtime YAML configs.
- **High Frequency:** Built on ZeroMQ nodes for asynchronous, low-latency real-time control for reactive policies.

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
# if already cloned, or some of the submodules are incompletely cloned, run
git submodule update --init --recursive
cd robots_realtime
uv venv --python 3.11 && uv pip install -e .
```

---
## Configuration
If using YAM arms, configure YAM arms CAN chain according to instructions from the [I2RT repo](https://github.com/i2rt-robotics/i2rt)

## Usage / Quickstart

### Run a teleop data collection session in sim using [3d printed leader arms](https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader)

```bash
uv run rr-session configs/yam/yam_sim_gello_teleop.yaml
```
Then you should see the terminal populate with TUI (TerminalUI) 
```
╭───────────────────────────────────────────────────────────────────────────────────── robots_realtime ─────────────────────────────────────────────────────────────────────────────────────╮
│   NODE                                                   STATUS                                      HZ    TOPICS                                                                         │
│   gello_left                                             ● live                                   254.3    joint_pos                                                                      │
│   gello_right                                            ● live                                   254.3    joint_pos                                                                      │
│   yam                                                    ● live                                    29.7    left_state, right_state                                                        │
│ http://localhost:8765  (viser)  http://localhost:8012  (vr)                                                                                                                               │
│ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ ○  idle                                                                                                                                                 [r] record  [d] discard  [q] quit │
│ ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ [gello_left] === left YamActiveLeaderTeleoperator (port=/dev/ttyACM0) ===                                                                                               │
│ [gello_right] === right YamActiveLeaderTeleoperator (port=/dev/ttyACM1) ===                                                                                             │                                                                                             │
│ [yam] ╭────── viser (listening *:8765) ───────╮                                                                                                                                           │
│ [yam] │             ╷                         │                                                                                                                                           │
│ [yam] │   HTTP      │ http://localhost:8765   │                                                                                                                                           │
│ [yam] │   Websocket │ ws://localhost:8765     │                                                                                                                                           │
│ [yam] │             ╵                         │                                                                                                                                           │
│ [yam] ╰───────────────────────────────────────╯                                                                                                                                           │
│   logs: /tmp/rr_logs_n8map8js                                                                                                                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
Look under configs for other existing configs

### Replay an episode

```bash
uv run rr-replay recordings/20260323/episode_175805_0473b1bc/
```

Opens a Viser viewer at `http://localhost:8080`. Two modes: **qpos** (exact, restores recorded state) and **physics** (re-simulates from actions).

> Commands run against the project venv via `uv run`. Alternatively: `source .venv/bin/activate` and drop the prefix.
