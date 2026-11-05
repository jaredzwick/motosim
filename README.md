# motosim

First attempt at an Open-source motorcycle autonomy research project. Sim-first. Not self-driving — not yet.

https://github.com/user-attachments/assets/062b5ac2-26c3-44dd-b3e1-15544b20794f

The bike's FPV camera view through the slalom with live HSV cone detection
overlaid. Green bounding boxes are the raw classical-CV detections; the
"cones detected: N" HUD updates per frame. Rendered offscreen via
`see.py` at 640x480 / 30 fps.


Chase-cam view of the bike using the v1.1 MPC planner (casadi + ipopt,
kinematic bicycle model, 2s horizon, re-plans at 20 Hz) to navigate the
slalom with cone 0 placed dead-ahead at x=15. HUD shows the current MPC
steer command, bike state, and the FPV camera inset (bottom-left) with
live cone detections. Rendered via `see_autonomous.py` at 1280x720 / 30 fps.

**Status:** v1.1 — Whipple-topology CBR500R MJCF with suspension, four-channel actuator stack (throttle + rear_brake + front_brake + position-controlled steer), cone-slalom test track with a dead-ahead obstacle, perception pipeline (HSV detector + inverse perspective mapping), MPC trajectory planner, and PPO training infrastructure (v1.0 lean controller). Architecture-complete; MPC tuning for lean-dynamics mismatch is ongoing (see v1.2 roadmap).

## What this is

A simulation framework for motorcycle autonomy research, starting with the 2015
Honda CBR500R as a reference platform. Built on MuJoCo, published under MIT.

**Year one is simulation only.** No physical bike mods, no actuators, no public-road
anything. Validation runs in MuJoCo and later MJX for parallel RL training.

## What this is NOT

- Self-driving software for public roads
- A commercial product
- Safety-critical in any way
- Something to run on an actual motorcycle in 2026

## Six premises (the ground rules)

1. **Software and sensor layer only.** Not a motorcycle OEM.
2. **Perception + warning, not control.** Rider stays in the loop when hardware eventually enters.
3. **CBR500R is reference, not the only platform.** Code must stay vehicle-agnostic.
4. **Self-balancing is a hardware problem** (Honda Riding Assist geometry), not v1 software.
5. **Simulation first.** Sim-to-real only after sim maturity.
6. **No actuator-controlling release until year two.** Closed-course only.

## The CBR500R model (v0.1)

File: [`cbr500r.xml`](./cbr500r.xml)

Topology follows the Whipple bicycle model from Meijaard, Papadopoulos, Ruina,
and Schwab (2007, "Linearized dynamics equations for the balance and steer of a
bicycle: a benchmark and review", Proc. Roy. Soc. A). Four rigid bodies:

- `rear_frame` — chassis + engine + fuel tank (150 kg)
- `rear_wheel` — 20 kg, spins about lateral axis
- `front_frame` — fork + triple clamp + handlebars (12 kg), hinges about steer axis
- `front_wheel` — 12 kg, spins about lateral axis

Total mass: 194 kg (matches Honda CBR500R wet weight, excludes rider).

### Parameters (verified from Honda 2015 CBR500R spec sheet)

| Parameter | Value |
|---|---|
| Wheelbase | 1.410 m |
| Rake angle | 25° (0.436 rad) |
| Trail | 0.102 m |
| Front tire | 120/70 ZR17 (R = 0.306 m) |
| Rear tire | 160/60 ZR17 (R = 0.312 m) |
| Seat height | 0.785 m |
| Wet weight (no rider) | 194 kg |

### Known approximations

- **No suspension.** Rigid fork and swingarm. v0.2.
- **No Pacejka tire model.** MuJoCo elliptic friction cone only. Lateral grip above ~30° lean will diverge from real bike behavior. v0.3+ if ever.
- **Cylinder wheels, not toroidal.** Edge contact approximation. v0.3.
- **Inertia tensors are rough.** First-pass estimates. If self-stability validation fails, tune these.
- **No rider body.** 194 kg is wet weight only. Project is autonomous use case.

## Running it

### Requirements

- Python 3.12 (pin via venv — `tensorstore` has no wheels for 3.14+)
- macOS Apple Silicon or Linux

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Interactive view

```bash
python view.py
```

Opens the MuJoCo viewer. Drag to orbit. Space to pause. The bike will fall
sideways within a second because it has zero forward velocity and no
gyroscopic stability — this is expected.

### Self-stability validation (v0.1 acceptance gate)

```bash
python validate.py           # headless
mjpython validate.py --render  # watch it happen (needs mjpython on macOS)
```

The bike is pushed to 5 m/s with a small 2.9° lean perturbation, then released
with no control input. Expected: self-stabilizes within 2 seconds, holds speed
for 15 seconds. This is the analytic prediction from Meijaard 2007 for
Whipple-topology bikes between their weave and capsize eigenvalue speeds.

If it fails, the inertia tensors in `cbr500r.xml` need tuning. The topology is
correct by construction.

### Actuator validation (v0.3 acceptance gate)

```bash
python brake_test.py
```

Runs four tests headless: full-throttle acceleration 0→10 m/s, then braking
distances under rear-only, front-only, and combined 60F/40R braking strategies.
Pass criteria: bike reaches 10 m/s, stops within 15 seconds on each brake test,
and chassis stays above z=0.3 (doesn't go over the bars). Expected ordering:
front_only < combined < rear_only stopping distance.

### Drive the slalom (v0.3 cone course)

```bash
mjpython drive.py
```

Bike spawns at origin at 5 m/s, PD lean controller weaves through 8 cones at
x=10,17,24,…,59 (y alternates ±2 m). Chase camera follows. Press `[` / `]` to
switch to fixed cameras (chase | side | overhead), space to pause, R to reset.

### Learned lean controller (v1.0 PPO training)

```bash
# Short local run on CPU (~30-60 min on M4):
python train.py

# Longer / production training (GPU recommended):
python train.py --num-timesteps 2_000_000 --num-envs 1024

# Play the trained policy in the native MuJoCo viewer:
mjpython rollout.py                           # loads latest checkpoint
mjpython rollout.py --command 0.1             # fixed commanded lean 0.1 rad
mjpython rollout.py --command sine            # oscillating lean command
```

Trains a tiny MLP (two hidden layers of 64) via PPO on the `MotorcycleBalance`
MJX environment to track commanded lean angles without falling over. Everything
other than steer torque (throttle, brakes) uses fixed controllers — only the
balance skill is learned.

**Observation (6-dim, proprioceptive):** lean, lean_rate, steer_angle,
steer_rate, forward_speed, commanded_lean. No vision.

**Action (1-dim):** steer torque in [-10, +10] N·m.

**Reward:** alive_bonus − 10 × (lean − commanded)² − 0.01 × action² − 100 × fallen.

**Episode:** 5 seconds (2500 physics steps at timestep 0.004). Commanded lean
resamples every 1 second to a random value in [-0.15, +0.15] rad.

**Where to train:** CPU works but is slow (~1k env-steps/sec per env on M4).
Cloud GPU is dramatically faster — Lambda Labs or vast.ai A100 at ~$1–2/hour
finishes a 2M-step run in 10–15 minutes. The checkpoint pickle transfers back
to your laptop and loads into `rollout.py`.

**Known deps gotcha:** `brax`, `jax`, `jaxlib`, `flax` tend to lag Python
releases by 6-12 months. If `pip install brax` fails on 3.14+, use a Python
3.12 venv for training (native MuJoCo on 3.15 is fine for everything else).

### Autonomous slalom (v0.5 perception-driven)

```bash
mjpython autonomous.py                       # interactive viewer
mjpython autonomous.py --drop 0.3            # drop 30% of perception cycles (failure test)
mjpython autonomous.py --noise 0.4           # add 0.4 m gaussian noise per detection
mjpython autonomous.py --duration 30         # auto-exit after 30s wall-clock
```

Unlike `drive.py` (which uses a hardcoded cone-position schedule), `autonomous.py`
runs the FPV camera at 20 Hz, detects cones each frame, projects every detection
to world coordinates via ground-plane ray intersection, and picks the lean
target from the nearest cone *ahead of the bike*. The controller does NOT know
where the cones are placed — it only knows what perception says.

Press **D** in the viewer to toggle a per-perception-cycle terminal readout
showing bike position, how many cones were seen, and the current lean target.

Failure testing:

- `--drop <frac>` — fraction of perception cycles that drop all detections.
  Simulates camera dropout or inference failures. Controller falls back to
  "zero lean" (go straight) when no detections are current.
- `--noise <std>` — meters of gaussian noise added to each world-xy estimate.
  Simulates calibration error, pose drift, or motion blur.

Try `--drop 0.5` — the bike mostly makes it through even when half of perception
frames are blank, because it's always acting on the most recent valid detections.
At `--drop 0.8` or `--noise 1.5` it starts eating cones.

### Perception demo (v0.4 cone detection video)

```bash
python see.py                           # 15s sim, saves perception.mp4 at 640x480 30fps
python see.py --duration 30             # longer run
python see.py --resolution 1280 720     # 720p for YouTube
python see.py --no-overlay              # raw FPV camera, no bounding boxes
```

Runs the slalom controller, renders the FPV camera (frame-mounted, not
steering-mounted), applies classical HSV cone detection on each frame, and
writes an MP4 with bounding boxes overlaid on detected cones.

Expected output: a 15-second clip where you see the bike's POV weaving
through the slalom, with green bounding boxes drawn around each visible
orange cone and a "cones detected: N" HUD label in the top-left.

The detector is `perception.py` — a 70-line classical HSV + connected-
component implementation. Zero ML dependencies. For the simple case of
orange cones on grey asphalt it works perfectly; for production motorcycle
autonomy you'd eventually swap it for a trained detector (YOLOv10, DETR,
or similar). See the module docstring for the upgrade path.

### Actuator interface (for future contributors)

Four control channels, each a direct torque motor in `cbr500r.xml`:

| Actuator | Joint | ctrlrange (N·m) | Semantics |
|---|---|---|---|
| `throttle` | `rear_spin` | `0` to `+70` | positive torque only; engines don't push backward |
| `rear_brake` | `rear_spin` | `-80` to `0` | negative torque; use friction-model gating in controller |
| `front_brake` | `front_spin` | `-200` to `0` | ~60-70% of total braking authority |
| `steer_motor` | `steer` | `-10` to `+10` | direct torque, bidirectional |

Set via `data.ctrl[model.actuator("<name>").id] = value`. See `drive.py` for a
reference controller implementation.

## Roadmap

- ~~**v0.1**: primitive MJCF, self-stability validation~~
- ~~**v0.2**: add telescopic fork + swingarm suspension, re-validate~~
- ~~**v0.3**: throttle + brake + steer actuators (sim-only, no hardware)~~
- ~~**v0.4**: perception stack — FPV camera + HSV cone detection, offline demo video~~
- ~~**v0.5**: perception-in-the-loop — inverse perspective mapping, `autonomous.py` driving the slalom on perception alone, `--drop` and `--noise` failure flags~~
- ~~**v1.0**: PPO lean controller on brax/MJX — `motosim_env.py` (Brax env), `train.py` (PPO), `rollout.py` (play learned policy). Baseline lean-command tracking task.~~
- ~~**v1.1**: MPC trajectory planner (casadi + ipopt) replacing reactive controller. Kinematic bicycle model, 2s horizon, 20 Hz re-planning, soft obstacle costs + steer-rate constraints. Architecture-complete; tuning for the real motorcycle's lean-dynamics response is v1.2.~~
- **v1.2** (next): extend MPC planner with lean-dynamics state (adds 1 state + 1 equation). Fixes the current oscillation when lean-lag causes real-bike response to diverge from planner prediction. Or: train a learned trajectory tracker (PPO) to convert planner-trajectory to steer commands that properly handle lean dynamics.
- **v2.0** (year 2): second vehicle (MT-07 URDF) to prove vehicle-agnostic architecture

Public-road actuation: never, under this license.

## License

MIT. See LICENSE.

## References

- Meijaard, Papadopoulos, Ruina, Schwab. "Linearized dynamics equations for the balance and steer of a bicycle: a benchmark and review." Proc. Roy. Soc. A, 2007.
- Cossalter, Vittore. *Motorcycle Dynamics.* 2nd ed. 2006.
- Honda CBR500R/CBR500R ABS 2013–2015 Service Manual.
- MuJoCo documentation: https://mujoco.readthedocs.io/
