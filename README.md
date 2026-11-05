# motosim

Open-source motorcycle autonomy research. Sim-first. Not self-driving — not yet.

**Status:** v0.1 scaffold. First working CBR500R MJCF, no actuators, no controller.

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
pip install mujoco numpy
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
python validate.py --render  # watch it happen
```

The bike is pushed to 5 m/s with a small 2.9° lean perturbation, then released
with no control input. Expected: self-stabilizes within 2 seconds, holds speed
for 15 seconds. This is the analytic prediction from Meijaard 2007 for
Whipple-topology bikes between their weave and capsize eigenvalue speeds.

If it fails, the inertia tensors in `cbr500r.xml` need tuning. The topology is
correct by construction.

## Roadmap

- **v0.1** (current): primitive MJCF, self-stability validation
- **v0.2**: add telescopic fork + swingarm suspension, re-validate
- **v0.3**: throttle + brake + steer actuators (sim-only, no hardware)
- **v0.4**: perception stack (camera sensor, YOLO-class object detection)
- **v0.5**: visual mesh overlay from CBR500R scan (cosmetic)
- **v1.0**: PPO lean controller trained via brax/MJX, documented sim-to-sim gap
- **v2.0** (year 2): second vehicle (MT-07 URDF) to prove vehicle-agnostic architecture

Public-road actuation: never, under this license.

## License

MIT. See LICENSE.

## References

- Meijaard, Papadopoulos, Ruina, Schwab. "Linearized dynamics equations for the balance and steer of a bicycle: a benchmark and review." Proc. Roy. Soc. A, 2007.
- Cossalter, Vittore. *Motorcycle Dynamics.* 2nd ed. 2006.
- Honda CBR500R/CBR500R ABS 2013–2015 Service Manual.
- MuJoCo documentation: https://mujoco.readthedocs.io/

## Why "motosim" and not "motopilot"

Naming constrains ambition. "motopilot" overclaims. "motosim" describes exactly
what this is, exactly what we can validate, exactly where we are in the arc.
When it does more than sim, we rename.
