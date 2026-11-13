# garmi_sim

Isaac Sim (and MuJoCo) benchmark for the GARMI bimanual robot: generate expert demonstrations of a
bottle-grasping task, convert them to LeRobot datasets, and evaluate π₀.₅ policies against the same scenes.

The policies themselves live in [openpi](https://github.com/Physical-Intelligence/openpi); this repository is the
simulation side only.

## What is here

```
garmi_bench/          the simulator
  spec.py             scene sampling: table, room, lighting, bottle asset and placement, all from a seed
  expert.py           the scripted bimanual expert (approach, grasp, lift; optional observer arm)
  scene.py            MuJoCo scene
  isaac_env.py        Isaac Sim environment
  kinematics.py       FK/IK against the GARMI URDF
  camera_models.py    head and wrist camera intrinsics, matched to the real robot's 640x480 wrist cameras
  rtc_sim.py          real-time chunking and the robot client's rate limits, so evaluation matches deployment
scripts/
  record_demos.py     record expert demonstrations
  convert_to_lerobot.py   raw episodes -> a LeRobot dataset (optionally with auxiliary target dimensions)
  eval_policy.py      run a served policy against an evaluation suite
  make_eval_suites.py build fixed evaluation suites
  ...                 asset building, calibration and diagnostics
assets/*.json         scene configuration: reference grasps copied from real demonstrations, evaluation
                      suites, camera calibration, reachability maps
```

## Assets not in this repository

`assets/` here holds only the JSON configuration. The 3D assets it refers to are about 1.7 GB of binary data and
are kept outside git:

| directory | size | what |
|---|---|---|
| `assets/bottles`, `bottles_v1`, `bottles_v2` | ~500 MB | bottle meshes and textures as USD, for Isaac |
| `assets/bottles_mj` | 181 MB | the same bottles for MuJoCo |
| `assets/hdri`, `hdri_mj` | ~1 GB | HDRI environment maps for dome lighting |
| `assets/garmi` | 18 MB | the robot USD, converted from [garmi_description](https://github.com/geriatronics/garmi_description) |
| `assets/v16_lab` | 2.4 MB | the lab-twin scene |

`scripts/build_bottle_assets.py` and `build_bottle_assets_mj.py` rebuild the bottle assets from source meshes;
the HDRIs are downloaded. Copy them from the group's machine rather than rebuilding if you have access.

## The robot model

The GARMI model — URDF, meshes and joint limits — comes from
**[geriatronics/garmi_description](https://github.com/geriatronics/garmi_description)**, which is not vendored
here. Check it out next to this repository; both `kinematics.py` and `robot.py` currently expect it at
`~/sim/garmi_description/garmi_description` (a hard-coded path, not yet configurable).

- `garmi_bench/kinematics.py` reads its URDF directly for forward and inverse kinematics and for joint limits.
- `garmi_bench/robot.py` builds the Isaac USD from it. It does **not** import the URDF unmodified: Isaac Sim 6.2's
  importer and PhysX need four fixes, applied to a copy at import time — `package://` mesh paths resolved to
  absolute ones; `<gazebo>`, `<ros2_control>` and Franka's `<position_based_velocity_limits>` stripped (the
  converter crashes on them); `<mimic>` removed, because PhysX mimic joints drove the articulation to NaN within
  about 30 steps, so the coupled joints — the second finger and the upper lift stage — are commanded together
  instead; and the USD's "Physics" variant set switched to "physx" after import. Joint drives come in with zero
  gains and are set by `set_drive_gains`.

Nothing in `garmi_bench` redefines the robot's geometry, so a change upstream flows through — but if the upstream
URDF changes shape, re-check those four fixes.

## Task and conventions

The task is to grasp a bottle standing on a table and lift it. State is 34-dimensional: 7 left joints,
7 right joints, two finger widths, then 9 values of left and right end-effector pose. Actions are 16-dimensional
(14 joints + 2 fingers), optionally padded with auxiliary gripper-centric targets. Control runs at 15 Hz with an
action chunk of 32.

Scenes are a pure function of their seed, so a recording is reproducible: `record_demos.py` writes a
`generation.json` with the command, library versions and checksums of every input, and every attempted seed with
why it was kept or dropped goes to `record_log.json`.

### Which arm grasps

The bottle is placed on the grasping arm's own side of `ARM_SIDE_Y = 0.10` in the torso frame. With the default
jitter the bottle spreads over y -0.70 .. 0.98, so `--bottle-y LO HI` selects where on the table it stands:
`0.35 1.00` (left), `-0.05 0.35` (middle, which straddles the arm boundary so both arms appear) and
`-0.70 -0.05` (right) take a third of the draws each.

### Episode rejection

Episodes are dropped, not kept and flagged, when the demonstration would teach something the robot cannot do:

- `--max-track-err` (default 0.08 rad): either arm ended up that far from the joint position it was commanded.
  An episode like that records actions the arm did not carry out. Clean episodes stay under 0.059 rad.
- `--observer-max-err`: the observer camera never got pointed at the cap.
- the bottle was not lifted `LIFT_SUCCESS_M` (0.20 m) off the table.

## Running

Isaac Sim is needed for `record_demos.py` and `eval_policy.py`; the MuJoCo path (`scene.py`, `mj_*` scripts) runs
without it.

```bash
# record 30 demonstrations, 10 per bottle band, with the observer arm
python scripts/record_demos.py --episodes 10 --first-seed 0 --max-attempts 40 --max-seconds 70 \
    --split train --observer --bottle-y 0.35 1.00 \
    --aim-error 0.1 --start-real --bow 0.25 --approach-curve 0.07 --grasp-jitter 0.08 --cap-jitter 0.05 \
    --orient-jitter 10 --speed-jitter 0.4 --approach-tilt 15 --standoff 0.09 --recover-prob 0.25 \
    --out demos/raw_left

# convert to a LeRobot dataset (several raw folders may feed one split)
HF_LEROBOT_HOME=demos/lerobot python scripts/convert_to_lerobot.py \
    --raw train=demos/raw_left train=demos/raw_mid train=demos/raw_right \
    --aux16 --repo-id garmi/garmi_sim_lift_bottle

# evaluate a policy served by openpi's scripts/serve_policy.py
python scripts/eval_policy.py --suite assets/eval_suites.json --port 8000
```

`--help` on any script is authoritative; the flags mirror the argparse definitions.
