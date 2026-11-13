"""Scene description and seeded sampling for the GARMI "lift the bottle" task.

Pure Python (no simulator imports), so the same scenes can be rebuilt in Isaac Sim or MuJoCo. All
positions are in GARMI's `base_link` frame (x forward, y left, z up), in meters. Nominal values and
ranges come from the real v16 recordings (fabiomanz/garmi_grasp_bottle_v16): fingertips at grasp time
were at x 0.98-1.34 / y -0.40..0.74 / z ~0.95 (lift at 0), the arm on the bottle's side grasps it.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os

import numpy as np

# Arm joint start pose: median first frame of the 203 v16 episodes (p90-p10 spread < 0.02 rad),
# identical to the real controller's start_joint_configuration. Grippers open (0.035 m per finger).
START_JOINTS = {
    "left": (0.1954, -1.0258, -0.9474, -1.6800, -0.3601, 2.8894, 0.6811),
    "right": (-0.1931, -1.0249, 0.9547, -1.6830, 0.3722, 2.8960, -0.6690),
}
START_FINGER = 0.035

LIFT_SUCCESS_M = 0.20  # success: bottle bottom at least this far above the table surface

# Bottles: RoboTwin-OD objects (MIT), converted by scripts/build_bottle_assets.py. The Franka hand opens to 80 mm
# and may approach a bottle from any side, so what has to fit is the bottle's widest horizontal diameter in any
# direction (square bottles are widest across the diagonal), not its x/y bounding box (`width` in the catalog).
# That diameter comes from scripts/survey_bottles.py (assets/bottle_survey.json, original size). Bottles are scaled
# down so that it is at most MAX_BOTTLE_DIAMETER (>= 7.5 mm clearance per finger); models that would need shrinking
# below MIN_BOTTLE_SCALE are not used.
CATALOG = os.path.join(os.path.dirname(__file__), "..", "assets", "bottles", "catalog.json")
SURVEY = os.path.join(os.path.dirname(__file__), "..", "assets", "bottle_survey.json")
MAX_BOTTLE_DIAMETER = 0.065
MIN_BOTTLE_SCALE = 0.65
# The v16 demos lift drink bottles with screw caps (PET water bottles), grasped at the cap: only the drink-bottle
# categories are used (no oil / soy-sauce / vinegar bottles).
BOTTLE_CATEGORIES = ("001_bottle", "114_bottle")
MAX_CAP_DIAMETER = 0.038  # top 15 mm of the bottle (survey_bottles.py `cap_diameter_mm`), after scaling
MAX_SCALE_JITTER = 1.05  # upper end of SceneRanges.bottle_scale


def bottle_assets(split: str) -> list[dict]:
    """'train': seen bottles; 'unseen': every 4th usable bottle (catalog order), held out of all training.

    Each asset dict gets `diameter` (m, widest horizontal, original size). Models that fall over by themselves on
    a table (stability.json, scripts/check_bottle_stability.py: tilt > 10 deg after 2 s; 3 models, whose
    collision meshes have no flat base) are not used.
    """
    with open(SURVEY) as f:
        survey = {b["asset"] + ".usda": b for b in json.load(f)["bottles"]}
    with open(CATALOG) as f:
        assets = [{**a, "diameter": survey[a["asset"]]["max_diameter_mm"] / 1000.0,
                   "cap_diameter": survey[a["asset"]]["cap_diameter_mm"] / 1000.0} for a in json.load(f)["assets"]]
    assets = [a for a in assets if a["category"] in BOTTLE_CATEGORIES]
    assets = [a for a in assets if MAX_BOTTLE_DIAMETER / a["diameter"] >= MIN_BOTTLE_SCALE]
    # A cap the fingers can close on like on v16's PET-bottle caps (~31 mm), at the largest scale the bottle gets.
    assets = [a for a in assets if a["cap_diameter"] * min(MAX_SCALE_JITTER, MAX_BOTTLE_DIAMETER / a["diameter"]) <= MAX_CAP_DIAMETER]
    stability = os.path.join(os.path.dirname(CATALOG), "stability.json")
    if os.path.exists(stability):
        with open(stability) as f:
            tilt = {k: v["tilt_deg"] for k, v in json.load(f).items()}
        assets = [a for a in assets if tilt.get(a["asset"], 0.0) <= 10.0]
    unseen = {a["asset"] for k, a in enumerate(assets) if k % 4 == 3}
    if split == "unseen":
        return [a for a in assets if a["asset"] in unseen]
    return [a for a in assets if a["asset"] not in unseen]


# Where each arm can do the task (scripts/reach_map.py, scripts/grasp_map.py). A bottle position is used only if
# the scripted expert succeeded there in every grasp-map trial (nominal 0.85 m table) and the arm reaches it
# kinematically at the scene's table height (nearest mapped height).
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
GRASP_MIN_SUCCESS = 1.0

# Which arm grasps follows the bottle's side, as in the real v16 demos: there the left arm grasped exactly when the
# fingertip was at y > +0.10 m at the grasp (203 of 203 episodes; left +0.11..+0.74, right -0.40..+0.10). The arm is
# still drawn 50/50 for balance, and the bottle is placed on that arm's side of ARM_SIDE_Y, so the arm can be told
# from the scene (with a random arm wherever both reach, half of the v1 demos had an arm the images do not show).
ARM_SIDE_Y = 0.10

# The v16 grasps (scripts/extract_v16_grasps.py): every scene copies one real episode of its arm -- bottle where the
# real cap was grasped (+ GRASP_JITTER_M), cap at the real grasp height -- and the expert copies that episode's
# approach angles, lift height and timing. The cap is grasped CAP_BELOW_TOP below the bottle's top.
V16_GRASPS = os.path.join(ASSETS, "v16_grasps.json")
GRASP_JITTER_M = 0.02
CAP_BELOW_TOP = 0.005  # fingertip centre on the cap (deeper, the tilted 70 mm fingertips reach the widening neck)
TABLE_HEIGHT_LIMITS = (0.70, 0.95)  # a table whose height follows from cap height and bottle size must stay plausible
START_JITTER_RAD = 0.01  # v16's start pose varies by ~0.02 rad (p90 - p10)
# Only real episodes that lift the bottle at least this far are copied (the task's success is 20 cm; 46 of the 203
# v16 demos lift less than 25 cm).
MIN_REF_LIFT_M = 0.22


@functools.lru_cache(maxsize=None)
def v16_grasps() -> tuple[dict, ...]:
    with open(V16_GRASPS) as f:
        return tuple(json.load(f)["grasps"])


def v16_grasp(episode: int) -> dict:
    return next(g for g in v16_grasps() if g["episode"] == episode)


HDRI_DIR = os.path.join(ASSETS, "hdri")


@functools.lru_cache(maxsize=None)
def hdri_files(split: str) -> tuple[str, ...]:
    """Room HDRIs for 'train' or 'unseen' (assets/hdri/hdri.json); empty if not downloaded."""
    path = os.path.join(HDRI_DIR, "hdri.json")
    if not os.path.exists(path):
        return ()
    with open(path) as f:
        return tuple(json.load(f)[split])


@functools.lru_cache(maxsize=None)
def _maps():
    with open(os.path.join(ASSETS, "reach_map.json")) as f:
        reach = json.load(f)
    path = os.path.join(ASSETS, "grasp_map.json")
    cells = {}
    if os.path.exists(path):
        with open(path) as f:
            grasp = json.load(f)
        step = float(grasp["step"])
        for side in ("left", "right"):
            g = grasp["maps"][side]
            cells[side] = np.array([(x, y) for i, x in enumerate(grasp["xs"]) for j, y in enumerate(grasp["ys"])
                                    if g[i][j] is not None and g[i][j] >= GRASP_MIN_SUCCESS])
    else:  # before scripts/grasp_map.py has run (it samples scenes itself): kinematic reach only
        step = float(reach["xs"][1] - reach["xs"][0])
        for side in ("left", "right"):
            g = reach["maps"][f"{side}@0.85"]
            cells[side] = np.array([(x, y) for i, x in enumerate(reach["xs"]) for j, y in enumerate(reach["ys"]) if g[i][j]])
    return cells, step, reach


def _kinematically_reachable(arm: str, x: float, y: float, h: float) -> bool:
    _, _, reach = _maps()
    hh = min(reach["heights"], key=lambda v: abs(v - h))
    xs, ys = np.array(reach["xs"]), np.array(reach["ys"])
    return bool(reach["maps"][f"{arm}@{hh}"][int(np.abs(xs - x).argmin())][int(np.abs(ys - y).argmin())])


def workspace_point(arm: str, h: float, x_lim: tuple, y_lim: tuple, rng: np.random.Generator) -> tuple | None:
    """A random bottle position on the table (x_lim, y_lim) inside the arm's workspace, or None."""
    cells, step, _ = _maps()
    c = cells[arm]
    inside = c[(c[:, 0] >= x_lim[0]) & (c[:, 0] <= x_lim[1]) & (c[:, 1] >= y_lim[0]) & (c[:, 1] <= y_lim[1])]
    for _ in range(20):
        if not len(inside):
            return None
        x, y = inside[int(rng.integers(len(inside)))] + rng.uniform(-step / 2, step / 2, 2)
        x, y = float(np.clip(x, *x_lim)), float(np.clip(y, *y_lim))
        if _kinematically_reachable(arm, x, y, h):
            return round(x, 4), round(y, 4)
    return None


@dataclasses.dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


@dataclasses.dataclass(frozen=True)
class SceneRanges:
    """Randomization ranges. `TRAIN` is used for demos; `EVAL_NOMINAL` fixes the table."""

    table_height: Range = Range(0.81, 0.89)  # table top z (nominal 0.85: grasp z ~0.95 at mid-bottle)
    table_front_x: Range = Range(0.70, 0.80)  # near edge, distance from base_link
    table_depth: Range = Range(0.60, 0.80)
    table_width: Range = Range(1.30, 1.60)
    table_center_y: Range = Range(0.05, 0.20)  # the reachable area is shifted left (v16 y range)
    # Bottle position: inside the grasping arm's workspace on the table (workspace_point), no fixed range.
    bottle_split: str = "train"  # which bottle assets: "train" or "unseen" (see bottle_assets)
    bottle_scale: Range = Range(0.92, 1.05)  # uniform scale jitter of the asset
    bottle_mass: Range = Range(0.25, 0.60)
    light_intensity: Range = Range(150.0, 450.0)  # dome light: ambient fill
    # Sunlight (distant light, sharp shadows), present with probability sun_probability.
    sun_probability: float = 0.6
    sun_elevation_deg: Range = Range(15.0, 70.0)
    sun_intensity: Range = Range(800.0, 3000.0)
    sun_color_temp: Range = Range(4500.0, 6500.0)
    # Indoor ceiling lights (rect panels facing down, soft shadows).
    ceiling_lights: tuple = (1, 4)  # min, max count
    ceiling_height: Range = Range(2.4, 2.8)
    ceiling_x: Range = Range(-0.5, 2.5)
    ceiling_y: Range = Range(-1.5, 1.8)
    ceiling_size: Range = Range(0.3, 1.0)  # panel edge length (m)
    ceiling_intensity: Range = Range(4000.0, 15000.0)
    ceiling_color_temp: Range = Range(2700.0, 6500.0)
    table_thickness: Range = Range(0.025, 0.05)  # table top slab
    table_leg_width: Range = Range(0.04, 0.07)  # square legs
    table_leg_inset: Range = Range(0.03, 0.08)  # leg distance from the top's edges
    wall_gap: Range = Range(0.40, 1.50)  # back wall distance behind the table's far edge
    side_wall_y: Range = Range(1.4, 2.4)  # side wall distance from GARMI's center line (side is random)
    # Table tops: wood tones (v16's lab table is light orange wood) and common office colors.
    table_colors: tuple = ((0.78, 0.52, 0.27), (0.70, 0.45, 0.24), (0.85, 0.62, 0.38), (0.55, 0.40, 0.28),
                           (0.78, 0.74, 0.68), (0.30, 0.30, 0.32), (0.92, 0.92, 0.90), (0.45, 0.33, 0.22))
    # Walls and floor: hue/saturation/value ranges (light, muted walls; floors from grey to warm parquet tones).
    wall_hsv: tuple = ((0.0, 1.0), (0.0, 0.25), (0.70, 0.97))
    floor_hsv: tuple = ((0.0, 0.15), (0.15, 0.65), (0.30, 0.75))
    edge_margin: float = 0.06  # bottle center at least this far inside the table edges
    # Surroundings: a real indoor 360-degree photo (Poly Haven, CC0; assets/hdri/hdri.json) as the dome light,
    # rotated at random. It is the background every camera sees (rooms, windows, furniture, like v16's lab) and the
    # main light. The flat back / side walls are only used with no HDRI.
    hdri_split: str = "train"  # "train" or "unseen" (6 rooms held out of all training)
    hdri_intensity: Range = Range(550.0, 1250.0)  # head image brightness ~ v16's (wrists slightly brighter)
    walls: bool = False
    # The "v16 lab" twin (v16_lab.py, MuJoCo only): the real lab panorama, the real table, clear PET bottles.
    lab_twin: bool = False


TRAIN = SceneRanges()
EVAL_NOMINAL = dataclasses.replace(
    TRAIN,
    table_height=Range(0.85, 0.85),
    table_front_x=Range(0.75, 0.75),
    table_depth=Range(0.70, 0.70),
    table_width=Range(1.40, 1.40),
    table_center_y=Range(0.15, 0.15),
    table_thickness=Range(0.04, 0.04),
    table_leg_width=Range(0.05, 0.05),
    table_leg_inset=Range(0.05, 0.05),
    wall_gap=Range(0.80, 0.80),
    side_wall_y=Range(1.8, 1.8),
    bottle_scale=Range(1.0, 1.0),
)
# Generalization splits: nominal table, bottles never seen in training / rooms (HDRIs) never seen in training.
EVAL_UNSEEN_OBJECTS = dataclasses.replace(EVAL_NOMINAL, bottle_split="unseen")
EVAL_UNSEEN_ROOMS = dataclasses.replace(EVAL_NOMINAL, hdri_split="unseen")
# The real lab of the v16 recordings, rebuilt from them (MuJoCo only; see v16_lab.py).
EVAL_V16_LAB = dataclasses.replace(EVAL_NOMINAL, lab_twin=True)


@dataclasses.dataclass(frozen=True)
class SceneSpec:
    seed: int
    table_height: float
    table_front_x: float
    table_depth: float
    table_width: float
    table_center_y: float
    table_color: tuple
    bottle_x: float
    bottle_y: float
    bottle_yaw: float
    bottle_asset: str  # path relative to assets/bottles
    bottle_scale: float
    bottle_height: float  # after scaling (m)
    bottle_width: float  # after scaling (m)
    bottle_mass: float
    light_intensity: float
    table_thickness: float
    table_leg_width: float
    table_leg_inset: float
    wall_gap: float
    side_wall_y: float  # signed: + left of GARMI, - right
    wall_color: tuple
    floor_color: tuple
    # Lighting. Sun: None, or (azimuth_deg, elevation_deg, intensity, color_temp_K), coming from that direction.
    sun: tuple | None
    # Ceiling panels: ((x, y, z, edge_m, intensity, color_temp_K), ...), all facing down.
    ceiling: tuple
    arm: str  # "left" or "right": the arm that grasps (drawn 50/50; the bottle is placed in its workspace)
    ref_episode: int = -1  # the v16 episode this scene copies (bottle position, cap height, expert's grasp)
    cap_z: float = 0.0  # height of the grasp point at the cap (m)
    start_offsets: tuple = ()  # 14 arm-joint offsets of the start pose from START_JOINTS (rad)
    hdri: str = ""  # assets/hdri/<file> as the dome light ("": plain dome, back and side walls)
    hdri_rotation_deg: float = 0.0  # about the vertical axis
    hdri_intensity: float = 0.0
    walls: bool = True
    cap_radius: float = 0.0  # PET bottles of the v16 lab twin only (m); 0 -> v16_lab.CAP_RADIUS

    @property
    def table_top_center(self) -> tuple[float, float, float]:
        """Center of the table top slab (its upper surface at table_height)."""
        return (self.table_front_x + self.table_depth / 2, self.table_center_y, self.table_height - self.table_thickness / 2)

    @property
    def table_leg_xy(self) -> list[tuple[float, float]]:
        """Centers of the four legs."""
        d = self.table_leg_inset + self.table_leg_width / 2
        xs = (self.table_front_x + d, self.table_front_x + self.table_depth - d)
        ys = (self.table_center_y - self.table_width / 2 + d, self.table_center_y + self.table_width / 2 - d)
        return [(x, y) for x in xs for y in ys]

    @property
    def back_wall_x(self) -> float:
        return self.table_front_x + self.table_depth + self.wall_gap

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


def _sample_lab_scene(seed: int, ranges: SceneRanges) -> SceneSpec:
    """A scene in the v16 lab twin: the real table and room, a clear PET bottle with a random cap color, placed where
    a real v16 episode of the drawn arm grasped the cap (+ GRASP_JITTER_M, on the table, on the arm's side)."""
    from garmi_bench import v16_lab

    rng = np.random.default_rng(seed)
    rng_obj = np.random.default_rng([seed, 2])
    lab = v16_lab.table()
    fx, d, w, cy, h = (lab[k] for k in ("table_front_x", "table_depth", "table_width", "table_center_y", "table_height"))
    m = ranges.edge_margin
    arm = "left" if rng.random() < 0.5 else "right"
    pool = [g for g in v16_grasps() if g["arm"] == arm and g["lift_m"] >= MIN_REF_LIFT_M]
    for _ in range(200):
        ref = pool[int(rng.integers(len(pool)))]
        bx, by = ref["x"] + rng.normal(0.0, GRASP_JITTER_M), ref["y"] + rng.normal(0.0, GRASP_JITTER_M)
        by = max(by, ARM_SIDE_Y + 1e-3) if arm == "left" else min(by, ARM_SIDE_Y)
        if fx + m <= bx <= fx + d - m and cy - w / 2 + m <= by <= cy + w / 2 - m:
            break
    else:
        raise RuntimeError(f"seed {seed}: no v16 grasp of the {arm} arm on the lab table")
    # The measured standard 0.5 l bottle (v16_lab.py), full (~0.52 kg); the table height is fitted to it.
    bottle_height, cap_radius = v16_lab.PET_HEIGHT, v16_lab.CAP_RADIUS
    cap = list(v16_lab.CAP_COLORS)[int(rng_obj.integers(len(v16_lab.CAP_COLORS)))]
    rng_start = np.random.default_rng([seed, 4])
    start_offsets = tuple(round(float(v), 5) for v in rng_start.uniform(-START_JITTER_RAD, START_JITTER_RAD, 14))
    return SceneSpec(
        seed=seed, table_height=h, table_front_x=fx, table_depth=d, table_width=w, table_center_y=cy,
        table_color=tuple(lab["table_color_display"]),
        bottle_x=round(float(bx), 4), bottle_y=round(float(by), 4), bottle_yaw=float(rng.uniform(-np.pi, np.pi)),
        bottle_asset=v16_lab.pet_asset(cap), bottle_scale=1.0, bottle_height=bottle_height,
        bottle_width=2 * v16_lab.PET_RADIUS, bottle_mass=0.52, light_intensity=300.0,
        table_thickness=0.03, table_leg_width=0.05, table_leg_inset=0.05, wall_gap=1.0, side_wall_y=2.0,
        wall_color=(0.9, 0.9, 0.9), floor_color=tuple(lab["floor_color_display"]), sun=None, ceiling=(), arm=arm,
        ref_episode=int(ref["episode"]), cap_z=round(float(h + bottle_height - v16_lab.CAP_HEIGHT / 2), 4),
        start_offsets=start_offsets, hdri=v16_lab.ROOM, hdri_rotation_deg=0.0, hdri_intensity=1000.0, walls=False,
        cap_radius=cap_radius,
    )


def _hsv_color(rng: np.random.Generator, hsv: tuple) -> tuple:
    import colorsys

    h, s, v = (float(rng.uniform(lo, hi)) for lo, hi in hsv)
    return tuple(round(c, 4) for c in colorsys.hsv_to_rgb(h, s, v))


def sample_scene(seed: int, ranges: SceneRanges = TRAIN, grasp_jitter_m: float = GRASP_JITTER_M,
                 cap_jitter_m: float = 0.01) -> SceneSpec:
    """Deterministic: the same seed, ranges and asset catalog always give the same scene.

    Table and bottle placement, room and bottle asset are drawn from separate generators, so changing one
    group of ranges does not reshuffle the others. grasp_jitter_m / cap_jitter_m: std of the bottle position around
    the copied real grasp and of the cap height (which sets the table height); larger values spread the scenes, and
    so the trajectories, beyond the real demos.
    """
    if ranges.lab_twin:
        return _sample_lab_scene(seed, ranges)
    rng = np.random.default_rng(seed)
    rng_room = np.random.default_rng([seed, 1])
    rng_obj = np.random.default_rng([seed, 2])
    rng_light = np.random.default_rng([seed, 3])
    sun = None
    if rng_light.random() < ranges.sun_probability:
        sun = (round(float(rng_light.uniform(0.0, 360.0)), 2), round(ranges.sun_elevation_deg.sample(rng_light), 2),
               round(ranges.sun_intensity.sample(rng_light), 1), round(ranges.sun_color_temp.sample(rng_light), 0))
    lo, hi = ranges.ceiling_lights
    ceiling = tuple(
        tuple(round(v, 3) for v in (ranges.ceiling_x.sample(rng_light), ranges.ceiling_y.sample(rng_light),
                                    ranges.ceiling_height.sample(rng_light), ranges.ceiling_size.sample(rng_light),
                                    ranges.ceiling_intensity.sample(rng_light), ranges.ceiling_color_temp.sample(rng_light)))
        for _ in range(int(rng_light.integers(lo, hi + 1)))
    )
    assets = bottle_assets(ranges.bottle_split)
    asset = assets[int(rng_obj.integers(len(assets)))]
    scale = ranges.bottle_scale.sample(rng_obj)
    scale = min(scale, MAX_BOTTLE_DIAMETER / asset["diameter"])  # never scale a bottle past the gripper limit
    arm = "left" if rng.random() < 0.5 else "right"
    bottle_height = asset["height"] * scale
    m = ranges.edge_margin
    pool = [g for g in v16_grasps() if g["arm"] == arm and g["lift_m"] >= MIN_REF_LIFT_M]
    fixed_table = ranges.table_height.lo == ranges.table_height.hi
    for _ in range(100):  # copy a real grasp; redraw if the table it implies is implausible
        ref = pool[int(rng.integers(len(pool)))]
        bx, by = ref["x"] + rng.normal(0.0, grasp_jitter_m), ref["y"] + rng.normal(0.0, grasp_jitter_m)
        by = max(by, ARM_SIDE_Y + 1e-3) if arm == "left" else min(by, ARM_SIDE_Y)
        if fixed_table:  # evaluation suites with a fixed table: the cap height follows from it
            h = ranges.table_height.lo
        else:
            h = ref["z"] + rng.normal(0.0, cap_jitter_m) + CAP_BELOW_TOP - bottle_height
        if not TABLE_HEIGHT_LIMITS[0] <= h <= TABLE_HEIGHT_LIMITS[1]:
            continue
        fx, d, w, cy = (r.sample(rng) for r in (ranges.table_front_x, ranges.table_depth, ranges.table_width, ranges.table_center_y))
        fx = min(fx, bx - m - 0.05)  # the bottle stands on the table, at least the edge margin inside
        d = max(d, bx + m - fx)
        cy = float(np.clip(cy, by - w / 2 + m, by + w / 2 - m))
        break
    else:
        raise RuntimeError(f"seed {seed}: no plausible table for the {arm} arm's v16 grasps")
    rng_bg = np.random.default_rng([seed, 5])
    rooms = hdri_files(ranges.hdri_split)
    hdri = rooms[int(rng_bg.integers(len(rooms)))] if rooms else ""
    hdri_rot = round(float(rng_bg.uniform(0.0, 360.0)), 2)
    hdri_int = round(ranges.hdri_intensity.sample(rng_bg), 1)
    rng_start = np.random.default_rng([seed, 4])
    start_offsets = tuple(round(float(v), 5) for v in rng_start.uniform(-START_JITTER_RAD, START_JITTER_RAD, 14))
    return SceneSpec(
        seed=seed,
        table_height=h, table_front_x=fx, table_depth=d, table_width=w, table_center_y=cy,
        table_color=ranges.table_colors[rng.integers(len(ranges.table_colors))],
        bottle_x=round(float(bx), 4),
        bottle_y=round(float(by), 4),
        bottle_yaw=float(rng.uniform(-np.pi, np.pi)),
        bottle_asset=asset["asset"],
        bottle_scale=scale,
        bottle_height=bottle_height,
        bottle_width=asset["diameter"] * scale,
        bottle_mass=ranges.bottle_mass.sample(rng_obj),
        light_intensity=ranges.light_intensity.sample(rng),
        table_thickness=ranges.table_thickness.sample(rng_room),
        table_leg_width=ranges.table_leg_width.sample(rng_room),
        table_leg_inset=ranges.table_leg_inset.sample(rng_room),
        wall_gap=ranges.wall_gap.sample(rng_room),
        side_wall_y=ranges.side_wall_y.sample(rng_room) * (1 if rng_room.random() < 0.5 else -1),
        wall_color=_hsv_color(rng_room, ranges.wall_hsv),
        floor_color=_hsv_color(rng_room, ranges.floor_hsv),
        sun=sun,
        ceiling=ceiling,
        arm=arm,
        ref_episode=int(ref["episode"]),
        cap_z=round(float(h + bottle_height - CAP_BELOW_TOP), 4),
        start_offsets=start_offsets,
        hdri=hdri, hdri_rotation_deg=hdri_rot, hdri_intensity=hdri_int, walls=ranges.walls,
    )
