"""The "v16 lab" twin: the real lab where the v16 demonstrations were recorded, rebuilt in MuJoCo for testing.

From the v16 recordings themselves (scripts/build_v16_lab.py, assets/v16_lab/): a panorama of the lab seen by the
calibrated head camera (arms masked out, median over 2000 frames) as the background dome, and the real table top
fitted to the edges in the head images. The bottles are standard 0.5 l PET bottles as measured on the real ones (cap
28 mm x 17 mm, total height 21.5-23.5 cm, body 64-68 mm) with the short neck, support ring and rounded shoulder seen
in the v16 wrist images (a lathe-turned profile: transparent body, opaque cap). The head images alone cannot fix the
table height (their edges fit 0.70-0.86 m about equally well); it follows from the bottle and the real cap grasps
(fingers closed to <= 14.5 mm, 82 of 203 episodes), whose fingertip height is 0.951 m (10-90 %: 0.936-0.968 m).
MuJoCo only.
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np

LAB_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "v16_lab")
ROOM = "v16_lab"  # SceneSpec.hdri value of the lab panorama
PET_PREFIX = "pet_bottle_"  # SceneSpec.bottle_asset: pet_bottle_<cap color>
# Cap colors seen in the v16 wrist images (water, "Lift", Coca-Cola Zero).
CAP_COLORS = {"white": (0.92, 0.92, 0.90), "green": (0.30, 0.45, 0.15), "black": (0.06, 0.06, 0.06)}
# Standard 0.5 l PET bottle (m), z from the bottom.
PET_HEIGHT = 0.225  # with the cap (21.5-23.5 cm)
PET_RADIUS = 0.033  # body (64-68 mm)
CAP_RADIUS = 0.014  # 28 mm
CAP_HEIGHT = 0.017
NECK_RADIUS = 0.0125  # 25 mm thread / neck (28 mm closure)
RING_RADIUS = 0.0165  # support ring, 33 mm
RING_Z = (0.201, 0.203)
SHOULDER_Z = (0.150, 0.193)  # rounded dome from the body to the neck
TABLE_HEIGHT = round(0.951 - (PET_HEIGHT - CAP_HEIGHT / 2), 3)  # the real cap grasps' fingertip height: 0.735 m


@functools.lru_cache(maxsize=None)
def table() -> dict:
    with open(os.path.join(LAB_DIR, "table.json")) as f:
        return json.load(f)


def panorama_path() -> str:
    return os.path.join(LAB_DIR, "panorama.png")


def _shoulder() -> list[tuple[float, float]]:
    """Rounded dome (quarter ellipse) from the body radius to the neck: convex, so one collision part."""
    z0, z1 = SHOULDER_Z
    return [(float(z0 + (z1 - z0) * np.sin(a)), float(NECK_RADIUS + (PET_RADIUS - NECK_RADIUS) * np.cos(a)))
            for a in np.linspace(0.0, np.pi / 2, 12)]


def _profile(height: float) -> list[tuple[float, float]]:
    """(z, radius) of the bottle below the cap: rounded base, ribbed body, shoulder, neck with support ring."""
    h_body = height - CAP_HEIGHT
    pts = [(0.0, 0.0), (0.0, PET_RADIUS - 0.006), (0.004, PET_RADIUS)]
    for z in np.linspace(0.008, SHOULDER_Z[0], 14)[:-1]:  # two shallow ribs on the body
        rib = 0.0012 * (np.cos(2 * np.pi * z / 0.035) < -0.7)
        pts.append((float(z), PET_RADIUS - rib))
    pts += _shoulder()
    pts += [(RING_Z[0], NECK_RADIUS), (RING_Z[0], RING_RADIUS), (RING_Z[1], RING_RADIUS), (RING_Z[1], NECK_RADIUS)]
    pts += [(h_body, NECK_RADIUS), (h_body, 0.0)]
    return pts


def collision_parts(height: float) -> list[dict]:
    """Convex parts of the bottle below the cap: MuJoCo collides a mesh as its convex hull, and the hull of the whole
    body would be a cone from the body to the cap (no neck: the fingers would close on its slope). Each part:
    cylinder (radius, z0, z1) or mesh (path)."""
    h_body = height - CAP_HEIGHT
    shoulder = [(SHOULDER_Z[0], 0.0)] + _shoulder() + [(SHOULDER_Z[1], 0.0)]
    return [{"cylinder": (PET_RADIUS, 0.0, SHOULDER_Z[0])},
            {"mesh": _lathe(shoulder, os.path.join(LAB_DIR, f"pet_shoulder_{round(1000 * height)}mm.obj"))},
            {"cylinder": (NECK_RADIUS, SHOULDER_Z[1], h_body)},
            {"cylinder": (RING_RADIUS, *RING_Z)}]


def lathe_obj(height: float, n: int = 48) -> str:
    """Body mesh of revolution (closed) of a bottle `height` tall with its cap, origin at the bottom center, z up."""
    return _lathe(_profile(height), os.path.join(LAB_DIR, f"pet_body_{round(1000 * height)}mm.obj"), n)


def _lathe(prof: list[tuple[float, float]], path: str, n: int = 48) -> str:
    if os.path.exists(path):
        return path
    verts, faces = [], []
    for z, r in prof:
        for k in range(n):
            a = 2 * np.pi * k / n
            verts.append((r * np.cos(a), r * np.sin(a), z))
    for i in range(len(prof) - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            c, d = a + n, b + n
            faces += [(a, b, d), (a, d, c)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in verts) + "\n")
        f.write("\n".join(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces) + "\n")
    return path


@functools.lru_cache(maxsize=None)
def key_light() -> np.ndarray:
    """Direction towards the lab's brightest region (windows, ceiling lights) in the panorama, base_link frame."""
    import cv2

    pano = cv2.imread(panorama_path())[..., ::-1].astype(np.float32) / 255.0
    lum = pano @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    h, w = lum.shape
    el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    az = ((np.arange(w) + 0.5) / w) * 2 * np.pi
    weight = np.broadcast_to(np.cos(el)[:, None], lum.shape)
    energy = lum * weight
    top = energy >= np.percentile(energy, 99.0)
    ee, aa = np.meshgrid(el, az, indexing="ij")
    d = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1)
    v = (d[top] * energy[top][:, None]).sum(0)
    return v / np.linalg.norm(v)


def dome_texture(tone_k: float, gain: float) -> str:
    """The panorama holds final camera values; MuJoCo's renders go through camera_models.tone_lut(tone_k), so the
    dome gets the inverse tone curve (times `gain`) to show the lab at its recorded brightness. Cached per values."""
    import cv2

    path = os.path.join(LAB_DIR, f"panorama_dome_k{tone_k:g}_g{gain:.3g}.png")
    if not os.path.exists(path):
        y = cv2.imread(panorama_path()).astype(np.float64) / 255.0
        x = y if tone_k < 1e-6 else -np.log(1 - np.clip(y, 0, 0.999) * (1 - np.exp(-tone_k))) / tone_k
        cv2.imwrite(path, np.clip(x * gain * 255.0, 0, 255).astype(np.uint8))
    return path


# The table as the wrist cameras see it (scripts/mj_calibrate_lab.py): wood pixels of real v16 wrist frames before the
# grasp have median RGB (133, 117, 94) and the twin's plain table rendered (149, 131, 110); the head-image color is
# scaled by the ratio, and the grain texture (mean 1) gives the real frames' +-7 gray-level spread.
WRIST_TABLE_GAIN = (133 / 149, 117 / 131, 94 / 110)
WOOD_TEXTURE_M = 0.5  # one texture tile per 0.5 m


def wood_texture(size: int = 1024) -> str:
    """Grayscale wood grain (fine stripes, wavy, with darker streaks), values 0.85..1.0 x 255; tileable."""
    import cv2

    path = os.path.join(LAB_DIR, f"wood_{size}.png")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(16)
    y, x = np.mgrid[0:size, 0:size] / size  # one tile, periodic in both directions
    warp = sum(a * np.sin(2 * np.pi * (fx * x + fy * y) + rng.uniform(0, 2 * np.pi))
               for a, fx, fy in ((0.05, 1, 0), (0.03, 2, 1), (0.02, 3, -1), (0.01, 5, 2)))
    stripes = np.sin(2 * np.pi * (30 * (y + warp)))  # ~1.7 cm grain spacing at 0.5 m per tile
    streaks = np.sin(2 * np.pi * (7 * (y + 1.5 * warp) + 0.3))
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), 1.5)
    v = 0.925 + 0.03 * stripes - 0.025 * np.maximum(streaks, 0) + 0.012 * fine
    cv2.imwrite(path, np.clip(v / v.max() * 255, 0, 255).astype(np.uint8))
    return path


def pet_asset(cap: str) -> str:
    return f"{PET_PREFIX}{cap}"


def is_pet(asset: str) -> bool:
    return asset.startswith(PET_PREFIX)


def pet_cap(asset: str) -> str:
    return asset[len(PET_PREFIX):]
