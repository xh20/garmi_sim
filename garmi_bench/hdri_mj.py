"""The scene's HDRI room in MuJoCo: an inward-facing textured dome around the scene, plus light estimated from it.

Isaac lights the scene with the HDRI as a dome light (scene.py); MuJoCo cannot light from an image, so:
  * background: a large sphere with the room photo on its inside (equirectangular UVs), emissive and unlit by
    the scene's lights, rotated like Isaac's dome (SceneSpec.hdri_rotation_deg);
  * light: ambient plus one directional key light from the room's brightest
    direction (the windows / lamps), split by that region's share of the light; the foreground's overall brightness
    follows only the dome intensity (Isaac's renders keep a similar exposure across rooms).
Brightness gains are fitted to Isaac renders of the same scenes (scripts/mj_calibrate_lights.py).
"""

from __future__ import annotations

import functools
import os

import cv2
import numpy as np

from garmi_bench.spec import HDRI_DIR

CACHE = os.path.join(os.path.dirname(__file__), "..", "assets", "hdri_mj")
DOME_RADIUS = 20.0
TEX_SIZE = (2048, 1024)  # equirectangular texture (width, height)
# The latlong image's azimuth origin relative to Isaac's dome (deg) and mirror (Isaac shows the dome from inside):
# found by matching the background edges of MuJoCo and Isaac head images of the same 8 scenes over 24 offsets x
# mirror: mirrored, 0 deg (edge correlation 0.469; next best 0.417, median of all candidates 0.377).
AZIMUTH_OFFSET_DEG = 0.0
MIRROR = True


@functools.lru_cache(maxsize=None)
def _load(name: str) -> np.ndarray:
    """The room in units of its median luminance. The .hdr files hold absolute radiance in which a few window /
    lamp pixels are thousands of times brighter than the walls and dominate the mean; exposing for the median puts
    walls and furniture in range and lets the windows saturate, as Isaac's tone-mapped renders show them."""
    hdr = cv2.imread(os.path.join(HDRI_DIR, name), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    hdr = cv2.resize(hdr, TEX_SIZE, interpolation=cv2.INTER_AREA)
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return hdr / float(np.median(lum))


@functools.lru_cache(maxsize=None)
def key_light(name: str) -> tuple[np.ndarray, float, float]:
    """(direction towards the brightest region in the latlong's own frame, its share of the total light, mean)."""
    hdr = _load(name)
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    h, w = lum.shape
    el = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    az = ((np.arange(w) + 0.5) / w) * 2 * np.pi
    weight = np.broadcast_to(np.cos(el)[:, None], lum.shape)  # solid angle per pixel
    energy = lum * weight
    top = energy >= np.percentile(energy, 99.5)
    ee, aa = np.meshgrid(el, az, indexing="ij")
    d = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1)
    v = (d[top] * energy[top][:, None]).sum(0)
    return v / np.linalg.norm(v), float(energy[top].sum() / energy.sum()), float((lum * weight).sum() / weight.sum())


def texture_png(name: str, gain: float) -> str:
    """The room photo, linear values times `gain`, clipped to 8 bit (MuJoCo's output goes through the same tone
    curve as everything else). Cached per (room, gain)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{os.path.splitext(name)[0]}_{gain:.4g}.png")
    if not os.path.exists(path):
        img = np.clip(_load(name) * gain * 255.0, 0, 255).astype(np.uint8)
        if MIRROR:
            img = img[:, ::-1]
        cv2.imwrite(path, img[..., ::-1])
    return path


def dome_obj() -> str:
    """UV sphere with faces wound inwards and equirectangular texture coordinates (u = azimuth, v = elevation)."""
    path = os.path.join(CACHE, "dome.obj")
    if os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    n_az, n_el = 96, 48
    lines = []
    for i in range(n_el + 1):
        el = np.pi / 2 - np.pi * i / n_el
        for j in range(n_az + 1):
            az = 2 * np.pi * j / n_az
            lines.append(f"v {DOME_RADIUS * np.cos(el) * np.cos(az):.4f} {DOME_RADIUS * np.cos(el) * np.sin(az):.4f} {DOME_RADIUS * np.sin(el):.4f}")
    for i in range(n_el + 1):
        for j in range(n_az + 1):
            lines.append(f"vt {j / n_az:.5f} {1 - i / n_el:.5f}")
    for i in range(n_el):
        for j in range(n_az):
            a, b = i * (n_az + 1) + j + 1, i * (n_az + 1) + j + 2
            c, d = a + n_az + 1, b + n_az + 1
            lines.append(f"f {a}/{a} {b}/{b} {d}/{d}")  # wound so the normal points to the centre
            lines.append(f"f {a}/{a} {d}/{d} {c}/{c}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
