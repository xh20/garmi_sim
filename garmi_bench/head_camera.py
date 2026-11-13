"""GARMI's head camera as v16 saw it, from the calibration fitted to the v16 recordings.

scripts/calibrate_head_camera.py fits, for v16's 256x256 left-eye image: head tilt / pan, image roll, and an
equidistant fisheye with a cubic term r = f (theta + k2 theta^3) around (cx, cy). The arms only cover angles up to
~65 deg off the axis, where the cubic term is reliable; beyond THETA_FIT the lens curve is continued with its slope
there up to the measured image circle (CIRCLE_PX, black outside, as on the real camera). Simulator-independent:
both scenes render at the camera's native resolution (RENDER_SIZE) and downsample to 256 like v16 (downsample()).
"""

from __future__ import annotations

import functools
import io
import json
import os

import numpy as np
from PIL import Image

CALIBRATION = os.path.join(os.path.dirname(__file__), "..", "assets", "head_camera_calibration.json")
OUT_SIZE = 256  # v16's head image
RENDER_SIZE = 1000  # one eye of the Olive's 1000x2000 stereo frame
# The wrist camera model (render size, intrinsics) lives in camera_models.wrist_render() / wrist_view().
WRIST_OUT = (256, 192)
THETA_FIT = 1.15  # rad: arms seen up to here during the calibration


@functools.lru_cache(maxsize=None)
def calibration() -> dict:
    with open(CALIBRATION) as f:
        return json.load(f)


def _r_cubic(theta, c):
    return c["f"] * (theta + c["k2"] * theta**3)


@functools.lru_cache(maxsize=None)
def lens_table() -> tuple[np.ndarray, np.ndarray]:
    """(theta [rad], r [px at OUT_SIZE]) of the lens, monotonic from the axis to the image circle."""
    c = calibration()
    theta = np.linspace(0.0, THETA_FIT, 400)
    r = _r_cubic(theta, c)
    slope = c["f"] * (1 + 3 * c["k2"] * THETA_FIT**2)
    theta_edge = THETA_FIT + (c["image_circle_px"] - r[-1]) / slope
    t2 = np.linspace(THETA_FIT, theta_edge, 200)[1:]
    return np.concatenate([theta, t2]), np.concatenate([r, r[-1] + slope * (t2 - THETA_FIT)])


def fov_deg() -> float:
    return float(np.degrees(2 * lens_table()[0][-1]))


def r_of_theta(theta: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    t, r = lens_table()
    return np.interp(theta, t, r) * size / OUT_SIZE


def theta_of_r(r: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    t, rr = lens_table()
    return np.interp(np.asarray(r) * OUT_SIZE / size, rr, t, right=np.nan)


def optical_center(size: int = OUT_SIZE) -> tuple[float, float]:
    c = calibration()
    return c["cx"] * size / OUT_SIZE, c["cy"] * size / OUT_SIZE


def head_joints() -> dict[str, float]:
    c = calibration()
    return {"o1_motor_2": c["tilt"], "o1_motor_1": c["pan"]}


def roll() -> float:
    return calibration()["roll"]


def isaac_ftheta(size: int = RENDER_SIZE) -> dict:
    """Isaac Sim f-theta parameters: theta = k0 + k1 r + ... + k4 r^4 (r in pixels at `size`)."""
    t, r = lens_table()
    r = r * size / OUT_SIZE
    # Least squares through the origin (theta = 0 at the optical centre): theta = k1 r + k2 r^2 + k3 r^3 + k4 r^4.
    a = np.stack([r, r**2, r**3, r**4], axis=1)
    k = np.linalg.lstsq(a / r.max() ** np.arange(1, 5), t, rcond=None)[0] / r.max() ** np.arange(1, 5)
    return {"nominal_width": size, "nominal_height": size, "optical_center": optical_center(size),
            "max_fov": fov_deg(), "distortion_coefficients": [0.0] + [float(x) for x in k]}


def pixel_rays(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit ray per pixel in the optical frame (x right, y down, z forward) and a mask of the image circle."""
    cx, cy = optical_center(size)
    v, u = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    dx, dy = u - cx, v - cy
    theta = theta_of_r(np.hypot(dx, dy), size)
    valid = ~np.isnan(theta)
    theta = np.nan_to_num(theta)
    phi = np.arctan2(dy, dx) - roll()  # the image is rotated by +roll relative to the lens frame
    ray = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    return ray, valid


def circle_mask(size: int) -> np.ndarray:
    return pixel_rays(size)[1]


def downsample(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(width, height) with PIL's bilinear filter, which averages when shrinking (openpi's resize)."""
    return np.asarray(Image.fromarray(img[..., :3]).resize(size, Image.BILINEAR))


# The robot's cameras publish JPEG (Olive head: "jpeg" 1000x2000 stereo; RealSense wrists: "rgb8; jpeg compressed",
# garmi_vla_client.py), so every v16 image was JPEG-compressed at full resolution before it was cropped and
# downsampled. The quality is not recorded; 80 is ROS image_transport's default.
JPEG_QUALITY = 80


def camera_frame(img: np.ndarray, size: tuple[int, int], jpeg_quality: int = JPEG_QUALITY) -> np.ndarray:
    """v16's image path for a native-resolution render: the camera's JPEG, then downsampling to `size`."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[..., :3])).save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return downsample(np.asarray(Image.open(buf).convert("RGB")), size)
