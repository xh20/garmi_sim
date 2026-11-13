"""Camera models shared by the MuJoCo scene: v16's wrist letterbox and the head camera's fisheye.

Simulator-independent (NumPy + OpenCV). The Isaac Sim scene (scene.py) renders the same models natively.
"""

from __future__ import annotations

import cv2
import numpy as np

# Cube faces of the head fisheye: (forward, image right, image up) in the camera's optical frame
# (ROS convention: x right, y down, z forward). Five faces cover a 190 deg lens; the back face is never hit.
CUBE_FACES = {
    "front": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "right": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "left": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "up": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "down": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
}


# Wrist cameras (D435i color) as on the robot: 640x480 with the camera_info of 2026-11-11 (plumb_bob, zero
# distortion) as fx, fy, cx, cy. The same values are in examples/garmi/vla_client/garmi_vla_client.py.
WRIST_INTRINSICS_640 = {
    "left": (605.3610229492188, 604.5435791015625, 316.593505859375, 250.85704040527344),
    "right": (609.8031005859375, 609.947998046875, 329.8813171386719, 253.70968627929688),
}
# "real": the camera image is the robot's 640x480. Both simulators render a centred pinhole slightly wider than it
#   (WRIST_RENDER at focal length WRIST_RENDER_FX px, ~1:1 with the camera's pixels) and warp it to
#   WRIST_INTRINSICS_640 (off-centre principal point, fx != fy) -- wrist_view().
# "legacy": until 2026-11-11, a centred pinhole with a 54.5 deg horizontal FOV rendered at 960x720 directly
#   (1280x720 at 69 deg cropped to 4:3, from the D435i data sheet). Datasets up to v7obs were recorded with it.
# Either image then goes through v16's JPEG + downsampling to 256x192 + letterbox (head_camera.camera_frame, letterbox).
_WRIST = {"model": "real"}  # set_wrist_model()
WRIST_SIZE = (640, 480)
WRIST_RENDER_FX = 600.0
WRIST_RENDER = (672, 528)  # covers both wrists' views: half-width 336/600 = 0.560 > 0.541, half-height 0.440 > 0.416
WRIST_SIZE_LEGACY = (960, 720)
WRIST_HFOV_LEGACY_DEG = 54.5


def set_wrist_model(model: str) -> None:
    """Select the wrist camera model ("real" | "legacy") before a scene is built."""
    if model not in ("real", "legacy"):
        raise ValueError(f"unknown wrist camera model {model!r}")
    _WRIST["model"] = model


def wrist_size() -> tuple[int, int]:
    """(width, height) of the wrist camera image (before JPEG + downsampling)."""
    return WRIST_SIZE_LEGACY if _WRIST["model"] == "legacy" else WRIST_SIZE


def wrist_render() -> tuple[tuple[int, int], float]:
    """(render (width, height), horizontal FOV in degrees) of the centred pinhole each simulator renders."""
    if _WRIST["model"] == "legacy":
        return WRIST_SIZE_LEGACY, WRIST_HFOV_LEGACY_DEG
    w, _ = WRIST_RENDER
    return WRIST_RENDER, float(2 * np.degrees(np.arctan(w / 2 / WRIST_RENDER_FX)))


def wrist_intrinsics(side: str) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy of a wrist camera image of wrist_size() (pixel centres at integer coordinates)."""
    if _WRIST["model"] == "legacy":
        w, h = WRIST_SIZE_LEGACY
        f = w / 2 / np.tan(np.radians(WRIST_HFOV_LEGACY_DEG) / 2)
        return f, f, (w - 1) / 2, (h - 1) / 2
    return WRIST_INTRINSICS_640[side]


def wrist_view(render: np.ndarray, side: str) -> np.ndarray:
    """The wrist camera image from a wrist_render() render (the same camera centre and orientation)."""
    if _WRIST["model"] == "legacy":
        return render
    fx, fy, cx, cy = wrist_intrinsics(side)
    h, w = render.shape[:2]
    f = WRIST_RENDER_FX
    # output pixel (u, v) -> ray ((u - cx) / fx, (v - cy) / fy) -> render pixel (f * x + (w - 1) / 2, f * y + (h - 1) / 2)
    m = np.array([[f / fx, 0.0, (w - 1) / 2 - f * cx / fx], [0.0, f / fy, (h - 1) / 2 - f * cy / fy]])
    return cv2.warpAffine(np.ascontiguousarray(render[..., :3]), m, WRIST_SIZE,
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def letterbox(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Pad a (size*3/4, size) image to (size, size) with black bars, like v16's wrist images (= scene.letterbox)."""
    h = img.shape[0]
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - h) // 2
    out[top:top + h] = img[..., :3]
    return out


class FthetaFromCube:
    """Equidistant (f-theta) fisheye from five 90 deg cube-face renders, as Isaac's head camera:
    `set_ftheta_properties(nominal_width=w, optical_center=(w/2, h/2), max_fov=190, k1=half_fov / (w/2))`,
    i.e. angle from the optical axis = k1 * pixel radius; the image circle spans the width, outside it is black.
    """

    # face_size 256: ~2.8 px/deg on the faces vs 1.35 px/deg at the fisheye's center, so bilinear sampling
    # stays close to one face pixel per output pixel (little aliasing) without rendering more than needed.
    def __init__(self, size: int = 256, face_size: int = 256, max_fov_deg: float = 190.0, rays: np.ndarray | None = None,
                 valid: np.ndarray | None = None):
        """rays / valid: a calibrated lens (per-pixel unit rays in the optical frame and the image circle, e.g.
        head_camera.pixel_rays); otherwise the equidistant model above."""
        self.size, self.face_size = size, face_size
        if rays is None:
            half_fov = np.radians(max_fov_deg) / 2
            k1 = half_fov / (size / 2)
            v, u = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
            dx, dy = u - size / 2, v - size / 2
            theta, phi = k1 * np.hypot(dx, dy), np.arctan2(dy, dx)
            rays = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
            valid = theta <= half_fov
        ray = rays
        self.valid = valid
        names = list(CUBE_FACES)
        fwd = np.array([CUBE_FACES[n][0] for n in names], float)
        face = np.argmax(ray @ fwd.T, axis=-1)  # the face whose axis is closest to the ray
        map_x, map_y = np.zeros((size, size), np.float32), np.zeros((size, size), np.float32)
        for k, name in enumerate(names):
            f, r, up = (np.array(a, float) for a in CUBE_FACES[name])
            sel = face == k
            depth = ray[sel] @ f
            a, b = (ray[sel] @ r) / depth, (ray[sel] @ up) / depth  # in [-1, 1] on a 90 deg face
            map_x[sel] = k * face_size + (a + 1) / 2 * face_size - 0.5
            map_y[sel] = (1 - b) / 2 * face_size - 0.5
        self.map_x, self.map_y, self.face_names = map_x, map_y, names

    def remap(self, faces: dict[str, np.ndarray], outside: int = 0) -> np.ndarray:
        """faces: name -> (face_size, face_size, 3) uint8 render of that cube face (image up = the face's up).
        outside: gray level outside the image circle (Isaac's RTX fills it with the dome light's color)."""
        atlas = np.concatenate([faces[n] for n in self.face_names], axis=1)
        out = cv2.remap(atlas, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[~self.valid] = outside
        return out


def tone_lut(k: float) -> np.ndarray:
    """uint8 LUT of the tone curve (1 - exp(-k x)) / (1 - exp(-k)): lifts mid-tones and rolls off highlights
    like Isaac's tone mapping does (its renders never clip), applied to MuJoCo's linear-clipped output."""
    x = np.arange(256) / 255.0
    y = x if k < 1e-6 else (1 - np.exp(-k * x)) / (1 - np.exp(-k))
    return np.clip(np.round(255 * y), 0, 255).astype(np.uint8)


def tone_inverse(y: float, k: float) -> float:
    """x with tone curve(x) = y (both in [0, 1])."""
    return y if k < 1e-6 else float(-np.log(1 - y * (1 - np.exp(-k))) / k)
