"""Build the "v16 lab" twin from the v16 recordings: a panorama of the real lab and the real table's geometry.

    ~/sim/mujoco_venv/bin/python scripts/build_v16_lab.py --episodes 60

The head camera does not move during the v16 recordings and is calibrated (head_camera.py), so every direction
around GARMI maps to a fixed head-image pixel. Per frame the arms are masked out (their meshes projected at the
recorded joint angles, as in calibrate_head_camera.py); the per-pixel median over many frames and episodes is the
lab without arms and bottles:
  * background.png: the median head image (camera view), used to fit the table;
  * panorama.png: the same in an equirectangular panorama around base_link (azimuth from +x, counter-clockwise;
    elevation +90 at the top row), for MuJoCo's dome; directions the head camera never saw are filled in;
  * table.json: the real table top (front edge, depth, width, centre y, height), fitted so that its projected edges
    lie on the strongest edges of background.png, and its color, and the floor color.
Writes assets/v16_lab/.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from calibrate_head_camera import ArmPoints  # noqa: E402
from calibrate_head_camera import HeadCamera  # noqa: E402
from calibrate_head_camera import occupancy  # noqa: E402
from calibrate_head_camera import project  # noqa: E402
from garmi_bench import head_camera  # noqa: E402

V16 = os.path.expanduser("~/garmi_vlapi/data/garmi_grasp_bottle_v16")
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "v16_lab")
SIZE = head_camera.OUT_SIZE
PANO = (1024, 512)  # width, height


def calib() -> dict:
    return {k: head_camera.calibration()[k] for k in ("tilt", "pan", "roll", "f", "k2", "cx", "cy")}


def lens_project(points: np.ndarray, pose) -> tuple[np.ndarray, np.ndarray]:
    """World points -> head-image pixels with the full calibrated lens (incl. the extension beyond the arms)."""
    pos, rot = pose
    c = (points - pos) @ rot
    theta = np.arctan2(np.hypot(c[:, 0], c[:, 1]), c[:, 2])
    phi = np.arctan2(c[:, 1], c[:, 0]) + head_camera.roll()
    r = head_camera.r_of_theta(theta)
    cx, cy = head_camera.optical_center()
    return np.stack([cx + r * np.cos(phi), cy + r * np.sin(phi)], axis=1), theta


def frames_and_masks(episodes, stride, arm, pose):
    p = calib()
    for ep in episodes:
        df = pd.read_parquet(sorted(glob.glob(f"{V16}/data/chunk-*/episode_{ep:06d}.parquet"))[0], columns=["observation.state"])
        states = np.stack(df["observation.state"].to_numpy())
        video = sorted(glob.glob(f"{V16}/videos/chunk-*/observation.images.head/episode_{ep:06d}.mp4"))[0]
        frames = iio.imread(video, plugin="pyav")
        for k in range(0, min(len(states), len(frames)), stride):
            occ = occupancy(*project(arm(states[k]), pose, p), np.pi)
            occ = cv2.dilate(occ, np.ones((9, 9), np.uint8))  # margin around the arms (and the held bottle)
            yield frames[k], occ.astype(bool)


def median_stack(images: list[np.ndarray], valid: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(images).astype(np.float32)
    ok = np.stack(valid)
    stack[~ok] = np.nan
    med = np.nanmedian(stack, axis=0)
    have = ok.any(axis=0)
    return np.nan_to_num(med).astype(np.uint8), have


def fit_table(background: np.ndarray, pose) -> dict:
    """Table top rectangle whose projected top edges lie on the strongest image edges."""
    gray = cv2.GaussianBlur(cv2.cvtColor(background, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 1.0)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    def edges(x):
        fx, d, w, cy, h = x
        corners = np.array([[fx, cy - w / 2], [fx + d, cy - w / 2], [fx + d, cy + w / 2], [fx, cy + w / 2]])
        pts = []
        for a, b in zip(corners, np.roll(corners, -1, axis=0), strict=True):
            t = np.linspace(0.02, 0.98, 60)[:, None]
            pts.append(np.column_stack([a + t * (b - a), np.full(60, h)]))
        return np.concatenate(pts)

    def score(x):
        uv, theta = lens_project(edges(x), pose)
        ok = (uv[:, 0] > 2) & (uv[:, 0] < SIZE - 3) & (uv[:, 1] > 2) & (uv[:, 1] < SIZE - 3)
        if ok.mean() < 0.5:
            return 0.0
        vals = cv2.remap(grad, uv[ok, 0].astype(np.float32)[None], uv[ok, 1].astype(np.float32)[None], cv2.INTER_LINEAR)
        return -float(vals.mean()) * ok.mean()

    best = None
    for fx in np.arange(0.70, 1.01, 0.05):
        for d in np.arange(0.6, 1.21, 0.1):
            for w in np.arange(1.0, 2.01, 0.2):
                for h in np.arange(0.70, 0.81, 0.025):
                    for cy in (-0.1, 0.0, 0.1, 0.2):
                        v = score((fx, d, w, cy, h))
                        if best is None or v < best[0]:
                            best = (v, (fx, d, w, cy, h))
    res = minimize(score, np.array(best[1]), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 800})
    fx, d, w, cy, h = res.x
    return {"table_front_x": round(float(fx), 3), "table_depth": round(float(d), 3), "table_width": round(float(w), 3),
            "table_center_y": round(float(cy), 3), "table_height": round(float(h), 3), "edge_score": round(-float(res.fun), 2)}


def mean_color(img: np.ndarray, mask: np.ndarray) -> list[float]:
    return [round(float(c) / 255.0, 4) for c in np.median(img[mask], axis=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    n_eps = len(glob.glob(f"{V16}/data/chunk-*/episode_*.parquet"))
    eps = np.linspace(0, n_eps - 1, args.episodes).astype(int).tolist()
    arm, cam = ArmPoints(), HeadCamera()
    pose = cam(head_camera.calibration()["tilt"], head_camera.calibration()["pan"])

    imgs, valid = [], []
    for img, occ in frames_and_masks(eps, args.stride, arm, pose):
        imgs.append(img)
        valid.append(~occ)
    background, have = median_stack(imgs, valid)
    cv2.imwrite(os.path.join(OUT, "background.png"), background[..., ::-1])
    print(f"[lab] {len(imgs)} frames from {len(eps)} episodes; pixels never free of the arms: {100 * (~have).mean():.1f} %", flush=True)

    # Panorama: each panorama direction -> head-image pixel (fixed camera), sample the background there.
    w, h = PANO
    az = (np.arange(w) + 0.5) / w * 2 * np.pi
    el = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    ee, aa = np.meshgrid(el, az, indexing="ij")
    dirs = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1).reshape(-1, 3)
    uv, theta = lens_project(pose[0] + 50.0 * dirs, pose)  # far away: direction only
    edge = head_camera.lens_table()[0][-1]
    inside = (theta <= edge) & (uv[:, 0] >= 0) & (uv[:, 0] < SIZE - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE - 1)
    mx, my = uv[:, 0].reshape(h, w).astype(np.float32), uv[:, 1].reshape(h, w).astype(np.float32)
    pano = cv2.remap(background, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    seen = inside.reshape(h, w) & (cv2.remap(have.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0)
    # Directions the head camera never saw (behind it, below the robot): inpaint at low resolution, then blend in.
    small = cv2.resize(pano, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    hole = cv2.resize((~seen).astype(np.uint8), (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
    filled = cv2.resize(cv2.inpaint(small[..., ::-1], hole, 15, cv2.INPAINT_TELEA)[..., ::-1], (w, h), interpolation=cv2.INTER_LINEAR)
    pano[~seen] = filled[~seen]
    cv2.imwrite(os.path.join(OUT, "panorama.png"), pano[..., ::-1])
    print(f"[lab] panorama {w}x{h}: {100 * seen.mean():.0f} % of directions seen by the head camera", flush=True)

    table = fit_table(background, pose)
    # Colors: median of the table top and of the floor between the robot and the table (camera view).
    corners = np.array([[table["table_front_x"] + 0.05, table["table_center_y"] - table["table_width"] / 2 + 0.05],
                        [table["table_front_x"] + table["table_depth"] - 0.05, table["table_center_y"] + table["table_width"] / 2 - 0.05]])
    xs, ys = np.meshgrid(np.linspace(*corners[:, 0], 20), np.linspace(*corners[:, 1], 40))
    top_uv, _ = lens_project(np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, table["table_height"])]), pose)
    ok = (top_uv[:, 0] >= 0) & (top_uv[:, 0] < SIZE) & (top_uv[:, 1] >= 0) & (top_uv[:, 1] < SIZE)
    m_top = np.zeros((SIZE, SIZE), bool)
    m_top[top_uv[ok, 1].astype(int), top_uv[ok, 0].astype(int)] = True
    floor_uv, _ = lens_project(np.column_stack([np.full(40, 0.55), np.linspace(-0.9, -0.5, 40), np.full(40, -0.03)]), pose)
    okf = (floor_uv[:, 0] >= 0) & (floor_uv[:, 0] < SIZE) & (floor_uv[:, 1] >= 0) & (floor_uv[:, 1] < SIZE)
    m_floor = np.zeros((SIZE, SIZE), bool)
    m_floor[floor_uv[okf, 1].astype(int), floor_uv[okf, 0].astype(int)] = True
    table["table_color_display"] = mean_color(background, m_top & have)
    table["floor_color_display"] = mean_color(background, m_floor & have) if (m_floor & have).any() else [0.6, 0.4, 0.2]
    with open(os.path.join(OUT, "table.json"), "w") as f:
        json.dump({**table, "episodes": eps, "frames": len(imgs), "panorama": "panorama.png", "size": [w, h]}, f, indent=1)
    print(f"[lab] table {json.dumps(table)}", flush=True)
    # Overlay of the fitted table on the background, for checking.
    over = background.copy()
    fx, d, wd, cy, ht = (table[k] for k in ("table_front_x", "table_depth", "table_width", "table_center_y", "table_height"))
    rect = np.array([[fx, cy - wd / 2, ht], [fx + d, cy - wd / 2, ht], [fx + d, cy + wd / 2, ht], [fx, cy + wd / 2, ht]])
    for a, b in zip(rect, np.roll(rect, -1, axis=0), strict=True):
        pts, _ = lens_project(a + np.linspace(0, 1, 80)[:, None] * (b - a), pose)
        for u, v in pts.astype(int):
            if 0 <= u < SIZE and 0 <= v < SIZE:
                over[v, u] = (255, 0, 255)
    cv2.imwrite(os.path.join(OUT, "table_fit.png"), over[..., ::-1])


if __name__ == "__main__":
    main()
