"""GARMI in Isaac Sim 6.2: URDF -> USD import and articulation setup.

The garmi_description URDF needs four fixes for Isaac Sim's importer and PhysX (found 2026-10-23):
  1. package:// mesh paths -> absolute paths;
  2. <gazebo>, <ros2_control> and Franka's <position_based_velocity_limits> removed (the converter
     cannot store them and crashes);
  3. <mimic> removed: PhysX mimic joints made the articulation blow up to NaN within ~30 steps.
     The coupled joints (second finger, upper lift stage) are commanded together instead;
  4. after import, physics lives in the USD variant set "Physics", which must be set to "physx".
Joint drives are not imported (all gains 0), so `set_drive_gains` sets them.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

DESCRIPTION_DIR = os.path.expanduser("~/sim/garmi_description/garmi_description")
ASSET_DIR = os.path.expanduser("~/sim/garmi_bench/assets/garmi")
ARTICULATION_ROOT = "Geometry/base_link/chassis_link"  # relative to the robot prim


def write_sim_urdf(out_path: str) -> str:
    src = os.path.join(DESCRIPTION_DIR, "urdf/garmi.urdf")
    tree = ET.parse(src)
    root = tree.getroot()
    for tag in ("gazebo", "ros2_control"):
        for e in root.findall(tag):
            root.remove(e)
    for j in root.findall("joint"):
        for tag in ("position_based_velocity_limits", "mimic"):
            for e in j.findall(tag):
                j.remove(e)
    for m in root.iter("mesh"):
        m.set("filename", re.sub(r"^package://garmi_description/", DESCRIPTION_DIR + "/", m.get("filename")))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path)
    return out_path


def import_garmi_usd(force: bool = False) -> str:
    """Convert the URDF once; returns the USD path (cached in ASSET_DIR)."""
    usd = os.path.join(ASSET_DIR, "garmi", "garmi.usda")
    if os.path.exists(usd) and not force:
        return usd
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    import omni.kit.app

    omni.kit.app.get_app().update()
    from isaacsim.asset.importer.urdf import URDFImporter
    from isaacsim.asset.importer.urdf import URDFImporterConfig

    os.system(f"rm -rf {os.path.join(ASSET_DIR, 'garmi')}")
    urdf = write_sim_urdf(os.path.join(ASSET_DIR, "garmi.urdf"))
    cfg = URDFImporterConfig(urdf_path=urdf, usd_path=ASSET_DIR, fix_base=True, merge_fixed_joints=False)
    return URDFImporter(cfg).import_urdf()


def drive_gains(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Stiff arm drives stand in for the real robot's gravity-compensated impedance control."""
    kp, kd = np.zeros(len(names)), np.zeros(len(names))
    for i, n in enumerate(names):
        if "_fr3_joint" in n:
            kp[i], kd[i] = 3000.0, 150.0
        elif "finger" in n:
            kp[i], kd[i] = 2000.0, 100.0
        elif n.startswith("lift"):
            kp[i], kd[i] = 1e5, 1e3
        elif n.startswith("o1_motor"):
            kp[i], kd[i] = 100.0, 10.0
    return kp, kd


def start_positions(names: list[str], current: np.ndarray) -> np.ndarray:
    from garmi_bench.spec import START_FINGER
    from garmi_bench.spec import START_JOINTS

    q = np.array(current, dtype=float).copy()
    for side, vals in START_JOINTS.items():
        for i, v in enumerate(vals):
            q[names.index(f"{side}_fr3_joint{i + 1}")] = v
        for f in (1, 2):
            q[names.index(f"{side}_fr3_finger_joint{f}")] = START_FINGER
    for i, n in enumerate(names):
        if n.startswith(("lift", "o1_motor")):
            q[i] = 0.0
    return q
