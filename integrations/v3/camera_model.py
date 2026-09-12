"""Camera calibration for the runner: intrinsics, distortion, mount extrinsics.

Everything comes from sensor_config.yaml. Without calibration entries the
model is the ideal pinhole the simulator renders with (fx = fy from the
horizontal FOV, principal point at the image centre, no distortion, mount =
height + pitch), so simulator runs are unchanged. A real camera fills in its
calibrated matrix, distortion coefficients and base-to-camera transform.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np


def _parse_mapping(value: Any, name: str) -> Optional[dict]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping (or its JSON text)")
    return value


def _parse_sequence(value: Any, name: str) -> Optional[list[float]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith("[") else [float(v) for v in value.split(",")]
    return [float(v) for v in value]


class CameraModel:
    """Pinhole (+ optional calibration) shared by the nav camera and previews."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        hfov_deg: float,
        intrinsics: Any = None,
        distortion: Any = None,
        extrinsics: Any = None,
        height_m: float = 1.25,
        pitch_deg: float = 0.0,
    ) -> None:
        self.width, self.height, self.hfov_deg = int(width), int(height), float(hfov_deg)
        calib = _parse_mapping(intrinsics, "camera.intrinsics")
        self.calibrated: Optional[tuple[float, float, float, float]] = None
        if calib:
            missing = {"fx", "fy", "cx", "cy"} - set(calib)
            if missing:
                raise ValueError(f"camera.intrinsics needs fx, fy, cx, cy (missing {sorted(missing)})")
            self.calibrated = tuple(float(calib[k]) for k in ("fx", "fy", "cx", "cy"))
        coeffs = _parse_sequence(distortion, "camera.distortion")
        self.distortion = np.asarray(coeffs, dtype=np.float64) if coeffs and any(abs(c) > 0 for c in coeffs) else None
        if self.distortion is not None and self.calibrated is None:
            raise ValueError("camera.distortion needs camera.intrinsics")
        mount = _parse_mapping(extrinsics, "camera.extrinsics")
        if mount:
            xyz = _parse_sequence(mount.get("xyz"), "camera.extrinsics.xyz")
            rpy = _parse_sequence(mount.get("rpy_deg"), "camera.extrinsics.rpy_deg")
            if xyz is None or len(xyz) != 3 or rpy is None or len(rpy) != 3:
                raise ValueError("camera.extrinsics needs xyz: [x, y, z] and rpy_deg: [roll, pitch, yaw]")
            self.position = tuple(xyz)
            self.rpy_deg = tuple(rpy)
        else:
            self.position = (0.0, float(height_m), 0.0)
            self.rpy_deg = (0.0, float(pitch_deg), 0.0)
        self._maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # -- geometry ---------------------------------------------------------

    @property
    def height_m(self) -> float:
        """Camera centre above the base (the agent's floor plane)."""
        return float(self.position[1])

    @property
    def pitch_deg(self) -> float:
        return float(self.rpy_deg[1])

    def orientation_rad(self) -> tuple[float, float, float]:
        """Habitat sensor orientation [about x, about y, about z] = [pitch, yaw, roll]."""
        roll, pitch, yaw = self.rpy_deg
        return (math.radians(pitch), math.radians(yaw), math.radians(roll))

    def matrix(self, width: int, height: int) -> np.ndarray:
        """3x3 K for an image of this size (calibrated values scale with it)."""
        if self.calibrated is None:
            focal = 0.5 * width / math.tan(math.radians(self.hfov_deg) / 2.0)
            return np.array(((focal, 0, (width - 1) / 2), (0, focal, (height - 1) / 2), (0, 0, 1)), dtype=np.float64)
        fx, fy, cx, cy = self.calibrated
        sx, sy = width / self.width, height / self.height
        return np.array(((fx * sx, 0, cx * sx), (0, fy * sy, cy * sy), (0, 0, 1)), dtype=np.float64)

    # -- images -----------------------------------------------------------

    def undistort(self, rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rectify both images with the calibrated model; identity without distortion."""
        if self.distortion is None:
            return rgb, depth
        import cv2

        height, width = depth.shape[:2]
        key = (width, height)
        if key not in self._maps:
            k = self.matrix(width, height)
            self._maps[key] = cv2.initUndistortRectifyMap(k, self.distortion, None, k, (width, height), cv2.CV_32FC1)
        map_x, map_y = self._maps[key]
        rgb_out = cv2.remap(np.ascontiguousarray(rgb), map_x, map_y, cv2.INTER_LINEAR)
        # Nearest neighbour keeps depth edges sharp instead of blending a wall into the floor.
        depth_out = cv2.remap(np.ascontiguousarray(depth.astype(np.float32)), map_x, map_y, cv2.INTER_NEAREST)
        return rgb_out, depth_out.astype(depth.dtype, copy=False)

    def describe(self) -> str:
        parts = [f"{self.width}x{self.height} hfov={self.hfov_deg:g}"]
        parts.append("K=calibrated(fx={:.1f}, fy={:.1f}, cx={:.1f}, cy={:.1f})".format(*self.calibrated) if self.calibrated else "K=from-hfov")
        parts.append("distortion={}".format("yes" if self.distortion is not None else "none"))
        parts.append("mount xyz={} rpy_deg={}".format(tuple(round(v, 3) for v in self.position), tuple(round(v, 1) for v in self.rpy_deg)))
        return " ".join(parts)


__all__ = ("CameraModel",)
