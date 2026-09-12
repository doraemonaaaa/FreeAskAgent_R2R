"""sensor_config.yaml calibration entries -> the runner's camera model."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

V3 = Path(__file__).resolve().parents[1] / "integrations/v3"
sys.path.insert(0, str(V3))
from camera_model import CameraModel  # noqa: E402
from run_config import RunConfig  # noqa: E402


def test_simulator_model_is_the_ideal_pinhole_from_hfov():
    cam = CameraModel(width=640, height=480, hfov_deg=90.0, height_m=1.25, pitch_deg=-15)
    k = cam.matrix(640, 480)
    focal = 0.5 * 640 / math.tan(math.radians(45.0))
    assert np.allclose(k, [[focal, 0, 319.5], [0, focal, 239.5], [0, 0, 1]])
    assert cam.calibrated is None and cam.distortion is None
    assert cam.position == (0.0, 1.25, 0.0) and cam.height_m == 1.25 and cam.pitch_deg == -15
    assert np.allclose(cam.orientation_rad(), (math.radians(-15), 0.0, 0.0))
    rgb, depth = np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.float32)
    assert cam.undistort(rgb, depth) == (rgb, depth)


def test_calibrated_intrinsics_scale_with_the_image_size():
    cam = CameraModel(width=640, height=480, hfov_deg=90.0,
                      intrinsics='{"fx": 600, "fy": 590, "cx": 330, "cy": 235}')
    assert np.allclose(cam.matrix(640, 480), [[600, 0, 330], [0, 590, 235], [0, 0, 1]])
    assert np.allclose(cam.matrix(320, 240), [[300, 0, 165], [0, 295, 117.5], [0, 0, 1]])  # a half-size preview


def test_extrinsics_replace_height_and_pitch():
    cam = CameraModel(width=640, height=480, hfov_deg=90.0, height_m=1.25, pitch_deg=-15,
                      extrinsics={"xyz": [0.05, 0.62, -0.1], "rpy_deg": [1.0, -20.0, 0.5]})
    assert cam.position == (0.05, 0.62, -0.1) and cam.height_m == 0.62 and cam.pitch_deg == -20.0
    assert np.allclose(cam.orientation_rad(), (math.radians(-20.0), math.radians(0.5), math.radians(1.0)))


def test_distortion_requires_intrinsics_and_is_ignored_when_all_zero():
    with pytest.raises(ValueError, match="needs camera.intrinsics"):
        CameraModel(width=640, height=480, hfov_deg=90.0, distortion="0.1,0,0,0")
    cam = CameraModel(width=640, height=480, hfov_deg=90.0, intrinsics={"fx": 600, "fy": 600, "cx": 320, "cy": 240},
                      distortion=[0, 0, 0, 0, 0])
    assert cam.distortion is None


def test_undistort_rectifies_both_images_with_the_same_map():
    cv2 = pytest.importorskip("cv2")
    cam = CameraModel(width=64, height=48, hfov_deg=90.0, intrinsics={"fx": 60, "fy": 60, "cx": 32, "cy": 24},
                      distortion=[-0.3, 0.1, 0, 0, 0])
    rgb = np.zeros((48, 64, 3), np.uint8); rgb[20:28, 28:36] = 255
    depth = np.full((48, 64), 2.0, np.float32); depth[20:28, 28:36] = 1.0
    out_rgb, out_depth = cam.undistort(rgb, depth)
    assert out_rgb.shape == rgb.shape and out_depth.shape == depth.shape and out_depth.dtype == depth.dtype
    assert set(np.unique(out_depth)).issubset({0.0, 1.0, 2.0})  # nearest neighbour: no blended depths
    assert out_rgb[24, 32].max() == 255  # the centre is fixed by the model


def test_run_config_hands_calibration_to_the_runner_as_json(tmp_path):
    (tmp_path / "s.yaml").write_text(
        "camera:\n  height_m: 1.0\n  pitch_deg: 0\n  hfov_deg: 87\n  width: 848\n  height: 480\n"
        "  intrinsics: {fx: 615.2, fy: 615.0, cx: 424.1, cy: 241.7}\n  distortion: [0.1, -0.2, 0.001, 0.0, 0.05]\n"
        "  extrinsics:\n    xyz: [0.0, 0.6, 0.0]\n    rpy_deg: [0.0, -12.0, 0.0]\n"
        "depth:\n  min_m: 0.3\n  max_m: 6.0\n")
    (tmp_path / "c.yaml").write_text("sensors: s.yaml\n")
    defaults = RunConfig.load(tmp_path / "c.yaml").argparse_defaults()
    cam = CameraModel(width=defaults["image_width"], height=defaults["image_height"], hfov_deg=defaults["depth_hfov"],
                      intrinsics=defaults["camera_intrinsics"], distortion=defaults["camera_distortion"],
                      extrinsics=defaults["camera_extrinsics"], height_m=defaults["camera_height_m"],
                      pitch_deg=defaults["camera_pitch_deg"])
    assert cam.calibrated == (615.2, 615.0, 424.1, 241.7) and cam.distortion is not None
    assert cam.height_m == 0.6 and cam.pitch_deg == -12.0
    assert json.loads(defaults["camera_intrinsics"])["fx"] == 615.2
