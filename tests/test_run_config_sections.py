"""config.yaml + sensor_config.yaml -> runner defaults and agent environment."""

import json
from pathlib import Path
import sys

import pytest

V3 = Path(__file__).resolve().parents[1] / "integrations/v3"
sys.path.insert(0, str(V3))
from run_config import RunConfig  # noqa: E402


def test_repo_config_exposes_sensors_robot_and_method_sections():
    config = RunConfig.load(V3 / "config.yaml")
    defaults = config.argparse_defaults()
    assert defaults["camera_height_m"] == 1.25 and defaults["camera_pitch_deg"] == -15
    assert defaults["image_width"] == 640 and defaults["image_height"] == 480
    assert defaults["depth_max_m"] == 10.0 and defaults["depth_normalize"] is False
    assert defaults["forward_step_m"] == 0.25 and defaults["turn_angle_deg"] == 15
    env = config.agent_env()
    spatial = json.loads(env["VLN_SPATIAL_MEMORY_PARAMS"])
    nav = json.loads(env["VLN_NAV_PARAMS"])
    robot = json.loads(env["VLN_ROBOT_PARAMS"])
    assert spatial["resolution_m"] == 0.1 and spatial["route_inflate_m"] == [0.25, 0.15]
    assert nav["collision_translation_m"] == 0.05 and nav["final_approach_stop_m"] == 1.5
    assert robot == {"forward_step_m": 0.25, "turn_angle_deg": 15}
    assert env["CAPTIONER_OPENING_LOOK_AROUND"] == "1"
    manifest = config.manifest()
    assert manifest["sensors"]["camera"]["hfov_deg"] == 90 and "navigation" in manifest


def test_config_without_sensors_key_uses_the_repo_sensor_file(tmp_path):
    (tmp_path / "old.yaml").write_text("runner:\n  max_steps: 7\n  camera_pitch_deg: -5\n")
    config = RunConfig.load(tmp_path / "old.yaml")
    assert config.sensor_path == V3 / "sensor_config.yaml"
    defaults = config.argparse_defaults()
    assert defaults["camera_height_m"] == 1.25
    assert defaults["camera_pitch_deg"] == -5  # an explicit runner key still wins
    assert defaults["max_steps"] == 7
    assert "VLN_NAV_PARAMS" not in config.agent_env()


def test_named_sensor_file_is_relative_to_the_config(tmp_path):
    (tmp_path / "robot_sensors.yaml").write_text("camera:\n  height_m: 0.6\n  pitch_deg: -20\n  hfov_deg: 87\n  width: 848\n  height: 480\ndepth:\n  min_m: 0.3\n  max_m: 6.0\n")
    (tmp_path / "robot.yaml").write_text("sensors: robot_sensors.yaml\nrobot:\n  forward_step_m: 0.2\n  turn_angle_deg: 15\n")
    defaults = RunConfig.load(tmp_path / "robot.yaml").argparse_defaults()
    assert defaults["camera_height_m"] == 0.6 and defaults["depth_hfov"] == 87 and defaults["depth_max_m"] == 6.0
    assert defaults["forward_step_m"] == 0.2
    (tmp_path / "broken.yaml").write_text("sensors: missing.yaml\n")
    with pytest.raises(FileNotFoundError):
        RunConfig.load(tmp_path / "broken.yaml")
