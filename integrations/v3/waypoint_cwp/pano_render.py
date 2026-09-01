"""12-view ring renderer for the CWP waypoint predictor.

Builds a habitat_sim Simulator per scene with the sensor layout the predictor
was trained on (RGB 224x224 / depth 256x256, HFOV 90, level camera at 1.25 m),
and renders the 12 clockwise slots by yawing the agent -30 deg per slot.
Depth is returned in [0,1] (= meters/10, clipped) to match DDPPO pretraining.
"""
import math
import os

import numpy as np

HABITAT_DATA = "/data/pengyh/workspace/habitat/data"
CAMERA_HEIGHT = 1.25
DEPTH_SCALE_M = 10.0


def make_sim(scene_id, gpu_device_id=0, agent_view_pitch_deg=None):
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    import habitat_sim

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = os.path.join(HABITAT_DATA, "scene_datasets", scene_id)
    cfg.gpu_device_id = gpu_device_id
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "cwp_rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [224, 224]; rgb.hfov = 90
    rgb.position = [0.0, CAMERA_HEIGHT, 0.0]; rgb.orientation = [0.0, 0.0, 0.0]
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid = "cwp_depth"; dep.sensor_type = habitat_sim.SensorType.DEPTH
    dep.resolution = [256, 256]; dep.hfov = 90
    dep.position = [0.0, CAMERA_HEIGHT, 0.0]; dep.orientation = [0.0, 0.0, 0.0]
    specs = [rgb, dep]
    if agent_view_pitch_deg is not None:
        # the deployment agent's forward view (for floor-openings candidates)
        fo_rgb = habitat_sim.CameraSensorSpec()
        fo_rgb.uuid = "fo_rgb"; fo_rgb.sensor_type = habitat_sim.SensorType.COLOR
        fo_rgb.resolution = [480, 640]; fo_rgb.hfov = 90
        fo_rgb.position = [0.0, CAMERA_HEIGHT, 0.0]
        fo_rgb.orientation = [math.radians(agent_view_pitch_deg), 0.0, 0.0]
        fo_dep = habitat_sim.CameraSensorSpec()
        fo_dep.uuid = "fo_depth"; fo_dep.sensor_type = habitat_sim.SensorType.DEPTH
        fo_dep.resolution = [480, 640]; fo_dep.hfov = 90
        fo_dep.position = [0.0, CAMERA_HEIGHT, 0.0]
        fo_dep.orientation = [math.radians(agent_view_pitch_deg), 0.0, 0.0]
        specs += [fo_rgb, fo_dep]
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))


def _quat_yaw(yaw_rad):
    import quaternion  # noqa: registered by habitat_sim import
    return quaternion.from_rotation_vector([0.0, yaw_rad, 0.0])


def render_ring(sim, position, heading_rad, num_slots=12):
    """Render the clockwise 12-view ring at `position` facing `heading_rad`.

    heading convention: habitat yaw (radians, CCW-positive) of the forward view.
    slot i = forward yaw - i*30deg (i.e. rotating right), matching SmartWay.
    Returns (rgb_list[12] HxWx3 uint8, depth_list[12] 256x256 float in [0,1]).
    Restores the agent state afterwards.
    """
    agent = sim.get_agent(0)
    saved = agent.get_state()
    rgbs, deps = [], []
    step = 2.0 * math.pi / num_slots
    for i in range(num_slots):
        state = agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = _quat_yaw(heading_rad - i * step)
        state.sensor_states = {}
        agent.set_state(state, reset_sensors=True)
        obs = sim.get_sensor_observations()
        rgbs.append(np.asarray(obs["cwp_rgb"])[..., :3].copy())
        d = np.asarray(obs["cwp_depth"], dtype=np.float32)
        deps.append(np.clip(d / DEPTH_SCALE_M, 0.0, 1.0))
    agent.set_state(saved, reset_sensors=True)
    return rgbs, deps
