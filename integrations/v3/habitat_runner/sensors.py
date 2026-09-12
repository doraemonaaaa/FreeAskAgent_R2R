"""Habitat sensor configuration and per-step observation geometry."""

import numpy as np
from PIL import Image

from . import settings

import habitat  # noqa: E402,F401  (path set up by settings)


def motion_overrides(forward_step_m, turn_angle_deg):
    return [
        "habitat.simulator.forward_step_size={}".format(float(forward_step_m)),
        "habitat.simulator.turn_angle={}".format(int(turn_angle_deg)),
        "habitat.task.measurements.success.success_distance=3.0",
    ]


def sensor_overrides(camera, width, height, hfov_deg, depth_min_m, depth_max_m, depth_normalize):
    """Habitat RGB-D sensor settings from sensor_config.yaml. RGB and depth
    must share pixel coordinates and the camera frame: the agent back-projects
    the RGB-selected pixel through this depth map. The mount (position and
    orientation) comes from the camera model: extrinsics when calibrated,
    height + pitch otherwise."""
    out = []
    px, py, pz = camera.position
    ox, oy, oz = camera.orientation_rad()
    for sensor in ("rgb", "depth"):
        prefix = "habitat.simulator.agents.main_agent.sim_sensors.{}_sensor.".format(sensor)
        out += [
            prefix + "height={}".format(int(height)),
            prefix + "width={}".format(int(width)),
            prefix + "hfov={}".format(int(round(float(hfov_deg)))),  # Habitat's hfov is an integer field
            prefix + "position=[{},{},{}]".format(float(px), float(py), float(pz)),
            prefix + "orientation=[{},{},{}]".format(float(ox), float(oy), float(oz)),
        ]
    prefix = "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor."
    out += [
        prefix + "type=HabitatSimDepthSensor",
        prefix + "min_depth={}".format(float(depth_min_m)),
        prefix + "max_depth={}".format(float(depth_max_m)),
        prefix + "normalize_depth={}".format("true" if depth_normalize else "false"),
    ]
    return out



def rgb_depth(sensor_values):
    """Extract the RGB-D pair from any observation dict.

    ``sim.get_observations_at`` renders the simulator's own sensors only, so the
    task's instruction sensor is absent from a preview observation; reading it
    stays at the Env level in ``unpack_observation``.
    """
    rgb = np.asarray(sensor_values["rgb"])[..., :3]
    depth_key = depth_sensor_key(sensor_values)
    depth = np.asarray(sensor_values[depth_key])
    if depth.ndim == 3:
        depth = depth[..., 0]
    # Identity unless sensor_config.yaml lists distortion; previews go through here too.
    rgb, depth = settings.CAMERA.undistort(rgb, depth)
    return (
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(depth),
    )


def unpack_observation(observation):
    rgb, depth = rgb_depth(observation)
    return rgb, depth, observation["instruction"]["text"]


def depth_sensor_key(sensor_values):
    """Support Habitat releases that expose the depth sensor under either key."""
    for key in ("depth", "depth_sensor"):
        if key in sensor_values:
            return key
    raise KeyError(
        "Depth sensor is missing; expected one of ('depth', 'depth_sensor'), got {}".format(
            sorted(sensor_values)
        )
    )


def camera_intrinsics(width, height, hfov_degrees):
    """K for an image of this size: the calibrated matrix when the camera
    model has one, else the ideal pinhole from ``hfov_degrees``."""
    if settings.CAMERA.calibrated is not None:
        return settings.CAMERA.matrix(width, height)
    focal = 0.5 * width / np.tan(np.deg2rad(hfov_degrees) / 2.0)
    return np.array(((focal, 0, (width - 1) / 2), (0, focal, (height - 1) / 2), (0, 0, 1)), dtype=np.float64)


def semantic_region_id(env, cache={}):
    """MP3D semantic region (room) the agent stands in, or None.

    Ground truth for diagnosing the map's doorway-crossing events: a change
    of region id between steps is a real room transition. Uses the region
    AABBs with a generous y pad (region boxes hug the floor slab).
    """
    try:
        scene_id = env.current_episode.scene_id
        boxes = cache.get(scene_id)
        if boxes is None:
            import numpy as _np

            boxes = []
            for region in env.sim.semantic_scene.regions:
                low = _np.asarray(region.aabb.min, dtype=float)
                high = _np.asarray(region.aabb.max, dtype=float)
                boxes.append((region.id, low, high))
            cache.clear()
            cache[scene_id] = boxes
        position = env.sim.get_agent_state().position
        best = None
        for region_id, low, high in boxes:
            if (
                low[0] <= position[0] <= high[0]
                and low[2] <= position[2] <= high[2]
                and low[1] - 1.2 <= position[1] <= high[1] + 1.2
            ):
                # Prefer the smallest box that contains the point: nested
                # region boxes (hallway spanning the floor) otherwise win.
                area = (high[0] - low[0]) * (high[2] - low[2])
                if best is None or area < best[1]:
                    best = (region_id, area)
        return best[0] if best else None
    except Exception:
        return None


def camera_to_world_matrix(env):
    """Build a Habitat camera-to-world matrix from the depth sensor state."""
    from habitat_sim.utils.common import quat_to_magnum

    sensor_states = env.sim.get_agent_state().sensor_states
    state = sensor_states[depth_sensor_key(sensor_states)]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(quat_to_magnum(state.rotation).to_matrix())
    transform[:3, 3] = np.asarray(state.position)
    return transform


def downscale_view(rgb, depth, intrinsics, scale):
    """Shrink one rendered view and rescale its intrinsics to match.

    Preview payloads cross a pipe as base64 every previewed step, and the VLM
    resizes them to its own budget anyway. Depth uses nearest-neighbour so a
    resampled pixel is always a real measured range rather than an interpolated
    value straddling a depth discontinuity.
    """
    if scale >= 1.0:
        return rgb, depth, intrinsics

    height, width = rgb.shape[:2]
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)

    small_rgb = np.asarray(
        Image.fromarray(rgb).resize((new_width, new_height), Image.BILINEAR)
    )
    rows = (np.arange(new_height) * height // new_height).clip(0, height - 1)
    columns = (np.arange(new_width) * width // new_width).clip(0, width - 1)
    small_depth = np.ascontiguousarray(depth[np.ix_(rows, columns)])

    # The principal point is expressed in pixels, so every intrinsic scales
    # with the axis it belongs to.
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[0, 0] *= new_width / width
    scaled[0, 2] *= new_width / width
    scaled[1, 1] *= new_height / height
    scaled[1, 2] *= new_height / height
    return small_rgb, small_depth, scaled


def render_preview_views(env, yaws_deg, hfov_deg, scale=1.0):
    """Render extra headings without consuming an episode step.

    ``get_observations_at`` teleports, renders through habitat-lab's sensor
    suite, and restores the pose. It never goes through ``Env.step``, so the
    episode's step budget and every measurement stay untouched. Each view keeps
    its own intrinsics and camera transform, so a waypoint chosen inside it
    back-projects in its own frame.

    ``yaws_deg`` follows the agent's own sign convention: positive is to the
    right, negative to the left, matching ``yaw_delta_deg`` everywhere else.
    The quaternion is built from the negated angle because a right-handed
    rotation about +y turns the camera the other way.
    """
    from habitat_sim.utils.common import quat_from_angle_axis

    saved = env.sim.get_agent_state()
    views = []
    try:
        for yaw_deg in sorted(yaws_deg):
            rotation = saved.rotation * quat_from_angle_axis(
                float(np.deg2rad(-yaw_deg)), np.array([0.0, 1.0, 0.0])
            )
            observation = env.sim.get_observations_at(
                position=saved.position,
                rotation=rotation,
                # Hold the pose so this heading's own sensor transform can be
                # read; the finally block restores it.
                keep_agent_at_new_pose=True,
            )
            if observation is None:
                continue
            rgb, depth = rgb_depth(observation)
            intrinsics = camera_intrinsics(
                rgb.shape[1], rgb.shape[0], hfov_deg
            )
            rgb, depth, intrinsics = downscale_view(
                rgb, depth, intrinsics, scale
            )
            views.append({
                "yaw_deg": float(yaw_deg),
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world_matrix(env),
            })
    finally:
        env.sim.set_agent_state(
            saved.position, saved.rotation, reset_sensors=False
        )
    return views


def navigable_window(env, radius_m=6.0, resolution_m=0.25, *, include_heights=False):
    """Navmesh traversability on a grid around the agent, at its floor level."""
    state = env.sim.get_agent_state()
    pathfinder = env.sim.pathfinder
    x0, y0, z0 = (float(v) for v in state.position)
    cells = int(round(2 * radius_m / resolution_m))
    origin = (x0 - radius_m, z0 - radius_m)
    mask = np.zeros((cells, cells), dtype=np.bool_)
    heights = np.full((cells, cells), np.nan, dtype=np.float32) if include_heights else None
    for row in range(cells):
        z = origin[1] + (row + 0.5) * resolution_m
        for col in range(cells):
            x = origin[0] + (col + 0.5) * resolution_m
            mask[row, col] = pathfinder.is_navigable([x, y0, z], 0.5)
            if heights is not None and mask[row, col]:
                snapped = np.asarray(pathfinder.snap_point([x, y0, z]), dtype=float)
                if np.isfinite(snapped).all():
                    heights[row, col] = snapped[1]
    result = {"origin_xz": origin, "resolution_m": resolution_m, "mask": mask}
    if heights is not None:
        result["height_m"] = heights
        settings = getattr(pathfinder, "nav_mesh_settings", None)
        result["height_cell_m"] = float(getattr(settings, "cell_height", 0.0))
    return result

