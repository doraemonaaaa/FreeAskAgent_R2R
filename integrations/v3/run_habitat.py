"""Run the v3 RGB-D waypoint actor on Habitat R2R-CE with oracle local control.

The actor receives RGB, depth, the instruction, and camera calibration, then
returns a Habitat world-space waypoint.  ``ShortestPathFollower`` is strictly
the low-level controller that converts that waypoint to one discrete R2R-CE
action; it is not used to select the waypoint.
"""

import argparse
import base64
import io
import json
import logging
import os
import select
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

for _name in ("habitat", "habitat_sim", "magnum", "corrade", "transformers", "torch"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.captureWarnings(True)
warnings.filterwarnings("ignore")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

ROOT = Path(__file__).resolve().parents[2]
HABITAT_ROOT = ROOT.parent / "habitat" / "habitat-lab"
HABITAT_DATA = ROOT.parent / "habitat" / "data"
AGENTFLOW_ROOT = ROOT.parent / "FreeAskAgent"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HABITAT_ROOT / "habitat-lab"))

import habitat  # noqa: E402
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # noqa: E402

# The actor asks for turns in degrees and this runner executes whole repeats of
# the simulator's turn primitive, so this value is half of a contract with the
# actor's TURN_STEP_DEG rather than a private simulator setting. A mismatch
# would silently round every requested turn down.
TURN_ANGLE_DEG = 15

R2R_CE_OVERRIDES = [
    "habitat.simulator.forward_step_size=0.25",
    "habitat.simulator.turn_angle={}".format(TURN_ANGLE_DEG),
    "habitat.task.measurements.success.success_distance=3.0",
]


# Height of both cameras above the agent's base. The actor needs the same
# number to tell floor pixels from wall pixels when it snaps a waypoint, so it
# is defined once and passed through rather than repeated in the worker.
SENSOR_HEIGHT_M = 1.25

DEPTH_SENSOR_OVERRIDES = [
    # RGB and depth must share pixel coordinates and the same camera frame;
    # vln_agent_2 back-projects the RGB-selected pixel through this depth map.
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=480",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=640",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position=[0,{},0]".format(SENSOR_HEIGHT_M),
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.type=HabitatSimDepthSensor",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=480",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=640",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position=[0,{},0]".format(SENSOR_HEIGHT_M),
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth=0.0",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=10.0",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=false",
]


class WaypointActorProcess:
    """Keep the Python 3.12 vision model out of Habitat's Python process."""

    def __init__(self, python, worker, model_path, gpu_id=None, timeout=600, camera_height_m=SENSOR_HEIGHT_M):
        command = [
            str(python), str(worker), "--model-path", str(model_path),
            "--camera-height-m", repr(float(camera_height_m)),
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(AGENTFLOW_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        # Only pin the worker when --gpu-id is given; otherwise inherit the
        # caller's CUDA_VISIBLE_DEVICES so the shell setting is not silently lost.
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=environment, text=True, bufsize=1,
        )
        self.timeout = timeout

    @staticmethod
    def _png(rgb):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _array(values):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(values), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            raise RuntimeError("Waypoint actor timed out after {} seconds".format(self.timeout))
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("Waypoint actor process exited unexpectedly")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def prepare(self, instruction):
        """Initialize the worker's task memory before an episode starts."""
        return self._request({"operation": "prepare", "instruction": instruction})

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world, navigable=None, oracle_goal=None):
        encode_started = time.perf_counter()
        request = {
            "operation": "act",
            "rgb": self._png(rgb), "depth": self._array(depth),
            "instruction": instruction, "intrinsics": np.asarray(intrinsics).tolist(),
            "camera_to_world": np.asarray(camera_to_world).tolist(),
            # Ask for the agent's own map and marker frame only when a video
            # is being recorded: they cost a PNG encode per step.
            "want_visuals": bool(getattr(self, "want_visuals", False)),
        }
        if oracle_goal is not None:
            # Diagnostic only (--som-oracle): the goal position lets the
            # worker pick the best set-of-mark candidate without the model,
            # which bounds what perfect choices could achieve.
            request["oracle_goal_xyz"] = [float(v) for v in oracle_goal]
        if navigable is not None:
            # The controller's own traversability around the agent: what the
            # follower can reach. Floor seen through glass or past a railing
            # looks walkable in depth but is not, and a target there only
            # makes the follower turn in place.
            request["navigable"] = {
                "origin_xz": list(navigable["origin_xz"]),
                "resolution_m": navigable["resolution_m"],
                "mask": self._array(navigable["mask"]),
            }
        encode_ms = (time.perf_counter() - encode_started) * 1000
        roundtrip_started = time.perf_counter()
        result = self._request(request)
        # Serialization and pipe transfer are measured separately from the
        # worker's own model time so a slow step can be attributed to one side.
        result["encode_ms"] = encode_ms
        result["roundtrip_ms"] = (time.perf_counter() - roundtrip_started) * 1000
        if result.get("stop"):
            return None, result
        if "world_xyz" not in result:
            # A PREVIEW decision carries no waypoint by design: the actor is
            # asking to inspect the surrounding views before committing. The
            # caller distinguishes it from STOP by reading "action_mode".
            return None, result
        return np.asarray(result["world_xyz"], dtype=np.float32), result

    def act_on_preview(self, views, instruction):
        """Answer a PREVIEW decision with the headings Habitat just rendered."""
        encode_started = time.perf_counter()
        request = {
            "operation": "act_on_preview",
            "instruction": instruction,
            "want_visuals": bool(getattr(self, "want_visuals", False)),
            "views": [
                {
                    "yaw_deg": view["yaw_deg"],
                    "rgb": self._png(view["rgb"]),
                    "depth": self._array(view["depth"]),
                    "intrinsics": np.asarray(view["intrinsics"]).tolist(),
                    "camera_to_world": np.asarray(
                        view["camera_to_world"]
                    ).tolist(),
                }
                for view in views
            ],
        }
        encode_ms = (time.perf_counter() - encode_started) * 1000
        roundtrip_started = time.perf_counter()
        result = self._request(request)
        result["encode_ms"] = encode_ms
        result["roundtrip_ms"] = (time.perf_counter() - roundtrip_started) * 1000
        if result.get("stop"):
            return None, result
        if "world_xyz" not in result:
            return None, result
        return np.asarray(result["world_xyz"], dtype=np.float32), result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _rgb_depth(sensor_values):
    """Extract the RGB-D pair from any observation dict.

    ``sim.get_observations_at`` renders the simulator's own sensors only, so the
    task's instruction sensor is absent from a preview observation; reading it
    stays at the Env level in ``_observation``.
    """
    rgb = np.asarray(sensor_values["rgb"])[..., :3]
    depth_key = _depth_sensor_key(sensor_values)
    depth = np.asarray(sensor_values[depth_key])
    if depth.ndim == 3:
        depth = depth[..., 0]
    return (
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(depth),
    )


def _observation(observation):
    rgb, depth = _rgb_depth(observation)
    return rgb, depth, observation["instruction"]["text"]


def _depth_sensor_key(sensor_values):
    """Support Habitat releases that expose the depth sensor under either key."""
    for key in ("depth", "depth_sensor"):
        if key in sensor_values:
            return key
    raise KeyError(
        "Depth sensor is missing; expected one of ('depth', 'depth_sensor'), got {}".format(
            sorted(sensor_values)
        )
    )


def _intrinsics(width, height, hfov_degrees):
    focal = 0.5 * width / np.tan(np.deg2rad(hfov_degrees) / 2.0)
    return np.array(((focal, 0, (width - 1) / 2), (0, focal, (height - 1) / 2), (0, 0, 1)), dtype=np.float64)


def _camera_to_world(env):
    """Build a Habitat camera-to-world matrix from the depth sensor state."""
    from habitat_sim.utils.common import quat_to_magnum

    sensor_states = env.sim.get_agent_state().sensor_states
    state = sensor_states[_depth_sensor_key(sensor_states)]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(quat_to_magnum(state.rotation).to_matrix())
    transform[:3, 3] = np.asarray(state.position)
    return transform


def _downscale_view(rgb, depth, intrinsics, scale):
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


def _preview_views(env, yaws_deg, hfov_deg, scale=1.0):
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
            rgb, depth = _rgb_depth(observation)
            intrinsics = _intrinsics(
                rgb.shape[1], rgb.shape[0], hfov_deg
            )
            rgb, depth, intrinsics = _downscale_view(
                rgb, depth, intrinsics, scale
            )
            views.append({
                "yaw_deg": float(yaw_deg),
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "camera_to_world": _camera_to_world(env),
            })
    finally:
        env.sim.set_agent_state(
            saved.position, saved.rotation, reset_sensors=False
        )
    return views


def _navigable_window(env, radius_m=6.0, resolution_m=0.25):
    """Navmesh traversability on a grid around the agent, at its floor level."""
    state = env.sim.get_agent_state()
    pathfinder = env.sim.pathfinder
    x0, y0, z0 = (float(v) for v in state.position)
    cells = int(round(2 * radius_m / resolution_m))
    origin = (x0 - radius_m, z0 - radius_m)
    mask = np.zeros((cells, cells), dtype=np.bool_)
    for row in range(cells):
        z = origin[1] + (row + 0.5) * resolution_m
        for col in range(cells):
            x = origin[0] + (col + 0.5) * resolution_m
            mask[row, col] = pathfinder.is_navigable([x, y0, z], 0.5)
    return {"origin_xz": origin, "resolution_m": resolution_m, "mask": mask}


def _build_navmesh_map(env, resolution=1024):
    """Render the scene navmesh for the optional trajectory visualization."""
    from habitat.utils.visualizations.maps import get_topdown_map_from_sim

    return get_topdown_map_from_sim(
        env.sim, map_resolution=resolution, draw_border=True
    )


def _render_topdown(
    env,
    navmesh_map,
    positions,
    goal_position,
    output_height,
    waypoints=(),
    landmark_marks=(),
):
    """Draw the executed trajectory, start, current position, and goal.

    ``waypoints`` are the world-space targets the actor asked for, which show
    where it intended to go as opposed to where the follower took it.
    ``landmark_marks`` are ``(position, kind)`` pairs recording where the
    tracker reported standing at or crossing the active subgoal's landmark.
    """
    import cv2
    from habitat.utils.visualizations import maps

    image = maps.colorize_topdown_map(navmesh_map.copy())
    rows, columns = navmesh_map.shape[:2]

    def to_pixel(position):
        row, col = maps.to_grid(
            position[2], position[0], navmesh_map.shape,
            pathfinder=env.sim.pathfinder,
        )
        # A requested waypoint can land off the navmesh, and to_grid does not
        # clamp; an out-of-bounds point would otherwise be drawn nowhere.
        return (
            int(np.clip(col, 0, columns - 1)),
            int(np.clip(row, 0, rows - 1)),
        )

    path = [to_pixel(position) for position in positions]
    if len(path) > 1:
        cv2.polylines(image, [np.asarray(path, dtype=np.int32)], False, (0, 80, 255), 3)
    # Drawn under the trajectory endpoints so the executed path stays legible.
    for waypoint in waypoints:
        cv2.circle(image, to_pixel(waypoint), 3, _REQUESTED_COLOR, -1, cv2.LINE_AA)
    if waypoints and path:
        cv2.line(
            image, path[-1], to_pixel(waypoints[-1]), _REQUESTED_COLOR, 1,
            cv2.LINE_AA,
        )
    for position, kind in landmark_marks:
        cv2.drawMarker(
            image, to_pixel(position),
            (230, 80, 230) if kind == "PASSED" else _LANDMARK_COLORS["AT"],
            cv2.MARKER_DIAMOND if kind == "PASSED" else cv2.MARKER_TRIANGLE_UP,
            14, 2,
        )
    if path:
        cv2.circle(image, path[0], 7, (0, 180, 0), -1)
        cv2.circle(image, path[-1], 7, (255, 80, 0), -1)
    if goal_position is not None:
        cv2.drawMarker(image, to_pixel(goal_position), (255, 0, 0), cv2.MARKER_STAR, 16, 2)
    height, width = image.shape[:2]
    return cv2.resize(
        image, (int(width * output_height / height), output_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _topdown_panel(rgb, topdown, agent_map=None):
    """First-person view | the agent's own map (when given) | true top-down.

    The middle panel is what Spatial Memory believes: unknown grey, free
    white, occupied black, trail blue, landmarks green, committed target red,
    set-of-mark candidates yellow, the agent orange. Comparing it with the
    ground-truth map on the right shows where the belief went wrong.
    """
    panels = [rgb]
    if agent_map is not None:
        height = rgb.shape[0]
        if agent_map.shape[0] != height:
            agent_map = np.asarray(
                Image.fromarray(agent_map).resize((height, height), Image.Resampling.NEAREST)
            )
        panels.append(agent_map)
    panels.append(topdown)
    return np.concatenate(panels, axis=1)


def _decode_visuals(decision):
    """(agent_map, marker_frame) from a worker response, either may be None."""
    visuals = (decision or {}).get("visuals") or {}

    def _png(key):
        encoded = visuals.get(key)
        if not encoded:
            return None
        try:
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
        except Exception:
            return None

    return _png("map_png"), _png("som_png")


def _clean_video_frame(rgb):
    """Return an unannotated RGB frame for the saved video."""
    return rgb.copy()


# Requested amber, executed green: the same pair the legend names.
_REQUESTED_COLOR = (255, 190, 0)
_APPLIED_COLOR = (0, 220, 90)

# The landmark tracker reports a proximity class rather than a distance, so the
# overlay carries it as color: cool when far, warm as the camera closes in.
_LANDMARK_COLORS = {
    "FAR": (110, 170, 255),
    "NEAR": (255, 150, 40),
    "AT": (60, 235, 140),
    "UNKNOWN": (170, 170, 170),
}


def _landmark_state(decision):
    """Return this step's landmark reading, or None when it never ran."""
    return (decision.get("debug") or {}).get("landmark")


def _landmark_mark_kind(landmark):
    """Classify a landmark reading for the top-down trajectory markers.

    Only the two states that pin the route to a place are marked: crossing the
    landmark, and standing at it. FAR/NEAR sightings happen on most steps and
    would bury the map.
    """
    if not landmark:
        return None
    if landmark.get("passed"):
        return "PASSED"
    if landmark.get("visible") and landmark.get("proximity") == "AT":
        return "AT"
    return None


def _draw_landmark_point(image, decision):
    """Plot the landmark the tracker located, colored by its proximity.

    The pixel is optional by design: when the model returns no usable ``u``/``v``
    there is simply no marker. The structured state remains available in the
    terminal debug output.
    """
    import cv2

    landmark = _landmark_state(decision)
    pixel = (decision.get("debug") or {}).get("landmark_pixel_uv")
    if not landmark or not pixel:
        return
    height, width = image.shape[:2]
    center = (
        int(np.clip(int(pixel[0]), 0, width - 1)),
        int(np.clip(int(pixel[1]), 0, height - 1)),
    )
    proximity = landmark.get("proximity") or "UNKNOWN"
    color = _LANDMARK_COLORS.get(proximity, _LANDMARK_COLORS["UNKNOWN"])
    # A diamond, so the landmark never reads as one of the waypoint circles.
    cv2.drawMarker(
        image, center, color, cv2.MARKER_DIAMOND, 22, 2, cv2.LINE_AA,
    )
    cv2.circle(image, center, 3, color, -1, cv2.LINE_AA)
    label = "LM {}{}".format(proximity, " PASSED" if landmark.get("passed") else "")
    cv2.putText(
        image, label, (center[0] + 14, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
    )


def _previewed_view(decision):
    """Return the surrounding view a previewed step committed to, if any."""
    block = decision.get("decision") or {}
    for key in ("execution", "exploration"):
        inner = block.get(key) or {}
        if inner.get("view_index") is not None:
            return inner
    return None


def _draw_preview_indicator(image, decision):
    """Show which surrounding heading a PREVIEW decision selected."""
    import cv2

    previewed = _previewed_view(decision)
    if previewed is None:
        return

    yaw_deg = float(previewed.get("view_yaw_deg") or 0.0)
    yaw_rad = np.deg2rad(yaw_deg)
    height, width = image.shape[:2]
    origin = (width // 2, max(int(height * 0.20), 55))
    length = max(int(min(width, height) * 0.14), 45)
    endpoint = (
        int(np.clip(origin[0] + length * np.sin(yaw_rad), 12, width - 13)),
        int(np.clip(origin[1] - length * np.cos(yaw_rad), 12, height - 13)),
    )
    color = (40, 225, 255)
    cv2.circle(image, origin, 7, color, 2, cv2.LINE_AA)
    cv2.arrowedLine(
        image, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.28,
    )
    label = "PREVIEW {:+.0f}deg".format(yaw_deg)
    cv2.putText(
        image,
        label,
        (max(origin[0] - 72, 4), origin[1] + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_turn_arrow(image, turn_deg):
    """Overlay the actor's requested in-place turn on an RGB video frame.

    Positive angles point right and negative angles point left, matching the
    convention used by ``_turn_primitive``.  A bent arrow is used instead of a
    straight horizontal arrow so it cannot be confused with an image-space
    waypoint direction.
    """
    import cv2

    if turn_deg is None or int(turn_deg) == 0:
        return

    turn_deg = int(turn_deg)
    height, width = image.shape[:2]
    direction = 1 if turn_deg > 0 else -1
    center_x = width // 2
    bend_y = max(int(height * 0.27), 42)
    stem_y = min(int(height * 0.46), height - 24)
    tip_x = int(np.clip(
        center_x + direction * max(int(width * 0.18), 55),
        24,
        width - 25,
    ))
    thickness = max(3, int(round(min(width, height) / 120.0)))
    color = (255, 210, 0)

    # A small translucent backing keeps the symbol readable in both bright
    # rooms and dark corridors without hiding much of the observation.
    overlay = image.copy()
    pad = 18
    left = max(min(center_x, tip_x) - pad, 0)
    right = min(max(center_x, tip_x) + pad, width - 1)
    top = max(bend_y - pad - 22, 0)
    bottom = min(stem_y + pad, height - 1)
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.38, image, 0.62, 0, dst=image)

    cv2.line(
        image, (center_x, stem_y), (center_x, bend_y),
        color, thickness, cv2.LINE_AA,
    )
    cv2.arrowedLine(
        image, (center_x, bend_y), (tip_x, bend_y),
        color, thickness, cv2.LINE_AA, tipLength=0.28,
    )
    label = "{} {}deg".format(
        "RIGHT" if direction > 0 else "LEFT", abs(turn_deg)
    )
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2,
    )
    text_x = int(np.clip(center_x - text_width // 2, 2, width - text_width - 2))
    text_y = max(top + text_height + 4, text_height + 2)
    cv2.putText(
        image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 2, cv2.LINE_AA,
    )


def _wrap_overlay_text(cv2, text, max_width, scale, thickness):
    """Wrap one label by rendered pixel width without dropping source text."""
    words = str(text or "-").split()
    if not words:
        return ["-"]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_task_context(image, decision):
    """Show the original instruction and the currently active subgoal."""
    import cv2

    task = decision.get("task_memory") or {}
    debug = decision.get("debug") or {}
    current = debug.get("subgoal_after")
    instruction = task.get("goal") or "-"
    if current is None:
        subgoal_text = (
            "COMPLETE"
            if task.get("current_subgoal_id") is None
            else str(task.get("current_subgoal_id"))
        )
    else:
        subgoal_text = "{}: {}".format(
            current.get("subgoal_id", "-"),
            current.get("description") or "-",
        )

    scale, thickness = 0.44, 1
    margin, line_height = 8, 18
    max_width = max(image.shape[1] - 2 * margin, 1)
    lines = _wrap_overlay_text(
        cv2, "Instruction: " + instruction, max_width, scale, thickness
    )
    lines.extend(_wrap_overlay_text(
        cv2, "Subgoal: " + subgoal_text, max_width, scale, thickness
    ))
    strip_height = margin + line_height * len(lines) + 4
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1] - 1, min(strip_height, image.shape[0]) - 1),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, dst=image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (margin, margin + line_height * (index + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _annotated_video_frame(rgb, decision, _steps):
    """Visualize only navigation actions and perception decisions.

    ``requested_pixel_uv`` is the location the waypoint policy asked for;
    ``pixel_uv`` is where the depth map allowed that waypoint to land. Drawing
    both, joined by a line, separates a bad model selection from a good
    selection that the walkable-pixel snap pulled somewhere else. The landmark
    decision is a diamond, PREVIEW is a cyan heading arrow, and an in-place
    turn is a yellow bent arrow. The top strip contains only the route
    instruction and active subgoal; detailed diagnostics remain in terminal
    logs.
    """
    import cv2

    image = rgb.copy()
    height, width = image.shape[:2]
    debug = decision.get("debug") or {}
    _draw_task_context(image, decision)

    def to_pixel(value):
        if not value:
            return None
        return (
            int(np.clip(int(value[0]), 0, width - 1)),
            int(np.clip(int(value[1]), 0, height - 1)),
        )

    # A previewed step chose its pixel inside a surrounding view, so those
    # coordinates address a different image than this one; drawing them here
    # would put markers on unrelated scenery.
    previewed = _previewed_view(decision)
    requested = (
        None if previewed else to_pixel(debug.get("requested_pixel_uv"))
    )
    applied = None if previewed else to_pixel(decision.get("pixel_uv"))
    if requested is not None and applied is not None and requested != applied:
        cv2.line(image, requested, applied, (255, 255, 255), 1, cv2.LINE_AA)
    if requested is not None:
        cv2.circle(image, requested, 9, _REQUESTED_COLOR, 2, cv2.LINE_AA)
        cv2.drawMarker(
            image, requested, _REQUESTED_COLOR, cv2.MARKER_CROSS, 14, 1,
        )
    if applied is not None:
        cv2.circle(image, applied, 6, _APPLIED_COLOR, -1, cv2.LINE_AA)
        cv2.circle(image, applied, 6, (255, 255, 255), 1, cv2.LINE_AA)
    _draw_landmark_point(image, decision)
    _draw_preview_indicator(image, decision)
    _draw_turn_arrow(image, decision.get("turn_deg"))
    return image


def _combined_step_timings(decision):
    """Combine the first and second halves of a PREVIEW step.

    ``act_on_preview`` returns its own timing dictionary, so without this merge
    the initial Captioner and waypoint request disappear from the runner log.
    """
    resolved = decision.get("timings") or {}
    preview = decision.get("preview") or {}
    initial = preview.get("act_timings") or {}

    def total(key):
        return float(initial.get(key, 0.0)) + float(
            resolved.get(key, 0.0)
        )

    return {
        "rgb_ms": total("rgb_ms"),
        "memory_ms": total("memory_ms"),
        "captioner_ms": total("captioner_ms"),
        "depth_ms": total("depth_ms"),
        "select_pixel_ms": total("select_pixel_ms"),
        "preview_select_ms": total("preview_select_ms"),
        "waypoint_ms": total("waypoint_ms"),
        "worker_ms": float(initial.get("total_ms", 0.0)) + float(
            resolved.get("total_ms", 0.0)
        ),
        "encode_ms": float(preview.get("act_encode_ms", 0.0)) + float(
            decision.get("encode_ms", 0.0)
        ),
        "roundtrip_ms": float(
            preview.get("act_roundtrip_ms", 0.0)
        ) + float(decision.get("roundtrip_ms", 0.0)),
        "preview_render_ms": float(preview.get("render_ms", 0.0)),
    }


def _latency_line(episode_id, steps, decision, step_ms, render_ms, env_ms):
    """Attribute one step's wall time to the model, the IPC, and the simulator."""
    timings = _combined_step_timings(decision)
    roundtrip_ms = timings["roundtrip_ms"]
    worker_ms = timings["worker_ms"]
    memory_ms = timings.get("memory_ms", 0.0)
    captioner_ms = timings.get("captioner_ms", 0.0)
    accounted = (
        timings.get("rgb_ms", 0.0) + memory_ms + timings.get("depth_ms", 0.0)
        + timings.get("select_pixel_ms", 0.0)
        + timings.get("preview_select_ms", 0.0)
        + timings.get("waypoint_ms", 0.0)
    )
    return (
        "episode={} step={} LATENCY step={:.0f}ms | encode={:.0f} ipc={:.0f} "
        "worker={:.0f} [rgb={:.0f} memory={:.0f} (captioner={:.0f} rules={:.0f}) "
        "depth={:.0f} select_pixel={:.0f} preview_select={:.0f} "
        "waypoint={:.0f} other={:.0f}] preview_render={:.0f} "
        "render={:.0f} env={:.0f}".format(
            episode_id, steps, step_ms,
            timings["encode_ms"],
            # Whatever the round trip spent outside the worker's own act() call.
            roundtrip_ms - worker_ms,
            worker_ms,
            timings.get("rgb_ms", 0.0),
            memory_ms, captioner_ms, memory_ms - captioner_ms,
            timings.get("depth_ms", 0.0), timings.get("select_pixel_ms", 0.0),
            timings.get("preview_select_ms", 0.0),
            timings.get("waypoint_ms", 0.0),
            # Non-zero here means act() spends time outside every named phase.
            worker_ms - accounted,
            timings["preview_render_ms"],
            render_ms, env_ms,
        )
    )


def _captioner_line(episode_id, steps, decision):
    """Report the Captioner's judgement beside its raw model text."""
    return (
        "episode={} step={} CAPTIONER ran={} history={} evidence={} "
        "evidence_ids={} completed={} error={} "
        "mode={} confidence={:.2f} evidence={!r} latency={:.0f}ms "
        "response={!r} analysis_error={!r}".format(
            episode_id, steps,
            decision.get("captioner_ran_this_step"),
            decision.get("temporal_frames"),
            decision.get("completion_evidence_frames"),
            decision.get("completion_frame_ids"),
            decision.get("captioner_completed"),
            decision.get("captioner_error"),
            decision.get("captioner_error_mode"),
            decision.get("captioner_error_confidence", 0.0),
            decision.get("captioner_error_evidence"),
            decision.get("captioner_latency_ms", 0.0),
            decision.get("captioner_raw_response"),
            decision.get("temporal_error"),
        )
    )


def _action_summary(decision, action):
    """Describe the executed action for one log line.

    Four decisions reach this point and only one of them has coordinates, so
    the pixel is read last rather than assumed.
    """
    if decision.get("stop"):
        return "STOP"
    if decision.get("turn_deg"):
        turn_deg = int(decision["turn_deg"])
        return "TURN {:+d}deg x{} a={}".format(
            turn_deg, abs(turn_deg) // TURN_ANGLE_DEG, int(action)
        )
    if decision.get("forward_steps"):
        return "FWD x{} a={}".format(int(decision["forward_steps"]), int(action))
    pixel = decision.get("pixel_uv")
    if pixel is None:
        return "PREVIEW a={}".format(int(action))
    return "({},{}) d={:.2f} a={}".format(
        pixel[0], pixel[1], decision.get("depth_m", 0.0), int(action)
    )


def _step_line(episode_id, steps, decision, step_ms, action):
    """One line per step: where the time went, memory state, and the action.

    ``--debug-memory`` adds the full per-memory dumps below this line.
    """
    timings = _combined_step_timings(decision)
    task = decision.get("task_memory") or {}
    temporal = decision.get("temporal_memory") or {}
    waypoint_ms = timings.get("waypoint_ms", 0.0)
    select_ms = timings.get("select_pixel_ms", 0.0)
    captioner_ms = timings.get("captioner_ms", 0.0)
    preview_select_ms = timings.get("preview_select_ms", 0.0)
    preview_render_ms = timings.get("preview_render_ms", 0.0)
    debug = decision.get("debug") or {}
    analyzed = debug.get("analyzed_subgoal") or {}
    analyzed_id = analyzed.get("subgoal_id")
    current_id = task.get("current_subgoal_id")
    line = (
        "ep={} s={} {:.0f}ms [wp={:.0f} sel={:.0f} cap={:.0f} "
        "pre={:.0f} rest={:.0f}] "
        "sg={}->{} mode={} win={} obs={} | cap={} act={}".format(
            episode_id, steps, step_ms,
            waypoint_ms, select_ms, captioner_ms, preview_select_ms,
            step_ms - waypoint_ms - select_ms - captioner_ms
            - preview_select_ms - preview_render_ms,
            analyzed_id, current_id,
            temporal.get("active_error_mode"),
            len(temporal.get("frame_ids") or ()),
            task.get("observation_count"),
            _caption_summary(decision),
            _action_summary(decision, action),
        )
    )
    preview = decision.get("preview") or {}
    if preview:
        selected = _previewed_view(decision) or {}
        line += " | preview=view{}/{} yaw={:+.0f} render={:.0f}ms".format(
            selected.get("view_index", "-"),
            len(preview.get("yaws_deg") or ()),
            float(selected.get("view_yaw_deg") or 0.0),
            preview_render_ms,
        )
    spatial = debug.get("spatial_summary")
    if spatial and spatial != "sp=-":
        line += " | " + spatial
    if debug.get("som_choice") is not None:
        line += " som={}/{}".format(
            debug.get("som_choice"), len(debug.get("som_candidates") or ())
        )
    # Surface only the abnormal cases inline; the rest stays behind the flag.
    if decision.get("temporal_error"):
        line += " ANALYSIS_ERROR={!r}".format(decision.get("temporal_error"))
    if debug.get("spatial_error"):
        line += " SPATIAL_ERROR={!r}".format(debug.get("spatial_error"))
    return line


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def _turn_primitive(turn_deg):
    """Split a requested turn into one primitive and a repeat count.

    Positive is to the right, matching ``yaw_delta_deg`` everywhere else. The
    request is rejected rather than rounded when it is not whole repeats of the
    simulator's turn angle: rounding would execute a smaller turn than the actor
    asked for and nothing downstream would notice.
    """
    turn_deg = int(turn_deg)
    if turn_deg == 0 or turn_deg % TURN_ANGLE_DEG:
        raise ValueError(
            "turn_deg={} is not a non-zero multiple of the simulator's "
            "turn_angle={}".format(turn_deg, TURN_ANGLE_DEG)
        )
    action = 3 if turn_deg > 0 else 2  # turn_right / turn_left
    return action, abs(turn_deg) // TURN_ANGLE_DEG


def _turn_frame(
    env,
    observation,
    decision,
    steps,
    args,
    navmesh_map,
    positions,
    goal_position,
    waypoint_targets,
    landmark_marks,
):
    """Render one intermediate frame of a multi-primitive turn.

    Without these the video would jump the whole turn at once, which reads as a
    teleport and hides how many steps the turn actually cost.
    """
    rgb, _ = _rgb_depth(observation)
    agent_map, _ = _decode_visuals(decision)
    debug_rgb = (
        _clean_video_frame(rgb)
        if args.clean_video
        else _annotated_video_frame(rgb, decision, steps)
    )
    if navmesh_map is None:
        return debug_rgb
    return _topdown_panel(
        debug_rgb,
        _render_topdown(
            env, navmesh_map, positions, goal_position, rgb.shape[0],
            waypoints=waypoint_targets,
            landmark_marks=landmark_marks,
        ),
        agent_map=agent_map,
    )


def _fallback_action_for_follower_stop(decision):
    """Choose a safe primitive when a nonterminal waypoint is unreachable."""
    debug = decision.get("debug") or {}
    recovery_mode = debug.get("recovery_mode")
    if recovery_mode in (
        "WALL_STUCK",
        "GET_NOWHERE",
        "NO_VALID_DEPTH",
    ):
        return 2  # HabitatSimActions.turn_left
    return 1  # HabitatSimActions.move_forward


def _navigation_debug_lines(
    episode_id,
    steps,
    decision,
    follower_action,
    action,
    position_before,
    position_after,
    distance_before,
    distance_after,
):
    """Explain one model-to-Habitat decision in reader-facing layers."""
    debug = decision.get("debug") or {}
    analyzed = debug.get("analyzed_subgoal") or {}
    before = debug.get("subgoal_before") or {}
    after = debug.get("subgoal_after") or {}
    moved = float(
        np.linalg.norm(
            np.asarray(position_after) - np.asarray(position_before)
        )
    )
    follower_value = (
        None if follower_action is None else int(follower_action)
    )
    forced_forward = (
        not decision.get("stop")
        and (follower_value is None or follower_value == 0)
        and int(action) == 1
    )
    return (
        "DEBUG_NAV episode={} step={}".format(episode_id, steps),
        "  STATE pos={} -> {} moved={:.3f}m dtg={:.2f} -> {:.2f}".format(
            np.asarray(position_before).round(3).tolist(),
            np.asarray(position_after).round(3).tolist(),
            moved,
            float(distance_before),
            float(distance_after),
        ),
        "  SUBGOAL analyzed={} before={} after={} transition={}".format(
            analyzed.get("subgoal_id"),
            before.get("subgoal_id"),
            after.get("subgoal_id"),
            debug.get("subgoal_transition"),
        ),
        "    instruction={!r}".format(analyzed.get("description")),
        "    completion_criteria={!r}".format(
            analyzed.get("completion_criteria")
        ),
        "  CAPTION completed={} history={} evidence={} ids={} raw={!r} "
        "error={!r} mode={} "
        "confidence={:.2f} evidence={!r}".format(
            decision.get("captioner_completed"),
            decision.get("temporal_frames"),
            decision.get("completion_evidence_frames"),
            decision.get("completion_frame_ids"),
            decision.get("captioner_raw_response"),
            decision.get("temporal_error"),
            decision.get("captioner_error_mode"),
            decision.get("captioner_error_confidence", 0.0),
            decision.get("captioner_error_evidence"),
        ),
        "  LANDMARK state={} raw={!r} error={!r}".format(
            debug.get("landmark"),
            debug.get("landmark_raw_response"),
            debug.get("landmark_error"),
        ),
        "  BEHAVIOR recent={}".format(
            (debug.get("behavior_history") or [])[-3:]
        ),
        "  PREVIEW requested={} headings={} selected_view={} yaw={} "
        "selection={} guard={!r}".format(
            bool(decision.get("preview")),
            (decision.get("preview") or {}).get("yaws_deg"),
            debug.get("preview_view_index"),
            debug.get("preview_yaw_deg"),
            debug.get("preview_selection"),
            debug.get("preview_guard_reason"),
        ),
        "  WAYPOINT phase={} heading_lock={} model_intent={} "
        "applied_intent={} confidence={} guard={!r} evidence={!r} raw={!r} "
        "normalized={} requested={} validated={} depth={} "
        "world={} error_candidate={} guard={!r} recovery={} "
        "stop_disposition={} stop_reason={}".format(
            debug.get("navigation_phase"),
            debug.get("corridor_heading_yaw_deg"),
            debug.get("waypoint_model_intent"),
            debug.get("waypoint_applied_intent"),
            debug.get("waypoint_confidence"),
            debug.get("waypoint_guard_reason"),
            debug.get("waypoint_evidence"),
            debug.get("waypoint_raw_response"),
            debug.get("requested_normalized_uv"),
            debug.get("requested_pixel_uv"),
            decision.get("pixel_uv"),
            decision.get("depth_m"),
            decision.get("world_xyz"),
            debug.get("error_candidate"),
            debug.get("error_guard_reason"),
            debug.get("recovery_mode"),
            debug.get("waypoint_stop_disposition"),
            debug.get("stop_reason"),
        ),
        "  SPATIAL {} target={} som_choice={} candidates={}".format(
            debug.get("spatial_summary"),
            ((decision.get("spatial_memory") or {}).get("target") or {}).get("world_xyz"),
            debug.get("som_choice"),
            [
                (c.get("label"), c.get("kind"), c.get("distance_m"), c.get("bearing_deg"), c.get("world_xyz"))
                for c in (debug.get("som_candidates") or ())
            ],
        ),
        "  CONTROL follower={} forced_forward={} habitat_action={}".format(
            (
                "NONE"
                if follower_value is None
                else ACTION_NAMES.get(follower_value, follower_value)
            ),
            forced_forward,
            ACTION_NAMES.get(int(action), int(action)),
        ),
    )


def _caption_summary(decision):
    """Condense this step's Captioner verdict, or mark that it did not run."""
    if not decision.get("captioner_ran_this_step"):
        return "-"
    return "{}{}".format(
        "DONE" if decision.get("captioner_completed") else "wip",
        "" if decision.get("captioner_error_mode") == "NONE"
        else "/" + str(decision.get("captioner_error_mode")),
    )


def _task_memory_line(episode_id, steps, decision):
    """Report the Task Memory state that waypoint selection reads this step."""
    state = decision.get("task_memory") or {}
    return (
        "episode={} step={} TASK_MEMORY subgoal={} status={!r} temporal={!r} "
        "observations={} observation={!r} events={} temporal_events={}".format(
            episode_id, steps,
            state.get("current_subgoal_id"),
            state.get("subgoal_completion_status"),
            state.get("temporal_status"),
            state.get("observation_count"),
            state.get("latest_observation"),
            state.get("events"),
            state.get("temporal_events"),
        )
    )


def _temporal_memory_line(episode_id, steps, decision):
    """Report the Temporal Memory window and its latest stored analysis."""
    state = decision.get("temporal_memory") or {}
    return (
        "episode={} step={} TEMPORAL_MEMORY subgoal={} frames={} "
        "active_error_mode={} pending_events={} latest_result={} "
        "analysis_error={!r}".format(
            episode_id, steps,
            state.get("current_subgoal_id"),
            state.get("frame_ids"),
            state.get("active_error_mode"),
            state.get("pending_events"),
            state.get("latest_result"),
            state.get("last_analysis_error"),
        )
    )


METRIC_NAMES = ("success", "spl", "distance_to_goal")


def _empty_totals():
    return {name: 0.0 for name in METRIC_NAMES}


def _write_rank_summary(output_dir, rank, count, totals):
    """Emit this shard's totals in the layout aggregate_r2r_ce_results.py reads."""
    result = {"rank": rank, "count": count, "totals": totals}
    print("rank_summary={}".format(json.dumps(result, sort_keys=True)), flush=True)
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rank_{}.json".format(rank)).open("w") as handle:
        json.dump(result, handle, sort_keys=True)


def _parse_episode_ids(spec):
    """``"1,2,3"`` or ``"@file"`` -> list of id strings, in order."""
    if not spec:
        return None
    if spec.startswith("@"):
        lines = Path(spec[1:]).read_text().splitlines()
        items = [line.split("#", 1)[0].strip() for line in lines]
        items = [item.split(",", 1)[0].strip() for item in items if item]
    else:
        items = [item.strip() for item in spec.split(",") if item.strip()]
    return items or None


def _select_episodes(
    available,
    *,
    episode_id,
    episode_count,
    rank,
    world_size,
    episode_ids=None,
):
    """Select one exact episode, an explicit id list, or the eval prefix.

    ``episode_ids`` keeps the list's own order so a stratified evaluation
    set is sharded evenly across ranks category by category.
    """
    available = list(available)
    if episode_ids:
        by_id = {str(episode.episode_id): episode for episode in available}
        missing = [item for item in episode_ids if item not in by_id]
        if missing:
            raise ValueError(
                "episode ids not present in this split/scene: {}".format(
                    ", ".join(missing)
                )
            )
        selected = [by_id[item] for item in episode_ids]
    elif episode_id is not None:
        selected = [
            episode
            for episode in available
            if str(episode.episode_id) == str(episode_id)
        ]
        if not selected:
            raise ValueError(
                "episode_id {} is not present in this split/scene".format(
                    episode_id
                )
            )
    else:
        selected = (
            available
            if episode_count == 0
            else available[:episode_count]
        )
    return selected[rank::world_size]


def main():
    parser = argparse.ArgumentParser(description="Run RGB-D waypoint Actor on Habitat R2R-CE")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--scene-id", default="all", help="Restrict evaluation to one MP3D scene.")
    parser.add_argument("--episodes", type=int, default=1, help="0 = all episodes in the split.")
    parser.add_argument(
        "--episode-id",
        help="Run exactly one episode ID; overrides --episodes.",
    )
    parser.add_argument(
        "--episode-ids",
        help="Comma-separated episode IDs, or @path to a file with one ID per "
        "line (an optional ',category' suffix and # comments are ignored); "
        "overrides --episodes and --episode-id.",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0, help="This process's shard index.")
    parser.add_argument("--world-size", type=int, default=1, help="Number of evaluation processes.")
    parser.add_argument("--output-dir", type=Path, help="Write this rank's totals for later aggregation.")
    parser.add_argument(
        "--model-path",
        default=str(
            AGENTFLOW_ROOT / "models" / "JoyAI-VL-Interaction"
        ),
    )
    parser.add_argument("--actor-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--actor", choices=("waypoint", "awarevln"), default="waypoint",
        help="awarevln: drive Habitat with the AwareVLN policy served by integrations/v3/serve_awarevln.py "
        "(native forward/turn/stop protocol, 512x512 RGB like its reference evaluation).",
    )
    parser.add_argument("--awarevln-url", default="http://127.0.0.1:8600/v1")
    parser.add_argument("--awarevln-model", default="awarevln")
    parser.add_argument("--waypoint-radius", type=float, default=0.25)
    parser.add_argument("--depth-hfov", type=float, default=90.0)
    parser.add_argument(
        "--preview-yaws",
        type=str,
        default="-90,-45,0,45,90",
        help=(
            "Comma-separated heading offsets in degrees rendered for a "
            "PREVIEW decision; negative is left and positive is right. The "
            "forward view is "
            "rendered through the same path so all views share one scale."
        ),
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help=(
            "Downscale factor applied to preview views before they cross the "
            "pipe. The VLM resizes them to its own budget anyway."
        ),
    )
    parser.add_argument("--som-oracle", action="store_true", help="DIAGNOSTIC: give the worker the goal position so set-of-mark picks the geometrically best marker instead of asking the model.")
    parser.add_argument("--debug-memory", action="store_true", help="Dump both memories and the full latency breakdown under each step.")
    parser.add_argument(
        "--debug-navigation",
        action="store_true",
        help="Explain position, subgoal, Captioner, waypoint, and control decisions every step.",
    )
    parser.add_argument("--record-video", action="store_true", help="Save RGB plus top-down trajectory MP4s.")
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument(
        "--clean-video",
        action="store_true",
        help="Record unannotated RGB instead of overlaying the agent's waypoint pixels.",
    )
    args = parser.parse_args()
    # Resolve the id list now: the evaluator chdirs into the Habitat root
    # before the episodes are selected, which breaks a relative @file path.
    episode_ids = _parse_episode_ids(args.episode_ids)
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be in [0, --world-size).")
    try:
        args.preview_yaws = tuple(
            float(value)
            for value in args.preview_yaws.split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--preview-yaws must be comma-separated numbers.")
    if not args.preview_yaws:
        parser.error("--preview-yaws must name at least one heading.")
    if not 0 < args.preview_scale <= 1.0:
        parser.error("--preview-scale must be in (0, 1].")
    if args.output_dir is not None:
        # Habitat changes the working directory below, so pin this now.
        args.output_dir = args.output_dir.expanduser().resolve()

    overrides = R2R_CE_OVERRIDES + DEPTH_SENSOR_OVERRIDES + [
        "habitat.dataset.split={}".format(args.split),
        "habitat.dataset.data_path='{}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz'".format(HABITAT_DATA),
        "habitat.dataset.scenes_dir={}/scene_datasets".format(HABITAT_DATA),
        "habitat.environment.max_episode_steps={}".format(args.max_steps),
    ]
    if args.scene_id != "all":
        overrides.append("habitat.dataset.content_scenes=[{}]".format(args.scene_id))
    if args.actor == "awarevln":
        # AwareVLN was evaluated on square 512x512 frames; keep depth aligned.
        overrides += [
            "habitat.simulator.agents.main_agent.sim_sensors.{}_sensor.{}=512".format(sensor, side)
            for sensor in ("rgb", "depth") for side in ("height", "width")
        ]
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
    if args.actor == "awarevln":
        from awarevln_actor import AwareVLNActor

        actor = AwareVLNActor(args.awarevln_url, args.awarevln_model)
    else:
        actor = WaypointActorProcess(args.actor_python, ROOT / "integrations/v3/vln_waypoint_worker.py", args.model_path, args.gpu_id)
    actor.want_visuals = bool(args.record_video)
    if args.record_video:
        # Habitat changes the working directory below; keep media paths pinned
        # to the directory from which this runner was launched.
        args.video_dir = args.video_dir.expanduser().resolve()
        from habitat.utils.visualizations.utils import images_to_video
        args.video_dir.mkdir(parents=True, exist_ok=True)
        (args.video_dir / "topdown").mkdir(exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(HABITAT_ROOT)
        with habitat.Env(config=config) as env:
            episodes = _select_episodes(
                env.episodes,
                episode_id=args.episode_id,
                episode_count=args.episodes,
                rank=args.rank,
                world_size=args.world_size,
                episode_ids=episode_ids,
            )
            env.episodes = episodes
            if not episodes:
                print("rank={} has no episodes".format(args.rank), flush=True)
                _write_rank_summary(args.output_dir, args.rank, 0, _empty_totals())
                return
            totals = _empty_totals()
            for index, episode in enumerate(episodes, start=1):
                observation = env.reset()
                # Habitat owns episode ordering; trust the environment over the
                # list index so logged IDs and goals match the active episode.
                episode = env.current_episode
                steps = 0
                _, _, instruction = _observation(observation)
                print(
                    "episode={} instruction={!r} start_position={} goal_position={} "
                    "reference_geodesic={}".format(
                        episode.episode_id,
                        instruction,
                        env.sim.get_agent_state().position.tolist(),
                        (
                            episode.goals[0].position
                            if episode.goals
                            else None
                        ),
                        (episode.info or {}).get("geodesic_distance"),
                    ),
                    flush=True,
                )
                preparation = actor.prepare(instruction)
                subgoals = preparation.get("subgoals", [])
                # Printed once per episode, so each subgoal gets its own lines:
                # the completion criteria is what the Captioner judges against.
                print(
                    "episode={} prepared_subgoals={}".format(
                        episode.episode_id, len(subgoals)
                    ),
                    flush=True,
                )
                for subgoal in subgoals:
                    print(
                        "  [{}] {}\n      proof: {}".format(
                            subgoal.get("subgoal_id"),
                            subgoal.get("description"),
                            subgoal.get("completion_criteria"),
                        ),
                        flush=True,
                    )
                frames = [] if args.record_video else None
                navmesh_map = None
                if frames is not None:
                    try:
                        navmesh_map = _build_navmesh_map(env)
                    except Exception as exc:
                        print(
                            "Top-down video fallback to RGB: {}: {}".format(
                                type(exc).__name__, exc
                            ),
                            flush=True,
                        )
                positions = [env.sim.get_agent_state().position.copy()]
                goal_position = episode.goals[0].position if episode.goals else None
                last_agent_map = None
                # Requested waypoints and landmark events accumulate over the
                # episode so the top-down map shows the whole intended route
                # beside the executed one.
                waypoint_targets = []
                landmark_marks = []
                previous_landmark_mark = None
                follower = ShortestPathFollower(
                    env.sim, args.waypoint_radius, return_one_hot=False
                )
                step_started = time.perf_counter()
                while not env.episode_over:
                    position_before = (
                        env.sim.get_agent_state().position.copy()
                    )
                    distance_before = float(
                        env.get_metrics().get("distance_to_goal", 0.0)
                    )
                    rgb, depth, instruction = _observation(observation)
                    intrinsics = _intrinsics(rgb.shape[1], rgb.shape[0], args.depth_hfov)
                    waypoint, decision = actor.act(
                        rgb, depth, instruction, intrinsics, _camera_to_world(env),
                        navigable=_navigable_window(env),
                        oracle_goal=(goal_position if args.som_oracle else None),
                    )
                    if decision.get("action_mode") == "PREVIEW":
                        # The actor asked to look around before committing.
                        # Rendering is not a simulator step, so this costs the
                        # episode nothing but one extra model call.
                        preview_started = time.perf_counter()
                        views = _preview_views(
                            env,
                            args.preview_yaws,
                            args.depth_hfov,
                            args.preview_scale,
                        )
                        preview_render_ms = (
                            time.perf_counter() - preview_started
                        ) * 1000
                        preview_request = decision
                        waypoint, decision = actor.act_on_preview(
                            views, instruction
                        )
                        decision["preview"] = {
                            "render_ms": preview_render_ms,
                            "yaws_deg": [view["yaw_deg"] for view in views],
                            "act_timings": preview_request.get("timings") or {},
                            "act_encode_ms": preview_request.get(
                                "encode_ms", 0.0
                            ),
                            "act_roundtrip_ms": preview_request.get(
                                "roundtrip_ms", 0.0
                            ),
                            "requested_by": preview_request.get(
                                "decision"
                            ),
                        }
                    if waypoint is not None:
                        waypoint_targets.append(
                            np.asarray(waypoint, dtype=np.float64)
                        )
                    landmark_mark = _landmark_mark_kind(
                        _landmark_state(decision)
                    )
                    # Only the transition is marked: the tracker holds AT or
                    # passed for several consecutive steps, and one marker per
                    # step would bury the map.
                    if (
                        landmark_mark is not None
                        and landmark_mark != previous_landmark_mark
                    ):
                        landmark_marks.append(
                            (position_before.copy(), landmark_mark)
                        )
                    previous_landmark_mark = landmark_mark
                    render_started = time.perf_counter()
                    agent_map, marker_frame = _decode_visuals(decision)
                    last_agent_map = agent_map if agent_map is not None else last_agent_map
                    # The frame the model chose from, markers included, when
                    # this step asked the set-of-mark question.
                    shown_rgb = (
                        marker_frame
                        if marker_frame is not None and marker_frame.shape == rgb.shape
                        else rgb
                    )
                    debug_rgb = (
                        _clean_video_frame(shown_rgb)
                        if args.clean_video
                        else _annotated_video_frame(shown_rgb, decision, steps)
                    )
                    if frames is not None:
                        if navmesh_map is None:
                            frames.append(debug_rgb)
                        else:
                            frames.append(_topdown_panel(
                                debug_rgb, _render_topdown(
                                    env, navmesh_map, positions, goal_position,
                                    rgb.shape[0],
                                    waypoints=waypoint_targets,
                                    landmark_marks=landmark_marks,
                                ),
                                agent_map=last_agent_map,
                            ))
                    render_ms = (time.perf_counter() - render_started) * 1000
                    follower_action = None
                    if decision.get("stop"):
                        action = 0  # HabitatSimActions.stop, chosen by Actor.
                        repeats = 1
                    elif decision.get("turn_deg"):
                        action, repeats = _turn_primitive(
                            decision["turn_deg"]
                        )
                    elif decision.get("forward_steps"):
                        # A discrete policy (AwareVLN) asked for N x 25 cm.
                        action = 1  # HabitatSimActions.move_forward
                        repeats = max(1, int(decision["forward_steps"]))
                    elif waypoint is None:
                        # No waypoint, no turn and no stop: the previewed
                        # heading had no valid depth, or a PREVIEW went
                        # unanswered. Turning in place keeps the episode alive
                        # instead of handing the follower a None target.
                        action = 2  # HabitatSimActions.turn_left
                        repeats = 1
                    else:
                        repeats = 1
                        follower_action = follower.get_next_action(waypoint)
                        if follower_action is None or int(follower_action) == 0:
                            # STOP is reserved exclusively for the Actor. A
                            # local follower can emit STOP for a nearby or
                            # unreachable waypoint, but that is not task end.
                            # During lateral stuck recovery, forcing forward
                            # repeats the collision that recovery is meant to
                            # escape; execute a stable turn primitive instead.
                            action = _fallback_action_for_follower_stop(
                                decision
                            )
                        else:
                            action = follower_action
                    env_started = time.perf_counter()
                    # A requested turn is several primitives. Every one of them
                    # is a real simulator step against the episode budget, so
                    # they are counted and recorded individually rather than
                    # collapsed into the single step the actor was consulted on.
                    executed = 0
                    for repeat in range(repeats):
                        # A turn can reach the episode's step limit partway
                        # through; stop issuing primitives rather than stepping
                        # an environment that has already finished.
                        if repeat and env.episode_over:
                            break
                        observation = env.step({"action": action})
                        executed += 1
                        positions.append(
                            env.sim.get_agent_state().position.copy()
                        )
                        if repeat < repeats - 1 and hasattr(actor, "observe"):
                            # Queued primitives are steps the policy was not
                            # asked about; it still sees their frames.
                            actor.observe(_observation(observation)[0])
                        if frames is not None and repeat:
                            # The frame recorded before the loop already covers
                            # the first primitive, so this starts at the second
                            # and the video stays one frame per executed step.
                            frames.append(_turn_frame(
                                env, observation, decision, steps + repeat,
                                args, navmesh_map, positions, goal_position,
                                waypoint_targets, landmark_marks,
                            ))
                    env_ms = (time.perf_counter() - env_started) * 1000
                    position_after = (
                        env.sim.get_agent_state().position.copy()
                    )
                    distance_after = float(
                        env.get_metrics().get("distance_to_goal", 0.0)
                    )
                    now = time.perf_counter()
                    step_ms = (now - step_started) * 1000
                    print(
                        _step_line(episode.episode_id, steps, decision, step_ms, action)
                        + " dtg={:.2f}".format(distance_after),
                        flush=True,
                    )
                    if args.debug_memory:
                        for line in (
                            "episode={} step={} follower_action={} model_response={!r}".format(
                                episode.episode_id, steps,
                                None if decision.get("stop") else follower_action,
                                decision.get("raw_model_response"),
                            ),
                            _task_memory_line(episode.episode_id, steps, decision),
                            _temporal_memory_line(episode.episode_id, steps, decision),
                            _captioner_line(episode.episode_id, steps, decision),
                            _latency_line(
                                episode.episode_id, steps, decision,
                                step_ms, render_ms, env_ms,
                            ),
                        ):
                            print("  " + line, flush=True)
                    if args.debug_navigation:
                        for line in _navigation_debug_lines(
                            episode.episode_id,
                            steps,
                            decision,
                            follower_action,
                            action,
                            position_before,
                            position_after,
                            distance_before,
                            distance_after,
                        ):
                            print(line, flush=True)
                    step_started = now
                    # Counted from what the loop actually executed: a turn is
                    # several steps, and an early break makes it fewer than
                    # were asked for.
                    steps += executed
                metrics = env.get_metrics()
                for name in totals:
                    totals[name] += float(metrics.get(name, 0.0))
                print("rank={} [{}/{}] id={} steps={} success={:.3f} spl={:.3f} dtg={:.2f}".format(args.rank, index, len(episodes), episode.episode_id, steps, float(metrics.get("success", 0)), float(metrics.get("spl", 0)), float(metrics.get("distance_to_goal", 0))), flush=True)
                if frames:
                    rgb, _, _ = _observation(observation)
                    if navmesh_map is None:
                        frames.append(rgb.copy())
                    else:
                        frames.append(_topdown_panel(
                            rgb, _render_topdown(
                                env, navmesh_map, positions, goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                landmark_marks=landmark_marks,
                            ),
                            agent_map=last_agent_map,
                        ))
                    episode_id = str(episode.episode_id).replace("/", "_")
                    images_to_video(frames, str(args.video_dir), episode_id, fps=10)
                    if navmesh_map is not None:
                        Image.fromarray(
                            _render_topdown(
                                env, navmesh_map, positions, goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                landmark_marks=landmark_marks,
                            )
                        ).save(str(args.video_dir / "topdown" / (episode_id + ".png")))
            _write_rank_summary(args.output_dir, args.rank, len(episodes), totals)
    finally:
        actor.close()
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
