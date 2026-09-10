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
from integrations.v3.preview_protocol import preview_headings_for_request, preview_for_unseen_frame, execution_observation
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

    def __init__(self, python, worker, model_path, gpu_id=None, timeout=600, camera_height_m=SENSOR_HEIGHT_M,
                 base_url=None, evidence_dir=None):
        command = [
            str(python), str(worker), "--model-path", str(model_path),
            "--camera-height-m", repr(float(camera_height_m)),
        ]
        if base_url:
            command += ["--base-url", str(base_url)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(AGENTFLOW_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        if evidence_dir is not None:
            # Recording a debug/video evaluation must retain the original
            # images used by its auditable decisions, not just an overlay MP4.
            # Respect an explicitly configured location (including opt-out "").
            environment.setdefault("JOYAI_EVIDENCE_DIR", str(Path(evidence_dir).resolve()))
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

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world, navigable=None, oracle_goal=None, cwp_candidates=None, temporal_observed=False, preview_views=(), preview_request_id="", previous_execution=None):
        encode_started = time.perf_counter()
        request = {
            "operation": "act",
            "rgb": self._png(rgb), "depth": self._array(depth),
            "instruction": instruction, "intrinsics": np.asarray(intrinsics).tolist(),
            "camera_to_world": np.asarray(camera_to_world).tolist(),
            # Ask for the agent's own map and marker frame only when a video
            # is being recorded: they cost a PNG encode per step.
            "want_visuals": bool(getattr(self, "want_visuals", False)),
            "temporal_observed": bool(temporal_observed),
            "previous_execution": previous_execution,
        }
        if preview_views:
            request["preview_request_id"] = preview_request_id
            request["preview_views"] = [
                {
                    "yaw_deg": v["yaw_deg"], "rgb": self._png(v["rgb"]),
                    "depth": self._array(v["depth"]),
                    "intrinsics": np.asarray(v["intrinsics"]).tolist(),
                    "camera_to_world": np.asarray(v["camera_to_world"]).tolist(),
                } for v in preview_views
            ]
        if cwp_candidates is not None:
            # Externally supplied (runner-side CWP) waypoint candidates; the
            # worker substitutes them for its floor-openings generator.
            request["cwp_candidates"] = cwp_candidates
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
            if navigable.get("height_m") is not None:
                request["navigable"]["height_m"] = self._array(navigable["height_m"])
                request["navigable"]["height_cell_m"] = float(navigable.get("height_cell_m", 0.0))
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

    def act_on_preview(self, views, instruction, cwp_candidates=None):
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
        if cwp_candidates is not None:
            request["cwp_candidates"] = cwp_candidates
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

    def observe(self, rgb, camera_to_world, *, depth=None, intrinsics=None):
        """Send one queued simulator primitive to Temporal Memory only."""
        return self._request(
            {
                "operation": "observe",
                "rgb": self._png(rgb),
                "camera_to_world": np.asarray(camera_to_world).tolist(),
                **({"depth": self._array(depth), "intrinsics": np.asarray(intrinsics).tolist()}
                   if depth is not None and intrinsics is not None else {}),
            }
        )

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


def _semantic_region_id(env, cache={}):
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


def _navigable_window(env, radius_m=6.0, resolution_m=0.25, *, include_heights=False):
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


def _floor_map_key(height):
    """Stable half-metre cache key; floors remain separate while stairs blend."""
    return round(float(height) * 2.0) / 2.0


def _build_navmesh_map(env, resolution=1024, height=None):
    """Render one floor slice of the scene navmesh."""
    from habitat.utils.visualizations.maps import get_topdown_map

    if height is None:
        height = env.sim.get_agent_state().position[1]
    return get_topdown_map(
        env.sim.pathfinder,
        float(height),
        map_resolution=resolution,
        draw_border=True,
    )


def _navmesh_map_for_height(env, cache, height):
    """Return the current floor slice without rebuilding it every video frame."""
    key = _floor_map_key(height)
    if key not in cache:
        cache[key] = _build_navmesh_map(env, height=height)
    return key, cache[key]


def _render_topdown(
    env,
    navmesh_map,
    positions,
    goal_position,
    output_height,
    waypoints=(),
    landmark_marks=(),
    floor_height=None,
):
    """Draw the executed trajectory, start, current position, and goal.

    ``waypoints`` are the world-space targets the actor asked for, which show
    where it intended to go as opposed to where the follower took it.
    ``landmark_marks`` are ``(position, kind)`` pairs recording where the
    tracker reported standing at or crossing the active subgoal's landmark.
    """
    import cv2
    from habitat.utils.visualizations import maps

    if floor_height is None:
        floor_height = float(env.sim.get_agent_state().position[1])
    floor_tolerance = 0.75
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

    floor_positions = [
        position for position in positions
        if abs(float(position[1]) - floor_height) <= floor_tolerance
    ]
    floor_waypoints = [
        waypoint for waypoint in waypoints
        if abs(float(waypoint[1]) - floor_height) <= floor_tolerance
    ]
    floor_landmarks = [
        (position, kind) for position, kind in landmark_marks
        if abs(float(position[1]) - floor_height) <= floor_tolerance
    ]
    path = [to_pixel(position) for position in floor_positions]
    if len(path) > 1:
        cv2.polylines(image, [np.asarray(path, dtype=np.int32)], False, (0, 80, 255), 3)
    # Drawn under the trajectory endpoints so the executed path stays legible.
    for waypoint in floor_waypoints:
        cv2.circle(image, to_pixel(waypoint), 3, _REQUESTED_COLOR, -1, cv2.LINE_AA)
    if floor_waypoints and path:
        cv2.line(
            image, path[-1], to_pixel(floor_waypoints[-1]), _REQUESTED_COLOR, 1,
            cv2.LINE_AA,
        )
    for position, kind in floor_landmarks:
        cv2.drawMarker(
            image, to_pixel(position),
            (230, 80, 230) if kind == "PASSED" else _LANDMARK_COLORS["AT"],
            cv2.MARKER_DIAMOND if kind == "PASSED" else cv2.MARKER_TRIANGLE_UP,
            14, 2,
        )
    if path:
        cv2.circle(image, path[0], 7, (0, 180, 0), -1)
        cv2.circle(image, path[-1], 7, (255, 80, 0), -1)
    if (
        goal_position is not None
        and abs(float(goal_position[1]) - floor_height) <= floor_tolerance
    ):
        cv2.drawMarker(image, to_pixel(goal_position), (255, 0, 0), cv2.MARKER_STAR, 16, 2)

    # Habitat maps use the full scene bounds, which can leave the active floor
    # as a tiny island in a large blank canvas. Crop to the valid slice, then
    # resize to a fixed square so every MP4 frame keeps identical dimensions.
    valid_rows, valid_cols = np.where(navmesh_map != 0)
    if valid_rows.size and valid_cols.size:
        margin = max(8, int(0.02 * max(navmesh_map.shape)))
        row0 = max(0, int(valid_rows.min()) - margin)
        row1 = min(rows, int(valid_rows.max()) + margin + 1)
        col0 = max(0, int(valid_cols.min()) - margin)
        col1 = min(columns, int(valid_cols.max()) + margin + 1)
        image = image[row0:row1, col0:col1]
    image = cv2.resize(
        image, (output_height, output_height), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        image,
        "floor y={:+.2f}m".format(float(floor_height)),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return image


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
    return min(strip_height, image.shape[0])


CHAIN_DONE = (60, 180, 75)      # completed stage
CHAIN_ACTIVE = (250, 200, 40)   # active stage
CHAIN_TODO = (90, 90, 90)       # not reached
CHAIN_STUCK = (230, 90, 40)     # active stage held longer than CHAIN_STUCK_STEPS
CHAIN_STUCK_STEPS = 30


def new_subgoal_chain(subgoals):
    """Per-episode tracker for the subgoal-chain strip drawn on video frames."""
    return {
        "ids": [str(item.get("subgoal_id")) for item in subgoals],
        "stage_id": None,
        "entered_step": 0,
        "completed": [],
        "transition": False,
    }


def update_subgoal_chain(chain, decision, steps):
    """Advance the tracker from this step's decision; returns the tracker."""
    if chain is None:
        return None
    debug = decision.get("debug") or {}
    task = decision.get("task_memory") or {}
    after = debug.get("subgoal_after") or {}
    stage_id = after.get("subgoal_id")
    if stage_id is None:
        stage_id = task.get("current_subgoal_id")
    stage_id = None if stage_id is None else str(stage_id)
    chain["transition"] = False
    if stage_id != chain["stage_id"]:
        if chain["stage_id"] is not None:
            chain["transition"] = True
            # Every stage strictly before the new one counts as passed; the
            # planner may skip ahead (skip_to_final) over several stages.
            ids = chain["ids"]
            stop = ids.index(stage_id) if stage_id in ids else len(ids)
            for item in ids[:stop]:
                if item not in chain["completed"]:
                    chain["completed"].append(item)
        chain["stage_id"] = stage_id
        chain["entered_step"] = steps
    if decision.get("stop") and debug.get("stop_reason") == "ALL_SUBGOALS_COMPLETE":
        chain["completed"] = list(chain["ids"])
        chain["stage_id"] = None
    return chain


def _draw_subgoal_chain(image, decision, chain, steps, y_top):
    """One box per planned stage under the instruction strip.

    Completed stages are green, the active one yellow (orange once it has
    been held for CHAIN_STUCK_STEPS), unreached ones grey. The active box
    shows how many steps the agent has spent in it. A dot to the right
    reports the Captioner this step: grey ran/in-progress, green accepted
    completion, red completion claimed but rejected. The frame on which a
    stage advanced gets a yellow border so it is easy to find when scrubbing.
    """
    import cv2

    if not chain or not chain["ids"]:
        return y_top
    ids = chain["ids"]
    height, width = image.shape[:2]
    margin, gap, box_h = 8, 4, 20
    dot_room = 22
    usable = width - 2 * margin - dot_room
    box_w = max(min(90, (usable - gap * (len(ids) - 1)) // len(ids)), 14)
    y0 = y_top + 3
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y_top), (width - 1, y0 + box_h + 3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, dst=image)
    held = steps - chain["entered_step"]
    x = margin
    for item in ids:
        if item in chain["completed"]:
            color, label = CHAIN_DONE, item
        elif item == chain["stage_id"]:
            color = CHAIN_STUCK if held >= CHAIN_STUCK_STEPS else CHAIN_ACTIVE
            label = "{} +{}".format(item, held)
        else:
            color, label = CHAIN_TODO, item
        cv2.rectangle(image, (x, y0), (x + box_w, y0 + box_h), color, -1)
        cv2.putText(
            image, label, (x + 4, y0 + box_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 0, 0), 1, cv2.LINE_AA,
        )
        x += box_w + gap
    if decision.get("captioner_ran_this_step"):
        if decision.get("captioner_completed"):
            dot = CHAIN_DONE
        elif decision.get("captioner_model_completed_raw"):
            dot = (220, 40, 40)
        else:
            dot = (170, 170, 170)
        cv2.circle(image, (width - margin - 7, y0 + box_h // 2), 6, dot, -1)
    if chain.get("transition"):
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), CHAIN_ACTIVE, 4)
    return y0 + box_h + 3


def _annotated_video_frame(rgb, decision, steps, chain=None):
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
    strip_bottom = _draw_task_context(image, decision)
    _draw_subgoal_chain(image, decision, chain, steps, strip_bottom)

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
        "window_ids={} eligible_ids={} evidence_ids={} evidence_paths={} "
        "raw_completed={} completed={} evidence_valid={} rejection={!r} "
        "completion_confidence={:.2f} completion_evidence={!r} error={} "
        "mode={} error_confidence={:.2f} error_evidence={!r} latency={:.0f}ms "
        "response={!r} analysis_error={!r}".format(
            episode_id, steps,
            decision.get("captioner_ran_this_step"),
            decision.get("temporal_frames"),
            decision.get("completion_evidence_frames"),
            decision.get("completion_frame_ids"),
            decision.get("completion_eligible_frame_ids"),
            decision.get("completion_evidence_frame_ids"),
            decision.get("completion_evidence_frame_paths"),
            decision.get("captioner_model_completed_raw"),
            decision.get("captioner_completed"),
            decision.get("captioner_completion_evidence_valid"),
            decision.get("captioner_completion_rejection_reason"),
            decision.get("captioner_completion_confidence", 0.0),
            decision.get("captioner_completion_evidence"),
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
        return "TURN request={:+d}deg execute={:+d}deg x1 a={}".format(
            turn_deg,
            TURN_ANGLE_DEG if turn_deg > 0 else -TURN_ANGLE_DEG,
            int(action),
        )
    if decision.get("forward_steps"):
        return "FWD request=x{} execute=x1 a={}".format(
            int(decision["forward_steps"]), int(action)
        )
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
        line += " ANALYSIS_ERROR_STAGE={} ANALYSIS_ERROR={!r}".format(
            decision.get("captioner_failed_stage") or "unknown",
            decision.get("temporal_error"),
        )
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
    """Resolve a requested turn to exactly one simulator primitive.

    Positive is to the right, matching ``yaw_delta_deg`` everywhere else. The
    magnitude is validated against the simulator angle, but is deliberately not
    executed open-loop.  The next 15-degree observation must pass through the
    Actor and TemporalMemory before another primitive can be issued.
    """
    turn_deg = int(turn_deg)
    if turn_deg == 0 or turn_deg % TURN_ANGLE_DEG:
        raise ValueError(
            "turn_deg={} is not a non-zero multiple of the simulator's "
            "turn_angle={}".format(turn_deg, TURN_ANGLE_DEG)
        )
    action = 3 if turn_deg > 0 else 2  # turn_right / turn_left
    return action, 1


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
            floor_height=env.sim.get_agent_state().position[1],
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
        "  CAPTION raw_completed={} completed={} valid={} rejection={!r} "
        "confidence={:.2f} completion_evidence={!r} history={} evidence={} "
        "window_ids={} eligible_ids={} evidence_ids={} evidence_paths={} "
        "raw={!r} error={!r} mode={} error_confidence={:.2f} "
        "error_evidence={!r}".format(
            decision.get("captioner_model_completed_raw"),
            decision.get("captioner_completed"),
            decision.get("captioner_completion_evidence_valid"),
            decision.get("captioner_completion_rejection_reason"),
            decision.get("captioner_completion_confidence", 0.0),
            decision.get("captioner_completion_evidence"),
            decision.get("temporal_frames"),
            decision.get("completion_evidence_frames"),
            decision.get("completion_frame_ids"),
            decision.get("completion_eligible_frame_ids"),
            decision.get("completion_evidence_frame_ids"),
            decision.get("completion_evidence_frame_paths"),
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


def _write_step_trace(
    handle,
    *,
    episode_id,
    step,
    decision,
    action,
    executed,
    position_before,
    position_after,
    distance_before,
    distance_after,
):
    """Persist the complete non-image decision while the episode is running."""
    if handle is None:
        return
    traced_decision = dict(decision or {})
    # Base64 visualization panels can be reconstructed from the MP4 and make
    # a long JSONL trace unnecessarily huge. All model/memory evidence stays.
    traced_decision.pop("visuals", None)
    payload = {
        "episode_id": str(episode_id),
        "step": int(step),
        "action": int(action),
        "executed_primitives": int(executed),
        "position_before": [float(value) for value in position_before],
        "position_after": [float(value) for value in position_after],
        "distance_to_goal_before": float(distance_before),
        "distance_to_goal_after": float(distance_after),
        "decision": traced_decision,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


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


def _load_config(path, rank=0):
    """YAML -> (argparse defaults, agent environment). CLI and preset env win.

    Delegates to integrations/v3/run_config.py, the single parser of
    config.yaml shared with the launcher and serving scripts. ``rank`` selects
    the replica when a service lists several ``base_urls``.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_config import RunConfig

    config = RunConfig.load(path)
    return config.argparse_defaults(rank), config.agent_env(rank)


def main():
    parser = argparse.ArgumentParser(description="Run RGB-D waypoint Actor on Habitat R2R-CE")
    parser.add_argument(
        "--config", type=Path,
        default=(ROOT / "integrations/v3/config.yaml"),
        help="YAML with model/agent/runner defaults; CLI flags and preset env vars override it. "
        "Pass --no-config-file to ignore it.",
    )
    parser.add_argument("--no-config-file", action="store_true")
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
        help="Planner/set-of-mark model name: vllm-<served>, joyai[-<served>] or a local path.",
    )
    parser.add_argument("--base-url", default=None, help="Server for --model-path (default: env VLLM_BASE_URL).")
    parser.add_argument("--actor-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--actor", choices=("waypoint", "awarevln", "panohop"), default="waypoint",
        help="awarevln: drive Habitat with the AwareVLN policy served by integrations/v3/serve_awarevln.py "
        "(native forward/turn/stop protocol, 512x512 RGB like its reference evaluation). "
        "panohop: CWP 12-view look-around candidates + VLM chooser + follower hops "
        "(integrations/v3/panohop_actor.py; runs in this process).",
    )
    parser.add_argument("--awarevln-url", default="http://127.0.0.1:8600/v1")
    parser.add_argument("--awarevln-model", default="awarevln")
    parser.add_argument("--panohop-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--panohop-model", default="qwen3-vl-8b")
    parser.add_argument("--panohop-stop-verify", type=int, default=0,
                        help="1: route STOP votes through the verification call "
                        "(measured worse zero-shot; kept for ablations).")
    parser.add_argument(
        "--cwp-candidates", type=int, default=0,
        help="1: supply runner-side CWP waypoint candidates to the waypoint "
        "actor (replaces the worker's floor-openings generator; forward "
        "sector on normal steps, full ring on PREVIEW).",
    )
    parser.add_argument(
        "--panohop-mode", choices=("always", "selective"), default="always",
        help="always: look around at every decision point (upper bound). "
        "selective: monocular forward arc by default; the VLM chooses LOOK "
        "to spend a look-around (forced on first decision and failed hops).",
    )
    parser.add_argument("--waypoint-radius", type=float, default=0.25)
    parser.add_argument(
        "--camera-pitch-deg", type=float, default=0.0,
        help="Tilt both cameras about their x axis; negative looks down so more floor is visible. "
        "Back-projection, the floor mask, spatial memory and previews all read the live sensor pose, so no other setting changes.",
    )
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
        "--trace-jsonl",
        type=Path,
        help=(
            "Persist one complete non-image model/memory decision per executed "
            "decision step. With --debug-memory, defaults to output-dir or the "
            "launch directory as r2r_trace_rank_<rank>.jsonl."
        ),
    )
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
    pre, _ = parser.parse_known_args()
    if not pre.no_config_file and pre.config and Path(pre.config).exists():
        defaults, agent_env = _load_config(pre.config, rank=pre.rank)
        parser.set_defaults(**defaults)
        for key, value in agent_env.items():
            # A variable exported by the caller (e.g. an A/B script) wins.
            os.environ.setdefault(key, value)
        print("config: {} (defaults: {}; env: {})".format(
            pre.config, ", ".join(sorted(defaults)) or "-",
            ", ".join(sorted(agent_env)) or "-"), flush=True)
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
    if args.trace_jsonl is None and args.debug_memory:
        trace_parent = args.output_dir or Path.cwd()
        args.trace_jsonl = trace_parent / "r2r_trace_rank_{}.jsonl".format(
            args.rank
        )
    if args.trace_jsonl is not None:
        args.trace_jsonl = args.trace_jsonl.expanduser().resolve()

    overrides = R2R_CE_OVERRIDES + DEPTH_SENSOR_OVERRIDES + [
        "habitat.dataset.split={}".format(args.split),
        "habitat.dataset.data_path='{}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz'".format(HABITAT_DATA),
        "habitat.dataset.scenes_dir={}/scene_datasets".format(HABITAT_DATA),
        "habitat.environment.max_episode_steps={}".format(args.max_steps),
    ]
    if args.scene_id != "all":
        overrides.append("habitat.dataset.content_scenes=[{}]".format(args.scene_id))
    if args.camera_pitch_deg:
        pitch = float(np.deg2rad(args.camera_pitch_deg))
        overrides += [
            "habitat.simulator.agents.main_agent.sim_sensors.{}_sensor.orientation=[{},0,0]".format(sensor, pitch)
            for sensor in ("rgb", "depth")
        ]
    if args.actor == "awarevln":
        # AwareVLN was evaluated on square 512x512 frames; keep depth aligned.
        overrides += [
            "habitat.simulator.agents.main_agent.sim_sensors.{}_sensor.{}=512".format(sensor, side)
            for sensor in ("rgb", "depth") for side in ("height", "width")
        ]
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
    if args.actor == "panohop" or args.cwp_candidates:
        # CWP was trained on level 224 RGB / 256 depth-in-meters views; give it
        # dedicated sensors so the ring never depends on the nav camera pitch.
        import dataclasses as _dc

        from habitat.config.default_structured_configs import (
            HabitatSimDepthSensorConfig,
            HabitatSimRGBSensorConfig,
        )

        # A second sensor of the same type needs its own uuid; the base
        # configs have none (it defaults to "rgb"/"depth"), so add the field
        # exactly the way HeadRGBSensorConfig does upstream.
        @_dc.dataclass
        class _CwpRGBSensorConfig(HabitatSimRGBSensorConfig):
            uuid: str = "cwp_rgb"

        @_dc.dataclass
        class _CwpDepthSensorConfig(HabitatSimDepthSensorConfig):
            uuid: str = "cwp_depth"

        with habitat.config.read_write(config):
            sensors = config.habitat.simulator.agents.main_agent.sim_sensors
            sensors["cwp_rgb"] = _CwpRGBSensorConfig(
                height=224, width=224, hfov=90,
                position=[0.0, SENSOR_HEIGHT_M, 0.0], orientation=[0.0, 0.0, 0.0])
            sensors["cwp_depth"] = _CwpDepthSensorConfig(
                height=256, width=256, hfov=90,
                position=[0.0, SENSOR_HEIGHT_M, 0.0], orientation=[0.0, 0.0, 0.0],
                min_depth=0.0, max_depth=10.0, normalize_depth=False)
    if args.actor == "awarevln":
        from awarevln_actor import AwareVLNActor

        actor = AwareVLNActor(args.awarevln_url, args.awarevln_model)
    elif args.actor == "panohop":
        from panohop_actor import PanoHopActor

        actor = PanoHopActor(args.panohop_url, args.panohop_model,
                             mode=args.panohop_mode,
                             stop_verify=bool(args.panohop_stop_verify))
    else:
        actor = WaypointActorProcess(
            args.actor_python, ROOT / "integrations/v3/vln_waypoint_worker.py", args.model_path, args.gpu_id,
            base_url=args.base_url,
            evidence_dir=(args.output_dir / "evidence" if args.record_video or args.debug_memory else None),
        )
    actor.want_visuals = bool(args.record_video)
    if args.record_video:
        # Habitat changes the working directory below; keep media paths pinned
        # to the directory from which this runner was launched.
        args.video_dir = args.video_dir.expanduser().resolve()
        from habitat.utils.visualizations.utils import images_to_video
        args.video_dir.mkdir(parents=True, exist_ok=True)
        (args.video_dir / "topdown").mkdir(exist_ok=True)
    previous_directory = Path.cwd()
    trace_handle = None
    try:
        if args.trace_jsonl is not None:
            args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = args.trace_jsonl.open(
                "w", encoding="utf-8", buffering=1
            )
        os.chdir(HABITAT_ROOT)
        with habitat.Env(config=config) as env:
            if hasattr(actor, "attach_env"):
                actor.attach_env(env)
            cwp_feed = None
            if args.cwp_candidates and args.actor == "waypoint":
                from cwp_feed import CwpCandidateFeed

                cwp_feed = CwpCandidateFeed(env)
            episodes = _select_episodes(
                env.episodes,
                episode_id=args.episode_id,
                episode_count=args.episodes,
                rank=args.rank,
                world_size=args.world_size,
                episode_ids=episode_ids,
            )
            # Habitat Env.episodes has no setter. Assigning that name merely
            # creates a shadow attribute while reset() continues consuming the
            # iterator constructed for the full dataset. Replace the iterator
            # so explicit IDs, prefixes, and rank sharding are actually used.
            env.episode_iterator = iter(episodes)
            env.number_of_episodes = len(episodes)
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
                if cwp_feed is not None:
                    cwp_feed.reset()
                steps = 0
                pending_preview = None
                previous_execution = None
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
                subgoal_chain = new_subgoal_chain(subgoals)
                navmesh_map = None
                navmesh_floor_key = None
                navmesh_maps = {}
                if frames is not None:
                    try:
                        initial_height = float(
                            env.sim.get_agent_state().position[1]
                        )
                        navmesh_floor_key, navmesh_map = (
                            _navmesh_map_for_height(
                                env, navmesh_maps, initial_height
                            )
                        )
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
                temporal_observed = False
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
                    camera_to_world = _camera_to_world(env)
                    step_cands = (
                        cwp_feed.step(intrinsics, camera_to_world)
                        if cwp_feed is not None else None
                    )
                    if step_cands is not None and steps == 0:
                        print("cwp_feed: {} forward candidates at step 0".format(
                            len(step_cands)), flush=True)
                    joint_views = ()
                    capture_request = preview_for_unseen_frame(pending_preview, temporal_observed)
                    if args.actor == "waypoint" and capture_request:
                        preview_started = time.perf_counter()
                        # Include a rear view when the model explicitly needs
                        # it; all images belong to this step's actual pose.
                        yaws = preview_headings_for_request(args.preview_yaws, capture_request, camera_to_world)
                        joint_views = _preview_views(env, yaws, args.depth_hfov, args.preview_scale)
                        preview_render_ms = (time.perf_counter() - preview_started) * 1000
                    waypoint, decision = actor.act(
                        rgb, depth, instruction, intrinsics, camera_to_world,
                        navigable=_navigable_window(env, include_heights=bool(joint_views)),
                        oracle_goal=(goal_position if args.som_oracle else None),
                        cwp_candidates=step_cands,
                        **(
                            {
                                "temporal_observed": temporal_observed,
                                "preview_views": joint_views,
                                "previous_execution": previous_execution,
                                "preview_request_id": pending_preview["recovery_id"] if pending_preview else "",
                            }
                            if args.actor == "waypoint"
                            else {}
                        ),
                    )
                    temporal_observed = False
                    if joint_views:
                        decision["preview"] = {
                            "render_ms": preview_render_ms,
                            "yaws_deg": [v["yaw_deg"] for v in joint_views],
                            "recovery_id": pending_preview["recovery_id"],
                            "joint_inference": True,
                        }
                    pending_preview = decision.get("preview_request")
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
                        ring_cands = (
                            cwp_feed.ring(intrinsics, camera_to_world)
                            if cwp_feed is not None else None
                        )
                        waypoint, decision = actor.act_on_preview(
                            views, instruction, cwp_candidates=ring_cands
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
                    update_subgoal_chain(subgoal_chain, decision, steps)
                    debug_rgb = (
                        _clean_video_frame(shown_rgb)
                        if args.clean_video
                        else _annotated_video_frame(
                            shown_rgb, decision, steps, chain=subgoal_chain,
                        )
                    )
                    if frames is not None:
                        try:
                            navmesh_floor_key, navmesh_map = (
                                _navmesh_map_for_height(
                                    env, navmesh_maps, position_before[1]
                                )
                            )
                        except Exception as exc:
                            print(
                                "Top-down floor slice fallback to RGB: "
                                "{}: {}".format(type(exc).__name__, exc),
                                flush=True,
                            )
                        if navmesh_map is None:
                            frames.append(debug_rgb)
                        else:
                            frames.append(_topdown_panel(
                                debug_rgb, _render_topdown(
                                    env, navmesh_map, positions, goal_position,
                                    rgb.shape[0],
                                    waypoints=waypoint_targets,
                                    landmark_marks=landmark_marks,
                                    floor_height=position_before[1],
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
                        # A discrete policy asked for N x 25 cm. Execute only
                        # one real step, then consult the policy on the new
                        # observation rather than running an open-loop burst.
                        action = 1  # HabitatSimActions.move_forward
                        repeats = 1
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
                    # The control contract is one decision, one simulator
                    # primitive, one resulting observation.  The loop remains
                    # for protocol compatibility, but ``repeats`` is one for
                    # every action path above.
                    executed = 0
                    intermediate_traces = []
                    for repeat in range(repeats):
                        cancel_queued = False
                        # A turn can reach the episode's step limit partway
                        # through; stop issuing primitives rather than stepping
                        # an environment that has already finished.
                        if repeat and env.episode_over:
                            break
                        observation = env.step({"action": action})
                        previous_execution = execution_observation(action, env.get_metrics())
                        executed += 1
                        positions.append(
                            env.sim.get_agent_state().position.copy()
                        )
                        if (
                            repeat < repeats - 1
                            and args.actor == "waypoint"
                            and hasattr(actor, "observe")
                        ):
                            # Queued primitives are steps the policy was not
                            # asked about; it still sees their frames.
                            intermediate_rgb, intermediate_depth = _rgb_depth(observation)
                            observed = actor.observe(
                                intermediate_rgb,
                                _camera_to_world(env),
                                depth=intermediate_depth,
                                intrinsics=_intrinsics(intermediate_rgb.shape[1], intermediate_rgb.shape[0], args.depth_hfov),
                            )
                            observed_distance = float(
                                env.get_metrics().get("distance_to_goal", 0.0)
                            )
                            intermediate_traces.append(
                                {
                                    "step": steps + repeat + 1,
                                    "decision": {
                                        "operation": "observe",
                                        **observed,
                                    },
                                    "position_before": positions[-2],
                                    "position_after": positions[-1],
                                    "distance": observed_distance,
                                }
                            )
                            advanced = (
                                observed.get("task_complete")
                                or observed.get("subgoal_before")
                                != observed.get("subgoal_after")
                            )
                            preview_requested = bool(
                                observed.get("preview_requested")
                            )
                            pending_preview = observed.get("preview_request")
                            if advanced or preview_requested:
                                # Do not finish an old subgoal's queued turn
                                # after JoyAI advanced the task or requested a
                                # surrounding-view decision on this frame.
                                # The next act consumes this same frame for
                                # control but skips its temporal ingestion.
                                temporal_observed = True
                                cancel_queued = True
                        elif repeat < repeats - 1 and hasattr(actor, "observe"):
                            # Other actor protocols keep their existing
                            # observation contract.
                            actor.observe(_observation(observation)[0])
                        if frames is not None and repeat < repeats - 1:
                            # These are the intermediate observations Captioner
                            # receives through ``observe``. The final result is
                            # recorded by the next actor decision (or the final
                            # episode frame), avoiding both gaps and duplicates.
                            turn_height = float(
                                env.sim.get_agent_state().position[1]
                            )
                            try:
                                navmesh_floor_key, navmesh_map = (
                                    _navmesh_map_for_height(
                                        env, navmesh_maps, turn_height
                                    )
                                )
                            except Exception:
                                pass
                            frames.append(_turn_frame(
                                env, observation, decision, steps + repeat + 1,
                                args, navmesh_map, positions, goal_position,
                                waypoint_targets, landmark_marks,
                            ))
                        if cancel_queued:
                            break
                    env_ms = (time.perf_counter() - env_started) * 1000
                    position_after = (
                        env.sim.get_agent_state().position.copy()
                    )
                    distance_after = float(
                        env.get_metrics().get("distance_to_goal", 0.0)
                    )
                    _write_step_trace(
                        trace_handle,
                        episode_id=episode.episode_id,
                        step=steps,
                        decision=decision,
                        action=action,
                        executed=executed,
                        position_before=position_before,
                        position_after=position_after,
                        distance_before=distance_before,
                        distance_after=distance_after,
                    )
                    for trace in intermediate_traces:
                        _write_step_trace(
                            trace_handle,
                            episode_id=episode.episode_id,
                            step=trace["step"],
                            decision=trace["decision"],
                            action=action,
                            executed=1,
                            position_before=trace["position_before"],
                            position_after=trace["position_after"],
                            distance_before=trace["distance"],
                            distance_after=trace["distance"],
                        )
                    now = time.perf_counter()
                    step_ms = (now - step_started) * 1000
                    print(
                        _step_line(episode.episode_id, steps, decision, step_ms, action)
                        + " dtg={:.2f}".format(distance_after)
                        + " region={}".format(_semantic_region_id(env)),
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
                    final_height = float(
                        env.sim.get_agent_state().position[1]
                    )
                    try:
                        navmesh_floor_key, navmesh_map = (
                            _navmesh_map_for_height(
                                env, navmesh_maps, final_height
                            )
                        )
                    except Exception:
                        pass
                    if navmesh_map is None:
                        frames.append(rgb.copy())
                    else:
                        frames.append(_topdown_panel(
                            rgb, _render_topdown(
                                env, navmesh_map, positions, goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                landmark_marks=landmark_marks,
                                floor_height=final_height,
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
                                floor_height=final_height,
                            )
                        ).save(str(args.video_dir / "topdown" / (episode_id + ".png")))
                    for floor_key, floor_map in sorted(navmesh_maps.items()):
                        floor_name = "{}_floor_{:+.1f}.png".format(
                            episode_id, floor_key
                        )
                        Image.fromarray(
                            _render_topdown(
                                env,
                                floor_map,
                                positions,
                                goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                landmark_marks=landmark_marks,
                                floor_height=floor_key,
                            )
                        ).save(str(args.video_dir / "topdown" / floor_name))
            _write_rank_summary(args.output_dir, args.rank, len(episodes), totals)
    finally:
        if trace_handle is not None:
            trace_handle.close()
        actor.close()
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
