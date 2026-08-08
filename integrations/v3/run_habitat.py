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

R2R_CE_OVERRIDES = [
    "habitat.simulator.forward_step_size=0.25",
    "habitat.simulator.turn_angle=15",
    "habitat.task.measurements.success.success_distance=3.0",
]


DEPTH_SENSOR_OVERRIDES = [
    # RGB and depth must share pixel coordinates and the same camera frame;
    # vln_agent_2 back-projects the RGB-selected pixel through this depth map.
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=480",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=640",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90",
    "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position=[0,1.25,0]",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.type=HabitatSimDepthSensor",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=480",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=640",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position=[0,1.25,0]",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth=0.0",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=10.0",
    "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=false",
]


class WaypointActorProcess:
    """Keep the Python 3.12 vision model out of Habitat's Python process."""

    def __init__(self, python, worker, model_path, gpu_id, timeout=600):
        command = [str(python), str(worker), "--model-path", str(model_path)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(AGENTFLOW_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
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

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world):
        encode_started = time.perf_counter()
        request = {
            "operation": "act",
            "rgb": self._png(rgb), "depth": self._array(depth),
            "instruction": instruction, "intrinsics": np.asarray(intrinsics).tolist(),
            "camera_to_world": np.asarray(camera_to_world).tolist(),
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
        return np.asarray(result["world_xyz"], dtype=np.float32), result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _observation(observation):
    rgb = np.asarray(observation["rgb"])[..., :3]
    depth_key = _depth_sensor_key(observation)
    depth = np.asarray(observation[depth_key])
    if depth.ndim == 3:
        depth = depth[..., 0]
    return np.ascontiguousarray(rgb, dtype=np.uint8), np.ascontiguousarray(depth), observation["instruction"]["text"]


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


def _build_navmesh_map(env, resolution=1024):
    """Render the scene navmesh for the optional trajectory visualization."""
    from habitat.utils.visualizations.maps import get_topdown_map_from_sim

    return get_topdown_map_from_sim(
        env.sim, map_resolution=resolution, draw_border=True
    )


def _render_topdown(env, navmesh_map, positions, goal_position, output_height):
    """Draw the executed trajectory, start, current position, and goal."""
    import cv2
    from habitat.utils.visualizations import maps

    image = maps.colorize_topdown_map(navmesh_map.copy())

    def to_pixel(position):
        row, col = maps.to_grid(
            position[2], position[0], navmesh_map.shape,
            pathfinder=env.sim.pathfinder,
        )
        return col, row

    path = [to_pixel(position) for position in positions]
    if len(path) > 1:
        cv2.polylines(image, [np.asarray(path, dtype=np.int32)], False, (0, 80, 255), 3)
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


def _topdown_panel(rgb, topdown):
    """Place the first-person RGB image beside the current top-down map."""
    return np.concatenate((rgb, topdown), axis=1)


def _clean_video_frame(rgb):
    """Return an unannotated RGB frame for the saved video."""
    return rgb.copy()


def _latency_line(episode_id, steps, decision, step_ms, render_ms, env_ms):
    """Attribute one step's wall time to the model, the IPC, and the simulator."""
    timings = decision.get("timings") or {}
    roundtrip_ms = decision.get("roundtrip_ms", 0.0)
    worker_ms = timings.get("total_ms", 0.0)
    memory_ms = timings.get("memory_ms", 0.0)
    captioner_ms = timings.get("captioner_ms", 0.0)
    accounted = (
        timings.get("rgb_ms", 0.0) + memory_ms + timings.get("depth_ms", 0.0)
        + timings.get("select_pixel_ms", 0.0) + timings.get("waypoint_ms", 0.0)
    )
    return (
        "episode={} step={} LATENCY step={:.0f}ms | encode={:.0f} ipc={:.0f} "
        "worker={:.0f} [rgb={:.0f} memory={:.0f} (captioner={:.0f} rules={:.0f}) "
        "depth={:.0f} select_pixel={:.0f} waypoint={:.0f} other={:.0f}] "
        "render={:.0f} env={:.0f}".format(
            episode_id, steps, step_ms,
            decision.get("encode_ms", 0.0),
            # Whatever the round trip spent outside the worker's own act() call.
            roundtrip_ms - worker_ms,
            worker_ms,
            timings.get("rgb_ms", 0.0),
            memory_ms, captioner_ms, memory_ms - captioner_ms,
            timings.get("depth_ms", 0.0), timings.get("select_pixel_ms", 0.0),
            timings.get("waypoint_ms", 0.0),
            # Non-zero here means act() spends time outside every named phase.
            worker_ms - accounted,
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


def _step_line(episode_id, steps, decision, step_ms, action):
    """One line per step: where the time went, memory state, and the action.

    ``--debug-memory`` adds the full per-memory dumps below this line.
    """
    timings = decision.get("timings") or {}
    task = decision.get("task_memory") or {}
    temporal = decision.get("temporal_memory") or {}
    waypoint_ms = timings.get("waypoint_ms", 0.0)
    select_ms = timings.get("select_pixel_ms", 0.0)
    captioner_ms = timings.get("captioner_ms", 0.0)
    debug = decision.get("debug") or {}
    analyzed = debug.get("analyzed_subgoal") or {}
    analyzed_id = analyzed.get("subgoal_id")
    current_id = task.get("current_subgoal_id")
    pixel = decision.get("pixel_uv")
    line = (
        "ep={} s={} {:.0f}ms [wp={:.0f} sel={:.0f} cap={:.0f} rest={:.0f}] "
        "sg={}->{} mode={} win={} obs={} | cap={} act={}".format(
            episode_id, steps, step_ms,
            waypoint_ms, select_ms, captioner_ms,
            step_ms - waypoint_ms - select_ms - captioner_ms,
            analyzed_id, current_id,
            temporal.get("active_error_mode"),
            len(temporal.get("frame_ids") or ()),
            task.get("observation_count"),
            _caption_summary(decision),
            "STOP" if decision.get("stop") else
            "({},{}) d={:.2f} a={}".format(
                pixel[0], pixel[1], decision.get("depth_m", 0.0), int(action)
            ),
        )
    )
    # Surface only the abnormal cases inline; the rest stays behind the flag.
    if decision.get("temporal_error"):
        line += " ANALYSIS_ERROR={!r}".format(decision.get("temporal_error"))
    return line


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


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
        "  WAYPOINT phase={} heading_lock={} model_intent={} "
        "applied_intent={} guard={!r} evidence={!r} raw={!r} "
        "normalized={} requested={} validated={} depth={} "
        "world={} error_candidate={} guard={!r} recovery={} "
        "stop_disposition={} stop_reason={}".format(
            debug.get("navigation_phase"),
            debug.get("corridor_heading_yaw_deg"),
            debug.get("waypoint_model_intent"),
            debug.get("waypoint_applied_intent"),
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


def _select_episodes(
    available,
    *,
    episode_id,
    episode_count,
    rank,
    world_size,
):
    """Select one exact episode or the requested evaluation prefix."""
    available = list(available)
    if episode_id is not None:
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
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpu-id", type=int, default=0)
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
    parser.add_argument("--waypoint-radius", type=float, default=0.25)
    parser.add_argument("--depth-hfov", type=float, default=90.0)
    parser.add_argument("--debug-memory", action="store_true", help="Dump both memories and the full latency breakdown under each step.")
    parser.add_argument(
        "--debug-navigation",
        action="store_true",
        help="Explain position, subgoal, Captioner, waypoint, and control decisions every step.",
    )
    parser.add_argument("--record-video", action="store_true", help="Save RGB plus top-down trajectory MP4s.")
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be in [0, --world-size).")
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
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
    actor = WaypointActorProcess(args.actor_python, ROOT / "integrations/v3/vln_waypoint_worker.py", args.model_path, args.gpu_id)
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
                        rgb, depth, instruction, intrinsics, _camera_to_world(env)
                    )
                    render_started = time.perf_counter()
                    debug_rgb = _clean_video_frame(rgb)
                    if frames is not None:
                        if navmesh_map is None:
                            frames.append(debug_rgb)
                        else:
                            frames.append(_topdown_panel(
                                debug_rgb, _render_topdown(env, navmesh_map, positions, goal_position, rgb.shape[0])
                            ))
                    render_ms = (time.perf_counter() - render_started) * 1000
                    follower_action = None
                    if decision.get("stop"):
                        action = 0  # HabitatSimActions.stop, chosen by Actor.
                    else:
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
                    observation = env.step({"action": action})
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
                        _step_line(episode.episode_id, steps, decision, step_ms, action),
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
                    steps += 1
                    positions.append(env.sim.get_agent_state().position.copy())
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
                            rgb, _render_topdown(env, navmesh_map, positions, goal_position, rgb.shape[0])
                        ))
                    episode_id = str(episode.episode_id).replace("/", "_")
                    images_to_video(frames, str(args.video_dir), episode_id, fps=10)
                    if navmesh_map is not None:
                        Image.fromarray(
                            _render_topdown(env, navmesh_map, positions, goal_position, rgb.shape[0])
                        ).save(str(args.video_dir / "topdown" / (episode_id + ".png")))
            _write_rank_summary(args.output_dir, args.rank, len(episodes), totals)
    finally:
        actor.close()
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
