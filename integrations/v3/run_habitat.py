"""Run the v3 RGB-D waypoint actor on Habitat R2R-CE with oracle local control.

The actor receives RGB, depth, the instruction, and camera calibration, then
returns a Habitat world-space waypoint.  ``ShortestPathFollower`` (navmesh) or
the geometric follower (``--no-navmesh``) is strictly the low-level controller
that converts that waypoint to one discrete R2R-CE action.

The helpers live in ``integrations/v3/habitat_runner`` by function: settings
(paths, shared geometry), sensors, actor_process, control, video, step_log,
episodes.  This file is the argument parser and the episode loop.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integrations.v3.habitat_runner import settings  # noqa: E402  (quiets logs, sets sys.path)
from integrations.v3.habitat_runner.settings import AGENTFLOW_ROOT, HABITAT_DATA, HABITAT_ROOT, ROOT  # noqa: E402
from integrations.v3.habitat_runner.sensors import (  # noqa: E402
    camera_intrinsics, camera_to_world_matrix, motion_overrides, navigable_window,
    render_preview_views, semantic_region_id, sensor_overrides, unpack_observation,
)
from integrations.v3.habitat_runner.actor_process import WaypointActorProcess  # noqa: E402
from integrations.v3.habitat_runner.control import (  # noqa: E402
    GeometricFollower, fallback_action_for_follower_stop, turn_primitive,
)
from integrations.v3.habitat_runner.video import (  # noqa: E402
    annotated_video_frame, decode_visuals, navmesh_map_for_height, new_subgoal_chain,
    render_topdown, topdown_panel, update_subgoal_chain,
)
from integrations.v3.habitat_runner.step_log import (  # noqa: E402
    empty_totals, step_line, write_rank_summary, write_step_trace,
)
from integrations.v3.habitat_runner.episodes import load_config, parse_episode_ids, select_episodes  # noqa: E402
from integrations.v3.camera_model import CameraModel  # noqa: E402
from integrations.v3.preview_protocol import execution_observation, preview_headings_for_request  # noqa: E402

import habitat  # noqa: E402
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # noqa: E402

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
    parser.add_argument("--depth-hfov", type=float, default=90.0, help="sensor_config.yaml camera.hfov_deg")
    parser.add_argument("--camera-height-m", type=float, default=settings.SENSOR_HEIGHT_M, help="sensor_config.yaml camera.height_m")
    parser.add_argument("--image-width", type=int, default=640, help="sensor_config.yaml camera.width")
    parser.add_argument("--image-height", type=int, default=480, help="sensor_config.yaml camera.height")
    parser.add_argument("--depth-min-m", type=float, default=0.0, help="sensor_config.yaml depth.min_m")
    parser.add_argument("--depth-max-m", type=float, default=10.0, help="sensor_config.yaml depth.max_m")
    parser.add_argument("--depth-normalize", dest="depth_normalize", action=argparse.BooleanOptionalAction, default=False,
                        help="sensor_config.yaml depth.normalize")
    parser.add_argument("--camera-intrinsics", default=None,
                        help='sensor_config.yaml camera.intrinsics as JSON {"fx":..,"fy":..,"cx":..,"cy":..}; default: pinhole from --depth-hfov')
    parser.add_argument("--camera-distortion", default=None, help="sensor_config.yaml camera.distortion: k1,k2,p1,p2[,k3] (OpenCV)")
    parser.add_argument("--camera-extrinsics", default=None,
                        help='sensor_config.yaml camera.extrinsics as JSON {"xyz":[x,y,z],"rpy_deg":[roll,pitch,yaw]}; default: --camera-height-m + --camera-pitch-deg')
    parser.add_argument("--forward-step-m", type=float, default=settings.FORWARD_STEP_M, help="config.yaml robot.forward_step_m")
    parser.add_argument("--turn-angle-deg", type=int, default=settings.TURN_ANGLE_DEG,
                        help="config.yaml robot.turn_angle_deg; the actor's 45 deg turn requests must be a multiple of it")
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
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        help="Persist one complete non-image model/memory decision per executed decision step.",
    )
    parser.add_argument("--record-video", action="store_true", help="Save RGB plus top-down trajectory MP4s.")
    parser.add_argument(
        "--navmesh", dest="navmesh", action=argparse.BooleanOptionalAction, default=True,
        help="Use the simulator navmesh (yaml runner.navmesh). --no-navmesh is deployment mode: "
        "the agent gets no navmesh traversability window and waypoints are executed by a "
        "turn-then-forward follower instead of ShortestPathFollower.",
    )
    parser.add_argument(
        "--navmesh-candidates", dest="navmesh_candidates", action=argparse.BooleanOptionalAction,
        default=None,
        help="Ablation override: give the agent the navmesh traversability window (candidate "
        "filtering) regardless of --navmesh. Default: follow --navmesh.",
    )
    parser.add_argument(
        "--navmesh-follower", dest="navmesh_follower", action=argparse.BooleanOptionalAction,
        default=None,
        help="Ablation override: execute waypoints with Habitat's ShortestPathFollower "
        "regardless of --navmesh. Default: follow --navmesh.",
    )
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument(
        "--clean-video",
        action="store_true",
        help="Record unannotated RGB instead of overlaying the agent's waypoint pixels.",
    )
    pre, _ = parser.parse_known_args()
    if not pre.no_config_file and pre.config and Path(pre.config).exists():
        defaults, agent_env = load_config(pre.config, rank=pre.rank)
        parser.set_defaults(**defaults)
        for key, value in agent_env.items():
            # A variable exported by the caller (e.g. an A/B script) wins.
            os.environ.setdefault(key, value)
        print("config: {} (defaults: {}; env: {})".format(
            pre.config, ", ".join(sorted(defaults)) or "-",
            ", ".join(sorted(agent_env)) or "-"), flush=True)
    args = parser.parse_args()
    # Sensor / motion geometry is module state for the follower and the
    # preview ring; both come from the config now.
    camera = CameraModel(
        width=args.image_width, height=args.image_height, hfov_deg=args.depth_hfov,
        intrinsics=args.camera_intrinsics, distortion=args.camera_distortion, extrinsics=args.camera_extrinsics,
        height_m=args.camera_height_m, pitch_deg=args.camera_pitch_deg,
    )
    settings.configure(camera=camera, turn_angle_deg=args.turn_angle_deg, forward_step_m=args.forward_step_m)
    # With calibrated extrinsics the mount height / pitch are theirs.
    args.camera_height_m = camera.height_m
    args.camera_pitch_deg = camera.pitch_deg
    print("camera: " + camera.describe(), flush=True)
    if 45 % settings.TURN_ANGLE_DEG:
        raise SystemExit("--turn-angle-deg must divide 45 (the actor's turn request)")
    # Resolve the id list now: the evaluator chdirs into the Habitat root
    # before the episodes are selected, which breaks a relative @file path.
    episode_ids = parse_episode_ids(args.episode_ids)
    # Ablation overrides default to the umbrella --navmesh flag.
    if args.navmesh_candidates is None:
        args.navmesh_candidates = args.navmesh
    if args.navmesh_follower is None:
        args.navmesh_follower = args.navmesh
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
    if args.trace_jsonl is not None:
        args.trace_jsonl = args.trace_jsonl.expanduser().resolve()

    overrides = motion_overrides(args.forward_step_m, args.turn_angle_deg) + sensor_overrides(
        camera, args.image_width, args.image_height, args.depth_hfov,
        args.depth_min_m, args.depth_max_m, args.depth_normalize,
    ) + [
        "habitat.dataset.split={}".format(args.split),
        "habitat.dataset.data_path='{}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz'".format(HABITAT_DATA),
        "habitat.dataset.scenes_dir={}/scene_datasets".format(HABITAT_DATA),
        "habitat.environment.max_episode_steps={}".format(args.max_steps),
    ]
    if args.scene_id != "all":
        overrides.append("habitat.dataset.content_scenes=[{}]".format(args.scene_id))
    # (sensor orientation, incl. the pitch, is part of sensor_overrides now)
    if args.actor == "awarevln":
        # AwareVLN was evaluated on square 512x512 frames; keep depth aligned.
        overrides += [
            "habitat.simulator.agents.main_agent.sim_sensors.{}_sensor.{}=512".format(sensor, side)
            for sensor in ("rgb", "depth") for side in ("height", "width")
        ]
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
    if args.actor == "panohop":
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
                position=[0.0, settings.SENSOR_HEIGHT_M, 0.0], orientation=[0.0, 0.0, 0.0])
            sensors["cwp_depth"] = _CwpDepthSensorConfig(
                height=256, width=256, hfov=90,
                position=[0.0, settings.SENSOR_HEIGHT_M, 0.0], orientation=[0.0, 0.0, 0.0],
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
            base_url=args.base_url, camera_height_m=args.camera_height_m,
            evidence_dir=(args.output_dir / "evidence" if args.record_video else None),
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
            episodes = select_episodes(
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
                write_rank_summary(args.output_dir, args.rank, 0, empty_totals())
                return
            totals = empty_totals()
            for index, episode in enumerate(episodes, start=1):
                observation = env.reset()
                # Habitat owns episode ordering; trust the environment over the
                # list index so logged IDs and goals match the active episode.
                episode = env.current_episode
                steps = 0
                pending_preview = None
                previous_execution = None
                _, _, instruction = unpack_observation(observation)
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
                            navmesh_map_for_height(
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
                # Requested waypoints accumulate over the episode so the
                # top-down map shows the whole intended route beside the
                # executed one.
                waypoint_targets = []
                follower = (
                    ShortestPathFollower(env.sim, args.waypoint_radius, return_one_hot=False)
                    if args.navmesh_follower else GeometricFollower(env, args.waypoint_radius)
                )
                step_started = time.perf_counter()
                while not env.episode_over:
                    position_before = (
                        env.sim.get_agent_state().position.copy()
                    )
                    distance_before = float(
                        env.get_metrics().get("distance_to_goal", 0.0)
                    )
                    rgb, depth, instruction = unpack_observation(observation)
                    intrinsics = camera_intrinsics(rgb.shape[1], rgb.shape[0], args.depth_hfov)
                    camera_to_world = camera_to_world_matrix(env)
                    joint_views = ()
                    if args.actor == "waypoint" and pending_preview:
                        preview_started = time.perf_counter()
                        # Include a rear view when the model explicitly needs
                        # it; all images belong to this step's actual pose.
                        yaws = preview_headings_for_request(args.preview_yaws, pending_preview, camera_to_world)
                        joint_views = render_preview_views(env, yaws, args.depth_hfov, args.preview_scale)
                        preview_render_ms = (time.perf_counter() - preview_started) * 1000
                    waypoint, decision = actor.act(
                        rgb, depth, instruction, intrinsics, camera_to_world,
                        navigable=(navigable_window(env, include_heights=bool(joint_views))
                                   if args.navmesh_candidates else None),
                        **(
                            {
                                "preview_views": joint_views,
                                "previous_execution": previous_execution,
                                "preview_request_id": pending_preview["recovery_id"] if pending_preview else "",
                            }
                            if args.actor == "waypoint"
                            else {}
                        ),
                    )
                    if joint_views:
                        decision["preview"] = {
                            "render_ms": preview_render_ms,
                            "yaws_deg": [v["yaw_deg"] for v in joint_views],
                            "recovery_id": pending_preview["recovery_id"],
                            "joint_inference": True,
                        }
                    pending_preview = decision.get("preview_request")
                    if waypoint is not None:
                        waypoint_targets.append(
                            np.asarray(waypoint, dtype=np.float64)
                        )
                    render_started = time.perf_counter()
                    agent_map, marker_frame = decode_visuals(decision)
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
                        shown_rgb.copy()
                        if args.clean_video
                        else annotated_video_frame(
                            shown_rgb, decision, steps, chain=subgoal_chain,
                        )
                    )
                    if frames is not None:
                        try:
                            navmesh_floor_key, navmesh_map = (
                                navmesh_map_for_height(
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
                            frames.append(topdown_panel(
                                debug_rgb, render_topdown(
                                    env, navmesh_map, positions, goal_position,
                                    rgb.shape[0],
                                    waypoints=waypoint_targets,
                                    floor_height=position_before[1],
                                ),
                                agent_map=last_agent_map,
                            ))
                    render_ms = (time.perf_counter() - render_started) * 1000
                    follower_action = None
                    if decision.get("stop"):
                        action = 0  # HabitatSimActions.stop, chosen by Actor.
                    elif decision.get("turn_deg"):
                        action = turn_primitive(decision["turn_deg"])
                    elif decision.get("forward_steps"):
                        # A discrete policy asked for N x 25 cm. Execute only
                        # one real step, then consult the policy on the new
                        # observation rather than running an open-loop burst.
                        action = 1  # HabitatSimActions.move_forward
                    elif waypoint is None:
                        # No waypoint, no turn and no stop: the previewed
                        # heading had no valid depth. Turning in place keeps
                        # the episode alive instead of handing the follower a
                        # None target.
                        action = 2  # HabitatSimActions.turn_left
                    else:
                        follower_action = follower.get_next_action(waypoint)
                        if follower_action is None or int(follower_action) == 0:
                            # STOP is reserved exclusively for the Actor. A
                            # local follower can emit STOP for a nearby or
                            # unreachable waypoint, but that is not task end.
                            # During lateral stuck recovery, forcing forward
                            # repeats the collision that recovery is meant to
                            # escape; execute a stable turn primitive instead.
                            action = fallback_action_for_follower_stop(
                                decision
                            )
                        else:
                            action = follower_action
                    env_started = time.perf_counter()
                    # One decision, one simulator primitive, one observation.
                    observation = env.step({"action": action})
                    previous_execution = execution_observation(action, env.get_metrics())
                    positions.append(env.sim.get_agent_state().position.copy())
                    env_ms = (time.perf_counter() - env_started) * 1000
                    position_after = (
                        env.sim.get_agent_state().position.copy()
                    )
                    distance_after = float(
                        env.get_metrics().get("distance_to_goal", 0.0)
                    )
                    write_step_trace(
                        trace_handle,
                        episode_id=episode.episode_id,
                        step=steps,
                        decision=decision,
                        action=action,
                        position_before=position_before,
                        position_after=position_after,
                        distance_before=distance_before,
                        distance_after=distance_after,
                    )
                    now = time.perf_counter()
                    step_ms = (now - step_started) * 1000
                    print(
                        step_line(episode.episode_id, steps, decision, step_ms, action)
                        + " dtg={:.2f}".format(distance_after)
                        + " region={}".format(semantic_region_id(env)),
                        flush=True,
                    )
                    step_started = now
                    steps += 1
                metrics = env.get_metrics()
                for name in totals:
                    totals[name] += float(metrics.get(name, 0.0))
                print("rank={} [{}/{}] id={} steps={} success={:.3f} spl={:.3f} dtg={:.2f}".format(args.rank, index, len(episodes), episode.episode_id, steps, float(metrics.get("success", 0)), float(metrics.get("spl", 0)), float(metrics.get("distance_to_goal", 0))), flush=True)
                if frames:
                    rgb, _, _ = unpack_observation(observation)
                    final_height = float(
                        env.sim.get_agent_state().position[1]
                    )
                    try:
                        navmesh_floor_key, navmesh_map = (
                            navmesh_map_for_height(
                                env, navmesh_maps, final_height
                            )
                        )
                    except Exception:
                        pass
                    if navmesh_map is None:
                        frames.append(rgb.copy())
                    else:
                        frames.append(topdown_panel(
                            rgb, render_topdown(
                                env, navmesh_map, positions, goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                floor_height=final_height,
                            ),
                            agent_map=last_agent_map,
                        ))
                    episode_id = str(episode.episode_id).replace("/", "_")
                    images_to_video(frames, str(args.video_dir), episode_id, fps=10)
                    if navmesh_map is not None:
                        Image.fromarray(
                            render_topdown(
                                env, navmesh_map, positions, goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                floor_height=final_height,
                            )
                        ).save(str(args.video_dir / "topdown" / (episode_id + ".png")))
                    for floor_key, floor_map in sorted(navmesh_maps.items()):
                        floor_name = "{}_floor_{:+.1f}.png".format(
                            episode_id, floor_key
                        )
                        Image.fromarray(
                            render_topdown(
                                env,
                                floor_map,
                                positions,
                                goal_position,
                                rgb.shape[0],
                                waypoints=waypoint_targets,
                                floor_height=floor_key,
                            )
                        ).save(str(args.video_dir / "topdown" / floor_name))
            write_rank_summary(args.output_dir, args.rank, len(episodes), totals)
    finally:
        if trace_handle is not None:
            trace_handle.close()
        actor.close()
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
