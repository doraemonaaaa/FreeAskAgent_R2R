"""Evaluate the v1 agentflow.agents.vln_agent on Habitat R2R-CE."""

import argparse
import gzip
import json
import logging
import os
import sys
import warnings
from pathlib import Path

for _name in ("habitat", "habitat_sim", "magnum", "corrade", "transformers", "torch", "accelerate", "diffusers", "PIL", "vllm"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.captureWarnings(True)
warnings.filterwarnings("ignore")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
HABITAT_ROOT = ROOT.parent / "habitat" / "habitat-lab"
HABITAT_DATA = ROOT.parent / "habitat" / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HABITAT_ROOT / "habitat-lab"))

import habitat  # noqa: E402

from integrations.v1.r2r_ce_adapter import (  # noqa: E402
    HABITAT_ACTIONS,
    R2R_CE_OVERRIDES,
    VLNAgentProcess,
    habitat_action,
    unpack_observation,
)
from integrations.run_artifacts import RunArtifactWriter, safe_name  # noqa: E402


def _topdown_panel(rgb, topdown_info):
    """Return an RGB frame next to the map with Habitat's cumulative trajectory."""
    import cv2
    import numpy as np
    from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

    # TopDownMap mutates its map in place each step; copy it so video frames
    # retain the trajectory as it looked at this exact timestep.
    map_info = dict(topdown_info)
    map_info["map"] = topdown_info["map"].copy()
    if topdown_info.get("fog_of_war_mask") is not None:
        map_info["fog_of_war_mask"] = topdown_info["fog_of_war_mask"].copy()
    topdown = colorize_draw_agent_and_fit_to_height(map_info, rgb.shape[0])
    return np.concatenate((rgb, topdown), axis=1)


def _save_topdown_map(topdown_info, rgb_height, path):
    from PIL import Image
    from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

    map_info = dict(topdown_info)
    map_info["map"] = topdown_info["map"].copy()
    if topdown_info.get("fog_of_war_mask") is not None:
        map_info["fog_of_war_mask"] = topdown_info["fog_of_war_mask"].copy()
    image = colorize_draw_agent_and_fit_to_height(map_info, rgb_height)
    Image.fromarray(image).save(str(path))


def _validate_scene_selection(split, scene_id):
    """Fail before Habitat Env construction when filtering would select nothing."""
    if scene_id == "all":
        return
    dataset_path = HABITAT_DATA / "datasets/vln/mp3d/r2r/v1" / split / (split + ".json.gz")
    if not dataset_path.is_file():
        raise SystemExit("R2R-CE split is unavailable: {}".format(dataset_path))
    with gzip.open(str(dataset_path), "rt") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", payload)
    selected = [episode for episode in episodes if scene_id in episode.get("scene_id", "")]
    if selected:
        return
    available = sorted({Path(episode["scene_id"]).stem for episode in episodes})
    raise SystemExit(
        "No episodes for --split {} --scene-id {}. Use --scene-id all or choose one of: {}"
        .format(split, scene_id, ", ".join(available))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val_seen")
    parser.add_argument("--scene-id", default="17DRP5sb8fy")
    parser.add_argument("--episodes", type=int, default=1, help="Use 0 for all.")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=0, help="This process's shard index.")
    parser.add_argument("--world-size", type=int, default=1, help="Number of evaluation processes.")
    parser.add_argument("--gpu-id", type=int, help="GPU assigned to this process's model worker.")
    parser.add_argument("--output-dir", type=Path, help="Write this rank's totals for later aggregation.")
    parser.add_argument("--verbose", action="store_true", help="Print failure diagnostics as well as metrics.")
    parser.add_argument("--model-path", default=str(ROOT.parent / "FreeAskAgent" / "models" / "Qwen3-VL-8B-Instruct"))
    parser.add_argument("--planner-model-path", default=str(ROOT.parent / "FreeAskAgent" / "models" / "Qwen3-VL-8B-Instruct"))
    parser.add_argument(
        "--temporal-model-path",
        help="Optional video-understanding model; defaults to the planner model.",
    )
    parser.add_argument(
        "--memory-mode",
        choices=("temporal", "task", "task+temporal"),
        default="temporal",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Root directory for run-isolated results and media.",
    )
    parser.add_argument("--run-id", help="Shared run ID for all ranks.")
    parser.add_argument("--record-video", action="store_true", help="Save RGB + TopDownMap trajectory as MP4")
    parser.add_argument(
        "--video-dir",
        help="Legacy media root; used as artifact root when --artifact-dir is omitted.",
    )
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be in [0, --world-size).")
    _validate_scene_selection(args.split, args.scene_id)

    artifact_root = (
        args.artifact_dir
        or args.video_dir
        or (args.output_dir / "runs" if args.output_dir else None)
        or "artifacts"
    )
    artifacts = RunArtifactWriter(
        artifact_root,
        run_id=args.run_id or os.environ.get("R2R_RUN_ID"),
        rank=args.rank,
        run_config={
            "runner": "r2r_ce_example",
            "split": args.split,
            "scene_id": args.scene_id,
            "episodes_requested": args.episodes,
            "max_steps": args.max_steps,
            "rank": args.rank,
            "world_size": args.world_size,
            "gpu_id": args.gpu_id,
            "policy_model_path": args.model_path,
            "planner_model_path": args.planner_model_path,
            "temporal_model_path": args.temporal_model_path,
            "memory_mode": args.memory_mode,
            "record_video": args.record_video,
            "video_fps": 10,
        },
    )

    if args.record_video:
        from habitat.utils.visualizations.utils import images_to_video

    overrides = R2R_CE_OVERRIDES + [
        "habitat.dataset.split={}".format(args.split),
        "habitat.dataset.data_path='{}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz'".format(HABITAT_DATA),
        "habitat.dataset.scenes_dir={}/scene_datasets".format(HABITAT_DATA),
        "habitat.environment.max_episode_steps={}".format(args.max_steps),
    ]
    if args.scene_id != "all":
        overrides.append("habitat.dataset.content_scenes=[{}]".format(args.scene_id))
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)

    has_topdown = False
    if args.record_video:
        try:
            import habitat.config.default_structured_configs as hcfg
            config.habitat.task.measurements.top_down_map = hcfg.TopDownMapMeasurementConfig(
                type="TopDownMap",
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_shortest_path=True,
                draw_view_points=False,
                draw_goal_positions=True,
            )
            has_topdown = True
        except Exception:
            pass

    agent = VLNAgentProcess(
        ROOT / ".venv/bin/python",
        ROOT / "integrations/v1/vln_agent_worker.py",
        args.model_path,
        args.planner_model_path,
        gpu_id=args.gpu_id,
        temporal_model_path=args.temporal_model_path,
        memory_mode=args.memory_mode,
    )
    old_cwd = Path.cwd()
    try:
        os.chdir(HABITAT_ROOT)
        with habitat.Env(config=config) as env:
            requested = env.episodes if args.episodes == 0 else env.episodes[:args.episodes]
            env.episodes = requested[args.rank::args.world_size]
            count = len(env.episodes)
            if count == 0:
                print("rank={} has no episodes".format(args.rank), flush=True)
                if args.output_dir:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    with (args.output_dir / "rank_{}.json".format(args.rank)).open("w") as handle:
                        json.dump({"rank": args.rank, "count": 0, "totals": {
                            "success": 0.0, "spl": 0.0, "distance_to_goal": 0.0,
                        }}, handle, sort_keys=True)
                return
            totals = {"success": 0.0, "spl": 0.0, "distance_to_goal": 0.0}
            for index in range(count):
                observation = env.reset()
                rgb, instruction = unpack_observation(observation)
                initial_metrics = env.get_metrics()
                agent.reset(instruction)
                steps = 0
                actions = []
                invalid_action_error = None
                frames = [] if args.record_video else None
                topdown_info = initial_metrics.get("top_down_map") if has_topdown else None
                if frames is not None:
                    if topdown_info:
                        frames.append(_topdown_panel(rgb, topdown_info))
                    else:
                        frames.append(rgb.copy())
                while not env.episode_over:
                    try:
                        action = agent.act(rgb)
                    except RuntimeError as exc:
                        if "returned invalid action" not in str(exc):
                            raise
                        invalid_action_error = str(exc)
                        print(
                            "rank={} episode={} ignored_invalid_action={!r}".format(
                                args.rank, env.current_episode.episode_id, invalid_action_error
                            ),
                            flush=True,
                        )
                        break
                    actions.append(action)
                    observation = env.step(habitat_action(action))
                    steps += 1
                    rgb, _ = unpack_observation(observation)
                    post_metrics = env.get_metrics()
                    topdown_info = post_metrics.get("top_down_map") if has_topdown else None
                    if frames is not None:
                        if topdown_info:
                            frames.append(_topdown_panel(rgb, topdown_info))
                        else:
                            frames.append(rgb.copy())
                finish_response = agent.finish_episode(rgb)
                metrics = env.get_metrics()
                print(
                    "rank={} episode={}/{} id={} steps={} metrics={}".format(
                        args.rank, index + 1, count, env.current_episode.episode_id, steps, metrics
                    ),
                    flush=True,
                )
                video_path = None
                topdown_path = None
                artifact_episode_id = str(env.current_episode.episode_id)
                if frames:
                    images_to_video(
                        frames,
                        str(artifacts.video_dir),
                        safe_name(artifact_episode_id),
                        fps=10,
                    )
                    candidate = artifacts.media_path(
                        artifact_episode_id,
                        "video",
                    )
                    if candidate.is_file():
                        video_path = candidate
                if topdown_info:
                    map_path = artifacts.media_path(
                        artifact_episode_id,
                        "topdown",
                    )
                    _save_topdown_map(topdown_info, rgb.shape[0], map_path)
                    print("TopDownMap saved: {}".format(map_path), flush=True)
                    topdown_path = map_path
                artifacts.add_episode(
                    episode_id=artifact_episode_id,
                    instruction=instruction,
                    actions=actions,
                    steps=steps,
                    habitat_metrics=metrics,
                    video_path=video_path,
                    topdown_path=topdown_path,
                    memory_diagnostics=(
                        finish_response.get("memory")
                        if isinstance(finish_response, dict)
                        else agent.last_memory_diagnostics
                    ),
                    error=invalid_action_error,
                )
                if args.verbose and float(metrics.get("success", 0.0)) < 1.0:
                    # Habitat does not expose a separate termination reason in
                    # the standard metrics, so report the observable causes.
                    if actions and actions[-1] == "STOP":
                        reason = "STOP issued outside success radius"
                    elif steps >= args.max_steps:
                        reason = "maximum episode steps reached before success"
                    else:
                        reason = "episode terminated without successful STOP"
                    print(
                        "failure episode={} reason={} distance_to_goal={:.3f} "
                        "last_action={} action_counts={} instruction={!r}".format(
                            env.current_episode.episode_id,
                            reason,
                            float(metrics.get("distance_to_goal", float("nan"))),
                            actions[-1] if actions else None,
                            {name: actions.count(name) for name in HABITAT_ACTIONS},
                            instruction,
                        ),
                        flush=True,
                    )
                for name in totals:
                    totals[name] += float(metrics[name])
            result = {
                "rank": args.rank,
                "count": count,
                "totals": totals,
                "artifact_results": str(artifacts.results_path),
            }
            print("rank_summary={}".format(json.dumps(result, sort_keys=True)), flush=True)
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                with (args.output_dir / "rank_{}.json".format(args.rank)).open("w") as handle:
                    json.dump(result, handle, sort_keys=True)
    finally:
        agent.close()
        os.chdir(old_cwd)
    print("Run artifacts: {}".format(artifacts.results_path), flush=True)


if __name__ == "__main__":
    main()
