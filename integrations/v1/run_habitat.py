"""Minimal single-GPU Habitat + v1 VLN agent inference runner.

Usage:
  CUDA_VISIBLE_DEVICES=0 python integrations/v1/run_habitat.py \\
      --split val_unseen --episodes 0 --max-steps 500
  CUDA_VISIBLE_DEVICES=0 python integrations/v1/run_habitat.py \\
      --split val_unseen --episodes 10 --max-steps 500 --record-video
  # Output:
  #   videos/{episode_id}.mp4      — first-person navigation video
  #   videos/topdown/{episode_id}.png — top-down trajectory plot
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

for _name in ("habitat", "habitat_sim", "magnum", "corrade",
               "transformers", "torch", "accelerate", "diffusers", "PIL"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.captureWarnings(True)
warnings.filterwarnings("ignore")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parents[2]
HABITAT_ROOT = ROOT.parent / "habitat" / "habitat-lab"
HABITAT_DATA = ROOT.parent / "habitat" / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HABITAT_ROOT / "habitat-lab"))

import habitat  # noqa: E402

from integrations.v1.r2r_ce_adapter import (  # noqa: E402
    R2R_CE_OVERRIDES,
    VLNAgentProcess,
    habitat_action,
    unpack_observation,
)
from integrations.run_artifacts import RunArtifactWriter, safe_name  # noqa: E402


def _build_navmesh_map(env, resolution=1024):
    """Render the current scene floor's navigable regions from Habitat's navmesh."""
    from habitat.utils.visualizations.maps import get_topdown_map_from_sim

    return get_topdown_map_from_sim(
        env.sim, map_resolution=resolution, draw_border=True
    )


def _render_topdown(env, navmesh_map, positions, goal_position, output_height):
    """Overlay the actual 3D agent positions on the navigable-area map."""
    import cv2
    import numpy as np
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
    """Place the RGB observation next to a scene TopDownMap frame."""
    import numpy as np

    return np.concatenate((rgb, topdown), axis=1)


def main():
    parser = argparse.ArgumentParser(description="Run VLN agent on Habitat R2R-CE")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--scene-id", default="all")
    parser.add_argument("--episodes", type=int, default=1, help="0 = all episodes in split")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpu-id", type=int, default=0)
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
    parser.add_argument("--run-id", help="Optional stable run ID.")
    parser.add_argument("--record-video", action="store_true", help="Save each episode as MP4")
    parser.add_argument(
        "--video-dir",
        help="Legacy media root; used as artifact root when --artifact-dir is omitted.",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_dir or args.video_dir or "artifacts"
    artifacts = RunArtifactWriter(
        artifact_root,
        run_id=args.run_id or os.environ.get("R2R_RUN_ID"),
        run_config={
            "runner": "run_habitat",
            "split": args.split,
            "scene_id": args.scene_id,
            "episodes_requested": args.episodes,
            "max_steps": args.max_steps,
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
            episodes = env.episodes if args.episodes == 0 else env.episodes[:args.episodes]
            env.episodes = episodes
            total = len(episodes)
            print("split={} episodes={} scene_id={}".format(args.split, total, args.scene_id))

            for idx, ep in enumerate(episodes):
                observation = env.reset()
                rgb, instruction = unpack_observation(observation)
                agent.reset(instruction)
                steps = 0
                actions = []
                episode_error = None
                frames = [] if args.record_video else None
                navmesh_map = None
                if args.record_video:
                    try:
                        navmesh_map = _build_navmesh_map(env)
                    except Exception as exc:
                        episode_error = (
                            "Top-down video fallback to RGB: "
                            "{}: {}".format(type(exc).__name__, exc)
                        )
                goal_position = ep.goals[0].position if ep.goals else None
                positions = [env.sim.get_agent_state().position.copy()]
                if frames is not None:
                    if navmesh_map is None:
                        frames.append(rgb.copy())
                    else:
                        topdown = _render_topdown(
                            env,
                            navmesh_map,
                            positions,
                            goal_position,
                            rgb.shape[0],
                        )
                        frames.append(_topdown_panel(rgb, topdown))
                while not env.episode_over:
                    try:
                        action = agent.act(rgb)
                    except RuntimeError as exc:
                        if "returned invalid action" not in str(exc):
                            raise
                        print(
                            "episode={} ignored_invalid_action={!r}".format(
                                env.current_episode.episode_id, str(exc)
                            ),
                            flush=True,
                        )
                        episode_error = str(exc)
                        break
                    actions.append(action)
                    observation = env.step(habitat_action(action))
                    steps += 1
                    rgb, _ = unpack_observation(observation)
                    positions.append(env.sim.get_agent_state().position.copy())
                    if frames is not None:
                        if navmesh_map is None:
                            frames.append(rgb.copy())
                        else:
                            topdown = _render_topdown(
                                env,
                                navmesh_map,
                                positions,
                                goal_position,
                                rgb.shape[0],
                            )
                            frames.append(_topdown_panel(rgb, topdown))
                finish_response = agent.finish_episode(rgb)
                metrics = env.get_metrics()
                status = "OK" if float(metrics.get("success", 0)) >= 1.0 else "FAIL"
                print(
                    "[{:>4}/{}] {}  steps={:<4}  {}  s={:.3f}  spl={:.3f}  dtg={:.2f}".format(
                        idx + 1, total, status, steps,
                        env.current_episode.episode_id,
                        float(metrics.get("success", 0)),
                        float(metrics.get("spl", 0)),
                        float(metrics.get("distance_to_goal", 0)),
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
                if navmesh_map is not None:
                    from PIL import Image

                    map_path = artifacts.media_path(
                        artifact_episode_id,
                        "topdown",
                    )
                    topdown = _render_topdown(
                        env, navmesh_map, positions, goal_position, rgb.shape[0]
                    )
                    Image.fromarray(topdown).save(str(map_path))
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
                    error=episode_error,
                )
    finally:
        agent.close()
        os.chdir(old_cwd)
    print("Run artifacts: {}".format(artifacts.results_path), flush=True)


if __name__ == "__main__":
    main()
