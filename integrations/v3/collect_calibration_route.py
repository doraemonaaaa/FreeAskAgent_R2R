"""Offline visual trajectory collector within the official TRAIN dataset.

Frozen training scenes are the default; validation scenes require explicit mode.
Test scenes remain sealed. Reference replay is not closed-loop evaluation.

Reference waypoints drive locomotion, not semantic labels. Oracle metadata stays
in collector_private.json; observations contain only sensed/executed information.
No Captioner, Planner, model service or navigation worker is constructed.
"""
import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path


def training_episode(dataset_path, plan_path, episode_id, *, calibration_split='train'):
    if calibration_split not in {'train', 'validation'}:
        raise ValueError('only explicit train or validation collection allowed; test stays sealed')
    dataset_bytes = Path(dataset_path).read_bytes()
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    if sha256(dataset_bytes).hexdigest() != plan["dataset_sha256"]["train"]:
        raise ValueError("official training dataset checksum differs from frozen plan")
    episodes = json.loads(gzip.decompress(dataset_bytes))["episodes"]
    matches = [e for e in episodes if str(e["episode_id"]) == str(episode_id)]
    if len(matches) != 1:
        raise ValueError("one official training episode required")
    episode = matches[0]
    if plan["scene_splits"].get(episode["scene_id"]) != calibration_split:
        raise ValueError("reserved scene does not match explicit calibration split")
    return episode, sha256(plan_bytes).hexdigest()


def next_route_action(follower, targets, index):
    """Advance completed waypoints without emitting intermediate STOP actions."""
    while index < len(targets):
        action = follower.get_next_action(targets[index])
        if action is None:
            raise RuntimeError("reference follower could not produce an action")
        if int(action) != 0:
            return int(action), index
        index += 1
    return 0, index


def save_preview_snapshot(env, frame, output, renderer):
    """Task-independent simultaneous views, in a sidecar not the time stream."""
    import numpy as np
    from PIL import Image
    views = renderer(env, [-90, -45, 0, 45, 90], 90, scale=1.0)
    if len(views) != 5:
        raise ValueError("incomplete fixed-direction Preview capture")
    records = []
    for index, view in enumerate(views):
        stem = "preview_{:06d}_view_{}".format(frame, index)
        rgb_path = output / (stem + ".png")
        depth_path = output / (stem + ".depth.npy")
        Image.fromarray(view["rgb"]).save(rgb_path)
        np.save(depth_path, np.asarray(view["depth"], dtype=np.float32), allow_pickle=False)
        records.append(dict(view=index, yaw_deg=float(view["yaw_deg"]),
            rgb=rgb_path.name, rgb_sha256=sha256(rgb_path.read_bytes()).hexdigest(),
            depth=depth_path.name, depth_file_sha256=sha256(depth_path.read_bytes()).hexdigest(),
            intrinsics=view["intrinsics"].tolist(), camera_to_world=view["camera_to_world"].tolist(),
            depth_max_m=10.0))
    return dict(frame_id=frame, kind="simultaneous_preview", views=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument('--calibration-split', choices=('train', 'validation'), default='train')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--camera-pitch-deg", type=float, default=-15)
    parser.add_argument("--preview-interval", type=int, default=0,
                        help="Optional fixed-direction sidecar every N real frames, starting at frame 1; 0 disables")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("positive max steps required")
    if args.preview_interval < 0:
        parser.error("preview interval must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    dataset = root.parent / "habitat/data/datasets/vln/mp3d/r2r/v1/train/train.json.gz"
    episode, plan_hash = training_episode(dataset, args.split_plan, args.episode_id, calibration_split=args.calibration_split)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Imports deliberately follow split/provenance validation; unit tests need no GPU.
    import numpy as np
    from PIL import Image
    from integrations.v3.run_habitat import (habitat, HABITAT_ROOT, HABITAT_DATA, CAMERA,
        FORWARD_STEP_M, TURN_ANGLE_DEG, motion_overrides, sensor_overrides,
        rgb_depth, camera_intrinsics, camera_to_world_matrix, render_preview_views)
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # oracle replay of the GT route
    overrides = motion_overrides(FORWARD_STEP_M, TURN_ANGLE_DEG) + sensor_overrides(
        CAMERA, CAMERA.width, CAMERA.height, CAMERA.hfov_deg, 0.0, 10.0, False) + [
        "habitat.dataset.split=train",
        "habitat.dataset.data_path='{}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz'".format(HABITAT_DATA),
        "habitat.dataset.scenes_dir={}/scene_datasets".format(HABITAT_DATA),
        "habitat.environment.max_episode_steps={}".format(args.max_steps),
        "habitat.dataset.content_scenes=[{}]".format(Path(episode["scene_id"]).stem)]
    for sensor in ("rgb", "depth"):
        overrides.append("habitat.simulator.agents.main_agent.sim_sensors.{}_sensor.orientation=[{},0,0]".format(
            sensor, float(np.deg2rad(args.camera_pitch_deg))))  # the collector's own pitch wins
    config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
    private = dict(episode=episode, split_plan_sha256=plan_hash, collector="reference_route",
                   calibration_split=args.calibration_split,
                   meaning="ORACLE COLLECTION CONTROL ONLY; never judgment input or semantic labels")
    (output / "collector_private.json").write_text(json.dumps(private, indent=2))
    cwd = Path.cwd()
    try:
        os.chdir(HABITAT_ROOT)
        with habitat.Env(config=config) as env:
            env.episodes = [e for e in env.episodes if str(e.episode_id) == str(args.episode_id)]
            if len(env.episodes) != 1:
                raise ValueError("Habitat training episode selection mismatch")
            observation = env.reset()
            targets = list(episode["reference_path"]) + [episode["goals"][0]["position"]]
            follower = ShortestPathFollower(env.sim, 0.3, return_one_hot=False)
            target_index, previous_action, steps, frame = 0, None, 0, 0
            reached = False
            preview_records = []
            with (output / "observations.jsonl").open("x") as stream:
                while True:
                    frame += 1
                    rgb, depth = rgb_depth(observation)
                    rgb_path = output / "frame_{:06d}.png".format(frame)
                    depth_path = output / "frame_{:06d}.depth.npy".format(frame)
                    Image.fromarray(rgb).save(rgb_path)
                    np.save(depth_path, np.asarray(depth, dtype=np.float32), allow_pickle=False)
                    row = dict(frame_id=frame, instruction=episode["instruction"]["instruction_text"].strip(),
                        rgb=rgb_path.name, rgb_sha256=sha256(rgb_path.read_bytes()).hexdigest(),
                        depth=depth_path.name, depth_file_sha256=sha256(depth_path.read_bytes()).hexdigest(),
                        depth_max_m=10.0, intrinsics=camera_intrinsics(rgb.shape[1], rgb.shape[0], 90).tolist(),
                        camera_to_world=camera_to_world_matrix(env).tolist(),
                        previous_action=previous_action, collision=bool(env.sim.previous_step_collided))
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if args.preview_interval and (frame - 1) % args.preview_interval == 0:
                        before = camera_to_world_matrix(env).copy()
                        preview_records.append(save_preview_snapshot(env, frame, output, render_preview_views))
                        if not np.allclose(before, camera_to_world_matrix(env), atol=1e-6, rtol=0):
                            raise ValueError("Preview rendering did not restore the real camera pose")
                        camera_height = float(before[1, 3] - env.sim.get_agent_state().position[1])
                        preview_records[-1]["camera_height_m"] = camera_height
                    if env.episode_over or steps >= args.max_steps:
                        break
                    action, target_index = next_route_action(follower, targets, target_index)
                    if action == 0:
                        reached = True
                        break
                    observation = env.step(action)
                    previous_action = action
                    steps += 1
            result = dict(frames=frame, actions=steps, collector_route_finished=reached,
                          calibration_split=args.calibration_split,
                          semantic_labels=0, model_calls=0,
                          observations_sha256=sha256((output / "observations.jsonl").read_bytes()).hexdigest())
            if args.preview_interval:
                preview_path = output / "preview_observations.jsonl"
                with preview_path.open("x") as stream:
                    for record in preview_records:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                result.update(preview_interval=args.preview_interval, preview_snapshots=len(preview_records),
                              preview_sha256=sha256(preview_path.read_bytes()).hexdigest())
            (output / "collection_result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
