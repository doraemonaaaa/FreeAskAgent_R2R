import gzip
from hashlib import sha256
import json

import pytest

from integrations.v3.collect_calibration_route import next_route_action, training_episode


def test_intermediate_waypoint_stop_never_becomes_navigation_stop():
    class Follower:
        def get_next_action(self, target):
            return {"reached": 0, "next": 2}[target]
    assert next_route_action(Follower(), ["reached", "next"], 0) == (2, 1)
    assert next_route_action(Follower(), ["reached"], 0) == (0, 1)


def test_follower_failure_is_not_arrival():
    class Follower:
        def get_next_action(self, target):
            return None
    with pytest.raises(RuntimeError, match="could not produce"):
        next_route_action(Follower(), ["target"], 0)


def test_training_scene_and_dataset_are_locked(tmp_path):
    dataset = tmp_path / "train.json.gz"
    dataset.write_bytes(gzip.compress(json.dumps({"episodes": [{"episode_id": 1, "scene_id": "scene"}]}).encode()))
    plan = tmp_path / "plan.json"
    content = dict(dataset_sha256={"train": sha256(dataset.read_bytes()).hexdigest()}, scene_splits={"scene": "train"})
    plan.write_text(json.dumps(content))
    assert training_episode(dataset, plan, "1")[0]["episode_id"] == 1
    content["scene_splits"]["scene"] = "test"
    plan.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="reserved"):
        training_episode(dataset, plan, "1")
    with pytest.raises(ValueError, match='reserved'):
        training_episode(dataset, plan, '1', calibration_split='validation')
    content['scene_splits']['scene'] = 'validation'
    plan.write_text(json.dumps(content))
    assert training_episode(dataset, plan, '1', calibration_split='validation')[0]['episode_id'] == 1
    with pytest.raises(ValueError, match='reserved'):
        training_episode(dataset, plan, '1')
    with pytest.raises(ValueError, match='sealed'):
        training_episode(dataset, plan, '1', calibration_split='test')
    dataset.write_bytes(gzip.compress(b'{"episodes": []}'))
    with pytest.raises(ValueError, match="checksum"):
        training_episode(dataset, plan, "1")
def test_preview_sidecar_is_fixed_direction_and_not_temporal(tmp_path):
    import numpy as np
    from hashlib import sha256
    from integrations.v3.collect_calibration_route import save_preview_snapshot
    calls = []
    def render(env, yaws, hfov, scale):
        calls.append((yaws, hfov, scale))
        return [dict(yaw_deg=yaw, rgb=np.zeros((2, 3, 3), dtype=np.uint8),
                     depth=np.ones((2, 3)), intrinsics=np.eye(3), camera_to_world=np.eye(4))
                for yaw in yaws]
    record = save_preview_snapshot(None, 9, tmp_path, render)
    assert calls == [([-90, -45, 0, 45, 90], 90, 1.0)]
    assert record['frame_id'] == 9 and record['kind'] == 'simultaneous_preview'
    assert len(record['views']) == 5
    assert not (tmp_path / 'observations.jsonl').exists()
    assert set(record) == {'frame_id', 'kind', 'views'}
    for view in record['views']:
        assert sha256((tmp_path / view['rgb']).read_bytes()).hexdigest() == view['rgb_sha256']
        assert sha256((tmp_path / view['depth']).read_bytes()).hexdigest() == view['depth_file_sha256']
