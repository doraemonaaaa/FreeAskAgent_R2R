from integrations.v3.preview_protocol import preview_for_unseen_frame, preview_headings, execution_observation
from integrations.v3.preview_protocol import preview_headings_for_request
import math
import pytest


def test_directional_coverage_and_rear_request():
    angles = (-90, -45, 0, 45, 90)
    assert preview_headings(angles, "LEFT") == (-90, -45, 0)
    assert preview_headings(angles, "RIGHT") == (0, 45, 90)
    assert preview_headings(angles, "BACK") == (-90, 90, 180)
    assert preview_headings(angles, "UNKNOWN") == angles
    assert preview_headings(angles, "CENTER") == angles


def test_custom_angles_are_preserved_not_replaced_with_scene_specific_views():
    assert preview_headings((-70, -20, 10, 60), "LEFT") == (-70, -20)
    assert preview_headings((20,), "LEFT") == (20,)


def test_preview_waits_for_an_unanalyzed_frame_without_changing_request():
    pending = {"recovery_id": "episode:preview:1", "direction": "LEFT"}
    assert preview_for_unseen_frame(pending, True) is None
    assert preview_for_unseen_frame(pending, False) is pending
    assert preview_for_unseen_frame(None, False) is None


def test_execution_feedback_never_leaks_oracle_navigation_metrics():
    assert execution_observation(1, {"distance_to_goal": 12, "success": False,
                                     "collisions": {"is_collision": True, "count": 4}}) == {
        "action": 1, "collision": True,
    }
    assert execution_observation(3, {}) == {"action": 3, "collision": None}


def camera_pose(yaw, pitch=0, origin=(0, 1.5, 0)):
    y, p = math.radians(yaw), math.radians(pitch)
    cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    return [[cy, -sy * sp, -sy * cp, origin[0]],
            [0, cp, -sp, origin[1]],
            [sy, cy * sp, cy * cp, origin[2]], [0, 0, 0, 1]]


@pytest.mark.parametrize("start,turn", [(0, 30), (120, -90), (170, 30), (-170, -45)])
@pytest.mark.parametrize("direction", ["LEFT", "RIGHT", "BACK", "UNKNOWN"])
def test_preview_bearings_survive_turns_translation_and_pitch(start, turn, direction):
    configured = (-90, -45, 0, 45, 90)
    request = {"direction": direction, "source_camera_to_world": camera_pose(start, 15)}
    actual = preview_headings_for_request(configured, request, camera_pose(start + turn, -10, (3, 2, -7)))
    expected = tuple((v - turn + 180) % 360 - 180 for v in preview_headings(configured, direction))
    assert actual == pytest.approx(expected)


def test_legacy_preview_without_pose_remains_supported():
    assert preview_headings_for_request((0, 45, 90), {"direction": "RIGHT"}, camera_pose(30)) == (0, 45, 90)


def test_invalid_heading_fails_instead_of_rendering_a_different_direction():
    with pytest.raises(ValueError, match="heading"):
        preview_headings_for_request((0,), {"source_camera_to_world": camera_pose(0, 90)}, camera_pose(0))
