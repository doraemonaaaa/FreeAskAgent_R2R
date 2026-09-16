from integrations.v3.preview_protocol import preview_headings, execution_observation
from integrations.v3.preview_protocol import preview_headings_for_request
import math
import pytest


def test_every_configured_heading_is_rendered_once_in_order():
    angles = (0, -45, 45, -90, 90, 180)
    assert preview_headings(angles) == (0.0, -45.0, 45.0, -90.0, 90.0, 180.0)
    assert preview_headings((0, 90, 90, -90)) == (0.0, 90.0, -90.0)


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
def test_preview_bearings_survive_turns_translation_and_pitch(start, turn):
    configured = (0, -45, 45, -90, 90, 180)
    request = {"source_camera_to_world": camera_pose(start, 15)}
    actual = preview_headings_for_request(configured, request, camera_pose(start + turn, -10, (3, 2, -7)))
    expected = tuple((v - turn + 180) % 360 - 180 for v in preview_headings(configured))
    assert actual == pytest.approx(expected)


def test_preview_without_source_pose_keeps_camera_relative_headings():
    assert preview_headings_for_request((0, 45, 90), {}, camera_pose(30)) == (0, 45, 90)


def test_invalid_heading_fails_instead_of_rendering_a_different_direction():
    with pytest.raises(ValueError, match="heading"):
        preview_headings_for_request((0,), {"source_camera_to_world": camera_pose(0, 90)}, camera_pose(0))
