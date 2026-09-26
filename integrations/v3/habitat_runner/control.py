"""Turning a waypoint or turn request into one Habitat primitive."""

import math

import numpy as np

from . import settings


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def turn_primitive(turn_deg):
    """Resolve a requested turn to exactly one simulator primitive.

    Positive is to the right, matching ``yaw_delta_deg`` everywhere else. The
    magnitude is validated against the simulator angle, but is deliberately not
    executed open-loop.  The next 15-degree observation must pass through the
    Actor and TemporalMemory before another primitive can be issued.
    """
    turn_deg = int(turn_deg)
    if turn_deg == 0 or turn_deg % settings.TURN_ANGLE_DEG:
        raise ValueError(
            "turn_deg={} is not a non-zero multiple of the simulator's "
            "turn_angle={}".format(turn_deg, settings.TURN_ANGLE_DEG)
        )
    return 3 if turn_deg > 0 else 2  # turn_right / turn_left


class GeometricFollower:
    """Waypoint execution as a real robot's local controller would do it.

    The agent already plans on its own occupancy grid and hands over the next
    point along a free path, so the controller only has to face the point,
    then step. ``get_next_action`` returns 0 when the point is inside the goal
    radius, otherwise one primitive.
    """

    def __init__(self, env, goal_radius, turn_angle_deg=None):
        self.env = env
        self.goal_radius = float(goal_radius)
        turn_angle_deg = settings.TURN_ANGLE_DEG if turn_angle_deg is None else turn_angle_deg
        self.turn_angle_deg = float(turn_angle_deg)

    def get_next_action(self, waypoint):
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.env.sim.get_agent_state()
        position = np.asarray(state.position, dtype=np.float64)
        goal = np.asarray(waypoint, dtype=np.float64)
        dx, dz = float(goal[0] - position[0]), float(goal[2] - position[2])
        if math.hypot(dx, dz) < self.goal_radius:
            return 0
        forward = quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
        # Signed bearing about +Y; positive means the point is to the left,
        # which is the direction Habitat's turn_left rotates.
        bearing = math.degrees(math.atan2(
            forward[2] * dx - forward[0] * dz, forward[0] * dx + forward[2] * dz))
        # One turn unit of slack: stepping 25 cm at up to 15 deg off the line
        # still closes distance, and the next observation re-aims anyway.
        if abs(bearing) <= self.turn_angle_deg:
            return 1  # move_forward
        return 2 if bearing > 0 else 3  # turn_left / turn_right


def fallback_action_for_follower_stop(decision):
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

