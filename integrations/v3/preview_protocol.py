"""Pure Preview transport helpers, independent of Habitat and scene content."""

import math


def preview_headings(configured, direction):
    """Use the requested half-space; keep broad coverage without a prior.

    Angles are camera-relative, not world bearings or episode-specific turns.
    Explicit rear observations include 180 even with a front-only configuration.
    """
    headings = tuple(dict.fromkeys(float(y) for y in configured))
    center = {"LEFT": -90.0, "RIGHT": 90.0, "BACK": 180.0}.get(direction)
    if center is None:
        return headings
    if direction == "BACK" and 180.0 not in headings:
        headings += (180.0,)
    selected = tuple(y for y in headings if abs((y - center + 180) % 360 - 180) <= 90)
    return selected or headings


def preview_headings_for_request(configured, request, camera_to_world):
    """Keep the requested world bearings across queued physical turns.

    Source/current transforms are observations, not route or goal metadata.
    Translation does not change the bearing; the model receives fresh views
    from the current position and must judge whether the old question still fits.
    Legacy requests without a source pose retain camera-relative behavior.
    """
    headings = preview_headings(configured, request.get("direction"))
    source = request.get("source_camera_to_world")
    if source is None:
        return headings

    def yaw(pose):
        # Camera forward is -Z; positive yaw is to the agent's right.
        x, z = -float(pose[0][2]), -float(pose[2][2])
        if not math.isfinite(x) or not math.isfinite(z) or math.hypot(x, z) < 1e-8:
            raise ValueError("Preview requires a finite horizontal camera heading")
        return math.degrees(math.atan2(x, -z))

    offset = yaw(source) - yaw(camera_to_world)
    return tuple((angle + offset + 180.0) % 360.0 - 180.0 for angle in headings)


def execution_observation(action, metrics):
    """Only proprioceptive feedback, never oracle distance/goal/route labels."""
    collisions = metrics.get("collisions")
    collision = collisions.get("is_collision") if isinstance(collisions, dict) else None
    return {"action": int(action), "collision": collision}
