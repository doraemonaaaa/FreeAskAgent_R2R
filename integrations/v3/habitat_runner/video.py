"""Video recording: top-down maps, the agent's own map panel and frame overlays."""

import base64
import io

import numpy as np
from PIL import Image


def floor_map_key(height):
    """Stable half-metre cache key; floors remain separate while stairs blend."""
    return round(float(height) * 2.0) / 2.0


def build_navmesh_map(env, resolution=1024, height=None):
    """Render one floor slice of the scene navmesh."""
    from habitat.utils.visualizations.maps import get_topdown_map

    if height is None:
        height = env.sim.get_agent_state().position[1]
    return get_topdown_map(
        env.sim.pathfinder,
        float(height),
        map_resolution=resolution,
        draw_border=True,
    )


def navmesh_map_for_height(env, cache, height):
    """Return the current floor slice without rebuilding it every video frame."""
    key = floor_map_key(height)
    if key not in cache:
        cache[key] = build_navmesh_map(env, height=height)
    return key, cache[key]


def render_topdown(
    env,
    navmesh_map,
    positions,
    goal_position,
    output_height,
    waypoints=(),
    floor_height=None,
):
    """Draw the executed trajectory, start, current position, and goal.

    ``waypoints`` are the world-space targets the actor asked for, which show
    where it intended to go as opposed to where the follower took it.
    """
    import cv2
    from habitat.utils.visualizations import maps

    if floor_height is None:
        floor_height = float(env.sim.get_agent_state().position[1])
    floor_tolerance = 0.75
    image = maps.colorize_topdown_map(navmesh_map.copy())
    rows, columns = navmesh_map.shape[:2]

    def to_pixel(position):
        row, col = maps.to_grid(
            position[2], position[0], navmesh_map.shape,
            pathfinder=env.sim.pathfinder,
        )
        # A requested waypoint can land off the navmesh, and to_grid does not
        # clamp; an out-of-bounds point would otherwise be drawn nowhere.
        return (
            int(np.clip(col, 0, columns - 1)),
            int(np.clip(row, 0, rows - 1)),
        )

    floor_positions = [
        position for position in positions
        if abs(float(position[1]) - floor_height) <= floor_tolerance
    ]
    floor_waypoints = [
        waypoint for waypoint in waypoints
        if abs(float(waypoint[1]) - floor_height) <= floor_tolerance
    ]
    path = [to_pixel(position) for position in floor_positions]
    if len(path) > 1:
        cv2.polylines(image, [np.asarray(path, dtype=np.int32)], False, (0, 80, 255), 3)
    # Drawn under the trajectory endpoints so the executed path stays legible.
    for waypoint in floor_waypoints:
        cv2.circle(image, to_pixel(waypoint), 3, _REQUESTED_COLOR, -1, cv2.LINE_AA)
    if floor_waypoints and path:
        cv2.line(
            image, path[-1], to_pixel(floor_waypoints[-1]), _REQUESTED_COLOR, 1,
            cv2.LINE_AA,
        )
    if path:
        cv2.circle(image, path[0], 7, (0, 180, 0), -1)
        cv2.circle(image, path[-1], 7, (255, 80, 0), -1)
    if (
        goal_position is not None
        and abs(float(goal_position[1]) - floor_height) <= floor_tolerance
    ):
        cv2.drawMarker(image, to_pixel(goal_position), (255, 0, 0), cv2.MARKER_STAR, 16, 2)

    # Habitat maps use the full scene bounds, which can leave the active floor
    # as a tiny island in a large blank canvas. Crop to the valid slice, then
    # resize to a fixed square so every MP4 frame keeps identical dimensions.
    valid_rows, valid_cols = np.where(navmesh_map != 0)
    if valid_rows.size and valid_cols.size:
        margin = max(8, int(0.02 * max(navmesh_map.shape)))
        row0 = max(0, int(valid_rows.min()) - margin)
        row1 = min(rows, int(valid_rows.max()) + margin + 1)
        col0 = max(0, int(valid_cols.min()) - margin)
        col1 = min(columns, int(valid_cols.max()) + margin + 1)
        image = image[row0:row1, col0:col1]
    image = cv2.resize(
        image, (output_height, output_height), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        image,
        "floor y={:+.2f}m".format(float(floor_height)),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return image


def topdown_panel(rgb, topdown, agent_map=None):
    """First-person view | the agent's own map (when given) | true top-down.

    The middle panel is what Spatial Memory believes: unknown grey, free
    white, occupied black, trail blue, landmarks green, committed target red,
    set-of-mark candidates yellow, the agent orange. Comparing it with the
    ground-truth map on the right shows where the belief went wrong.
    """
    panels = [rgb]
    if agent_map is not None:
        height = rgb.shape[0]
        # Both dimensions, not only the height: a map that is already the right
        # height but a different width used to widen the whole frame, and a
        # video whose frames change size is rejected by the encoder.
        if agent_map.shape[:2] != (height, height):
            agent_map = np.asarray(
                Image.fromarray(agent_map).resize((height, height), Image.Resampling.NEAREST)
            )
        panels.append(agent_map)
    panels.append(topdown)
    return np.concatenate(panels, axis=1)


PREVIEW_STRIP_HEIGHT = 136


def preview_strip(views, width, *, selected_yaw=None, label=None,
                  height=PREVIEW_STRIP_HEIGHT, stale=False):
    """A filmstrip of one surrounding-view ring, read left to right as a panorama.

    ``views`` is a sequence of ``(yaw_deg, rgb)`` pairs; they are ordered by
    heading, not by capture order, so the strip reads from the robot's left to
    its right. The view the step committed to is outlined in cyan - a strip
    with no outlined cell is a ring the model looked at and passed over, which
    is what the recordings could not show before. ``stale`` dims a ring carried
    over from an earlier step so it is never mistaken for a fresh one.
    """
    import cv2

    def heading(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    band = np.zeros((height, width, 3), dtype=np.uint8)
    band[:] = (24, 24, 24)
    # An unlabelled view is still shown, last: dropping it would hide the very
    # thing the strip exists to reveal.
    ordered = sorted(((heading(yaw), rgb) for yaw, rgb in views),
                     key=lambda item: (item[0] is None, item[0] or 0.0))
    if not ordered:
        if label:
            import cv2 as _cv2
            _cv2.putText(band, label, (6, height // 2), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.42, (120, 120, 120), 1, _cv2.LINE_AA)
        return band
    margin, gap, text_h = 6, 4, 18
    cell_w = max((width - 2 * margin - gap * (len(ordered) - 1)) // len(ordered), 24)
    cell_h = height - 2 * margin - text_h
    bright, dim = (40, 225, 255), (120, 120, 120)
    for index, (yaw, rgb) in enumerate(ordered):
        x0 = margin + index * (cell_w + gap)
        thumb = np.asarray(Image.fromarray(np.asarray(rgb)).convert("RGB")
                           .resize((cell_w, cell_h), Image.Resampling.BILINEAR))
        if stale:
            thumb = (thumb * 0.45).astype(np.uint8)
        band[margin:margin + cell_h, x0:x0 + cell_w] = thumb
        chosen = (yaw is not None and selected_yaw is not None
                  and abs(yaw - float(selected_yaw)) < 1e-6)
        cv2.rectangle(band, (x0, margin), (x0 + cell_w - 1, margin + cell_h - 1),
                      bright if chosen else dim, 2 if chosen else 1, cv2.LINE_AA)
        caption = ("?deg" if yaw is None else "{:+.0f}deg".format(yaw))
        caption += " SELECTED" if chosen else ""
        cv2.putText(band, caption, (x0 + 3, height - margin - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, bright if chosen else dim, 1, cv2.LINE_AA)
    if label:
        # Top right, on its own dark backing: the bottom row belongs to the
        # per-view headings and a label there covers the last one.
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x0 = max(width - text_width - 2 * margin, margin)
        cv2.rectangle(band, (x0 - 4, 0), (width - 1, text_height + 9), (24, 24, 24), -1)
        cv2.putText(band, label, (x0, text_height + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, dim if stale else bright, 1, cv2.LINE_AA)
    return band


def decode_visuals(decision):
    """(agent_map, marker_frame) from a worker response, either may be None."""
    visuals = (decision or {}).get("visuals") or {}

    def _png(key):
        encoded = visuals.get(key)
        if not encoded:
            return None
        try:
            return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
        except Exception:
            return None

    return _png("map_png"), _png("som_png")


# Requested amber, executed green: the same pair the legend names.
_REQUESTED_COLOR = (255, 190, 0)
_APPLIED_COLOR = (0, 220, 90)

def previewed_view(decision):
    """Return the surrounding view a previewed step committed to, if any."""
    inner = (decision.get("decision") or {}).get("execution") or {}
    return inner if inner.get("view_index") is not None else None


def preview_ring_yaws(decision):
    """Headings the surrounding-view ring of this step was captured at."""
    yaws = (decision.get("preview") or {}).get("yaws_deg")
    if not isinstance(yaws, (list, tuple)):
        return ()
    return tuple(float(yaw) for yaw in yaws
                 if isinstance(yaw, (int, float)) and np.isfinite(yaw))


def draw_preview_indicator(image, decision):
    """Show the surrounding views this step looked at, and which it chose.

    One ray per captured heading, right positive from the current heading, at
    the angles actually captured - the robot reports measured yaws, which drift
    from the requested ring. The bright ray is the view the step committed to;
    a ring with no bright ray settled nothing, which is what a Preview loop
    looks like from outside (the Go2 run of 2026-09-18 drew seven of them).
    """
    import cv2

    previewed = previewed_view(decision)
    yaws = preview_ring_yaws(decision)
    if previewed is None and not yaws:
        return

    height, width = image.shape[:2]
    origin = (width // 2, max(int(height * 0.20), 55))
    length = max(int(min(width, height) * 0.14), 45)
    bright, dim = (40, 225, 255), (110, 150, 165)
    accent = bright if previewed is not None else dim

    def endpoint(yaw_deg, radius):
        yaw_rad = np.deg2rad(yaw_deg)
        return (int(np.clip(origin[0] + radius * np.sin(yaw_rad), 12, width - 13)),
                int(np.clip(origin[1] - radius * np.cos(yaw_rad), 12, height - 13)))

    for yaw in yaws:
        tip = endpoint(yaw, length)
        cv2.line(image, origin, tip, dim, 2, cv2.LINE_AA)
        cv2.circle(image, tip, 3, dim, -1, cv2.LINE_AA)
    if yaws:
        cv2.circle(image, origin, length, dim, 1, cv2.LINE_AA)
    cv2.circle(image, origin, 7, accent, 2, cv2.LINE_AA)
    if previewed is not None:
        yaw_deg = float(previewed.get("view_yaw_deg") or 0.0)
        cv2.arrowedLine(image, origin, endpoint(yaw_deg, length), bright, 3,
                        cv2.LINE_AA, tipLength=0.28)
        label = "PREVIEW {:+.0f}deg".format(yaw_deg)
        if yaws:
            label += " of {}".format(len(yaws))
    else:
        label = "PREVIEW {} views, no selection".format(len(yaws))
    (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    cv2.putText(
        image,
        label,
        (int(np.clip(origin[0] - text_width // 2, 4, width - text_width - 4)),
         min(origin[1] + length + 19, height - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        accent,
        2,
        cv2.LINE_AA,
    )


def draw_turn_arrow(image, turn_deg):
    """Overlay the actor's requested in-place turn on an RGB video frame.

    Positive angles point right and negative angles point left, matching the
    convention used by ``turn_primitive``.  A bent arrow is used instead of a
    straight horizontal arrow so it cannot be confused with an image-space
    waypoint direction.
    """
    import cv2

    if turn_deg is None or int(turn_deg) == 0:
        return

    turn_deg = int(turn_deg)
    height, width = image.shape[:2]
    direction = 1 if turn_deg > 0 else -1
    center_x = width // 2
    bend_y = max(int(height * 0.27), 42)
    stem_y = min(int(height * 0.46), height - 24)
    tip_x = int(np.clip(
        center_x + direction * max(int(width * 0.18), 55),
        24,
        width - 25,
    ))
    thickness = max(3, int(round(min(width, height) / 120.0)))
    color = (255, 210, 0)

    # A small translucent backing keeps the symbol readable in both bright
    # rooms and dark corridors without hiding much of the observation.
    overlay = image.copy()
    pad = 18
    left = max(min(center_x, tip_x) - pad, 0)
    right = min(max(center_x, tip_x) + pad, width - 1)
    top = max(bend_y - pad - 22, 0)
    bottom = min(stem_y + pad, height - 1)
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.38, image, 0.62, 0, dst=image)

    cv2.line(
        image, (center_x, stem_y), (center_x, bend_y),
        color, thickness, cv2.LINE_AA,
    )
    cv2.arrowedLine(
        image, (center_x, bend_y), (tip_x, bend_y),
        color, thickness, cv2.LINE_AA, tipLength=0.28,
    )
    label = "{} {}deg".format(
        "RIGHT" if direction > 0 else "LEFT", abs(turn_deg)
    )
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2,
    )
    text_x = int(np.clip(center_x - text_width // 2, 2, width - text_width - 2))
    text_y = max(top + text_height + 4, text_height + 2)
    cv2.putText(
        image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 2, cv2.LINE_AA,
    )


def wrap_overlay_text(cv2, text, max_width, scale, thickness):
    """Wrap one label by rendered pixel width without dropping source text."""
    words = str(text or "-").split()
    if not words:
        return ["-"]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_task_context(image, decision):
    """Show the original instruction and the currently active subgoal."""
    import cv2

    task = decision.get("task_memory") or {}
    debug = decision.get("debug") or {}
    current = debug.get("subgoal_after")
    instruction = task.get("goal") or "-"
    if current is None:
        subgoal_text = (
            "COMPLETE"
            if task.get("current_subgoal_id") is None
            else str(task.get("current_subgoal_id"))
        )
    else:
        subgoal_text = "{}: {}".format(
            current.get("subgoal_id", "-"),
            current.get("description") or "-",
        )

    scale, thickness = 0.44, 1
    margin, line_height = 8, 18
    max_width = max(image.shape[1] - 2 * margin, 1)
    lines = wrap_overlay_text(
        cv2, "Instruction: " + instruction, max_width, scale, thickness
    )
    lines.extend(wrap_overlay_text(
        cv2, "Subgoal: " + subgoal_text, max_width, scale, thickness
    ))
    strip_height = margin + line_height * len(lines) + 4
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1] - 1, min(strip_height, image.shape[0]) - 1),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, dst=image)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (margin, margin + line_height * (index + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return min(strip_height, image.shape[0])


CHAIN_DONE = (60, 180, 75)      # completed stage
CHAIN_ACTIVE = (250, 200, 40)   # active stage
CHAIN_TODO = (90, 90, 90)       # not reached
CHAIN_STUCK = (230, 90, 40)     # active stage held longer than CHAIN_STUCK_STEPS
CHAIN_STUCK_STEPS = 30


def new_subgoal_chain(subgoals):
    """Per-episode tracker for the subgoal-chain strip drawn on video frames."""
    return {
        "ids": [str(item.get("subgoal_id")) for item in subgoals],
        "stage_id": None,
        "entered_step": 0,
        "completed": [],
        "transition": False,
    }


def update_subgoal_chain(chain, decision, steps):
    """Advance the tracker from this step's decision; returns the tracker."""
    if chain is None:
        return None
    debug = decision.get("debug") or {}
    task = decision.get("task_memory") or {}
    after = debug.get("subgoal_after") or {}
    stage_id = after.get("subgoal_id")
    if stage_id is None:
        stage_id = task.get("current_subgoal_id")
    stage_id = None if stage_id is None else str(stage_id)
    chain["transition"] = False
    if stage_id != chain["stage_id"]:
        if chain["stage_id"] is not None:
            chain["transition"] = True
            # Every stage strictly before the new one counts as passed; the
            # planner may skip ahead (skip_to_final) over several stages.
            ids = chain["ids"]
            stop = ids.index(stage_id) if stage_id in ids else len(ids)
            for item in ids[:stop]:
                if item not in chain["completed"]:
                    chain["completed"].append(item)
        chain["stage_id"] = stage_id
        chain["entered_step"] = steps
    if decision.get("stop") and debug.get("stop_reason") == "ALL_SUBGOALS_COMPLETE":
        chain["completed"] = list(chain["ids"])
        chain["stage_id"] = None
    return chain


def draw_subgoal_chain(image, decision, chain, steps, y_top):
    """One box per planned stage under the instruction strip.

    Completed stages are green, the active one yellow (orange once it has
    been held for CHAIN_STUCK_STEPS), unreached ones grey. The active box
    shows how many steps the agent has spent in it. A dot to the right
    reports the Captioner this step: grey ran/in-progress, green accepted
    completion. The frame on which a
    stage advanced gets a yellow border so it is easy to find when scrubbing.
    """
    import cv2

    if not chain or not chain["ids"]:
        return y_top
    ids = chain["ids"]
    height, width = image.shape[:2]
    margin, gap, box_h = 8, 4, 20
    dot_room = 22
    usable = width - 2 * margin - dot_room
    box_w = max(min(90, (usable - gap * (len(ids) - 1)) // len(ids)), 14)
    y0 = y_top + 3
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y_top), (width - 1, y0 + box_h + 3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, dst=image)
    held = steps - chain["entered_step"]
    x = margin
    for item in ids:
        if item in chain["completed"]:
            color, label = CHAIN_DONE, item
        elif item == chain["stage_id"]:
            color = CHAIN_STUCK if held >= CHAIN_STUCK_STEPS else CHAIN_ACTIVE
            label = "{} +{}".format(item, held)
        else:
            color, label = CHAIN_TODO, item
        cv2.rectangle(image, (x, y0), (x + box_w, y0 + box_h), color, -1)
        cv2.putText(
            image, label, (x + 4, y0 + box_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (0, 0, 0), 1, cv2.LINE_AA,
        )
        x += box_w + gap
    if decision.get("captioner_ran_this_step"):
        dot = CHAIN_DONE if decision.get("captioner_completed") else (170, 170, 170)
        cv2.circle(image, (width - margin - 7, y0 + box_h // 2), 6, dot, -1)
    if chain.get("transition"):
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), CHAIN_ACTIVE, 4)
    return y0 + box_h + 3


def annotated_video_frame(rgb, decision, steps, chain=None):
    """Visualize only navigation actions and perception decisions.

    ``requested_pixel_uv`` is the location the waypoint policy asked for;
    ``pixel_uv`` is where the depth map allowed that waypoint to land. Drawing
    both, joined by a line, separates a bad model selection from a good
    selection that the walkable-pixel snap pulled somewhere else. PREVIEW is a
    cyan compass of every captured heading, the chosen one drawn as an arrow;
    an in-place turn is a yellow bent arrow. The top strip
    holds the route instruction and active subgoal; detailed diagnostics
    remain in terminal logs.
    """
    import cv2

    image = rgb.copy()
    height, width = image.shape[:2]
    debug = decision.get("debug") or {}
    strip_bottom = draw_task_context(image, decision)
    draw_subgoal_chain(image, decision, chain, steps, strip_bottom)

    def to_pixel(value):
        if not value:
            return None
        return (
            int(np.clip(int(value[0]), 0, width - 1)),
            int(np.clip(int(value[1]), 0, height - 1)),
        )

    # A previewed step chose its pixel inside a surrounding view, so those
    # coordinates address a different image than this one; drawing them here
    # would put markers on unrelated scenery.
    previewed = previewed_view(decision)
    requested = (
        None if previewed else to_pixel(debug.get("requested_pixel_uv"))
    )
    applied = None if previewed else to_pixel(decision.get("pixel_uv"))
    if requested is not None and applied is not None and requested != applied:
        cv2.line(image, requested, applied, (255, 255, 255), 1, cv2.LINE_AA)
    if requested is not None:
        cv2.circle(image, requested, 9, _REQUESTED_COLOR, 2, cv2.LINE_AA)
        cv2.drawMarker(
            image, requested, _REQUESTED_COLOR, cv2.MARKER_CROSS, 14, 1,
        )
    if applied is not None:
        cv2.circle(image, applied, 6, _APPLIED_COLOR, -1, cv2.LINE_AA)
        cv2.circle(image, applied, 6, (255, 255, 255), 1, cv2.LINE_AA)
    draw_preview_indicator(image, decision)
    draw_turn_arrow(image, decision.get("turn_deg"))
    return image

