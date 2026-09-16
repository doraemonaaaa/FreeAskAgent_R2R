"""Per-step log lines, rank summaries and the JSONL decision trace."""

import json

from . import settings
from .video import previewed_view


def step_timings(decision):
    """The worker's phase timings plus transport and Preview rendering."""
    timings = dict(decision.get("timings") or {})
    timings["worker_ms"] = float(timings.get("total_ms", 0.0))
    timings["encode_ms"] = float(decision.get("encode_ms", 0.0))
    timings["roundtrip_ms"] = float(decision.get("roundtrip_ms", 0.0))
    timings["preview_render_ms"] = float((decision.get("preview") or {}).get("render_ms", 0.0))
    return timings



def action_summary(decision, action):
    """Describe the executed action for one log line.

    Four decisions reach this point and only one of them has coordinates, so
    the pixel is read last rather than assumed.
    """
    if decision.get("stop"):
        return "STOP"
    if decision.get("turn_deg"):
        turn_deg = int(decision["turn_deg"])
        return "TURN request={:+d}deg execute={:+d}deg x1 a={}".format(
            turn_deg,
            settings.TURN_ANGLE_DEG if turn_deg > 0 else -settings.TURN_ANGLE_DEG,
            int(action),
        )
    if decision.get("forward_steps"):
        return "FWD request=x{} execute=x1 a={}".format(
            int(decision["forward_steps"]), int(action)
        )
    pixel = decision.get("pixel_uv")
    if pixel is None:
        return "PREVIEW a={}".format(int(action))
    return "({},{}) d={:.2f} a={}".format(
        pixel[0], pixel[1], decision.get("depth_m", 0.0), int(action)
    )


def step_line(episode_id, steps, decision, step_ms, action):
    """One line per step: where the time went, memory state, and the action.

    ``--debug-memory`` adds the full per-memory dumps below this line.
    """
    timings = step_timings(decision)
    task = decision.get("task_memory") or {}
    temporal = decision.get("temporal_memory") or {}
    select_ms = timings.get("select_pixel_ms", 0.0)
    captioner_ms = timings.get("captioner_ms", 0.0)
    preview_render_ms = timings.get("preview_render_ms", 0.0)
    debug = decision.get("debug") or {}
    analyzed = debug.get("analyzed_subgoal") or {}
    analyzed_id = analyzed.get("subgoal_id")
    current_id = task.get("current_subgoal_id")
    line = (
        "ep={} s={} {:.0f}ms [sel={:.0f} cap={:.0f} rest={:.0f}] "
        "sg={}->{} mode={} win={} obs={} | cap={} act={}".format(
            episode_id, steps, step_ms,
            select_ms, captioner_ms,
            step_ms - select_ms - captioner_ms - preview_render_ms,
            analyzed_id, current_id,
            temporal.get("active_error_mode"),
            len(temporal.get("frame_ids") or ()),
            task.get("observation_count"),
            caption_summary(decision),
            action_summary(decision, action),
        )
    )
    preview = decision.get("preview") or {}
    if preview:
        selected = previewed_view(decision) or {}
        line += " | preview=view{}/{} yaw={:+.0f} render={:.0f}ms".format(
            selected.get("view_index", "-"),
            len(preview.get("yaws_deg") or ()),
            float(selected.get("view_yaw_deg") or 0.0),
            preview_render_ms,
        )
    spatial = debug.get("spatial_summary")
    if spatial and spatial != "sp=-":
        line += " | " + spatial
    if debug.get("som_choice") is not None:
        line += " som={}/{}".format(
            debug.get("som_choice"), len(debug.get("som_candidates") or ())
        )
    # Surface only the abnormal cases inline; the rest stays behind the flag.
    if decision.get("temporal_error"):
        line += " ANALYSIS_ERROR_STAGE={} ANALYSIS_ERROR={!r}".format(
            decision.get("captioner_failed_stage") or "unknown",
            decision.get("temporal_error"),
        )
    if debug.get("spatial_error"):
        line += " SPATIAL_ERROR={!r}".format(debug.get("spatial_error"))
    return line



def caption_summary(decision):
    """Condense this step's Captioner verdict, or mark that it did not run."""
    if not decision.get("captioner_ran_this_step"):
        return "-"
    return "{}{}".format(
        "DONE" if decision.get("captioner_completed") else "wip",
        "" if decision.get("captioner_error_mode") == "NONE"
        else "/" + str(decision.get("captioner_error_mode")),
    )



# Habitat measurements, then the runner's own: oracle_success (came within
# the 3 m success radius at any step), path_length (metres walked), steps, and
# nDTW / SDTW against the dense reference path (path_metrics.py; only when the
# split has a ``{split}_gt.json.gz``).
METRIC_NAMES = ("success", "spl", "distance_to_goal", "oracle_success", "path_length", "steps", "ndtw", "sdtw")


def empty_totals():
    return {name: 0.0 for name in METRIC_NAMES}


def write_rank_summary(output_dir, rank, count, totals):
    """Emit this shard's totals in the layout aggregate_r2r_ce_results.py reads."""
    result = {"rank": rank, "count": count, "totals": totals}
    print("rank_summary={}".format(json.dumps(result, sort_keys=True)), flush=True)
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rank_{}.json".format(rank)).open("w") as handle:
        json.dump(result, handle, sort_keys=True)


def write_step_trace(
    handle,
    *,
    episode_id,
    step,
    decision,
    action,
    position_before,
    position_after,
    distance_before,
    distance_after,
):
    """Persist the complete non-image decision while the episode is running."""
    if handle is None:
        return
    traced_decision = dict(decision or {})
    # Base64 visualization panels can be reconstructed from the MP4 and make
    # a long JSONL trace unnecessarily huge. All model/memory evidence stays.
    traced_decision.pop("visuals", None)
    payload = {
        "episode_id": str(episode_id),
        "step": int(step),
        "action": int(action),
        "position_before": [float(value) for value in position_before],
        "position_after": [float(value) for value in position_after],
        "distance_to_goal_before": float(distance_before),
        "distance_to_goal_after": float(distance_after),
        "decision": traced_decision,
    }
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

