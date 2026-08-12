"""JSON-lines worker that owns the Python 3.12 RGB-D waypoint actor."""

import argparse
import base64
import io
import json
import sys

import numpy as np
from PIL import Image


def _decode_rgb(encoded):
    return np.asarray(
        Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    )


def _decode_array(encoded):
    with io.BytesIO(base64.b64decode(encoded)) as buffer:
        return np.load(buffer, allow_pickle=False)


def _temporal_report(actor):
    """Expose the Captioner's own judgement and raw text for step debugging."""
    memory = getattr(actor, "temporal_memory", None)
    if memory is None:
        return {}
    # Prefer this step's own analysis: reading Temporal Memory's latest_result
    # clears it whenever that analysis completed the subgoal.
    result = actor.last_caption or memory.latest_result
    diagnostics = memory.diagnostics()
    report = {
        "temporal_frames": len(memory.recent_frames()),
        "completion_evidence_frames": diagnostics.get(
            "completion_window_size"
        ),
        "completion_frame_ids": diagnostics.get(
            "completion_frame_ids", []
        ),
        "temporal_error": memory.last_analysis_error,
        "captioner_ran_this_step": actor.last_caption is not None,
    }
    if result is not None:
        report.update({
            "captioner_completed": result.completed,
            "captioner_error": result.error,
            "captioner_error_mode": result.error_mode,
            "captioner_error_confidence": result.error_confidence,
            "captioner_error_evidence": result.error_evidence,
            "captioner_latency_ms": result.latency_ms,
            "captioner_raw_response": result.raw_response,
        })
    return report


def _memory_state(actor):
    """Forward both memories' own diagnostics so each step's state is visible.

    ``pending_events`` is reported as a count because Temporal Memory never
    drains it in this worker, so the list itself only grows with the episode.
    """
    state = {}
    task_memory = getattr(actor, "task_memory", None)
    if task_memory is not None:
        state["task_memory"] = task_memory.diagnostics()
    memory = getattr(actor, "temporal_memory", None)
    if memory is not None:
        temporal = memory.diagnostics()
        temporal["pending_events"] = len(temporal["pending_events"])
        state["temporal_memory"] = temporal
    return state


def _subgoal_debug(actor, subgoal_id):
    """Return the exact planner text used for one subgoal ID."""
    for subgoal in getattr(actor, "subgoals", ()):
        if str(subgoal.subgoal_id) == str(subgoal_id):
            return {
                "subgoal_id": subgoal.subgoal_id,
                "description": subgoal.description,
                "completion_criteria": subgoal.completion_criteria,
            }
    return None


def _act_response(actor, decision):
    """Build the response shared by ``act`` and ``act_on_preview``."""
    response = {
        # The one nested object the runner should read: the mode decision with
        # its waypoint underneath.
        "decision": _decision_payload(actor, decision),
        # Kept flat alongside it so the existing runner call sites keep working
        # while they migrate onto "decision".
        "stop": decision.stop,
        "action_mode": decision.action_mode,
        # Flat as well, because it is how the runner tells a turn apart from
        # the other two decisions that carry no world point: a PREVIEW and a
        # stop. None on every step that steers to a waypoint.
        "turn_deg": decision.turn_deg,
        "raw_model_response": decision.raw_response,
        "timings": actor.last_timings,
    }
    response.update(_temporal_report(actor))
    response.update(_memory_state(actor))
    response.update(_agent_debug_state(actor, decision))
    if decision.point is not None:
        response.update({
            "pixel_uv": decision.point.pixel_uv,
            "depth_m": decision.point.depth_m,
            "camera_xyz": decision.point.camera_xyz,
            "world_xyz": decision.point.world_xyz,
        })
    return response


def _decision_payload(actor, decision):
    """Mirror the actor's own wire schema so one shape crosses every boundary.

    The mode is the discriminator and the waypoint lives under the block that
    mode names, exactly as the model emits it; the geometry the RGB-D layer
    resolved is added alongside the normalized coordinates it came from.
    """
    payload = {
        "action_mode": decision.action_mode,
        "confidence": getattr(actor, "last_waypoint_confidence", None),
        "evidence": getattr(actor, "last_waypoint_evidence", None),
    }
    if decision.action_mode == "PREVIEW":
        payload["preview"] = True
        return payload

    block = {
        "intent": getattr(actor, "last_waypoint_applied_intent", None),
    }
    if decision.turn_deg is not None:
        # An in-place turn has no coordinates at all: the controller repeats the
        # simulator's turn primitive instead of steering to a point.
        block["turn_deg"] = decision.turn_deg
        payload["execution"] = block
        return payload

    block["normalized_uv"] = getattr(actor, "last_requested_normalized", None)
    # Present only when this action was resolved from surrounding views, in
    # which case the coordinates address that view rather than the forward one.
    view_index = getattr(actor, "last_preview_view_index", None)
    if view_index is not None:
        block.update({
            "view_index": view_index,
            "view_yaw_deg": getattr(actor, "last_preview_yaw_deg", None),
        })
    if decision.point is not None:
        block.update({
            "pixel_uv": list(decision.point.pixel_uv),
            "depth_m": decision.point.depth_m,
            "camera_xyz": list(decision.point.camera_xyz),
            "world_xyz": list(decision.point.world_xyz),
        })
    if decision.action_mode == "EXPLORATION":
        payload["exploration"] = block
    else:
        block["stop"] = decision.stop
        payload["execution"] = block
    return payload


def _agent_debug_state(actor, decision):
    """Expose v3 decisions without asking the runner to infer internal state."""
    analyzed_id = (
        actor.last_caption.subgoal_id
        if getattr(actor, "last_caption", None) is not None
        else None
    )
    before_id = getattr(actor, "last_subgoal_before", None)
    after_id = getattr(actor, "last_subgoal_after", None)
    if decision.stop:
        if getattr(actor, "last_waypoint_stop_disposition", None):
            stop_reason = "WAYPOINT_FINAL_STOP"
        elif actor.task_memory.is_task_complete():
            stop_reason = "ALL_SUBGOALS_COMPLETE"
        else:
            stop_reason = "UNCLASSIFIED_STOP"
    else:
        stop_reason = "CONTINUE"
    landmark = getattr(actor, "last_landmark", None)
    return {
        "debug": {
            "analyzed_subgoal": _subgoal_debug(actor, analyzed_id),
            "subgoal_before": _subgoal_debug(actor, before_id),
            "subgoal_after": _subgoal_debug(actor, after_id),
            "subgoal_transition": before_id != after_id,
            "requested_pixel_uv": getattr(
                actor, "last_requested_pixel", None
            ),
            "requested_normalized_uv": getattr(
                actor, "last_requested_normalized", None
            ),
            "requested_turn_deg": getattr(
                actor, "last_requested_turn_deg", None
            ),
            "navigation_phase": getattr(
                actor, "_navigation_phase", None
            ),
            "corridor_heading_yaw_deg": getattr(
                actor, "_corridor_heading_yaw_deg", None
            ),
            "waypoint_model_intent": getattr(
                actor, "last_waypoint_model_intent", None
            ),
            "waypoint_applied_intent": getattr(
                actor, "last_waypoint_applied_intent", None
            ),
            "waypoint_model_action_mode": getattr(
                actor, "last_waypoint_model_action_mode", None
            ),
            "waypoint_applied_action_mode": getattr(
                actor, "last_waypoint_applied_action_mode", None
            ),
            "waypoint_guard_reason": getattr(
                actor, "last_waypoint_guard_reason", None
            ),
            "waypoint_evidence": getattr(
                actor, "last_waypoint_evidence", None
            ),
            "waypoint_confidence": getattr(
                actor, "last_waypoint_confidence", None
            ),
            "error_candidate": getattr(
                actor, "last_error_candidate", None
            ),
            "error_guard_reason": getattr(
                actor, "last_error_guard_reason", None
            ),
            "recovery_mode": getattr(
                actor, "last_recovery_mode", None
            ),
            "landmark": (
                landmark.model_dump()
                if landmark is not None
                else None
            ),
            # The tracker keeps its last state after a subgoal advances, so the
            # visualization needs this to tell a fresh reading from a stale one.
            "landmark_subgoal_id": getattr(
                actor, "_landmark_subgoal_id", None
            ),
            "landmark_pixel_uv": getattr(
                actor, "last_landmark_pixel", None
            ),
            "landmark_normalized_uv": getattr(
                actor, "last_landmark_normalized", None
            ),
            "landmark_raw_response": getattr(
                actor, "last_landmark_raw_response", None
            ),
            "landmark_error": getattr(
                actor, "last_landmark_error", None
            ),
            "behavior_history": list(
                actor.behavior_history()
                if hasattr(actor, "behavior_history")
                else ()
            ),
            "waypoint_raw_response": getattr(
                actor, "last_waypoint_raw_response", None
            ),
            "waypoint_stop_disposition": getattr(
                actor, "last_waypoint_stop_disposition", None
            ),
            "stop_reason": stop_reason,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    from agentflow.agents.vln_agent_4 import PreviewView, VLNAgent

    actor = VLNAgent(
        args.model_path,
        debug_performance=False,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "prepare":
                subgoals = actor.prepare_task(request["instruction"])
                response = {
                    "subgoals": [
                        {
                            "subgoal_id": subgoal.subgoal_id,
                            "description": subgoal.description,
                            "completion_criteria": subgoal.completion_criteria,
                        }
                        for subgoal in subgoals
                    ]
                }
            elif request.get("operation", "act") == "act":
                decision = actor.act(
                    _decode_rgb(request["rgb"]),
                    _decode_array(request["depth"]),
                    request["instruction"],
                    np.asarray(request["intrinsics"], dtype=np.float64),
                    np.asarray(request["camera_to_world"], dtype=np.float64),
                    normalized_depth=bool(request.get("normalized_depth", False)),
                    depth_min_m=request.get("depth_min_m"),
                    depth_max_m=request.get("depth_max_m"),
                )
                response = _act_response(actor, decision)
            elif request.get("operation") == "act_on_preview":
                # The second half of a previewed step: the runner rendered the
                # headings the actor asked for, and the actor commits to one.
                # It must not re-enter ``act``, which would advance motion,
                # landmark, and temporal state a second time for one step.
                decision = actor.act_on_preview(
                    [
                        PreviewView(
                            yaw_deg=float(view["yaw_deg"]),
                            rgb=_decode_rgb(view["rgb"]),
                            depth=_decode_array(view["depth"]),
                            intrinsics=np.asarray(
                                view["intrinsics"], dtype=np.float64
                            ),
                            camera_to_world=np.asarray(
                                view["camera_to_world"], dtype=np.float64
                            ),
                        )
                        for view in request["views"]
                    ],
                    request["instruction"],
                    normalized_depth=bool(
                        request.get("normalized_depth", False)
                    ),
                    depth_min_m=request.get("depth_min_m"),
                    depth_max_m=request.get("depth_max_m"),
                )
                response = _act_response(actor, decision)
            else:
                raise ValueError("Unsupported operation: {!r}".format(request["operation"]))
        except Exception as exc:
            response = {"error": "{}: {}".format(type(exc).__name__, exc)}
        protocol_stdout.write(json.dumps(response) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
