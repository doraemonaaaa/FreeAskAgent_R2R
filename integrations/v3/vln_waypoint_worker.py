"""JSON-lines worker that owns the Python 3.12 RGB-D waypoint actor."""

import argparse
import base64
import io
import json
import sys
from dataclasses import asdict

import numpy as np
from PIL import Image


def _decode_rgb(encoded):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


def _decode_array(encoded):
    with io.BytesIO(base64.b64decode(encoded)) as buffer:
        return np.load(buffer, allow_pickle=False)


def _encode_png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _temporal_report(actor):
    """Expose the Captioner's own judgement and raw text for step debugging."""
    memory = actor.temporal_memory
    # Prefer this step's own analysis: reading Temporal Memory's latest_result
    # clears it whenever that analysis completed the subgoal.
    result = actor.last_caption
    diagnostics = memory.diagnostics()
    captioner = diagnostics.get("captioner") or {}
    report = {
        "temporal_frames": len(memory.recent_frames()),
        "completion_frame_ids": diagnostics.get("frame_ids", []),
        "completion_eligible_frame_ids": diagnostics.get("completion_eligible_frame_ids", []),
        "temporal_error": memory.last_analysis_error,
        "captioner_ran_this_step": result is not None,
        "captioner_model_calls": captioner.get("model_calls", 0),
        "captioner_stage_timings_ms": captioner.get("last_stage_timings_ms", {}),
        "captioner_stage_budgets_ms": captioner.get("last_stage_budgets_ms", {}),
        "captioner_failed_stage": captioner.get("last_failed_stage"),
        "captioner_execution_history": captioner.get("execution_history", {}),
    }
    if result is not None:
        report.update({
            "captioner_completed": result.completed,
            "captioner_completion_evidence": result.completion_evidence,
            "completion_evidence_frame_ids": list(result.completion_evidence_frame_ids),
            "completion_evidence_frame_paths": list(result.completion_evidence_frame_paths),
            "captioner_error": result.error,
            "captioner_error_mode": result.error_mode,
            "captioner_error_evidence": result.error_evidence,
            "captioner_latency_ms": result.latency_ms,
            "captioner_raw_response": result.raw_response,
            "captioner_decision": result.decision,
            "captioner_error_evidence_frame_ids": list(result.error_evidence_frame_ids),
            "captioner_recovery_id": result.recovery_id,
            "captioner_preview_direction": result.preview_direction,
            "captioner_route_action": result.route_action,
            "captioner_route_reason": result.route_reason,
        })
    return report


def _memory_state(actor):
    """Forward the memories' own diagnostics so each step's state is visible."""
    temporal = actor.temporal_memory.diagnostics()
    # Events are reported as a count: the worker never drains them, so the
    # list itself only grows with the episode.
    temporal["events"] = len(temporal["events"])
    state = {"task_memory": actor.task_memory.diagnostics(), "temporal_memory": temporal}
    if actor.use_spatial_memory:
        state["spatial_memory"] = actor.spatial_memory.diagnostics()
    return state


def _subgoal_debug(actor, subgoal_id):
    """Return the exact planner text used for one subgoal ID."""
    for subgoal in actor.subgoals:
        if str(subgoal.subgoal_id) == str(subgoal_id):
            return {
                "subgoal_id": subgoal.subgoal_id,
                "description": subgoal.description,
                "completion_criteria": subgoal.completion_criteria,
            }
    return None


def _visuals(actor):
    """What the agent believes, for the video: its map and the marker frame."""
    out = {}
    if actor.use_spatial_memory:
        try:
            points = [c["world_xyz"] for c in actor.last_som_candidates if c.get("world_xyz")]
            out["map_png"] = _encode_png(actor.spatial_memory.visual_map(extra_points=points))
        except Exception as exc:  # a broken picture must not break the step
            out["map_error"] = f"{type(exc).__name__}: {exc}"
    if actor.last_som_image is not None:
        out["som_png"] = _encode_png(actor.last_som_image)
    return out


def _act_response(actor, decision, want_visuals=False):
    response = {
        "decision": _decision_payload(actor, decision),
        "stop": decision.stop,
        # How the runner tells a turn apart from a stop and a steer; None on
        # every step that steers to a waypoint.
        "turn_deg": decision.turn_deg,
        "raw_model_response": decision.raw_response,
        "timings": actor.last_timings,
    }
    response.update(_temporal_report(actor))
    response.update(_memory_state(actor))
    response["debug"] = _agent_debug_state(actor, decision)
    response["preview_request"] = actor.temporal_memory.preview_request()
    response["preview_execution"] = actor._preview_execution
    if want_visuals:
        response["visuals"] = _visuals(actor)
    if decision.point is not None:
        response.update({
            "pixel_uv": decision.point.pixel_uv,
            "depth_m": decision.point.depth_m,
            "camera_xyz": decision.point.camera_xyz,
            "world_xyz": decision.point.world_xyz,
        })
    return response


def _decision_payload(actor, decision):
    """The step's action with the geometry the RGB-D layer resolved."""
    block = {"intent": actor.last_waypoint_intent, "stop": decision.stop}
    if decision.turn_deg is not None:
        # An in-place turn has no coordinates: the controller repeats the
        # simulator's turn primitive instead of steering to a point.
        block["turn_deg"] = decision.turn_deg
    else:
        block["normalized_uv"] = actor.last_requested_normalized
        # Present only when this action was resolved from surrounding views, in
        # which case the coordinates address that view rather than the forward one.
        if actor.last_preview_view_index is not None:
            block.update({"view_index": actor.last_preview_view_index, "view_yaw_deg": actor.last_preview_yaw_deg})
        if decision.point is not None:
            block.update({
                "pixel_uv": list(decision.point.pixel_uv),
                "depth_m": decision.point.depth_m,
                "camera_xyz": list(decision.point.camera_xyz),
                "world_xyz": list(decision.point.world_xyz),
            })
    return {
        "confidence": actor.last_waypoint_confidence,
        "evidence": actor.last_waypoint_evidence,
        "execution": block,
    }


def _agent_debug_state(actor, decision):
    """Expose v3 decisions without asking the runner to infer internal state."""
    analyzed_id = actor.last_caption.subgoal_id if actor.last_caption is not None else None
    if decision.stop:
        stop_reason = "ALL_SUBGOALS_COMPLETE" if actor.task_memory.is_task_complete() else "UNCLASSIFIED_STOP"
    else:
        stop_reason = "CONTINUE"
    return {
        "analyzed_subgoal": _subgoal_debug(actor, analyzed_id),
        "subgoal_before": _subgoal_debug(actor, actor.last_subgoal_before),
        "subgoal_after": _subgoal_debug(actor, actor.last_subgoal_after),
        "subgoal_transition": actor.last_subgoal_before != actor.last_subgoal_after,
        "requested_pixel_uv": actor.last_requested_pixel,
        "requested_normalized_uv": actor.last_requested_normalized,
        "requested_turn_deg": actor.last_requested_turn_deg,
        "navigation_phase": actor._navigation_phase,
        "waypoint_intent": actor.last_waypoint_intent,
        "waypoint_guard_reason": actor.last_waypoint_guard_reason,
        "waypoint_evidence": actor.last_waypoint_evidence,
        "waypoint_confidence": actor.last_waypoint_confidence,
        "waypoint_raw_response": actor.last_waypoint_raw_response,
        "preview_view_index": actor.last_preview_view_index,
        "preview_yaw_deg": actor.last_preview_yaw_deg,
        "preview_selection": asdict(actor.last_preview_selection) if actor.last_preview_selection is not None else None,
        "preview_guard_reason": actor.last_preview_guard_reason,
        "error_candidate": actor.last_error_candidate,
        "error_guard_reason": actor.last_error_guard_reason,
        "recovery_mode": actor.last_recovery_mode,
        "behavior_history": list(actor.behavior_history()),
        "spatial_summary": actor.last_spatial_summary,
        "spatial_error": actor.last_spatial_error,
        "som_choice": actor.last_som_choice,
        "som_candidates": actor.last_som_candidates,
        "som_error": actor.last_som_error,
        "som_raw_response": actor.last_som_raw_response,
        "stop_reason": stop_reason,
    }


def _navigable_window(payload):
    if not payload:
        return None
    window = {
        "origin_xz": tuple(payload["origin_xz"]),
        "resolution_m": float(payload["resolution_m"]),
        "mask": _decode_array(payload["mask"]),
        "height_cell_m": float(payload.get("height_cell_m", 0.0)),
    }
    if "height_m" in payload:
        window["height_m"] = _decode_array(payload["height_m"])
    return window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Planner/set-of-mark model name (see create_vln_engine).")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--camera-height-m", type=float, default=None,
        help="Camera height above the agent's base; enables floor-level waypoint validation in the actor.",
    )
    args = parser.parse_args()

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    from agentflow.agents.vln_agent_4 import PreviewView, VLNAgent

    actor = VLNAgent(args.model_path, base_url=args.base_url, camera_height_m=args.camera_height_m)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("operation", "act")
            if operation == "prepare":
                response = {"subgoals": [
                    {"subgoal_id": s.subgoal_id, "description": s.description, "completion_criteria": s.completion_criteria}
                    for s in actor.prepare_task(request["instruction"])
                ]}
            elif operation == "act":
                decision = actor.act(
                    _decode_rgb(request["rgb"]),
                    _decode_array(request["depth"]),
                    request["instruction"],
                    np.asarray(request["intrinsics"], dtype=np.float64),
                    np.asarray(request["camera_to_world"], dtype=np.float64),
                    normalized_depth=bool(request.get("normalized_depth", False)),
                    depth_min_m=request.get("depth_min_m"),
                    depth_max_m=request.get("depth_max_m"),
                    navigable_window=_navigable_window(request.get("navigable")),
                    oracle_goal_xyz=request.get("oracle_goal_xyz"),
                    cwp_candidates=request.get("cwp_candidates"),
                    preview_views=[
                        PreviewView(
                            yaw_deg=float(view["yaw_deg"]),
                            rgb=_decode_rgb(view["rgb"]),
                            depth=_decode_array(view["depth"]),
                            intrinsics=np.asarray(view["intrinsics"], dtype=np.float64),
                            camera_to_world=np.asarray(view["camera_to_world"], dtype=np.float64),
                        ) for view in request.get("preview_views", [])
                    ],
                    preview_request_id=request.get("preview_request_id", ""),
                    previous_execution=request.get("previous_execution"),
                )
                response = _act_response(actor, decision, want_visuals=bool(request.get("want_visuals")))
            else:
                raise ValueError("Unsupported operation: {!r}".format(operation))
        except Exception as exc:
            response = {"error": "{}: {}".format(type(exc).__name__, exc)}
        protocol_stdout.write(json.dumps(response) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
