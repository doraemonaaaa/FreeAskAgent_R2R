"""Own and call the v1 agentflow.agents.vln_agent.AsyncThinkActVLN."""

import argparse
import base64
import io
import inspect
import json
import re
import sys

from PIL import Image


ACTION_ALIASES = {
    "TURN_RIGHT_15_DEGREES": "TURN_RIGHT",
    "TURN_LEFT_15_DEGREES": "TURN_LEFT",
    "MOVE_FORWARD": "FORWARD",
    "MOVE_FORWARD_0.1_METERS": "FORWARD",
    "MOVE_FORWARD_0.25_METERS": "FORWARD",
}


def _call_with_supported_kwargs(callable_object, *args, **kwargs):
    """Pass new protocol fields only when a legacy callable can accept them."""
    parameters = inspect.signature(callable_object).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported = {
        key: value
        for key, value in kwargs.items()
        if value is not None and (accepts_kwargs or key in parameters)
    }
    return callable_object(*args, **supported)


def act_with_alias_recovery(
    agent,
    image,
):
    """Recover extended action labels rejected by the legacy Actor parser."""
    try:
        return agent.act(image)
    except ValueError as exc:
        match = re.search(r"returned invalid action ['\"]([^'\"]+)['\"]", str(exc))
        if match:
            action = match.group(1).strip().upper()
            if action in ACTION_ALIASES:
                return ACTION_ALIASES[action]
        raise


def reset_agent(agent, goal, episode_id=None):
    """Reset the public agent API, with a fallback for the pre-memory agent."""
    reset = getattr(agent, "reset", None)
    if reset is not None:
        parameters = inspect.signature(reset).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "goal" in parameters:
            kwargs = {"goal": goal}
            if episode_id is not None:
                kwargs["episode_id"] = episode_id
            elif "episode_id" in parameters:
                kwargs["episode_id"] = "legacy-episode"
            return _call_with_supported_kwargs(reset, **kwargs)
        return reset(goal)

    agent.goal = goal
    thinker = agent.thinker
    thinker_reset = getattr(thinker, "reset", None)
    if callable(thinker_reset):
        thinker_reset(goal)
    else:
        # Do not wait for ModelA's asynchronous planning request. It may still
        # be decoding when an episode ends.
        with thinker._condition:
            thinker._pending = None
        with thinker._lock:
            thinker.goal = goal
            thinker._subtask_tracker = None
            thinker._directive = goal
            actions = getattr(thinker, "_actions", None)
            if actions is not None:
                actions.clear()

    memory = getattr(agent, "temporal_memory", None)
    if memory is None:
        memory = getattr(agent, "task_memory", None)
    if memory is not None and hasattr(memory, "reset"):
        memory.reset(
            episode_id=str(episode_id or "legacy-episode"),
            goal=goal,
        )


def finish_agent_episode(
    agent,
    image,
):
    """Finalize the last pending transition when the environment terminates."""
    finish_episode = getattr(agent, "finish_episode", None)
    if finish_episode is None:
        return None
    return finish_episode(image)


def memory_diagnostics(agent, include_raw_response=False):
    """Return JSON-safe diagnostics from the configured agent memory."""
    memory = getattr(agent, "memory", None)
    if memory is None:
        memory = getattr(agent, "temporal_memory", None)
    if memory is None:
        memory = getattr(agent, "task_memory", None)
    if memory is None:
        return None

    diagnostics = getattr(memory, "diagnostics", None)
    if callable(diagnostics):
        payload = _call_with_supported_kwargs(
            diagnostics,
            include_raw_response=include_raw_response,
        )
    else:
        latest = getattr(memory, "latest_record", None)
        latest_payload = None
        if latest is not None:
            if hasattr(latest, "to_memory_dict"):
                latest_payload = latest.to_memory_dict()
            elif hasattr(latest, "model_dump"):
                latest_payload = latest.model_dump(mode="json")
            else:
                latest_payload = str(latest)
            if (
                include_raw_response
                and isinstance(latest_payload, dict)
                and hasattr(latest, "raw_response")
            ):
                latest_payload["raw_response"] = latest.raw_response
        payload = {
            "episode_id": getattr(memory, "episode_id", None),
            "last_analysis_error": getattr(
                memory, "last_analysis_error", None
            ),
            "latest_analysis": latest_payload,
        }

    # The wire protocol is intentionally strict JSON.  Raise here, close to the
    # producer, rather than leaving the parent process with a truncated reply.
    json.dumps(payload, ensure_ascii=False)
    return payload


def temporal_diagnostics(agent, include_raw_response=False):
    """Compatibility alias for callers of the earlier temporal-only helper."""
    return memory_diagnostics(
        agent,
        include_raw_response=include_raw_response,
    )


def _analysis_step_ids(diagnostics):
    modules = (diagnostics or {}).get("modules") or {}
    temporal = modules.get("temporal_memory")
    if isinstance(temporal, dict):
        diagnostics = temporal
    latest = (diagnostics or {}).get("latest_analysis") or {}
    captions = latest.get("step_captions") or []
    step_ids = [
        item.get("step_id")
        for item in captions
        if isinstance(item, dict) and item.get("step_id") is not None
    ]
    if step_ids:
        return [int(value) for value in step_ids]
    timeline = latest.get("action_timeline") or []
    step_ids = [
        item.get("step_id")
        for item in timeline
        if isinstance(item, dict) and item.get("step_id") is not None
    ]
    return [int(value) for value in step_ids]


def log_temporal_analysis_once(
    diagnostics,
    last_event_key=None,
    stream=None,
):
    """Log each successful or failed analyzed window once and return its key."""
    if not diagnostics:
        return last_event_key
    stream = stream or sys.stderr
    root_diagnostics = diagnostics
    modules = diagnostics.get("modules") or {}
    temporal = modules.get("temporal_memory")
    if isinstance(temporal, dict):
        diagnostics = temporal
    episode_id = (
        diagnostics.get("episode_id")
        or root_diagnostics.get("episode_id")
    )
    latest = diagnostics.get("latest_analysis")
    error = diagnostics.get("last_analysis_error")
    step_ids = _analysis_step_ids(diagnostics)
    attempted_step_id = diagnostics.get("last_analyzed_step_id")
    latest_step_id = step_ids[-1] if step_ids else None

    # A failed newer window keeps the previous valid caption.  Give that
    # failure its own event identity instead of silently treating it as the
    # already-logged successful window.
    if error and attempted_step_id != latest_step_id:
        failure_step_id = attempted_step_id
        if failure_step_id is None:
            completed = diagnostics.get("completed_step_ids") or []
            failure_step_id = completed[-1] if completed else None
        event_key = ("error", episode_id, failure_step_id, str(error))
        payload = {
            "event": "temporal_analysis",
            "status": "error",
            "episode_id": episode_id,
            "attempted_step_id": failure_step_id,
            "completed_step_ids": diagnostics.get("completed_step_ids"),
            "last_analysis_error": error,
            "raw_response": diagnostics.get("last_failed_raw_response"),
        }
    elif latest is not None:
        window = latest.get("window") or {}
        event_key = (
            "success",
            episode_id,
            tuple(step_ids),
            window.get("end_seconds"),
        )
        payload = {
            "event": "temporal_analysis",
            "status": "ok",
            "episode_id": episode_id,
            "step_ids": step_ids,
            "model_latency_ms": latest.get("model_latency_ms"),
            "latency_budget_ms": latest.get("latency_budget_ms"),
            "latency_budget_met": latest.get("latency_budget_met"),
            "peak_gpu_memory_bytes": latest.get("peak_gpu_memory_bytes"),
            "last_analysis_error": error,
            "raw_response": latest.get("raw_response"),
        }
    else:
        return last_event_key

    if event_key == last_event_key:
        return last_event_key
    print(
        "TEMPORAL_ANALYSIS "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=stream,
        flush=True,
    )
    return event_key


def decode_image(request):
    return Image.open(
        io.BytesIO(base64.b64decode(request["image"]))
    ).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="Navigate safely to the requested destination.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--planner-model-path", required=True)
    parser.add_argument("--temporal-model-path")
    parser.add_argument(
        "--memory-mode",
        choices=("temporal", "task", "task+temporal"),
        default="temporal",
    )
    args = parser.parse_args()

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr
    from agentflow.agents.vln_agent import AsyncThinkActVLN

    agent = AsyncThinkActVLN(
        goal=args.goal,
        policy_model_path=args.model_path,
        planner_model_path=args.planner_model_path,
        temporal_model_path=args.temporal_model_path,
        memory_mode=args.memory_mode,
        debug_performance=False,
    )
    agent.thinker.show_output = False
    last_logged_temporal_event = None
    episode_counter = 0

    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if "reset" in request:
                    goal = request["reset"]
                    episode_counter += 1
                    reset_agent(
                        agent,
                        goal,
                        "worker-episode-{}".format(episode_counter),
                    )
                    last_logged_temporal_event = None
                    response = {
                        "ok": True,
                        "memory": memory_diagnostics(agent),
                    }
                elif request.get("finish_episode"):
                    finish_agent_episode(
                        agent,
                        decode_image(request),
                    )
                    diagnostics = memory_diagnostics(
                        agent,
                        include_raw_response=True,
                    )
                    last_logged_temporal_event = (
                        log_temporal_analysis_once(
                            diagnostics,
                            last_logged_temporal_event,
                        )
                    )
                    response = {
                        "ok": True,
                        "memory": diagnostics,
                    }
                else:
                    action = act_with_alias_recovery(
                        agent,
                        decode_image(request),
                    )
                    diagnostics = memory_diagnostics(
                        agent,
                        include_raw_response=True,
                    )
                    last_logged_temporal_event = (
                        log_temporal_analysis_once(
                            diagnostics,
                            last_logged_temporal_event,
                        )
                    )
                    response = {
                        "action": action,
                        "memory": diagnostics,
                    }
            except Exception as exc:
                response = {"error": "{}: {}".format(type(exc).__name__, exc)}
            protocol_stdout.write(json.dumps(response) + "\n")
            protocol_stdout.flush()
    finally:
        agent.close(timeout=2)


if __name__ == "__main__":
    main()
