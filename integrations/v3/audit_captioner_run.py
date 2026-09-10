"""Read-only run audit; geometric metrics are diagnostics, not semantic verdicts."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    return round(values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo), 3)


def audit(trace_path, log_path):
    text = log_path.read_text(errors="replace")
    completed = {m[1]: dict(steps=int(m[2]), success=float(m[3]), spl=float(m[4]))
        for m in re.finditer(r"rank=\d+ \[\d+/\d+\] id=(\d+) steps=(\d+) success=([\d.]+) spl=([\d.]+)", text)}
    times = defaultdict(list)
    for m in re.finditer(r"^ep=(\d+) s=\d+ (\d+)ms", text, re.M):
        times[m[1]].append(int(m[2]))
    episodes = defaultdict(list)
    if trace_path.exists():
        with trace_path.open() as source:
            for line in source:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episodes[str(value["episode_id"])].append(value)
    result = []
    for eid, steps in episodes.items():
        actions = Counter(t["action"] for t in steps)
        still = longest = turns = 0
        route_ids = set()
        progress = []
        error_modes = Counter()
        failures = 0
        path = 0.
        for t in steps:
            d = t["decision"]
            distance = math.dist(t["position_before"][::2], t["position_after"][::2])
            path += distance
            still = still + 1 if distance < .01 else 0
            longest = max(longest, still)
            turns += t["action"] in (2, 3) and distance < .01
            ex = d.get("preview_execution") or {}
            if ex.get("world_xyz") is not None:
                route_ids.add(ex.get("frame_id"))
            failures += bool(d.get("captioner_failed_stage") or d.get("temporal_error"))
            error_modes[d.get("captioner_error_mode")] += 1
            if d.get("captioner_completed"):
                # This is an index for visual audit, NOT a false-completion rule.
                progress.append(dict(step=t["step"], path_so_far_m=round(path - distance, 3),
                    path_after_action_m=round(path, 3),
                    evidence=d.get("completion_evidence_frame_ids"), raw=d.get("captioner_raw_response")))
        row = dict(episode_id=eid, state="finished" if eid in completed else "partial",
            trace_steps=len(steps), **completed.get(eid, {}), actions=dict(actions),
            stationary_turn_steps=turns, longest_stationary_run=longest, path_xz_m=round(path, 3),
            initial_distance_m=steps[0]["distance_to_goal_before"],
            closest_distance_m=min(min(t["distance_to_goal_before"], t["distance_to_goal_after"]) for t in steps),
            final_distance_m=steps[-1]["distance_to_goal_after"], preview_commits=len(route_ids),
            route_actions=dict(Counter(t["decision"].get("captioner_route_action", "LEGACY") for t in steps)),
            inference_failures=failures, error_modes=dict(error_modes), completion_audit=progress,
            step_p50_ms=percentile(times[eid], .5), step_p95_ms=percentile(times[eid], .95),
            latency_samples=len(times[eid]),
            trace_matches_reported_steps=(len(steps) == completed[eid]["steps"] if eid in completed else None),
            step_max_ms=max(times[eid], default=None), over_10s=sum(t > 10000 for t in times[eid]))
        result.append(row)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.trace, args.log), ensure_ascii=False, indent=2))
