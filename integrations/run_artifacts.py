"""Run-isolated result and media artifact helpers for R2R evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import uuid


SCHEMA_VERSION = 1


def make_run_id(prefix="r2r"):
    """Return a filesystem-safe run ID with sub-second and process uniqueness."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return "{}-{}-{}-{}".format(
        prefix,
        timestamp,
        os.getpid(),
        uuid.uuid4().hex[:8],
    )


def safe_name(value):
    """Convert an episode/run identifier to a safe filename component."""
    normalized = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    ).strip("._")
    return normalized or "unknown"


def json_safe(value):
    """Recursively convert common Habitat/numpy values to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "value"):
        return json_safe(value.value)
    raise TypeError(
        "Value of type {} is not JSON serializable".format(
            type(value).__name__
        )
    )


def compact_habitat_metrics(metrics):
    """Keep scalar Habitat metrics and omit array-heavy visualization payloads."""
    compact = {}
    for name, value in (metrics or {}).items():
        try:
            converted = json_safe(value)
        except TypeError:
            continue
        if converted is None or isinstance(converted, (str, bool, int, float)):
            compact[str(name)] = converted
    return compact


def memory_timing(memory_diagnostics):
    """Extract the canonical timing payload without inventing missing samples."""
    if not isinstance(memory_diagnostics, dict):
        return {}
    timing = memory_diagnostics.get("timing")
    if isinstance(timing, dict):
        return json_safe(timing)

    # Compatibility for diagnostics that expose timing below an active-memory
    # entry (for example a task+temporal composite).
    collected = {}
    for name, payload in memory_diagnostics.items():
        if not isinstance(payload, dict):
            continue
        nested = payload.get("timing")
        if isinstance(nested, dict):
            collected[str(name)] = json_safe(nested)
    return collected


def _component_count(component):
    for name in (
        "inference_count",
        "count",
        "call_count",
        "attempt_count",
        "attempts",
    ):
        value = component.get(name)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def _component_total_ms(component):
    for name in (
        "total_inference_ms",
        "total_ms",
        "inference_total_ms",
    ):
        value = component.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _timing_components(payload, prefix=""):
    """Yield timing leaves that provide additive count and total-ms fields."""
    if not isinstance(payload, dict):
        return
    count = _component_count(payload)
    total_ms = _component_total_ms(payload)
    if count is not None and total_ms is not None:
        yield prefix or "memory", payload, count, total_ms
        return
    for name, value in payload.items():
        if isinstance(value, dict):
            path = "{}.{}".format(prefix, name) if prefix else str(name)
            yield from _timing_components(value, path)


def aggregate_episode_timing(episodes):
    """Aggregate timing components using sample counts, never episode means."""
    totals = {}
    for episode in episodes:
        for path, component, count, total_ms in _timing_components(
            episode.get("timing") or {}
        ):
            aggregate = totals.setdefault(
                path,
                {
                    "inference_count": 0,
                    "total_inference_ms": 0.0,
                },
            )
            aggregate["inference_count"] += count
            aggregate["total_inference_ms"] += total_ms
            for field in (
                "success_count",
                "successful_inference_count",
                "failure_count",
                "failed_inference_count",
                "latency_budget_met_count",
            ):
                value = component.get(field)
                if isinstance(value, (int, float)):
                    aggregate[field] = aggregate.get(field, 0) + int(value)

    for aggregate in totals.values():
        count = aggregate["inference_count"]
        aggregate["average_inference_ms"] = (
            aggregate["total_inference_ms"] / count
            if count
            else None
        )
    return totals


def aggregate_episodes(episodes):
    """Return scalar Habitat averages and weighted timing for episode rows."""
    scalar_names = sorted(
        {
            name
            for episode in episodes
            for name, value in episode["habitat_metrics"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    averages = {}
    for name in scalar_names:
        values = [
            float(episode["habitat_metrics"][name])
            for episode in episodes
            if isinstance(
                episode["habitat_metrics"].get(name),
                (int, float),
            )
            and not isinstance(
                episode["habitat_metrics"].get(name),
                bool,
            )
        ]
        if values:
            averages[name] = sum(values) / len(values)
    return {
        "episode_count": len(episodes),
        "total_steps": sum(episode["steps"] for episode in episodes),
        "average_habitat_metrics": averages,
        "timing": aggregate_episode_timing(episodes),
    }


class RunArtifactWriter:
    """Incrementally write one self-contained artifact directory per run/rank."""

    def __init__(
        self,
        artifact_root,
        *,
        run_config,
        run_id=None,
        rank=None,
    ):
        self.run_id = safe_name(run_id or make_run_id())
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.run_dir = self.artifact_root / self.run_id
        self.rank = None if rank is None else int(rank)
        self.output_dir = (
            self.run_dir / "rank_{}".format(self.rank)
            if self.rank is not None
            else self.run_dir
        )
        self.video_dir = self.output_dir / "videos"
        self.topdown_dir = self.output_dir / "topdown"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.topdown_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "results.json"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.run_config = json_safe(run_config)
        self.episodes = []
        self.write()

    def media_path(self, episode_id, kind):
        """Return the expected absolute media path for an episode."""
        filename = safe_name(episode_id)
        if kind == "video":
            return self.video_dir / (filename + ".mp4")
        if kind == "topdown":
            return self.topdown_dir / (filename + ".png")
        raise ValueError("Unknown media kind: {!r}".format(kind))

    def relative_path(self, path):
        if path is None:
            return None
        return str(Path(path).resolve().relative_to(self.run_dir))

    def add_episode(
        self,
        *,
        episode_id,
        instruction,
        actions,
        steps,
        habitat_metrics,
        video_path=None,
        topdown_path=None,
        memory_diagnostics=None,
        error=None,
    ):
        diagnostics = json_safe(memory_diagnostics)
        episode = {
            "episode_id": str(episode_id),
            "instruction": str(instruction),
            "actions": [str(action) for action in actions],
            "steps": int(steps),
            "habitat_metrics": compact_habitat_metrics(habitat_metrics),
            "artifacts": {
                "video": self.relative_path(video_path),
                "topdown": self.relative_path(topdown_path),
            },
            "memory": diagnostics,
            "timing": memory_timing(diagnostics),
            "error": None if error is None else str(error),
        }
        self.episodes.append(episode)
        self.write()
        return episode

    def _aggregate(self):
        return aggregate_episodes(self.episodes)

    def payload(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "rank": self.rank,
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config": self.run_config,
            "episodes": list(self.episodes),
            "aggregate": self._aggregate(),
        }

    def write(self):
        payload = self.payload()
        temporary = self.results_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        temporary.replace(self.results_path)
        return self.results_path
