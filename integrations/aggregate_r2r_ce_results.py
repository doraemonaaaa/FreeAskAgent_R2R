"""Aggregate per-rank R2R-CE results produced by either agent version."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from integrations.run_artifacts import aggregate_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    results = []
    for rank in range(args.world_size):
        path = args.output_dir / "rank_{}.json".format(rank)
        if not path.is_file():
            raise SystemExit("Missing result from rank {}: {}".format(rank, path))
        with path.open() as handle:
            results.append(json.load(handle))

    count = sum(result["count"] for result in results)
    if count == 0:
        raise SystemExit("No episodes were evaluated.")
    totals = {
        name: sum(result["totals"][name] for result in results)
        for name in ("success", "spl", "distance_to_goal")
    }
    print("average_metrics={}".format({name: value / count for name, value in totals.items()}))
    print("episodes={}".format(count))

    artifact_paths = [
        Path(result["artifact_results"])
        for result in results
        if result.get("artifact_results")
    ]
    if len(artifact_paths) == len(results) and all(
        path.is_file() for path in artifact_paths
    ):
        rank_payloads = []
        for path in artifact_paths:
            with path.open(encoding="utf-8") as handle:
                rank_payloads.append(json.load(handle))
        run_ids = {payload["run_id"] for payload in rank_payloads}
        if len(run_ids) != 1:
            raise SystemExit(
                "Rank artifacts have different run IDs: {}".format(
                    sorted(run_ids)
                )
            )
        run_directories = {path.parent.parent for path in artifact_paths}
        if len(run_directories) != 1:
            raise SystemExit(
                "Rank artifacts do not share one run directory: {}".format(
                    sorted(str(path) for path in run_directories)
                )
            )
        episodes = [
            episode
            for payload in rank_payloads
            for episode in payload["episodes"]
        ]
        combined = {
            "schema_version": 1,
            "run_id": next(iter(run_ids)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ranks": sorted(
                payload.get("rank") for payload in rank_payloads
            ),
            "configs_by_rank": {
                str(payload.get("rank")): payload.get("config", {})
                for payload in rank_payloads
            },
            "episodes": episodes,
            "aggregate": aggregate_episodes(episodes),
        }
        combined_path = next(iter(run_directories)) / "results.json"
        temporary = combined_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                combined,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        temporary.replace(combined_path)
        print("artifact_results={}".format(combined_path))


if __name__ == "__main__":
    main()
