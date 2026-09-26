"""Freeze the planner's subgoal plans for an evaluation set.

Runs VLNAgent.prepare_task (the exact production planning path: prompt,
strict parsing, retries) once per episode and writes the plans in the
VLN_FROZEN_PLAN_FILE format, so every arm of an A/B comparison judges the same
stages. Only the planner endpoint is contacted.

    VLN_PLANNER_MODEL=qwen3-vl-8b VLN_PLANNER_BASE_URL=http://127.0.0.1:8301/v1 \
    .venv/bin/python freeze_plans.py --episode-set eval_sets/val_unseen_200.txt \
        --out plans.json [--shard 0/3]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "FreeAskAgent"))

from integrations.v3.habitat_runner.episodes import parse_episode_ids  # noqa: E402


class _NoTemporalMemory:
    def reset(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-set", required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", default="0/1", help="r/n: plan episodes[r::n]")
    parser.add_argument("--compact", action="store_true", help="VLN_COMPACT_SUBGOALS planner prompt")
    args = parser.parse_args()
    if not os.environ.get("VLN_PLANNER_MODEL"):
        raise SystemExit("set VLN_PLANNER_MODEL / VLN_PLANNER_BASE_URL to the planner endpoint")
    os.environ["VLN_COMPACT_SUBGOALS"] = "1" if args.compact else "0"
    os.environ.pop("VLN_FROZEN_PLAN_FILE", None)

    from agentflow.agents.vln.agent import VLNAgent
    from agentflow.agents.vln.memory.task_memory import TaskMemory

    dataset = ROOT.parent / "habitat" / "data" / "datasets" / "vln" / "mp3d" / "r2r" / "v1" / args.split / f"{args.split}.json.gz"
    episodes = {str(e["episode_id"]): e for e in json.load(gzip.open(dataset))["episodes"]}
    ids = parse_episode_ids("@" + args.episode_set)
    r, n = (int(v) for v in args.shard.split("/"))
    ids = ids[r::n]
    # The actor engine is never called: planning goes through the VLN_PLANNER role.
    agent = VLNAgent(engine=object(), task_memory=TaskMemory("placeholder"), temporal_memory=_NoTemporalMemory())
    plans, failures = [], []
    for index, episode_id in enumerate(ids, 1):
        instruction = episodes[episode_id]["instruction"]["instruction_text"]
        try:
            subgoals = agent.prepare_task(instruction)
        except Exception as exc:  # recorded, not fatal: the cohort must be complete before use
            failures.append({"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(ids)}] {episode_id} FAILED {exc}", flush=True)
            continue
        plans.append({
            "episode_id": episode_id, "instruction": instruction, "compact": args.compact,
            "subgoals": [{"subgoal_id": s.subgoal_id, "description": s.description,
                          "completion_criteria": s.completion_criteria} for s in subgoals],
        })
        print(f"[{index}/{len(ids)}] {episode_id} {len(subgoals)} stages", flush=True)
    payload = {
        "source": f"{Path(args.episode_set).name} shard {args.shard}; planner {os.environ['VLN_PLANNER_MODEL']}",
        "official_split": args.split,
        "cohort_sha256": hashlib.sha256(Path(args.episode_set).read_bytes()).hexdigest(),
        "plans": plans, "failures": failures,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(plans)} plans, {len(failures)} failures -> {args.out}")


if __name__ == "__main__":
    main()
