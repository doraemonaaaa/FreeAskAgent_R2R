"""Shrink a runner output directory to what the metrics need, for archiving under results/.

    python -m benchmark.compact_trace SRC_DIR DST_DIR

A FreeAskAgent ``rank_*_trace.jsonl`` line carries the whole decision payload (camera
pose, prompt, VLM reply): ~11 KB per step, ~38 GB for the six 200-set runs.
``common.load_run`` reads five fields per line and nothing else, so archiving
the full payload buys nothing and costs three orders of magnitude.

This keeps ``episode_id``, ``step``, ``action``, ``position_before/after`` and
``distance_to_goal_before/after`` -- the exact input of every metric -- the optional
per-step cost fields (``common.COST_FIELDS``) when a runner writes them, plus the
``rank_*.log`` files unchanged (they hold the per-episode result lines and the
per-step summaries the failure analysis parses). Runs whose traces are already
minimal (CA-Nav, AwareVLN) pass through unchanged.
"""
import argparse
import json
from pathlib import Path

from .common import COST_FIELDS

KEEP = ("episode_id", "step", "action", "position_before", "position_after",
        "distance_to_goal_before", "distance_to_goal_after") + COST_FIELDS
KEY = '"distance_to_goal_after": '


def compact_line(line):
    """Parse the tail of a trace line, so a truncated decision payload never breaks it."""
    at = line.rfind(KEY)
    if at < 0:
        return None
    try:
        tail = json.loads("{" + line[at:])
    except ValueError:
        return None
    head = {}
    for key in ("episode_id", "step", "action", "position_before", "distance_to_goal_before") + COST_FIELDS:
        marker = '"{}": '.format(key)
        # cost fields are top-level and written after any nested payload: take the last one
        start = line.rfind(marker) if key in COST_FIELDS else line.find(marker)
        if start < 0:
            continue
        try:
            head[key] = json.JSONDecoder().raw_decode(line, start + len(marker))[0]
        except ValueError:
            pass
    head.update(tail)
    return {k: head[k] for k in KEEP if k in head}


def compact_dir(src, dst):
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    before = after = 0
    for trace in sorted(src.glob("rank_*_trace.jsonl")):
        before += trace.stat().st_size
        target = dst / trace.name
        with open(trace, errors="replace") as handle, open(target, "w") as out:
            for line in handle:
                row = compact_line(line)
                if row is not None:
                    # sorted: load_run parses the tail from "distance_to_goal_after",
                    # so episode_id / position_* must sort after it, as the runner writes them
                    out.write(json.dumps(row, sort_keys=True) + "\n")
        after += target.stat().st_size
    for log in sorted(src.glob("rank_*.log")):
        target = dst / log.name
        if not target.exists():
            target.write_bytes(log.read_bytes())
    return before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    before, after = compact_dir(args.src, args.dst)
    mb = 1024.0 * 1024.0
    print("{} -> {}  traces {:.0f} MB -> {:.1f} MB ({:.0f}x)".format(
        args.src, args.dst, before / mb, after / mb, before / max(after, 1)))


if __name__ == "__main__":
    main()
