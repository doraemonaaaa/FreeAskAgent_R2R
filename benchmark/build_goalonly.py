"""GOAL-ONLY variant: keep only the last sub-instruction (the one with the stop condition).

    python3 -m benchmark.build_goalonly

Episodes with K = 1 are excluded (nothing to remove). Writes
  benchmark/data/goalonly_<set>.json          per-episode text + removed prefix
  <habitat r2r>/<set>_goalonly/...            split for the v19 runner (+ gt, + ids file)
CA-Nav / AwareVLN inputs: ``benchmark.canav build`` / ``benchmark.awarevln build`` (they read this json).
"""
import argparse
import re

from .build_variants import write_split
from .common import DATA_DIR, dump_json, load_episodes, load_gt, load_json


BARE_STOP = re.compile(r"^(?:and |then )?(?:stop|wait|stand|end|halt|walk forward)(?: there| here| right there| right here| immediatly| immediately)?\.?$", re.IGNORECASE)


def goal_only_text(instruction, last_span):
    text = instruction[last_span[0]:last_span[1]].strip()
    text = re.sub(r"^(?:and|then|,)\s+", "", text, flags=re.IGNORECASE)
    return text[0].upper() + text[1:] if text else text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subgoals", default=str(DATA_DIR / "subgoals_val_unseen_200.json"))
    parser.add_argument("--name", default="val_unseen_200")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--no-splits", action="store_true")
    args = parser.parse_args()
    subgoals = load_json(args.subgoals)["episodes"]
    out = {}
    for eid, rec in subgoals.items():
        if rec["K"] < 2 or not all(s["span"] for s in rec["subgoals"]):
            continue
        last = rec["subgoals"][-1]
        span = list(last["span"])
        kept = 1
        # a bare "Stop." carries no goal: keep the preceding chunk as well
        if BARE_STOP.match(last["text"].strip()) and rec["K"] >= 3:
            span[0] = rec["subgoals"][-2]["span"][0]
            kept = 2
        elif BARE_STOP.match(last["text"].strip()):
            continue
        out[eid] = dict(K=rec["K"], kept_chunks=kept, instruction=goal_only_text(rec["instruction"], span),
                        goal_text=last["text"], removed_text=rec["instruction"][: span[0]].strip())
    meta = dict(name=args.name, split=args.split, goalonly_split="{}_goalonly".format(args.name), goalonly=out)
    path = DATA_DIR / "goalonly_{}.json".format(args.name)
    dump_json(meta, path)
    print("goal-only episodes={} (K=1 excluded {})".format(len(out), len(subgoals) - len(out)))
    print("wrote", path)
    if not args.no_splits:
        episodes, raw = load_episodes(args.split)
        gt = load_gt(args.split)
        directory = write_split(meta["goalonly_split"], raw, episodes, gt, {eid: v["instruction"] for eid, v in out.items()})
        ids = DATA_DIR / "{}_goalonly_ids.txt".format(args.name)
        ids.write_text("# episode ids present in split {}_goalonly\n".format(args.name) + "".join(eid + "\n" for eid in out))
        print("wrote split", directory, "ids", ids)


if __name__ == "__main__":
    main()
