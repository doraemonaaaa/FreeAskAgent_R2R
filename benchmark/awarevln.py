"""AwareVLN adapter: ORIG / FLIP / GOAL-ONLY splits in its data dir, and result import.

AwareVLN reads the instruction text at run time (no offline parse), so a
variant is just an R2R_VLNCE_v1-3 split with the new ``instruction_text``;
each variant uses its own eligible episode subset. FLIP has its own episode set
(build_flip, full val_unseen) and writes <set>_flip_orig and <set>_flip over it.

    python -m benchmark.awarevln build
    python -m benchmark.awarevln import --results <RESULTS_DIR>/awarevln/VLN-CE-v1/<split> --out benchmark/results/awarevln/<variant>

``import`` reads the evaluator's ``traj_<split>_*.jsonl`` (written by the
Position measure + trainer patch) and ``<split>_N-i.json`` stats into the
runner trace layout that ``benchmark.metrics`` consumes.
"""
import argparse
import glob
import gzip
import json
from pathlib import Path

from .common import DEFAULT_SET, data_path, dump_json, load_json, write_runner_run

AWARE = Path("/data/pengyh/workspace/Reproductions/AwareVLN/evaluation")
AWARE_DATA = AWARE / "data/datasets/R2R_VLNCE_v1-3_preprocessed"


def build(args):
    subgoals = load_json(args.subgoals)["episodes"]
    with gzip.open(str(AWARE_DATA / "val_unseen/val_unseen.json.gz"), "rt") as handle:
        base = json.load(handle)
    with gzip.open(str(AWARE_DATA / "val_unseen/val_unseen_gt.json.gz"), "rt") as handle:
        gt = json.load(handle)
    tables = {}
    if "orig" in args.variants:
        tables["{}_orig".format(args.name)] = {eid: rec["instruction"] for eid, rec in subgoals.items()}
    if "flip" in args.variants and args.flip:
        meta = load_json(args.flip)
        flips = {eid: meta["flip"][eid] for eid in meta["balanced"]}
        tables["{}_flip_orig".format(meta["name"])] = {eid: v["original_instruction"] for eid, v in flips.items()}
        tables["{}_flip".format(meta["name"])] = {eid: v["instruction"] for eid, v in flips.items()}
    if "goalonly" in args.variants and args.goalonly:
        tables["{}_goalonly".format(args.name)] = {eid: v["instruction"] for eid, v in load_json(args.goalonly)["goalonly"].items()}
    for split, texts in tables.items():
        directory = AWARE_DATA / split
        directory.mkdir(parents=True, exist_ok=True)
        data = dict(base)
        data["episodes"] = []
        for episode in base["episodes"]:
            eid = str(episode["episode_id"])
            if eid in texts:
                episode = json.loads(json.dumps(episode))
                episode["instruction"]["instruction_text"] = texts[eid]
                data["episodes"].append(episode)
        with gzip.open(str(directory / "{}.json.gz".format(split)), "wt") as handle:
            json.dump(data, handle)
        with gzip.open(str(directory / "{}_gt.json.gz".format(split)), "wt") as handle:
            json.dump({eid: gt[eid] for eid in texts}, handle)
        print("{}: {} episodes -> {}".format(split, len(data["episodes"]), directory))


def import_run(args):
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for fn in glob.glob(str(results / "*_[0-9]*-[0-9]*.json")):
        stats.update(load_json(fn))
    seen = write_runner_run(glob.glob(str(results / "traj_*.jsonl")), out)
    keys = ["success", "oracle_success", "spl", "distance_to_goal", "path_length", "steps_taken", "ndtw"]
    rows = [stats[e] for e in seen if e in stats]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys} if rows else {}
    if rows:
        summary["sdtw"] = sum(r["success"] * r["ndtw"] for r in rows) / len(rows)
    dump_json(dict(results=str(results), n=len(rows), summary=summary), out / "awarevln_summary.json")
    print("imported {} episodes -> {}".format(len(seen), out))
    if rows:
        print("AwareVLN metrics  n={} SR={success:.3f} OSR={oracle_success:.3f} SPL={spl:.3f} NE={distance_to_goal:.2f} PL={path_length:.2f}m Steps={steps_taken:.1f} nDTW={ndtw:.3f} SDTW={sdtw:.3f}".format(len(rows), **summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--subgoals", default=str(data_path("subgoals")))
    b.add_argument("--name", default=DEFAULT_SET)
    b.add_argument("--variants", default="orig,flip,goalonly", type=lambda v: v.split(","),
                   help="comma list of orig, flip, goalonly")
    b.add_argument("--flip", default=str(data_path("flip", "val_unseen")))
    b.add_argument("--goalonly", default=str(data_path("goalonly")))
    i = sub.add_parser("import")
    i.add_argument("--results", required=True, help="AwareVLN results dir holding traj_*.jsonl and <split>_N-i.json")
    i.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args) if args.cmd == "build" else import_run(args)


if __name__ == "__main__":
    main()
