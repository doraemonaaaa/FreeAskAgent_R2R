"""AwareVLN adapter: ORIG / SWAP / DROP-k splits in its data dir, and result import.

AwareVLN reads the instruction text at run time (no offline parse), so a
variant is just an R2R_VLNCE_v1-3 split with the new ``instruction_text``;
all 200 / 200 / 176 episodes are usable.

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

from .common import DATA_DIR, dump_json, load_json

AWARE = Path("/data/pengyh/workspace/Reproductions/AwareVLN/evaluation")
AWARE_DATA = AWARE / "data/datasets/R2R_VLNCE_v1-3_preprocessed"


def build(args):
    subgoals = load_json(args.subgoals)["episodes"]
    variants = load_json(args.variants)
    with gzip.open(str(AWARE_DATA / "val_unseen/val_unseen.json.gz"), "rt") as handle:
        base = json.load(handle)
    with gzip.open(str(AWARE_DATA / "val_unseen/val_unseen_gt.json.gz"), "rt") as handle:
        gt = json.load(handle)
    tables = dict(
        orig={eid: rec["instruction"] for eid, rec in subgoals.items()},
        swap={eid: v["instruction"] for eid, v in variants["swap"].items()},
        drop={eid: v["instruction"] for eid, v in variants["drop"].items()},
    )
    if args.flip:
        tables["flip"] = {eid: v["instruction"] for eid, v in load_json(args.flip)["flip"].items()}
    if args.goalonly:
        tables["goalonly"] = {eid: v["instruction"] for eid, v in load_json(args.goalonly)["goalonly"].items()}
    for variant, texts in tables.items():
        split = "{}_{}".format(args.name, variant)
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
    seen = set()
    with open(out / "rank_0_trace.jsonl", "w") as trace, open(out / "rank_0.log", "w") as log:
        for fn in sorted(glob.glob(str(results / "traj_*.jsonl"))):
            for line in open(fn):
                rec = json.loads(line)
                eid = rec["episode_id"]
                if eid in seen:
                    continue
                seen.add(eid)
                pos, dist = rec["positions"], rec["distances"]
                for step in range(1, len(pos)):
                    trace.write(json.dumps(dict(episode_id=eid, step=step - 1, position_before=pos[step - 1],
                                                position_after=pos[step], distance_to_goal_before=dist[step - 1],
                                                distance_to_goal_after=dist[step]), sort_keys=True) + "\n")
                m = rec["metric"]
                log.write("rank=0 id={} steps={} success={:.3f} spl={:.3f} dtg={:.2f} osr={:.0f} path_length={:.2f} ndtw={:.3f}\n".format(
                    eid, int(m["steps_taken"]), m["success"], m["spl"], m["distance_to_goal"], m["oracle_success"],
                    m["path_length"], m["ndtw"]))
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
    b.add_argument("--subgoals", default=str(DATA_DIR / "subgoals_val_unseen_200.json"))
    b.add_argument("--variants", default=str(DATA_DIR / "variants_val_unseen_200.json"))
    b.add_argument("--name", default="val_unseen_200")
    b.add_argument("--flip", default=str(DATA_DIR / "flip_val_unseen_200.json"))
    b.add_argument("--goalonly", default=str(DATA_DIR / "goalonly_val_unseen_200.json"))
    i = sub.add_parser("import")
    i.add_argument("--results", required=True, help="AwareVLN results dir holding traj_*.jsonl and <split>_N-i.json")
    i.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args) if args.cmd == "build" else import_run(args)


if __name__ == "__main__":
    main()
