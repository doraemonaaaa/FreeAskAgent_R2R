"""SWAP and DROP-k instruction variants of an evaluation set, as extra habitat splits.

    python -m benchmark.build_variants --subgoals benchmark/data/subgoals_val_unseen_200.json \
        --name val_unseen_200 --seed 20260917

Writes
  benchmark/data/variants_<name>.json           per-episode variant metadata
  <r2r data>/<name>_swap/<name>_swap.json.gz    same episodes, instruction of a
                                                same-scene donor trajectory
  <r2r data>/<name>_drop/<name>_drop.json.gz    same episodes, sub-instruction k removed
  <name>_{swap,drop}/<name>_{swap,drop}_gt.json.gz  copy of the original dense gt
  benchmark/data/<name>_{swap,drop}_ids.txt      EPISODE_IDS file for each variant run
The runner takes them with SPLIT=<name>_swap (EPISODE_IDS may stay the 200-set
file: the variant split holds exactly those ids). Episode ids are unchanged,
so traces of the three runs line up by id.

SWAP donors: same scene, different trajectory, preferring a donor whose start
viewpoint is within 0.5 m of the episode's own start (otherwise the nearest
start). The donor's own subgoals are stored so PathAttrib / SGCR' can score the
trajectory against the donor's reference path.
DROP-k: k uniform in 1..K-1 (the last sub-instruction carries the stop
condition and is never removed); episodes with K = 1 are excluded.
"""
import argparse
import gzip
import json
import math
import random
import re

from .build_subgoals import build_episode, load_fgr2r
from .common import DATA_DIR, R2R_DIR, dump_json, load_episodes, load_gt, load_json


def remove_span(text, span):
    cut = text[: span[0]] + text[span[1]:]
    cut = re.sub(r"[ \t]+", " ", cut).strip()
    # a chunk that followed the removed one often starts with a connective
    cut = re.sub(r"([.!?]\s+)(?:and|then|,)\s+", r"\1", cut, flags=re.IGNORECASE)
    cut = re.sub(r"^(?:and|then|,)\s+", "", cut, flags=re.IGNORECASE)
    cut = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), cut)
    return cut[0].upper() + cut[1:] if cut else cut


def pick_donor(episode, episodes, rng, near_m=0.5):
    same_scene = [e for e in episodes.values()
                  if e["scene_id"] == episode["scene_id"] and e["trajectory_id"] != episode["trajectory_id"]]
    if not same_scene:
        return None, None
    distances = {str(e["episode_id"]): math.dist(e["start_position"], episode["start_position"]) for e in same_scene}
    near = [e for e in same_scene if distances[str(e["episode_id"])] <= near_m]
    pool = near or [min(same_scene, key=lambda e: distances[str(e["episode_id"])])]
    donor = rng.choice(sorted(pool, key=lambda e: int(e["episode_id"])))
    return donor, distances[str(donor["episode_id"])]


def write_split(name, raw, episodes_by_id, gt, instruction_of):
    """Write ``<name>/<name>.json.gz`` with the selected episodes and new instruction texts."""
    directory = R2R_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(raw)
    data["episodes"] = []
    for episode_id, text in instruction_of.items():
        episode = json.loads(json.dumps(episodes_by_id[episode_id]))
        episode["instruction"]["instruction_text"] = text
        data["episodes"].append(episode)
    with gzip.open(str(directory / "{}.json.gz".format(name)), "wt") as handle:
        json.dump(data, handle)
    with gzip.open(str(directory / "{}_gt.json.gz".format(name)), "wt") as handle:
        json.dump({eid: dict(locations=gt[eid]) for eid in instruction_of}, handle)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subgoals", default=str(DATA_DIR / "subgoals_val_unseen_200.json"))
    parser.add_argument("--name", default="val_unseen_200")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--no-splits", action="store_true", help="only write the metadata json")
    args = parser.parse_args()

    subgoals = load_json(args.subgoals)["episodes"]
    episodes, raw = load_episodes(args.split)
    gt = load_gt(args.split)
    fgr2r = load_fgr2r(args.split)
    rng = random.Random(args.seed)

    swap, drop = {}, {}
    for episode_id, item in subgoals.items():
        episode = episodes[episode_id]
        donor, start_distance = pick_donor(episode, episodes, rng)
        if donor is not None:
            donor_id = str(donor["episode_id"])
            donor_subgoals = build_episode(donor, fgr2r[int(donor["trajectory_id"])], gt[donor_id])
            swap[episode_id] = dict(
                donor_episode_id=donor_id, donor_start_distance_m=start_distance,
                instruction=donor["instruction"]["instruction_text"],
                donor_K=donor_subgoals["K"], donor_subgoals=donor_subgoals["subgoals"],
                donor_path_length_m=donor_subgoals["path_length_m"],
            )
        if item["K"] >= 2 and all(s["span"] for s in item["subgoals"]):
            k = rng.randint(1, item["K"] - 1)
            span = item["subgoals"][k - 1]["span"]
            drop[episode_id] = dict(k=k, removed_text=item["subgoals"][k - 1]["text"],
                                    instruction=remove_span(item["instruction"], span))

    meta = dict(name=args.name, split=args.split, seed=args.seed, subgoals_file=args.subgoals,
                swap_split="{}_swap".format(args.name), drop_split="{}_drop".format(args.name),
                swap=swap, drop=drop)
    out = DATA_DIR / "variants_{}.json".format(args.name)
    dump_json(meta, out)
    near = sum(v["donor_start_distance_m"] <= 0.5 for v in swap.values())
    print("swap={} (same-start donors {}) drop={} (K=1 excluded {})".format(
        len(swap), near, len(drop), len(subgoals) - len(drop)))
    print("wrote", out)
    if not args.no_splits:
        for suffix, table in (("swap", swap), ("drop", drop)):
            directory = write_split("{}_{}".format(args.name, suffix), raw, episodes, gt,
                                    {eid: v["instruction"] for eid, v in table.items()})
            ids_file = DATA_DIR / "{}_{}_ids.txt".format(args.name, suffix)
            ids_file.write_text("# episode ids present in split {}_{}\n".format(args.name, suffix)
                                + "".join(eid + "\n" for eid in table))
            print("wrote split", directory, "ids", ids_file)


if __name__ == "__main__":
    main()
