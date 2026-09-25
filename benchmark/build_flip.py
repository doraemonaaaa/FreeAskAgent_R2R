"""FLIP variant: reverse left/right in the FIRST sub-instruction, where every agent starts.

    <habitat python> -m benchmark.build_flip [--name val_unseen] [--min-turn 45] [--no-navmesh]

The turn is at the start position, which every agent occupies, so every
selected episode is scorable for every system. (The earlier FLIP-k design
flipped a later sentence and could only score agents that reached it, so each
system was scored on a different, easier subset; it was retired 2026-09-24.)

The reference heading is the dataset's start rotation. R2R-CE v1-2 (the FreeAskAgent
runner's split) and v1-3 (CA-Nav, AwareVLN) share start positions but not start
rotations (median 96 deg apart on val_unseen), and only v1-3 agrees with the
instruction text: on first chunks saying "turn left/right" whose reference path
turns >= 45 deg, the sign matches 24/32 with v1-3 and 6/26 with v1-2. So the
v1-3 rotation is used, and the FreeAskAgent runner splits written here are the v1-2 episodes
with ONLY start_rotation replaced by v1-3's (ORIG and FLIP-1 alike: the ORIG
control has to be rerun on the same heading).

Selection (per episode, sub-instruction k = 1 only):
  * the chunk contains exactly one "left"/"right" token (a repeated word would
    leave a mixed instruction after flipping the first occurrence);
  * that token is a turn command ("turn left", "make a right", "go left",
    "veer to the right", "take the first left", ...), not a landmark relation
    ("with the table to your right", "the door on the left"): flipping a
    relation changes which landmark is meant, not which way to turn;
  * the reference path turns that way from the start heading: the largest
    signed angle between the heading and the displacement to any point in the
    first 5 m of the dense reference path has min_turn <= |angle| <= max_turn
    and the sign of the word (left < 0, right > 0 in habitat's x/z plane); near
    180 deg the side of a turn-around is arbitrary, hence the upper bound;
  * the mirrored direction is navigable: a point 2 m from the start along the
    outgoing direction reflected about the heading snaps to the navmesh within
    0.75 m (needs habitat_sim; --no-navmesh skips it).
``balanced`` lists an equal number of left and right episodes (the smaller
count, chosen by a seeded shuffle) so a left/right bias cannot pass for
instruction following.

Outputs
  benchmark/data/<set>/subgoals.json                 built for all ids of the split if missing
  benchmark/data/<set>/rule/flip.json                per-episode word, angle, headings, chunk span, flipped text
  benchmark/data/<set>/rule/ids/flip.txt             balanced episode ids
  <habitat r2r>/<set>_flip{_orig,}/...json.gz        FreeAskAgent runner splits (balanced ids, v1-3 start rotation)
  CA-Nav / AwareVLN inputs are written by benchmark.canav / benchmark.awarevln ``build``.
"""
import argparse
import gzip
import json
import random
import re

import numpy as np

from .build_subgoals import build as build_subgoals
from .common import (DIRECTION, R2R_V13_VAL_UNSEEN, R2R_DIR, data_path, dump_json, ids_path, load_episodes, load_gt, load_json,
                     signed_angle, write_split, xz)

OPPOSITE = {"left": "right", "right": "left"}


TURN_COMMAND = re.compile(
    r"\b(?:turn(?:ing)?|make|take|hang|go(?:ing)?|head(?:ing)?|veer|bear|swing|walk|move|step|exit|enter|proceed|continue)"
    r"(?:\s+(?:a|an|the|another|first|second|next|immediate|slight|sharp|hard|quick|90|degree|degrees|to|towards|toward|your|out|up|down|straight|and|then|immediately|slightly|sharply|back|around|over|through|into|onto|off|in|at|another))*"
    r"\s+(left|right)\b(?!\s+of\b)(?!\s+side\s+of\b)", re.IGNORECASE)
RELATION = re.compile(r"\bwith\b[^,.;]*\b(?:left|right)\b|\b(?:on|at|by|from)\s+(?:the|your)\s+(?:far\s+)?(?:left|right)\b",
                      re.IGNORECASE)


def is_turn_command(text):
    """The (single) left/right token of ``text`` follows a motion verb and is not a landmark relation."""
    return TURN_COMMAND.search(text) is not None and RELATION.search(text) is None


def flip_text(text, span, word):
    """Replace the direction word inside the span (first occurrence), keeping case."""
    segment = text[span[0]:span[1]]
    m = re.search(r"\b" + word + r"\b", segment, re.IGNORECASE)
    new = OPPOSITE[word]
    new = new.capitalize() if m.group(0)[0].isupper() else new
    segment = segment[: m.start()] + new + segment[m.end():]
    return text[: span[0]] + segment + text[span[1]:]


def navigable(pathfinders, scene_id, point, tol=0.75):
    import habitat_sim
    scene = scene_id.split("/")[1]
    if scene not in pathfinders:
        pf = habitat_sim.PathFinder()
        pf.load_nav_mesh(str(R2R_DIR.parents[4] / "scene_datasets" / "mp3d" / scene / (scene + ".navmesh")))
        pathfinders[scene] = pf
    pf = pathfinders[scene]
    snapped = np.asarray(pf.snap_point(np.asarray(point, dtype=np.float32)))
    if not np.isfinite(snapped).all():
        return False
    return float(np.linalg.norm(snapped - np.asarray(point))) <= tol and abs(snapped[1] - point[1]) < 0.5


def heading_xz(rotation):
    """Unit x/z forward vector (-z rotated) of an [x, y, z, w] start_rotation."""
    u = np.asarray(rotation[:3], dtype=np.float64)
    w = float(rotation[3])
    v = np.array([0.0, 0.0, -1.0])
    f = v + 2.0 * np.cross(u, np.cross(u, v) + w * v)
    f = np.array([f[0], f[2]])
    return f / np.linalg.norm(f)


def start_turn(points, heading, limit=5.0):
    """Largest signed angle from the start heading to any point in the first ``limit`` m (>= 0.5 m away)."""
    best, best_vec, walked = 0.0, None, 0.0
    for j in range(1, len(points)):
        walked += float(np.linalg.norm(xz(points[j]) - xz(points[j - 1])))
        if walked > limit:
            break
        v = xz(points[j]) - xz(points[0])
        n = float(np.linalg.norm(v))
        if n < 0.5:
            continue
        angle = signed_angle(heading, v / n)
        if abs(angle) > abs(best):
            best, best_vec = angle, v / n
    return best, best_vec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="val_unseen", help="eval set name under benchmark/data/")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--min-turn", type=float, default=45.0)
    parser.add_argument("--max-turn", type=float, default=135.0)
    parser.add_argument("--no-navmesh", action="store_true")
    parser.add_argument("--no-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes, raw = load_episodes(args.split)
    gt = load_gt(args.split)
    sg_path = data_path("subgoals", args.name)
    if not sg_path.exists():
        built, skipped = build_subgoals(list(episodes), args.split)
        dump_json(dict(split=args.split, ids_file="all ids of {}".format(args.split),
                       provenance="derived from the FGR2R human sub-instruction annotation (Hong et al. 2020)",
                       episodes=built, skipped=skipped), sg_path)
        print("wrote", sg_path, "episodes", len(built), "skipped", len(skipped))
    subgoals = load_json(sg_path)["episodes"]
    with gzip.open(str(R2R_V13_VAL_UNSEEN), "rt") as handle:
        v13 = {str(e["episode_id"]): e for e in json.load(handle)["episodes"]}

    pathfinders = {}
    out = {}
    rejected = {"no_word": 0, "repeated_or_both_words": 0, "no_span": 0, "not_turn_command": 0, "weak_turn": 0, "turn_around": 0, "wrong_sign": 0, "not_navigable": 0}
    for eid, rec in subgoals.items():
        first = rec["subgoals"][0]
        found = [w.lower() for w in DIRECTION.findall(first["text"])]
        if not found:
            rejected["no_word"] += 1
            continue
        if len(found) != 1:
            rejected["repeated_or_both_words"] += 1
            continue
        if not first["span"]:
            rejected["no_span"] += 1
            continue
        if not is_turn_command(first["text"]):
            rejected["not_turn_command"] += 1
            continue
        word = found[0]
        heading = heading_xz(v13[eid]["start_rotation"])
        angle, out_vec = start_turn(gt[eid], heading)
        if abs(angle) < args.min_turn:
            rejected["weak_turn"] += 1
            continue
        if abs(angle) > args.max_turn:
            rejected["turn_around"] += 1
            continue
        if (angle < 0) != (word == "left"):
            rejected["wrong_sign"] += 1
            continue
        mirrored = 2 * np.dot(out_vec, heading) * heading - out_vec
        start = np.asarray(gt[eid][0], dtype=np.float64)
        probe = start + np.array([mirrored[0] * 2.0, 0.0, mirrored[1] * 2.0])
        if not args.no_navmesh and not navigable(pathfinders, rec["scene_id"], probe):
            rejected["not_navigable"] += 1
            continue
        v12_heading = heading_xz(episodes[eid]["start_rotation"])
        out[eid] = dict(k=1, word=word, flipped=OPPOSITE[word], turn_deg=float(angle),
                        start_xyz=[float(v) for v in start], incoming_xz=[float(v) for v in heading],
                        outgoing_xz=[float(v) for v in out_vec], mirrored_xz=[float(v) for v in mirrored],
                        start_rotation_v13=v13[eid]["start_rotation"],
                        v12_heading_offset_deg=signed_angle(heading, v12_heading),
                        scene=rec["scene_id"].split("/")[1], original_text=first["text"], span=first["span"],
                        original_instruction=rec["instruction"],
                        instruction=flip_text(rec["instruction"], first["span"], word))

    lefts = sorted(e for e, v in out.items() if v["word"] == "left")
    rights = sorted(e for e, v in out.items() if v["word"] == "right")
    rng = random.Random(args.seed)
    rng.shuffle(lefts)
    rng.shuffle(rights)
    m = min(len(lefts), len(rights))
    balanced = sorted(lefts[:m] + rights[:m], key=int)

    meta = dict(name=args.name, split=args.split, k=1, min_turn=args.min_turn, max_turn=args.max_turn, navmesh=not args.no_navmesh, seed=args.seed,
                heading_source="R2R_VLNCE_v1-3 start_rotation (FreeAskAgent runner splits: v1-2 episodes with this rotation)",
                provenance="rule-based minimal pair: the single left/right word of sub-instruction 1 replaced, everything else byte-identical",
                n_candidates=len(out), n_left=len(lefts), n_right=len(rights), balanced=balanced,
                flip=out, rejected=rejected)
    path = data_path("flip", args.name)
    dump_json(meta, path)
    scenes = {out[e]["scene"] for e in balanced}
    offsets = np.abs([out[e]["v12_heading_offset_deg"] for e in balanced])
    print("candidates={} (left {}, right {}) balanced={} over {} scenes; rejected={}".format(
        len(out), len(lefts), len(rights), len(balanced), len(scenes), rejected))
    print("v1-2 start heading off by >45 deg on {}/{} balanced episodes".format(int((offsets > 45).sum()), len(balanced)))
    print("wrote", path)

    ids = ids_path("flip", args.name)
    ids.parent.mkdir(parents=True, exist_ok=True)
    ids.write_text("# FLIP balanced episode ids ({} left + {} right), set {}\n".format(m, m, args.name)
                   + "".join(e + "\n" for e in balanced))
    print("wrote ids", ids)
    if not args.no_splits:
        rotated = {e: dict(episodes[e], start_rotation=out[e]["start_rotation_v13"]) for e in balanced}
        for suffix, texts in (("flip_orig", {e: subgoals[e]["instruction"] for e in balanced}),
                              ("flip", {e: out[e]["instruction"] for e in balanced})):
            directory = write_split("{}_{}".format(args.name, suffix), raw, rotated, gt, texts)
            print("wrote split", directory)


if __name__ == "__main__":
    main()
