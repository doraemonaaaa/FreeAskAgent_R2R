"""FLIP-k variant: reverse left/right in one turning sub-instruction.

    <habitat python> -m benchmark.build_flip [--min-turn 45] [--no-navmesh]

Selection (per episode, one segment k >= 2):
  * the sub-instruction contains exactly one of "left" / "right";
  * the reference path turns that way inside the segment: signed angle between
    the incoming direction (last 1.5 m before B_{k-1}) and the displacement
    over the first 5 m of the segment reaches |angle| >= min_turn with the
    sign of the word (left < 0, right > 0 in habitat's x/z plane, verified on
    FGR2R val_unseen: 46/50 "turn left/right" segments agree);
  * the mirrored direction is navigable: a point 2 m from B_{k-1} along the
    outgoing direction reflected about the incoming direction snaps to the
    navmesh within 0.75 m (needs habitat_sim; --no-navmesh skips it);
  * k = 1 is excluded: the incoming heading there is the dataset's start
    rotation, which differs between R2R-CE v1-2 and v1-3.
When several segments qualify the one with the largest turn is used.

Outputs
  benchmark/data/flip_<set>.json                 per-episode k, word, angle, flipped text, anchor point
  <habitat r2r>/<set>_flip/<set>_flip.json.gz    split for the v19 runner (+ gt copy, + ids file)
  CA-Nav / AwareVLN splits are written by benchmark.canav / benchmark.awarevln ``build --flip``.
"""
import argparse
import math
import re

import numpy as np

from .common import DATA_DIR, R2R_DIR, dump_json, load_episodes, load_gt, load_json

WORD = re.compile(r"\b(left|right)\b", re.IGNORECASE)
OPPOSITE = {"left": "right", "right": "left"}


def xz(p):
    return np.asarray(p, dtype=np.float64)[[0, 2]]


def signed_angle(a, b):
    return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


def incoming_direction(pts, i, dist=1.5):
    j, acc = i, 0.0
    while j > 0 and acc < dist:
        acc += np.linalg.norm(xz(pts[j]) - xz(pts[j - 1]))
        j -= 1
    v = xz(pts[i]) - xz(pts[j])
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def segment_turn(pts, i0, i1, inc, limit=5.0):
    """Signed angle (and outgoing unit vector) of the largest heading change in the first ``limit`` m."""
    best, best_vec, acc = 0.0, None, 0.0
    for j in range(i0 + 1, i1 + 1):
        acc += np.linalg.norm(xz(pts[j]) - xz(pts[j - 1]))
        if acc > limit:
            break
        v = xz(pts[j]) - xz(pts[i0])
        n = np.linalg.norm(v)
        if n < 0.5:
            continue
        a = signed_angle(inc, v / n)
        if abs(a) > abs(best):
            best, best_vec = a, v / n
    return best, best_vec


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subgoals", default=str(DATA_DIR / "subgoals_val_unseen_200.json"))
    parser.add_argument("--name", default="val_unseen_200")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--min-turn", type=float, default=45.0)
    parser.add_argument("--no-navmesh", action="store_true")
    parser.add_argument("--no-splits", action="store_true")
    args = parser.parse_args()

    subgoals = load_json(args.subgoals)["episodes"]
    gt = load_gt(args.split)
    episodes, raw = load_episodes(args.split)
    pathfinders = {}
    out, rejected = {}, {"no_word": 0, "k1_only": 0, "weak_turn": 0, "wrong_sign": 0, "not_navigable": 0}
    for eid, rec in subgoals.items():
        pts = gt[eid]
        best = None
        had_word = had_k1 = False
        for s in rec["subgoals"]:
            words = {w.lower() for w in WORD.findall(s["text"])}
            if len(words) != 1 or not s["span"]:
                continue
            had_word = True
            k, word = s["index"], words.pop()
            if k == 1:
                had_k1 = True
                continue
            i0, i1 = rec["subgoals"][k - 2]["gt_index"], s["gt_index"]
            inc = incoming_direction(pts, i0)
            if inc is None:
                continue
            angle, out_vec = segment_turn(pts, i0, i1, inc)
            if abs(angle) < args.min_turn:
                rejected["weak_turn"] += 1
                continue
            if (angle < 0) != (word == "left"):
                rejected["wrong_sign"] += 1
                continue
            # mirror the outgoing direction about the incoming direction
            mirrored = 2 * np.dot(out_vec, inc) * inc - out_vec
            anchor = np.asarray(pts[i0], dtype=np.float64)
            probe = anchor + np.array([mirrored[0] * 2.0, 0.0, mirrored[1] * 2.0])
            if not args.no_navmesh and not navigable(pathfinders, rec["scene_id"], probe):
                rejected["not_navigable"] += 1
                continue
            if best is None or abs(angle) > abs(best["turn_deg"]):
                best = dict(k=k, word=word, turn_deg=float(angle), anchor_xyz=[float(v) for v in anchor],
                            incoming_xz=[float(v) for v in inc], outgoing_xz=[float(v) for v in out_vec],
                            mirrored_xz=[float(v) for v in mirrored], original_text=s["text"],
                            instruction=flip_text(rec["instruction"], s["span"], word))
        if best:
            out[eid] = best
        elif had_word and had_k1 and not had_word:
            pass
        elif not had_word:
            rejected["no_word"] += 1
        elif had_k1:
            rejected["k1_only"] += 1
    meta = dict(name=args.name, split=args.split, min_turn=args.min_turn, navmesh=not args.no_navmesh,
                flip_split="{}_flip".format(args.name), flip=out, rejected=rejected)
    path = DATA_DIR / "flip_{}.json".format(args.name)
    dump_json(meta, path)
    words = [v["word"] for v in out.values()]
    print("flip episodes={} (left {}, right {}) rejected={}".format(len(out), words.count("left"), words.count("right"), rejected))
    print("wrote", path)
    if not args.no_splits:
        from .build_variants import write_split
        directory = write_split(meta["flip_split"], raw, episodes, gt, {eid: v["instruction"] for eid, v in out.items()})
        ids = DATA_DIR / "{}_flip_ids.txt".format(args.name)
        ids.write_text("# episode ids present in split {}_flip\n".format(args.name) + "".join(eid + "\n" for eid in out))
        print("wrote split", directory, "ids", ids)


if __name__ == "__main__":
    main()
