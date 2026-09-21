"""Subgoal-level instruction-following metrics from runner traces.

    python -m benchmark.metrics --orig OUT_DIR[,OUT_DIR] [--flip OUT_DIR] [--goalonly OUT_DIR] \
        [--subgoals benchmark/data/<set>/subgoals.json] \
        [--radius 1.5] [--json out.json]

Per episode (design doc: subgoal metrics): e_k = first step at or after e_{k-1}
whose position lies within r_b of boundary B_k (the last boundary uses the 3 m
success radius); c = number of boundaries reached in this order, f = c + 1 is
the first failed segment.

ORIG run            SR, SGCR = mean(c / K), SGCR@k = P(c >= k | K >= k)
                    SGCR-eff: c counted with a per-segment path budget
                    (<= max(2 x reference segment length, 3 m)) so that a long
                    wandering trajectory does not collect boundaries by chance
All rates carry a bootstrap 95% CI over episodes.
"""
import argparse
import csv

import numpy as np

from .common import (SUCCESS_DISTANCE_M, data_path, dist_xz, dump_json, load_gt, load_json,
                      load_run, ndtw, path_walked, signed_angle, xz)


# ---------------------------------------------------------------- per-episode scoring
def entry_steps(positions, boundaries, radius, last_radius=SUCCESS_DISTANCE_M):
    """Sequential entry steps: e_k = first step at or after e_{k-1} inside B_k.

    Searching from the previous entry (instead of from the start) is what makes
    the order matter: reaching B_3 before B_2 does not complete 3, and a
    trajectory that merely passes B_k early on the way somewhere else gets no
    credit for it. The search stops at the first boundary never reached; later
    entries are None.
    """
    out = []
    start = 0
    for k, boundary in enumerate(boundaries):
        r = last_radius if k == len(boundaries) - 1 else radius
        hit = None
        for t in range(start, len(positions)):
            if dist_xz(positions[t], boundary) <= r:
                hit = t
                break
        out.append(hit)
        if hit is None:
            out.extend([None] * (len(boundaries) - k - 1))
            break
        start = hit
    return out


def completed_prefix(entries):
    """c: number of boundaries reached in order."""
    return sum(e is not None for e in entries)


def score_episode(positions, subgoals, radius, budget_factor=2.0, budget_min_m=3.0):
    """c: boundaries reached in order; c_eff: the same with a per-segment path budget.

    A long wandering trajectory (CA-Nav walks ~30 m in 250 steps) passes the
    boundaries of a small house in order by chance. c_eff only credits segment
    k when the path walked between e_{k-1} and e_k is at most
    max(budget_factor * reference segment length, budget_min_m); the first
    segment that blows the budget ends the efficient prefix.
    """
    boundaries = [s["boundary_xyz"] for s in subgoals]
    entries = entry_steps(positions, boundaries, radius)
    c = completed_prefix(entries)
    walked = path_walked(positions)
    c_eff = 0
    previous_step, previous_arc = 0, 0.0
    for k, (entry, sub) in enumerate(zip(entries, subgoals)):
        if entry is None:
            break
        budget = max(budget_factor * (sub["arc_end_m"] - previous_arc), budget_min_m)
        if walked[entry] - walked[previous_step] > budget:
            break
        c_eff = k + 1
        previous_step, previous_arc = entry, sub["arc_end_m"]
    return dict(K=len(subgoals), c=c, c_eff=c_eff, f=None if c == len(subgoals) else c + 1, entries=entries)


# ---------------------------------------------------------------- aggregation
def bootstrap_ci(values, rng, rounds=1000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(rounds)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def rate(name, values, rng):
    values = [float(v) for v in values]
    lo, hi = bootstrap_ci(values, rng)
    return dict(name=name, n=len(values), mean=float(np.mean(values)) if values else float("nan"), ci=[lo, hi])


def evaluate(subgoals, orig, radius=1.5, ids=None, seed=0):
    """subgoals: episode -> record from build_subgoals; orig: (positions, results)."""
    rng = np.random.default_rng(seed)
    per = {}
    episodes = [e for e in subgoals if e in orig[0] and (ids is None or e in ids)]
    for episode in episodes:
        record = subgoals[episode]
        row = dict(K=record["K"], sr=orig[1].get(episode, {}).get("success"))
        row.update({"orig_" + k: v for k, v in score_episode(orig[0][episode], record["subgoals"], radius).items()})
        per[episode] = row

    metrics = {}
    metrics["SR"] = rate("SR", [per[e]["sr"] for e in episodes if per[e]["sr"] is not None], rng)
    metrics["SGCR"] = rate("SGCR", [per[e]["orig_c"] / per[e]["K"] for e in episodes], rng)
    metrics["SGCR-eff"] = rate("SGCR-eff", [per[e]["orig_c_eff"] / per[e]["K"] for e in episodes], rng)
    max_k = max((per[e]["K"] for e in episodes), default=0)
    metrics["SGCR@k"] = {k: rate("SGCR@{}".format(k), [per[e]["orig_c"] >= k for e in episodes if per[e]["K"] >= k], rng)
                         for k in range(1, max_k + 1)}

    return metrics, per


# ---------------------------------------------------------------- FLIP-k
def turn_at_anchor(positions, anchor_step, fallback_incoming, scan_m=5.0, in_m=1.5, min_turn=30.0):
    """Signed turn the agent makes after reaching the anchor: 'left', 'right' or 'straight'.

    Incoming = displacement over the last ``in_m`` of walked path before the
    anchor; outgoing = displacement from the anchor to each later point within
    ``scan_m`` of walked path (points >= 0.5 m away); the largest-magnitude angle
    decides. None when the agent walks less than 0.5 m after the anchor.

    NOT bit-identical to build_flip's selection geometry, which walks the
    incoming window with a 2D arc accumulator (and overshoots it by one segment)
    where this uses the 3D cumulative arc. Measured on the 47 flip episodes the
    two incoming headings differ on 4, by at most 5.5 deg. Unifying them shifts
    AwareVLN's TurnMatch by ~2.5 points and leaves v19 / CA-Nav unchanged, so the
    split is kept deliberate rather than silently "fixed": the released flip set
    was selected with build_flip's version, and these numbers are already
    published. Change both together or neither.
    """
    walked = path_walked(positions)
    j = anchor_step
    while j > 0 and walked[anchor_step] - walked[j] < in_m:
        j -= 1
    inc = xz(positions[anchor_step]) - xz(positions[j])
    if np.linalg.norm(inc) < 0.5:
        inc = np.asarray(fallback_incoming, dtype=np.float64)
    inc = inc / np.linalg.norm(inc)
    best, moved = None, False
    for t in range(anchor_step + 1, len(positions)):
        if walked[t] - walked[anchor_step] > scan_m:
            break
        out = xz(positions[t]) - xz(positions[anchor_step])
        n = np.linalg.norm(out)
        if n < 0.5:
            continue
        moved = True
        angle = signed_angle(inc, out / n)
        if best is None or abs(angle) > abs(best):
            best = angle
    if not moved:
        return None, None
    if abs(best) < min_turn:
        return "straight", best
    return ("left" if best < 0 else "right"), best


def evaluate_flip(subgoals, orig, flip, flips, radius=1.5, seed=0):
    """FLIP-k metrics (paired ORIG / FLIP runs on the flip episodes).

    Reached      the agent reached B_{k-1} in order (in the ORIG / FLIP run)
    TurnMatch    ORIG run: turn after B_{k-1} agrees with the original word;
                 FLIP run: agrees with the flipped word (both | reached, moved >= 2 m)
    FlipFollow   FLIP run turned the flipped way while the ORIG run turned the original way
    Bk-reached   the original B_k was still reached afterwards (ORIG vs FLIP run)
    """
    rng = np.random.default_rng(seed)
    per, metrics = {}, {}
    ids = [e for e in flips if e in orig[0] and e in flip[0] and e in subgoals]
    for e in ids:
        meta = flips[e]
        k = meta["k"]
        bounds = subgoals[e]["subgoals"]
        row = dict(k=k, word=meta["word"], flipped=("right" if meta["word"] == "left" else "left"),
                   sr=orig[1].get(e, {}).get("success"), flip_sr=flip[1].get(e, {}).get("success"))
        for tag, run in (("orig", orig), ("flip", flip)):
            positions = run[0][e]
            entries = entry_steps(positions, [b["boundary_xyz"] for b in bounds], radius)
            anchor = entries[k - 2]
            row[tag + "_reached"] = anchor is not None
            row[tag + "_bk"] = entries[k - 1] is not None
            if anchor is not None:
                # refine: the step nearest to B_{k-1} within the next 3 m of walked path
                walked = path_walked(positions)
                point = bounds[k - 2]["boundary_xyz"]
                anchor = min((t for t in range(anchor, len(positions)) if walked[t] - walked[entries[k - 2]] <= 3.0),
                             key=lambda t: dist_xz(positions[t], point))
                turn, angle = turn_at_anchor(positions, anchor, meta["incoming_xz"])
                row[tag + "_turn"], row[tag + "_angle"] = turn, angle
            else:
                row[tag + "_turn"], row[tag + "_angle"] = None, None
        per[e] = row
    both = [e for e in ids if per[e]["orig_turn"] and per[e]["flip_turn"]]
    metrics["n_flip_episodes"] = dict(name="n", n=len(ids), mean=float(len(ids)), ci=[None, None])
    metrics["Reached_orig"] = rate("Reached_orig", [per[e]["orig_reached"] for e in ids], rng)
    metrics["Reached_flip"] = rate("Reached_flip", [per[e]["flip_reached"] for e in ids], rng)
    moved_o = [e for e in ids if per[e]["orig_turn"]]
    moved_f = [e for e in ids if per[e]["flip_turn"]]
    metrics["TurnMatch_orig"] = rate("TurnMatch_orig", [per[e]["orig_turn"] == per[e]["word"] for e in moved_o], rng)
    metrics["TurnMatch_flip"] = rate("TurnMatch_flip", [per[e]["flip_turn"] == per[e]["flipped"] for e in moved_f], rng)
    metrics["TurnOriginalWord_flip"] = rate("TurnOriginalWord_flip", [per[e]["flip_turn"] == per[e]["word"] for e in moved_f], rng)
    metrics["FlipFollow"] = rate("FlipFollow", [per[e]["orig_turn"] == per[e]["word"] and per[e]["flip_turn"] == per[e]["flipped"] for e in both], rng)
    metrics["TurnChanged"] = rate("TurnChanged", [per[e]["orig_turn"] != per[e]["flip_turn"] for e in both], rng)
    metrics["Bk_reached_orig"] = rate("Bk_reached_orig", [per[e]["orig_bk"] for e in ids], rng)
    metrics["Bk_reached_flip"] = rate("Bk_reached_flip", [per[e]["flip_bk"] for e in ids], rng)
    metrics["SR_orig(flip eps)"] = rate("SR_orig", [per[e]["sr"] for e in ids if per[e]["sr"] is not None], rng)
    metrics["SR_flip"] = rate("SR_flip", [per[e]["flip_sr"] for e in ids if per[e]["flip_sr"] is not None], rng)
    return metrics, per


# ---------------------------------------------------------------- GOAL-ONLY
def evaluate_goalonly(subgoals, orig, goal, goals, gt=None, radius=1.5, seed=0):
    """GOAL-ONLY (only the last sub-instruction) vs ORIG, paired.

    SR / SGCR-eff / intermediate-boundary rate / nDTW / PL / steps for both runs.
    Intermediate-boundary rate = fraction of B_1..B_{K-1} reached in order: with
    no route words this measures how much of the route the agent walks anyway
    (layout prior or goal search that happens to follow the route).
    """
    rng = np.random.default_rng(seed)
    per, metrics = {}, {}
    ids = [e for e in goals if e in orig[0] and e in goal[0] and e in subgoals]
    for e in ids:
        bounds = subgoals[e]["subgoals"]
        row = dict(K=len(bounds))
        for tag, run in (("orig", orig), ("goal", goal)):
            positions = run[0][e]
            scored = score_episode(positions, bounds, radius)
            inter = scored["entries"][:-1]
            row[tag + "_sr"] = run[1].get(e, {}).get("success")
            row[tag + "_steps"] = run[1].get(e, {}).get("steps")
            row[tag + "_sgcr_eff"] = scored["c_eff"] / len(bounds)
            row[tag + "_inter"] = sum(x is not None for x in inter) / max(len(inter), 1)
            row[tag + "_pl"] = path_walked(positions)[-1]
            row[tag + "_ndtw"] = ndtw(positions, gt[e]) if gt else None
        per[e] = row
    for key, label in (("sr", "SR"), ("sgcr_eff", "SGCR-eff"), ("inter", "IntermediateBoundaries"), ("ndtw", "nDTW"), ("pl", "PL_m"), ("steps", "Steps")):
        for tag in ("orig", "goal"):
            vals = [per[e][tag + "_" + key] for e in ids if per[e][tag + "_" + key] is not None]
            metrics["{}_{}".format(label, tag)] = rate("{}_{}".format(label, tag), vals, rng)
    metrics["SR_kept"] = rate("SR_kept", [per[e]["goal_sr"] for e in ids if per[e]["orig_sr"] == 1.0 and per[e]["goal_sr"] is not None], rng)
    metrics["SR_gained"] = rate("SR_gained", [per[e]["goal_sr"] for e in ids if per[e]["orig_sr"] == 0.0 and per[e]["goal_sr"] is not None], rng)
    return metrics, per


def format_table(metrics):
    lines = []
    def line(item, label=None):
        lo, hi = item["ci"]
        ci = "" if lo is None else "  [{:.3f}, {:.3f}]".format(lo, hi)
        lines.append("{:<32} n={:<4} {:.3f}{}".format(label or item["name"], item["n"], item["mean"], ci))
    for key, item in metrics.items():
        if key == "SGCR@k":
            for k, sub in item.items():
                line(sub, "SGCR@{}".format(k))
        else:
            line(item, key)
    return "\n".join(lines)


def _load(spec):
    return load_run(spec.split(",")) if spec else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orig", required=True, help="ORIG run output dir(s), comma separated")
    parser.add_argument("--flip", help="FLIP-k run output dir(s)")
    parser.add_argument("--flip-meta", default=str(data_path("flip")))
    parser.add_argument("--goalonly", help="GOAL-ONLY run output dir(s)")
    parser.add_argument("--goalonly-meta", default=str(data_path("goalonly")))
    parser.add_argument("--subgoals", default=str(data_path("subgoals")))
    parser.add_argument("--radius", type=float, default=1.5, help="boundary radius r_b in metres")
    parser.add_argument("--ids", help="restrict to an id list file")
    parser.add_argument("--json", help="write metrics + per-episode rows here")
    parser.add_argument("--csv", help="write per-episode rows here")
    args = parser.parse_args()

    subgoals = load_json(args.subgoals)
    gt = load_gt(subgoals["split"])
    ids = None
    if args.ids:
        from .common import read_id_list
        ids = set(read_id_list(args.ids))
    metrics, per = evaluate(subgoals["episodes"], _load(args.orig), radius=args.radius, ids=ids)
    print(format_table(metrics))
    if args.flip:
        flip_metrics, flip_per = evaluate_flip(subgoals["episodes"], _load(args.orig), _load(args.flip),
                                               load_json(args.flip_meta)["flip"], radius=args.radius)
        print("--- FLIP-k")
        print(format_table(flip_metrics))
        metrics.update({"flip:" + k: v for k, v in flip_metrics.items()})
        for e, row in flip_per.items():
            per.setdefault(e, {}).update({"flip_" + k if not k.startswith(("orig_", "flip_")) else k: v for k, v in row.items()})
    if args.goalonly:
        goal_metrics, goal_per = evaluate_goalonly(subgoals["episodes"], _load(args.orig), _load(args.goalonly),
                                                   load_json(args.goalonly_meta)["goalonly"], gt=gt, radius=args.radius)
        print("--- GOAL-ONLY")
        print(format_table(goal_metrics))
        metrics.update({"goalonly:" + k: v for k, v in goal_metrics.items()})
        for e, row in goal_per.items():
            per.setdefault(e, {}).update({"goalonly_" + k: v for k, v in row.items()})
    if args.json:
        dump_json(dict(args=vars(args), metrics=metrics, episodes=per), args.json)
    if args.csv:
        keys = sorted({k for row in per.values() for k in row if k != "orig_entries"})
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["episode_id"] + keys, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for episode, row in per.items():
                writer.writerow(dict(row, episode_id=episode))


if __name__ == "__main__":
    main()
