"""Subgoal-level instruction-following metrics from runner traces.

    python -m benchmark.metrics --orig OUT_DIR[,OUT_DIR] [--goalonly OUT_DIR] [--compare OUT_DIR] \
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
FLIP                python -m benchmark.metrics --orig <flip_orig run> --flip <flip run>:
                    turn at the start per run (follow / opposite / straight / none),
                    MeanFollow, Blind, WordEffect; see evaluate_flip
--compare / GOAL-ONLY  paired agreement with ORIG (FlipRate, SuccessKept,
                    Kappa, McNemar_p, EndpointGap, |dNE|); see evaluate_paired
All rates carry a bootstrap 95% CI over episodes.
"""
import argparse
import csv
import math

import numpy as np

from .build_flip import start_turn
from .common import (SUCCESS_DISTANCE_M, data_path, dist_xz, dump_json, load_gt, load_json,
                      load_run, ndtw, path_walked)


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


# ---------------------------------------------------------------- FLIP
TURN_THRESHOLD_DEG = 30.0


def classify_start_turn(positions, heading, given):
    """Which way the agent left the start relative to the start heading, against the word it was given.

    Same geometry as build_flip.start_turn (largest signed angle to any point
    within the first 5 m of x/z path, points >= 0.5 m from the start), applied
    to the agent's positions instead of the reference path. Returns
    (category, angle): 'follow' / 'opposite' / 'straight' (|angle| < 30 deg) /
    'none' (never got 0.5 m from the start within those 5 m).
    """
    angle, vec = start_turn(positions, np.asarray(heading, dtype=np.float64))
    if vec is None:
        return "none", None
    if abs(angle) < TURN_THRESHOLD_DEG:
        return "straight", angle
    side = "left" if angle < 0 else "right"
    return ("follow" if side == given else "opposite"), angle


def evaluate_flip(orig, flip, meta, ids=None, seed=0):
    """FLIP: paired ORIG / FLIP runs, turn at the start (build_flip), every episode scored.

    Per run, each episode is one of follow / opposite / straight / none (the
    four rates sum to 1; "follow" = turned the way of the word THAT run was
    given). Pooled over both runs:
    MeanFollow  mean follow rate of the two runs (1 = always turns as told)
    Blind       what a system that ignores the word (same turn in both runs)
                would score: (1 - mean non-turn rate) / 2
    WordEffect  MeanFollow - Blind, per episode (0 = word ignored, ~0.5 = perfect)
    BothFollow  followed in both runs; SameSide = turned the same physical side
                in both runs (left/right, whatever the word)
    FollowGiven_left/right  follow rate by the word given, pooled over both runs
                (a left/right bias shows up here; the episode set is balanced)
    """
    rng = np.random.default_rng(seed)
    flips = meta["flip"]
    wanted = meta["balanced"] if ids is None else [e for e in meta["balanced"] if e in ids]
    eps = [e for e in wanted if e in orig[0] and e in flip[0]]
    per = {}
    for e in eps:
        v = flips[e]
        row = dict(word=v["word"], flipped=v["flipped"],
                   orig_sr=orig[1].get(e, {}).get("success"), flip_sr=flip[1].get(e, {}).get("success"))
        for tag, run, given in (("orig", orig, v["word"]), ("flip", flip, v["flipped"])):
            positions = run[0][e]
            row[tag + "_start_offset_m"] = dist_xz(positions[0], v["start_xyz"])
            category, angle = classify_start_turn(positions, v["incoming_xz"], given)
            row[tag + "_turn"], row[tag + "_angle"] = category, angle
        per[e] = row

    metrics = dict(n=dict(name="n", n=len(eps), mean=float(len(eps)), ci=[None, None]))
    for tag in ("orig", "flip"):
        for category in ("follow", "opposite", "straight", "none"):
            key = "{}_{}".format(tag.upper(), category)
            metrics[key] = rate(key, [per[e][tag + "_turn"] == category for e in eps], rng)
    follow = lambda e, tag: per[e][tag + "_turn"] == "follow"
    nonturn = lambda e, tag: per[e][tag + "_turn"] in ("straight", "none")
    side = lambda e, tag: (per[e]["word"] if tag == "orig" else per[e]["flipped"]) if per[e][tag + "_turn"] == "follow" else \
        ((per[e]["flipped"] if tag == "orig" else per[e]["word"]) if per[e][tag + "_turn"] == "opposite" else None)
    metrics["MeanFollow"] = rate("MeanFollow", [(follow(e, "orig") + follow(e, "flip")) / 2 for e in eps], rng)
    metrics["Blind"] = rate("Blind", [(1 - (nonturn(e, "orig") + nonturn(e, "flip")) / 2) / 2 for e in eps], rng)
    metrics["WordEffect"] = rate("WordEffect", [(follow(e, "orig") + follow(e, "flip")) / 2
                                                - (1 - (nonturn(e, "orig") + nonturn(e, "flip")) / 2) / 2 for e in eps], rng)
    metrics["BothFollow"] = rate("BothFollow", [follow(e, "orig") and follow(e, "flip") for e in eps], rng)
    metrics["SameSide"] = rate("SameSide", [side(e, "orig") is not None and side(e, "orig") == side(e, "flip") for e in eps], rng)
    for word in ("left", "right"):
        given = [follow(e, "orig") for e in eps if per[e]["word"] == word] + [follow(e, "flip") for e in eps if per[e]["flipped"] == word]
        metrics["FollowGiven_" + word] = rate("FollowGiven_" + word, given, rng)
    metrics["SR_orig"] = rate("SR_orig", [per[e]["orig_sr"] for e in eps if per[e]["orig_sr"] is not None], rng)
    metrics["SR_flip"] = rate("SR_flip", [per[e]["flip_sr"] for e in eps if per[e]["flip_sr"] is not None], rng)
    offsets = [per[e][t + "_start_offset_m"] for e in eps for t in ("orig", "flip")]
    metrics["StartOffset_max_m"] = dict(name="StartOffset_max_m", n=len(offsets), mean=max(offsets, default=0.0), ci=[None, None])
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


# ---------------------------------------------------------------- paired agreement
def cohen_kappa(a, b):
    """Cohen's kappa of two binary vectors; nan when chance agreement is 1."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    po = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def evaluate_paired(orig, other, ids=None, seed=0):
    """Per-episode agreement of two runs on the same episodes (paraphrase arm, rerun, GOAL-ONLY).

    Equal SR can hide heavy churn: the two runs may solve different episodes.
    FlipRate     P(success differs) -- compare against an ORIG-vs-ORIG rerun
                 (noise floor) before attributing it to the instruction change
    SuccessKept  P(other succeeds | ORIG succeeds)
    Kappa        Cohen's kappa of the success vectors (1 = same episodes, 0 = chance)
    McNemar_p    exact test that the discordant pairs are symmetric (a real SR shift)
    EndpointGap  xz distance between the two final positions (threshold-free)
    |dNE|        |final distance to goal ORIG - other| (threshold-free)
    """
    rng = np.random.default_rng(seed)
    per, metrics = {}, {}
    common = [e for e in orig[1] if e in other[1] and (ids is None or e in ids)]
    for e in common:
        row = dict(orig_sr=orig[1][e]["success"], other_sr=other[1][e]["success"],
                   ne_gap=abs(orig[1][e]["dtg"] - other[1][e]["dtg"]))
        if orig[0].get(e) and other[0].get(e):
            row["end_gap"] = dist_xz(orig[0][e][-1], other[0][e][-1])
        per[e] = row
    a = np.array([per[e]["orig_sr"] > 0 for e in common])
    b = np.array([per[e]["other_sr"] > 0 for e in common])
    n_both, n_orig, n_other = int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b))
    count = lambda name, v: dict(name=name, n=len(common), mean=float(v), ci=[None, None])
    metrics["SR_orig"] = rate("SR_orig", a, rng)
    metrics["SR_other"] = rate("SR_other", b, rng)
    metrics["Solved_both"] = count("Solved_both", n_both)
    metrics["Solved_orig_only"] = count("Solved_orig_only", n_orig)
    metrics["Solved_other_only"] = count("Solved_other_only", n_other)
    metrics["FlipRate"] = rate("FlipRate", a != b, rng)
    metrics["SuccessKept"] = rate("SuccessKept", b[a], rng)
    kappas = []
    for _ in range(1000):
        idx = rng.integers(0, len(common), len(common))
        kappas.append(cohen_kappa(a[idx], b[idx]))
    kappas = [k for k in kappas if not np.isnan(k)]
    ci = [float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))] if kappas else [None, None]
    metrics["Kappa"] = dict(name="Kappa", n=len(common), mean=cohen_kappa(a, b) if common else float("nan"), ci=ci)
    metrics["McNemar_p"] = count("McNemar_p", mcnemar_exact(n_orig, n_other))
    metrics["EndpointGap_m"] = rate("EndpointGap_m", [per[e]["end_gap"] for e in common if "end_gap" in per[e]], rng)
    metrics["|dNE|_m"] = rate("|dNE|_m", [per[e]["ne_gap"] for e in common], rng)
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
    parser.add_argument("--flip", help="FLIP run output dir(s); --orig is then the matching flip_orig run")
    parser.add_argument("--flip-meta", default=str(data_path("flip", "val_unseen")))
    parser.add_argument("--goalonly", help="GOAL-ONLY run output dir(s)")
    parser.add_argument("--goalonly-meta", default=str(data_path("goalonly")))
    parser.add_argument("--compare", help="second run on the same episodes (paraphrase arm or ORIG rerun): paired agreement")
    parser.add_argument("--subgoals", help="default: the 200-set, or the FLIP set's subgoals when --flip is given")
    parser.add_argument("--radius", type=float, default=1.5, help="boundary radius r_b in metres")
    parser.add_argument("--ids", help="restrict to an id list file")
    parser.add_argument("--json", help="write metrics + per-episode rows here")
    parser.add_argument("--csv", help="write per-episode rows here")
    args = parser.parse_args()

    flip_meta = load_json(args.flip_meta) if args.flip else None
    args.subgoals = args.subgoals or str(data_path("subgoals", flip_meta["name"]) if flip_meta else data_path("subgoals"))
    subgoals = load_json(args.subgoals)
    gt = load_gt(subgoals["split"])
    ids = None
    if args.ids:
        from .common import read_id_list
        ids = set(read_id_list(args.ids))
    if flip_meta and ids is None:
        ids = set(flip_meta["balanced"])
    metrics, per = evaluate(subgoals["episodes"], _load(args.orig), radius=args.radius, ids=ids)
    print(format_table(metrics))
    if args.flip:
        flip_metrics, flip_per = evaluate_flip(_load(args.orig), _load(args.flip), flip_meta, ids=ids)
        print("--- FLIP (turn at the start)")
        print(format_table(flip_metrics))
        metrics.update({"flip:" + k: v for k, v in flip_metrics.items()})
        for e, row in flip_per.items():
            per.setdefault(e, {}).update({"flip_" + k: v for k, v in row.items()})
        flip_paired, _ = evaluate_paired(_load(args.orig), _load(args.flip), ids=set(flip_per))
        print("--- FLIP paired agreement (success)")
        print(format_table(flip_paired))
        metrics.update({"flip_paired:" + k: v for k, v in flip_paired.items()})
    if args.goalonly:
        goal_metrics, goal_per = evaluate_goalonly(subgoals["episodes"], _load(args.orig), _load(args.goalonly),
                                                   load_json(args.goalonly_meta)["goalonly"], gt=gt, radius=args.radius)
        print("--- GOAL-ONLY")
        print(format_table(goal_metrics))
        metrics.update({"goalonly:" + k: v for k, v in goal_metrics.items()})
        for e, row in goal_per.items():
            per.setdefault(e, {}).update({"goalonly_" + k: v for k, v in row.items()})
        goal_paired, _ = evaluate_paired(_load(args.orig), _load(args.goalonly),
                                         ids=set(load_json(args.goalonly_meta)["goalonly"]))
        print("--- GOAL-ONLY paired agreement")
        print(format_table(goal_paired))
        metrics.update({"goalonly_paired:" + k: v for k, v in goal_paired.items()})
    if args.compare:
        paired, paired_per = evaluate_paired(_load(args.orig), _load(args.compare), ids=ids)
        print("--- paired agreement (ORIG vs --compare)")
        print(format_table(paired))
        metrics.update({"paired:" + k: v for k, v in paired.items()})
        for e, row in paired_per.items():
            per.setdefault(e, {}).update({"paired_" + k: v for k, v in row.items()})
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
