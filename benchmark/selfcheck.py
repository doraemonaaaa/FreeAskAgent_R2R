"""Sanity check of the metrics with two fake agents, plus r_b calibration.

    python -m benchmark.selfcheck [--noise 0.3] [--repeats 20]

Reference-path agent (obeys every instruction)
  ORIG   walks the dense reference path
  SWAP   walks the donor's reference path
  DROP-k walks segments 1..k-1 and stops (no instruction for segment k)
Shortest-path oracle (ignores the instruction)
  walks the ORIG reference path in all three variants.

Expected (design doc section 5): the reference agent scores SGCR = 1,
PathAttrib = 1, LocFail high and Skip = 0; the oracle scores SR = 1 but
ISens ~ 0, PathAttrib ~ 0, Skip = 1. r_b is the smallest radius at which the
noisy reference agent keeps SGCR >= 0.98 on ORIG.
"""
import argparse

import numpy as np

from .common import DATA_DIR, load_gt, load_json, resample
from .metrics import evaluate, format_table


def noisy(positions, sigma, rng):
    """Localisation-style noise: a constant per-episode offset plus a slow random
    walk, both bounded by ``sigma`` (i.i.d. per-step noise would inflate the
    walked path length and defeat the SGCR-eff path budget)."""
    if sigma <= 0:
        return positions
    offset = rng.normal(0, sigma / 2, 2)
    drift = np.zeros(2)
    out = []
    for p in positions:
        drift = np.clip(drift + rng.normal(0, sigma / 10, 2), -sigma, sigma)
        out.append([p[0] + offset[0] + drift[0], p[1], p[2] + offset[1] + drift[1]])
    return out


def fake_runs(subgoals, variants, gt, agent, sigma, rng):
    orig_p, orig_r, swap_p, swap_r, drop_p, drop_r = {}, {}, {}, {}, {}, {}
    for episode, record in subgoals.items():
        path = gt[episode]
        orig_p[episode] = noisy(resample(path), sigma, rng)
        orig_r[episode] = dict(success=1.0)
        if episode in variants["swap"]:
            donor = variants["swap"][episode]["donor_episode_id"]
            swap_p[episode] = noisy(resample(gt[donor] if agent == "reference" else path), sigma, rng)
            swap_r[episode] = dict(success=float(agent == "oracle"))
        if episode in variants["drop"]:
            k = variants["drop"][episode]["k"]
            if agent == "reference":
                # no instruction for segment k: walk segments 1..k-1 and stop there
                bounds = record["subgoals"]
                walk = path[: (bounds[k - 2]["gt_index"] + 1) if k >= 2 else 1]
                drop_r[episode] = dict(success=0.0)
            else:
                walk = path
                drop_r[episode] = dict(success=1.0)
            drop_p[episode] = noisy(resample(walk), sigma, rng)
    return (orig_p, orig_r), (swap_p, swap_r), (drop_p, drop_r)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subgoals", default=str(DATA_DIR / "subgoals_val_unseen_200.json"))
    parser.add_argument("--variants", default=str(DATA_DIR / "variants_val_unseen_200.json"))
    parser.add_argument("--noise", type=float, default=0.3, help="position noise sigma in metres")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--radii", default="0.5,0.75,1.0,1.25,1.5,2.0")
    args = parser.parse_args()

    data = load_json(args.subgoals)
    subgoals = data["episodes"]
    variants = load_json(args.variants)
    gt = load_gt(data["split"])
    radii = [float(r) for r in args.radii.split(",")]

    print("== r_b calibration: noisy reference agent on ORIG (sigma={} m, {} repeats)".format(args.noise, args.repeats))
    chosen = radii[-1]
    for radius in radii:
        scores = []
        for repeat in range(args.repeats):
            rng = np.random.default_rng(repeat)
            orig, _, _ = fake_runs(subgoals, variants, gt, "reference", args.noise, rng)
            metrics, _ = evaluate(subgoals, orig, radius=radius, gt=gt)
            scores.append(metrics["SGCR"]["mean"])
        mean = float(np.mean(scores))
        flag = ""
        if mean >= 0.98 and chosen == radii[-1]:
            chosen = radius
            flag = "  <- smallest r_b with SGCR >= 0.98"
        print("r_b={:.2f}  SGCR={:.3f} (min {:.3f}){}".format(radius, mean, min(scores), flag))

    for agent in ("reference", "oracle"):
        rng = np.random.default_rng(0)
        orig, swap, drop = fake_runs(subgoals, variants, gt, agent, args.noise, rng)
        metrics, _ = evaluate(subgoals, orig, swap, drop, variants, radius=chosen, gt=gt)
        print("\n== {} agent, r_b={:.2f}, noise={} m".format(agent, chosen, args.noise))
        print(format_table(metrics))


if __name__ == "__main__":
    main()
