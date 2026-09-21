"""Calibrate the subgoal boundary radius with noisy reference-path trajectories.

    python -m benchmark.selfcheck [--noise 0.3] [--repeats 20]

Select the smallest tested radius with mean SGCR >= 0.98 on ORIG.
"""
import argparse

import numpy as np

from .common import data_path, load_gt, load_json, resample
from .metrics import evaluate


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


def reference_run(subgoals, gt, sigma, rng):
    positions = {e: noisy(resample(gt[e]), sigma, rng) for e in subgoals}
    results = {e: dict(success=1.0) for e in subgoals}
    return positions, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subgoals", default=str(data_path("subgoals")))
    parser.add_argument("--noise", type=float, default=0.3, help="position noise sigma in metres")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--radii", default="0.5,0.75,1.0,1.25,1.5,2.0")
    args = parser.parse_args()

    data = load_json(args.subgoals)
    subgoals = data["episodes"]
    gt = load_gt(data["split"])
    radii = [float(r) for r in args.radii.split(",")]

    print("== r_b calibration: noisy reference agent on ORIG (sigma={} m, {} repeats)".format(args.noise, args.repeats))
    chosen = radii[-1]
    for radius in radii:
        scores = []
        for repeat in range(args.repeats):
            rng = np.random.default_rng(repeat)
            orig = reference_run(subgoals, gt, args.noise, rng)
            metrics, _ = evaluate(subgoals, orig, radius=radius)
            scores.append(metrics["SGCR"]["mean"])
        mean = float(np.mean(scores))
        flag = ""
        if mean >= 0.98 and chosen == radii[-1]:
            chosen = radius
            flag = "  <- smallest r_b with SGCR >= 0.98"
        print("r_b={:.2f}  SGCR={:.3f} (min {:.3f}){}".format(radius, mean, min(scores), flag))


if __name__ == "__main__":
    main()
