"""Aggregate + failure-classify eval runs under outputs/experiments/ab/.

Usage:
  python bench/analyze_runs.py base200 panohop200 cwp200      # summaries
  python bench/analyze_runs.py --paired panohop200 cwp200     # + episode flips

Per run: SR / SPL / dtg / oracle (min dtg <= 3 m anywhere on the trajectory),
failure classes (early stop far, timeout far, wrong stop at goal, timeout
near goal), mean steps, per-category breakdown from the 200-set listing.
"""
import argparse
import glob
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
AB = os.path.join(ROOT, "outputs", "experiments", "ab")
EVAL_SET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "eval_sets", "val_unseen_200.txt")
SUCCESS_M = 3.0


def categories():
    cats = {}
    if os.path.exists(EVAL_SET):
        for line in open(EVAL_SET):
            if line.strip() and not line.startswith("#") and "," in line:
                i, c = line.strip().split(",")
                cats[i] = c
    return cats


def load(name):
    """Per-episode dict: succ, spl, steps, dtg, min (trajectory min dtg)."""
    eps = {}
    for f in glob.glob(os.path.join(AB, name, "rank_*.log")):
        cur = None
        for line in open(f, errors="replace"):
            m = re.match(r"episode=(\S+) instruction=", line)
            if m:
                cur = m.group(1)
                eps.setdefault(cur, {"min": 1e9})
            m = re.search(r"^ep=(\S+) .* dtg=([\d.]+)", line)
            if m and m.group(1) in eps:
                e = eps[m.group(1)]
                e["min"] = min(e["min"], float(m.group(2)))
            m = re.match(
                r"rank=\d+ \[\d+/\d+\] id=(\S+) steps=(\d+) "
                r"success=([\d.]+) spl=([\d.]+) dtg=([\d.]+)", line)
            if m:
                eps.setdefault(m.group(1), {"min": 1e9}).update(
                    steps=int(m.group(2)), succ=float(m.group(3)),
                    spl=float(m.group(4)), dtg=float(m.group(5)))
    return {k: v for k, v in eps.items() if "succ" in v}


def summarize(name, rows, cats, max_steps=150):
    n = len(rows)
    if not n:
        print(f"{name}: no episodes found under {os.path.join(AB, name)}")
        return
    vals = list(rows.values())
    fail = [e for e in vals if e["succ"] == 0]
    def frac(pred):
        return sum(1 for e in fail if pred(e))
    print(f"{name}: n={n} SR={sum(e['succ'] for e in vals)/n:.3f} "
          f"SPL={sum(e['spl'] for e in vals)/n:.3f} "
          f"dtg={sum(e['dtg'] for e in vals)/n:.2f} "
          f"oracle={sum(1 for e in vals if e['min'] <= SUCCESS_M)/n:.3f} "
          f"steps={statistics.mean(e['steps'] for e in vals):.0f}")
    print(f"  failures {len(fail)}: "
          f"early_stop_far={frac(lambda e: e['steps'] < max_steps and e['min'] > SUCCESS_M)} "
          f"timeout_far={frac(lambda e: e['steps'] >= max_steps and e['min'] > SUCCESS_M)} "
          f"wrong_stop@goal={frac(lambda e: e['steps'] < max_steps and e['min'] <= SUCCESS_M)} "
          f"timeout_near={frac(lambda e: e['steps'] >= max_steps and e['min'] <= SUCCESS_M)}")
    by = {}
    for k, v in rows.items():
        by.setdefault(cats.get(k, "?"), []).append(v)
    for c in sorted(by):
        v = by[c]
        print(f"  {c:14s} n={len(v):3d} SR={sum(x['succ'] for x in v)/len(v):.2f} "
              f"oracle={sum(1 for x in v if x['min'] <= SUCCESS_M)/len(v):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names under outputs/experiments/ab/")
    ap.add_argument("--paired", action="store_true",
                    help="also list episode success flips between the first two runs")
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()
    cats = categories()
    loaded = {}
    for name in args.runs:
        loaded[name] = load(name)
        summarize(name, loaded[name], cats, args.max_steps)
    if args.paired and len(args.runs) >= 2:
        a, b = (loaded[args.runs[0]], loaded[args.runs[1]])
        flips = [(k, a[k]["succ"], b[k]["succ"])
                 for k in a if k in b and a[k]["succ"] != b[k]["succ"]]
        won = [k for k, x, y in flips if y > x]
        lost = [k for k, x, y in flips if x > y]
        print(f"paired {args.runs[0]} -> {args.runs[1]}: +{len(won)} -{len(lost)}")
        print(f"  won:  {', '.join(sorted(won)) or '-'}")
        print(f"  lost: {', '.join(sorted(lost)) or '-'}")


if __name__ == "__main__":
    main()
