"""Summarize cwp_coverage points.jsonl: coverage by source and bucket."""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/pengyh/workspace/FreeAskAgent_R2R/outputs/experiments/cwp_coverage/points.jsonl"
rows = [json.loads(l) for l in open(path)]
print(f"points: {len(rows)}  episodes: {len({r['ep'] for r in rows})}")


def cov(sub, src):
    vals = [r[src]["hit"] for r in sub if r.get(src)]
    errs = [r[src]["min_err"] for r in sub if r.get(src) and r[src]["min_err"] is not None]
    ns = [r[src]["n"] for r in sub if r.get(src)]
    if not vals:
        return "  -"
    return "%3d%% (n=%4d, med|err|=%4.1f deg, cands=%.1f)" % (
        100 * sum(vals) / len(vals), len(vals), float(np.median(errs)), float(np.mean(ns)))


buckets = [("ALL", rows),
           ("straight", [r for r in rows if not r["turning"]]),
           ("turning", [r for r in rows if r["turning"]]),
           ("turning>60", [r for r in rows if abs(r["gt_rel"]) > 60]),
           ("behind>120", [r for r in rows if abs(r["gt_rel"]) > 120])]
for name, sub in buckets:
    if not sub:
        continue
    print(f"\n== {name}  ({len(sub)} pts, {100*len(sub)/len(rows):.0f}%)")
    for src in ("ring", "stale", "fo"):
        print(f"  {src:6s} {cov(sub, src)}")
    f45 = [r["front45"] for r in sub]
    print(f"  front45 ceiling: {100*sum(f45)/len(f45):3.0f}%")
    ent = [r["entropy"] for r in sub]
    top1 = [r["top1"] for r in sub]
    print(f"  ring entropy: mean {np.mean(ent):.3f} med {np.median(ent):.3f}   "
          f"top1: mean {np.mean(top1):.4f}")

print("\n== per category (ring / stale / fo hit%)")
for cat in sorted({r["cat"] for r in rows}):
    sub = [r for r in rows if r["cat"] == cat]
    line = f"  {cat:14s} n={len(sub):4d}  "
    for src in ("ring", "stale", "fo"):
        vals = [r[src]["hit"] for r in sub if r.get(src)]
        line += f"{src}={100*sum(vals)/max(1,len(vals)):3.0f}%  "
    print(line)

# entropy as a look-around trigger: does high entropy predict turning?
ent_t = [r["entropy"] for r in rows if r["turning"]]
ent_s = [r["entropy"] for r in rows if not r["turning"]]
if ent_t and ent_s:
    print(f"\n== entropy signal: turning med {np.median(ent_t):.3f} vs straight med {np.median(ent_s):.3f}")
    thr = float(np.median(rows and [r['entropy'] for r in rows]))
    hi = [r for r in rows if r["entropy"] >= thr]
    print(f"   trigger@median-entropy: fires {100*len(hi)/len(rows):.0f}% of pts, "
          f"catches {100*sum(r['turning'] for r in hi)/max(1,sum(r['turning'] for r in rows)):.0f}% of turning pts")
