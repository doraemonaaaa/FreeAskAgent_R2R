"""Recompute every benchmark run from raw traces with one scorer and write results/summary.json.

    python -m benchmark.summarize [--out benchmark/results/summary.json]

Per-run means (SR, SPL, nDTW, SGCR-eff, NE, PL, steps), paired agreement
(evaluate_paired + paired deltas of nDTW / SGCR-eff / PL with bootstrap CI) for
noise floors, GOAL-ONLY, PARAPHRASE (vs ORIG and A2-A4 vs A1), cross-system
agreement on ORIG, and FLIP (evaluate_flip) for every system with both runs.
REPORT.md numbers come from this file.
"""
import argparse
import json

import numpy as np

from .common import ROOT, data_path, load_gt, load_json, load_run, ndtw, path_walked
from .metrics import evaluate_flip, evaluate_paired, score_episode

R = ROOT / 'benchmark/results'
sub = load_json(data_path('subgoals'))
SG = sub['episodes']
GT = load_gt(sub['split'])
rng = np.random.default_rng(0)
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--out', default=str(R / 'summary.json'))
args = parser.parse_args()

RUNS = {
    'freeaskagent/orig': R / 'freeaskagent/orig', 'freeaskagent/orig2': R / 'freeaskagent/orig2', 'freeaskagent/goalonly': R / 'freeaskagent/goalonly',
    'canav/orig': R / 'canav/orig', 'canav/goalonly': R / 'canav/goalonly',
    'aware/orig': R / 'awarevln/orig', 'aware/goalonly': R / 'awarevln/goalonly',
}
for a in ('para_id', 'para_terse', 'para_natural', 'para_lm_shift'):
    RUNS['aware/' + a] = R / 'awarevln' / a
    RUNS['qwen/' + a] = R / 'canav_qwen' / a
RUNS['qwen/orig'] = R / 'canav_qwen/orig'
runs = {k: load_run(str(v)) for k, v in RUNS.items()}


def per_ep(run):
    pos, res = run
    out = {}
    for e, r in res.items():
        row = dict(sr=r['success'], spl=r['spl'], ne=r['dtg'], steps=r['steps'])
        if e in pos and len(pos[e]) > 1:
            row['pl'] = path_walked(pos[e])[-1]
            if e in GT:
                row['ndtw'] = ndtw(pos[e], GT[e])
            if e in SG:
                s = score_episode(pos[e], SG[e]['subgoals'], 1.5)
                row['sgcr_eff'] = s['c_eff'] / s['K']
                row['inter'] = sum(x is not None for x in s['entries'][:-1]) / max(len(s['entries']) - 1, 1)
        out[e] = row
    return out


P = {k: per_ep(v) for k, v in runs.items()}


def ci_mean(v):
    v = np.asarray(v, float)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(key, ids=None):
    p = P[key]
    ids = [e for e in p if ids is None or e in ids]
    out = dict(n=len(ids))
    for m in ('sr', 'spl', 'ndtw', 'sgcr_eff', 'ne', 'pl', 'steps'):
        v = [p[e][m] for e in ids if m in p[e]]
        out[m] = round(float(np.mean(v)), 3) if v else None
    return out


def delta(a, b, metric, ids=None):
    common = [e for e in P[a] if e in P[b] and metric in P[a][e] and metric in P[b][e] and (ids is None or e in ids)]
    d = [P[b][e][metric] - P[a][e][metric] for e in common]
    m, lo, hi = ci_mean(d)
    return dict(n=len(common), mean=round(m, 3), ci=[round(lo, 3), round(hi, 3)])


def paired(a, b, ids=None):
    m, _ = evaluate_paired(runs[a], runs[b], ids=ids)
    pick = lambda k: round(m[k]['mean'], 3)
    return dict(n=m['FlipRate']['n'], sr_a=pick('SR_orig'), sr_b=pick('SR_other'),
                both=int(m['Solved_both']['mean']), a_only=int(m['Solved_orig_only']['mean']),
                b_only=int(m['Solved_other_only']['mean']), flip=pick('FlipRate'),
                kept=pick('SuccessKept'), kappa=pick('Kappa'), kappa_ci=[round(x, 2) for x in m['Kappa']['ci']],
                mcnemar=round(m['McNemar_p']['mean'], 3), endgap=pick('EndpointGap_m'), dne=pick('|dNE|_m'),
                d_ndtw=delta(a, b, 'ndtw', ids), d_sgcr=delta(a, b, 'sgcr_eff', ids), d_pl=delta(a, b, 'pl', ids))


out = dict(summary={k: summary(k) for k in P})
pairs = {
    'noise:freeaskagent orig~orig2': ('freeaskagent/orig', 'freeaskagent/orig2'),
    'noise:canav gpt~qwen parse': ('canav/orig', 'qwen/orig'),
    'goal:freeaskagent': ('freeaskagent/orig', 'freeaskagent/goalonly'), 'goal:canav': ('canav/orig', 'canav/goalonly'), 'goal:aware': ('aware/orig', 'aware/goalonly'),
}
for sysk in ('aware', 'qwen'):
    for a in ('para_id', 'para_terse', 'para_natural', 'para_lm_shift'):
        pairs[f'para:{sysk} orig~{a}'] = (f'{sysk}/orig', f'{sysk}/{a}')
    for a in ('para_terse', 'para_natural', 'para_lm_shift'):
        pairs[f'para:{sysk} A1~{a}'] = (f'{sysk}/para_id', f'{sysk}/{a}')
out['paired'] = {k: paired(*v) for k, v in pairs.items()}

# cross-system agreement on ORIG (same 200 episodes)
cross = {}
for a, b in (('freeaskagent/orig', 'canav/orig'), ('freeaskagent/orig', 'aware/orig'), ('canav/orig', 'aware/orig'), ('qwen/orig', 'aware/orig')):
    cross[f'{a}~{b}'] = paired(a, b)
out['cross'] = cross
common = set(P['freeaskagent/orig']) & set(P['canav/orig']) & set(P['aware/orig'])
s = {k: {e for e in common if P[k][e]['sr'] > 0} for k in ('freeaskagent/orig', 'canav/orig', 'aware/orig')}
out['orig_union'] = dict(n=len(common), all3=len(s['freeaskagent/orig'] & s['canav/orig'] & s['aware/orig']),
                         any=len(s['freeaskagent/orig'] | s['canav/orig'] | s['aware/orig']),
                         freeaskagent_or_canav_not_aware=len((s['freeaskagent/orig'] | s['canav/orig']) - s['aware/orig']))


# FLIP (turn at the start, build_flip): results/<system>/flip_orig vs flip, once both runs exist.
flip_meta = load_json(data_path('flip', 'val_unseen'))
out['flip'] = {}
for system in ('freeaskagent', 'canav', 'awarevln'):
    a, b = R / system / 'flip_orig', R / system / 'flip'
    if not (a.is_dir() and b.is_dir()):
        continue
    fm, _ = evaluate_flip(load_run(str(a)), load_run(str(b)), flip_meta)
    out['flip'][system] = {k: dict(mean=round(v['mean'], 3), ci=[None if x is None else round(x, 3) for x in v['ci']], n=v['n'])
                           for k, v in fm.items()}
with open(args.out, 'w') as handle:
    json.dump(out, handle, indent=1)
print('wrote', args.out)
