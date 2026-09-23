"""Prepare PARAPHRASE inputs and queue sequential FP16 AwareVLN evaluation.

    python -m benchmark.run_paraphrase prepare
    python -m benchmark.run_paraphrase awarevln --wait-hours 24

CA-Nav inputs include parser tasks, NOT reused ORIG replies. Supply newly
parsed llm_reply.json for every arm before running its existing launcher.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import awarevln, canav
from .common import ROOT, data_path, dump_json, load_json

ARMS = ('para_id', 'para_terse', 'para_natural', 'para_lm_shift')
SET = 'val_unseen_200'
RUN = ROOT / 'outputs' / 'paraphrase_20260920'


def prepare():
    import ast
    import gzip

    meta_path = data_path('paraphrase')
    meta = load_json(meta_path)
    source_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = RUN / 'manifest.json'
    if manifest.exists() and load_json(manifest)['paraphrase_sha256'] != source_sha:
        raise RuntimeError('Source changed: use a new run directory to avoid mixing inputs')
    # Read only the prompt assignment; never execute the upstream API client/key code.
    tree = ast.parse((canav.CANAV / 'vlnce_baselines/common/instruction_tools.py').read_text())
    prompt_node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'prompt_template' for t in n.targets))
    prompt = eval(compile(ast.Expression(prompt_node), '<upstream prompt>', 'eval'), {'__builtins__': {}})
    for dataset, destination in ((awarevln.AWARE_DATA, 'awarevln'),
                                 (canav.CANAV_DATASET.parent.parent, 'canav')):
        with gzip.open(dataset / 'val_unseen/val_unseen.json.gz', 'rt') as handle:
            base = json.load(handle)
        with gzip.open(dataset / 'val_unseen/val_unseen_gt.json.gz', 'rt') as handle:
            gt = json.load(handle)
        for letter, arm in zip(('A1', 'A2', 'A3', 'A4'), ARMS):
            split = f'{SET}_{arm}'
            texts = {e: v[letter]['instruction'] for e, v in meta['episodes'].items()}
            if destination == 'canav':
                directory = canav.write_canav_split(split, base, gt, texts)
                dump_json(sorted(texts, key=int), directory / 'episode_ids.json')
                with (directory / 'parse_tasks.jsonl').open('w') as handle:
                    for e in sorted(texts, key=int):
                        handle.write(json.dumps({'episode_id': e, 'model': 'gpt-4',
                            'temperature': 0, 'prompt': prompt + '"""' + texts[e] + '"""'}) + '\n')
            else:
                directory = dataset / split
                directory.mkdir(parents=True, exist_ok=True)
                episodes = []
                for original in base['episodes']:
                    e = str(original['episode_id'])
                    if e in texts:
                        episode = json.loads(json.dumps(original))
                        episode['instruction']['instruction_text'] = texts[e]
                        episodes.append(episode)
                assert len(episodes) == len(texts) == 200
                with gzip.open(directory / f'{split}.json.gz', 'wt') as handle:
                    json.dump(dict(base, episodes=episodes), handle)
                with gzip.open(directory / f'{split}_gt.json.gz', 'wt') as handle:
                    json.dump({e: gt[e] for e in texts}, handle)
            print(destination, arm, len(texts), directory, flush=True)
    dump_json({'paraphrase_sha256': source_sha, 'arms': list(ARMS),
               'awarevln_precision': 'FP16', 'canav_parser_model': 'gpt-4',
               'data_note': 'A4 episodes 903/1455 have unresolved semantic review concerns; inputs unchanged.'}, manifest)


def status(**fields):
    dump_json(dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **fields), RUN / 'awarevln_status.json')
    print(fields, flush=True)


def run_aware(args):
    import fcntl
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'awarevln.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = load_json(RUN / 'manifest.json')
    assert manifest['paraphrase_sha256'] == hashlib.sha256(data_path('paraphrase').read_bytes()).hexdigest()
    python = '/data/pengyh/miniconda3/envs/awarevln-eval/bin/python'
    scoring_python = ROOT / '.venv/bin/python'
    episodes = load_json(data_path('paraphrase'))['episodes']
    expected = set(episodes)
    deadline = time.monotonic() + args.wait_hours * 3600
    for arm in ARMS:
        split = f'{SET}_{arm}'
        raw = RUN / 'awarevln_raw' / 'awarevln' / 'VLN-CE-v1' / split
        stats_file = raw / f'{split}_1-0.json'
        trace_file = raw / f'traj_{split}_1-0.jsonl'
        completed = {}
        if trace_file.exists():
            letter = ('A1', 'A2', 'A3', 'A4')[ARMS.index(arm)]
            for line in trace_file.read_text().splitlines():
                row = json.loads(line)
                eid = str(row['episode_id'])
                assert eid in expected and eid not in completed, 'Unexpected/duplicate episode'
                assert row['instruction'] == episodes[eid][letter]['instruction']
                assert all(k in row['metric'] for k in ('success', 'spl', 'distance_to_goal', 'ndtw'))
                completed[eid] = row['metric']
        remaining = sorted(expected - completed.keys(), key=int)
        if remaining:
            # The upstream evaluator skips any existing stats file, even partial ones.
            if stats_file.exists():
                raise RuntimeError(f'Incomplete stats file requires inspection: {stats_file}')
            while True:
                query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
                candidates = [(int(free.strip()), int(gpu.strip())) for gpu, free in
                              (line.split(',') for line in query.strip().splitlines())]
                free, gpu = max(candidates)
                if free >= args.min_free_mib:
                    break
                status(state='waiting_for_gpu', arm=arm, required_free_mib=args.min_free_mib,
                       best_free_mib=free, best_gpu=gpu)
                if time.monotonic() >= deadline:
                    status(state='wait_timeout', arm=arm)
                    return 2
                time.sleep(60)
            env = dict(os.environ, GPU=str(gpu), VARIANT=arm, PYTHON=python,
                       RESULTS_DIR=str(RUN / 'awarevln_raw'), HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
                       EPISODES=json.dumps(remaining))
            status(state='running', arm=arm, gpu=gpu, precision='FP16',
                   completed_before_resume=len(completed), remaining=len(remaining))
            with (RUN / f'awarevln_{arm}.log').open('a') as log:
                result = subprocess.run(['bash', 'scripts/eval/bench_local.sh'],
                                        cwd=awarevln.AWARE, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                status(state='failed', arm=arm, returncode=result.returncode)
                return result.returncode
            # The evaluator only writes this invocation's stats. Merge the prior
            # completed episode metrics from its durable trajectory journal.
            new_stats = load_json(stats_file)
            assert set(new_stats) == set(remaining), 'Incomplete resumed evaluation'
            dump_json(dict(completed, **new_stats), stats_file)
        elif not stats_file.exists():
            dump_json(completed, stats_file)
        if len(load_json(stats_file)) != 200:
            status(state='incomplete', arm=arm, stats=str(stats_file))
            return 3
        out = ROOT / 'benchmark/results/awarevln' / arm
        subprocess.run([str(scoring_python), '-m', 'benchmark.awarevln', 'import',
                        '--results', str(raw), '--out', str(out)], cwd=ROOT, check=True)
        subprocess.run([str(scoring_python), '-m', 'benchmark.metrics', '--orig', str(out),
                        '--json', str(out.parent / f'metrics_{arm}.json'),
                        '--csv', str(out / 'per_episode.csv')], cwd=ROOT, check=True)
        status(state='arm_complete', arm=arm)
    status(state='complete', episodes=800)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'awarevln'])
    parser.add_argument('--wait-hours', type=float, default=24)
    parser.add_argument('--min-free-mib', type=int, default=28000)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        sys.exit(run_aware(args))


if __name__ == '__main__':
    main()
