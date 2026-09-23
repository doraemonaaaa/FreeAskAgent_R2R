"""CA-Nav with Qwen parsing: isolated ORIG + four PARAPHRASE runs.

Uses the upstream prompt and temperature=0, validates every reply before
navigation, caches successful replies with hashes, and resumes both stages.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from . import canav
from .common import ROOT, data_path, dump_json, load_json
from .run_paraphrase import ARMS

RUN = ROOT / 'outputs/paraphrase_20260920/canav_qwen'
SET = 'val_unseen_200_qwen'
VARIANTS = ('orig',) + ARMS


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    dump_json(value, tmp)
    tmp.replace(path)


def status(**fields):
    fields = dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), **fields)
    atomic_json(fields, RUN / 'status.json')
    atomic_json(fields, RUN.parent / 'canav_status.json')
    print(fields, flush=True)


def validate(reply):
    if not isinstance(reply, dict):
        raise ValueError('Reply must be an object')
    if not isinstance(reply.get('destination'), str) or not reply['destination'].strip():
        raise ValueError('Missing destination')
    subs = reply.get('sub-instructions')
    if not isinstance(subs, list) or not subs or not all(isinstance(s, str) and s.strip() for s in subs):
        raise ValueError('Invalid sub-instructions')
    keys = {str(i) for i in range(len(subs))}
    if set(reply.get('state-constraints', {})) != keys or set(reply.get('decisions', {})) != keys:
        raise ValueError('Non-contiguous constraint/decision indices')
    for i in keys:
        constraints = reply['state-constraints'][i]
        if not isinstance(constraints, list):
            raise ValueError('Invalid constraints list')
        for c in constraints:
            if not isinstance(c, list) or len(c) != 2 or c[0] not in ('location constraint', 'object constraint', 'direction constraint') or not isinstance(c[1], str) or not c[1]:
                raise ValueError('Invalid constraint')
            # Upstream supports forward/backward and aliases; unknown strings
            # use its existing ambiguous-direction fallback. Preserve parser output.
        decision = reply['decisions'][i]
        landmarks, directions = decision.get('landmarks'), decision.get('directions')
        if not isinstance(landmarks, list) or not isinstance(directions, list):
            raise ValueError('Invalid decision lists')
        for landmark in landmarks:
            if not isinstance(landmark, list) or len(landmark) != 2 or not isinstance(landmark[0], str) or not landmark[0] or not isinstance(landmark[1], str) or not landmark[1]:
                raise ValueError('Invalid landmark/action')
        # ZS_Evaluator_mp consumes landmarks and state-constraints, not this
        # descriptive field. Preserve e.g. 'turn around' instead of inventing a turn.
        if any(not isinstance(d, str) or not d.strip() for d in directions):
            raise ValueError('Invalid direction description')
    return reply


def prepare(args):
    meta = load_json(data_path('paraphrase'))
    original = load_json(data_path('subgoals'))['episodes']
    tree = ast.parse((canav.CANAV / 'vlnce_baselines/common/instruction_tools.py').read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt_template' for t in n.targets))
    prompt = eval(compile(ast.Expression(node), '<upstream prompt>', 'eval'), {'__builtins__': {}})
    tables = {'orig': {e: v['instruction'] for e, v in original.items()}}
    for letter, arm in zip(('A1', 'A2', 'A3', 'A4'), ARMS):
        tables[arm] = {e: v[letter]['instruction'] for e, v in meta['episodes'].items()}
    manifest = dict(model=args.model, base_url=args.base_url, temperature=0,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    inputs_sha256=hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest(),
                    parser='Qwen3-VL-8B-Instruct', variants=list(VARIANTS),
                    note='New Qwen ORIG control; A4 903/1455 review concerns remain in unchanged inputs.')
    path = RUN / 'manifest.json'
    if path.exists() and load_json(path) != manifest:
        raise ValueError('Run provenance changed; use a new run directory')
    atomic_json(manifest, path)
    with gzip.open(canav.CANAV_DATASET, 'rt') as f:
        base = json.load(f)
    with gzip.open(canav.CANAV_GT, 'rt') as f:
        gt = json.load(f)
    tasks = []
    for arm, texts in tables.items():
        directory = canav.write_canav_split(f'{SET}_{arm}', base, gt, texts)
        atomic_json(sorted(texts, key=int), directory / 'episode_ids.json')
        for eid, text in texts.items():
            tasks.append((arm, eid, prompt + '"""' + text + '"""'))
    return tables, tasks


def parse_one(args, task):
    arm, eid, prompt = task
    cache = RUN / 'replies' / arm / f'{eid}.json'
    if cache.exists():
        return arm, eid, validate(load_json(cache)['reply'])
    body = dict(model=args.model, temperature=0, max_tokens=3072,
                messages=[dict(role='user', content=prompt)])
    error = None
    attempts = []
    for attempt in range(3):
        result = None
        try:
            req = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as response:
                result = json.load(response)
            content = result['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            reply = validate(json.loads(content))
            if result['choices'][0].get('finish_reason') == 'length':
                raise ValueError('Truncated response')
            atomic_json(dict(reply=reply, raw=result, model=args.model,
                             correction_attempts=attempts), cache)
            return arm, eid, reply
        except Exception as exc:
            error = exc
            attempts.append(dict(attempt=attempt + 1, error=str(exc), raw=result))
            atomic_json(attempts, RUN / 'parse_attempts' / arm / f'{eid}.json')
            if result is not None:
                content = result['choices'][0]['message'].get('content') or ''
                body['messages'] = [dict(role='user', content=prompt),
                    dict(role='assistant', content=content),
                    dict(role='user', content=(
                        f'The response failed the original JSON format contract: {exc}. '
                        'Return the complete corrected JSON object, with no commentary. '
                        'Preserve the original instruction meaning; do not invent landmarks or directions. '
                        'decisions directions must be lists of strings (empty lists are allowed). '
                        'Constraint types are exactly '
                        'location constraint, object constraint, direction constraint; a direction '
                        'constraint value is a direction string; empty constraint lists are allowed. Both indexed objects must have '
                        'keys 0..N-1 for the N sub-instructions. Landmark actions are approach, '
                        'move away, approach then move away.'))]
            time.sleep(1)
    raise RuntimeError(f'{arm}/{eid}: {type(error).__name__}: {error}')


def run(args):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tables, tasks = prepare(args)
        status(state='waiting_for_qwen', model=args.model)
        deadline = time.monotonic() + 900
        needs_parsing = any(not (RUN / 'replies' / arm / f'{eid}.json').exists() for arm, eid, _ in tasks)
        while needs_parsing:
            try:
                with urllib.request.urlopen(args.base_url.rstrip('/') + '/models', timeout=3) as response:
                    models = json.load(response)
                assert args.model in {m['id'] for m in models['data']}
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('Qwen endpoint did not become ready in 15 minutes')
                time.sleep(10)
        atomic_json(load_json(RUN / 'manifest.json'), ROOT / 'benchmark/results/canav_qwen/manifest.json')
        replies = {arm: {} for arm in VARIANTS}
        status(state='parsing', completed=0, total=len(tasks), model=args.model)
        failures = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(parse_one, args, task) for task in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                try:
                    arm, eid, reply = future.result()
                    replies[arm][eid] = reply
                except Exception as exc:
                    failures.append(str(exc))
                if n % 20 == 0 or n == len(tasks):
                    status(state='parsing', completed=sum(len(v) for v in replies.values()),
                           processed=n, failed=len(failures), total=len(tasks), model=args.model)
        if failures:
            atomic_json(failures, RUN / 'parse_failures.json')
            raise RuntimeError(f'{len(failures)} parses failed; see parse_failures.json')
        for arm in VARIANTS:
            assert set(replies[arm]) == set(tables[arm])
            atomic_json(replies[arm], canav.CANAV_BENCH / f'{SET}_{arm}/llm_reply.json')
        status(state='parsed', completed=len(tasks), total=len(tasks))
        if args.server_pid:
            proc = Path('/proc') / str(args.server_pid)
            if proc.exists() and proc.stat().st_uid == os.getuid():
                cmd = (proc / 'cmdline').read_bytes()
                if b'vllm' in cmd and b'8302' in cmd:
                    os.kill(args.server_pid, signal.SIGTERM)
        python = '/data/pengyh/miniconda3/envs/CA-Nav/bin/python'
        scoring_python = str(ROOT / '.venv/bin/python')
        for arm in VARIANTS:
            exp = f'exp_bench_qwen_{arm}'
            status(state='running', arm=arm, gpus=args.gpus, parser_model=args.model)
            env = dict(os.environ, VARIANT=arm, SET=SET, GPUS=args.gpus,
                       NPROC=str(len(args.gpus.split(','))), EXP_NAME=exp, PYTHON=python,
                       PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with (RUN / f'{arm}.log').open('a') as log:
                result = subprocess.run(['bash', 'run_r2r/bench_local.sh'], cwd=canav.CANAV,
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'CA-Nav {arm} exited {result.returncode}; see {RUN / (arm + ".log")}')
            done = {}
            for path in (canav.CANAV / 'data/checkpoints' / exp).glob('stats_ep_ckpt_*.json'):
                done.update(load_json(path))
            if set(done) != set(tables[arm]):
                raise RuntimeError(f'{arm}: incomplete stats ({len(done)}/200); rerun to resume')
            out = ROOT / 'benchmark/results/canav_qwen' / arm
            subprocess.run([scoring_python, '-m', 'benchmark.canav', 'import', '--exp', exp,
                            '--out', str(out)], cwd=ROOT, check=True)
            subprocess.run([scoring_python, '-m', 'benchmark.metrics', '--orig', str(out),
                            '--json', str(out.parent / f'metrics_{arm}.json'),
                            '--csv', str(out / 'per_episode.csv')], cwd=ROOT, check=True)
            status(state='arm_complete', arm=arm, episodes=200)
        status(state='complete', episodes=1000, parser_model=args.model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8302/v1')
    p.add_argument('--model', default='qwen3-vl-8b')
    p.add_argument('--gpus', default='1,2,3')
    p.add_argument('--server-pid', type=int)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        status(state='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
