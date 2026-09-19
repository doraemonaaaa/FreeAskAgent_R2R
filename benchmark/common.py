"""Shared loading helpers. Self-contained: only numpy (fastdtw optional)."""
import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
R2R_DIR = ROOT.parent / "habitat" / "data" / "datasets" / "vln" / "mp3d" / "r2r" / "v1"
EVAL_SETS = ROOT / "integrations" / "v3" / "eval_sets"
SUCCESS_DISTANCE_M = 3.0
FORWARD_STEP_M = 0.25

try:
    from fastdtw import fastdtw as _fastdtw
except ImportError:
    _fastdtw = None


# ---------------------------------------------------------------- datasets
def split_dir(split):
    return R2R_DIR / split


def load_episodes(split="val_unseen"):
    """episode_id (str) -> raw R2R-CE episode dict."""
    with gzip.open(str(split_dir(split) / "{}.json.gz".format(split)), "rt") as handle:
        data = json.load(handle)
    return {str(ep["episode_id"]): ep for ep in data["episodes"]}, data


def load_gt(split="val_unseen"):
    """episode_id -> dense reference locations (shortest-path follower positions)."""
    with gzip.open(str(split_dir(split) / "{}_gt.json.gz".format(split)), "rt") as handle:
        data = json.load(handle)
    return {str(key): value["locations"] for key, value in data.items()}


def episode_nodes(episode):
    """Full viewpoint sequence of an R2R-CE episode: start + reference_path + goal.

    R2R-CE stores the trajectory's interior viewpoints in ``reference_path`` and
    the two end viewpoints as ``start_position`` / ``goals[0].position``, so the
    concatenation lines up 1:1 with FGR2R's ``path`` (verified on val_unseen).
    """
    return [list(episode["start_position"])] + [list(p) for p in episode["reference_path"]] + [list(episode["goals"][0]["position"])]


def read_id_list(path):
    """Eval-set file: ``#`` comments, ``id[,category]`` rows -> list of id strings."""
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            items.append(line.split(",", 1)[0].strip())
    return items


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=1)


# ---------------------------------------------------------------- run outputs
RESULT = re.compile(r" id=(\S+) steps=(\d+) success=([\d.]+) spl=([\d.]+) dtg=([\d.]+)")


def load_run(out_dirs):
    """Runner output dirs -> (positions, results).

    positions[episode] = [start, position after step 0, position after step 1, ...]
    results[episode]   = dict(steps, success, spl, dtg)
    Trace lines are the runner's ``rank_*_trace.jsonl`` records; only the
    position / distance tail is needed so a truncated decision payload never
    breaks parsing.
    """
    if isinstance(out_dirs, (str, Path)):
        out_dirs = [out_dirs]
    positions, results = {}, {}
    key = '"distance_to_goal_after": '
    for directory in out_dirs:
        for log in sorted(glob.glob(str(Path(directory) / "rank_*.log"))):
            with open(log, errors="replace") as handle:
                for line in handle:
                    m = RESULT.search(line)
                    if m:
                        results[m.group(1)] = dict(steps=int(m.group(2)), success=float(m.group(3)),
                                                   spl=float(m.group(4)), dtg=float(m.group(5)))
        for trace in sorted(glob.glob(str(Path(directory) / "rank_*_trace.jsonl"))):
            with open(trace, errors="replace") as handle:
                for line in handle:
                    at = line.rfind(key)
                    if at < 0:
                        continue
                    try:
                        tail = json.loads("{" + line[at:])
                    except ValueError:
                        continue
                    episode = str(tail["episode_id"])
                    seq = positions.setdefault(episode, [])
                    if not seq:
                        seq.append([float(v) for v in tail["position_before"]])
                    seq.append([float(v) for v in tail["position_after"]])
    return positions, results


# ---------------------------------------------------------------- geometry
def xz(point):
    return np.asarray([point[0], point[2]], dtype=np.float64)


def dist_xz(a, b):
    return float(np.linalg.norm(xz(a) - xz(b)))


def dedup(positions):
    out = []
    for p in positions:
        p = [float(v) for v in p]
        if not out or p != out[-1]:
            out.append(p)
    return out


def _euclid(a, b):
    return float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)))


def dtw_distance(path, reference):
    if _fastdtw is not None:
        return float(_fastdtw(path, reference, dist=_euclid)[0])
    a = np.asarray(path, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    total = np.full((len(a) + 1, len(b) + 1), np.inf)
    total[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j], total[i, j - 1], total[i - 1, j - 1])
    return float(total[-1, -1])


def ndtw(positions, reference):
    """VLN-CE nDTW of an agent position sequence against a dense reference path."""
    return float(np.exp(-dtw_distance(dedup(positions), reference) / (len(reference) * SUCCESS_DISTANCE_M)))


def resample(points, step=FORWARD_STEP_M):
    """Walk a polyline in fixed steps (fake-agent trajectories)."""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    out = [pts[0]]
    carry = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length == 0:
            continue
        direction = seg / length
        pos = carry
        while pos + step <= length + 1e-9:
            pos += step
            out.append(a + direction * pos)
        carry = pos - length
    if np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1])
    return [list(map(float, p)) for p in out]
