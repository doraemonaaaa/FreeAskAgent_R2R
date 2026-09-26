"""Shared loading helpers. Self-contained: only numpy (fastdtw optional)."""
import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
FGR2R_DIR = DATA_DIR / "fgr2r"
R2R_DIR = ROOT.parent / "habitat" / "data" / "datasets" / "vln" / "mp3d" / "r2r" / "v1"
# R2R-CE v1-3 val_unseen (CA-Nav's and AwareVLN's copies are byte-identical). Same
# episodes, positions and texts as v1-2 above, but different start rotations; only
# v1-3's agree with the instructions' first turn (see build_flip).
R2R_V13_VAL_UNSEEN = ROOT.parent / "Reproductions" / "CA-Nav-code" / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed" / "val_unseen" / "val_unseen.json.gz"
EVAL_SETS = ROOT / "integrations" / "v3" / "eval_sets"
# A final sub-instruction that names nothing ("and stop immediatly.", "Wait there.").
# build_goalonly keeps the preceding chunk for these; build_paraphrase skips its
# final-chunk landmark gate on them -- there is no landmark to preserve.
BARE_STOP = re.compile(r"^(?:and |then )?(?:stop|wait|stand|end|halt|walk forward)"
                       r"(?: there| here| right there| right here| immediatly| immediately)?\.?$", re.IGNORECASE)
SUCCESS_DISTANCE_M = 3.0
FORWARD_STEP_M = 0.25

# Everything derived for one evaluation set lives under data/<set>/, so adding a
# second set (the full val_unseen the FLIP suite wants) adds a directory instead
# of another dozen files in a flat folder.
DEFAULT_SET = "val_unseen_200"

# Inside a set, variants are filed by HOW THEY WERE MADE, because that decides how
# they may be read. A rule/ variant is a minimal pair -- one word flipped, one span
# deleted -- so it can be compared against ORIG
# directly. An llm/ variant re-words the whole instruction, so its arms may only be
# read against their own para_id control, never against ORIG. subgoals.json sits
# above both: it is the FGR2R-derived boundary set every variant and metric builds on.
FAMILY = {
    "flip": "rule", "goalonly": "rule",
    "paraphrase": "llm",
}
VARIANT_FAMILY = {
    "flip": "rule", "goalonly": "rule",
    "para_id": "llm", "para_terse": "llm", "para_natural": "llm", "para_lm_shift": "llm",
}


def variant_parser(doc):
    """argparse parser with the four options every variant builder takes.

    --subgoals is the boundary file the variant is cut from, --name the eval set,
    --split the underlying R2R-CE split, --no-splits stops before writing the
    habitat split (metadata only).
    """
    import argparse
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subgoals", default=str(data_path("subgoals")))
    parser.add_argument("--name", default=DEFAULT_SET)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--no-splits", action="store_true")
    return parser


def set_dir(name=DEFAULT_SET):
    return DATA_DIR / name


def data_path(kind, name=DEFAULT_SET):
    """data/<set>/subgoals.json, or data/<set>/{rule,llm}/<kind>.json for a variant."""
    if kind == "subgoals":
        return set_dir(name) / "subgoals.json"
    if kind not in FAMILY:
        raise KeyError("unknown data kind {!r}; known: subgoals, {}".format(kind, ", ".join(sorted(FAMILY))))
    return set_dir(name) / FAMILY[kind] / "{}.json".format(kind)


def ids_path(variant, name=DEFAULT_SET):
    """data/<set>/{rule,llm}/ids/<variant>.txt -- the EPISODE_IDS file a variant run takes."""
    if variant not in VARIANT_FAMILY:
        raise KeyError("unknown variant {!r}; known: {}".format(variant, ", ".join(sorted(VARIANT_FAMILY))))
    return set_dir(name) / VARIANT_FAMILY[variant] / "ids" / "{}.txt".format(variant)


def gen_dir(name=DEFAULT_SET):
    """data/<set>/llm/paraphrase_gen/ -- the rewrites paraphrase.json is built from."""
    return set_dir(name) / "llm" / "paraphrase_gen"

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


# ---------------------------------------------------------------- variant splits
# Optional per-step cost fields a runner may write as TOP-LEVEL keys of each trace line.
# None of the systems emits them yet; load_costs / metrics.evaluate_cost skip runs without them.
COST_FIELDS = ("tokens_in", "tokens_out", "model_calls", "step_time_s")


def _field(line, key):
    """Value of ``key`` in a raw trace line: the last occurrence, so a nested payload
    written before the top-level fields does not shadow them."""
    at = line.rfind('"{}": '.format(key))
    if at < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(line, at + len(key) + 4)[0]
    except ValueError:
        return None


def load_costs(out_dirs):
    """Runner output dirs -> {episode: {steps, tokens_in, tokens_out, model_calls, step_time_s}} (sums).

    Only steps that carry at least one cost field count; episodes without any are absent.
    """
    if isinstance(out_dirs, (str, Path)):
        out_dirs = [out_dirs]
    costs = {}
    for directory in out_dirs:
        for trace in sorted(glob.glob(str(Path(directory) / "rank_*_trace.jsonl"))):
            with open(trace, errors="replace") as handle:
                for line in handle:
                    values = {k: _field(line, k) for k in COST_FIELDS}
                    values = {k: float(v) for k, v in values.items() if isinstance(v, (int, float))}
                    episode = _field(line, "episode_id")
                    if not values or episode is None:
                        continue
                    row = costs.setdefault(str(episode), dict(steps=0))
                    row["steps"] += 1
                    for k, v in values.items():
                        row[k] = row.get(k, 0.0) + v
    return costs


def write_split(name, raw, episodes_by_id, gt, instruction_of):
    """Write ``<r2r data>/<name>/<name>.json.gz`` with new instruction texts, plus a gt copy.

    Every variant builder (flip, goalonly, paraphrase) emits its split
    this way. It lives here rather than in one of them so the others do not have
    to import a sibling builder just to write a file.
    """
    directory = R2R_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(raw)
    data["episodes"] = []
    for episode_id, text in instruction_of.items():
        episode = json.loads(json.dumps(episodes_by_id[episode_id]))
        episode["instruction"]["instruction_text"] = text
        data["episodes"].append(episode)
    with gzip.open(str(directory / "{}.json.gz".format(name)), "wt") as handle:
        json.dump(data, handle)
    with gzip.open(str(directory / "{}_gt.json.gz".format(name)), "wt") as handle:
        json.dump({eid: dict(locations=gt[eid]) for eid in instruction_of}, handle)
    return directory


def write_runner_run(traj_files, out_dir):
    """External ``traj_*.jsonl`` (positions / distances / metric per episode) -> runner format.

    CA-Nav and AwareVLN both dump one record per episode; the metric stack reads
    the runner's ``rank_*_trace.jsonl`` + ``rank_*.log`` instead, so both adapters
    converted them with the same twenty lines. Duplicate episodes (a rerun
    appended to the same file) keep their first occurrence.

    ``sdtw`` goes into the log line only when the source reports it -- load_run's
    result regex stops at ``dtg=``, so trailing fields are free-form.
    Returns the set of episode ids written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(out_dir / "rank_0_trace.jsonl", "w") as trace, open(out_dir / "rank_0.log", "w") as log:
        for path in sorted(traj_files):
            for line in open(path):
                record = json.loads(line)
                episode = record["episode_id"]
                if episode in seen:
                    continue
                seen.add(episode)
                positions, distances = record["positions"], record["distances"]
                for step in range(1, len(positions)):
                    # sort_keys: load_run parses the record from its "distance_to_goal_after"
                    # key onwards, exactly as the runner's own traces are laid out
                    trace.write(json.dumps(dict(
                        episode_id=episode, step=step - 1,
                        position_before=positions[step - 1], position_after=positions[step],
                        distance_to_goal_before=distances[step - 1],
                        distance_to_goal_after=distances[step]), sort_keys=True) + "\n")
                m = record["metric"]
                row = ("rank=0 id={} steps={} success={:.3f} spl={:.3f} dtg={:.2f} osr={:.0f}"
                       " path_length={:.2f} ndtw={:.3f}").format(
                    episode, int(m["steps_taken"]), m["success"], m["spl"], m["distance_to_goal"],
                    m["oracle_success"], m["path_length"], m["ndtw"])
                if "sdtw" in m:
                    row += " sdtw={:.3f}".format(m["sdtw"])
                log.write(row + "\n")
    return seen


# ---------------------------------------------------------------- geometry
DIRECTION = re.compile(r"\b(left|right)\b", re.IGNORECASE)


def signed_angle(a, b):
    """Degrees from 2-vector ``a`` to ``b``; negative is left, positive is right in habitat x/z."""
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])))


def path_walked(positions):
    """Cumulative 3D path length at every step."""
    out = [0.0]
    for a, b in zip(positions[:-1], positions[1:]):
        out.append(out[-1] + float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64))))
    return out


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
