"""CA-Nav adapter: build its ORIG / FLIP / GOAL-ONLY inputs and import its outputs.

CA-Nav (Reproductions/CA-Nav-code) does not read the instruction text at run
time: it executes a GPT-4 parse (sub-instructions, state constraints,
decisions) stored per episode id in ``llm_reply_valunseen1839.json``. So a
variant needs both a dataset split with the new text and a matching reply file:

  ORIG      the 200-set, replies unchanged
  FLIP      reverse left/right in the matching parsed sub-instructions
  GOAL-ONLY keep the parsed sub-instructions overlapping the retained final chunks

    python -m benchmark.canav build            # writes into CA-Nav's data/datasets/benchmark/
    python -m benchmark.canav import --exp exp_bench_orig --out benchmark/results/canav/orig

``import`` converts ``traj_*.jsonl`` + ``stats_ep_ckpt_*.json`` of an experiment
into the runner's ``rank_0.log`` / ``rank_0_trace.jsonl`` layout that
``benchmark.metrics`` reads, and prints the original CA-Nav metrics.
"""
import argparse
import glob
import gzip
import json
import re
from pathlib import Path

from .common import DEFAULT_SET, DIRECTION, data_path, dump_json, load_json, write_runner_run

CANAV = Path("/data/pengyh/workspace/Reproductions/CA-Nav-code")
CANAV_DATASET = CANAV / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
CANAV_GT = CANAV / "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz"
CANAV_REPLIES = CANAV / "data/datasets/LLM_REPLYS_VAL_UNSEEN/llm_reply_valunseen1839.json"
CANAV_BENCH = CANAV / "data/datasets/benchmark"


def _norm_tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def locate(text, fragment, cursor=0):
    """Character span of a GPT sub-instruction inside the instruction.

    Sub-instructions come in reading order, so the search starts where the
    previous one ended; the span runs from the first matched token to the last,
    and short unmatched tokens are skipped.
    """
    lowered = text.lower()
    tokens = _norm_tokens(fragment)
    if not tokens:
        return None
    start = end = None
    for token in tokens:
        at = lowered.find(token, end if end is not None else cursor)
        if at < 0 or at > (end if end is not None else cursor) + 40:
            continue
        if start is None:
            start = at
        end = at + len(token)
    return None if start is None else (start, end)


def _walk(reply, text, span):
    """Walk the reply's sub-instructions once, reporting which ones cover ``span``.

    ``locate`` is cursor-driven, so the two rewriters below must advance the
    cursor identically or they would disagree about where a sub-instruction
    sits. Sharing the walk is what guarantees that.

    Yields (index, sub, located, covers_span); ``located`` is None when the
    sub-instruction could not be found in the text, and covers_span is then False.
    """
    cursor = 0
    for index, sub in enumerate(reply["sub-instructions"]):
        located = locate(text, sub, cursor)
        if located is None:
            yield index, sub, None, False
            continue
        cursor = located[1]
        overlap = max(0, min(located[1], span[1]) - max(located[0], span[0]))
        yield index, sub, located, overlap >= 0.5 * (located[1] - located[0])


def _renumbered(reply, keep):
    """Reply with only the sub-instructions in ``keep``, renumbered 0..len(keep)-1.

    state-constraints and decisions are keyed by the sub-instruction index as a
    string, so dropping one means rewriting every later key -- done in one place
    because getting it wrong silently misaligns constraints with instructions.
    """
    out = dict(reply)
    out["sub-instructions"] = [reply["sub-instructions"][i] for i in keep]
    out["state-constraints"] = {str(n): reply["state-constraints"][str(i)] for n, i in enumerate(keep)}
    out["decisions"] = {str(n): reply["decisions"][str(i)] for n, i in enumerate(keep)}
    return out


FLIP_WORD = DIRECTION


def _flip_words(text):
    return FLIP_WORD.sub(lambda m: {"left": "right", "right": "left"}[m.group(0).lower()], text)


def flip_reply(reply, text, span):
    """Flip left/right in the GPT sub-instructions overlapping ``span`` (text, constraints, decisions)."""
    out = json.loads(json.dumps(reply))
    flipped = []
    for index, sub, located, covers in _walk(reply, text, span):
        if covers and FLIP_WORD.search(sub):
            out["sub-instructions"][index] = _flip_words(sub)
            out["state-constraints"][str(index)] = [[c[0], _flip_words(c[1]) if c[0] == "direction constraint" else c[1]]
                                                    for c in reply["state-constraints"][str(index)]]
            decision = out["decisions"][str(index)]
            decision["directions"] = [_flip_words(d) for d in decision.get("directions", [])]
            flipped.append(index)
    return (out if flipped else None), flipped


def keep_last_reply(reply, text, last_span):
    """Keep only the sub-instructions overlapping the last FGR2R chunk; renumber."""
    keep = [index for index, sub, located, covers in _walk(reply, text, last_span) if covers]
    if not keep:
        keep = [len(reply["sub-instructions"]) - 1]
    if len(keep) == len(reply["sub-instructions"]):
        return None
    return _renumbered(reply, keep)


def write_canav_split(name, base, gt, texts):
    directory = CANAV_BENCH / name
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(base)
    data["episodes"] = []
    for episode in base["episodes"]:
        eid = str(episode["episode_id"])
        if eid in texts:
            episode = json.loads(json.dumps(episode))
            episode["instruction"]["instruction_text"] = texts[eid]
            data["episodes"].append(episode)
    with gzip.open(str(directory / "{}.json.gz".format(name)), "wt") as handle:
        json.dump(data, handle)
    with gzip.open(str(directory / "{}_gt.json.gz".format(name)), "wt") as handle:
        json.dump({eid: gt[eid] for eid in texts}, handle)
    return directory


def build(args):
    subgoals = load_json(args.subgoals)["episodes"]
    with gzip.open(str(CANAV_DATASET), "rt") as handle:
        base = json.load(handle)
    with gzip.open(str(CANAV_GT), "rt") as handle:
        gt = json.load(handle)
    replies = load_json(CANAV_REPLIES)

    orig_text = {eid: rec["instruction"] for eid, rec in subgoals.items()}
    orig_reply = {eid: replies[eid] for eid in subgoals}
    flip_text, flip_reply_table, flip_excluded = {}, {}, {}
    if args.flip:
        flips = load_json(args.flip)["flip"]
        for eid, v in flips.items():
            span = subgoals[eid]["subgoals"][v["k"] - 1]["span"]
            reply, flipped = flip_reply(replies[eid], subgoals[eid]["instruction"], span)
            if reply is None:
                flip_excluded[eid] = "no GPT sub-instruction with left/right maps onto chunk {}".format(v["k"])
                continue
            flip_text[eid] = v["instruction"]
            flip_reply_table[eid] = dict(reply, _flipped_sub_instructions=[replies[eid]["sub-instructions"][i] for i in flipped])
        print("flip: {} episodes, excluded {}".format(len(flip_text), len(flip_excluded)))
        dump_json(dict(excluded_from_flip=flip_excluded), CANAV_BENCH / "{}_flip_excluded.json".format(args.name))

    goal_text, goal_reply_table = {}, {}
    if args.goalonly:
        goals = load_json(args.goalonly)["goalonly"]
        for eid, v in goals.items():
            chunks = subgoals[eid]["subgoals"]
            last_span = (chunks[-v.get("kept_chunks", 1)]["span"][0], chunks[-1]["span"][1])
            reply = keep_last_reply(replies[eid], subgoals[eid]["instruction"], last_span)
            if reply is None:
                continue
            goal_text[eid] = v["instruction"]
            goal_reply_table[eid] = dict(reply, _kept_for_goal_only=True)
        print("goalonly: {} episodes (excluded {})".format(len(goal_text), len(goals) - len(goal_text)))

    for name, texts, table in (("orig", orig_text, orig_reply),
                               ("flip", flip_text, flip_reply_table), ("goalonly", goal_text, goal_reply_table)):
        if not texts:
            continue
        split = "{}_{}".format(args.name, name)
        directory = write_canav_split(split, base, gt, texts)
        dump_json(table, directory / "llm_reply.json")
        dump_json(sorted(texts, key=int), directory / "episode_ids.json")
        print("{}: {} episodes -> {}".format(split, len(texts), directory))


def import_run(args):
    exp = CANAV / "data/checkpoints" / args.exp
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for fn in glob.glob(str(exp / "stats_ep_ckpt_*.json")):
        stats.update(load_json(fn))
    seen = write_runner_run(glob.glob(str(exp / "traj_*.jsonl")), out)
    keys = ["success", "oracle_success", "spl", "distance_to_goal", "path_length", "steps_taken", "ndtw", "sdtw"]
    rows = [stats[e] for e in seen if e in stats]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in keys} if rows else {}
    dump_json(dict(exp=args.exp, n=len(rows), summary=summary), out / "canav_summary.json")
    print("imported {} episodes -> {}".format(len(seen), out))
    if rows:
        print("CA-Nav metrics  n={} SR={success:.3f} OSR={oracle_success:.3f} SPL={spl:.3f} NE={distance_to_goal:.2f} PL={path_length:.2f}m Steps={steps_taken:.1f} nDTW={ndtw:.3f} SDTW={sdtw:.3f}".format(len(rows), **summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--subgoals", default=str(data_path("subgoals")))
    b.add_argument("--name", default=DEFAULT_SET)
    b.add_argument("--flip", default=str(data_path("flip")), help="flip json from build_flip ('' to skip)")
    b.add_argument("--goalonly", default=str(data_path("goalonly")), help="goal-only json ('' to skip)")
    i = sub.add_parser("import")
    i.add_argument("--exp", required=True, help="CA-Nav experiment name under data/checkpoints/")
    i.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args) if args.cmd == "build" else import_run(args)


if __name__ == "__main__":
    main()
