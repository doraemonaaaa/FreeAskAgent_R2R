"""FGR2R sub-instruction chunks -> per-episode subgoal boundaries.

    python -m benchmark.build_subgoals --ids integrations/v3/eval_sets/val_unseen_200.txt \
        --out benchmark/data/subgoals_val_unseen_200.json

For every episode: K subgoals, each with its instruction text span, the FGR2R
node range, the boundary point B_k (the end viewpoint of the chunk, start +
reference_path + goal indexing) and the arc length of that boundary along the
dense reference path. Consecutive chunks that end on the same viewpoint (FGR2R
often splits "... and stop" into its own zero-length chunk) are merged, so
boundaries are strictly increasing along the path.
"""
import argparse
import ast
import re

import numpy as np

from .common import DATA_DIR, EVAL_SETS, dist_xz, dump_json, episode_nodes, load_episodes, load_gt, load_json


def _as_list(value):
    return value if isinstance(value, list) else ast.literal_eval(value)


def _norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_fgr2r(split="val_unseen"):
    return {int(d["path_id"]): d for d in load_json(DATA_DIR / "FGR2R_{}.json".format(split))}


def match_instruction(record, text):
    """Index of ``text`` among the trajectory's three FGR2R instructions."""
    wanted = _norm(text)
    for index, candidate in enumerate(_as_list(record["instructions"])):
        if _norm(candidate) == wanted:
            return index
    raise KeyError("instruction text not found in FGR2R record")


def token_spans(text, chunks):
    """Character spans of each token chunk inside the original instruction text.

    FGR2R tokens are lower-cased and lemmatised ("are standing" -> "be stand"),
    so a token matches when the text contains it, or its first three letters,
    within a short window after the previous match; unmatched tokens are
    skipped. Returns None when a whole chunk finds no token at all (then the
    caller joins the tokens instead).
    """
    spans = []
    cursor = 0
    lowered = text.lower()
    for tokens in chunks:
        start = None
        end = cursor
        for token in tokens:
            token = token.lower()
            at = -1
            for needle in (token, token[:3]):
                if len(needle) < 2:
                    continue
                found = lowered.find(needle, end)
                if 0 <= found <= end + 40:
                    at = found
                    break
            if at < 0:
                continue
            if start is None:
                start = at
            end = at + len(token) if lowered.startswith(token, at) else at + 3
            while end < len(text) and text[end].isalpha():
                end += 1
        if start is None:
            return None
        while end < len(text) and text[end] in " .,;!?\r\n":
            end += 1
        spans.append([start, end])
        cursor = end
    if spans:
        spans[0][0] = 0
        spans[-1][1] = len(text)
    return spans


def arc_lengths(locations):
    pts = np.asarray(locations, dtype=np.float64)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def nearest_arc(locations, cumulative, point):
    pts = np.asarray(locations, dtype=np.float64)
    d = np.linalg.norm(pts[:, [0, 2]] - np.asarray([point[0], point[2]]), axis=1)
    index = int(np.argmin(d))
    return float(cumulative[index]), index


def build_episode(episode, record, gt_locations):
    text = episode["instruction"]["instruction_text"]
    which = match_instruction(record, text)
    chunks = _as_list(record["new_instructions"])[which]
    view = record["chunk_view"][which]
    nodes = episode_nodes(episode)
    assert len(nodes) == len(record["path"]), "viewpoint count mismatch"
    spans = token_spans(text, chunks)
    cumulative = arc_lengths(gt_locations)

    raw = []
    for k, ((node_start, node_end), tokens) in enumerate(zip(view, chunks)):
        raw.append(dict(
            text=text[spans[k][0]:spans[k][1]].strip() if spans else " ".join(tokens),
            span=spans[k] if spans else None,
            node_start=int(node_start), node_end=int(node_end),
        ))
    # merge chunks that end on the same viewpoint as the previous chunk
    merged = []
    for item in raw:
        if merged and item["node_end"] == merged[-1]["node_end"]:
            prev = merged[-1]
            prev["text"] = (prev["text"] + " " + item["text"]).strip()
            prev["span"] = [prev["span"][0], item["span"][1]] if prev["span"] and item["span"] else None
            prev["merged_from"].append(len(merged) + len(prev["merged_from"]))
            continue
        merged.append(dict(item, merged_from=[]))
    subgoals = []
    for k, item in enumerate(merged, start=1):
        boundary = nodes[item["node_end"] - 1]
        arc, gt_index = nearest_arc(gt_locations, cumulative, boundary)
        subgoals.append(dict(
            index=k, text=item["text"], span=item["span"],
            node_start=item["node_start"], node_end=item["node_end"],
            boundary_xyz=[float(v) for v in boundary],
            arc_end_m=arc, gt_index=gt_index,
            merged_chunks=1 + len(item["merged_from"]),
        ))
    return dict(
        episode_id=str(episode["episode_id"]), trajectory_id=int(episode["trajectory_id"]),
        scene_id=episode["scene_id"], instruction=text, K=len(subgoals),
        path_length_m=float(cumulative[-1]), nodes=nodes, subgoals=subgoals,
        fgr2r_instruction_index=which, span_aligned=spans is not None,
    )


def build(ids, split="val_unseen"):
    episodes, _ = load_episodes(split)
    gt = load_gt(split)
    fgr2r = load_fgr2r(split)
    out, skipped = {}, []
    for episode_id in ids:
        episode = episodes[episode_id]
        record = fgr2r.get(int(episode["trajectory_id"]))
        if record is None:
            skipped.append(episode_id)
            continue
        out[episode_id] = build_episode(episode, record, gt[episode_id])
    return out, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default=str(EVAL_SETS / "val_unseen_200.txt"))
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--out", default=str(DATA_DIR / "subgoals_val_unseen_200.json"))
    args = parser.parse_args()
    from .common import read_id_list
    ids = read_id_list(args.ids)
    subgoals, skipped = build(ids, args.split)
    dump_json(dict(split=args.split, ids_file=args.ids, episodes=subgoals, skipped=skipped), args.out)
    ks = [v["K"] for v in subgoals.values()]
    gaps = [b["arc_end_m"] - a["arc_end_m"] for v in subgoals.values() for a, b in zip(v["subgoals"][:-1], v["subgoals"][1:])]
    print("episodes={} skipped={} K mean={:.2f} min={} max={} unaligned_text={} min_boundary_gap={:.2f}m boundaries<1m={}".format(
        len(subgoals), len(skipped), sum(ks) / len(ks), min(ks), max(ks),
        sum(not v["span_aligned"] for v in subgoals.values()), min(gaps), sum(g < 1.0 for g in gaps)))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
