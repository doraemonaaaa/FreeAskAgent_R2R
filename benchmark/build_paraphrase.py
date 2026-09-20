"""PARAPHRASE suite: the same route, re-worded, with the FGR2R segmentation kept.

    python -m benchmark.build_paraphrase export            # generation tasks -> data/<set>/paraphrase_gen/tasks.jsonl
    python -m benchmark.build_paraphrase ingest            # data/<set>/paraphrase_gen/*.json -> metadata + habitat splits

Rewriting is chunk-wise on purpose: each sub-instruction is rewritten on its
own and the K chunks are re-joined, so K, the chunk -> viewpoint alignment and
every boundary B_k stay exactly as ``build_subgoals`` computed them. The whole
existing metric stack (SGCR, SGCR-eff, per-hop survival, GOAL-ONLY's
intermediate-boundary rate) therefore applies unchanged, and the generator only
ever sees one short sentence at a time.

Arms (all on the same 200 episodes; start / goal / reference path untouched):
  A1 PARA-ID   meaning-preserving paraphrase, same register.  The CONTROL: its
               SR drop is the confound floor of the rewrite pipeline itself,
               and only the extra drop of A2..A4 relative to A1 is attributable
               to the manipulation.  It deliberately absorbs the punctuation /
               whitespace normalisation that re-joining introduces.
  A2 TERSE     clipped imperatives, the way someone commands a robot.
  A3 NATURAL   chatty, hedged, filler words, the way a non-expert speaks.
  A4 LM-SHIFT  the same physical referents, described WITHOUT their original
               head noun ("the sofa" -> "the large upholstered seat").  Naming
               different objects would need scene knowledge the generator does
               not have; re-describing the same one is checkable (the original
               noun must be gone) and still separates "matches a landmark word"
               from "understands what is being pointed at".

Gates in ``ingest`` (see ``check_episode``): chunk count, non-empty text, and
an exact match on the sequence of left/right words -- a generator that quietly
drops or flips a direction word would silently turn every arm into a FLIP run.
New landmark nouns are reported (expected for A4, suspicious for A1..A3).
"""
import argparse
import difflib
import json
import re
from pathlib import Path

from .common import BARE_STOP, DIRECTION, DEFAULT_SET, write_split, data_path, gen_dir, ids_path, dump_json, load_episodes, load_gt, load_json

ARMS = ("A1", "A2", "A3", "A4")
ARM_LABEL = {"A1": "para_id", "A2": "para_terse", "A3": "para_natural", "A4": "para_lm_shift"}
GEN_DIR = gen_dir()

END_PUNCT = ".!?"
# Words that carry no landmark information, so they never count as a "new noun".
# The gate exists to catch a HALLUCINATED LANDMARK; every function word, motion
# verb and discourse marker left in here would just bury that signal in noise.
STOP = set("""a an the this that these those and or but then so if when while as at by for from in into of off on onto out over
past through to toward towards under up down upon with without after before again around across along behind below beneath
beside between beyond during near next until you your yours i it its is are was were be been being am do does did doing have
has had will would shall should can could may might must go goes going went gone come comes coming walk walks walking move
moves moving turn turns turning head heads heading continue continues continuing proceed carry carries stop stops stopping
halt halts wait waits waiting enter enters entering exit exits exiting leave leaves leaving take takes taking keep keeps
keeping make makes making stand stands standing sit sits sitting face faces facing follow follows following pass passes
passing cross crosses reach reaches get gets got find finds see sees look looks looking veer veers step steps travel begin
begins start starts end ends lead leads leading open opens opened block blocks blocked mark marks marked contain contains
containing straight forward ahead back backward front side sides one two three four five first second third fourth fifth
last final there here now once just about all any some more most other another same very much pretty basically actually
really quite bit way ways point thing things okay alright well yeah like directly immediately slightly once still even also
where which who what how your yourself themselves via toward onto out
want wants need needs going gonna far short put puts able let lets sure going""".split())
WORD = re.compile(r"[a-z]+")


def _stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def nouns(text):
    """Crude content-word set: lower-cased alphabetic tokens that are not function words."""
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def retains_landmark(before, after):
    """Does ``after`` still name something ``before`` named?

    Fuzzy on purpose: the originals contain typos the rewrites normalise
    ("dooryway" -> "doorway", "immediatly" -> "immediately", "stair case" ->
    "staircase"), and those are spelling changes, not a lost landmark. A pronoun
    standing in for the landmark ("the bedroom" -> "there") matches nothing and
    is still caught.
    """
    want, got = nouns(before), nouns(after)
    if not want:
        return True
    for w in want:
        for g in got:
            if difflib.SequenceMatcher(None, w, g).ratio() >= 0.8:
                return True
    return False


def new_nouns(after, reference):
    """Content words in ``after`` that the episode's original instruction never mentions.

    ``reference`` is the WHOLE original instruction, not just the matching chunk:
    a chatty rewrite legitimately refers back to the previous chunk's landmark
    ("from the rope, carry on to ..."), and that is not a hallucination. Matches
    are also tried against the space-stripped original so that a compound
    ("hallway") is not flagged when the original writes it apart ("hall way"),
    and against a crude stem so plurals and -ing forms do not count as new.
    """
    known = nouns(reference)
    known |= {_stem(w) for w in known}
    squashed = re.sub(r"[^a-z]", "", reference.lower())
    out = set()
    for word in nouns(after):
        if word in known or _stem(word) in known or word in squashed:
            continue
        out.add(word)
    return out


def directions(text):
    return [m.group(0).lower() for m in DIRECTION.finditer(text)]


def join_chunks(chunks):
    """Re-join rewritten chunks into one instruction, and report each chunk's span.

    FGR2R splits mid-sentence ("Go through the living room" | "and turn left
    ..."), and the original instruction joins those with a space and no full
    stop. Terminating every chunk would manufacture "living room. and turn
    left", which is broken English in all four arms and pure added noise, so a
    chunk is only closed off when the next one opens a new sentence.
    """
    pieces = [c.strip() for c in chunks]
    parts, spans, cursor = [], [], 0
    for index, piece in enumerate(pieces):
        nxt = pieces[index + 1] if index + 1 < len(pieces) else None
        continues = bool(nxt) and (nxt[0].islower() or nxt[0] in ",;")
        if piece and not continues:
            piece = piece.rstrip(",;") or piece
            if piece[-1] not in END_PUNCT:
                piece += "."
        if parts:
            cursor += 1  # the single space we join with
        spans.append([cursor, cursor + len(piece)])
        cursor += len(piece)
        parts.append(piece)
    return " ".join(parts), spans


# ---------------------------------------------------------------- export
def export(subgoals, out):
    with open(out, "w") as handle:
        for episode_id, record in subgoals.items():
            handle.write(json.dumps(dict(
                episode_id=episode_id, K=record["K"], instruction=record["instruction"],
                chunks=[s["text"] for s in record["subgoals"]],
                directions=[directions(s["text"]) for s in record["subgoals"]],
            )) + "\n")
    return len(subgoals)


# ---------------------------------------------------------------- ingest
def check_episode(record, arm, chunks):
    """Hard errors (reject) and soft notes (report) for one episode / arm."""
    errors, notes = [], []
    original = [s["text"] for s in record["subgoals"]]
    reference = record["instruction"]
    if len(chunks) != record["K"]:
        return ["chunk count {} != K {}".format(len(chunks), record["K"])], notes
    for k, (before, after) in enumerate(zip(original, chunks), start=1):
        after = (after or "").strip()
        if not after:
            errors.append("chunk {} empty".format(k))
            continue
        if directions(before) != directions(after):
            errors.append("chunk {}: direction words {} -> {}".format(k, directions(before), directions(after)))
        if arm == "A4":
            # the point of this arm is to drop the original head nouns
            kept = nouns(after) & nouns(before)
            if kept == nouns(before) and nouns(before):
                notes.append("chunk {}: A4 reuses every original noun {}".format(k, sorted(kept)))
        else:
            added = new_nouns(after, reference)
            if added:
                notes.append("chunk {}: new nouns {}".format(k, sorted(added)))
    # The last sub-instruction carries the stop condition, so replacing its landmark
    # with a pronoun ("Stop in the bedroom." -> "Halt there.") quietly removes the
    # destination from the arm. Coreference makes it recoverable, but the arm is then
    # weaker than ORIG for a reason that has nothing to do with the manipulation.
    if arm != "A4" and not BARE_STOP.match(original[-1].strip()) \
            and not retains_landmark(original[-1], chunks[-1].strip()):
        errors.append("final chunk keeps no landmark from {!r}".format(original[-1]))
    # An arm byte-identical to ORIG is a no-op for that episode: it carries none of
    # the rewrite confound A1 exists to measure, and none of the manipulation the
    # other arms exist to measure.
    if [c.strip() for c in chunks] == [c.strip() for c in original]:
        errors.append("identical to the original instruction")
    if arm == "A2" and sum(map(len, chunks)) > sum(map(len, original)):
        notes.append("TERSE arm is longer than the original")
    if arm == "A3" and sum(map(len, chunks)) < sum(map(len, original)):
        notes.append("NATURAL arm is shorter than the original")
    return errors, notes


def ingest(subgoals, generated):
    out, problems = {}, []
    for episode_id, record in subgoals.items():
        entry = generated.get(episode_id)
        if entry is None:
            problems.append((episode_id, "-", ["missing from generation"]))
            continue
        arms = {}
        for arm in ARMS:
            chunks = entry.get(arm)
            if not chunks:
                problems.append((episode_id, arm, ["missing arm"]))
                continue
            errors, notes = check_episode(record, arm, chunks)
            if errors:
                problems.append((episode_id, arm, errors))
                continue
            if notes:
                problems.append((episode_id, arm, ["NOTE " + n for n in notes]))
            text, spans = join_chunks(chunks)
            arms[arm] = dict(instruction=text, chunks=[c.strip() for c in chunks], spans=spans)
        if len(arms) == len(ARMS):
            out[episode_id] = arms
    return out, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "ingest"])
    parser.add_argument("--subgoals", default=str(data_path("subgoals")))
    parser.add_argument("--name", default=DEFAULT_SET)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--tasks", default=str(gen_dir() / "tasks.jsonl"))
    parser.add_argument("--gen", default=str(GEN_DIR))
    parser.add_argument("--no-splits", action="store_true")
    args = parser.parse_args()

    subgoals = load_json(args.subgoals)["episodes"]
    if args.action == "export":
        n = export(subgoals, args.tasks)
        print("wrote {} tasks ({} chunks) to {}".format(
            n, sum(v["K"] for v in subgoals.values()), args.tasks))
        return

    generated, generators = {}, []
    files = sorted(Path(args.gen).glob("*.json"))
    for path in files:
        payload = load_json(path)
        generated.update(payload.get("episodes", payload))
        who = payload.get("generator")
        if who and who not in generators:
            generators.append(who)
    print("loaded {} generation files, {} episodes, generator(s): {}".format(
        len(files), len(generated), ", ".join(generators) or "UNRECORDED"))

    out, problems = ingest(subgoals, generated)
    hard = [p for p in problems if not p[2][0].startswith("NOTE")]
    soft = [p for p in problems if p[2][0].startswith("NOTE")]
    print("accepted {}/{} episodes; {} hard rejects, {} notes".format(len(out), len(subgoals), len(hard), len(soft)))
    for episode_id, arm, messages in hard[:40]:
        print("  REJECT {} {}: {}".format(episode_id, arm, "; ".join(messages)))
    for episode_id, arm, messages in soft[:20]:
        print("  note   {} {}: {}".format(episode_id, arm, "; ".join(messages)))
    if len(soft) > 20:
        print("  ... {} more notes".format(len(soft) - 20))

    meta = dict(name=args.name, split=args.split,
                provenance="LLM rewrite, chunk-wise; NOT a minimal pair -- read A1 (para_id) as the confound floor",
                generator=generators or None, arms=ARM_LABEL,
                splits={arm: "{}_{}".format(args.name, ARM_LABEL[arm]) for arm in ARMS},
                n_hard_rejects=len(hard), episodes=out)
    path = data_path("paraphrase", args.name)
    dump_json(meta, path)
    print("wrote", path)
    lengths = {arm: sum(len(v[arm]["instruction"]) for v in out.values()) for arm in ARMS}
    original = sum(len(subgoals[e]["instruction"]) for e in out)
    print("mean instruction chars: ORIG {:.0f} | {}".format(
        original / max(len(out), 1),
        " | ".join("{} {:.0f}".format(a, lengths[a] / max(len(out), 1)) for a in ARMS)))

    if not args.no_splits:
        episodes, raw = load_episodes(args.split)
        gt = load_gt(args.split)
        for arm in ARMS:
            name = meta["splits"][arm]
            directory = write_split(name, raw, episodes, gt, {e: v[arm]["instruction"] for e, v in out.items()})
            ids = ids_path(ARM_LABEL[arm], args.name)
            ids.write_text("# episode ids present in split {}\n".format(name) + "".join(e + "\n" for e in out))
            print("wrote split", directory, "ids", ids)


if __name__ == "__main__":
    main()
