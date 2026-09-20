"""Join the blind reviewers' verdicts with the key and score them.

The reviewers never saw which arm an item came from, nor that 90 of the 890
items were controls. The controls come first: a reviewer that misses a
left/right flip cannot be believed when it says the real rewrites are fine.
"""
import json
import collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
key = json.load(open(HERE / "key.json"))

verdicts = {}
missing_files = []
for shard in range(6):
    path = HERE / "out_{}.json".format(shard)
    if not path.exists():
        missing_files.append(path.name)
        continue
    for row in json.load(open(path)):
        verdicts[row["id"]] = row

print("reviewed {}/{} items{}".format(
    len(verdicts), len(key), "  (missing: %s)" % ", ".join(missing_files) if missing_files else ""))

# ---------------------------------------------------------------- controls
print("\n" + "=" * 78)
print("1. Reviewer reliability, measured on the 90 hidden controls")
print("=" * 78)
EXPECT = {"ctrl_same": "same", "ctrl_other": "different", "ctrl_flip": "different"}
LABEL = {"ctrl_same": "identical text (expect 'same')",
         "ctrl_other": "another episode's route (expect 'different')",
         "ctrl_flip": "every left/right reversed (expect 'different')"}
control_rows = collections.defaultdict(list)
for item, (kind, episode, arm) in key.items():
    if kind.startswith("ctrl") and item in verdicts:
        control_rows[kind].append(verdicts[item])

for kind in ("ctrl_same", "ctrl_other", "ctrl_flip"):
    rows = control_rows[kind]
    if not rows:
        continue
    hit = sum(r["verdict"] == EXPECT[kind] for r in rows)
    line = "  {:<46} {:>2}/{:<3} = {:.2f}".format(LABEL[kind], hit, len(rows), hit / len(rows))
    if kind == "ctrl_flip":
        caught = sum(bool(r.get("direction_changed")) for r in rows)
        line += "   direction_changed flagged {}/{}".format(caught, len(rows))
    print(line)
    for r in rows:
        if r["verdict"] != EXPECT[kind]:
            print("      MISS {} -> {!r} {}".format(r["id"], r["verdict"], r.get("note", "")[:70]))

false_alarm = [r for r in control_rows["ctrl_same"] if r["verdict"] != "same"]
print("\n  false-alarm rate on identical text: {}/{}".format(len(false_alarm), len(control_rows["ctrl_same"])))

# ---------------------------------------------------------------- real data
print("\n" + "=" * 78)
print("2. The 800 real rewrites, re-attached to their arms")
print("=" * 78)
ARM_NAME = {"A1": "para_id", "A2": "para_terse", "A3": "para_natural", "A4": "para_lm_shift"}
by_arm = collections.defaultdict(list)
for item, (kind, episode, arm) in key.items():
    if kind == "real" and item in verdicts:
        by_arm[arm].append((episode, verdicts[item]))

print("  {:<14} {:>5} {:>7} {:>8} {:>11} {:>10} {:>10}".format(
    "arm", "n", "same", "weaker", "different", "dir_chg", "ref_chg"))
flagged = []
for arm in ("A1", "A2", "A3", "A4"):
    rows = by_arm[arm]
    if not rows:
        continue
    c = collections.Counter(r["verdict"] for _, r in rows)
    d = sum(bool(r.get("direction_changed")) for _, r in rows)
    f = sum(bool(r.get("referent_changed")) for _, r in rows)
    print("  {:<14} {:>5} {:>7} {:>8} {:>11} {:>10} {:>10}".format(
        ARM_NAME[arm], len(rows), c["same"], c["weaker"], c["different"], d, f))
    for episode, r in rows:
        if r["verdict"] != "same" or r.get("direction_changed"):
            flagged.append((arm, episode, r))

# A4 is the arm whose whole point is to re-describe the same object, so a
# referent_changed there is the finding this review exists to produce.
print("\n" + "=" * 78)
print("3. Every flagged real item (verdict != same, or a direction change)")
print("=" * 78)
if not flagged:
    print("  none")
for arm, episode, r in sorted(flagged, key=lambda x: (x[0], int(x[1]))):
    print("  {} ep {:<6} {:<10} dir={} ref={}  {}".format(
        ARM_NAME[arm], episode, r["verdict"], int(bool(r.get("direction_changed"))),
        int(bool(r.get("referent_changed"))), r.get("note", "")[:90]))

print("\n" + "=" * 78)
print("4. A4 referent changes (the layer the machine gates cannot see)")
print("=" * 78)
a4 = [(e, r) for e, r in by_arm["A4"] if r.get("referent_changed")]
print("  {}/{} A4 items judged to point at a DIFFERENT object".format(len(a4), len(by_arm["A4"])))
for episode, r in sorted(a4, key=lambda x: int(x[0])):
    print("    ep {:<6} {}".format(episode, r.get("note", "")[:96]))
json.dump({"flagged": [(a, e, r) for a, e, r in flagged]}, open(HERE / "flagged.json", "w"), indent=1)
