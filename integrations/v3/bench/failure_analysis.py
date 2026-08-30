"""Classify why each episode of an A/B config failed. usage: failure_analysis.py <config> [<config>...]"""
import re, glob, sys, os, json, collections, statistics as st
S = os.environ.get("AB_ROOT", "/tmp/claude-2001/-data-pengyh/6c06fa75-c808-4e1c-a492-c760302e8758/scratchpad")
cats = {}
for line in open("/data/pengyh/workspace/FreeAskAgent_R2R/integrations/v3/eval_sets/val_unseen_40.txt"):
    line = line.split("#")[0].strip()
    if line: i, c = line.split(","); cats[i] = c
def stage_type(desc):
    d = desc.lower()
    if re.search(r"\bturn\b", d) and not re.search(r"\bwalk|\bgo\b|\bhead\b", d): return "turn"
    if re.search(r"\bstair", d): return "stairs"
    if re.search(r"\b(door|doorway|exit|enter|out of|into the|through the|entrance|archway)\b", d): return "doorway"
    if re.search(r"\b(straight|hallway|hall|corridor|along)\b", d) and not re.search(r"\b(to|toward|until|past)\s+the\b", d): return "corridor"
    if re.search(r"\b(stop|wait|stand)\b", d): return "stop"
    return "landmark"
def analyze(cfg):
    eps = []
    for f in glob.glob(f"{S}/ab/{cfg}/rank_*.log"):
        t = open(f, errors="replace").read()
        for blk in re.finditer(r"^episode=(\d+) instruction=(['\"])(.*?)\2 start.*?(?=^episode=\d+ instruction=|\Z)", t, re.M | re.S):
            ep, instr, body = blk.group(1), blk.group(3), blk.group(0)
            subs = re.findall(r"^  \[(\d+)\] (.*)$", body, re.M)
            r = re.search(r"steps=(\d+) success=([\d.]+) spl=([\d.]+) dtg=([\d.]+)", body)
            if not r: continue
            steps, succ, spl, dtg = int(r[1]), float(r[2]), float(r[3]), float(r[4])
            sg = re.findall(r"^ep=\d+ s=(\d+) .*?sg=\S+->(\d+).*?dtg=([\d.]+)$", body, re.M)
            per_stage = collections.OrderedDict()
            for s_, g, d in sg: per_stage.setdefault(int(g), []).append((int(s_), float(d)))
            n = len(subs); last = max(per_stage) if per_stage else 0
            reached_final = n > 0 and last == n
            stopped = steps < 150 and "act=STOP" in body
            dtg0 = float(sg[0][2]) if sg else dtg
            if succ > 0: cls = "success"
            elif stopped: cls = "wrong_stop"
            elif reached_final: cls = "final_timeout_near" if dtg <= 3.5 else "final_timeout_far"
            else:
                stalled = per_stage.get(last, [])
                stype = stage_type(dict((int(i), d) for i, d in subs).get(last, ""))
                dd = stalled[-1][1] - stalled[0][1] if stalled else 0
                cls = f"stalled_{stype}"
            eps.append(dict(ep=ep, cat=cats.get(ep, "?"), cls=cls, stages=n, last=last, steps=steps, dtg=dtg, dtg0=dtg0,
                            stall_steps=len(per_stage.get(last, [])), stall_desc=dict((int(i), d) for i, d in subs).get(last, "")[:60]))
    return eps
for cfg in sys.argv[1:]:
    eps = analyze(cfg)
    print(f"\n===== {cfg}: n={len(eps)} SR={st.mean(e['dtg']<=3 and e['cls']=='success' for e in eps):.2f}")
    counts = collections.Counter(e["cls"] for e in eps)
    for c, k in counts.most_common(): print(f"  {c:22s} {k:2d}  ({100*k/len(eps):.0f}%)")
    stall = [e for e in eps if e["cls"].startswith("stalled")]
    if stall: print(f"  stalled episodes: median steps on the stalled stage={st.median(e['stall_steps'] for e in stall):.0f}, median dtg change {st.median(e['dtg']-e['dtg0'] for e in stall):+.1f} m")
    print("  by category:", {c: collections.Counter(e['cls'] for e in eps if e['cat']==c).most_common(2) for c in ("doorway","walk","stairs","abstract_stop")})
    print("  examples of stalled stages:")
    for e in sorted(stall, key=lambda e: -e["stall_steps"])[:8]: print(f"    ep{e['ep']:>5} {e['cat']:13s} stage {e['last']}/{e['stages']} {e['stall_steps']:3d} steps: {e['stall_desc']}")
    json.dump(eps, open(f"{S}/ab/{cfg}/failure_classes.json", "w"), indent=1)
