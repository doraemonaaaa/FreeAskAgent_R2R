"""Score three preview-selection variants on benchmark v2.
usage: score_preview_bench2.py <bench> <model> <url> [limit]
variants: five (current selector, views -90..+90) | eight (same prompt, 8 views) | eight+facts (8 views + open-floor distance per view)"""
import sys, json, time, math, collections, io
import numpy as np
from PIL import Image
from agentflow.agents.engine.remote_qwen3vl import RemoteQwen3VL
from agentflow.agents.models_embodied_v2.memory.temporal_memory.temporal_captioner import TemporalCaptioner
from agentflow.agents.models_embodied_v2.data_models import Subgoal, TemporalCaptionerConfig
from agentflow.agents.models_embodied_v2.skiils.planning import parse_subgoal_plan
from agentflow.agents.models_embodied_v2.skiils.protocol import SUBGOAL_PROMPT
from agentflow.agents.models_embodied_v2.skiils.preview import PREVIEW_SELECTION_PROMPT, parse_preview_selection

bench, model, url = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
meta = json.load(open(f"{bench}/meta.json"))[:limit]
engine = RemoteQwen3VL(model, base_url=url)
captioner = TemporalCaptioner(engine=engine, config=TemporalCaptionerConfig(max_tokens=256))

class View:
    def __init__(self, yaw, rgb): self.yaw_deg, self.rgb = yaw, rgb
def wrap(a): return (a + 180) % 360 - 180
def png(rgb, max_edge=448):
    im = Image.fromarray(rgb); s = max_edge / max(im.size)
    if s < 1: im = im.resize((int(im.width * s), int(im.height * s)))
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()

plans = {}
def subgoal_for(m):
    ep = m["episode_id"]
    if ep not in plans:
        try:
            resp = engine([f"Navigation instruction: {m['instruction']}"], system_prompt=SUBGOAL_PROMPT, max_tokens=1024, temperature=0)
            plans[ep] = parse_subgoal_plan(str(resp), instruction=m["instruction"])
        except Exception:
            plans[ep] = [Subgoal("1", m["instruction"], "The destination is reached.")]
    subs = plans[ep]
    return subs[min(len(subs) - 1, int(m["sample_index"] * len(subs) / max(1, m["n_samples"])))]

def select_with_facts(sg, views, opens):
    content = [f"Active subgoal: {sg.description}\nCompletion criterion: {sg.completion_criteria}\n"
               f"Available simultaneous views: {len(views)}. Each view lists how far the floor is walkable straight ahead in that direction (from the depth sensor)."]
    for i, v in enumerate(views):
        content.append(f"view_index={i}; yaw_deg={v.yaw_deg:+.1f}; walkable_ahead={opens[str(int(v.yaw_deg))]:.1f} m")
        content.append(png(v.rgb))
    resp = engine(content, system_prompt=PREVIEW_SELECTION_PROMPT, max_tokens=96, temperature=0)
    return parse_preview_selection(dict(captioner._json_value(str(resp))), view_count=len(views))

results = {}
for variant in ("five", "eight", "eight+facts"):
    rows = []; t_all = 0.0
    for m in meta:
        yaws = [y for y in m["yaws"] if (variant != "five" or abs(y) <= 90)]
        views = [View(y, np.asarray(Image.open(f"{bench}/{m['episode_id']}/{m['k']:02d}/view_{int(y):+04d}.png").convert("RGB"))) for y in yaws]
        sg = subgoal_for(m); gt = m["gt_bearing_deg"]
        t = time.perf_counter()
        try:
            sel = select_with_facts(sg, views, m["open_m"]) if variant == "eight+facts" else captioner.select(subgoal=sg, views=views)
        except Exception:
            sel = None
        t_all += time.perf_counter() - t
        chosen = yaws[sel.view_index] if sel is not None else None
        best = min(yaws, key=lambda y: abs(wrap(y - gt)))
        rows.append(dict(ep=m["episode_id"], k=m["k"], gt=gt, best=best, chosen=chosen, hit=(chosen == best), forward_hit=(best == 0.0),
                         err=(abs(wrap(chosen - gt)) if chosen is not None else None)))
    ok = [r for r in rows if r["chosen"] is not None]; turn = [r for r in ok if r["best"] != 0.0]
    behind = [r for r in ok if abs(r["gt"]) > 112.5]
    results[variant] = dict(n=len(rows), answered=len(ok), acc=100*np.mean([r["hit"] for r in ok]), forward=100*np.mean([r["forward_hit"] for r in rows]),
                            turn_n=len(turn), turn_acc=100*np.mean([r["hit"] for r in turn]) if turn else 0, turn_forward=100*np.mean([r["chosen"]==0.0 for r in turn]) if turn else 0,
                            behind_n=len(behind), behind_acc=100*np.mean([r["hit"] for r in behind]) if behind else 0,
                            err=np.mean([r["err"] for r in ok]), ms=t_all/len(rows)*1000)
    json.dump(rows, open(f"{bench}/scores_{model}_{variant}.json", "w"), indent=1)
    r = results[variant]
    print(f"{model} {variant:12s} n={r['n']} acc={r['acc']:.0f}% (forward-baseline {r['forward']:.0f}%, random {100/ (5 if variant=='five' else 8):.0f}%)  turning n={r['turn_n']} acc={r['turn_acc']:.0f}% picked-forward={r['turn_forward']:.0f}%  behind n={r['behind_n']} acc={r['behind_acc']:.0f}%  mean|err|={r['err']:.0f}°  {r['ms']:.0f} ms", flush=True)
json.dump(results, open(f"{bench}/summary_{model}.json", "w"), indent=1)
