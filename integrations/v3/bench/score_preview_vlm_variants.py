"""Preview variants that use the views for RECOGNITION, not direction choice.
usage: score_preview_bench3.py <bench> <model> <url> [limit]"""
import sys, json, time, math, re, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from agentflow.agents.engine.remote_qwen3vl import RemoteQwen3VL
from agentflow.agents.models_embodied_v2.data_models import Subgoal
from agentflow.agents.models_embodied_v2.skiils.planning import parse_subgoal_plan, landmark_phrase
from agentflow.agents.models_embodied_v2.skiils.protocol import SUBGOAL_PROMPT

bench, model, url = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
meta = json.load(open(f"{bench}/meta.json"))[:limit]
engine = RemoteQwen3VL(model, base_url=url)
def wrap(a): return (a + 180) % 360 - 180
def png(im, max_edge=448):
    s = max_edge / max(im.size)
    if s < 1: im = im.resize((int(im.width * s), int(im.height * s)))
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()
def jsonobj(text):
    t = str(text).strip()
    if t.startswith("```"): t = "\n".join(t.splitlines()[1:-1])
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])

plans = {}
def stage_for(m):
    ep = m["episode_id"]
    if ep not in plans:
        try:
            resp = engine([f"Navigation instruction: {m['instruction']}"], system_prompt=SUBGOAL_PROMPT, max_tokens=1024, temperature=0)
            plans[ep] = parse_subgoal_plan(str(resp), instruction=m["instruction"])
        except Exception:
            plans[ep] = [Subgoal("1", m["instruction"], "The destination is reached.")]
    subs = plans[ep]
    return subs[min(len(subs) - 1, int(m["sample_index"] * len(subs) / max(1, m["n_samples"])))]

LOCATE8 = """You see several simultaneous views from one standing position, each labelled
with view_index and its heading offset (negative = left, positive = right,
+180 = behind). Find the named thing. Reply with exactly one single-line JSON
object; angle brackets are values you fill in:
{"visible":<true|false>,"view_index":<int or null>,"u":<int 0-1000 or null>,"v":<int 0-1000 or null>,"confidence":<float 0-1>,"note":"<at most 12 words>"}
visible=false when it is in none of the views; a look-alike is NOT it."""

PANO = """The image is a panorama made of 8 tiles from one standing position; each tile is
labelled with its heading offset in degrees (negative = left, positive = right,
+180 = behind). Choose the tile whose direction best continues the active
navigation subgoal. Reply with exactly one single-line JSON object:
{"heading_deg":<one of the labelled headings>,"confidence":<float 0-1>,"evidence":"<at most 15 words>"}"""

def direction_prior(desc):
    d = desc.lower()
    if re.search(r"\b(turn around|go back|behind you|reverse)\b", d): return "BACK"
    if re.search(r"\bturn\s+left\b|\bon (your|the) left\b|\bto (your|the) left\b|\bleft\b", d): return "LEFT"
    if re.search(r"\bturn\s+right\b|\bon (your|the) right\b|\bto (your|the) right\b|\bright\b", d): return "RIGHT"
    if re.search(r"\b(straight|forward|ahead|continue|down the hall|along)\b", d): return "AHEAD"
    return None
SIDE_VIEWS = {"LEFT": [-45.0, -90.0, -135.0], "RIGHT": [45.0, 90.0, 135.0], "AHEAD": [0.0, -45.0, 45.0], "BACK": [180.0, -135.0, 135.0]}

def views_of(m):
    return {y: Image.open(f"{bench}/{m['episode_id']}/{m['k']:02d}/view_{int(y):+04d}.png").convert("RGB") for y in m["yaws"]}

def locate8(m, sg, views):
    phrase = landmark_phrase(sg.description)
    content = [f"Thing to find: {phrase}\nRoute stage: {sg.description}\nFull route instruction: {m['instruction']}\nViews: {len(views)}."]
    for i, (y, im) in enumerate(views.items()):
        content.append(f"view_index={i}; heading_deg={y:+.0f}"); content.append(png(im, 320))
    r = jsonobj(engine(content, system_prompt=LOCATE8, max_tokens=96, temperature=0))
    if r.get("visible") and r.get("view_index") is not None:
        return list(views.keys())[int(r["view_index"])], float(r.get("confidence") or 0)
    return None, 0.0

def panorama(m, sg, views):
    tiles = [(y, im.resize((320, 240))) for y, im in views.items()]
    canvas = Image.new("RGB", (320 * 4, (240 + 24) * 2), (20, 20, 20)); draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for i, (y, im) in enumerate(tiles):
        x, yy = (i % 4) * 320, (i // 4) * 264
        canvas.paste(im, (x, yy + 24)); draw.text((x + 6, yy + 3), f"heading {y:+.0f}", fill=(255, 255, 0), font=font)
    content = [f"Active subgoal: {sg.description}\nCompletion criterion: {sg.completion_criteria}\nFull route instruction: {m['instruction']}", png(canvas, 900)]
    r = jsonobj(engine(content, system_prompt=PANO, max_tokens=96, temperature=0))
    h = float(r["heading_deg"]); return min(views.keys(), key=lambda y: abs(wrap(y - h)))

def prior_pick(m, sg):
    side = direction_prior(sg.description)
    cands = SIDE_VIEWS.get(side, [0.0])
    return max(cands, key=lambda y: m["open_m"].get(str(int(y)), 0.0)), side

def run(variant):
    rows = []; t0 = time.perf_counter()
    for m in meta:
        sg = stage_for(m); views = views_of(m); gt = m["gt_bearing_deg"]; best = min(views.keys(), key=lambda y: abs(wrap(y - gt)))
        chosen = None; how = ""
        try:
            if variant == "locate8":
                chosen, _ = locate8(m, sg, views); how = "landmark" if chosen is not None else "none"
                if chosen is None: chosen = 0.0
            elif variant == "panorama":
                chosen = panorama(m, sg, views); how = "pano"
            elif variant == "prior":
                chosen, side = prior_pick(m, sg); how = side or "none"
            elif variant == "prior+locate":
                chosen, _ = locate8(m, sg, views); how = "landmark"
                if chosen is None:
                    chosen, side = prior_pick(m, sg); how = side or "none"
        except Exception as e:
            chosen, how = 0.0, f"err:{type(e).__name__}"
        rows.append(dict(ep=m["episode_id"], k=m["k"], gt=gt, best=best, chosen=chosen, hit=(chosen == best), how=how, forward_hit=(best == 0.0)))
    turn = [r for r in rows if r["best"] != 0.0]
    by = {}
    for r in rows: by.setdefault(r["how"], []).append(r["hit"])
    print(f"{model} {variant:13s} n={len(rows)} acc={100*np.mean([r['hit'] for r in rows]):.0f}% (forward {100*np.mean([r['forward_hit'] for r in rows]):.0f}%)  turning n={len(turn)} acc={100*np.mean([r['hit'] for r in turn]):.0f}%  by-how={ {k: f'{100*np.mean(v):.0f}%/{len(v)}' for k, v in by.items()} }  {(time.perf_counter()-t0)/len(rows)*1000:.0f} ms/pt", flush=True)
    json.dump(rows, open(f"{bench}/scores3_{model}_{variant}.json", "w"), indent=1)

for v in ("prior", "locate8", "panorama", "prior+locate"):
    run(v)
