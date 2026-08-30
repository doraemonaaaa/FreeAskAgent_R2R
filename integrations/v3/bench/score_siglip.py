import sys, json, re, time, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, __import__("os").path.dirname(__file__))
from landmark_np import landmark_np
bench = sys.argv[1]
meta = json.load(open(f"{bench}/meta.json")); plans = json.load(open(f"{bench}/plans.json"))
name = "/data/pengyh/workspace/FreeAskAgent/models/siglip-so400m-patch14-384"
model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval(); proc = AutoProcessor.from_pretrained(name)
def wrap(a): return (a + 180) % 360 - 180
def stage(m):
    subs = plans.get(m["episode_id"]) or [m["instruction"]]; return subs[min(len(subs) - 1, int(m["sample_index"] * len(subs) / max(1, m["n_samples"])))]
ROOM = {"hallway", "hall", "kitchen", "bedroom", "bathroom", "lobby", "corridor", "living room", "dining room", "office", "patio", "garage", "closet", "stairs", "staircase", "stairway", "laundry room", "foyer", "entryway"}
rows = []; t0 = time.perf_counter()
for m in meta:
    desc = stage(m); noun = landmark_np(desc); gt = m["gt_bearing_deg"]
    if not noun: continue
    ims = [Image.open(f"{bench}/{m['episode_id']}/{m['k']:02d}/view_{int(y):+04d}.png").convert("RGB") for y in m["yaws"]]
    with torch.no_grad():
        inp = proc(text=[f"a photo of a {noun}", "a photo of a wall"], images=ims, padding="max_length", return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        logits = model(**inp).logits_per_image.float().cpu().numpy()[:, 0]
    # smooth over adjacent views (a landmark straddles two 45-degree views)
    order = list(m["yaws"]); sm = np.array([logits[i] + 0.5 * (logits[(i - 1) % 8] + logits[(i + 1) % 8]) for i in range(8)])
    chosen = order[int(np.argmax(sm))]; best = min(order, key=lambda y: abs(wrap(y - gt)))
    rows.append(dict(ep=m["episode_id"], k=m["k"], noun=noun, room=(noun in ROOM), chosen=chosen, best=best, err=abs(wrap(chosen - gt)), gt=gt, margin=float(sm.max() - np.median(sm))))
def rep(name, rs):
    if not rs: return
    print(f"{name:12s} n={len(rs):3d} nearest-view {100*np.mean([r['chosen']==r['best'] for r in rs]):.0f}%  err<=45°: {100*np.mean([r['err']<=45 for r in rs]):.0f}%  forward<=45°: {100*np.mean([abs(r['gt'])<=45 for r in rs]):.0f}%")
print(f"SigLIP  {(time.perf_counter()-t0)/max(1,len(rows))*1000:.0f} ms/pt")
rep("all", rows); rep("room nouns", [r for r in rows if r["room"]]); rep("object nouns", [r for r in rows if not r["room"]])
turn = [r for r in rows if abs(r["gt"]) > 30]; rep("turning", turn); rep("turning+room", [r for r in turn if r["room"]])
hi = sorted(rows, key=lambda r: -r["margin"])[:40]; rep("top-40 margin", hi)
json.dump(rows, open(f"{bench}/scores_siglip.json", "w"), indent=1)
