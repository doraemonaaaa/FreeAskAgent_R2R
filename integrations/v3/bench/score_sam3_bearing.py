import sys, json, re, time, collections, numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
sys.path.insert(0, __import__("os").path.dirname(__file__))
from landmark_np import landmark_np
bench = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
meta = json.load(open(f"{bench}/meta.json")); plans = json.load(open(f"{bench}/plans.json"))
model = Sam3Model.from_pretrained("/data/pengyh/workspace/FreeAskAgent/models/sam3", torch_dtype=torch.bfloat16).cuda().eval()
proc = Sam3Processor.from_pretrained("/data/pengyh/workspace/FreeAskAgent/models/sam3")
def wrap(a): return (a + 180) % 360 - 180
def stage(m):
    subs = plans.get(m["episode_id"]) or [m["instruction"]]; return subs[min(len(subs) - 1, int(m["sample_index"] * len(subs) / max(1, m["n_samples"])))]
APPROACH = re.compile(r"\b(to|toward|towards|into|enter|reach|through|up the|down the|climb|stop (at|beside|next to|in front of|under|by)|wait)\b", re.I)
AWAY = re.compile(r"\b(out of|past|away|leave|exit|around)\b", re.I)
HFOV = 90.0
rows = []; t0 = time.perf_counter()
for m in meta:
    desc = stage(m); noun = landmark_np(desc); gt = m["gt_bearing_deg"]
    kind = "away" if AWAY.search(desc) else ("approach" if APPROACH.search(desc) else "other")
    dets = []
    if noun:
        for y in m["yaws"]:
            im = Image.open(f"{bench}/{m['episode_id']}/{m['k']:02d}/view_{int(y):+04d}.png").convert("RGB")
            inp = proc(images=im, text=noun, return_tensors="pt").to("cuda"); inp = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
            with torch.no_grad(): out = model(**inp)
            res = proc.post_process_instance_segmentation(out, threshold=thr, mask_threshold=0.5, target_sizes=inp["original_sizes"].tolist())[0]
            W = im.width
            for b, s in zip(res["boxes"], res["scores"]):
                cx = (float(b[0]) + float(b[2])) / 2 / W
                dets.append((float(s), wrap(y + (cx - 0.5) * HFOV), y))
    dets.sort(reverse=True)
    top = dets[0] if dets else None
    err = abs(wrap(top[1] - gt)) if top else None
    any30 = any(abs(wrap(d[1] - gt)) <= 30 for d in dets[:3])
    rows.append(dict(ep=m["episode_id"], k=m["k"], noun=noun, kind=kind, gt=gt, n_det=len(dets), top_bearing=(top[1] if top else None), top_score=(top[0] if top else 0), err=err, any30=any30, forward_err=abs(gt)))
def rep(name, rs):
    vis = [r for r in rs if r["err"] is not None]
    if not rs: return
    print(f"{name:10s} n={len(rs):3d} detected={len(vis):3d} ({100*len(vis)/len(rs):.0f}%)  top-det bearing err<=30°: {100*np.mean([r['err']<=30 for r in vis]) if vis else 0:.0f}%  <=45°: {100*np.mean([r['err']<=45 for r in vis]) if vis else 0:.0f}%  any-of-top3<=30°: {100*np.mean([r['any30'] for r in vis]) if vis else 0:.0f}%  | forward<=30°: {100*np.mean([r['forward_err']<=30 for r in rs]):.0f}%  (forward on detected pts: {100*np.mean([r['forward_err']<=30 for r in vis]) if vis else 0:.0f}%)")
print(f"SAM3 bearing scoring thr={thr}  {(time.perf_counter()-t0)/len(rows)*1000:.0f} ms/pt")
rep("all", rows); rep("approach", [r for r in rows if r["kind"] == "approach"]); rep("away", [r for r in rows if r["kind"] == "away"]); rep("other", [r for r in rows if r["kind"] == "other"])
turn = [r for r in rows if r["forward_err"] > 30]; rep("turning", turn)
json.dump(rows, open(f"{bench}/scores_sam3_bearing_{thr}.json", "w"), indent=1)
