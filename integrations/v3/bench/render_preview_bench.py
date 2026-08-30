"""Preview benchmark v2: points every ~1.5 m along the reference path, 8 views
(every 45 deg incl. behind), per-view open-floor distance from depth, geodesic GT."""
import os, sys, json, math
sys.path.insert(0, "/data/pengyh/workspace/FreeAskAgent_R2R/integrations/v3")
import numpy as np
from PIL import Image
import run_habitat as rh
import habitat, habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_magnum

OUT = sys.argv[1]; SPACING = 1.5
ids = [l.split("#")[0].strip().split(",")[0] for l in open("/data/pengyh/workspace/FreeAskAgent_R2R/integrations/v3/eval_sets/val_unseen_40.txt") if l.split("#")[0].strip()]
YAWS = [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
overrides = rh.R2R_CE_OVERRIDES + rh.DEPTH_SENSOR_OVERRIDES + [
    "habitat.dataset.split=val_unseen",
    "habitat.dataset.data_path='{}/datasets/vln/mp3d/r2r/v1/{{split}}/{{split}}.json.gz'".format(rh.HABITAT_DATA),
    "habitat.dataset.scenes_dir={}/scene_datasets".format(rh.HABITAT_DATA),
    "habitat.environment.max_episode_steps=5"]
config = habitat.get_config("benchmark/nav/vln_r2r.yaml", overrides=overrides)
os.chdir(rh.HABITAT_ROOT)

def yaw_of(rotation):
    R = np.asarray(quat_to_magnum(rotation).to_matrix()); f = -R[:, 2]
    return math.degrees(math.atan2(f[0], -f[2]))
def bearing(pos, yaw_deg, target):
    dx, dz = target[0] - pos[0], target[2] - pos[2]; y = math.radians(yaw_deg)
    return math.degrees(math.atan2(dx * math.cos(y) + dz * math.sin(y), dx * math.sin(y) - dz * math.cos(y)))
def open_distance(depth):
    """Walkable distance straight ahead in this view: median depth of the
    lower-central band that is floor-like (depth grows with row)."""
    h, w = depth.shape; band = depth[int(h * 0.55):int(h * 0.95), int(w * 0.4):int(w * 0.6)]
    band = band[np.isfinite(band) & (band > 0.2)]
    return float(np.percentile(band, 85)) if band.size else 0.0

meta = []
with habitat.Env(config=config) as env:
    by_id = {str(e.episode_id): e for e in env.episodes}
    env.episodes = [by_id[i] for i in ids if i in by_id]
    for ep in env.episodes:
        env.reset(); sim = env.sim; pf = sim.pathfinder
        ref = [np.asarray(p, dtype=np.float64) for p in ep.reference_path]
        goal = np.asarray(ep.goals[0].position, dtype=np.float64)
        # densify: walk the geodesic between consecutive reference points
        samples = []
        for a, b in zip(ref[:-1], ref[1:]):
            path = habitat_sim.ShortestPath(); path.requested_start = a.astype(np.float32); path.requested_end = b.astype(np.float32)
            pts = [np.asarray(p, dtype=np.float64) for p in path.points] if pf.find_path(path) else [a, b]
            acc = 0.0; last = pts[0]; samples.append(pts[0])
            for p in pts[1:]:
                seg = np.linalg.norm((p - last)[[0, 2]]); acc += seg
                while acc >= SPACING:
                    t = 1 - (acc - SPACING) / max(seg, 1e-6); samples.append(last + (p - last) * t); acc -= SPACING
                last = p
        kept = 0
        for k, raw in enumerate(samples):
            pos = np.asarray(pf.snap_point(raw.astype(np.float32)), dtype=np.float64)
            if not np.all(np.isfinite(pos)) or np.linalg.norm((goal - pos)[[0, 2]]) < 1.0:
                continue
            path = habitat_sim.ShortestPath(); path.requested_start = pos.astype(np.float32); path.requested_end = goal.astype(np.float32)
            if not pf.find_path(path) or len(path.points) < 2:
                continue
            nxt = next((np.asarray(q, dtype=np.float64) for q in path.points[1:] if np.linalg.norm((np.asarray(q) - pos)[[0, 2]]) >= 0.75), goal)
            # heading = direction of arrival along the route (start rotation at the first point)
            if k == 0:
                rotation = sim.get_agent_state().rotation
            else:
                d = pos - np.asarray(pf.snap_point(samples[k - 1].astype(np.float32)), dtype=np.float64)
                if np.linalg.norm(d[[0, 2]]) < 1e-3: d = nxt - pos
                rotation = quat_from_angle_axis(math.atan2(-d[0], -d[2]), np.array([0.0, 1.0, 0.0]))
            sim.set_agent_state(pos.astype(np.float32), rotation, reset_sensors=True)
            yaw = yaw_of(sim.get_agent_state().rotation); gt = bearing(pos, yaw, nxt)
            views = rh._preview_views(env, YAWS, 90.0, 1.0)
            if any((v["rgb"].max(axis=2) < 8).mean() > 0.3 for v in views):
                continue
            d = f"{OUT}/{ep.episode_id}/{kept:02d}"; os.makedirs(d, exist_ok=True)
            opens = {}
            for v in views:
                Image.fromarray(v["rgb"]).save(f"{d}/view_{int(v['yaw_deg']):+04d}.png")
                opens[str(int(v["yaw_deg"]))] = round(open_distance(np.asarray(v["depth"], dtype=np.float64).squeeze()), 2)
            meta.append({"episode_id": str(ep.episode_id), "k": kept, "n_samples": len(samples), "sample_index": k,
                         "instruction": ep.instruction.instruction_text, "yaws": [v["yaw_deg"] for v in views],
                         "gt_bearing_deg": gt, "open_m": opens, "dist_to_goal_m": float(np.linalg.norm((goal - pos)[[0, 2]]))})
            kept += 1
        print(ep.episode_id, "points", kept, flush=True)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print("decision points:", len(meta))
