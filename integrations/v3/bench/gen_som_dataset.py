"""Generate SoM direction-choice training samples from R2R-CE train GT paths.

For every episode, walk the reference path; at ~1.2 m spaced decision points
render the agent's view (640x480, HFOV 90, pitch -15 like deployment),
build set-of-mark candidates the same way the agent does (floor openings from
the depth frame + L/R/B turn options), and label the candidate whose bearing
best matches the geodesic direction toward the episode goal. Output: one
annotated PNG + one JSONL row per sample, ready for LoRA SFT.

Run inside the ``habitat`` conda env:
  python integrations/v3/bench/gen_som_dataset.py --shard 0 --num-shards 8 --out datasets/som_v1
"""
import argparse, gzip, importlib.util, json, math, os, random, re, sys
import numpy as np

ROOT = "/data/pengyh/workspace/FreeAskAgent_R2R"
AGENT = "/data/pengyh/workspace/FreeAskAgent"
HABITAT_DATA = "/data/pengyh/workspace/habitat/data"
HABITAT_ROOT = "/data/pengyh/workspace/habitat/habitat-lab"

# --- load the deployment candidate module under py3.9 (slots shim) ----------
import dataclasses
_orig_dataclass = dataclasses.dataclass
def _compat_dataclass(*a, **k):
    k.pop("slots", None)
    return _orig_dataclass(*a, **k)
dataclasses.dataclass = _compat_dataclass
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
_sm = AGENT + "/agentflow/agents/models_embodied_v2/memory/spatial_memory"
occupancy_grid = _load("sm_pkg.occupancy_grid", _sm + "/occupancy_grid.py")
sys.modules["sm_pkg"] = type(sys)("sm_pkg"); sys.modules["sm_pkg"].occupancy_grid = occupancy_grid
cand_src = open(_sm + "/candidates.py").read().replace("from .occupancy_grid import Frontier",
                                                       "from sm_pkg.occupancy_grid import Frontier")
cand_mod = type(sys)("sm_candidates"); cand_mod.__dict__["__name__"] = "sm_candidates"
sys.modules["sm_candidates"] = cand_mod
exec(compile(cand_src, "candidates.py", "exec"), cand_mod.__dict__)
Candidate = cand_mod.Candidate
dataclasses.dataclass = _orig_dataclass

class Intr:
    def __init__(self, w, h, hfov_deg):
        self.fx = self.fy = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx, self.cy = w / 2.0, h / 2.0

def floor_mask_from(depth, cam, h=1.25, band=0.30, intr=None):
    hh, ww = depth.shape
    us = np.arange(ww)[None, :]; vs = np.arange(hh)[:, None]
    x = (us - intr.cx) * depth / intr.fx
    y = -(vs - intr.cy) * depth / intr.fy
    z = -depth
    row = cam[1, :3]
    y_w = row[0] * x + row[1] * y + row[2] * z + cam[1, 3]
    return np.abs(y_w - (cam[1, 3] - h)) <= band

PHASE_RE = [(r"\bturn\s+left\b", "TURN_LEFT"), (r"\bturn\s+right\b", "TURN_RIGHT"),
            (r"\b(straight|hallway|corridor)\b", "FOLLOW_CORRIDOR")]
def phase_for(text):
    t = text.lower()
    for pat, ph in PHASE_RE:
        if re.search(pat, t):
            return ph
    return "APPROACH_LANDMARK"

def sentences(instr):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", instr.strip()) if p.strip()]
    return parts or [instr.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="datasets/som_v1")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all in shard")
    ap.add_argument("--spacing-m", type=float, default=1.2)
    ap.add_argument("--pitch-deg", type=float, default=-15.0)
    ap.add_argument("--wide-jitter", action="store_true", help="half the poses face up to 180 deg off-route so turning choices (L/R/B and far-side markers) get learned")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, args.out)
    img_dir = os.path.join(out_dir, "images"); os.makedirs(img_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, f"shard_{args.shard:02d}.jsonl")

    os.environ.setdefault("MAGNUM_LOG", "quiet"); os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    import habitat_sim
    from PIL import Image

    data = json.load(gzip.open(HABITAT_DATA + "/datasets/vln/mp3d/r2r/v1/train/train.json.gz"))
    episodes = data["episodes"]
    by_scene = {}
    for e in episodes:
        by_scene.setdefault(e["scene_id"], []).append(e)
    scenes = sorted(by_scene)
    my_scenes = scenes[args.shard::args.num_shards]
    random.seed(1234 + args.shard)
    written = 0
    with open(rows_path, "w") as rows:
        for scene in my_scenes:
            scene_path = os.path.join(HABITAT_DATA, "scene_datasets", scene)
            cfg = habitat_sim.SimulatorConfiguration()
            cfg.scene_id = scene_path
            cfg.gpu_device_id = 0
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = "rgb"; spec.sensor_type = habitat_sim.SensorType.COLOR
            spec.resolution = [480, 640]; spec.hfov = 90
            spec.position = [0.0, 1.25, 0.0]
            spec.orientation = [math.radians(args.pitch_deg), 0.0, 0.0]
            dspec = habitat_sim.CameraSensorSpec()
            dspec.uuid = "depth"; dspec.sensor_type = habitat_sim.SensorType.DEPTH
            dspec.resolution = [480, 640]; dspec.hfov = 90
            dspec.position = [0.0, 1.25, 0.0]
            dspec.orientation = [math.radians(args.pitch_deg), 0.0, 0.0]
            agent_cfg.sensor_specifications = [spec, dspec]
            try:
                sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
            except Exception as exc:
                print(f"scene {scene}: {type(exc).__name__}: {exc}", flush=True)
                continue
            pf = sim.pathfinder
            intr = Intr(640, 480, 90.0)
            eps = by_scene[scene]
            if args.episodes:
                eps = eps[: args.episodes]
            for e in eps:
                goal = np.asarray(e["goals"][0]["position"], dtype=np.float64)
                ref = [np.asarray(p, dtype=np.float64) for p in e["reference_path"]]
                instr = e["instruction"]["instruction_text"].strip()
                sents = sentences(instr)
                # densify the reference path
                points = []
                for a, b in zip(ref[:-1], ref[1:]):
                    seg = np.linalg.norm(b - a)
                    n = max(1, int(seg / args.spacing_m))
                    for k in range(n):
                        points.append(a + (b - a) * (k / n))
                points.append(ref[-1])
                total = len(points)
                for idx, pos in enumerate(points[:-1]):
                    snapped = pf.snap_point(pos)
                    if not np.isfinite(snapped).all():
                        continue
                    # GT direction: geodesic next point toward the goal
                    path = habitat_sim.ShortestPath()
                    path.requested_start = snapped; path.requested_end = pf.snap_point(goal)
                    if not pf.find_path(path) or len(path.points) < 2:
                        continue
                    nxt = np.asarray(path.points[1], dtype=np.float64)
                    look = nxt - np.asarray(snapped)
                    if np.linalg.norm(look[[0, 2]]) < 0.15:
                        continue
                    gt_yaw = math.degrees(math.atan2(look[0], -look[2]))
                    # face along the route with jitter so views vary
                    if args.wide_jitter and random.random() < 0.5:
                        face_yaw = gt_yaw + random.uniform(-180.0, 180.0)
                    else:
                        face_yaw = gt_yaw + random.uniform(-35.0, 35.0)
                    state = habitat_sim.AgentState()
                    state.position = snapped
                    state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
                        math.radians(-face_yaw), np.array([0.0, 1.0, 0.0]))
                    sim.get_agent(0).set_state(state, reset_sensors=True)
                    obs = sim.get_sensor_observations()
                    rgb = np.asarray(obs["rgb"])[..., :3]
                    depth = np.asarray(obs["depth"], dtype=np.float64)
                    sensor = sim.get_agent(0).get_state().sensor_states["rgb"]
                    import habitat_sim.utils.common as huc
                    cam = np.eye(4)
                    cam[:3, :3] = np.asarray(huc.quat_to_magnum(sensor.rotation).to_matrix())
                    cam[:3, 3] = np.asarray(sensor.position)
                    fmask = floor_mask_from(depth, cam, intr=intr)
                    cands = cand_mod.floor_openings(depth, fmask, intr, cam)
                    # navmesh filter + numbering left-to-right
                    cands = [c for c in cands if pf.is_navigable(
                        [c.world_xyz[0], float(snapped[1]), c.world_xyz[2]], 0.5)]
                    cands.sort(key=lambda c: c.pixel_uv[0] if c.pixel_uv else 0)
                    for i, c in enumerate(cands, start=1):
                        c.label = str(i)
                    if not cands:
                        continue
                    for lab, note in (("L", "turn left: area outside the current view"),
                                      ("R", "turn right: area outside the current view"),
                                      ("B", "turn around: the way back")):
                        cands.append(Candidate(lab, (0.0, 0.0, 0.0), 0.0,
                                               {"L": -90.0, "R": 90.0, "B": 180.0}[lab],
                                               "turn", None, note))
                    # label: candidate closest in bearing to the GT direction
                    rel = lambda c: abs((({"L": -90.0, "R": 90.0, "B": 180.0}.get(c.label, c.bearing_deg)
                                           if c.kind == "turn" else c.bearing_deg)
                                          - (gt_yaw - face_yaw) + 180) % 360 - 180)
                    in_view = [c for c in cands if c.kind != "turn"]
                    best = min(in_view, key=rel)
                    gt_rel = ((gt_yaw - face_yaw) + 180) % 360 - 180
                    if rel(best) > 50.0:
                        best = min((c for c in cands if c.kind == "turn"), key=rel)
                    prog = idx / max(1, total - 1)
                    sent_i = min(len(sents) - 1, int(prog * len(sents)))
                    subgoal = sents[sent_i]
                    text = "\n".join((
                        "Current subgoal ({} of {}): {}\nCompletion evidence: the view shows this part of the route done.".format(
                            sent_i + 1, len(sents), subgoal),
                        "Full route instruction: {}".format(instr),
                        "Required navigation phase: {}.".format(phase_for(subgoal)),
                        "", "",
                        "Options:",
                        cand_mod.describe_candidates(cands),
                        "Choose one option label.",
                    ))
                    annotated = cand_mod.annotate_image(rgb, cands)
                    name = "{}_{:03d}.png".format(e["episode_id"], idx)
                    Image.fromarray(annotated).save(os.path.join(img_dir, name))
                    rows.write(json.dumps({
                        "image": "images/" + name,
                        "prompt": text,
                        "label": best.label,
                        "target": '{"choice":"%s","confidence":0.9,"evidence":"toward the route direction"}' % best.label,
                        "meta": {"episode": e["episode_id"], "idx": idx, "gt_rel_deg": round(gt_rel, 1),
                                 "n_candidates": len(cands), "scene": scene},
                    }) + "\n")
                    written += 1
            sim.close()
            print(f"scene {scene} done, total rows {written}", flush=True)
    print("shard", args.shard, "written", written)

if __name__ == "__main__":
    main()
