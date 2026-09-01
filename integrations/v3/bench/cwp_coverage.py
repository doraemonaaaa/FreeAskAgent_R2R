"""Offline candidate-coverage benchmark along 200-set GT trajectories.

At each decision point (reference path densified at --spacing-m, agent facing
its incoming travel direction) we measure, per candidate source, whether any
candidate lies within +-30 deg of the GT next-hop direction:

  ring      full 12-view CWP ring at the current pose (look-around upper bound)
  fo        our deployed floor-openings generator (monocular, pitch -15 view)
  stale     CWP fed the previous decision point's ring, slot-aligned to the
            current heading, with slot 0 replaced by a fresh forward render
  front45   ceiling for ANY monocular method: GT direction within +-45 deg

Angles are reported right-positive degrees relative to the current heading.
Turning bucket: |gt_rel| > 30 deg.  Also logs CWP angle-entropy / top-1 prob
at every point (the look-around trigger signal candidates).

Run inside the habitat conda env from integrations/v3:
  python bench/cwp_coverage.py --episodes 0 --out outputs/experiments/cwp_coverage
"""
import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(V3))
sys.path.insert(0, V3)
sys.path.insert(0, os.path.join(V3, "bench"))

VAL_UNSEEN = "/data/pengyh/workspace/habitat/data/datasets/vln/mp3d/r2r/v1/val_unseen/val_unseen.json.gz"
EVAL_SET = os.path.join(V3, "eval_sets", "val_unseen_200.txt")
TOL_DEG = 30.0
FRONT_DEG = 45.0


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def world_yaw_deg(v):
    """Habitat world yaw of horizontal direction v, right-positive degrees.
    (agent forward -z, +x right; matches gen_som_dataset's convention)"""
    return math.degrees(math.atan2(v[0], -v[2]))


def quat_yaw_deg(sr):
    import quaternion
    q = quaternion.quaternion(sr[3], sr[0], sr[1], sr[2])
    fwd = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
    return world_yaw_deg(fwd)


def hit_stats(rel_degs, gt_rel):
    if not rel_degs:
        return {"hit": 0, "min_err": None, "n": 0}
    errs = [abs(wrap180(r - gt_rel)) for r in rel_degs]
    return {"hit": int(min(errs) <= TOL_DEG), "min_err": round(min(errs), 1), "n": len(rel_degs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="0 = all 200")
    ap.add_argument("--spacing-m", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/experiments/cwp_coverage"))
    ap.add_argument("--pitch-deg", type=float, default=-15.0, help="deployment agent view pitch (for fo)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    import habitat_sim
    import gen_som_dataset as gsd  # reuses the deployment candidates module (slots shim)
    from waypoint_cwp.pano_render import make_sim, render_ring
    from waypoint_cwp.predictor import CWPPredictor
    import habitat_sim.utils.common as huc

    ids = [l.split(",")[0].strip() for l in open(EVAL_SET)
           if l.strip() and not l.startswith("#")]
    cats = {l.split(",")[0].strip(): l.split(",")[1].strip() for l in open(EVAL_SET)
            if l.strip() and not l.startswith("#") and "," in l}
    if args.episodes:
        ids = ids[: args.episodes]
    data = json.load(gzip.open(VAL_UNSEEN))
    eps = {str(e["episode_id"]): e for e in data["episodes"] if str(e["episode_id"]) in set(ids)}

    by_scene = {}
    for eid in ids:
        if eid in eps:
            by_scene.setdefault(eps[eid]["scene_id"], []).append(eps[eid])

    pred = CWPPredictor()
    intr = gsd.Intr(640, 480, 90.0)
    rows_f = open(os.path.join(args.out, "points.jsonl"), "w")
    n_pts = 0
    t_start = time.time()

    for scene, scene_eps in sorted(by_scene.items()):
        try:
            sim = make_sim(scene, agent_view_pitch_deg=args.pitch_deg)
        except Exception as exc:
            print(f"scene {scene}: {type(exc).__name__}: {exc}", flush=True)
            continue
        pf = sim.pathfinder
        agent = sim.get_agent(0)
        for e in scene_eps:
            ref = [np.asarray(p, dtype=np.float64) for p in e["reference_path"]]
            pts = []
            for a, b in zip(ref[:-1], ref[1:]):
                seg = np.linalg.norm(b - a)
                n = max(1, int(seg / args.spacing_m))
                for k in range(n):
                    pts.append(a + (b - a) * (k / n))
            pts.append(ref[-1])

            prev_ring = None  # (rgbs, deps, heading_deg)
            for idx in range(len(pts) - 1):
                pos = pf.snap_point(pts[idx])
                if not np.isfinite(np.asarray(pos)).all():
                    prev_ring = None
                    continue
                pos = np.asarray(pos)
                look = pts[idx + 1] - pos
                if np.linalg.norm(look[[0, 2]]) < 0.15:
                    continue
                gt_yaw = world_yaw_deg(look)
                if idx == 0:
                    face_yaw = quat_yaw_deg(e["start_rotation"])
                else:
                    incoming = pos - pts[idx - 1]
                    if np.linalg.norm(incoming[[0, 2]]) < 0.05:
                        continue
                    face_yaw = world_yaw_deg(incoming)
                gt_rel = wrap180(gt_yaw - face_yaw)
                heading_rad = -math.radians(face_yaw)  # right+ deg -> habitat CCW+ rad

                # --- source 1: full ring CWP -------------------------------
                rgbs, deps = render_ring(sim, pos, heading_rad)
                out_ring = pred.predict(rgbs, deps)
                ring_rel = [-w["heading_deg"] for w in out_ring["waypoints"]]  # CCW+ -> right+

                # --- source 3: stale ring (prev ring aligned + fresh slot0) -
                stale = None
                if prev_ring is not None:
                    p_rgbs, p_deps, p_face = prev_ring
                    shift = int(round(wrap180(p_face - face_yaw) / 30.0)) % 12
                    s_rgbs = [p_rgbs[(j + shift) % 12] for j in range(12)]
                    s_deps = [p_deps[(j + shift) % 12] for j in range(12)]
                    f_rgb, f_dep = render_ring(sim, pos, heading_rad, num_slots=1)
                    s_rgbs[0], s_deps[0] = f_rgb[0], f_dep[0]
                    out_stale = pred.predict(s_rgbs, s_deps)
                    stale = [-w["heading_deg"] for w in out_stale["waypoints"]]

                # --- source 2: floor-openings (deployment monocular) --------
                state = habitat_sim.AgentState()
                state.position = pos
                state.rotation = huc.quat_from_angle_axis(
                    math.radians(-face_yaw), np.array([0.0, 1.0, 0.0]))
                agent.set_state(state, reset_sensors=True)
                obs = sim.get_sensor_observations()
                depth = np.asarray(obs["fo_depth"], dtype=np.float64)
                sensor = agent.get_state().sensor_states["fo_rgb"]
                cam = np.eye(4)
                cam[:3, :3] = np.asarray(huc.quat_to_magnum(sensor.rotation).to_matrix())
                cam[:3, 3] = np.asarray(sensor.position)
                fmask = gsd.floor_mask_from(depth, cam, intr=intr)
                try:
                    cands = gsd.cand_mod.floor_openings(depth, fmask, intr, cam)
                    cands = [c for c in cands if pf.is_navigable(
                        [c.world_xyz[0], float(pos[1]), c.world_xyz[2]], 0.5)]
                except Exception:
                    cands = []
                fo_rel = [c.bearing_deg for c in cands]

                row = {
                    "ep": str(e["episode_id"]), "cat": cats.get(str(e["episode_id"]), "?"),
                    "idx": idx, "gt_rel": round(gt_rel, 1),
                    "turning": int(abs(gt_rel) > TOL_DEG),
                    "ring": hit_stats(ring_rel, gt_rel),
                    "fo": hit_stats(fo_rel, gt_rel),
                    "stale": hit_stats(stale, gt_rel) if stale is not None else None,
                    "front45": int(abs(gt_rel) <= FRONT_DEG),
                    "entropy": round(out_ring["angle_entropy"], 3),
                    "top1": round(out_ring["angle_top1"], 4),
                }
                rows_f.write(json.dumps(row) + "\n")
                n_pts += 1
                prev_ring = (rgbs, deps, face_yaw)
        rows_f.flush()
        sim.close()
        print(f"scene {os.path.basename(scene)} done, {n_pts} pts, {time.time()-t_start:.0f}s",
              flush=True)
    rows_f.close()
    print("total points:", n_pts)


if __name__ == "__main__":
    main()
