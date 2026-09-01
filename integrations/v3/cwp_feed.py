"""Runner-side CWP candidate feed for the ORIGINAL waypoint actor.

Supplies navmesh-grounded waypoint candidates from the ported SmartWay CWP
predictor to the worker (which replaces its floor-openings generator with
them, keeping captioner/judge/spatial memory untouched).

Two operating points, mirroring the measured selective-look design:
  step(...)     normal (monocular) step: fresh forward 3 slots + the stale
                ring realigned to the current heading feed CWP; only
                candidates within +-45 deg of forward are returned
                (measured: 92-93% coverage on straight decision points).
  ring(...)     look-around (PREVIEW) moment: full fresh 12-slot ring;
                all candidates returned and the ring is cached as the new
                stale ring.

Every candidate: world_xyz (navmesh-snapped), bearing_deg (right+ relative
to the agent's heading), distance_m, score, and pixel_uv projected into the
agent's forward camera (None when outside the view).
"""
from __future__ import annotations

import math

import numpy as np

NUM_SLOTS = 12
SLOT_DEG = 30.0
DEPTH_SCALE_M = 10.0
MONO_ARC_DEG = 45.0
MIN_SCORE_FRAC = 0.15


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _agent_yaw(rotation):
    import quaternion  # noqa: registered by habitat import

    fwd = quaternion.rotate_vectors(rotation, np.array([0.0, 0.0, -1.0]))
    return math.atan2(-fwd[0], -fwd[2])


class CwpCandidateFeed:
    def __init__(self, env, device="cuda", max_predictions=5):
        from waypoint_cwp.predictor import CWPPredictor

        self.env = env
        self.predictor = CWPPredictor(device=device,
                                      max_predictions=max_predictions)
        self.last_ring = None  # (rgbs, deps, yaw_rad)

    def reset(self):
        self.last_ring = None

    # -- public ---------------------------------------------------------------
    def step(self, intrinsics=None, camera_to_world=None):
        """Forward-sector candidates for a normal (monocular) step."""
        pos, yaw = self._pose()
        if self.last_ring is None:
            rgbs, deps = self._render_ring(pos, yaw)
            self.last_ring = (rgbs, deps, yaw)
        else:
            p_rgbs, p_deps, p_yaw = self.last_ring
            shift = int(round(_wrap180(math.degrees(p_yaw - yaw)) / SLOT_DEG)) % NUM_SLOTS
            rgbs = [p_rgbs[(j + shift) % NUM_SLOTS] for j in range(NUM_SLOTS)]
            deps = [p_deps[(j + shift) % NUM_SLOTS] for j in range(NUM_SLOTS)]
            for slot in (11, 0, 1):
                r, d = self._render_slot(pos, yaw - math.radians(SLOT_DEG) * slot)
                rgbs[slot], deps[slot] = r, d
        cands = self._candidates(rgbs, deps, pos, yaw)
        cands = [c for c in cands if abs(c["bearing_deg"]) <= MONO_ARC_DEG]
        self._project(cands, intrinsics, camera_to_world)
        return cands

    def ring(self, intrinsics=None, camera_to_world=None):
        """Full look-around candidates (PREVIEW moment); refreshes the ring."""
        pos, yaw = self._pose()
        rgbs, deps = self._render_ring(pos, yaw)
        self.last_ring = (rgbs, deps, yaw)
        cands = self._candidates(rgbs, deps, pos, yaw)
        self._project(cands, intrinsics, camera_to_world)
        return cands

    # -- internals ------------------------------------------------------------
    def _pose(self):
        state = self.env.sim.get_agent_state()
        return np.asarray(state.position, dtype=np.float64), _agent_yaw(state.rotation)

    def _render_slot(self, pos, world_yaw):
        from habitat_sim.utils.common import quat_from_angle_axis

        rot = quat_from_angle_axis(world_yaw, np.array([0.0, 1.0, 0.0]))
        obs = self.env.sim.get_observations_at(
            position=pos, rotation=rot, keep_agent_at_new_pose=False)
        rgb = np.asarray(obs["cwp_rgb"])[..., :3].copy()
        dep = np.clip(np.asarray(obs["cwp_depth"], dtype=np.float32)
                      .reshape(256, 256) / DEPTH_SCALE_M, 0.0, 1.0)
        return rgb, dep

    def _render_ring(self, pos, yaw):
        rgbs, deps = [], []
        for i in range(NUM_SLOTS):
            r, d = self._render_slot(pos, yaw - math.radians(SLOT_DEG) * i)
            rgbs.append(r)
            deps.append(d)
        return rgbs, deps

    def _candidates(self, rgbs, deps, pos, yaw):
        out = self.predictor.predict(rgbs, deps)
        pf = self.env.sim.pathfinder
        cands = []
        for w in out["waypoints"]:
            world_yaw = yaw + w["heading_rad"]
            tgt = pos + np.array([-math.sin(world_yaw), 0.0,
                                  -math.cos(world_yaw)]) * w["distance_m"]
            snapped = pf.snap_point(tgt)
            if not np.isfinite(np.asarray(snapped)).all():
                continue
            snapped = np.asarray(snapped, dtype=np.float64)
            if abs(snapped[1] - pos[1]) > 1.2:
                continue
            if float(np.linalg.norm((snapped - pos)[[0, 2]])) < 0.3:
                continue
            if any(np.linalg.norm((np.asarray(c["world_xyz"]) - snapped)[[0, 2]]) < 0.5
                   for c in cands):
                continue
            cands.append({
                "world_xyz": snapped.tolist(),
                "bearing_deg": -math.degrees(w["heading_rad"]),  # right+
                "distance_m": w["distance_m"],
                "score": w["score"],
                "pixel_uv": None,
            })
        if cands:
            best = max(c["score"] for c in cands)
            cands = [c for c in cands if c["score"] >= best * MIN_SCORE_FRAC]
        cands.sort(key=lambda c: c["bearing_deg"])
        return cands

    def _project(self, cands, intrinsics, camera_to_world):
        """Fill pixel_uv for candidates visible in the agent's forward camera."""
        if intrinsics is None or camera_to_world is None:
            return
        K = np.asarray(intrinsics, dtype=np.float64)
        world_to_cam = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
        for c in cands:
            X = np.ones(4)
            X[:3] = c["world_xyz"]
            p = world_to_cam @ X
            # habitat camera looks down -z
            if p[2] >= -0.05:
                continue
            u = K[0, 0] * (p[0] / -p[2]) + K[0, 2]
            v = K[1, 1] * (-p[1] / -p[2]) + K[1, 2]
            if 0 <= u < 2 * K[0, 2] + 2 and 0 <= v < 2 * K[1, 2] + 2:
                c["pixel_uv"] = [int(round(u)), int(round(v))]
