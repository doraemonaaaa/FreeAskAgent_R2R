"""Pano-hop actor: CWP look-around candidates + VLM chooser + cached hop target.

Drop-in for ``WaypointActorProcess`` (prepare / act / close, in-process like
AwareVLNActor).  At each decision point it renders a 12-view clockwise ring
through the env's dedicated ``cwp_rgb``/``cwp_depth`` sensors, runs the ported
SmartWay CWP predictor (waypoint_cwp/), projects the surviving candidates into
an annotated panorama strip, and asks a VLM to pick a marker or STOP.  The
chosen waypoint is cached: while the hop is in progress ``act`` returns the
same world point without consulting the VLM, so the runner's existing
follower loop drives it one primitive at a time (zero changes to execution).

Phase 1 is the "always look around at every decision point" upper-bound mode;
the selective trigger policy is layered on later.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import time

import numpy as np
from PIL import Image, ImageDraw

NUM_SLOTS = 12
SLOT_DEG = 30.0
DEPTH_SCALE_M = 10.0
ARRIVE_RADIUS_M = 0.6
STUCK_WINDOW = 14  # > the 12 primitives a 180 deg turn needs, so turning is never "stuck"
STUCK_MIN_PROGRESS_M = 0.1
MIN_SCORE_FRAC = 0.15  # keep candidates scoring >= frac * best score
MIN_TRAVEL_BEFORE_STOP_M = 1.0  # R2R goals are >= 4 m away; a spawn STOP is never right


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _agent_yaw(rotation):
    """Habitat agent rotation (np.quaternion, yaw-only) -> CCW+ yaw radians."""
    import quaternion  # noqa: registered by habitat import

    fwd = quaternion.rotate_vectors(rotation, np.array([0.0, 0.0, -1.0]))
    return math.atan2(-fwd[0], -fwd[2])


def _describe(rel_deg, distance_m):
    a = _wrap180(rel_deg)
    if abs(a) <= 15:
        where = "straight ahead"
    elif abs(a) >= 150:
        where = "behind you"
    else:
        where = "{:.0f}° {}".format(abs(a), "right" if a > 0 else "left")
    return "{}, {:.1f} m away".format(where, distance_m)


def _png_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class PanoHopActor:
    def __init__(self, base_url, model, device="cuda", max_candidates=5, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.want_visuals = False
        self.env = None
        self._predictor = None
        self._device = device
        self._reset_episode_state()

    # -- wiring --------------------------------------------------------------
    def attach_env(self, env):
        self.env = env

    def _predictor_lazy(self):
        if self._predictor is None:
            from waypoint_cwp.predictor import CWPPredictor

            self._predictor = CWPPredictor(device=self._device,
                                           max_predictions=self.max_candidates)
        return self._predictor

    def _reset_episode_state(self):
        self.hop_target = None          # np.ndarray[3] world
        self.hop_started_dist = 0.0
        self.hop_step_count = 0
        self.hop_budget = 0
        self.hop_dists = []             # recent distances to target (stuck check)
        self.start_pos = None           # spawn position (STOP suppression)
        self.hop_history = []           # text lines for the prompt
        self.last_progress = ""
        self.decision_count = 0
        self.abandoned_note = ""
        self.stop_pending = False       # first STOP vote awaiting confirmation

    # -- protocol ------------------------------------------------------------
    def prepare(self, instruction):
        self._reset_episode_state()
        return {"subgoals": []}

    def close(self):
        pass

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world,
            navigable=None, oracle_goal=None):
        state = self.env.sim.get_agent_state()
        pos = np.asarray(state.position, dtype=np.float64)
        if self.start_pos is None:
            self.start_pos = pos.copy()

        # ---- hop in progress: keep steering to the cached target ----------
        if self.hop_target is not None:
            dist = float(np.linalg.norm((self.hop_target - pos)[[0, 2]]))
            self.hop_dists.append(dist)
            self.hop_step_count += 1
            arrived = dist <= ARRIVE_RADIUS_M
            over_budget = self.hop_step_count > self.hop_budget
            stuck = (len(self.hop_dists) > STUCK_WINDOW and
                     self.hop_dists[-STUCK_WINDOW] - dist < STUCK_MIN_PROGRESS_M)
            if not (arrived or over_budget or stuck):
                return self.hop_target.astype(np.float32), {
                    "stop": False,
                    "action_mode": "PANO_HOP",
                    "world_xyz": self.hop_target.tolist(),
                    "pixel_uv": [0, 0],
                    "depth_m": dist,
                    "camera_xyz": pos.tolist(),
                    "debug": {"hop": "follow d={:.2f} step={}/{}".format(
                        dist, self.hop_step_count, self.hop_budget)},
                }
            outcome = ("reached it" if arrived else
                       "gave up (stuck)" if stuck else "gave up (budget)")
            if self.hop_history:
                self.hop_history[-1] += " -> " + outcome
            if not arrived:
                self.abandoned_note = (
                    "Note: the previous waypoint was unreachable; pick a different one.")
            self.hop_target = None

        # ---- decision point: look around ----------------------------------
        yaw = _agent_yaw(state.rotation)
        started = time.perf_counter()
        rgbs, deps = self._render_ring(pos, yaw)
        render_ms = (time.perf_counter() - started) * 1000
        out = self._predictor_lazy().predict(rgbs, deps)
        candidates = self._ground_candidates(out["waypoints"], pos, yaw)
        self.decision_count += 1

        if not candidates:
            # nothing navigable: let the runner's keep-alive turn fire
            return None, {"stop": False, "action_mode": "PANO_HOP",
                          "debug": {"hop": "no candidates"}}

        strip = self._annotate_strip(rgbs, candidates)
        choice, progress, raw, vlm_ms = self._choose(strip, candidates, instruction)
        self.abandoned_note = ""
        if progress:
            self.last_progress = progress

        timings = {"panohop_render_ms": render_ms, "panohop_vlm_ms": vlm_ms,
                   "panohop_decision": self.decision_count}
        visuals = {"som_png": self._encode_png(strip)} if self.want_visuals else None

        travelled = float(np.linalg.norm((pos - self.start_pos)[[0, 2]]))
        if choice is None and travelled < MIN_TRAVEL_BEFORE_STOP_M and candidates:
            # Measured guard, not a heuristic: the task's goals are all >= 4 m
            # from the spawn, so completing the instruction here is impossible.
            choice = max(candidates, key=lambda c: c["score"])
            self.hop_history.append(
                "hop {}: STOP suppressed (has not moved yet)".format(self.decision_count))
        if choice is None and not self.stop_pending:
            # First STOP vote: require a second consecutive confirmation (the
            # runner's keep-alive turn nudges the view before the re-decision).
            self.stop_pending = True
            self.hop_history.append(
                "hop {}: proposed STOP (awaiting confirmation)".format(self.decision_count))
            return None, {"stop": False, "action_mode": "PANO_HOP",
                          "raw_model_response": raw, "timings": timings,
                          **({"visuals": visuals} if visuals else {}),
                          "debug": {"hop": "STOP pending confirmation"}}
        if choice is None:  # confirmed STOP
            self.hop_history.append("hop {}: STOP".format(self.decision_count))
            return None, {"stop": True, "action_mode": "PANO_HOP",
                          "raw_model_response": raw, "timings": timings,
                          **({"visuals": visuals} if visuals else {}),
                          "debug": {"hop": "STOP", "progress": self.last_progress}}

        cand = choice
        self.stop_pending = False
        self.hop_target = np.asarray(cand["world_xyz"], dtype=np.float64)
        self.hop_step_count = 0
        self.hop_dists = []
        self.hop_started_dist = cand["distance_m"]
        self.hop_budget = int(cand["distance_m"] / 0.25) * 2 + 10
        self.hop_history.append("hop {}: went {}".format(
            self.decision_count, _describe(cand["rel_deg"], cand["distance_m"])))
        del self.hop_history[:-12]
        return self.hop_target.astype(np.float32), {
            "stop": False,
            "action_mode": "PANO_HOP",
            "world_xyz": self.hop_target.tolist(),
            "pixel_uv": [0, 0],
            "depth_m": cand["distance_m"],
            "camera_xyz": pos.tolist(),
            "raw_model_response": raw,
            "timings": timings,
            **({"visuals": visuals} if visuals else {}),
            "debug": {"hop": "new marker={} {}".format(
                cand["label"], _describe(cand["rel_deg"], cand["distance_m"])),
                "progress": self.last_progress},
        }

    # -- internals -----------------------------------------------------------
    def _render_ring(self, pos, yaw):
        from habitat_sim.utils.common import quat_from_angle_axis

        rgbs, deps = [], []
        for i in range(NUM_SLOTS):
            rot = quat_from_angle_axis(yaw - math.radians(SLOT_DEG) * i,
                                       np.array([0.0, 1.0, 0.0]))
            obs = self.env.sim.get_observations_at(
                position=pos, rotation=rot, keep_agent_at_new_pose=False)
            rgbs.append(np.asarray(obs["cwp_rgb"])[..., :3].copy())
            d = np.asarray(obs["cwp_depth"], dtype=np.float32)
            deps.append(np.clip(d.reshape(256, 256) / DEPTH_SCALE_M, 0.0, 1.0))
        return rgbs, deps

    def _ground_candidates(self, waypoints, pos, yaw):
        pf = self.env.sim.pathfinder
        out = []
        for w in waypoints:
            world_yaw = yaw + w["heading_rad"]
            tgt = pos + np.array([-math.sin(world_yaw), 0.0, -math.cos(world_yaw)]) \
                * w["distance_m"]
            snapped = pf.snap_point(tgt)
            if not np.isfinite(np.asarray(snapped)).all():
                continue
            snapped = np.asarray(snapped, dtype=np.float64)
            if abs(snapped[1] - pos[1]) > 1.2:  # other floor
                continue
            if float(np.linalg.norm((snapped - pos)[[0, 2]])) < 0.3:
                continue
            rel_deg = -math.degrees(w["heading_rad"])  # CCW+ -> right+
            dup = next((c for c in out if np.linalg.norm(
                (np.asarray(c["world_xyz"]) - snapped)[[0, 2]]) < 0.5), None)
            if dup is not None:
                continue
            out.append({"world_xyz": snapped.tolist(), "rel_deg": rel_deg,
                        "distance_m": w["distance_m"], "score": w["score"]})
        if out:
            best = max(c["score"] for c in out)
            out = [c for c in out if c["score"] >= best * MIN_SCORE_FRAC]
        out.sort(key=lambda c: c["rel_deg"])
        for i, c in enumerate(out, start=1):
            c["label"] = str(i)
        return out

    def _annotate_strip(self, rgbs, candidates):
        tile = rgbs[0].shape[0]
        strip = Image.fromarray(np.concatenate(rgbs, axis=1))
        draw = ImageDraw.Draw(strip)
        f = (tile / 2.0)  # focal for hfov 90: (w/2)/tan(45) = w/2
        for c in candidates:
            # clockwise angle from forward in [0,360)
            cw = (-(-c["rel_deg"]) + 360.0) % 360.0  # rel_deg is right+ = clockwise+
            slot = int((cw + SLOT_DEG / 2.0) // SLOT_DEG) % NUM_SLOTS
            in_tile = _wrap180(cw - slot * SLOT_DEG)
            u = slot * tile + tile / 2.0 + math.tan(math.radians(in_tile)) * f
            v = tile / 2.0 + f * 1.25 / max(0.5, c["distance_m"])
            v = min(v, tile - 14)
            r = 11
            draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 60, 60),
                         outline=(255, 255, 255), width=2)
            draw.text((u - (4 if len(c["label"]) == 1 else 8), v - 7),
                      c["label"], fill=(255, 255, 255))
        for i in range(NUM_SLOTS):
            draw.line([i * tile, 0, i * tile, tile], fill=(255, 255, 255), width=1)
        return strip

    def _choose(self, strip, candidates, instruction):
        options = "\n".join("  {}: {}".format(c["label"], _describe(
            c["rel_deg"], c["distance_m"])) for c in candidates)
        history = "; ".join(self.hop_history[-8:]) or "just started"
        prompt = (
            "You are a robot navigating a building. Follow this instruction:\n"
            '"{}"\n\n'
            "Moves so far: {}.\n"
            "{}"
            "The image is a full 360° look-around: 12 views left-to-right, each "
            "30° further to the RIGHT; the leftmost view faces your current forward "
            "direction, the 7th view faces behind you.\n"
            "Red numbered markers are reachable waypoints:\n{}\n\n"
            "Pick the waypoint that best continues the instruction given the moves "
            "already made. Only answer STOP if the whole instruction is complete and "
            "you are standing at the final described location.\n"
            'Answer with JSON only: {{"progress": "<short clause: what part of the '
            'instruction is already done>", "remaining": "<what is still left to do, '
            'or \\"none\\">", "choice": <marker number or "STOP">}}'
        ).format(instruction.strip(),
                 history,
                 ("Progress so far: {}.\n".format(self.last_progress)
                  if self.last_progress else "")
                 + (self.abandoned_note + "\n" if self.abandoned_note else "")
                 + ("You proposed STOP at the previous decision. Answer STOP again "
                    "ONLY if the instruction is fully complete; otherwise pick a "
                    "waypoint.\n" if self.stop_pending else ""),
                 options)
        started = time.perf_counter()
        raw = self._query(strip, prompt)
        vlm_ms = (time.perf_counter() - started) * 1000
        choice, progress = self._parse(raw, candidates)
        return choice, progress, raw, vlm_ms

    def _parse(self, raw, candidates):
        by_label = {c["label"]: c for c in candidates}
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                progress = str(data.get("progress", "")).strip()[:200]
                remaining = str(data.get("remaining", "")).strip().lower()
                choice = str(data.get("choice", "")).strip().upper()
                if choice == "STOP":
                    if remaining not in ("", "none", "nothing", "n/a", "-", "done"):
                        # The model itself says work remains: not a real STOP.
                        return max(candidates, key=lambda c: c["score"]), progress
                    return None, progress
                if choice in by_label:
                    return by_label[choice], progress
            except (ValueError, TypeError):
                pass
        if re.search(r"\bSTOP\b", raw or ""):
            return None, ""
        for token in re.findall(r"\d+", raw or ""):
            if token in by_label:
                return by_label[token], ""
        # unparseable: take the highest-scoring candidate rather than dying
        return max(candidates, key=lambda c: c["score"]), ""

    def _query(self, strip, prompt):
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 220,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _png_data_url(strip)}},
                {"type": "text", "text": prompt},
            ]}],
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"] or ""

    @staticmethod
    def _encode_png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
