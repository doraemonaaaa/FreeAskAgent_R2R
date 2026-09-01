"""Pano-hop actor: CWP look-around candidates + VLM chooser + cached hop target.

Drop-in for ``WaypointActorProcess`` (prepare / act / close), in-process like
AwareVLNActor.  At each decision point it obtains waypoint candidates from the
ported SmartWay CWP predictor (waypoint_cwp/), grounds them on the navmesh,
draws them as numbered markers on a view strip, and asks a VLM to pick a
marker or STOP.  The chosen world point is cached: while the hop is in
progress ``act`` returns the same point without consulting the VLM, so the
runner's follower loop drives it one primitive at a time (execution loop
unchanged).  Measured on the 200-set: SR 0.150 / SPL 0.134 / oracle 0.250
(always mode, qwen3-vl-8b) vs the 0.095 primitive-step baseline.

Modes (--panohop-mode):
  always     look around (12-view ring render) at every decision point.
             The upper-bound reference; the headline configuration.
  selective  monocular by default: fresh forward 3 slots + the realigned
             stale ring feed CWP, only forward +-45 deg candidates offered.
             A ring fires on: first decision, failed hop, no forward
             candidate, turn word in the first instruction step, 4 mono
             decisions in a row.  Measured: SR 0.125 at 7.0 look-arounds/ep
             (always: 10.5) - navigation reach intact (oracle 0.245).

Ablation switches, all default OFF (measured worse; kept reproducible):
  --panohop-stop-verify   STOP votes go through a verification call with a
                          3-consecutive-vote override (200d SR 0.065 / 200e
                          0.109 vs bare STOP 0.150).
  (backtrack)             B-options after a failed hop; chosen 0/2101 times
                          zero-shot (200bt2, SR-neutral).  Left active since
                          it is never taken; part of the training action
                          space later.

The load-bearing prompt line is the model's own "progress" clause fed back
at every decision - dropping it collapsed oracle 0.250 -> 0.095 (docs
2026-08-30 §21.1).  VLN_PANOHOP_STRIP_DIR=<dir> dumps each decision's
annotated strip.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
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
MONO_ARC_DEG = 45.0  # candidates offered on a monocular decision
MONO_STREAK_LIMIT = 4  # forced ring after this many mono decisions


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
    def __init__(self, base_url, model, device="cuda", max_candidates=5,
                 timeout=300, mode="always", stop_verify=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_candidates = max_candidates
        self.mode = mode
        self.stop_verify = stop_verify
        self.want_visuals = False
        self.strip_dir = os.environ.get("VLN_PANOHOP_STRIP_DIR") or None
        self.episode_counter = 0
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
        self.hop_step_count = 0
        self.hop_budget = 0
        self.hop_dists = []             # recent distances to target (stuck check)
        self.start_pos = None           # spawn position (STOP travel guard)
        self.hop_history = []           # text lines for the prompt
        self.last_progress = ""         # model's own progress clause, fed back
        self.decision_count = 0
        self.abandoned_note = ""
        self.stop_votes = 0             # consecutive STOP votes (verifier cap)
        self.last_ring = None           # (rgbs, deps, yaw_rad) of last look-around
        self.lookaround_count = 0
        self.force_look = True          # first decision always looks around
        self.nodes = []                 # visited look-around nodes (backtrack)
        self.offer_backtrack = False    # only after a failed hop (offering
                                        # B-options every decision collapsed
                                        # choice quality: oracle 0.250 -> 0.095)
        self.mono_streak = 0            # mono decisions since the last ring
        self.subgoals = []              # instruction steps (selective trigger)

    # -- protocol ------------------------------------------------------------
    def prepare(self, instruction):
        self._reset_episode_state()
        self.episode_counter += 1
        self.instruction_text = instruction.strip()
        self.subgoals = self._split_subgoals(instruction)
        return {"subgoals": [
            {"subgoal_id": i + 1, "description": s, "completion_criteria": ""}
            for i, s in enumerate(self.subgoals)]}

    def close(self):
        pass

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world,
            navigable=None, oracle_goal=None, cwp_candidates=None):
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
                self.force_look = True  # a failed hop means the plan was wrong
                self.offer_backtrack = True
            self.hop_target = None

        # ---- decision point ------------------------------------------------
        yaw = _agent_yaw(state.rotation)
        self.decision_count += 1
        first_step = self.subgoals[0] if self.subgoals else ""
        full_look = (self.mode == "always" or self.force_look
                     or self.last_ring is None
                     or self._wants_turn(first_step)
                     or self.mono_streak >= MONO_STREAK_LIMIT)
        started = time.perf_counter()
        candidates, strip = self._propose(pos, yaw, full_look)
        render_ms = (time.perf_counter() - started) * 1000

        if not candidates:
            if not full_look:
                # nothing ahead: looking around is the measured answer
                full_look = True
                candidates, strip = self._propose(pos, yaw, True)
            if not candidates:
                return None, {"stop": False, "action_mode": "PANO_HOP",
                              "debug": {"hop": "no candidates"}}

        back = (self._backtrack_options(pos)
                if (full_look and self.offer_backtrack) else [])
        self.offer_backtrack = False
        choice, progress, raw, vlm_ms = self._choose(
            strip, candidates + back, panorama=full_look)
        self.abandoned_note = ""
        if progress:
            self.last_progress = progress
        print("PANOHOP dec={} {} cands={} choice={} la={} prog={!r}".format(
            self.decision_count, "ring" if full_look else "mono",
            len(candidates),
            choice["label"] if isinstance(choice, dict) else "STOP",
            self.lookaround_count, (progress or "")[:60]), flush=True)

        if self.strip_dir:
            os.makedirs(self.strip_dir, exist_ok=True)
            strip.save(os.path.join(self.strip_dir, "ep{:03d}_dec{:02d}_{}.png".format(
                self.episode_counter, self.decision_count,
                choice["label"] if isinstance(choice, dict) else "STOP")))
        timings = {"panohop_render_ms": render_ms, "panohop_vlm_ms": vlm_ms,
                   "panohop_decision": self.decision_count,
                   "panohop_lookarounds": self.lookaround_count}
        visuals = {"som_png": self._encode_png(strip)} if self.want_visuals else None

        travelled = float(np.linalg.norm((pos - self.start_pos)[[0, 2]]))
        if choice is None and travelled < MIN_TRAVEL_BEFORE_STOP_M and candidates:
            # Measured guard, not a heuristic: the task's goals are all >= 4 m
            # from the spawn, so completing the instruction here is impossible.
            choice = max(candidates, key=lambda c: c["score"])
            self.hop_history.append(
                "hop {}: STOP suppressed (has not moved yet)".format(self.decision_count))
        if choice is None and self.stop_verify:
            # ABLATION (--panohop-stop-verify, default OFF: 200d SR 0.065 /
            # 200e 0.109 both lost to bare STOP 0.150). A STOP vote is checked
            # by an independently-framed verification call; 3 consecutive
            # votes override the verifier.
            self.stop_votes += 1
            ok, evidence = self._verify_stop(strip)
            print("PANOHOP stop-verify dec={} ok={} votes={} ev={!r}".format(
                self.decision_count, ok, self.stop_votes, evidence[:80]), flush=True)
            if not ok and self.stop_votes < 3 and candidates:
                self.hop_history.append(
                    "hop {}: STOP rejected by verification".format(self.decision_count))
                # Instruction-aware redirect: re-ask the chooser with the
                # verifier's evidence (a bare best-score hop is instruction-
                # blind and measured to cause wandering, 200d).
                self.abandoned_note = (
                    "A verification check says you are NOT at the final location "
                    "yet ({}). Continue following the instruction.".format(
                        evidence[:140]))
                choice2, progress2, raw2, vlm2 = self._choose(
                    strip, candidates, panorama=True)
                vlm_ms += vlm2
                raw += " || redirect: " + raw2
                if isinstance(choice2, dict):
                    choice = choice2
                    if progress2:
                        progress = progress2
                else:
                    choice = max(candidates, key=lambda c: c["score"])
                self.abandoned_note = ""
        else:
            self.stop_votes = 0
        if choice is None:  # STOP
            self.hop_history.append("hop {}: STOP".format(self.decision_count))
            return None, {"stop": True, "action_mode": "PANO_HOP",
                          "raw_model_response": raw, "timings": timings,
                          **({"visuals": visuals} if visuals else {}),
                          "debug": {"hop": "STOP", "progress": self.last_progress,
                                    "lookarounds": self.lookaround_count}}

        cand = choice
        self.hop_target = np.asarray(cand["world_xyz"], dtype=np.float64)
        self.hop_step_count = 0
        self.hop_dists = []
        budget_factor = 4 if cand.get("kind") == "backtrack" else 2
        self.hop_budget = int(cand["distance_m"] / 0.25) * budget_factor + 10
        self.hop_history.append("hop {}: {}".format(
            self.decision_count,
            "backtracked to an earlier position" if cand.get("kind") == "backtrack"
            else "went " + _describe(cand["rel_deg"], cand["distance_m"])))
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
                "progress": self.last_progress,
                "lookarounds": self.lookaround_count},
        }

    # -- candidate proposal ---------------------------------------------------
    def _propose(self, pos, yaw, full_look):
        """Returns (candidates, annotated strip) for a ring or mono decision."""
        if full_look:
            rgbs, deps = self._render_ring(pos, yaw)
            self.last_ring = ([r.copy() for r in rgbs],
                              [d.copy() for d in deps], yaw)
            self.lookaround_count += 1
            self.force_look = False
            self.mono_streak = 0
            self._remember_node(pos, rgbs[0])
            out = self._predictor_lazy().predict(rgbs, deps)
            candidates = self._ground_candidates(out["waypoints"], pos, yaw)
            return candidates, self._annotate(rgbs, candidates, slots=range(NUM_SLOTS))
        # monocular: fresh forward arc (slots 11,0,1), stale ring elsewhere
        p_rgbs, p_deps, p_yaw = self.last_ring
        shift = int(round(_wrap180(math.degrees(p_yaw - yaw)) / SLOT_DEG)) % NUM_SLOTS
        rgbs = [p_rgbs[(j + shift) % NUM_SLOTS] for j in range(NUM_SLOTS)]
        deps = [p_deps[(j + shift) % NUM_SLOTS] for j in range(NUM_SLOTS)]
        for slot in (11, 0, 1):
            r, d = self._render_slot(pos, yaw - math.radians(SLOT_DEG) * slot)
            rgbs[slot], deps[slot] = r, d
        out = self._predictor_lazy().predict(rgbs, deps)
        candidates = self._ground_candidates(out["waypoints"], pos, yaw)
        candidates = [c for c in candidates if abs(c["rel_deg"]) <= MONO_ARC_DEG]
        for i, c in enumerate(candidates, start=1):
            c["label"] = str(i)
        self.mono_streak += 1
        return candidates, self._annotate(
            [rgbs[11], rgbs[0], rgbs[1]], candidates, slots=(11, 0, 1))

    def _remember_node(self, pos, fwd_tile):
        for node in self.nodes:
            if np.linalg.norm((np.asarray(node["pos"]) - pos)[[0, 2]]) < 1.0:
                node["pos"] = pos.copy()
                node["tile"] = fwd_tile.copy()
                node["idx"] = self.decision_count
                return
        self.nodes.append({"pos": pos.copy(), "tile": fwd_tile.copy(),
                           "idx": self.decision_count})

    def _backtrack_options(self, pos, max_back=3, min_dist=1.5):
        opts = []
        for node in reversed(self.nodes):
            d = float(np.linalg.norm((np.asarray(node["pos"]) - pos)[[0, 2]]))
            if d < min_dist:
                continue
            opts.append({"label": "B{}".format(len(opts) + 1),
                         "world_xyz": np.asarray(node["pos"]).tolist(),
                         "rel_deg": 0.0, "distance_m": d, "score": 0.0,
                         "kind": "backtrack", "tile": node["tile"],
                         "idx": node["idx"]})
            if len(opts) >= max_back:
                break
        return opts

    def _render_ring(self, pos, yaw):
        rgbs, deps = [], []
        for i in range(NUM_SLOTS):
            r, d = self._render_slot(pos, yaw - math.radians(SLOT_DEG) * i)
            rgbs.append(r)
            deps.append(d)
        return rgbs, deps

    def _render_slot(self, pos, world_yaw):
        from habitat_sim.utils.common import quat_from_angle_axis

        rot = quat_from_angle_axis(world_yaw, np.array([0.0, 1.0, 0.0]))
        obs = self.env.sim.get_observations_at(
            position=pos, rotation=rot, keep_agent_at_new_pose=False)
        rgb = np.asarray(obs["cwp_rgb"])[..., :3].copy()
        dep = np.clip(np.asarray(obs["cwp_depth"], dtype=np.float32)
                      .reshape(256, 256) / DEPTH_SCALE_M, 0.0, 1.0)
        return rgb, dep

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
            if any(np.linalg.norm((np.asarray(c["world_xyz"]) - snapped)[[0, 2]]) < 0.5
                   for c in out):
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

    def _annotate(self, tiles, candidates, slots):
        """Draw numbered markers on a strip of the given ring slots."""
        tile = tiles[0].shape[0]
        strip = Image.fromarray(np.concatenate(tiles, axis=1))
        draw = ImageDraw.Draw(strip)
        f = tile / 2.0  # focal for hfov 90
        slot_center = {s: idx for idx, s in enumerate(slots)}
        for c in candidates:
            cw = (c["rel_deg"] + 360.0) % 360.0  # right+ == clockwise slot angle
            slot = int((cw + SLOT_DEG / 2.0) // SLOT_DEG) % NUM_SLOTS
            if slot not in slot_center:
                continue
            in_tile = _wrap180(cw - slot * SLOT_DEG)
            u = slot_center[slot] * tile + tile / 2.0 \
                + math.tan(math.radians(in_tile)) * f
            v = tile / 2.0 + f * 1.25 / max(0.5, c["distance_m"])
            v = min(v, tile - 14)
            r = 11
            draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 60, 60),
                         outline=(255, 255, 255), width=2)
            draw.text((u - (4 if len(c["label"]) == 1 else 8), v - 7),
                      c["label"], fill=(255, 255, 255))
        for i in range(1, len(tiles)):
            draw.line([i * tile, 0, i * tile, tile], fill=(255, 255, 255), width=1)
        return strip

    # -- chooser --------------------------------------------------------------
    def _choose(self, strip, candidates, panorama):
        lines, extras = [], []
        for c in candidates:
            if c.get("kind") == "backtrack":
                extras.append(c["tile"])
                lines.append(
                    "  {}: go back to a place you visited earlier (decision {}, "
                    "~{:.1f} m away; its forward view is extra image {})".format(
                        c["label"], c["idx"], c["distance_m"], len(extras)))
            else:
                lines.append("  {}: {}".format(
                    c["label"], _describe(c["rel_deg"], c["distance_m"])))
        options = "\n".join(lines)
        history = "; ".join(self.hop_history[-8:]) or "just started"
        image_desc = (
            "The image is a full 360° look-around: 12 views left-to-right, each "
            "30° further to the RIGHT; the leftmost view faces your current forward "
            "direction, the 7th view faces behind you.\n" if panorama else
            "The image shows your forward 90° arc: three views (30° left, ahead, "
            "30° right). You can NOT see what is beside or behind you.\n")
        prompt = (
            "You are a robot navigating a building. Follow this instruction:\n"
            '"{}"\n\n'
            "Moves so far: {}.\n"
            "{}"
            "{}"
            "Red numbered markers are reachable waypoints:\n{}\n\n"
            "Pick the waypoint that best continues the instruction given the moves "
            "already made. Only answer STOP if the whole instruction is complete and "
            "you are standing at the final described location.\n"
            'Answer with JSON only: {{"progress": "<short clause: what part of the '
            'instruction is already done>", "choice": <marker number or "STOP">}}'
        ).format(self.instruction_text,
                 history,
                 ("Progress so far: {}.\n".format(self.last_progress)
                  if self.last_progress else "")
                 + (self.abandoned_note + "\n" if self.abandoned_note else ""),
                 image_desc,
                 options)
        started = time.perf_counter()
        raw = self._query(strip, prompt, extras=extras)
        vlm_ms = (time.perf_counter() - started) * 1000
        choice, progress = self._parse(raw, candidates)
        return choice, progress, raw, vlm_ms

    _TURN_WORDS = re.compile(
        r"\b(turn|left|right|around|exit|enter|door|doorway|stairs|stairway|"
        r"upstairs|downstairs|behind|corner|hallway|out of)\b", re.I)

    def _wants_turn(self, step_text):
        # Only the NEXT action matters: match the first clause, not the whole
        # tail (turn-heavy instructions otherwise force a ring every time).
        if not step_text:
            return False
        first = re.split(r"[,.;]| then | and ", step_text, maxsplit=1)[0]
        return bool(self._TURN_WORDS.search(first))

    def _parse(self, raw, candidates):
        by_label = {c["label"]: c for c in candidates}
        progress = ""
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                progress = str(data.get("progress", "")).strip()[:200]
                choice = str(data.get("choice", "")).strip().upper()
                if choice == "STOP":
                    return None, progress
                if choice in by_label:
                    return by_label[choice], progress
            except (ValueError, TypeError):
                pass
        if re.search(r"\bSTOP\b", raw or ""):
            return None, progress
        found = re.search(r'choice[\"\s:]*([B]?[0-9]+)', raw or "")
        if found and found.group(1) in by_label:
            return by_label[found.group(1)], progress
        for token in re.findall(r"\d+", raw or ""):
            if token in by_label:
                return by_label[token], progress
        # unparseable: take the highest-scoring candidate rather than dying
        return max(candidates, key=lambda c: c["score"]), progress

    def _verify_stop(self, strip):
        prompt = (
            "You are a robot that was following this instruction:\n"
            '"{}"\n\n'
            "You believe you have finished. The image shows your surroundings "
            "from your current position.\n"
            "Verify strictly: are you standing AT the final location the "
            "instruction describes (not merely seeing it in the distance)?\n"
            'Answer with JSON only: {{"at_final_location": true or false, '
            '"evidence": "<what in the view proves or disproves it>"}}'
        ).format(self.instruction_text)
        try:
            raw = self._query(strip, prompt)
            found = re.search(r"\{.*\}", raw or "", re.DOTALL)
            data = json.loads(found.group(0))
            return bool(data.get("at_final_location")), str(data.get("evidence", ""))
        except Exception as exc:
            return True, "verifier error: {}".format(exc)  # fail open: allow STOP

    def _split_subgoals(self, instruction):
        prompt = (
            "Split this navigation instruction into an ordered list of short "
            "atomic steps (one motion or one 'stop at X' each). Answer with a "
            "JSON array of strings only.\n\nInstruction: {!r}".format(
                instruction.strip()))
        try:
            raw = self._query_text(prompt)
            found = re.search(r"\[.*\]", raw or "", re.DOTALL)
            steps = [str(s).strip() for s in json.loads(found.group(0))
                     if str(s).strip()]
            if steps:
                return steps[:12]
        except Exception:
            pass
        parts = [p.strip() for p in re.split(r"[.;\n]|\bthen\b", instruction)
                 if p.strip()]
        return parts[:12] or [instruction.strip()]

    # -- transport ------------------------------------------------------------
    def _query_text(self, prompt):
        body = json.dumps({
            "model": self.model, "temperature": 0.0, "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        return self._post(body)

    def _query(self, strip, prompt, extras=None):
        content = [{"type": "image_url",
                    "image_url": {"url": _png_data_url(strip)}}]
        for tile in (extras or []):
            content.append({"type": "image_url", "image_url": {
                "url": _png_data_url(Image.fromarray(np.asarray(tile)))}})
        content.append({"type": "text", "text": prompt})
        body = json.dumps({
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 220,
            "messages": [{"role": "user", "content": content}],
        }).encode("utf-8")
        return self._post(body)

    def _post(self, body, attempts=4):
        """One transient 502 from the server must not kill a whole eval shard
        (measured: panohop200e rank_3 died at episode 24/25)."""
        import urllib.request

        for attempt in range(attempts):
            try:
                request = urllib.request.Request(
                    self.base_url + "/chat/completions", data=body,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    payload = json.load(r)
                return payload["choices"][0]["message"]["content"] or ""
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt)

    @staticmethod
    def _encode_png(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
