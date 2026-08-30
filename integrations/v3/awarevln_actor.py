"""AwareVLN as the actor inside this runner.

AwareVLN (NaVILA-style VILA, Llama-3 8B + SigLIP) is a navigation policy, not a
general VLM: whatever the prompt, it answers ``<BEGIN_OF_ACTION> ...`` or
``<BEGIN_OF_REASONING> ...``.  It therefore cannot serve the waypoint worker's
JSON prompts; instead this actor speaks its native protocol (8 ordered frames +
instruction -> forward / turn / stop) and maps the answer onto the runner's
existing primitives.  The prompt, frame sampling, reasoning loop and action
parsing mirror ``AwareVLN/evaluation/vlnce_baselines/awarevln_trainer.py``.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.request

import numpy as np
from PIL import Image

# The llama_3 conversation template's own system text; sent verbatim so the
# served prompt is byte-identical to the reference evaluation.
LLAMA3_SYSTEM = (
    "You are a helpful language and vision assistant. You are able to understand the visual "
    "content that the user provides, and assist the user with a variety of tasks using natural language."
)
REASON_TOKEN = "<BEGIN_OF_REASONING>"
ACT_TOKEN = "<BEGIN_OF_ACTION>"
MAX_CONSECUTIVE_REASONING = 3


def sample_and_pad_images(frames, num_frames=8, size=(512, 512)):
    frames = list(frames)
    while len(frames) < num_frames:
        frames.insert(0, Image.new("RGB", size, color=(0, 0, 0)))
    latest = frames[-1]
    sampled = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    return [frames[i] for i in sampled] + [latest]


def extract_reasoning(text):
    if not text:
        return ""
    text = re.sub(r"(new\s*reasoning\s*is[:\s]*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[?ph_reasoning_token_?\]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|?end_of_text\|?>", "", text, flags=re.IGNORECASE)
    for token in ("<begin_of_reasoning>", "<begin_of_action>", "<end_of_reasoning>"):
        text = re.sub(token, "", text, flags=re.IGNORECASE)
    return text.strip(" '\"\n\t")


def parse_action(text):
    """Return (kind, amount): ('stop', 0) | ('forward', cm) | ('left'|'right', deg)."""
    if re.search(r"\bstop\b", text, re.IGNORECASE):
        return "stop", 0
    if re.search(r"move forward", text, re.IGNORECASE):
        match = re.search(r"move forward (\d+) cm", text)
        distance = int(match.group(1)) if match else 25
        if distance % 25:
            distance = min([25, 50, 75], key=lambda x: abs(x - distance))
        return "forward", max(25, distance)
    for kind in ("left", "right"):
        if re.search(r"turn {}".format(kind), text, re.IGNORECASE):
            match = re.search(r"turn {} (\d+) degree".format(kind), text)
            degree = int(match.group(1)) if match else 15
            if degree % 15:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
            return kind, max(15, degree)
    return "forward", 25


def _png_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class AwareVLNActor:
    """Drop-in for ``WaypointActorProcess``: ``prepare`` / ``act`` / ``observe`` / ``close``."""

    def __init__(self, base_url, model="awarevln", num_frames=8, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_frames = num_frames
        self.timeout = timeout
        self.want_visuals = False
        self.history = []
        self.step_index = 0
        self.last_reasoning = ""
        self.last_reason_step = 0

    # -- protocol -----------------------------------------------------------
    def prepare(self, instruction):
        self.history = []
        self.step_index = 0
        self.last_reasoning = ""
        self.last_reason_step = 0
        return {"subgoals": []}

    def observe(self, rgb):
        """Record a frame the policy was not consulted on (queued primitives)."""
        self.history.append(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB"))
        self.step_index += 1

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world, navigable=None, oracle_goal=None):
        current = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        frames = sample_and_pad_images(self.history + [current], self.num_frames, current.size)
        started = time.perf_counter()
        raw_outputs, reasoning_rounds = [], 0
        while True:
            raw = self._query(frames, instruction)
            raw_outputs.append(raw)
            if raw.startswith(REASON_TOKEN) and reasoning_rounds < MAX_CONSECUTIVE_REASONING:
                self.last_reasoning = extract_reasoning(raw[len(REASON_TOKEN):])
                self.last_reason_step = self.step_index
                reasoning_rounds += 1
                continue
            break
        content = raw[len(ACT_TOKEN):].strip() if raw.startswith(ACT_TOKEN) else raw
        kind, amount = parse_action(content)
        roundtrip_ms = (time.perf_counter() - started) * 1000
        decision = {
            "stop": kind == "stop",
            "action_mode": "AWAREVLN",
            "raw_model_response": " || ".join(raw_outputs),
            "roundtrip_ms": roundtrip_ms,
            "timings": {"awarevln_ms": roundtrip_ms, "reasoning_rounds": reasoning_rounds},
            "debug": {"reasoning": self.last_reasoning, "history_frames": len(self.history)},
        }
        if kind == "forward":
            decision["forward_steps"] = amount // 25
        elif kind in ("left", "right"):
            decision["turn_deg"] = amount if kind == "right" else -amount
        self.history.append(current)
        self.step_index += 1
        if kind == "stop":
            self.step_index = 0
        return None, decision

    def act_on_preview(self, views, instruction):
        raise RuntimeError("AwareVLNActor never requests previews")

    def close(self):
        pass

    # -- model --------------------------------------------------------------
    def _question_parts(self, frames, instruction):
        parts = ["Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations "]
        for frame in frames[:-1]:
            parts.append(frame)
        parts.append(", and current observation ")
        parts.append(frames[-1])
        tail = '. Your assigned task is: "{}". '.format(instruction)
        if self.last_reasoning:
            tail += 'The reasoning from {} steps ago was: "{}". '.format(
                self.step_index - self.last_reason_step, self.last_reasoning
            )
        tail += (
            "Analyze this series of images to decide whether to predict the next action or to perform reasoning. "
            "If action prediction, decide your next action, which could be turning left or right by a specific degree, "
            "moving forward a certain distance, or stop if the task is completed. "
            "If reasoning, describe your current observations, assess task progress, and provide a high-level plan for the next steps."
        )
        parts.append(tail)
        return parts

    def _query(self, frames, instruction):
        content = []
        for part in self._question_parts(frames, instruction):
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
            else:
                content.append({"type": "image_url", "image_url": {"url": _png_data_url(part)}})
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": LLAMA3_SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 256,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, headers={"Content-Type": "application/json"}
        )
        # Bypass http_proxy from the environment: the server is local, and a
        # proxied loopback request comes back as 502 / connection reset.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        last_error = None
        for attempt in range(5):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return (payload["choices"][0]["message"]["content"] or "").strip()
            except Exception as exc:  # transient server / socket errors
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("AwareVLN server unreachable: {}".format(last_error))
