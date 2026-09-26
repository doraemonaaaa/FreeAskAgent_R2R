"""The Python 3.12 waypoint-actor worker behind a JSON-lines pipe."""

import base64
import io
import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import settings


class WaypointActorProcess:
    """Keep the Python 3.12 vision model out of Habitat's Python process."""

    def __init__(self, python, worker, model_path, gpu_id=None, timeout=600, camera_height_m=None,
                 base_url=None, evidence_dir=None):
        if camera_height_m is None:
            camera_height_m = settings.SENSOR_HEIGHT_M
        command = [
            str(python), str(worker), "--model-path", str(model_path),
            "--camera-height-m", repr(float(camera_height_m)),
        ]
        if base_url:
            command += ["--base-url", str(base_url)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(settings.AGENTFLOW_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        if evidence_dir is not None:
            # Recording a debug/video evaluation must retain the original
            # images used by its auditable decisions, not just an overlay MP4.
            # Respect an explicitly configured location (including opt-out "").
            environment.setdefault("JOYAI_EVIDENCE_DIR", str(Path(evidence_dir).resolve()))
        # Only pin the worker when --gpu-id is given; otherwise inherit the
        # caller's CUDA_VISIBLE_DEVICES so the shell setting is not silently lost.
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            env=environment, text=True, bufsize=1,
        )
        self.timeout = timeout

    @staticmethod
    def _png(rgb):
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _array(values):
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(values), allow_pickle=False)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            raise RuntimeError("Waypoint actor timed out after {} seconds".format(self.timeout))
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("Waypoint actor process exited unexpectedly")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def prepare(self, instruction):
        """Initialize the worker's task memory before an episode starts."""
        return self._request({"operation": "prepare", "instruction": instruction})

    def act(self, rgb, depth, instruction, intrinsics, camera_to_world, preview_views=(), preview_request_id="", previous_execution=None):
        encode_started = time.perf_counter()
        request = {
            "operation": "act",
            "rgb": self._png(rgb), "depth": self._array(depth),
            "instruction": instruction, "intrinsics": np.asarray(intrinsics).tolist(),
            "camera_to_world": np.asarray(camera_to_world).tolist(),
            # Ask for the agent's own map and marker frame only when a video
            # is being recorded: they cost a PNG encode per step.
            "want_visuals": bool(getattr(self, "want_visuals", False)),
            "previous_execution": previous_execution,
        }
        if preview_views:
            request["preview_request_id"] = preview_request_id
            request["preview_views"] = [
                {
                    "yaw_deg": v["yaw_deg"], "rgb": self._png(v["rgb"]),
                    "depth": self._array(v["depth"]),
                    "intrinsics": np.asarray(v["intrinsics"]).tolist(),
                    "camera_to_world": np.asarray(v["camera_to_world"]).tolist(),
                } for v in preview_views
            ]
        encode_ms = (time.perf_counter() - encode_started) * 1000
        roundtrip_started = time.perf_counter()
        result = self._request(request)
        # Serialization and pipe transfer are measured separately from the
        # worker's own model time so a slow step can be attributed to one side.
        result["encode_ms"] = encode_ms
        result["roundtrip_ms"] = (time.perf_counter() - roundtrip_started) * 1000
        if result.get("stop") or "world_xyz" not in result:
            return None, result
        return np.asarray(result["world_xyz"], dtype=np.float32), result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()

