"""Python 3.9 Habitat adapter for the v1 Python 3.12 VLN agent process."""

import base64
import io
import json
import os
import select
import subprocess

import numpy as np
from PIL import Image


HABITAT_ACTIONS = {
    "FORWARD": "move_forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "STOP": "stop",
}

R2R_CE_OVERRIDES = [
    "habitat.simulator.forward_step_size=0.25",
    "habitat.simulator.turn_angle=15",
    "habitat.task.measurements.success.success_distance=3.0",
]


def habitat_action(action):
    try:
        return {"action": HABITAT_ACTIONS[action]}
    except KeyError as exc:
        raise ValueError("Unsupported VLN action: {!r}".format(action)) from exc


def unpack_observation(observation):
    rgb = np.asarray(observation["rgb"])
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected HWC RGB, got {}".format(rgb.shape))
    return (
        np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8),
        observation["instruction"]["text"],
    )


class VLNAgentProcess:
    def __init__(
        self,
        python,
        worker,
        model_path,
        planner_model_path,
        gpu_id=None,
        response_timeout=600,
        temporal_model_path=None,
        memory_mode="temporal",
    ):
        self.command = [
            str(python),
            str(worker),
            "--model-path",
            str(model_path),
            "--planner-model-path",
            str(planner_model_path),
        ]
        if temporal_model_path is not None:
            self.command.extend(
                ["--temporal-model-path", str(temporal_model_path)]
            )
        self.command.extend(["--memory-mode", str(memory_mode)])
        self.response_timeout = response_timeout
        self.last_memory_diagnostics = None
        environment = os.environ.copy()
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        # The worker uses a separate virtual environment; prefer this checkout
        # over any stale agentflow package installed in that environment.
        project_root = str(worker.parent.parent)
        environment["PYTHONPATH"] = project_root + os.pathsep + environment.get("PYTHONPATH", "")
        if gpu_id is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Keep model import/CUDA failures in the owning rank log. Protocol
            # messages still use the dedicated stdout pipe.
            stderr=None,
            env=environment,
            text=True,
            bufsize=1,
        )

    def reset(self, instruction):
        response = self._request({"reset": instruction})
        self.last_memory_diagnostics = response.get("memory")
        return response

    @staticmethod
    def _encode_image(rgb):
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def act(self, rgb):
        response = self._request({"image": self._encode_image(rgb)})
        self.last_memory_diagnostics = response.get("memory")
        return response["action"]

    def finish_episode(self, rgb):
        request = {
            "finish_episode": True,
            "image": self._encode_image(rgb),
        }
        response = self._request(request)
        self.last_memory_diagnostics = response.get("memory")
        return response

    def _request(self, request):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("VLN agent process exited unexpectedly")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.response_timeout)
        if not ready:
            raise RuntimeError("VLN agent worker timed out after {} seconds".format(self.response_timeout))
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("VLN agent process exited unexpectedly")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self, timeout=2):
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None
