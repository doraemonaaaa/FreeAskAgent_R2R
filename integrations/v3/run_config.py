"""Single-source run configuration for the v3 VLN stack (integrations/v3/config.yaml).

The YAML names the model *services* (one vLLM endpoint each) and maps the
agent's five model *roles* onto them:

    observer  JoyAI observation (Captioner)            -> JOYAI_API_BASE
    decision  completion + route decision, also the    -> CAPTIONER_DECISION_MODEL/_BASE_URL
              isolated completion gate (same engine)
    planner   instruction -> subgoals                  -> VLN_PLANNER_MODEL/_BASE_URL
    actor     waypoint actor engine (fallback engine)  -> --model-path/--base-url, VLLM_BASE_URL
    som       set-of-mark marker choice                -> VLN_SOM_MODEL/_BASE_URL

Everything the agent used to read from ad-hoc shell exports (captioner flags,
agent switches, runner knobs) also lives here. Precedence, lowest to highest:
YAML -> environment variables already exported by the caller -> CLI flags.

Used by run_habitat.py (per-rank defaults + env), by the launcher scripts
(``--shell`` prints bash assignments) and by serve_vllm.sh (``--serve NAME``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "integrations/v3/config.yaml"

ROLES = ("observer", "decision", "planner", "actor", "som")
# Role -> role whose service is used when the role is not configured.
ROLE_FALLBACK = {"planner": "decision", "som": "actor"}

# captioner.<key> -> environment variable (bool flags become "1"/"0").
CAPTIONER_ENV = {
    "native_minimal": "JOYAI_NATIVE_MINIMAL",
    "observe_then_judge": "JOYAI_OBSERVE_THEN_JUDGE",
    "judge_preview_images": "JOYAI_JUDGE_PREVIEW_IMAGES",
    "reasoned_control": "JOYAI_REASONED_CONTROL",
    "inference_only_completion": "CAPTIONER_INFERENCE_ONLY_COMPLETION",
    "route_identity": "CAPTIONER_ROUTE_IDENTITY",
    "sparse_judgement_images": "CAPTIONER_SPARSE_JUDGEMENT_IMAGES",
    "preview_candidates": "CAPTIONER_PREVIEW_CANDIDATES",
    "step_deadline_s": "CAPTIONER_STEP_DEADLINE_S",
    "evidence_dir": "JOYAI_EVIDENCE_DIR",
}
# agent.<key> -> environment variable.
AGENT_ENV = {
    "spatial_memory": "VLN_SPATIAL_MEMORY",
    "som": "VLN_SOM",
    "compact_subgoals": "VLN_COMPACT_SUBGOALS",
    "filter_route_candidates": "VLN_FILTER_ROUTE_CANDIDATES",
    "structured_vlm_max_tokens": "VLN_STRUCTURED_VLM_MAX_TOKENS",
    "vlm_image_max_pixels": "VLN_IMAGE_MAX_PIXELS",
    "frozen_plan_file": "VLN_FROZEN_PLAN_FILE",
    "som_trace": "VLN_SOM_TRACE",
}
# runner.<key> -> run_habitat.py argparse default of the same name.
RUNNER_KEYS = (
    "camera_pitch_deg", "max_steps", "waypoint_radius", "depth_hfov", "actor",
    "record_video", "cwp_candidates", "panohop_mode", "panohop_url", "panohop_model",
)


def _same_endpoint(a: dict | None, b: dict | None) -> bool:
    return bool(a and b and a["served_name"] == b["served_name"] and a["base_urls"] == b["base_urls"])


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class RunConfig:
    def __init__(self, data: dict, path: Path | None = None) -> None:
        self.path = path
        self.data = data or {}
        self.services: dict[str, dict] = dict(self.data.get("services") or {})
        self.roles: dict[str, dict] = {}
        self._resolve_roles()

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "RunConfig":
        import yaml

        path = Path(path or os.environ.get("V3_CONFIG") or DEFAULT_CONFIG)
        with open(path) as handle:
            data = yaml.safe_load(handle) or {}
        return cls(data, path)

    # -- model roles ---------------------------------------------------------

    def _resolve_roles(self) -> None:
        models = self.data.get("models") or {}
        legacy = self.data.get("model") or {}
        if not models and legacy:
            # Old layout: model.actor.{path,base_url} (or flat model.path).
            actor = legacy.get("actor") or legacy
            models = {"actor": {"served_name": str(actor.get("path", "")).replace("vllm-", "", 1),
                                "base_url": actor.get("base_url")}}
        for role in ROLES:
            spec = models.get(role)
            if spec is None:
                fallback = ROLE_FALLBACK.get(role)
                if fallback and fallback in self.roles:
                    self.roles[role] = dict(self.roles[fallback], inherited_from=fallback)
                continue
            if isinstance(spec, str):
                spec = {"service": spec}
            service_name = spec.get("service")
            service = self.services.get(service_name, {}) if service_name else {}
            urls = spec.get("base_urls") or service.get("base_urls")
            if not urls:
                url = spec.get("base_url") or service.get("base_url")
                urls = [url] if url else []
            if not urls:
                raise ValueError(f"models.{role}: no base_url (service={service_name!r})")
            served = spec.get("served_name") or service.get("served_name")
            if not served:
                raise ValueError(f"models.{role}: no served_name (service={service_name!r})")
            self.roles[role] = {
                "service": service_name,
                "served_name": str(served),
                "base_urls": [str(u) for u in urls],
                "kind": str(spec.get("kind") or service.get("kind") or "vllm"),
            }

    def role(self, name: str) -> dict | None:
        return self.roles.get(name)

    def role_url(self, name: str, rank: int = 0) -> str | None:
        spec = self.roles.get(name)
        if not spec:
            return None
        urls = spec["base_urls"]
        return urls[rank % len(urls)]

    def actor_model_path(self) -> str | None:
        """``--model-path`` for the worker (see create_vln_engine)."""
        spec = self.roles.get("actor")
        if not spec:
            return None
        if spec["kind"] == "joyai":
            return f"joyai-{spec['served_name']}"
        return f"vllm-{spec['served_name']}"

    # -- what run_habitat.py needs -------------------------------------------

    def argparse_defaults(self, rank: int = 0) -> dict:
        defaults: dict[str, Any] = {}
        model_path = self.actor_model_path()
        if model_path:
            defaults["model_path"] = model_path
            defaults["base_url"] = self.role_url("actor", rank)
        runner = self.data.get("runner") or {}
        for key in RUNNER_KEYS:
            if runner.get(key) is not None:
                defaults[key] = runner[key]
        return defaults

    def agent_env(self, rank: int = 0) -> dict[str, str]:
        env: dict[str, str] = {}
        actor = self.roles.get("actor")
        if actor:
            env["VLLM_BASE_URL"] = self.role_url("actor", rank)
        observer = self.roles.get("observer")
        if observer:
            env["JOYAI_API_BASE"] = self.role_url("observer", rank)
            env["JOYAI_MODEL_NAME"] = observer["served_name"]
        decision = self.roles.get("decision")
        if decision:
            env["CAPTIONER_DECISION_MODEL"] = decision["served_name"]
            env["CAPTIONER_DECISION_BASE_URL"] = self.role_url("decision", rank)
        planner = self.roles.get("planner")
        if planner and not _same_endpoint(planner, actor):
            env["VLN_PLANNER_MODEL"] = planner["served_name"]
            env["VLN_PLANNER_BASE_URL"] = self.role_url("planner", rank)
        som = self.roles.get("som")
        if som and not _same_endpoint(som, actor):
            env["VLN_SOM_MODEL"] = som["served_name"]
            env["VLN_SOM_BASE_URL"] = self.role_url("som", rank)
        for section, mapping in (("captioner", CAPTIONER_ENV), ("agent", AGENT_ENV)):
            values = self.data.get(section) or {}
            for key, env_key in mapping.items():
                if values.get(key) is not None:
                    env[env_key] = _env_value(values[key])
        return env

    # -- what the launcher / serve scripts need ------------------------------

    def eval_section(self) -> dict:
        return dict(self.data.get("eval") or {})

    def shell_exports(self, prefix: str = "CFG_") -> str:
        """Bash assignments describing the resolved configuration."""
        lines = [f"{prefix}CONFIG={shlex.quote(str(self.path))}"]
        for role in ROLES:
            spec = self.roles.get(role)
            if not spec:
                continue
            key = role.upper()
            lines.append(f"{prefix}{key}_MODEL={shlex.quote(spec['served_name'])}")
            lines.append(f"{prefix}{key}_URLS={shlex.quote(','.join(spec['base_urls']))}")
        eval_cfg = self.eval_section()
        for key in ("episode_set", "split", "world_size", "rank_gpus", "max_steps", "episodes"):
            if eval_cfg.get(key) is not None:
                value = eval_cfg[key]
                if isinstance(value, list):
                    value = ",".join(str(v) for v in value)
                lines.append(f"{prefix}EVAL_{key.upper()}={shlex.quote(str(value))}")
        runner = self.data.get("runner") or {}
        for key in ("max_steps", "camera_pitch_deg"):
            if runner.get(key) is not None:
                lines.append(f"{prefix}RUNNER_{key.upper()}={shlex.quote(str(runner[key]))}")
        return "\n".join(lines) + "\n"

    def serve_env(self, service_name: str) -> dict[str, str]:
        """Environment for serve_vllm.sh / start_joyai_captioner.sh."""
        service = self.services.get(service_name)
        if service is None:
            raise KeyError(f"unknown service {service_name!r}; known: {sorted(self.services)}")
        serve = dict(service.get("serve") or {})
        kind = service.get("kind", "vllm")
        port = serve.get("port")
        if port is None and service.get("base_url"):
            port = service["base_url"].rsplit(":", 1)[-1].split("/", 1)[0]
        env: dict[str, str] = {}
        if kind == "joyai":
            table = {
                "model_dir": "JOYAI_MODEL_PATH", "gpu": "JOYAI_GPU", "tp": "JOYAI_TENSOR_PARALLEL_SIZE",
                "gpu_util": "JOYAI_GPU_MEMORY_UTILIZATION", "max_model_len": "JOYAI_MAX_MODEL_LEN",
                "max_num_seqs": "JOYAI_MAX_NUM_SEQS", "max_images": "JOYAI_MAX_IMAGES",
                "max_num_batched_tokens": "JOYAI_MAX_NUM_BATCHED_TOKENS",
                "cudagraph_capture_sizes": "JOYAI_CUDAGRAPH_CAPTURE_SIZES",
            }
            env["JOYAI_MODEL_NAME"] = str(service.get("served_name", ""))
            if port is not None:
                env["JOYAI_PORT"] = str(port)
        else:
            table = {
                "model_dir": "MODEL_PATH", "gpu": "GPU", "tp": "TP", "gpu_util": "GPU_UTIL",
                "max_model_len": "MAX_MODEL_LEN", "max_num_seqs": "MAX_NUM_SEQS",
                "max_images": "MAX_IMAGES", "trust_remote_code": "TRUST_REMOTE_CODE",
                "extra_args": "VLLM_EXTRA_ARGS",
            }
            env["SERVED_NAME"] = str(service.get("served_name", ""))
            if port is not None:
                env["PORT"] = str(port)
        for key, env_key in table.items():
            if serve.get(key) is not None:
                env[env_key] = _env_value(serve[key])
        return env

    def manifest(self) -> dict:
        return {
            "config": str(self.path),
            "services": {name: {k: v for k, v in spec.items() if k != "serve"}
                         for name, spec in self.services.items()},
            "models": self.roles,
            "captioner": dict(self.data.get("captioner") or {}),
            "agent": dict(self.data.get("agent") or {}),
            "runner": dict(self.data.get("runner") or {}),
            "eval": self.eval_section(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=None, help="YAML path (default: $V3_CONFIG or integrations/v3/config.yaml)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shell", action="store_true", help="print CFG_* bash assignments")
    group.add_argument("--serve", metavar="SERVICE", help="print KEY=VALUE serving env for one service")
    group.add_argument("--agent-env", action="store_true", help="print the agent environment (KEY=VALUE) for --rank")
    group.add_argument("--manifest", action="store_true", help="print the resolved configuration as JSON")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    if args.shell:
        print(config.shell_exports(), end="")
    elif args.serve:
        for key, value in config.serve_env(args.serve).items():
            print(f"{key}={shlex.quote(value)}")
    elif args.agent_env:
        for key, value in config.agent_env(args.rank).items():
            print(f"{key}={shlex.quote(value)}")
    else:
        print(json.dumps(config.manifest(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
