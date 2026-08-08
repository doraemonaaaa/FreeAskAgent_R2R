# Temporal Memory R2R evaluation (v1)

Habitat sends only the episode instruction and RGB observations to the VLN
agent. Memory construction, timestamps, action/post-frame pairing, optical
flow, and video understanding remain inside the agent worker.

Example:

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python integrations/v1/run_habitat.py \
  --split val_unseen \
  --episodes 1 \
  --max-steps 8 \
  --memory-mode task+temporal \
  --record-video \
  --artifact-dir outputs/vln_memory \
  --run-id task-plus-temporal
```

The run is written to:

```text
outputs/vln_memory/task-plus-temporal/
├── results.json
├── videos/<episode_id>.mp4
└── topdown/<episode_id>.png
```

`results.json` contains the instruction, actions, number of steps, scalar
Habitat metrics, media paths, full memory diagnostics (including the raw video
model response), and weighted timing:

- Task Memory average observation-update time.
- Temporal Memory average eligible three-step analysis time.
- Temporal Memory image-only interface average observation-update time.
- Video Understanding Foundation Model average synchronized inference time.

The default live configuration uses a three-step window and a three-step
analysis stride. Rules and optical flow still update after every action, while
the video model runs once per three new completed actions. The recorded
end-to-end Temporal Memory latency budget is 5000 ms.

For ablations, change `--memory-mode` to `task`, `temporal`, or
`task+temporal`. The eight-GPU launcher accepts the same choice through
`MEMORY_MODE`.
