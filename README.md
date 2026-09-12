# FreeAskAgent × Habitat R2R-CE

FreeAskAgent 的 VLN agent（`FreeAskAgent/agentflow/agents/vln`）在 Habitat R2R-CE
上的评测适配：单卡推理、可视化和多卡并行评测。

## 目录结构

```text
integrations/v3/
  config.yaml                     运行配置的唯一来源（服务、模型角色、agent 开关、
                                  spatial_memory / navigation 参数、eval / runner 形状）
  sensor_config.yaml              传感器硬件参数（相机高度/俯仰/FOV/分辨率、深度范围；
                                  可选标定内参、畸变、外参）
  run_config.py                   config.yaml 解析器：--manifest / --shell / --serve NAME
  run_r2r_ce_inference_only_multigpu.sh   多 rank 正式评测入口
  run_habitat.py                  单个 rank 的 Habitat 评测 runner：参数解析 + episode 循环
  habitat_runner/                 runner 按功能拆分的模块
    settings.py                   路径、静默日志、共享几何（转角/步长/相机模型）
    sensors.py                    Habitat 传感器 override、RGB-D 读取、内参、相机位姿、
                                  preview 视角渲染、navmesh 可通行窗口
    actor_process.py              WaypointActorProcess：通过 JSON-lines 管道驱动 worker
    control.py                    waypoint / 转角 → 一个 Habitat 原语（含无 navmesh 的几何 follower）
    video.py                      俯视图、agent 自己的地图面板、帧叠加、subgoal 链
    step_log.py                   每步日志行、rank 汇总、JSONL 决策 trace
    episodes.py                   episode 选择与分片、config 加载
  vln_waypoint_worker.py          Python 3.12 worker：加载 agentflow.agents.vln.agent.VLNAgent
  camera_model.py                 相机模型（标定 K 缩放、去畸变、安装外参）
  preview_protocol.py             preview 视角请求与执行反馈协议
  awarevln_actor.py / serve_awarevln.py   AwareVLN 基线
  panohop_actor.py + waypoint_cwp/        pano-hop（CWP）基线
  eval_sets/                      固定评测集（val_unseen_40 / _200 / preview_10）
  bench/                          离线基准与分析脚本
  tests/                          runner 单元测试（用 habitat 环境运行）
integrations/aggregate_r2r_ce_results.py  多 rank 结果汇总
integrations/run_artifacts.py             运行产物工具
tests/                            run_config / camera_model 单元测试
docs/experiments/                 实验记录（RESULTS.md）
```

## 路径约定

```text
项目目录：      /data/pengyh/workspace/FreeAskAgent_R2R
agent 代码：    /data/pengyh/workspace/FreeAskAgent
Habitat：       /data/pengyh/workspace/habitat/habitat-lab
Habitat 数据：  /data/pengyh/workspace/habitat/data
Habitat Python：/data/pengyh/miniconda3/envs/habitat/bin/python
worker Python： /data/pengyh/workspace/FreeAskAgent/.venv/bin/python
```

Habitat 主进程用 habitat conda 环境（py3.9），模型 worker 用 FreeAskAgent 的 `.venv`（py3.12）。

## 模型服务

`config.yaml` 的 `services:` 描述每个服务（JoyAI 观察器、Qwen3-VL-8B 决策/规划/SoM），
`models:` 把角色映射到服务。启动一个服务：

```bash
integrations/v3/serve_service.sh qwen3-vl-8b   # -> serve_vllm.sh
integrations/v3/serve_service.sh joyai         # -> FreeAskAgent/scripts/start_joyai_captioner.sh
```

## 评测

多 rank 评测（每个 `runner.rank_gpus` 条目一个 Habitat worker）：

```bash
bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh
```

只有运行形状从环境变量取，其余全部来自 yaml（`V3_CONFIG=` 换配置文件）：

```text
SPLIT EPISODES EPISODE_IDS(csv 或 @file) MAX_STEPS RANK_GPUS
TRACE_JSONL=1 EVIDENCE_ARCHIVE_DIR VLN_FROZEN_PLAN_FILE R2R_RUN_ID OUTPUT_DIR
```

输出目录（默认 `outputs/<run_id>`）：`rank_<r>.log`、`rank_<r>.json`、`manifest.json`、
`rank_<r>_trace.jsonl`（TRACE_JSONL=1）、`aggregate.log`。启动器拒绝覆盖已存在的输出目录。

单卡调试：

```bash
CUDA_VISIBLE_DEVICES=4 /data/pengyh/miniconda3/envs/habitat/bin/python integrations/v3/run_habitat.py \
  --episode-ids 121 --max-steps 40 --record-video --video-dir videos
```

`--no-navmesh` 为部署模式：agent 得不到 navmesh 可通行窗口，waypoint 由转向-前进的几何
follower 执行。`--navmesh-candidates` / `--navmesh-follower` 可分别覆盖（消融）。

## 参数

- 传感器：`sensor_config.yaml`（`camera.height_m / pitch_deg / hfov_deg / width / height`、
  `depth.min_m / max_m / normalize`；可选 `camera.intrinsics / distortion / extrinsics`）。
- 机器人原语：`config.yaml` `robot:`（`forward_step_m`、`turn_angle_deg`）。
- 算法：`config.yaml` `spatial_memory:`（栅格与路线几何）和 `navigation:`
  （agent 常量，键名为 `agentflow/agents/vln/config.py` 常量的小写；未知键报错）。

## 指标

每个 episode 打印 `rank=<r> [i/n] id=<ep> steps=<n> success=<x> spl=<x> dtg=<x>`；
`success` 是在 3 m 成功半径内执行 STOP，`spl` 按路径长度加权，`dtg` 为结束时到目标的距离。
全部 rank 完成后启动器自动调用 `integrations/aggregate_r2r_ce_results.py`，
按 episode 数加权汇总到 `aggregate.log`。

## 测试

```bash
/data/pengyh/miniconda3/envs/habitat/bin/python -m pytest -q tests integrations/v3/tests
```
