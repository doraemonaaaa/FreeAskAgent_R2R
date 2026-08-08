# FreeAskAgent × Habitat R2R-CE 使用说明

本目录提供 FreeAskAgent VLN agent 在 Habitat R2R-CE 数据集上的单卡推理、可视化和八卡并行评测适配代码。

## 目录结构

推理代码按 agent 版本分开，公共的结果与产物工具保留在 `integrations/`：

| 路径 | 用途 |
| --- | --- |
| `integrations/v1/` | 第一版 `AsyncThinkActVLN`：RGB 输入、离散动作输出 |
| `integrations/v1/run_habitat.py` | 第一版单卡推理入口 |
| `integrations/v1/run_r2r_ce_8gpu.sh` | 第一版八卡评测入口 |
| `integrations/v1/r2r_ce_example.py` | 第一版支持分片的评测 runner |
| `integrations/v1/r2r_ce_adapter.py` | 第一版 Habitat 与模型 worker 的通信适配器 |
| `integrations/v1/vln_agent_worker.py` | 加载第一版 `agentflow.agents.vln_agent` |
| `integrations/v3/` | 第三版 `vln_agent_3.Actor`：RGB-D 输入、waypoint 输出 |
| `integrations/v3/run_habitat.py` | 第三版单卡及分片评测 runner |
| `integrations/v3/run_r2r_ce_8gpu.sh` | 第三版八卡评测入口 |
| `integrations/v3/vln_waypoint_worker.py` | 加载第三版 `agentflow.agents.vln_agent_3` |
| `integrations/aggregate_r2r_ce_results.py` | 两个版本共用的多 rank 结果汇总 |
| `integrations/run_artifacts.py` | 公共运行产物管理 |

## 路径约定

当前代码默认使用以下路径：

```text
项目目录：
/data/pengyh/workspace/FreeAskAgent_R2R

Habitat：
/data/pengyh/workspace/habitat/habitat-lab

Habitat 数据：
/data/pengyh/workspace/habitat/data

Habitat Python：
/data/pengyh/miniconda3/envs/habitat/bin/python

模型：
/data/pengyh/workspace/FreeAskAgent/models/JoyAI-VL-Interaction
```

Habitat 主进程使用 Habitat Conda 环境，模型 worker 使用本项目的 `.venv/bin/python`。

## 单卡推理

运行一个 `val_unseen` episode 并保存可视化：

```bash
CUDA_VISIBLE_DEVICES=0 /data/pengyh/miniconda3/envs/habitat/bin/python integrations/v3/run_habitat.py \
  --split val_unseen \
  --episodes 1 \
  --max-steps 100 \
  --record-video \
  --video-dir videos
```

默认 `--scene-id all`，会从指定 split 中选择有效 episode，不需要手动指定场景。

输出：

```text
videos/<episode_id>.mp4
videos/topdown/<episode_id>.png
```

MP4 左侧为第一视角 RGB，右侧为当前楼层的 Habitat navmesh 俯视图：

- 灰色：可导航区域
- 白色：不可通行区域
- 轨迹线：agent 实际运动轨迹
- 起点、终点和目标：使用不同颜色标记

仅运行指标、不保存视频：

```bash
CUDA_VISIBLE_DEVICES=0 python integrations/v1/run_habitat.py \
  --split val_unseen \
  --episodes 1 \
  --max-steps 500
```

`--episodes 0` 表示运行指定 split 的全部 episode。

## 八卡批量评测

第一版配置写在 `integrations/v1/run_r2r_ce_8gpu.sh` 顶部：

```bash
split="val_unseen"
scene_id="all"
episodes=0
max_steps=500
record_video=false
video_dir="${root_dir}/videos/r2r_8gpu"
output_dir="${root_dir}/outputs/r2r_ce_8gpu"
```

修改配置后直接运行，不需要附加命令行参数：

```bash
bash integrations/v1/run_r2r_ce_8gpu.sh
```

### GPU 分配

八个 rank 独立运行：

```text
rank 0 → GPU 0
rank 1 → GPU 1
...
rank 7 → GPU 7
```

每个 rank 拥有一个 Habitat evaluator 和一个常驻模型 worker。worker 继承对应 rank 的 `CUDA_VISIBLE_DEVICES`，不会跨卡加载模型。

脚本启动时会自动终止本项目之前遗留的 `r2r_ce_example.py` 和 `vln_agent_worker.py` 进程，避免旧任务占用显存。不会匹配其他项目的 Python/CUDA 进程。

### Episode 分片

完整 episode 列表按以下方式无重叠分片：

```python
episodes[rank::8]
```

例如 `val_unseen` 共 1839 条，每个 rank 约运行 229–230 条。

### 查看进度

```bash
tail -f outputs/r2r_ce_8gpu/rank_*.log
```

统计当前完成的 episode 数：

```bash
grep -h '^rank=.*episode=' outputs/r2r_ce_8gpu/rank_*.log | wc -l
```

检查 rank 与 GPU 映射：

```bash
head -1 outputs/r2r_ce_8gpu/rank_*.log
nvidia-smi
```

日志第一行应类似：

```text
rank=3 CUDA_VISIBLE_DEVICES=3
```

## 指标和汇总

每个 episode 会打印：

```text
rank=0 episode=1/230 id=1212 steps=500 metrics={
  'distance_to_goal': 7.14,
  'success': 0.0,
  'spl': 0.0
}
```

指标含义：

- `success`：agent 是否在成功半径内执行 STOP
- `spl`：Success weighted by Path Length
- `distance_to_goal`：episode 结束时到目标的距离

每个 rank 完成后将结果写入：

```text
outputs/r2r_ce_8gpu/rank_0.json
...
outputs/r2r_ce_8gpu/rank_7.json
```

八个 rank 全部完成后，启动脚本自动调用：

```bash
python integrations/aggregate_r2r_ce_results.py \
  outputs/r2r_ce_8gpu \
  --world-size 8
```

汇总是按实际 episode 数加权计算，不是简单平均八个 rank 的平均值。

## 模型复用与非法动作

模型 worker 在每个 rank 启动时加载一次，后续 episode 仅重置目标和 episode 状态，不重复加载模型。

支持的 Habitat 动作：

```text
FORWARD
TURN_LEFT
TURN_RIGHT
STOP
```

如果模型返回 `TURN_DOWN`、`TURN_AROUND` 等非法动作：

- 不向 Habitat 发送该动作
- 不转换为 STOP
- 当前 episode 中止并保留当时指标
- 继续运行下一个 episode
- 不会导致整个 rank 或八卡任务退出

## 常见问题

### 日志路径不存在

当前八卡输出目录是：

```text
outputs/r2r_ce_8gpu
```

应使用：

```bash
tail -f outputs/r2r_ce_8gpu/rank_*.log
```

### GPU 利用率为 0，日志长时间不更新

检查 evaluator 是否阻塞：

```bash
ps -eo pid,ppid,etime,stat,wchan,cmd | \
  grep -E 'r2r_ce_example|vln_agent_worker'
```

adapter 对 worker 响应设置了 600 秒超时，超时后会打印明确错误，不再永久等待。

### CUDA OOM

常见原因是旧 R2R worker 尚未退出，或多个评测任务同时使用同一组 GPU。重新运行八卡脚本时会自动清理本项目旧任务。也可先通过以下命令检查：

```bash
nvidia-smi
pgrep -af 'integrations/v1/(r2r_ce_example|vln_agent_worker)'
```

### `dataset should have non-empty episodes list`

指定的 `scene-id` 不属于当前 split。八卡脚本默认使用：

```text
split=val_unseen
scene_id=all
```

因此不会产生场景与 split 不匹配的问题。

### SemanticScene `.scn` 警告

MP3D 缺少语义标注时可能打印 `SSD Load Failure` 或 `active scene does not contain semantic annotations`。R2R-CE 的 RGB 导航和 navmesh 通常仍可运行；应以是否出现 Python traceback、episode 指标是否继续更新为准。

## 推荐工作流

先运行少量 episode 验证环境：

```bash
# 临时将 run_r2r_ce_8gpu.sh 中 episodes 改为 8
bash integrations/v1/run_r2r_ce_8gpu.sh
```

确认八个 rank 都能输出指标后，再将：

```bash
episodes=0
```

用于完整 `val_unseen` 评测。完整评测耗时较长；开启视频还会显著增加磁盘占用，建议全量指标评测使用 `record_video=false`。
