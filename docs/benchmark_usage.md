# Benchmark 运行与文件说明

返回 [benchmark 总览](../benchmark/README.md)。

以下命令在项目根目录执行。数据生成需要本地 R2R-CE 数据；FLIP 构造还需要
安装 Habitat-Sim 的 Python 环境和场景 navmesh。推理使用对应系统的运行环境。
仅重算归档指标可使用项目 `.venv/bin/python`（需要 NumPy，fastdtw 可选）。
代码块中的 `<cfg.yaml>`、`<out_orig>` 等是需替换的占位符。

```bash
cd /data/pengyh/workspace/FreeAskAgent_R2R

# 数据已生成；更换评测集时重新构造
python3 -m benchmark.build_subgoals --ids integrations/v3/eval_sets/val_unseen_200.txt
python3 -m benchmark.build_goalonly
python3 -m benchmark.selfcheck

# 运行 GOAL-ONLY
SPLIT=val_unseen_200_goalonly EPISODE_IDS=@benchmark/data/val_unseen_200/rule/ids/goalonly.txt \
  TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_goalonly> \
  bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh

# PARAPHRASE 四个 arm 都要跑，para_id 是对照组
for ARM in para_id para_terse para_natural para_lm_shift; do
  SPLIT=val_unseen_200_$ARM EPISODE_IDS=@benchmark/data/val_unseen_200/llm/ids/${ARM}.txt \
    TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_$ARM> \
    bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh
done

# 用归档轨迹重算
python3 -m benchmark.metrics --orig benchmark/results/freeaskagent/orig \
  --goalonly benchmark/results/freeaskagent/goalonly \
  --json /tmp/freeaskagent_metrics.json --csv /tmp/freeaskagent_per_episode.csv
```

CA-Nav / AwareVLN 的适配入口分别为 `python3 -m benchmark.canav build` 和
`python3 -m benchmark.awarevln build`，支持 ORIG、FLIP、GOAL-ONLY（`--variants` 可选其中几项）；
`import` 子命令导入轨迹。
PARAPHRASE 可将每个 arm 的轨迹传给 `metrics --orig`，使用同一组子目标边界评分。
CA-Nav 的 PARAPHRASE 必须先重新解析改写指令。

nDTW 优先使用 fastdtw，未安装时使用精确 DTW；两者不完全相同，跨实验比较时应统一环境。

## FLIP(起点转弯,2026-09-24)

FLIP 使用自己的集合:从全量 val_unseen 中选第一句含转弯指令、且参考路径在起点
前 5 m 内按该词方向转 45°–135° 的 episode,左右配平后 82 集。转弯发生在起点,
每个系统的每一集都能评分。起始朝向统一用 R2R-CE v1-3(v1-2 的朝向与指令不符),
所以 **ORIG 对照也要在同一批 episode、同一朝向上重跑**(`flip_orig`)。

```bash
# 数据(需要 habitat_sim 和 navmesh;CA-Nav 环境可用)
/data/pengyh/miniconda3/envs/CA-Nav/bin/python -m benchmark.build_flip     # data/val_unseen/rule/flip.json + FreeAskAgent splits
python3 -m benchmark.canav build --variants flip                             # CA-Nav: val_unseen_flip{_orig,}
python3 -m benchmark.awarevln build --variants flip                          # AwareVLN: val_unseen_flip{_orig,}

# FreeAskAgent(两次运行都用 v1-3 起始朝向的 split)
for V in flip_orig flip; do
  SPLIT=val_unseen_$V EPISODE_IDS=@benchmark/data/val_unseen/rule/ids/flip.txt \
    TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_$V> \
    bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh
done

# CA-Nav(在 CA-Nav-code 下;实验名不能用旧的 exp_bench_flip,评测会跳过目录里已有的 episode)
SET=val_unseen VARIANT=flip_orig EXP_NAME=exp_startflip_orig bash run_r2r/bench_local.sh
SET=val_unseen VARIANT=flip      EXP_NAME=exp_startflip      bash run_r2r/bench_local.sh

# AwareVLN(在 AwareVLN/evaluation 下)
SET=val_unseen VARIANT=flip_orig GPU=<id> bash scripts/eval/bench_local.sh
SET=val_unseen VARIANT=flip      GPU=<id> bash scripts/eval/bench_local.sh

# 导入到 results/<system>/{flip_orig,flip}/ 后评分
python3 -m benchmark.canav import --exp exp_startflip_orig --out benchmark/results/canav/flip_orig
python3 -m benchmark.metrics --orig benchmark/results/<system>/flip_orig --flip benchmark/results/<system>/flip
```

评分(`metrics.evaluate_flip`):每次运行的每一集按离开起点的方向归为按指令词转 /
反着转 / 直行(< 30°)/ 没离开起点四类,四类之和为 1;两次运行合并给出
MeanFollow、Blind(不看方向词时的期望)与 WordEffect = MeanFollow − Blind
(完全跟随约 +0.5,不看方向词为 0),以及按 left / right 分开的跟随率。
CA-Nav 的 FLIP 只改解析里包含该方向词的那条子指令中的这一个词。

## 数据构建与改写导入

```bash
python3 -m benchmark.build_paraphrase export
# 由外部生成器完成任务，将 batch_*.json 放入下面的 paraphrase_gen 目录
python3 -m benchmark.build_paraphrase ingest
```

`export` 导出生成任务，`ingest` 校验并导入已有改写；这两个命令不会自行调用 LLM。
源文件位于 `benchmark/data/val_unseen_200/llm/paraphrase_gen/`。
`ingest` 会写元数据和四个 Habitat split，添加 `--no-splits` 可仅写元数据。

## 文件入口

| 路径（相对 benchmark/） | 用途 |
|---|---|
| `build_subgoals.py` | 从 FGR2R 人工分段构建路径边界 |
| `build_flip.py` / `build_goalonly.py` / `build_paraphrase.py` | 三类实验的数据构建 |
| `metrics.py` / `selfcheck.py` | 轨迹评分 / 带噪参考轨迹半径标定 |
| `canav.py` / `awarevln.py` | 外部系统输入适配和轨迹导入 |
| `common.py` / `compact_trace.py` | 公共读写与几何 / 轨迹归档压缩 |
| `data/fgr2r/` | 第三方人工标注及 LICENSE |
| `data/<set>/subgoals.json` | 所有实验共用的子目标边界 |
| `data/<set>/rule/` | GOAL-ONLY 元数据和 `ids/`;FLIP 在 `data/val_unseen/rule/` |
| `data/val_unseen/subgoals.json` | 全量 val_unseen 的子目标边界(FLIP 用) |
| `data/<set>/llm/` | PARAPHRASE 元数据、生成批次、`ids/` 和 `review/` |
| `results/<model>/orig/`、`goalonly/`、`flip_orig/`、`flip/` | 原始轨迹和日志 |
| `results/<model>/metrics*.json` | 指标及逐任务结果 |
| `results/<model>/per_episode.csv` | ORIG 逐任务指标 |
| `results/freeaskagent/orig2/` | 第二次 ORIG(47 集),估计随机波动 |

增加评测集时新建 `data/<set>/`；数据路径统一由 `common.data_path`、`ids_path`、
`gen_dir` 管理。`rule/` 是规则变换，`llm/` 是整体措辞改写；元数据中的
`provenance` 记录来源，PARAPHRASE 的 `generator` 记录批次所声明的生成器。
默认评测集为 `val_unseen_200`，对应 `integrations/v3/eval_sets/val_unseen_200.txt`。

轨迹导入示例：

```bash
python3 -m benchmark.canav import --exp exp_bench_orig --out benchmark/results/canav/orig
python3 -m benchmark.awarevln import --results /path/to/awarevln/results --out benchmark/results/awarevln/orig
```

其他系统可导出兼容的 `rank_*_trace.jsonl` 和 `rank_*.log`；
位置与结果字段以 `common.load_run` / `write_runner_run` 为准。

## 当前 PARAPHRASE 批次（2026-09-20）

`benchmark.run_paraphrase prepare` 为两个系统构造四组各 200 条输入，并为 CA-Nav
写出 `parse_tasks.jsonl`；不会复用原文解析。输入哈希和运行状态在
`outputs/paraphrase_20260920/`。A4 的已知疑点保留在本批次记录中。

`benchmark.run_paraphrase awarevln --wait-hours 24` 等待至少 28 GB 空闲显存，
随后按 A1→A4 顺序运行原 FP16 配置，完成每组后导入轨迹和评分。
续跑时核对输入哈希及已完成轨迹，跳过已完成任务；分组结束后合并完整统计。
进度见 `awarevln_status.json`，日志为 `awarevln_queue.log` 和 `awarevln_<arm>.log`。
锁文件防止重复启动；不要删除仍被工作进程持有的锁文件。

CA-Nav 仍需可用的 GPT-4 服务生成每组新的 `llm_reply.json`，之后用
`VARIANT=para_id` 等参数运行其 `run_r2r/bench_local.sh`。旧解析端点本次返回 HTTP 405。

## CA-Nav 使用 Qwen 解析（2026-09-23）

运行入口：`.venv/bin/python -m benchmark.canav_qwen --gpus 1,2,3`。
默认连接 `http://127.0.0.1:8302/v1`，模型为本地 Qwen3-VL-8B-Instruct
（服务名 `qwen3-vl-8b`），使用上游解析提示词和 temperature=0。
每条输出经结构校验后缓存，全部通过后才运行导航；导航沿用 `exp1_nogate`。

为控制解析器变更的影响，该批次包含 Qwen 解析的 ORIG 与 A1～A4，共 1000 条。
数据 split 使用 `val_unseen_200_qwen_*`，结果保存到 `benchmark/results/canav_qwen/`；
与旧 GPT-4 解析的 `canav/` 分开。输入及提示词哈希、解析模型记录在 `manifest.json`。
过程日志和可续跑解析缓存位于 `outputs/paraphrase_20260920/canav_qwen/`，
进度见 `status.json`。A4 的既有语义疑点仍保留在该批次输入中。

Qwen 解析若不符合结构约定，会带原始指令、失败响应和校验反馈重新请求（最多三次）；
保留原始失败输出和纠错记录，不直接把不支持的方向词映射成另一个方向。
解析进度按成功与失败分别统计；已成功的缓存会复用。

校验以导航实际读取的字段为准：`decisions.directions` 未被当前 CA-Nav 导航器使用，
因此保留 `turn around` 等字符串描述，不强制限制为 left/right/forward；
实际使用的约束也按原导航器接受的类型校验：允许空约束；方向字符串保留，
由上游处理 forward/backward、别名或未知方向默认分支。动作字符串同样原样保留，
上游主要区分 `move away` 与其他值。结构校验不代表解析语义正确。
