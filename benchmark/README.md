# Subgoal 级指令跟随 benchmark

设计文档:`docs/subgoal_benchmark_design.md`。只看轨迹,不要求被测系统输出任何额外信息。

## 文件

| 路径 | 内容 |
|---|---|
| `common.py` | 数据路径、R2R-CE / 稠密参考路径加载、runner 输出(`rank_*.log`、`rank_*_trace.jsonl`)解析、nDTW |
| `build_subgoals.py` | FGR2R 切分 -> 每条 episode 的 subgoal 边界(`data/subgoals_<set>.json`) |
| `build_variants.py` | SWAP / DROP-k 指令变体,写成额外的 habitat split(`<r2r data>/<set>_swap`、`<set>_drop`)+ 元数据 + 各自的 EPISODE_IDS 文件 |
| `metrics.py` | SGCR、SGCR-eff(每段路程预算 ≤ max(2×参考段长, 3 m))、SGCR@k、ISens-SGCR(-eff)、PathAttrib、SGCR'、LocFail、PrefixKeep、Skip,带 bootstrap 95% CI |
| `build_goalonly.py` | GOAL-ONLY:只保留最后一段子指令(光秃秃的 "Stop." 时多保留前一段);指标 `metrics.py --goalonly` |
| `build_flip.py` | **FLIP-k 主实验**:选 k≥2 的转弯句反转左/右(navmesh 检查反方向可通行),写 v19 split;CA-Nav / AwareVLN 的 flip 输入由各自适配器 `build` 生成 |
| `selfcheck.py` | 参考路径 agent 与最短路 oracle 两个假 agent,r_b 标定 |
| `awarevln.py` | AwareVLN 适配:`build` 写 v1-3 格式的四个 split,`import` 转 trace 格式 |
| `canav.py` | CA-Nav 适配:`build` 生成它的 ORIG/SWAP/DROP 数据与 GPT 解析文件(写到 CA-Nav 的 `data/datasets/benchmark/`),`import` 把它的 `traj_*.jsonl` 转成 runner trace 格式并打印原始指标 |
| `data/FGR2R_val_unseen.json` | FGR2R 原始标注(Hong et al. 2020,许可见 `data/FGR2R_LICENSE`) |
| `data/subgoals_val_unseen_200.json` | 200-set 的 subgoal 边界 |
| `data/variants_val_unseen_200.json` | 200-set 的 SWAP donor 与 DROP-k 元数据 |
| `data/val_unseen_200_{swap,drop}_ids.txt` | 两个变体 split 的 episode id 列表 |
| `results/` | 已算出的结果(`v19_ring6_200_orig.*`) |

## 数据事实(200-set)

- FGR2R 覆盖 200/200。R2R-CE 的 `start_position + reference_path + goal` 与 FGR2R 的 viewpoint 序列一一对应(已用 connectivity 图核对)。
- 同一 viewpoint 结尾的相邻 chunk 合并后,K 均值 2.94,K=1 的 24 条不参与 DROP-k,DROP 集 176 条。
- SWAP donor:200 条都有同场景 donor,其中 128 条 donor 起点与自身起点距离 ≤ 0.5 m;`PathAttrib(same-start donors)` 只在这 128 条上算。
- 相邻边界间距最小 0.25 m,44 对 < 1 m。判定用顺序搜索(e_k 从 e_{k-1} 之后找),所以近距离边界不会因半径重叠而乱序。
- r_b 标定:参考路径加 0.3 m 高斯噪声,r_b ≥ 0.75 m 即 SGCR = 1;默认用 1.5 m,真实 agent 偏离更大。

## 精简协议(2026-09-19 起)

每个系统:ORIG 一次(复用)+ **FLIP 一次(主指标)** + GOAL-ONLY 一次(过拟合诊断)。随机性系统另跑一次 ORIG 作噪声底线,所有变体共用。SWAP / DROP 是第一版指标,保留作附录,新系统不必跑。报告见 `results/REPORT.md`。

## 用法

```bash
cd /data/pengyh/workspace/FreeAskAgent_R2R

# 1. 数据(已生成,改 eval set 时重跑)
python3 -m benchmark.build_subgoals --ids integrations/v3/eval_sets/val_unseen_200.txt
python3 -m benchmark.build_variants --name val_unseen_200

# 2. 自检
python3 -m benchmark.selfcheck

# 3. 跑三遍(ORIG 已有则只补 SWAP / DROP);SPLIT 与 EPISODE_IDS 换成变体的
SPLIT=val_unseen_200_swap EPISODE_IDS=@benchmark/data/val_unseen_200_swap_ids.txt \
  TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_swap> \
  bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh
SPLIT=val_unseen_200_drop EPISODE_IDS=@benchmark/data/val_unseen_200_drop_ids.txt \
  TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_drop> \
  bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh

# 3b. FLIP-k(主实验):SPLIT=val_unseen_200_flip + benchmark/data/val_unseen_200_flip_ids.txt,再跑一遍 ORIG 作噪声底线
#     python3 -m benchmark.metrics --orig <out_orig> --flip <out_flip>      # 转向指标
#     python3 -m benchmark.metrics --orig <out_orig> --flip <out_orig2>     # 噪声底线

# 4. 指标(nDTW 用 fastdtw 时请用 habitat 环境的 python;否则退化为精确 DTW,慢但结果一致)
python3 -m benchmark.metrics --orig <out_orig> --swap <out_swap> --drop <out_drop> \
  --json results/<model>.json --csv results/<model>.csv
```

CA-Nav 的三轮用 `Reproductions/CA-Nav-code/run_r2r/bench_local.sh`(`VARIANT=orig|swap|drop`,CA-Nav conda 环境),结果 `python3 -m benchmark.canav import --exp exp_bench_<variant> --out results/canav/<variant>`。其他系统(AwareVLN)只要把轨迹写成同样的 `rank_*_trace.jsonl` 尾部字段(`episode_id`、`position_before`、`position_after`、`distance_to_goal_before/after`)和 `rank_*.log` 的结果行(`id= steps= success= spl= dtg=`),就能用同一套指标。

## 自检结果(偏移 + 漂移噪声 0.3 m, r_b 0.75)

| agent | SGCR | ISens-SGCR | PathAttrib | LocFail | Skip |
|---|---|---|---|---|---|
| 参考路径 agent | 1.000 | 0.536 | 1.000 | 0.938 | 0.057 |
| 最短路 oracle | 1.000 | 0.000 | 0.010 | 0.000 | 1.000 |

两者 SR 都是 1,但在 ISens / PathAttrib / LocFail / Skip 上完全分开。参考 agent 的 Skip 0.057 来自相邻边界 < 1 m 的 episode:停在第 k-1 段末尾时已落进 B_k。
