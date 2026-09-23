# 导航指令跟随 Benchmark

## 1. 目的

**测导航智能体是否真正执行了指令，而不只看它有没有到达终点。**
做法是保持场景和起点不变，修改指令，再比较实际运动轨迹。
只需要位置轨迹和运行结果，不要求模型输出推理过程或内部子目标。

当前使用 R2R-CE 未见场景验证集的 **200 个任务**，通过 FGR2R 人工分段标注，
将每段指令对应到参考路径上的子目标边界。

## 2. 实验总览

| 实验 | 怎么改指令 | 回答什么问题 | 任务数 |
|---|---|---|---|
| **ORIG：原始基线** | 使用完整原文 | 原指令下表现如何？ | 200 |
| **FLIP：方向反转** | 只把一处 left/right 互换 | 方向要求变了，动作会跟着变吗？ | 47；CA-Nav 可映射 42 |
| **GOAL-ONLY：只给目标** | 保留最后的目标与停止描述，删除过程句 | 完成任务有多依赖过程信息？ | 174 |
| **PARAPHRASE：保义改写** | 相同路线和目标换一种表达 | 表达方式变了，还能跟随吗？ | 每组 200，共四组 |

GOAL-ONLY 若末段只有“Stop”等停止命令，会连同前一段保留；无法有效删除过程信息的任务不纳入。
FLIP 改变了方向要求，因此主要看转向，原终点成功率仅作辅助参考。

PARAPHRASE 的 **A1～A4 按改写类型分类，不是难度等级，也不保证改动程度依次增大**：

| 组别 | 改写方式 | 比较对象 |
|---|---|---|
| **A1 / para_id** | 同风格的保义改写，作为改写对照组 | ORIG |
| **A2 / para_terse** | 简短命令，减少冗余表达 | A1 |
| **A3 / para_natural** | 日常口语，增加连接词和填充词 | A1 |
| **A4 / para_lm_shift** | 同一地标换描述，如 sofa → upholstered seat | A1 |

四组都应保留路线、目标和关键约束，按原子指令分段改写。
先用 A1 与 ORIG 比较普通改写的影响，再看 A2～A4 相对 A1 的差异；
只有语义等价得到保证，才适合将差异解释为表达变化的影响。

完整方案为 **ORIG + FLIP + GOAL-ONLY + 四组 PARAPHRASE**，已有 ORIG 可复用。
对有随机性的系统，再跑一次相同配置的 ORIG，估计自然波动。

## 3. 怎么评分

| 指标 | 通俗解释 | 主要用途 |
|---|---|---|
| **SR：成功率** | 最后是否成功到达目标 | ORIG、GOAL-ONLY、PARAPHRASE |
| **SGCR：子目标完成率** | 按顺序完成了多少段指令对应的路径边界 | 检查过程执行 |
| **SGCR-eff：有路程限制的完成率** | 排除长时间绕路、碰巧经过边界造成的虚高 | 检查执行效率 |
| **转向匹配率** | 在指定位置，是否按当前指令向左或向右转 | FLIP |
| **nDTW / 中间边界到达率** | 路线与参考路径有多接近、是否经过中间目标 | 辅助解释行为变化 |

FLIP 同时报告到达转向位置的比例和两次运行均可判定的样本数。
比较时使用同一批任务，结合 bootstrap 95% 置信区间与重复 ORIG 的波动，
避免把随机差异当成指令作用。公式、转向窗口与半径见 [设计文档](../docs/subgoal_benchmark_design.md)。

## 4. 怎么运行、怎么看结果

从项目根目录执行；下面的命令使用已有轨迹重算指标，无需重新运行导航：

```bash
.venv/bin/python -m benchmark.metrics \
  --orig benchmark/results/v19/orig \
  --flip benchmark/results/v19/flip \
  --goalonly benchmark/results/v19/goalonly \
  --json /tmp/v19_metrics.json --csv /tmp/v19_per_episode.csv
```

- [运行指南](../docs/benchmark_usage.md)：数据构建、模型运行、PARAPHRASE 导入与目录说明。
- [实验报告](results/REPORT.md) / [系统比较](results/COMPARISON.md)：已有结果及分析。
- `results/{v19,canav,awarevln}/`：各系统的轨迹、日志和 `metrics*.json`。

PARAPHRASE 每组的轨迹分别用 `metrics --orig <该组输出目录>` 评分，
复用同一份子目标边界，再比较 ORIG、A1 与其他组。
CA-Nav 使用离线指令解析，运行 PARAPHRASE 前必须重新解析四组指令。

## 5. 当前状态和限制

- **已完成**：v19、CA-Nav、AwareVLN 的 ORIG、FLIP、GOAL-ONLY；v19 另有重复 ORIG。
- **PARAPHRASE**：四组数据已生成并通过自动校验，已有盲审记录；AwareVLN 四组各 200 条导航评测已完成。CA-Nav 正使用 Qwen 重新解析，并补跑同解析器的 ORIG 对照；结果单独保存到 `results/canav_qwen/`。
- **语义质量待复核**：抽样发现 A4 中任务 903、1455 有属性弱化或新增属性的疑点。
  自动校验及同模型家族盲审不能保证语义等价，详见 [数据审查记录](../docs/benchmark_data_review.md)。
- **FLIP 限制**：样本较少；选样与评分的转向几何有已知差异，详见设计文档。
  GOAL-ONLY 的表现变化也不能单独证明是否过拟合，需结合路径指标解释。
