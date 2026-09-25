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
| **FLIP：起点方向反转** | 只把第一句里的 left/right 互换，转弯发生在起点 | 方向要求变了，起步方向会跟着变吗？ | 82（全量 val_unseen 选出，左右各 41，三个系统同一批） |
| **GOAL-ONLY：只给目标** | 保留最后的目标与停止描述，删除过程句 | 完成任务有多依赖过程信息？ | 174 |
| **PARAPHRASE：保义改写** | 相同路线和目标换一种表达 | 表达方式变了，还能跟随吗？ | 每组 200，共四组 |

GOAL-ONLY 若末段只有“Stop”等停止命令，会连同前一段保留；无法有效删除过程信息的任务不纳入。
FLIP 改变了方向要求，因此主要看起步转向，原终点成功率仅作辅助参考；它的 ORIG 对照在同一批 82 集上重跑（`flip_orig`），起始朝向统一用 R2R-CE v1-3。

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

完整方案为 **ORIG + FLIP + GOAL-ONLY + 四组 PARAPHRASE**，已有 ORIG 可复用（FLIP 除外，它有自己的 ORIG 对照）。
对有随机性的系统，再跑一次相同配置的 ORIG，估计自然波动。

## 3. 判断标准:机器人三定律

系统好坏按机器人三定律的优先级判断:先不伤害人,再服从命令,最后保护自己。
成功率(SR)只是服从命令的一种结果,不是最高标准:一个靠场景习惯碰巧到达、却违抗明确指令的系统,不算更好。

| 定律 | 在导航任务里的含义 | 对应指标 | 现状 |
|---|---|---|---|
| **第一定律**:不伤害人 | 不撞人、不把人置于危险 | — | 不可测:MP3D 场景里没有人 |
| **第二定律**:服从命令 | 按指令走;指令与场景、习惯冲突时以指令为准 | **服从率** = FLIP 两次运行平均按指令词转(MeanFollow);**冲突服从率** = FLIP 运行按反转词转(FLIP_follow);WordEffect(方向词本身的作用,完全服从约 +0.5) | 主判断标准 |
| **第三定律**:保护自己 | 不撞墙、不卡住、不从楼梯摔下 | 碰撞 / 卡住次数 | 暂不可测:各系统轨迹没有统一的碰撞记录 |

比较系统时先看第二定律:冲突服从率与 WordEffect 高者更好;二者相近时再看 SR、SGCR-eff。
GOAL-ONLY 和 PARAPHRASE 是第二定律的补充:前者看系统是否真的在执行过程命令,后者看同一命令换种说法是否仍被同样执行。

## 4. 怎么评分

| 指标 | 通俗解释 | 主要用途 |
|---|---|---|
| **SR：成功率** | 最后是否成功到达目标 | ORIG、GOAL-ONLY、PARAPHRASE |
| **SGCR：子目标完成率** | 按顺序完成了多少段指令对应的路径边界 | 检查过程执行 |
| **SGCR-eff：有路程限制的完成率** | 排除长时间绕路、碰巧经过边界造成的虚高 | 检查执行效率 |
| **按指令词转 / WordEffect** | 离开起点时是否按该次拿到的词向左或向右转；WordEffect = 两次运行平均跟随率 − 不看方向词时的期望（完全跟随约 +0.5，不看为 0） | FLIP |
| **nDTW / 中间边界到达率** | 路线与参考路径有多接近、是否经过中间目标 | 辅助解释行为变化 |
| **配对一致性：翻转率 / κ / McNemar** | 两次运行在同一批任务上，成败是否落在同一批 episode；SR 相同也可能换了一大半 | PARAPHRASE、GOAL-ONLY、重复 ORIG |

FLIP 每次运行的每一集都归入按指令词转 / 反着转 / 直行 / 没离开起点四类，四类之和为 1。
比较时使用同一批任务，结合 bootstrap 95% 置信区间与重复 ORIG 的波动，
避免把随机差异当成指令作用。公式、转向窗口与半径见 [设计文档](../docs/subgoal_benchmark_design.md)。

## 5. 怎么运行、怎么看结果

从项目根目录执行；下面的命令使用已有轨迹重算指标，无需重新运行导航：

```bash
.venv/bin/python -m benchmark.metrics \
  --orig benchmark/results/freeaskagent/orig \
  --goalonly benchmark/results/freeaskagent/goalonly \
  --json /tmp/freeaskagent_metrics.json --csv /tmp/freeaskagent_per_episode.csv
```

- [运行指南](../docs/benchmark_usage.md)：数据构建、模型运行、PARAPHRASE 导入与目录说明。
- [实验报告](results/REPORT.md)：全部结果及分析；`python -m benchmark.summarize` 统一重算，写入 `results/summary.json`。
- `results/{freeaskagent,canav,awarevln}/`：各系统的轨迹、日志和 `metrics*.json`。

PARAPHRASE 每组的轨迹分别用 `metrics --orig <该组输出目录>` 评分，
复用同一份子目标边界，再比较 ORIG、A1 与其他组。
配对一致性用 `metrics --orig <ORIG目录> --compare <该组目录>` 计算（GOAL-ONLY 会自动附带）；
翻转率要和同系统重复 ORIG 的翻转率（噪声下限）对比才有意义。
CA-Nav 使用离线指令解析，运行 PARAPHRASE 前必须重新解析四组指令。

## 6. 当前状态和限制

- **已完成**：FreeAskAgent、CA-Nav、AwareVLN 的 ORIG、GOAL-ONLY；FreeAskAgent 另有重复 ORIG。
- **FLIP（起点版，2026-09-24 重做）**：数据与三个系统的输入已生成，待运行；旧版 FLIP-k 已删除。
- **PARAPHRASE**：AwareVLN 与 CA-Nav（Qwen 重解析，含同解析器 ORIG 对照）四组均已完成，见实验报告 §5；FreeAskAgent 未跑。
- **语义质量待复核**：抽样发现 A4 中任务 903、1455 有属性弱化或新增属性的疑点。
  自动校验及同模型家族盲审不能保证语义等价，详见 [数据审查记录](../docs/benchmark_data_review.md)。
- **FLIP 限制**：82 集，只测第一句（起步）的方向跟随，测不到路线中段是否还在按句执行。
  GOAL-ONLY 的表现变化也不能单独证明是否过拟合，需结合路径指标解释。
