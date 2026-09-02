# 实验结果总账(RESULTS.md)

评测:R2R-CE val_unseen,成功半径 3m,150 步预算(全量跑 500 步)。
**决策标准 = 200 分层集**(`integrations/v3/eval_sets/val_unseen_200.txt`,50/类,测地 4–14m);
40 集仅作冒烟。oracle = 轨迹上 min dtg ≤ 3m 的比例。
聚合工具:`python integrations/v3/bench/analyze_runs.py <run...> [--paired]`。
详细过程与失败归类:`2026-08-30-grounding-and-preview.md`(§ 号对应)。

## 一、我们的系统

### 旧管线(actor+captioner+judge+空间记忆,无 CWP)
| 跑名 | 模型 | 集数 | SR | SPL | oracle | 产物 | 文档 |
|---|---|---|---|---|---|---|---|
| full_val_unseen_pitch15_4b | 4B, pitch−15, 全录像 | **1839** | **0.091** | 0.065 | – | `outputs/full_val_unseen_pitch15_4b_20260830/`(1839 mp4, 3.4GB) | §9 |
| **base200(判定基准)** | 8B | 200 | **0.095** | 0.070 | ~0.15 | `outputs/experiments/ab/base200/` | §10 |
| lora200(som-v1 方向头) | 4B+LoRA | 200 | 0.085 | – | – | `outputs/experiments/ab/lora200/`;权重 `FreeAskAgent/models/lora-som-v1` | §17 |

失败结构(base200):卡地标 66%、错停 25%。40 集读数(4B 0.20 等)属小样本偏乐观,已退役。

### pano-hop 系列(CWP 环视 + VLM 选标 + hop 缓存;全部 200 集、8B)
| 跑名 | 机制 | SR | SPL | oracle | 判定 | 文档 |
|---|---|---|---|---|---|---|
| **panohop200(Phase 1 定型)** | 每步环视 + 裸 STOP + 1m 守卫 | **0.150** | **0.134** | 0.250 | ✅ 唯一大幅正收益(动作空间重构) | §20.2 |
| panohop200b | +双票确认+自由文本 remaining | 0.000 | – | 0.335 | ❌ 进度文本从不更新,守卫杀掉全部停 | §20.4 |
| panohop200c | 子目标指针 | 0.030 | – | 0.045 | ❌ 指针盲推进,连选路一起拖垮 | §20.5 |
| panohop200d | STOP 验证器+3票封顶 | 0.065 | – | 0.155 | ❌ 早停换游荡 | §20.6 |
| panohop200e | +指令感知重定向 | 0.109(n=175) | – | 0.196 | ❌ 仍不敌裸 STOP;rank_3 遇 502 崩(已加重试) | §20.6 |
| panohop200bt | 回溯选项每步提供 | 0.040 | – | 0.095 | ❌ 提示污染 + 误删进度回灌行(根因见 §21.1) | §21.1 |
| panohop200bt2 | 回溯仅失败后提供 | 0.145 | – | 0.240 | ❌ 回溯 0/2101 使用,中性 | §21.2 |
| panohop200sel | selective 环视(规则触发) | 0.125 | 0.113 | 0.245 | ◐ 环视 7.0/集 vs 10.5(−33% 预算 −17% SR,导航无损) | §20.7 |

关键机理:模型自身 progress 短语的逐决策回灌是承重墙(删掉 oracle 0.25→0.095);
LOOK 自选环视被否定(13/16 决策选看,不会自我节制)。

### CWP 注入旧管线系列(200 集、8B)
| 跑名 | 机制 | SR | SPL | oracle | 判定 | 文档 |
|---|---|---|---|---|---|---|
| **cwp200** | 旧管线 + CWP 候选替换 floor-openings | 0.110 | 0.077 | **0.365** | ✅ 到达能力全场最佳;❌ 停止转化率 30%(51 集到了没停对) | §23.1 |
| cwp200w | +stage 看门狗(8m,全录像) | 0.105 | 0.075 | 0.330 | ❌ 失败类别搬运(到了没停 38→29,早停 28→49);视频 `outputs/videos/cwp200w/` | §25 |

STOP 通道嫁接(S 票)机理级证伪:stage 追踪器停摆时过期子目标文本支配模型注意力,提示层无法绕过(§23.2)。

### 离线基准
| 基准 | 结果 | 产物 | 文档 |
|---|---|---|---|
| CWP 候选覆盖率(587 决策点) | 全环 96%/转向 95%;floor-openings 62%/28%;单目前扇区天花板转向仅 13% → 转向环视物理必需 | `outputs/experiments/cwp_coverage/` | §19.3–19.5 |
| 熵触发环视 | ❌ 否定:转向/直行熵 4.456 vs 4.447 无分离 | 同上 | §19.5 |
| som-v1 held-out | fwd 0.349→0.750,turn 0.127→0.274(单步选择学的是几何先验) | `FreeAskAgent/models/lora-som-v1/train.log` | §16–17 |

## 二、外部对照(同 benchmark)
| 系统 | 集数 | SR | SPL | 说明 | 位置 |
|---|---|---|---|---|---|
| **InternVLA-N1 S2(本机复现)** | 1839 | **0.585** | 0.538 | Qwen2.5-VL-7B 全参微调+DAgger;单目+低头像素目标;STOP 5×过采样,到达后停止精度 88.8% | `Reproductions/InternNav/logs/habitat/test_s2/` |
| InternVLA-N1(论文,双系统) | 1839 | 0.641–0.643 | 0.581–0.585 | NavDP/DualVLN 权重不在盘上 | InternNav README |
| AwareVLN(训练 8B) | 1839 | 0.647 | 0.563 | OS 0.744, NE 4.10;40 集同 0.65 | `Reproductions/AwareVLN`;§8 |
| SmartWay | 190 | 0.205 | – | 训练版 CWP+全景每步看,qwen3-vl-8b navigator | `Reproductions/SmartWay-Code/logs/eval_results/vu_merged` |
| Open-Nav | 100 | 0.10 | – | 原版 CWP,Qwen3-8B LLM | `Reproductions/Open-Nav/`(full100fix) |

## 三、核心结论(截至 2026-09-02)
1. 零样本天花板:动作空间重构值 +58%(0.095→0.150),此外所有规则/提示层修补
   (STOP×4、指针、验证器、回溯、看门狗)全部中性或负——**规则能决定"何时允许停",
   不能判断"这里是不是该停的地方"**。
2. 两套架构互补:cwp200 的腿(oracle 0.365)× pano-hop 的停止转化率(60%)≈ 0.22 上限。
3. 与训练系统的差距 = 训练本身:InternVLA-N1 配方(7B 全参、域内轨迹×4 形态、DAgger、
   STOP 同 vocab+5×过采样、低头像素目标、8 步杠杆)是下一阶段(Phase 3)的参考实现;
   其 S2 checkpoint 可作我们框架内基线(原生协议接入)或训练 init。
