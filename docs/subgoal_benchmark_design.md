# Subgoal 级指令跟随评测 Benchmark 设计

状态:v0.4(2026-09-17),代码在 `benchmark/`(用法见 `benchmark/README.md`)。只评一件事:**agent 的行为是否由指令的每一句决定**。不评 agent 的内部进度信念,不要求被测系统输出任何额外信息,只看轨迹。

## 1. 动机

R2R / R2R-CE 只报终点指标,一条轨迹里"哪句指令在哪段路被执行"不可见,而终点指标可以被与指令无关的策略刷高:CA-Nav 靠不早停拿到 SR 0.243;AwareVLN val_unseen SR 0.65 但真机上换指令分布后几乎没有指令泛化。要区分"听懂了指令"和"学到了 R2R 路径先验",必须测**分数对指令的敏感度**和**失败位置与被改句子的对应关系**,而不是绝对分。

## 2. 术语

| 名称 | 定义 |
|---|---|
| episode | R2R-CE 记录:场景、起点位姿、指令、参考节点路径 `reference_path`、稠密参考路径 `locations`(`{split}_gt.json.gz`,本地 `habitat/data/datasets/vln/mp3d/r2r/v1/val_unseen/`)。|
| subgoal k | 子指令 s_k 与参考子路径 P_k,k = 1..K,来自 FGR2R 的人工切分。|
| 边界 B_k | P_k 末节点为圆心、半径 r_b 的圆。r_b 由 5 节自检标定,初值 1.5 m。B_K 用官方 3 m 成功半径。|
| 进入事件 e_k | 从 e_{k-1} 起(含)轨迹首次进入 B_k 的步号;未进入记 ∞,之后的 e 全部记 ∞。顺序搜索保证"先到 B_3 再回头到 B_2"不算完成 3,也保证 B_K 的 3 m 半径不会在到达 B_{K-1} 之前被提前触发。|

## 3. 数据构建

**来源**:FGR2R(Hong et al., EMNLP 2020)对 R2R train / val_seen / val_unseen 提供人工对齐的子指令与子路径节点区间。R2R-CE 的 `reference_path` 是同一 trajectory 的节点位置列表,节点区间直接映射,不需要新标注。

**范围**:`integrations/v3/eval_sets/val_unseen_200.txt` 的 200 条,FGR2R 覆盖 200/200。R2R-CE 的 `start_position + reference_path + goal` 与 FGR2R 的 viewpoint 序列一一对应(用 MP3D connectivity 图核对过)。同一 viewpoint 结尾的相邻 chunk 合并;合并后 K 均值 2.94,K=1 的 24 条不参与 DROP-k。

**产出**:`benchmark/data/subgoals_val_unseen_200.json`,每条含 K、子指令原文片段(按 FGR2R 词元对齐回原句,保留大小写与标点)、节点区间、B_k 圆心、边界弧长。

**诊断变体**(不改代码,只换指令文件):

| 变体 | 生成 | 备注 |
|---|---|---|
| ORIG | 原指令 | 基准 |
| SWAP | 换成同场景另一条 trajectory 的指令,起点与参考路径不变;优先选起点距离 ≤ 0.5 m 的 donor(200 条里 128 条有),否则取起点最近的 | 被换入的指令自带它自己的参考路径 P' 与边界 B'_k,用于 4.3 |
| DROP-k | 删掉第 k 句,k 在 1..K−1 均匀采样,每条 episode 一个变体;最后一句(含停止条件)不删 | 切分直接用 FGR2R,不用 LLM;176 条 |

变体写成额外的 habitat split(`val_unseen_200_swap` / `val_unseen_200_drop`),episode id 不变,runner 用 `SPLIT=` 切换,三次运行的 trace 按 id 对齐。

## 4. 指标

### 4.1 逐段完成判定

subgoal k 完成,当且仅当 e_1..e_k 全部有限,即前 k 个边界按顺序依次被进入(e_k 的搜索从 e_{k-1} 开始,允许同一步)。顺序约束是核心:先到 B_3 再回头到 B_2 不算完成 3;半路经过 B_k 再去别处也不算。

由此得每条 episode 的**完成前缀长度** c = max{k : subgoal 1..k 全部完成},c ∈ 0..K,以及**首次失败段** f = c + 1(c = K 时无失败)。

### 4.2 SGCR 与逐段曲线

- **SGCR** = mean(c / K),前缀式子任务完成率。
- **SGCR-eff**:同上,但第 k 段只有在 e_{k-1} 到 e_k 之间走过的路程 ≤ max(2 × 参考段长, 3 m) 时才计入,超预算的段终止前缀。CA-Nav 这类走满 250 步、路程 30 m 的 agent 在小房子里会顺路按序经过所有边界,SGCR 会虚高,SGCR-eff 用来剔除这种"乱逛撞上"的完成。ISens 同时报两版。
- **SGCR@k** = P(c ≥ k),按段序号报一条曲线。这条曲线的形状本身有信息:靠先验的模型在 k = 1 已经低(第一句往往是"转身"类,先验帮不上),而在后段与 SR 接近;真正逐句执行的模型曲线平滑下降。
- **SR 与 SGCR 的差**:SR 高而 SGCR 低,说明终点是靠先验或最短路到的,不是沿指令到的。

### 4.3 ISens:指令敏感度

对每条 episode 配对比较 ORIG 与 SWAP:

- **ISens-SGCR** = SGCR(ORIG) − SGCR(SWAP),SWAP 下 c 按原路径的 B_k 算。理解指令的模型应显著为正;靠先验的模型接近 0。
- **PathAttrib(路径归属率)**:SWAP 下轨迹到换入指令参考路径 P' 的 nDTW 记 d',到原参考路径 P 的 nDTW 记 d。PathAttrib = P(d' > d)。它比 ISens-SGCR 更灵敏:一个模型即使两条指令都没走完,只要它听指令,轨迹就会偏向 P'。听指令的模型接近 1,完全不听的接近 0(自检里最短路 oracle 为 0.01)。同时单独报 donor 起点 ≤ 0.5 m 的 128 条,排除起点不同带来的偏置。
- **SGCR'(SWAP 指令自身完成率)**:SWAP 下按 B'_k 算的前缀完成率。与 ORIG 的 SGCR 对照,量化"换一条同场景指令后执行能力保留多少"。

三个数配对报,附 bootstrap 95% 置信区间(200 条 episode 重采样 1000 次)。

### 4.4 LocFail:局部失败定位率

对 DROP-k 变体:

- 定义 f_drop 为 DROP-k 下的首次失败段。
- **LocFail** = P(f_drop = k | c_orig ≥ k),条件在原指令下能走到第 k 段(否则第 k 段本来就失败,不能归因于删句)。
- **前缀保持率** = P(c_drop ≥ k−1 | c_orig ≥ k−1),删第 k 句不应影响前 k−1 段。低于 0.9 说明模型不是逐句执行,而是整条指令一起编码。
- **跳段率** = P(c_drop ≥ k | c_orig ≥ k),删了第 k 句仍然"完成"了第 k 段。这就是先验的直接证据:没有指令也走到了该走的地方。
- 随机基线:LocFail 的随机期望 ≈ 1/K̄(K̄ 为平均段数,R2R 约 3 到 4),报表附上。

### 4.5 辅助

沿用现有 SR、OSR、SPL、nDTW、SDTW、PL、Steps,只作为对照列。

## 5. 自检(上线前必过)

两个假 agent,只生成轨迹,不跑模型:

1. **参考路径 agent**:ORIG 沿 `locations` 走;SWAP 沿 donor 的 `locations` 走;DROP-k 走完第 k−1 段就停。位置加 0.3 m 高斯噪声重复 20 次标定 r_b。
2. **最短路 oracle**:不看指令,三个变体下都沿 ORIG 的 `locations` 走。

实测(`python3 -m benchmark.selfcheck`):r_b ≥ 0.75 m 时噪声参考 agent SGCR = 1(默认仍用 1.5 m,真实 agent 偏离更大);参考 agent ISens 0.54 / PathAttrib 1.00 / LocFail 0.94 / Skip 0.06,oracle ISens 0.00 / PathAttrib 0.01 / LocFail 0.00 / Skip 1.00,两者 SR 均为 1。顺序搜索改掉之前的"全轨迹首次进入"定义后才通过:原定义下 B_K 的 3 m 半径会先于 B_{K-1} 触发,无噪声参考 agent 有 69 条失败。



## 6. 被测系统与第一张表

被测:v19、AwareVLN,各跑 ORIG / SWAP / DROP-k 三遍 200-set。v19 ORIG 已有(ring6_200 trace,结果在 `benchmark/results/v19_ring6_200_orig.txt`:SR 0.255,SGCR 0.471,SGCR@1..4 = 0.665 / 0.500 / 0.361 / 0.348),补两遍。

第一张表,每系统一行:

SR | SGCR | SGCR-eff | ISens-SGCR | ISens-SGCR-eff | PathAttrib | LocFail | 前缀保持率 | 跳段率

外加两张图:SGCR@k 曲线(两模型叠加);每条 episode 的 (d, d') 散点。

**判据**:AwareVLN 的 ISens-SGCR 与 PathAttrib 明显低于 v19、跳段率明显高,则 benchmark 价值成立。两者分不开则回头改指标,不扩范围。

## 7. 进度

| 步骤 | 内容 | 状态 |
|---|---|---|
| 1 | FGR2R 映射 200-set,`benchmark/data/subgoals_val_unseen_200.json` | 完成 |
| 2 | 进入事件、c/f、SGCR、SGCR@k,在 ring6_200 trace 上跑通,`benchmark/metrics.py` | 完成 |
| 3 | 两个假 agent 自检,r_b 标定,`benchmark/selfcheck.py` | 完成 |
| 4a | SWAP / DROP-k 变体 split 与 id 文件,`benchmark/build_variants.py` | 完成 |
| 4b | CA-Nav 三遍(`benchmark/canav.py` 适配:SWAP 用 donor 的 GPT 解析,DROP-k 删掉与 FGR2R 第 k 段重叠的 GPT 子指令,171/176 条可映射;evaluator 补了轨迹导出) | 2026-09-17 21:24 启动,exp_bench_{orig,swap,drop} |
| 4c | v19 SWAP / DROP-k(2026-09-18 02:38–17:05,单 JoyAI 副本) | 完成 |
| 5 | 第一张表:`benchmark/results/COMPARISON.md`(CA-Nav 与 v19 原始指标 + 新指标 + 读法) | 完成 |
| 6 | AwareVLN 三遍(`benchmark/awarevln.py` 适配,Position measure + 轨迹导出) | 2026-09-19 完成 |
| 7 | **FLIP-k 主实验**(`benchmark/build_flip.py`,47 集;转向指标在 `metrics.py evaluate_flip`,参考 agent 1.00)三个系统 + v19 噪声对照 | 2026-09-19 完成,见 `benchmark/results/COMPARISON.md` |

## 8. 已做的修订(2026-09-19)

第一张表暴露了两个设计问题:DROP-k 被房屋布局混淆(被删段多数只有一条路,任何 agent 都会"照样走到");PathAttrib 被 donor 共享前缀稀释。改为以 **FLIP-k** 为主实验:在 k ≥ 2、参考路径确实转弯 ≥ 45°、反方向可通行的句子里反转左/右,指标为到达锚点后的转向是否跟随反转的词,并用同系统两次 ORIG 之间的差异作噪声底线。SWAP / DROP 降为辅助。

## 9. 可扩的内容

- **RESET 协议**:每段从 B_{k-1} 圆心与参考朝向单独起跑,与 E2E 的差值即误差累积代价。需要 runner 支持指定起点与子指令。
- **子指令类型标签**(TURN / PASS / ENTER / ORDINAL / STOP)按类型报 SGCR 与 LocFail,回答"哪类指令不会执行"。
- **更多变体**:FLIP-k(方向或序数反转,应在第 k 段失败)、REWRITE-k(同义改写,SGCR 应不变)、RECOMB(同场景拼接子路径,打破长度先验)。
- **自建场景**:扫自己的楼建 Habitat 场景,同一模板写 20 到 40 条路线,对所有模型都是 unseen 分布,并与真机共享真值。

## 10. 开放问题

- SWAP 换入指令的起点与原 episode 相同,但换入指令的第一句可能默认了不同的初始朝向。可限制只从起点相同(同 trajectory 起点节点)的 episode 里换,R2R 每条路径有 3 条指令但那是同路径,不能用;需要看同场景同起点节点的不同路径有多少。
- r_b 在参考路径加噪声上标定,真实 agent 偏离分布更宽,第一张表出来后复核。
- FGR2R 对 val_unseen 200-set 的覆盖率未知,第 1 步先看。
