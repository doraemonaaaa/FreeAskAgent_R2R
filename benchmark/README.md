# Subgoal 级指令跟随 benchmark

设计文档:`docs/subgoal_benchmark_design.md`。只看轨迹,不要求被测系统输出任何额外信息。

## 布局

```
benchmark/
  common.py              唯一的共享层:路径(data_path/ids_path/gen_dir)、R2R-CE 与稠密
                         参考路径加载、runner 输出解析、write_split、转向几何、nDTW
  build_subgoals.py      FGR2R 人工切分 -> subgoal 边界 B_k  ← 全部变体与指标的地基
  build_variants.py      SWAP / DROP-k          ─┐
  build_flip.py          FLIP-k(主实验)         ├ 四个变体 builder,彼此不依赖
  build_goalonly.py      GOAL-ONLY               │  (只依赖 common + subgoals.json)
  build_paraphrase.py    PARAPHRASE 四个 arm    ─┘
  metrics.py             全部指标 + bootstrap 95% CI
  selfcheck.py           参考路径 agent / 最短路 oracle 两个假 agent,r_b 标定
  compact_trace.py       归档 runner 输出:只留指标需要的字段(v19 的 trace 压 ~280x)
  canav.py awarevln.py   两个外部系统的输入生成与 trace 导入

  data/
    fgr2r/                 第三方源标注(Hong et al. 2020)+ LICENSE
    <eval set>/            每个 eval set 一个目录
      subgoals.json          B_k 边界(FGR2R 人工标注派生)— rule 与 llm 共同的地基
      rule/                  规则构造:最小对,可以直接和 ORIG 比
        swap_drop.json         SWAP donor + DROP-k
        flip.json              FLIP 选段 + 淘汰统计
        goalonly.json          保留句 + 被删前缀
        ids/{swap,drop,flip,goalonly}.txt
      llm/                   LLM 生成:不是最小对,只能相对 para_id 读
        paraphrase.json        四个 arm 的文本 + span + generator
        paraphrase_gen/        逐 chunk 生成结果 batch_*.json(paraphrase.json 的来源)
        ids/{para_id,para_terse,para_natural,para_lm_shift}.txt

  results/               见下方"results 布局"
```

加一个新 eval set 就是多一个 `data/<set>/` 目录,不是多十几个平铺文件。变体按**构造方式**分进
`rule/` 和 `llm/`,因为这决定了它们**能怎么读**(见下方 provenance)。

代码里的路径一律走 `common.data_path(kind, name)` / `ids_path(variant, name)` / `gen_dir(name)`,
家族归属由 `common.FAMILY` / `VARIANT_FAMILY` 查表决定,默认 `DEFAULT_SET = "val_unseen_200"`;
拼错 kind 或 variant 会直接抛 `KeyError` 并列出已知值,不会悄悄写到错地方。

habitat split 写在 `<r2r data>/<set>_<variant>/`,变体名:`swap` `drop` `flip` `goalonly`
`para_id` `para_terse` `para_natural` `para_lm_shift`。


### 数据来源(provenance)

目录名 `val_unseen_200` 是 **eval set 名**(对应 `integrations/v3/eval_sets/val_unseen_200.txt`),不是某一种构造方式。同一个 eval set 下既有规则派生的变体,也有 LLM 生成的,**区别是方法论上的**,所以它们分在 `rule/` 和 `llm/` 两个目录里,每个文件的顶层也自报来源:

| 文件 | `provenance` | 是最小对吗 |
|---|---|---|
| `subgoals.json` | FGR2R 人工标注派生 | — (地基) |
| `flip.json` | 规则:只换一个 left/right,其余逐字节不变 | **是** |
| `swap_drop.json` | 规则:donor 原文整条复制 / 按 span 删句 | 是(删除型) |
| `goalonly.json` | 规则:从原文逐字切出最后一句 | 是(删除型) |
| `paraphrase.json` | **LLM 改写**,逐 chunk;另有 `generator` 字段 | **否** |

`paraphrase.json` 额外记 `generator`(当前 `["Claude Opus 5"]`,`ingest` 时从 batch 文件收集,多个生成器会全部列出)。**它不是最小对**——措辞整体变了,所以 A2/A3/A4 的效应只能相对 A1(para_id)读,A1 的下降就是改写流程本身的混淆地板。规则变体没有这个问题,可以直接和 ORIG 比。

## 数据事实(200-set)

- FGR2R 覆盖 200/200。R2R-CE 的 `start_position + reference_path + goal` 与 FGR2R 的 viewpoint 序列一一对应(已用 connectivity 图核对)。
- 同一 viewpoint 结尾的相邻 chunk 合并后,K 均值 2.94,K=1 的 24 条不参与 DROP-k,DROP 集 176 条。
- SWAP donor:200 条都有同场景 donor,其中 128 条 donor 起点与自身起点距离 ≤ 0.5 m;`PathAttrib(same-start donors)` 只在这 128 条上算。
- 相邻边界间距最小 0.25 m,44 对 < 1 m。判定用顺序搜索(e_k 从 e_{k-1} 之后找),所以近距离边界不会因半径重叠而乱序。
- r_b 标定:参考路径加 0.3 m 高斯噪声,r_b ≥ 0.75 m 即 SGCR = 1;默认用 1.5 m,真实 agent 偏离更大。


## results 布局

```
results/
  REPORT.md            实验报告(FLIP 主实验 / GOAL-ONLY / SWAP-DROP 附录)
  COMPARISON.md        三系统汇总表
  goalonly_overlap.json  GOAL-ONLY 成功集重叠与难度分析
  {v19,canav,awarevln}/
    orig/ swap/ drop/ flip/ goalonly/   各轮的 rank_*.log 与 rank_*_trace.jsonl(原始证据)
    metrics.json / metrics.txt          ORIG + SWAP + DROP 指标
    metrics_flip.json                   FLIP-k 指标
    metrics_goalonly.json               GOAL-ONLY 指标
    per_episode.csv                     逐集 K / c / f / sr(各变体)
  v19/orig2/, v19/metrics_noise_floor.json   v19 的噪声底线(第二次 ORIG,仅 47 集 FLIP 子集)
```

指标文件都可以从 trace 重算,例如
`python3 -m benchmark.metrics --orig results/canav/orig --flip results/canav/flip --json results/canav/metrics_flip.json`。



### 一处已知的几何不一致(**未修**,需要决定)

FLIP 的**选段**几何(`build_flip`)与**打分**几何(`metrics.turn_at_anchor`)不是同一套:前者用 2D 弧长累加走入射窗口(且多走一段),后者用 3D 累计弧长。在 47 集上两者的入射朝向有 **4 集不同,最大差 5.5°**。

`metrics` 原来的 docstring 写着 "Same rule as the selection in build_flip",**这句是错的**,已订正。

统一到 `build_flip` 那套(定义了已发布 FLIP 集的那套)会让 **AwareVLN 的 3 个指标位移约 2.5 个点**(TurnMatch_orig 0.784→0.811、TurnMatch_flip 0.342→0.317、TurnChanged 0.457→0.486),v19 与 CA-Nav 不变。因为已跑完的实验和报告用的是现状数字,**没有擅自改**——要改就两边一起改并重算报告。

## 数据审查(2026-09-20)

生成数据和原始数据都过了一遍,结论与已知缺陷记在这里。

### 生成数据(llm/)

机器闸门之外,按 arm 统计"每个 chunk 保留了多少原文实义词",验证操作方向是否真的发生:

| arm | 保留率 | 全换掉的 chunk | 预期 |
|---|---|---|---|
| para_id | 97.7% | 2/566 | 高(保义)✓ |
| para_terse | 95.7% | 5/566 | 略低(精简)✓ |
| para_natural | 97.9% | 3/566 | 高 ✓ |
| para_lm_shift | **27.0%** | **263/566** | 低(刻意换掉中心名词)✓ |

A4 里 51 个 chunk 没换任何词,全是 `Turn right.` 这类纯方向句,没有 landmark 可换。

**查出并修掉的缺陷**:ep 1569(A2/A3)、1585(A2)、1772(A2)把**末句**的 landmark 换成了代词(`Stop in the bedroom.` → `Halt there.`)。末句承载停止条件,这等于削弱了该 arm 的目标信息。已改回具名,并加了永久闸门:**非 A4 的末句必须保留至少一个原实义词**(模糊匹配,容忍 `dooryway`→`doorway` 这类拼写归一;`BARE_STOP` 句豁免,因为原文本就没 landmark)。

**已知且保留的偏差**:A1/A2/A3 会把原文的明显拼写错误normalize(`dooryway`→`doorway`、`immediatly`→`immediately`、`stair case`→`staircase`)。这是改写流程的一部分,由 para_id 对照组一并吸收。

**闸门抓不到的**:用原有词汇拼出错误意思。**已做盲审**(见下)。

#### 盲审(2026-09-20)

四个 arm 的 800 条改写 + **90 条隐藏对照**打散成 6 片,交给 6 个全新上下文的审查者:不给 arm 标签、不说明文本由 LLM 生成、不暗示期望答案。每条判 `same` / `weaker` / `different`,外加 `direction_changed`、`referent_changed` 两个独立布尔量。原始判定归档在 `llm/review/`(`out_*.json` + `key.json` + `score.py`,可重跑)。

**先看对照项**——审查者不知道这 90 条是对照,它们决定了盲审结果能不能采信:

| 对照 | n | 命中 |
|---|---|---|
| 同一句 vs 自己(应判 same) | 30 | 30/30 |
| 换成另一集的整条指令(应判 different) | 30 | 30/30 |
| 左右词全部反转(应判 different) | 30 | 30/30,`direction_changed` 30/30 |
| 对同一文本的误报 | 30 | **0/30** |

既不放水也不乱报,且对方向变化完全敏感。

**真实 800 条的判定**:

| arm | n | same | weaker | different | dir_chg | ref_chg |
|---|---|---|---|---|---|---|
| para_id | 200 | 200 | 0 | 0 | 0 | 0 |
| para_terse | 200 | 200 | 0 | 0 | 0 | 0 |
| para_natural | 200 | 200 | 0 | 0 | 0 | 0 |
| para_lm_shift | 200 | 199 | **1** | 0 | 0 | **0** |

**A4 的 200 条里 0 条被判指向了另一个物体** —— 这正是机器闸门原理上看不到、盲审专门要查的一层。审查者把 `kitchen`→`cooking area`、`pool table`→`green felt games surface`、`eye chart`→`vision-testing poster` 一律认定为同物换描述,符合 A4 的设计意图。

**查出并修掉的唯一缺陷**:ep 1329 的 A4 把 `white flowers` 写成 `pale blooms`,模糊了终点 landmark 的区分性颜色。A4 只该换中心名词,`white` 是区分性属性,已改回 `white blooms`。

**残留限制**:审查者与生成者是同一模型家族(本地 Qwen3-VL 端点全部离线,8 张 GPU 被他人长任务占用,模型级独立这次做不到)。对照项证明它不是橡皮图章,但**同家族可能共享同一盲点**——如果我系统性地误描述某一类物体,审查者可能犯同样的错。要彻底闭环需要换一个模型家族或人工抽检重跑。

### 原始数据(R2R-CE + FGR2R)

- **边界几何干净**:389 对相邻边界,无零间距、无倒退,最小 0.25 m,11% 小于 1 m;末边界 B_K 与 episode goal **200/200 重合**。
- **13 集的 B_1 落在起点上**,第一段等于没走。这些集的 SGCR@1 恒为 1,把全集 SGCR@1 抬高了 v19 +0.023 / CA-Nav +0.016 / AwareVLN +0.009。量级小,但 SGCR@1 的绝对值要这么读。
- **5 集是罗列式指令**(ep 31/161/288/503/657,`Go to the plant / Go to the rope / ...`,其中 3 集原文带 `\r\n`),是路点清单而非路线描述,与其余 195 集不同分布。v19 在这 5 集上 SR 0.600 vs 其余 0.246(n=5,P(X≥3)=0.10,单看不显著),CA-Nav 0.400、AwareVLN 0.400 vs 0.631。样本太小不能下结论,但这是 eval set 里已知的异质成分。
- **原文拼写错误**(各 1 集):`kitche`(1023)、`dinning`(75)、`dooryway`(911)、`immediatly`(1544)、`dink`(869)、`cock`(1382),以及 `nest to`(45)、`Had past`(778)、`Exit and`(697)。**一律保留不改** —— 它们是被测系统真实要面对的输入,改了就不是 R2R 了。

## 精简协议(2026-09-19 起)

每个系统:ORIG 一次(复用)+ **FLIP 一次(主指标)** + GOAL-ONLY 一次(过拟合诊断)。随机性系统另跑一次 ORIG 作噪声底线,所有变体共用。SWAP / DROP 是第一版指标,保留作附录,新系统不必跑。报告见 `results/REPORT.md`。


## PARAPHRASE 套件(2026-09-20 新增)

回答报告第 5 节留下的问题:**指令风格换了还跟得住吗**。现有变体除 FLIP 外全是"删信息",没有一个测"同样的信息换个说法"。

**按 chunk 改写,不整条重写**:每个子指令单独改,再拼回去。K、chunk→viewpoint 对齐、边界 B_k 全部不变,所以 SGCR / SGCR-eff / per-hop survival / 中间边界率**整套指标一行不改就能用**。起点、目标、参考路径、GT 逐字节不变(已核对 200/200)。

| arm | split | 内容 | 词数 | 与原文词汇 Jaccard |
|---|---|---|---|---|
| ORIG | `val_unseen_200` | 原文 | 25.1 | 1.000 |
| **A1 para_id** | `val_unseen_200_para_para_id` | 保义改写,同语域。**对照组** | 26.9 | 0.666 |
| A2 terse | `val_unseen_200_para_terse` | 祈使短句,像指挥机器人 | 19.6 | 0.626 |
| A3 natural | `val_unseen_200_para_natural` | 口语化、啰嗦、带填充词 | 37.1 | 0.538 |
| A4 lm_shift | `val_unseen_200_para_lm_shift` | 同一物体,换掉原来的中心名词 | 29.2 | 0.346 |

**A1 是命脉**:它的 SR 下降 = 改写流程本身引入的混淆地板(同 v19 的噪声底线思路)。A2/A3/A4 相对 A1 的**额外**下降才可归因。没有 A1 的数据,其余三个 arm 不可解释。

**待验证的假设**(先写死,避免事后编故事):AwareVLN(R2R 上 SFT)在 A2/A3 的降幅显著大于 v19 与 CA-Nav(零样本 LLM)。成立则"零样本对指令风格更鲁棒"第一次有量化证据;不成立则该假设应当放弃。用 `reach-all x stop-correct` 二因子分解看降幅落在哪个因子。

**校验闸**(`check_episode`):chunk 数、非空、**left/right 词序列必须逐字相同**(硬性;否则任何 arm 都会悄悄变成一次 FLIP)。新增 landmark 名词只报告不拒绝(A4 预期会有)。生成期间这道闸抓到两类真实错误:非方向义的 `right`(`stop right there`)和 `left` 的过去式义(`after you have left the bathroom`)。最终 200/200 通过,0 硬拒;138 条 note 人工过了一遍,53 个"新词"全是虚词或良性换词(door→doorway、area→space),**0 个幻觉 landmark**。

生成器是 Claude Opus 5,与三个被测系统均不同族(v19=Qwen3-VL-8B、CA-Nav=GPT-4 离线解析、AwareVLN=SFT 8B),避免给任何一方送分。

```bash
python3 -m benchmark.build_paraphrase export    # 写生成任务(改 eval set 时重跑)
python3 -m benchmark.build_paraphrase ingest    # 校验 data/paraphrase_gen/*.json 并写四个 split
```

**注意**:CA-Nav 运行时不读指令原文,只读离线 GPT-4 解析。paraphrase 改不了解析,**必须对四个 arm 重新跑它的解析器**,否则测的不是 CA-Nav 在新指令上的表现。这会把"CA-Nav 解析器的鲁棒性"纳入测量——这是对的,它本来就是 CA-Nav 的一部分,但报告里要讲明口径。

## 用法

```bash
cd /data/pengyh/workspace/FreeAskAgent_R2R

# 1. 数据(已生成,改 eval set 时重跑)
python3 -m benchmark.build_subgoals --ids integrations/v3/eval_sets/val_unseen_200.txt
python3 -m benchmark.build_variants --name val_unseen_200

# 2. 自检
python3 -m benchmark.selfcheck

# 3. 跑三遍(ORIG 已有则只补 SWAP / DROP);SPLIT 与 EPISODE_IDS 换成变体的
SPLIT=val_unseen_200_swap EPISODE_IDS=@benchmark/data/val_unseen_200/rule/ids/swap.txt \
  TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_swap> \
  bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh
SPLIT=val_unseen_200_drop EPISODE_IDS=@benchmark/data/val_unseen_200/rule/ids/drop.txt \
  TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_drop> \
  bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh

# 3b. FLIP-k(主实验):SPLIT=val_unseen_200_flip + benchmark/data/val_unseen_200/rule/ids/flip.txt,再跑一遍 ORIG 作噪声底线
#     python3 -m benchmark.metrics --orig <out_orig> --flip <out_flip>      # 转向指标
#     python3 -m benchmark.metrics --orig <out_orig> --flip <out_orig2>     # 噪声底线

# 3c. PARAPHRASE(四个 arm);A1 是对照组,必须跑
for ARM in para_id para_terse para_natural para_lm_shift; do
  SPLIT=val_unseen_200_$ARM EPISODE_IDS=@benchmark/data/val_unseen_200/llm/ids/${ARM}.txt \
    TRACE_JSONL=1 V3_CONFIG=<cfg.yaml> OUTPUT_DIR=<out_$ARM> \
    bash integrations/v3/run_r2r_ce_inference_only_multigpu.sh
done

# 4. 指标(nDTW 用 habitat 环境的 python(有 fastdtw);没有则退化为精确 DTW。**两者不完全相同**——fastdtw 是近似算法,实测 40 集里 1 集差 8.5e-4,均值差 ~2e-4,其余指标不受影响)
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
