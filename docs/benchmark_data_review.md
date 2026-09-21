# Benchmark 数据审查记录（2026-09-20）

以下保留已有审查记录。自动检查和同模型家族盲审不等于语义等价保证；后续人工抽样发现了遗漏，见下方待复核项。

返回 [benchmark 总览](../benchmark/README.md)。

## 后续抽样：待复核项

固定随机种子 `20260920`，从 200 个任务中抽取 10 个，发现：

- **903 / A4**：`white table → pale outdoor surface` 弱化了颜色与物体类别；`yard → lawn`、`brick wall → masonry partition` 也不严格等价。
- **1455 / A4**：`atrium → skylit court` 加入了原文没有明确提供的采光特征。

这些是文本层面的语义疑点，尚未修订数据。需要复核 A4 的颜色、材质、物体类别及新增属性；不能将语义偏移解释为难度提升。

## 既有审查记录

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

在这 90 条对照上全部命中；这不能证明对真实改写中的细微语义偏移同样敏感。

**真实 800 条的判定**:

| arm | n | same | weaker | different | dir_chg | ref_chg |
|---|---|---|---|---|---|---|
| para_id | 200 | 200 | 0 | 0 | 0 | 0 |
| para_terse | 200 | 200 | 0 | 0 | 0 | 0 |
| para_natural | 200 | 200 | 0 | 0 | 0 | 0 |
| para_lm_shift | 200 | 199 | **1** | 0 | 0 | **0** |

**A4 的 200 条里 0 条被判指向了另一个物体** —— 这正是机器闸门原理上看不到、盲审专门要查的一层。审查者把 `kitchen`→`cooking area`、`pool table`→`green felt games surface`、`eye chart`→`vision-testing poster` 一律认定为同物换描述,符合 A4 的设计意图。

**该轮审查查出并修掉的缺陷**:ep 1329 的 A4 把 `white flowers` 写成 `pale blooms`,模糊了终点 landmark 的区分性颜色。A4 只该换中心名词,`white` 是区分性属性,已改回 `white blooms`。

**残留限制**:审查者与生成者是同一模型家族(本地 Qwen3-VL 端点全部离线,8 张 GPU 被他人长任务占用,模型级独立这次做不到)。对照项只能验证这些对照样例上的表现；**同家族可能共享同一盲点**——如果我系统性地误描述某一类物体,审查者可能犯同样的错。要彻底闭环需要换一个模型家族或人工抽检重跑。

### 原始数据(R2R-CE + FGR2R)

- **边界几何干净**:389 对相邻边界,无零间距、无倒退,最小 0.25 m,11% 小于 1 m;末边界 B_K 与 episode goal **200/200 重合**。
- **13 集的 B_1 落在起点上**,第一段等于没走。这些集的 SGCR@1 恒为 1,把全集 SGCR@1 抬高了 v19 +0.023 / CA-Nav +0.016 / AwareVLN +0.009。量级小,但 SGCR@1 的绝对值要这么读。
- **5 集是罗列式指令**(ep 31/161/288/503/657,`Go to the plant / Go to the rope / ...`,其中 3 集原文带 `\r\n`),是路点清单而非路线描述,与其余 195 集不同分布。v19 在这 5 集上 SR 0.600 vs 其余 0.246(n=5,P(X≥3)=0.10,单看不显著),CA-Nav 0.400、AwareVLN 0.400 vs 0.631。样本太小不能下结论,但这是 eval set 里已知的异质成分。
- **原文拼写错误**(各 1 集):`kitche`(1023)、`dinning`(75)、`dooryway`(911)、`immediatly`(1544)、`dink`(869)、`cock`(1382),以及 `nest to`(45)、`Had past`(778)、`Exit and`(697)。**一律保留不改** —— 它们是被测系统真实要面对的输入,改了就不是 R2R 了。

