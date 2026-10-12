## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-12
- Verification Status: UNVERIFIED（实验尚未运行；结果部分是先验范围估计）
- Version Label: experiment_plan_v1

# 实验设计与结果预估：层级奖励驱动的主动取证式 CWE 分类

## 1. 研究目标

本实验检验：在相同模型初始化、相同 CPG 工具和相同查询上限下，使用完整 CWE 层级 ALPHA 分数作为 GRPO 终局分类奖励，是否比 CWE 精确匹配奖励更能训练模型选择有用的代码查询，并改善最终 CWE 分类。

核心问题是层级奖励能否改善“查询—观察—分类”的策略，而不是重新证明 GRPO、CPG 工具或层级评估各自有效。Li 等已经研究了 CPG 工具和多轮 RL 漏洞检测；Adnan 等已经研究了单轮 CWE 分类的 ALPHA 奖励。因此本实验的窄贡献是二者交叉处：**多轮 CPG 探索中，完整的 CWE 层级距离是否比精确匹配反馈更适合作为策略优化信号。**

## 2. 预注册假设

- **H1（主要假设）**：在 PrimeVul 同一测试样本、最多 3 次查询的设置下，ALPHA-GRPO 的平均归一化 ALPHA 惩罚低于精确匹配 GRPO。
- **H2（查询行为）**：ALPHA-GRPO 在相同或略少的工具调用下取得更高的 ALPHA 分数；至少不应靠明显增加查询次数换取收益。
- **H3（分类护栏）**：CWE 改善不应伴随明显的二元 F1 退化。二元 F1 是评估护栏，不作为独立奖励项。
- **H4（外部泛化）**：在经过 CWE 标签审计、且与 PrimeVul 去重的 BigVul 子集上，收益方向应相同，但预计小于 PrimeVul，且不预设一定显著。

H1 是唯一主要检验；H2–H4 是次要或探索性检验。

## 3. 实验组

主实验采用三个最小条件，每个条件使用 3 个随机种子。每个种子先训练一个共同 SFT 起点，后续 GRPO 组都从该种子对应的同一 SFT checkpoint 初始化。

| 条件 | 训练 | CPG 工具 | CWE 奖励 | 作用 |
|---|---|---:|---|---|
| A：SFT 基线 | 普通监督微调 | 否 | 无 RL 奖励 | 衡量强化学习整体增益 |
| B：精确匹配 GRPO | SFT 后标准 GRPO | 是 | CWE 精确匹配 | 核心对照 |
| C：ALPHA-GRPO（方法） | SFT 后标准 GRPO | 是 | 完整归一化 ALPHA 层级分数 | 检验 H1；与 B 的奖励差异是主要自变量 |

所有组使用相同数据、提示模板、输出 JSON、工具定义、查询上限、模型和训练超参数。B 与 C 共用相同的证据和查询成本项，因此二者的主要区别只有 CWE 奖励：精确匹配或层级距离。

**可选但建议的工具消融**：增加一个“无工具 ALPHA-GRPO”条件。它与 C 使用相同 SFT 起点和终局分类奖励，只禁用 CPG 查询；这用于估计多轮环境带来的端到端收益。由于无工具策略无法取得工具返回的节点证据，该消融应被解读为“整套工具环境的系统效果”，不能单独声称是纯信息访问的因果效应。若要做严格工具效应比较，应额外提供 evidence 权重为 0 的配对版本，让有工具和无工具策略使用完全相同的分类与查询成本奖励。

## 4. 数据集和划分

### 4.1 PrimeVul：主要基准

- 使用 PrimeVul 官方 train/validation/test 划分；训练时只读取 train，选 checkpoint 和超参数只看 validation，test 只在最终评估时运行。
- 从有仓库 revision 信息的样本构建 Joern CPG。主要成对比较只使用 B、C 都能映射到同一仓库快照和函数节点的**共同可用测试子集**，避免工具覆盖率差异导致不公平；同时在完整官方测试集报告 SFT 基线结果。
- 报告 CPG 构建成功率、函数匹配率、不可用样本比例，以及按项目分组的结果。若可查询覆盖率低，结论必须限定在可构建 CPG 的子集，不能外推到完整 PrimeVul。
- 保留官方漏洞/修复配对关系；成对正确率作为补充指标。

PrimeVul 官方说明其采用更严格的标注、去重和时间切分，并覆盖 C/C++ 与 140 多种 CWE；v0.1 额外文件级元数据只覆盖成功补齐元数据的漏洞样本，因此实验应记录所用版本，避免把 v0.1 子集误称为全量数据。[PrimeVul 官方仓库](https://github.com/DLVulDet/PrimeVul)

### 4.2 BigVul：外部评估

- 主实验不把 BigVul 加进训练，先作为独立外部测试集，避免训练数据来源、规模和 CWE 标签噪声同时变化。
- 去除与 PrimeVul train、validation、test 重复的记录；优先使用规范化函数哈希，并利用可用的 CVE、提交、仓库/文件/函数位置元数据排查重复。所有去重规则和剔除数量写入结果。
- 二元指标可在全量去重样本上作为外部参考，但 CWE 指标只在人工复核标签的漏洞子集上报告。建议抽取约 300 个漏洞样本，按常见 CWE 分层；两位审阅者独立依据代码与修复上下文核对 CWE，分歧由第三次复核或讨论裁决。审阅者不查看模型预测。
- 将“只用 PrimeVul 训练”和“加入审计通过的 BigVul 训练样本”留作后续扩展，不混入第一轮主实验。

### 4.3 CWE 体系

冻结 MITRE CWE 4.20 XML，并在结果中保存 XML 文件 SHA-256 和 ALPHA 参数。训练奖励与评估必须使用同一份图快照。将测试 CWE 划分为训练中出现/未出现两组；未见 CWE 结果作为探索性分析，报告每类支持数，不用少数样本作强结论。

## 5. 环境与训练方法

### 5.1 只读多轮 Harness

使用预计算的 Joern CPG 快照，不运行模型生成的程序或漏洞利用代码。模型先看到目标函数；每次调用工具后得到新的代码观察及稳定节点 ID，最多查询 3 次，然后输出最终结果。

主工具维持简单的三种查询：`get_callers`、`get_callees`、`dataflow_slice`。模型最终输出固定 JSON：`verdict`、`cwe_id`、`evidence_node_ids`。Harness 记录每一轮的动作、参数、返回节点、查询次数和最终预测，便于复现和检查策略是否真的探索了新上下文。

这里 Harness 是多轮 RL 环境的一部分：若模型每轮只看到原始相同代码，就不能把重复生成文字称作主动探索。Li 等也显示，仅有终局判断的奖励可能导致不调查的策略；其 CPG 同时用于检索和证据核验。当前项目的证据分数更简单，因此报告中应称“节点引用有效率”，不要夸大成语义证据正确率。

### 5.2 奖励：不单设二元正确性项

继续使用项目当前的单一任务分数，不增加单独的 `R_binary`：

```text
R = 0.80 × R_task + 0.10 × R_evidence + 0.10 × R_efficiency
```

`R_task` 的定义：

- 真值为安全：预测安全得 1，预测漏洞得 0。
- 真值为漏洞：预测为漏洞时，方法组获得归一化 ALPHA 分数 `1 − penalty / max_penalty`；精确匹配对照仅正确 CWE 得 1；预测安全或没有合法 CWE 得 0。

这不是额外叠加的二元奖励权重；安全/漏洞判断只是最终分类任务的一部分。二元 F1 仅用于评估和退化检查。

`R_evidence` 衡量最终引用的节点 ID 是否出现在本轨迹的工具观察中；`R_efficiency = 1 − 查询次数/3`。奖励由程序确定，不使用 LLM-as-judge，也不把奖励数值回传给模型。

**开跑前的必要实现检查**：当前证据计分会把初始 CPG 观察中的节点也纳入 `observed_node_ids`，因此模型可能只引用初始节点而得到有效性分数；并且“节点来自观察”只证明引用可追溯，不证明它能支持漏洞判断。建议主实验将 `R_evidence` 限定为“至少一个引用节点来自本轨迹的工具返回”，并把该项命名为“查询证据引用有效性”；另外报告原始引用有效率。若不改这项，需把其权重结果解释为引用格式/来源奖励，不能声称验证了证据语义。此修正不改变“不单设二元奖励”的要求。

### 5.3 SFT 和 GRPO

- 基础模型：`Qwen/Qwen2.5-Coder-7B-Instruct`。
- SFT：PrimeVul 训练集上的普通函数级分类格式学习；提示中不放 CWE 真值、CVE 描述、补丁代码或漏洞位置。三个实验组从相同种子对应的 SFT checkpoint 开始。
- RL：标准 GRPO，不加自定义逐轮 advantage、手写 ΔALPHA shaping 或额外 critic；一条完整多轮轨迹按终局标量奖励优化。该算法选择遵循 GRPO 原始工作。[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- 推荐起始配置：rollout group size 8、每次更新 4 个 prompt、学习率 `3e-6`、clip ratio `0.2`、最多 3 次 CPG 查询、4×H100 80GB。固定 VERL、SGLang、Joern、模型和 tokenizer 版本/commit；VERL AgentLoop 当前文档标记 API 为 alpha，论文必须给出精确版本和配置。[VERL AgentLoop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- 每 100 个更新步在 validation 上评估并保存；建议将现有默认 1000 步作为上限，而非保证最优训练长度。最终 checkpoint 只按 validation 主指标选，不查看测试集结果后续调参。
- GRPO 前做一次不更新参数的 rollout pilot：要求 JSON 合法率至少 95%，至少 30% 轨迹产生有效工具调用，并且至少一半 prompt 的 8 条轨迹有非零组内奖励方差。若未达到，先修正提示/动作格式；仍未达到时，给所有实验组增加同一批仅用于学习工具调用格式的轻量 SFT 样例，再重新 pilot。不要用测试集调 pilot。

Adnan 等报告 Qwen2.5-Coder-7B 的 GRPO 在 Python CWE 预测上可降低 ALPHA 惩罚；这给层级奖励提供了方向性依据，但它是单轮分类，不能直接推出 C/C++ 多轮查询会得到同样幅度。[Adnan et al.](https://arxiv.org/abs/2607.21069)

## 6. 指标与统计分析

### 6.1 主指标

- **PrimeVul 漏洞样本上的平均归一化 ALPHA 惩罚**（越低越好），并同时报告平均归一化 ALPHA 分数（越高越好）。如果模型把漏洞判为安全，该样本按缺失 CWE/最大惩罚处理，避免只在模型判为漏洞的子集上计算 CWE 指标。
- 主要比较：C 对 B。比较同一测试样本上的配对差异，并报告 3 个种子的均值和标准差。

### 6.2 次要指标

- 漏洞样本的 CWE exact Accuracy、Macro-F1；附上 seen/unseen CWE 分层结果和各 CWE 支持数。这里的 CWE Accuracy 只在真值为漏洞的样本上计算，预测为 safe 或未输出 CWE 均算错。
- 全部样本的二元 Accuracy、Precision、Recall、F1、Balanced Accuracy、MCC 和 PR-AUC；另报告 PrimeVul vulnerable/fixed 配对两侧均判对的比例。由于 PrimeVul 安全样本较多，Accuracy 只作补充，避免单独用它得出模型优劣结论。
- 平均查询数、每次查询返回的新节点数、重复查询率、工具调用格式错误率、有效节点引用率。
- 固定最大预算 0、1、2、3 次时的 ALPHA 分数—查询次数曲线；比较曲线面积或达到同等 ALPHA 水平所需的平均调用数。这样可以检验“何时停止”，而不只看模型有没有调用工具。
- BigVul 审计子集上复算 CWE 层级与精确指标。没有审计的 CWE 标签不进入该项指标。

### 6.3 统计方法

- 三个种子使用同一数据顺序和配对初始化；对测试样本按仓库聚类做 10,000 次 paired bootstrap，并保留漏洞/修复配对结构，报告 95% 置信区间。
- 报告每个种子的结果、均值、标准差、相对变化；不要只报告合并后的单一 p 值。若资源允许，增加到 5 个种子以更可靠估计训练随机性。
- 对 H1 报告 ALPHA 惩罚差值及相对下降比例；对二元 F1 预设 −3 个百分点为警戒线（非劣效护栏），并明确其区间是否跨过该线。多个次要指标的显著性检验使用 Holm 校正。
- 在正式测试前，以 validation 的方差估算对 5% 相对 ALPHA 惩罚下降的检验功效；若功效不足，将结论限定为估计与置信区间，不把“不显著”解释为方法无效。

## 7. 结果预估（不是实测结果）

Adnan 等在单轮 Python CWE 预测中报告的 ALPHA 惩罚降幅为：特定默认设置约 8.3%，最佳数据设置最高约 27.9%；Li 等的多轮 CPG 智能体相对自身 SFT 报告了更高的配对正确率和更少的工具调用，但其奖励明显包含二元 verdict 权重，任务设置不等同于本项目。因此以下估计只作实验前预期，不把前作数字直接迁移成结果。

| 指标 | 方法 C 相对精确匹配 GRPO（B）的预估 | 预估可信度/解读 |
|---|---|---|
| PrimeVul 平均 ALPHA 惩罚 | 相对下降约 **5%–15%**，中心预期约 8%–10% | 低至中；已有层级奖励结果支持方向，但语言、数据、工具交互均不同 |
| CWE exact Macro-F1 | 约 **−1 至 +3 个百分点** | 低；层级奖励最可能先改善“错得更近”，不保证精确类别也提升 |
| 二元 Accuracy / F1 | 差异预计在 **−3 至 +2 个百分点** | 低；二者都不作优化目标，只作评估护栏；还需检查 Balanced Accuracy、MCC、PR-AUC 及召回/误报变化 |
| 平均 CPG 查询数 | 约 **1.2–2.4 次/样本**；C 与 B 之间预计差异在 **±0.3 次** 内 | 很低；策略可能用相似次数换取更有价值的上下文，不能预先承诺查询一定减少 |
| BigVul 审计 CWE 子集 | 预期收益较小，约 **0%–10% 的 ALPHA 惩罚相对下降**，区间很可能跨 0 | 很低；数据偏移和人工复核规模会明显影响估计 |

不预估绝对 F1 或绝对 ALPHA 分数：这些数值依赖实际 CPG 覆盖、数据过滤、提示和评估实现，目前没有本项目实测基线。若 CPG 只能覆盖少量 PrimeVul 样本，或 BigVul 审计标签的类别分布与全量不同，上表相对范围也需要重新估计。

## 8. 成功、失败与可证伪条件

- **支持核心假设**：C 相对 B 的 PrimeVul 平均 ALPHA 惩罚下降至少 5%，paired 95% CI 不跨 0；二元 F1 差值的点估计不低于 −3 pp；收益不是由查询次数大幅增加换来。
- **部分支持**：ALPHA 分数改善但 exact Macro-F1 不变，说明层级奖励改善了错误距离而非精确分类；应据实缩小论文主张。
- **否定“改善查询策略”**：C 的分类分数变好，但查询次数、预算曲线或新节点产出均无改进，且与 B 相比没有更好的单位查询收益。可以报告分类奖励有效，但不能声称查询策略更优。
- **否定层级奖励核心主张**：C 对 B 的 ALPHA 惩罚差异接近 0 或置信区间覆盖有意义的退化；或收益只在未审计 BigVul 标签上出现。
- **安全性/分类退化**：二元 F1 低于 −3 pp、或 PrimeVul 成对正确率退化时，检查模型是否通过预测安全来提高其它奖励分数；不应通过测试集调奖励权重。

## 9. 执行顺序与产物

1. 锁定数据版本、split、CWE 4.20 XML 哈希、模型/框架 commit 和样本去重清单。
2. 为固定仓库 revision 构建 CPG，先报告可覆盖样本数量与构图失败原因；建立三组共同可评估的样本清单。
3. 检查奖励实现，尤其是引用节点是否来自工具返回，以及 3 次查询预算是否在 Harness、VERL loop 和 reward scorer 中一致。
4. 运行 pilot，确认 GRPO 组内 reward 有变化且多轮工具轨迹能正确进入训练 token mask。
5. 训练 3 个种子的 SFT 起点；随后运行 A、B、C 主组。validation 挑选 checkpoint。
6. 最后一次性运行 PrimeVul test、BigVul 审计子集和预算曲线；生成 paired bootstrap 汇总。
7. 保存每个 seed 的训练配置、checkpoint、轨迹日志、预测 JSONL、指标 JSON/CSV、bootstrap 脚本版本和最终图表。

预期产物：

| 产物 | 建议路径 | 内容 |
|---|---|---|
| 配置与版本清单 | `artifacts/manifest.json` | 数据、taxonomy、模型、Joern、VERL/SGLang 版本与哈希 |
| 轨迹日志 | `artifacts/traces/{condition}-seed{seed}.jsonl` | 每轮动作、工具观察、最终预测和终局奖励分解 |
| 测试预测 | `artifacts/predictions/{dataset}-{condition}-seed{seed}.jsonl` | 可复算的逐样本输出 |
| 原始指标 | `artifacts/metrics/{dataset}-{condition}-seed{seed}.json` | 逐条件指标和支持数 |
| 配对统计汇总 | `artifacts/results_summary.csv` | 效应、置信区间、种子均值/方差 |
| 查询预算图 | `artifacts/figures/alpha_vs_queries.png` | 0–3 次预算下 ALPHA—调用数曲线 |

## 10. 主要参考资料

- Li et al. (2026). [Graph Is the Verifier: Agentic Reinforcement Learning for Interprocedural Vulnerability Detection](https://arxiv.org/abs/2607.26656). 支持使用 CPG 多轮环境、稳定节点 ID、SFT warm start 与终局 RL；其奖励设计和主要目标与本项目不同。
- Adnan et al. (2026). [From Evaluation to Optimisation: Hierarchy-Aware Training Signals for CWE Prediction in Python](https://arxiv.org/abs/2607.21069). 支持把归一化 ALPHA 惩罚直接作为 RL 奖励；其实验是 Python 单轮分类。
- Guo et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). GRPO 算法参考。
- [VERL Agent Loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)。多轮生成、工具和环境交互接口参考；实验须固定具体代码版本。
- [PrimeVul 数据与官方切分说明](https://github.com/DLVulDet/PrimeVul)。主基准和元数据说明。
