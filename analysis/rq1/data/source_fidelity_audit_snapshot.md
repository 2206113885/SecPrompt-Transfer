# RQ1 源端 30 条轨迹：冻结与论文协议核查（2026-10-10）

## 结论先行

现有 30 条完整轨迹已经单独封存，原文件未覆盖。逐样本指标、三票多数、五个划分在配对实验间的一致性、以及 30 次验证集候选选择均通过核查。Qwen2-7B + PE2 seed=41 的重复提案是真实的原始生成结果，不是候选解析把不同回答压成一种，也未发现应用层输出缓存或相同请求被重复提交。**但这只能排除已检查的调用链问题，不能从静态记录逻辑上证明 vLLM 内部绝无异常。**目前没有可证实的实现错误，因此不应为了追上论文的 0.945 改动协议或覆盖结果。

论文 Table 5 的 “Average wF1” 计算口径和主实验的确切 split ID 尚未公开到足以逐点对照。我们采用的是五次独立、分层的 30/30/40 holdout；论文主表描述的是一次随机划分。Qwen 还采用预先固定的模型专属 `package_minimal` 分类接口，Llama/Granite 使用 Appendix-F 接口。这些是**明确的协议差异**，而不是本轮发现的新程序错误。对照表应标为“部分协议对齐的独立复现/扩展”，不能写成完全同协议逐点重现。

## 1. 冻结范围与证据

- 权威源端矩阵：`/root/frozen_prompt_transfer/runs/source-optimization-v4/matrix/`；`source-optimization-current/matrix` 当前指向它。矩阵的 30 个 cell 路径分别指向既有 v1 Llama/Granite 和 v4 Qwen 运行目录；没有改动这些运行目录。
- 逐文件 SHA-256 清单：[`source_matrix_30cells_sha256.txt`](source_matrix_30cells_sha256.txt)，30/30 在服务器原路径通过 `sha256sum -c`。每个完整 JSON 约几十 MB，合计约 1.6 GB。
- 新增只读封存包：`/root/frozen_prompt_transfer/runs/source-optimization-v4/source-matrix-30cells-freeze-2026-10-10.tar.gz`，55 MB；内含 30 个完整 `result-full.json`、`matrix-manifest.json`、`matrix-summary.json`，共 32 项。归档 SHA-256：`6980c68109639305d08f7cbaa4669ed3ccbeb5f3fed1908914a1efc0ec95ad06`。封存是额外副本，不替代原始目录，也不意味着操作系统层面施加不可写锁。
- 另将 30 份 `run.log` 封存为 `/root/frozen_prompt_transfer/runs/source-optimization-v4/source-matrix-30logs-freeze-2026-10-10.tar.gz`，2.2 MB、30 项，SHA-256：`f094573dd625b8da081a1b3ee957f3adebe1ba64684b9c9c7b5a21e3a68af4a3`。这保留了 PE2 proposer 的独立调用证据。
- 原始源端代码版本：Llama/Granite `d36f8544d510d7299918eddf38c0dd683a612678`，Qwen v4 `5428dccfb0b5afc248fae93c1a9546fa5fb5802d`。已产生结果的实验分支保持不变；若日后证明错误，必须新建版本并注明更正原因。

## 2. 六项协议核查

| 项目 | 核查结果 | 判断 |
|---|---|---|
| 指标计算 | 从 [`source_test_predictions_slim.csv`](../rq1_source_side_diagnostics/source_test_predictions_slim.csv) 的 6,120 行（30 × 204）独立重算 `p0`、`p*` 的 macro-/weighted-F1；与 [`rq1_source_cells.csv`](../rq1_source_analysis_bundle/rq1_source_cells.csv) 的 30 个 cell 汇总零差异。三票多数与所存 majority prediction 也零差异。 | 没发现本研究评分或汇总错误。 |
| 数据集 | 使用 510 条 SecReq；数据 SHA-256 `d6d1535bacb754f3390a60638f9de38bacd72499f694f34b14808b01b5c77083`，与本地保存的作者 v2 复现包中 `SecReq_510_BTAPE.csv` 一致；全量为 Security 187、Non-security 323。 | 文件内容可核对；不能据此推定作者 Table 5 使用了相同的逐行划分。 |
| 划分 | seed=40–44 各自对全部 510 条做一次独立的、分层 30/30/40 划分，每次 pool/val/test 为 153/153/204。每个 seed 的 6 个 cell（3 模型 × 2 优化器）使用同一 split ID；五个 seed 的 ID 集不同。保存的 [`split manifests`](../rq1_source_analysis_bundle/splits/) 能复核。 | 本研究设计内部配对正确；与论文主表“一次随机 split”并不完全相同。 |
| 初始定义 | 30 个 cell 的初始定义 SHA-256 为 `ab6af83e8a679706a3587728c28e6ba6cc74a40a8186f1303788677f18ec1ccb`，对应 `definitions/security.txt`；语义与论文 Appendix F.4 的长篇 informed security/non-security 定义一致。`Security` 被显式指定为正类。该文件与作者复现包 `definitions/finegrained/security.txt` **不是逐字相同**（后者有压缩改写和额外示例）。 | 明确记录“按论文附录文字”而不是“复现包逐字定义”；作者实际主实验使用哪一版仍待确认。 |
| 推理模板 | 三模型均为原始 HF 权重、vLLM 0.6.4、FP16、temperature 0、batch size 1、每条三次投票、二选一 constrained decoding；无 INVALID。Llama/Granite 使用 Appendix F `Label: {class}`；Qwen v4 使用固定的 `System: You are a precise text classifier... {definitions}...` + `User: Text: {requirement}`，即 `package_minimal`。 | Qwen 的模板与论文 Appendix F 文字不同，虽是预先修复并固定的接口，不能称作严格 Appendix-F 复现。不同模型的源端分数也不应被解释为纯模型能力差异。 |
| 候选选择 | [`candidates_slim.jsonl`](../rq1_source_analysis_bundle/prompts/candidates_slim.jsonl) 共 630 条（每 cell 1 个 p0 + 20 个候选）；每 cell 恰好一个 selected best，均等于最高验证集 weighted-F1，平分时取较新迭代；best iteration 和得分与总表 30/30 一致。源端测试集仅在选择后评估 `p0` 与 `p*`。 | 没发现把 source-test 用于选候选的问题；“平分取较新”应在论文方法中明示，原论文对 PE2 的完全同分处理未写明。 |

本地代码定位：`src/pe2_btape/data.py`（数据/划分）、`metrics.py`（指标）、`prompts.py`（分类与 proposer 模板）、`optimizer.py`（PE2/候选选择）、`btape_optimizer.py`（BT-APE）、`backend.py`（HF-vLLM 请求和响应）。这些文件的当前本地内容与 Qwen v4 服务器版本哈希一致；Llama/Granite 的原运行以其各自保存的 commit 为准。

补充：BT-APE 的指针索引采用作者公开复现包的零基解释，并修正了该包回退时未恢复 definitions 的缺陷；空反馈象限采用公开包的确定性补位策略并逐轮记录。论文 Algorithm 1 对指针移动和空象限处理未给出足够明确的可执行约定。因此这里属于**公开记录的解释选择**，不是本轮新发现的 bug。当前代码的 20 项协议单元测试通过，但合成测试不能替代真实轨迹核查。

## 3. PE2 proposer 专项排查

权威源记录：`/root/frozen_prompt_transfer/runs/source-optimization-current/matrix/cells/qwen2_7b/pe2/seed-41/result-full.json`（SHA-256 `e7244254d5683f0589a4ee041133c7ab53c8de040fb7d32151965920913d0dbe`）。便于复核的抽取：[`qwen_pe2_seed41.jsonl`](../rq1_source_side_diagnostics/pe2_proposer_diagnostics/qwen_pe2_seed41.jsonl)。

- 20 次完整 HF chat-template 渲染输入有 **20 个不同 SHA-256**；20 个 error batch 也不同。`current` 一直是 p0，因为新候选的验证得分没有超过 p0；历史分数和错误反馈随轮次进入 meta-prompt。
- 服务器 `run.log` 有 **20 条不同 batch ID** 的 `HF_VLLM_BATCH` proposer 记录，单请求、`guided_choice=null`、`num_predict=1536`。代码每轮调用 `llm.generate()`；唯一的 `_render_cache` 以 `(system,user)` 为键，只缓存模板渲染结果，**不缓存生成结果**。
- 20 次保存的 **raw proposer output 完全相同**，不是仅解析后相同；20 次 `<revised_prompt>` 解析均成功。每次生成 279 tokens，未触及 1,536-token 限额。因此解析器折叠不同输出、截断成同一前缀这两种解释与记录不符。
- 对照诊断：Qwen seed=42 同为 20 个不同输入，但 2 种 raw output；Granite seed=43 是 20 个不同输入、3 种 raw output。这说明重复提案并非仅 seed=41 的抽取器错误，但目前不能把 Qwen 的搜索退化直接归咎于框架 bug。

**当前最节制的解释（推断）**：在固定 p0、相近的历史高分提示和贪心解码下，Qwen proposer 对变化的四条错误样本不敏感，反复收敛到同一段定义。它使 PE2 seed=41 的有效候选仅为 p0 加一种新文本，不能把“20 次迭代”当成“20 个独立的 prompt 探索”。这应作为实证限制和候选多样性结果报告。对 vLLM 内核层面的复用/确定性问题，仅凭持久化日志不能作绝对排除；若以后要证明底层原因，可另做隔离的两请求微型复测，但不得回写这 30 条轨迹。

## 4. 为什么暂时不追论文数字

论文 Table 5 列出 SecReq Qwen BT-APE `Average=0.945`、PE2 `0.937`，并称表格为 weighted F1。当前五个 seed 的 source-test weighted-F1 平均分别为 Qwen BT-APE **0.8318** 和 PE2 **0.8189**（范围分别 0.7928–0.8743、0.7777–0.8385），确有明显差距；但单 split 与五次分层 repeated holdout、Qwen 分类接口及可能的汇总口径均不同，不能将差额直接诊断成代码 bug。

另有一个应请作者解释的算术疑点：Table 5 Qwen BT-APE 的两类数值 0.908 与 0.981 的简单均值为 **0.9445**，按整份 SecReq 的 187/323 类别支持数加权则为 **0.9542**；表中 0.945 与前者近似。Qwen PE2 的两类数值 0.919 与 0.975 的均值为 **0.9470**、按该支持数加权为 **0.9545**，都不是表中 0.937。由于原测试集的类别支持数和列定义未公布，**这不是“论文算错”的证明**，但足以要求作者提供 Table 5 代码/逐样本预测和 split ID，才谈得上对齐 0.945。

更严格地说，**若**两类列值确实是同一个测试集上的逐类 F1，BT-APE 这一行要加权得到 0.945 需 Security 的测试支持率约 49.3%，PE2 那一行要得到 0.937 则需约 67.9%；它们不可能同时来自同一个固定测试集支持率。因此至少还存在未说明的汇总层级、列含义或表格数值问题。此结论以“逐类 F1”的条件为前提，不应直接写成作者计算错误。

已拟好 [`作者邮件草稿`](btape_author_query_draft.md)，收件人使用论文首页印出的邮箱。当前会话没有连接可发送邮件的账户，**邮件尚未发送**；需要研究者本人检查署名并寄出。

## 5. 停止条件与下一步

1. **立即停止重调旧 30 条以追高分。** 现阶段六项核查没有发现可证实的源端评分/候选选择/请求构造错误；保留现有 v1+v4 矩阵和独立封存包。论文写作须披露分层五 seed、Qwen 专属接口、PE2 有效候选重复和 Table 5 口径未确认。
2. **仅在出现可复现的错误证据时改版。** 例如作者提供精确 split/评分脚本后证明现用实现有协议错误，或隔离复测明确揭示输出被错误复用。届时先提交新的修正版本和预定重跑范围，统一重跑受影响的 source cell，再重新部署其受影响的 GitReq prompt；旧结果仍保留供审计。改版理由必须是纠错，而非“分数更接近论文”。
3. **若作者未给出精确 ID/计算方式且无实现错误，停止追 0.945，进入跨数据集迁移分析。** 现有 30 条轨迹可用于本研究的源域内部配对、候选排名与迁移分析，但不得据此宣称严格复现论文 Table 5。

论文依据：Zadenoori et al., *BT-APE*, [arXiv:2607.00427](https://arxiv.org/abs/2607.00427)，Table 5（PDF 第 24 页）、§3.3–3.6、Appendix A.1/B/F；作者 v2 [归档复现包](https://zenodo.org/records/20438927)。
