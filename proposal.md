# 基于可塑性 RNN 的 Few-Shot 传递推理行为学复现研究

## 一、研究背景

人类如何从极少量信息中推断出完整的排序结构，是认知科学长期关注的核心问题之一。传递推理（transitive inference）实验表明：被试在观察到若干局部比较（如 A>B、B>C）后，能够对从未直接见过的配对（如 A>C）做出正确判断，说明大脑能够在 few-shot 条件下构建隐性的全局排序。

Liu 等（2026）在 fMRI 与行为实验中进一步发现：即使所有被试接受完全相同的 few-shot 学习输入，他们仍会构建出个体化的全局排序——不同被试对同一 pair 的选择可能截然相反，且这种差异具有稳定的错误模式与跨被试 Beta 分布特征（如双峰 profile）。该工作将问题从“能否做传递推理”推进到“相同输入为何产生不同排序”，对理解学习中的个体差异与神经表征具有重要启发。

在计算建模一侧，Mermillod 等提出的可塑性 RNN（plastic RNN / `RetroModulRNN`）通过 meta-learning 训练慢权重，并在每个 episode 内用多巴胺调制的 Hebbian 可塑性（快权重 `pw`）实现快速学习，已在传递推理任务上取得良好效果。

然而，原 `fsrl` 仓库中的任务设计（随机相邻 pair 采样、学习阶段有选择与奖惩反馈）与 Liu 2026 的范式存在显著差异，也尚未配备与行为学一致的结构化分析流程。因此，将行为学范式迁移到可塑性 RNN 框架，并用同一套分析方法检验模型响应，是连接“神经计算机制”与“人类 few-shot 排序行为”的一条可行路径。

本研究旨在检验：一个已知可学习的神经网络，在观察学习、无测试反馈的设置下，能否复现行为学文章中的关键现象。

## 二、研究问题

本研究希望明确回答以下问题：

1. 能否将行为学的监督集 S、查询集 Q、block 重复呈现、非相邻约束等设计，迁移到 `simple_neo.py` 的可塑性 RNN 训练框架中，并稳定跑通？
2. 在测试阶段不提供行为反馈的前提下，模型能否仅靠观察学习（教师多巴胺信号）与 meta-train 辅助监督，完成对新 pair 的传递推理？
3. 模型的测试响应是否呈现与行为学一致的结构特征——如 S 与 QS 准确率差异、serial position effect、symbolic distance effect、HodgeRank 全局一致性、跨被试 Beta 分布分化？
4. 通过引入跨 episode 保留的可塑初态（`persistent_pw`），能否在相同输入下让不同“虚拟被试”产生不同的主观排序？

假设：


| 编号  | 假设                    | 对应指标                                   |
| --- | --------------------- | -------------------------------------- |
| H1  | 已监督 pair 比未监督 pair 更易 | `acc_supervised` vs `acc_unsupervised` |
| H2  | 模型能重建接近真实的全局排序        | Kendall τ、Spearman ρ                   |
| H3  | 个体差异可诱导分化排序           | `n_pair_bimodal`、主观排序分歧                |
| H4  | 远距离 pair 准确率更高        | distance effect 曲线（Fig 1G）             |




## 三、研究方法



### 3.1 计算模型

采用 `RetroModulRNN`（`simple_neo_backup.py`），其核心为：

- **慢权重**（`w`, `α`, `i2h`, `h2o` 等）：跨 episode 用 Adam meta-learning 更新；
- **快权重** `pw`：episode 内由多巴胺 `da` 与资格迹 `et` 驱动 Hebbian 可塑性更新；
- **隐状态** `hidden`：每个 trial 重置；`pw` 在 trial 间保留，可选跨 episode 保留（`persistent_pw`）。

学习阶段注入教师多巴胺 `teacher_da`（较强 cue 在左 → +1，在右 → -1），配合随机左右呈现，模拟观察学习而无显式动作；测试阶段网络自主采样动作，仅离线记录响应。

### 3.2 任务数据


| 要素         | 设定                                              |
| ---------- | ----------------------------------------------- |
| 刺激数量       | `nbcues = 8`，编号 0…7，真实全序 0 ≻ 1 ≻ … ≻ 7          |
| 刺激编码       | 每个 cue 为 15 维随机二值向量（±1），batch 内两两区分             |
| 监督集 S      | 8 个非相邻 pair，按秩差 Δ 排序选取（`build_supervision_set`） |
| 查询集 Q      | 全部 C(8,2)=28 个无序 pair（`build_query_set`）        |
| Episode 结构 | 4 学习 block ×                                    |
| 被试         | 每个 batch 位视为一个虚拟被试；`bs` 决定并行被试数                 |


相对原版 `fsrl` 的改动：弃用 `sample_trial_pair` 随机采样，改为 block schedule——每 block 开始前打乱 S 或 Q 的呈现顺序（`shuffle_schedule`），由 `prepare_trial` 构造每个 trial。

### 3.3 实验设计

**主实验配置（**`new_weak`**，E1 基线）**：


| 参数                         | 取值          |
| -------------------------- | ----------- |
| `persistent_pw`            | True        |
| `pw_init_std`              | 0.01        |
| `baux_learn` / `baux_test` | 0.5 / 1.0   |
| `batch_size`               | 16（训练与分析一致） |
| `nbiter` / `warmup`        | 5000 / 100  |
| `seed`                     | 42          |


**消融实验**：


| 目录                 | 改动                                  | 目的                |
| ------------------ | ----------------------------------- | ----------------- |
| `new_strong_quick` | `baux_test=0.3`, `pw_init_std=0.05` | 过强学习 → 个体差异消失     |
| `liu_hybrid`       | `baux_test=0`, `baux_learn=0.5`     | 测试无 meta 监督 → 学不会 |
| `liu_minimal_v2`   | `baux=0`                            | 完全无辅助 CE → 接近随机   |


**分析实验**（训练后 eval）：

1. 加载 `new_weak/net.dat` 与 `new_weak/subject_pw.pt`；
2. `batch_size=16`（必须与训练一致）；
3. 调用 `analysis.py` 的 `run_full_analysis`（含逐被试 HodgeRank）。

```bash
# 主实验训练
python simple_neo_backup.py \
  --output-dir new_weak \
  --persistent-pw --pw-init-std 0.01 \
  --baux-learn 0.5 --baux-test 1.0 \
  --batch-size 16 --seed 42 --nbiter 5000 --warmup 100

# 主实验分析
python run_analysis.py \
  --model-path new_weak/net.dat \
  --subject-pw-path new_weak/subject_pw.pt \
  --persistent-pw --batch-size 16 \
  --analysis-dir new_weak/analysis_v2
```

主实验超参数：`hs=200`, `lr=1e-4`, `lpw=1e-4`。

## 四、结果

*主数据来源：*`new_weak/analysis_v2/analysis_report.json`*（*`new_weak/net.dat` *+* `subject_pw.pt`*，*`batch_size=16`*，16 个训练虚拟被试）。*

### 4.1 任务迁移与群体表现

- 总体测试准确率 **62.0%**（`test_accuracy`），高于随机（50%），接近人类难 pair 群体均值（~67%）；
- 已监督 pair **59.9%**（`acc_S`），未监督 pair **62.8%**（`acc_Q\S`）——未监督略高，**H1 未支持**；
- 群体 HodgeRank 主观排序 **[0, 2, 1, 3, 4, 5, 7, 6]**，与真序局部混淆（0/1/2、6/7），**Kendall τ = 0.86**，**Spearman ρ = 0.95**（H2 部分成立）；
- `curl_ratio = 0.29`，`global_consistency = False`，2 个 circular triad——群体聚合排序接近真序但非完美。



### 4.2 结构效应

- **Distance effect**：秩差 1→7 准确率分别为 **62.4% → 61.8% → 61.4% → 61.3% → 65.4% → 61.3% → 58.1%**，中间秩差有上升、远端回落，**H4 仅弱成立**；
- **Serial position effect**：各秩位置准确率在 **59%–66%** 之间波动，未呈现人类明显的 U 形梯度。



### 4.3 个体结构


| 指标                           | `new_weak`          | 论文人类（参照）  |
| ---------------------------- | ------------------- | --------- |
| 被试间排序相似度 `inter_subj_τ`      | **0.33**（低）         | 低         |
| 偏离真序被试数                      | **16/16**           | 多数        |
| 自洽错序 `self_consistent_wrong` | **4/16**            | 64/69     |
| ≥80% 一致错误 pair               | **9/16**            | 63/69     |
| 逐被试 τ vs 真序（均值）              | **0.52**（0.21–0.93） | —         |
| pair 级 Beta 双峰               | **0/28**            | **15/28** |


- **个体排序分歧**：`per_subject_rankings.png` 显示 16 条被试曲线与真序（绿线）明显偏离，**方向上与 Liu 2026「相同输入、不同排序」一致**；
- **pair 双峰（Hypothesis I）**：仍为 0/28；难 pair 3–4 跨被试准确率有 0.1–1.0 散布，但中间值较多，Beta 判为 `high_accuracy` 而非双峰；
- **结论**：模型在**个体结构**上优于经典 Q-learning 模拟（低 `inter_subj_τ`、稳定错误模式），但尚未达到论文级双峰与大规模自洽错序。

## 五、研究意义

- 在统一任务范式下，证明 plastic RNN 可在**测试无行为反馈**时完成 few-shot 传递推理（62%），并在群体层面呈现弱的 distance effect；
- 建立模型—行为对齐分析流程（HodgeRank、Beta、逐被试排序），超越单一准确率；
- `persistent_pw` + 逐被试分析显示个体排序分歧（`inter_subj_τ=0.33`），为 constructive ranking 提供初步计算证据，同时揭示 independent-value 式共享 `net.dat` 难以产生 pair 双峰的限制。



## 附录 A：任务设计详述



### A.1 Episode 时间线

```text
Episode
├── 学习阶段（4 blocks）
│   └── 每 block：打乱 S 后依次呈现 8 个 pair
│       └── 每个 trial：2 步（展示，无 Go / 无决策）
└── 测试阶段（10 blocks）
    └── 每 block：打乱 Q 后依次呈现 28 个 pair
        └── 每个 trial：4 步（含 Go 信号与决策步）
```



### A.2 监督集与查询集

```text
S = build_supervision_set(nbcues, size=8)
  → 非相邻 pair (|i-j|>1)，按秩差 Δ 排序取前 8 个

Q = build_query_set(nbcues)
  → C(8,2) = 28 个无序对
```



### A.3 输入编码（`build_step_inputs`）


| 区间                 | 内容                   |
| ------------------ | -------------------- |
| `[0 : 2*cs)`       | 两个 cue 的二值特征拼接       |
| `[2*cs]`           | Go 信号（仅测试阶段）         |
| `[nbstimbits + 0]` | 常数 1.0               |
| `[nbstimbits + 1]` | 归一化时间                |
| `[nbstimbits + 2]` | reward（测试阶段恒为 0）     |
| `[nbstimbits + 3]` | 秩差 Δ 归一化             |
| 末尾 2 维             | 上一步动作 one-hot（仅测试阶段） |


---



## 附录 B：网络架构与 Episode 流程



### B.1 RetroModulRNN 结构

```text
inputs ──► i2h ──► tanh ──► h2o ──► action logits (2)
                    │              h2v ──► value (1)
                    │              h2DA ──► DA
              recurrent: (w + α⊙pw) · h
              pw ← pw + DA · et
              et ← (1-η)·et + η·Δet
```



### B.2 状态变量


| 符号             | 跨 trial    | 跨 episode             |
| -------------- | ---------- | --------------------- |
| `hidden`, `et` | 每 trial 重置 | —                     |
| `pw`           | 保留         | 可选 `persistent_pw` 保留 |
| 慢权重            | —          | Adam 更新               |




### B.3 单 trial 与 episode 流程

`run_single_trial`：构造输入 → 前向 → 测试采样动作 → 记录 `TestResponse` → trial 末 `pw.detach()` 截断 BPTT。

`_run_episode_blocks`：blank 预热 → 学习 block 循环 → 测试 block 循环 → 返回 `EpisodeRecord`。

---



## 附录 C：TrainConfig 主要参数


| 类别  | 参数                 | 默认    | 含义              |
| --- | ------------------ | ----- | --------------- |
| 网络  | `hs`               | 200   | 隐层大小            |
| 网络  | `bs`               | 32    | batch / 虚拟被试数   |
| 优化  | `lr`               | 1e-4  | 学习率             |
| 优化  | `nbiter`           | 5000  | 训练 episode 数    |
| 优化  | `warmup`           | 100   | 前 N episode 不更新 |
| 任务  | `nb_learn_blocks`  | 4     | 学习 block 数      |
| 任务  | `nb_test_blocks`   | 10    | 测试 block 数      |
| 任务  | `supervision_size` | 8     |                 |
| 损失  | `baux_learn`       | 0.5   | 学习步离线 CE        |
| 损失  | `baux_test`        | 1.0   | 测试决策步离线 CE      |
| 个体  | `pw_init_std`      | 0.0   | 可塑初态标准差         |
| 个体  | `persistent_pw`    | False | 跨 episode 保留 pw |


派生量：`n_learn_trials=32`, `n_test_trials=280`, `eplen=1184`, `inputsize=37`。

---



## 附录 D：分析模块与输出



### D.1 分析流水线

```text
TestResponse → R 矩阵 → acc_S / acc_Q\S → 错误模式 → circular triad
            → HodgeRank（群体 + 逐被试）→ 主观排序 → Kendall τ
            → Liu 扩展（serial position, distance, Beta, error consistency)
```



### D.2 输出文件


| 文件                               | 内容                                |
| -------------------------------- | --------------------------------- |
| `analysis_report.json`           | 全部数值指标（含 `per_subject_rankings`）  |
| `heatmap_R.png`                  | 响应矩阵热图                            |
| `acc_S_vs_QminusS.png`           | 监督 vs 未监督准确率                      |
| `true_vs_subjective_rank.png`    | 群体真实 vs 主观排序                      |
| `per_subject_rankings.png`       | **逐被试** HodgeRank vs 真序（Fig 3C/E） |
| `serial_position_effect.png`     | Fig 1F                            |
| `distance_effect.png`            | Fig 1G                            |
| `pair_beta_fits.png`             | pair 级 Beta 散点                    |
| `subject_beta_fits.png`          | 被试级 Beta 散点                       |
| `exemplar_pair_subject_hist.png` | 示例 pair 直方图                       |




### D.3 HodgeRank 简述

将 pairwise 偏好 `y_e = 2·R[i,j]−1` 建模为势函数梯度 `B·f ≈ y`；`curl_ratio` 衡量环流能量占比，越小越接近一致全序。

---



## 附录 E：关键函数索引


| 函数                                 | 文件       | 作用                   |
| ---------------------------------- | -------- | -------------------- |
| `build_supervision_set`            | backup   | 构造监督集 S              |
| `build_query_set`                  | backup   | 构造查询集 Q              |
| `prepare_trial`                    | backup   | schedule → trial     |
| `build_step_inputs`                | backup   | 逐步输入向量               |
| `run_single_trial`                 | backup   | 单 trial 执行           |
| `run_episode` / `run_episode_eval` | backup   | 训练 / 分析用 episode     |
| `build_response_matrix`            | analysis | 构造 R 矩阵              |
| `hodge_rank_scores`                | analysis | HodgeRank 分解         |
| `analyze_liu_behavioral_effects`   | analysis | Liu 2026 扩展          |
| `analyze_per_subject_rankings`     | analysis | 逐被试 HodgeRank（Fig 3） |
| `run_full_analysis`                | analysis | 分析总入口                |


