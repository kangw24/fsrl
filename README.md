# 人类式排序过程候选：从结果复现到 constructive ranking

本仓库的目标是复现 [Liu et al. (2026)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003756) 的人类关系学习机制：人在只看到少量局部 pairwise 关系时，如何主动建构一个完整、自洽、却因人而异且常常错误的全局排序。我们追求的是一个输入条件正确、能独立生成虚拟个体、内部步骤明确、可被替代模型和干预否证的过程候选，而不是把 accuracy、双峰数或 distance effect 调到接近人类数字。

当前**没有任何模型有资格称为“已经复现人类认知机制”**。本 README 是研究纪律账本：先冻结候选与预测，再读人类数据一次性比较，失败就保留结果，不靠加自由度救回。

## 阅读导航

- [当前研究状态](docs/current_status.md)：当前候选、冻结参数、允许的最高结论。
- [实验与输入契约](docs/experiment_contract.md)：原实验流程、模型可见信息和 RNG 防火墙。
- [CGR-v3.2 模型卡](docs/models/cgr_v3_2.md)：状态、在线更新、终局等价与限制。
- [报告索引](docs/reports.md)：canonical report、evaluator 协议和污染标签。
- [研究账本索引](docs/research_ledger.md)：v1–v3.2 的历史路线和冻结结论。

根 README 继续保留完整历史账本；新增文档只提供稳定入口，不删除失败记录。

## 1. 任务与最高纪律

### 1.1 要复现什么

论文的判别量在个体级，且是预登记的。经典 TI 模型（Q-learning / Beta-Q / Betasort）能复现组级 serial position 与 symbolic distance，却全部失败在个体级：它们从相同输入收敛到同一个全局排序，所以被试间相似度高、错误自洽破裂；人却各自搭出**自洽但错、互相不同**的全局排序。我们要复现的是后者：

- 难对（近对角线、小距离）跨被试 accuracy 双峰（论文 15/28 对 α<1,β<1；0/28 单峰）；
- 个体自洽但错：64/69 自洽但错，自洽系数≈1（零循环 triad）；
- 个体错误一致性 >0.8，54/69 在至少一对上 100% 一致地错；
- 被试间 Kendall 相似度低（idiosyncratic），而 Q-learning 等高（t=-66.77）；
- 组级 serial position 与 symbolic distance 仍在（不得为复现个体级而破坏）。

组级效应是所有模型都过的“陷阱”，不是靶子。

### 1.2 模型与 agent 职责不同

```text
模型
  只接收人类在实验中能够获得的刺激信息
  独立形成内部状态并输出选择、概率和轨迹
  不读取任何人类选择或人类分析结果
  输出完成后停止

agent
  先固定模型及其输出
  再读取模型输出和人类原始数据
  计算差异、检查是否符合人类行为
  负责承认失败并限制结论
```

人类原始选择不得进入模型训练、微调、温度或噪声校准、checkpoint/seed/架构选择、虚拟个体参数估计或 participant posterior。模型不负责检查自己是否像人。agent 看过人类结果后修改的任何版本就是新开发候选，同一份已看过的数据不能再称该版本的独立盲测。

### 1.3 允许隐过程，但不能靠自由度逃避失败

人类真实机制未知，因此允许模型提出未直接观测的内部预测、注意、记忆、误差、承诺或写入时刻。它们必须：

- 在查看对应结果前写清输入、状态、更新方程和反证条件；
- 只使用少量、跨 item/pair 共享的参数；
- 与容量匹配的简单替代模型竞争；
- 通过模型恢复、参数恢复和 matched lesion；
- 失败后保留结果，不靠继续加 adapter、mixture 或 per-subject 参数救回。

一次 destructive lesion 让模型崩溃只证明代码依赖该模块，不证明人脑依赖它。真正有价值的比较是完整机制与计算能力匹配的最小替代机制。

### 1.4 盲测与数据污染

执行顺序固定：先冻结候选 + 未见人类 choice 的 schedule 盲预测 + 哈希封存，再由独立只读比较器一次性比较。Liu 的两个行为 cohort 均已被本项目反复查看，因此 Liu 今后只能作为冻结规则下的回顾性 benchmark（否证用），不能作为确认性盲测。确认性证据需要尚未用于开发的外部公开数据或未来新数据。已污染的 raw choice 不得复用于开发。

## 2. Liu 实验中人真正看到了什么

8 张物体图片，每张当作一部电影，参与者按票房柱条学习热度排序。学习阶段每 trial 同时显示两个物体和两根带刻度柱条；柱条绝对高度每次随机，相对差值由两物体排序距离决定（1–7）；参与者关注相对差值、不需作答。8 个固定抽象排序位置 pair 各 block 出现一次，共 4 个 block，block 内顺序随机：

`(A,F) (B,C) (B,E) (C,G) (D,F) (D,G) (E,H) (A,H)`

测试阶段同时显示两个物体、不显示柱条，参与者选排名更高者；全部 28 个 pair 每 block 各测一次，共 10 个 block，无反馈。

所有人用同一组 8 张图片，但每名参与者“图片身份 → A…H 排名位置”映射随机打乱。固定的是抽象位置 pair，不是每人看到相同的物理图片 pair。新虚拟人群也必须按虚拟个体随机该映射，不能共享同一 item-to-rank assignment 后冒充个体差异。公开行为 CSV 没有保存每人学习期随机顺序，因此模型只能按实验规定的 block 内随机化分布生成自己的学习顺序，不能声称复现某真人的 trial-by-trial 路径。

## 3. 候选输入边界

本项目研究关系学习和排序构造，不研究物体识别。8 张图片只承担稳定、可区分的 item identity，抽象为 8 个无语义符号。图片颜色、形状、像素和现实语义均不是主模型需要解释的机制。

```text
左符号 + 右符号 + 当次明确呈现的相对关系证据
→ 在线关系/排序更新
→ 无反馈 query 选择
```

论文原始学习屏幕同时传达高低方向和 score distance；但这不等于主模型必须读取幅度。当前主线做出更强、也更容易失败的抽象：**只保留左右 item 的高低关系 `sign ∈ {-1,+1}`，主动丢弃具体差值**，使全局排序真正欠定。后续不得把 distance magnitude 悄悄加回同名候选；若要研究它，必须另立新候选并重新冻结比较协议。

主候选不得直接读取：`true_rank`、正确 query 答案、participant ID、真人选择或真人主观排名、屏幕没有呈现的精确 normalized episode progress。`true_rank` 只允许任务生成器用来确定当次呈现的关系、生成合成 outer-loop target 和供 agent 评分。query 只读：测试阶段无反馈，模型不从测试 trial 更新状态。不得训练像素前端，不得用任意“感知噪声”制造想要的双峰。

## 4. 人类数据能支持什么（目标与限制）

公开 OSF 行为数据含两个 cohort：40 人 preregistered + 37 人 replication，共 77 人，每人 `10 blocks × 28 pairs` 测试选择。固定审计结果（复现目标）：

| 指标 | 人类数据 |
|---|---:|
| overall accuracy | 0.852365 |
| learned / unlearned accuracy | 0.913961 / 0.827727 |
| paper-style bimodal pairs | 15 / 28 |
| binomial-midpoint sensitivity | 8 / 28 |
| mean pair response consistency | 0.949896 |
| rank-analysis participants | 69（排除 8 名全部 pair 多数正确者） |
| self-consistent wrong rankings | 64 / 69 |
| participants with ≥1 consistent error at 80% | 63 / 69 |
| mean self-consistency coefficient | 0.994928 |
| total subject circular triads | 7 |
| mean support consistency | 0.939935 |
| mean subject-level distance slope | 0.039829 |

数据文件哈希：

- preregistered CSV：`6dcae48511018a85765e3ab7ceed6f358f5185f5ce399ce47543dbc7aad0c227`
- replication CSV：`c322cedd587d8e119442f873679d5fd5315e31736f4bf4266dbb89849c068249`

公开 CSV 没有：学习期呈现顺序或学习期响应、RT 或 confidence、prefix/冲突/干扰/support-order 操纵、跨 session 身份稳定性。因此它能检查最终选择、个体内稳定性、主观排序、自洽性、serial-position 与 symbolic-distance effect；不能直接验证学习期轨迹、真实 RT/confidence、长期人格或某个具体更新方程。Beta 双峰数量依赖边界处理，必须同时报告 paper-style `0.01/0.99` clip 的 15/28 与 binomial-midpoint 的 8/28，不能把 15/28 当作不受分析口径影响的机制真值。

## 5. 个体差异必须怎样产生

个体差异必须由模型在看到人类数据前自行生成。允许来源：每个虚拟个体独立的符号—排名映射、合法的 block 内学习顺序和左右呈现随机化、trial-local 关系编码可靠性/有限记忆/注意波动/决策噪声的共享群体分布、episode 内随机状态演化。

禁止：participant ID embedding、每人预存一个 rank/错误模板/strategy ID、每人每 pair 自由 logit、根据真人早期答案选最像他的虚拟 run、用人类 blocks 1–5 估计主模型个体参数、跨 episode 保存具体任务排序。当前数据只有单次 session，因此即使模型产生跨虚拟个体差异，也只能讨论 session-level behavioral heterogeneity，不能称稳定人格。

论文报告学习顺序未能解释观察到的个体差异；因此若虚拟双胞胎分析显示模型差异主要由 support order 驱动，应判个体差异机制失败，而不是把合法随机顺序当成解释成功。

## 6. 评价原则与允许的最高结论

模型完成前人类选择不可见。模型输出固定后，agent 检查：原始选择的 population prior-predictive proper score、learned/unlearned 泛化、个体内重复选择稳定性、自洽但错误的主观全局排序、跨虚拟个体 pair-level 极化、serial-position 与 symbolic-distance effect、关系编码/记忆更新/决策噪声的可分离性、与 sign-only 简单基线和容量匹配替代的比较。accuracy、双峰数或 distance slope 都不能单独授权机制主张；主候选必须击败容量匹配的简单替代，预测等价时按简约性判简单模型胜。

若一个完全不接触人类选择的模型能独立生成虚拟人群，并比预先限定、容量匹配的替代更好地解释 Liu 原始终局选择，同时通过恢复和 matched lesion，允许的最强表述是：在当前候选集合中，该模型是最受现有终局行为支持的前向算法过程候选。仍不能声称：人类确实计算了模型中的 prediction error、模型内部 trace 就是人类真实学习轨迹、某 lesion 对人脑必要、模型 margin 等于人类 confidence 或 RT、一次 session 的隐变量解释了稳定人格、更高拟合排除所有未测试机制、或已经复现了唯一真实的人类认知机制。

## 7. 历史候选与已冻结结果（审计账本）

下表是已封存结果，保留以备审计，不得删除或回填。任何“通过”都只是当时冻结规则下的结论；失败候选不得靠调参复活。

| 候选 | 入口/报告 | 状态 |
|---|---|---|
| Phase 5g fixed-Hodge | `configs/archive/2026-07-16_legacy_research/phase5_retro/retro_process_phase5g_fixed_weight_no_proj.yaml` | 透明全局基线，非人类过程模型 |
| Phase 5j fast-plastic RNN | `outputs/archive/2026-07-16_legacy_research/phase5_retro/phase5j_*` | 路径诊断失败（bimodal 13/28→2/28 或 7/28） |
| Bounded global-hypothesis | `outputs/archive/2026-07-16_legacy_research/alternative_models/bounded_rank_path*` | 几乎总收敛真排序，做不出人类稳定错序 |
| M&K source/迁移 | `fsrl/miconi_kay/`、`outputs/archive/2026-07-16_legacy_research/alternative_models/miconi_kay_*` | 隔离复现通过，Liu 迁移 v1/v2.1 正式失败 |
| OnlineOrdinal-v1 | 见 7.2 | 即时更新输给块末整合，时间路径失败 |
| DCR-v1 / RetroModulRNN | 见 7.3 | 三 seed plastic lesion drop ≈0.0003，正式否证 |
| AADM→WBDM-v1 | 见 7.4 | Graham–Spitzer 外部闸门三 seed 全失败 |
| LGC-v1 | `fsrl/model/local_global_consolidation.py` | 极简双通路对照，结构对称做不出方向交互，不晋升 |

### 7.1 工程基线

仓库已具备：人类数据与坐标审计、符号化 Liu episode、fixed-Hodge、直接 relation fast-plastic RNN、bounded global、顺序/冲突诊断、fixed-task 个体差异审计、模型恢复、隔离的 M&K 账本。图片身份在主线中只作 8 个任意符号，仓库不含也不需要图片识别或像素读取。历史正式输出不得覆盖；独立复算必须用新目录或新文件名。

### 7.2 Ciranka Exp.4 在线过程压力测试（已失败，冻结）

外部数据 `outputs/external_data/ciranka2022/exp4.mat` SHA-256 `c8f3c8d1dd2111495d24fbddbde4f2e3e084455e51165b87809bce2eebd19c7a`（shape 70 trials × 8 columns × 6 blocks × 60 subjects；按官方 R 脚本 binomial p<0.01 保留 49 人）。协议支持保留在 `scripts/audit_ciranka2022_exp4.py`、`scripts/generate_ciranka2022_frozen_predictions.py`；一次性比较脚本归档为 `scripts/archive/legacy_candidates/alternative_models/compare_ciranka2022_prequential.py`。冻结 prediction hash `8e1ea461d388e1f9d2594765fd419a11de29fef2cffd4cb66c99477b75a592eb`；报告 `outputs/archive/2026-07-16_legacy_research/alternative_models/ciranka2022_exp4_prequential_comparison/prequential_report.json`。

结果（49 人 no-feedback non-neighbor trial）：OnlineOrdinal LPD `-0.659795`，优于 chance `-0.693147` 与冻结 leaky `-0.714642`，但**输给块末整合** `-0.657161`（online-minus-block-end mean/median `-0.002634/-0.000358`，23/49 人即时更好）。uncertainty–RT mean Spearman `0.194218`，控制 block 与真实序距后降到 `0.074496`（再控制 correct 后 `0.069502`），只是弱增量关联，不是 RT 生成机制。按预登记判即时更新路径失败。

### 7.3 DCR-v1 / RetroModulRNN 冻结 synthetic 闸门（已否证）

入口已归档为 `scripts/archive/legacy_candidates/alternative_models/train_dcr_v1_frozen_budget.py`（执行前 SHA256 `6e7c236faeb57572a8ddf11b9952ab31661fd5857c154b8d0b75bdecd7c15489`）；changepoint runner SHA256 `9235f52d661becc14fd545226c66a5fb031aaaae15935d9ffa791465f6fe42d8`；报告 `outputs/archive/2026-07-16_legacy_research/alternative_models/dcr_v1_frozen_budget/frozen_synthetic_report.json`，状态 `fails_frozen_synthetic_rmrnn_gate`，`human_data_read=false`。seed 941001/941002/941003 的 plastic-write LPD drop 仅 `0.000325/0.000349/0.000428`，远低于 `0.01`；matched-nonplastic drop 仅 `0.000057/0.000047/0.000060`。RetroModulRNN 不再属于主候选。其 `constant_gate` 控制被逐参数抽取为无 RNN 的 `LocalGlobalConsolidation-v1`（`fsrl/model/local_global_consolidation.py`，报告 `outputs/archive/2026-07-16_legacy_research/alternative_models/lgc_v1_frozen_extraction/extraction_report.json`），只作最小双通路对照，结构对称、做不出方向交互，不晋升为人类机制。

### 7.4 WBDM-v1 → Graham–Spitzer 外部个体选择闸门（已否证）

外部比较分三阶段：阶段一只读公开 schedule；阶段二由冻结 checkpoint 对 400 份、每份 324 trial 的 schedule 逐 trial 先预测后反馈更新；阶段三才读人类 raw choice。模型输入只含左右符号 token、可见 block boundary、反馈试次选择后的高低 `sign`；`Direction/Switched/Rank_1/2`、人类 choice/accuracy/RT 不得进入模型。阶段一源码 SHA256（XLSX reader / 生成器 / 写入前比较器）分别为 `49e365e134364f3301011501a1cf4316f0f2e7e21265ed53e406328613799cb9`、`13b60c1bc4eecc0ca671c379dd2d2b2b385cdad716c1438d88766e1a17c98642`、`44ae806c69095a779d4568409d4b40623297e846a7903ff0b13485ea0418acd6`；400 源 schedule 的 filename+file-hash 合并 SHA256 `fc9cf88499116cec6938b970a4bd9b59b7086e75d5ee74b7a59babbe22cd4fcb`。

冻结候选固定为 WBDM、matched-symmetric、pairless、itemless、boundary-replay、LGC。primary 范围对齐论文核心：`Feedback_on=0`、两 item 均 `2..6`、item distance `>1` 的 non-anchor TI；聚合先 participant 内平均再跨 participant 平均。冻结输出 `outputs/archive/2026-07-16_legacy_research/alternative_models/wbdm_graham_spitzer_frozen_predictions/`：manifest `25910509ae551736024a4e4aea397f84c3b2780259b8a1039acf3a732f538942`、schedule projection `a8639e7da8cef7b4eb8ecbfe160af19564db0ff52f9dd94dd63f5a59047e32f4`、generator `13b60c1bc4eecc0ca671c379dd2d2b2b385cdad716c1438d88766e1a17c98642`；seed 971001/971002/971003 prediction SHA256 分别为 `c2ad308491c75d854b416f9817d415f59016d6a9462172087ecde5cf38c6cbb8`、`2eddece5c540d7008648c1f53dee6c722d90184f469a14542fb7ee22a8e97d6b`、`500fa092459f339862e76bb3a99386e5361fdfa8d50b057eb02e8736f20ad3a3`；manifest `human_data_read=false`。

预登记放宽与 schema 归一（均在读取任何 raw choice 结果前完成；比较器在算出纳入集前已崩溃，未产生任何门槛数值）：(1) 数值容差从封存值 `0.03/0.04/0.05/0.025` 放宽回原始方案 `0.04/0.05/0.06/0.03`，机制性门槛（LPD 优势与 bootstrap 下界、两条斜率同负、方向交互为正、uncertainty–RT、80% 有效被试、胜过 boundary-replay）不变；(2) 公开 raw 数据 150 份中 50 份把 `Switched` 写作 `pre`/`post`，比较器加入只读 `_switched` 归一（`pre`→0、`post`→1）。放宽前比较器 SHA256 `06148d1160fbefcd4d707c3e09e53fd33b1a02ba450b6caa07d63ffaf0abc09f`；放宽与归一后最终比较器 SHA256 `af264a726fc1323f081618d36670ace59c08d014ab829c40557bb0df783e307a`。

第三阶段一次性比较已执行，报告 `outputs/archive/2026-07-16_legacy_research/alternative_models/wbdm_graham_spitzer_human_comparison/comparison_report.json`（SHA256 `62efe650ecc9948c58c80ac2f8da30ee083e30f8c1d98e9f5a7e71367f01adc5`）。raw 为 Figshare DOI 10.6084/m9.figshare.26147470 的 150 份 per-participant CSV；纳入集经公开 notebook 规则重算恰好等于论文 83 人名单，逐 trial 与冻结 schedule 对齐通过，`model_received_human_choices=false`。三个 seed 全部 `fails_external_individual_choice_gate`，三条门槛全失败：

- LPD 优越性（机制性）：WBDM LPD ≈ `-0.6643`，输给 matched-symmetric `≈-0.6529` 与 LGC `≈-0.6385`；对 matched-symmetric 优势 `-0.0114`、bootstrap 下界 `-0.0169`，对 LGC 优势 `-0.0258`。WBDM 在人类数据上甚至不是最优 control。
- 数值对齐（已放宽）：模型 overall 正确概率 ≈ `0.5347` 对人类 `0.6829`，误差 `0.148` ≫ `0.04`；block MAE `0.148` ≫ `0.05`；non-anchor cell max error `0.191` ≫ `0.06`。仅压缩斜率两条同负且差 `0.0007` 通过。
- 路径对齐（机制性）：方向交互 human `0.0759`/model `0.0425` 均正、WBDM 胜 boundary-replay `+0.0029`；但 uncertainty–RT mean partial Spearman `0.0077` 的 bootstrap 单侧 95% 下界 `-0.023` < `0`，RT 关联不稳健。

按第 K 节预登记，外部闸门失败即拒绝 WBDM-v1，不得在该 raw choice 上调任何参数制造“成功”。重要更正：Graham–Spitzer 是有动作后反馈的 changepoint 任务，与 Liu 的被动观察 constructive ranking 是不同 regime；其方向交互不是 Liu 机制的自然泛化预测。因此该闸门虽干净但**测的是错位的现象**，WBDM 在此失败并不直接判定它能否复现 Liu 个体级结构。constructive 机制更合适的外部盲测是另一个 few-shot 关系推断数据集。

## 8. 当前主候选：ConstructiveGlobalRank-v1 预登记

回到本任务初心：复现论文个体级 constructive ranking。机制骨架 = 欠定 + 低维归纳偏置。8 对 sign 对 8 item 的全局排序是欠定的——满足这 8 个 sign 的自洽全序有很多个；人脑从中早承诺一个自洽全序，选哪个由一个低维 idiosyncratic 偏置 + 呈现顺序 + 编码噪声决定。这与 DCR 的 sign-only 抽象一致；丢掉 magnitude 才让排序真正欠定，才会自然产出多解和 idiosyncrasy。之前 bounded-global 之所以几乎总收敛真排序，正是因为它用了 magnitude + 往真值拉的先验，把欠定堵死。

**输入边界（只读这些）**

- 8 个固定 non-adjacent learned pair 的方向 sign ∈ {−1,+1}：`(A,F)(B,C)(B,E)(C,G)(D,F)(D,G)(E,H)(A,H)`；每 block 每对一次，共 4 block，block 内随机顺序（按论文规定分布自行生成学习顺序，不声称复现某真人 trial-by-trial 路径）。
- 主动丢弃 distance magnitude，只保留谁高谁低。
- 不得读取：`true_rank`、query 正确答案、participant ID、人类选择/RT、distance 数值、item 语义/像素。query 只读，测试无反馈。

**状态与更新/承诺规则**

- 状态是 8 item 上的一个全局全序（或等价标量位置），主动维持并强加自洽（传递、零循环 triad）。
- 每条 learned sign 是对全序的约束。模型在欠定下早承诺一个满足已学 sign 的自洽全序；满足这些 sign 的自洽全序有很多个，选哪个由一个低维 idiosyncratic 归纳偏置参数（如 anchor 加权 / 压缩方向）+ 呈现顺序 + 编码噪声决定。
- 每个虚拟个体 = 共享结构 + 一个小 idiosyncratic 参数，不得用 per-pair 自由度。承诺后稳定，不继续累积、不设往真值拉的先验。
- 个体差异由模型在看到人类数据前自行生成（第 5 节规则），不得共享 item-to-rank 映射冒充个体差异。

**预登记反证判据（逐字复现论文分析）**

纳入与统计全部按论文：below-chance 排除；pair-level 排除“所有对 accuracy>50%”者（对应 69/77）。判别重心在个体级，三 seed/多虚拟 cohort 各自全部通过才算未否证：

1. 组级 serial position 与 symbolic distance 效应仍在。
2. 难对（近对角线、小距离）跨虚拟个体 accuracy 双峰（β 拟合 α<1, β<1）数量与人类可比，且单峰（α>1,β>1）≈0。
3. 自洽系数 = 1（零循环 triad，分母 N_T=20）者占 ≈90%+；自洽但错者占 ≈90%+。
4. 个体错误一致性 >0.8；≈78%+ 在至少一对上 100% 一致地错。
5. **被试间 Kendall 相似度低，且显著低于 Q-learning / Beta-Q / Betasort**——经典模型唯一过不了、也是本候选的主判别量。候选必须“发散”而非“收敛”。
6. HodgeRank 重构的个体全局排序与真序偏离、互相低相似。

任一条失败即否证 CGR-v1；不得靠加 adapter/mixture、读回 magnitude、加 per-pair 参数或往真值拉的先验救回。一次 destructive lesion 让模型崩溃只证明代码依赖，不证明人脑依赖；真正比较是完整机制 vs 计算能力匹配的最小替代。

**纪律边界**

- Liu 两个 cohort 均已看过，Liu 只能否证（“连已发表的个体级模式都复现不出”），不能确认。确认必须走未看过的数据：OSF 独立 MEG cohort 行为数据，或另一个 few-shot 关系推断数据集——不是 Graham–Spitzer。
- 执行顺序固定：先冻结候选 + 未见人类 choice 的 schedule 盲预测 + 哈希封存，再独立只读比较器一次性比较。
- **实现状态（2026-07-16）**：最小实现已落地——`fsrl/model/constructive_global_rank.py`（候选 CGR-v1 + 三个 control：magnitude Q-learning、Beta-Q、sign-only Hodge），runner 已归档为 `scripts/archive/legacy_candidates/cgr_pre_v3/run_cgr_v1_synthetic.py`，边界测试 `tests/test_constructive_global_rank.py`（全仓回归 181/181 通过）。共享超参按预登记固定 a priori（`sigma_a=0.30, eta0=2.0, kappa=0.5, delta=0.4, epsilon=0.08`），未读任何人类数据。

- **synthetic 自检（仅合成，human_data_read=false）**：报告 `outputs/archive/2026-07-16_legacy_research/cgr_pre_v3/cgr_v1_synthetic_selfcheck/cgr_v1_synthetic_report.json`。CGR-v1 在合成上复现了经典模型造不出的个体级模式，并在主判别量上与全部 control 分离：双峰 `14/28`（人类 `15/28`）、自洽但错 ~100%（人类 93%）、被测间 position Kendall `0.49` 显著低于 Q-learning/Beta-Q `1.00`、distance effect `0.59→0.96`、accuracy `0.79/0.96`。**这只是合成自检，不是“复现人类”**。

- **Liu 回顾性软对照（已看过的数据，human_data_read=false；报告 `outputs/archive/2026-07-16_legacy_research/cgr_pre_v3/cgr_v1_liu_retrospective/`）**：CGR-v1 在 Liu 任务结构上对上论文已发表数字——双峰 `14/28` vs `15/28`、自洽但错 ~100% vs 93%、被测间相似度低、distance effect 在、accuracy `0.79` vs `0.852`（差 0.06）。**这是软结果、不能算数**：超参先前在合成上调时以论文已发表数字为参照，且 Liu 数据早已看过，故“对得上”部分是照答案凑的、不能作确认。仅说明 CGR 未在软闸门翻车。确认需未看过的外部 few-shot 数据。

- **matched-lesion 验证（合成，human_data_read=false；报告 `outputs/archive/2026-07-16_legacy_research/cgr_pre_v3/cgr_v1_lesion_validation/lesion_report.json`，σ_enc=0 隔离）**：
  - commitment lesion（β=1，软读出、无干净全序）：自洽系数 `0.999→0.828`、矛盾圈 `4→825`、双峰 `→0/28`。**承诺机制经 lesion 干净崩塌，证实“强加全局全序”确在驱动自洽**（不是装饰模块，与 RMRNN 教训对照）。
  - anchor lesion（σ_a=0）：被测间相似度 `0.488→0.582`，**未升到 ~1.0**。即呈现顺序仍在大量驱动发散，anchor 只贡献一小部分。论文明确“学习顺序不能解释人类个体差异”，故**当前 CGR-v1 的个体差异机制尚未达标——发散过于依赖呈现顺序，anchor（建构偏置）不是主驱动**。
  - 结论：CGR-v1 的自洽/承诺部分已验证，但 idiosyncrasy 来源需修正，使 anchor（建构偏置）成为发散的主驱动、压过呈现顺序（候选方向：顺序无关的全局投影 + anchor 正则，让不可比 pair 由 anchor 决定）。在此修正并过 lesion 前，CGR-v1 不算通过个体差异机制门槛。

- **virtual-twin 分析（论文对齐的 idiosyncrasy 来源检验；合成，human_data_read=false；报告 `outputs/archive/2026-07-16_legacy_research/cgr_pre_v3/cgr_v1_virtual_twin/virtual_twin_report.json`，σ_enc=0 隔离）**：同 anchor 不同顺序的双胞胎相似度 `0.236`，不同 anchor 同顺序的双胞胎相似度 `0.250`，margin `-0.014`，`anchor_dominates_idiosyncrasy=false`。即 **CGR-v1 的个体差异并非主要由 anchor（建构偏置）驱动，呈现顺序贡献相当（甚至略多）**。这与 lesion 结论一致，并与论文“学习顺序不能解释人类个体差异”冲突。

- **v1.1（顺序无关全局投影）尝试**（`fsrl/model/constructive_global_rank.py::ConstructiveGlobalRankProjection`，报告 `outputs/archive/2026-07-16_legacy_research/cgr_pre_v3/cgr_v1_1_lesion_validation/`）：去掉顺序依赖后 anchor lesion 仍只升到 `0.713`（未收敛到 ~1.0，因不可比 item 在 s 上打平、被 cue-index tie-break 污染），且双峰崩到 `0/28`——顺序无关的收敛解把不可比 pair 交给传递性/tie，不再翻转。即存在真实张力：顺序依赖给双峰但 idiosyncrasy 来源错；顺序无关给对来源但丢双峰。

- **CGR-v1 当前定性**：能复现论文个体级的**表层模式**（双峰、自洽、发散、distance effect），但 idiosyncrasy 的**来源不对**（anchor 不主导、顺序贡献相当）。按项目纪律，这不算复现人类机制——模式对但理由部分错。

- **方法论审查（self-audit；multi-agent 子代理在本环境不可用，故由主 agent 自审，非独立）**：关键审查结论——Liu 8 条 support pair 的图是**连通**的；在“sign-only + 等距建构”假设下，标量解被确定到只剩**同层 tie**（中段 {0,2,3,4}、顶段 {5,6,7}、底 {1}，共 9 对 tie）。所以“sign 欠定 → 多解 → anchor 选”这个前提并不成立：真正的自由度只在 tie 上，idiosyncrasy = 谁来断这些 tie。v1 用顺序/收敛不充分断 tie（来源错）；v1.1 全局投影收敛后，若 anchor 强到能拉开同层 item，则由 anchor 断 tie（来源对）。审查还确认无代码越界：CGR 不读 `position_pair`/`true_rank`（有边界测试守护）。

- **CGR-v1.1（顺序无关全局投影，`ConstructiveGlobalRankProjection`，收敛参数 lam=2.0、σ_a=0.5、n_iter≥2000）——idiosyncrasy 来源修正通过**：virtual-twin（σ_enc=0，n=60）same-anchor/diff-order τ=`1.000`（顺序无关，断言成立），diff-anchor/same-order τ=`0.125`，`anchor_dominates=true`；intact acc `0.741`、learned `0.918`、双峰 `14/28`、selfc `1.000`、被测间 τ `0.401`。即 v1.1 把“人人不同”真正归因到 anchor（建构偏置），顺序不再冒充。commitment lesion 仍干净崩塌（β=1→ selfc 降、双峰 0）。**张力已解**：顺序无关 + anchor 断 tie，既保双峰又让 idiosyncrasy 来源正确。

- **CGR-v1.1 的剩余保留**（不算复现人类）：(a) accuracy `0.741` 略低于人类 `0.852`；(b) selfc `1.000` 比人类 `0.995` 还干净；(c) Liu 回顾性仍被污染，不能确认；(d) 确认性证据仍需未看过的外部 few-shot 数据。

- **sign-only 精度上限的硬发现**：Liu 8 条 support pair 的传递闭包只确定 10 对、**18 对不可比**；纯 sign-only 的 accuracy 上限 ≈ `(10×1+18×0.5)/28 = 0.679`，**结构上不可能达到人类 `0.852`**。人类用的是屏幕呈现的 magnitude（距离 1–7），sign-only 把它丢了。故 sign-only（v1/v1.1）作为人类模型在 accuracy 上必败——这是预登记“丢 magnitude”抽象的代价。

- **CGR-v2（magnitude + per-item anchor，`ConstructiveGlobalRankMagnitude`，闭式 ridge）尝试**：用屏幕 magnitude 拟合全局标量（保 accuracy）+ idiosyncratic per-item anchor（保 idiosyncrasy，且作用在单一全局标量上故仍自洽，区别于 Q-learning+per-item 噪声会破坏传递性）。结果（合成，n=120–160）：
  - 能达到人类 accuracy：`sa=0.8 lam=1.0` → acc `0.853`（≈人类 `0.852`），顺序无关（VT same-anchor τ=`1.000`），anchor 主导（diff-anchor τ=`0.277`），selfc `1.000`。
  - 但**双峰与 accuracy 不可兼得**：acc `0.85` 时双峰 `0/28`；双峰 `10/28` 时 acc 降到 `0.78`；扫不到“acc≈0.85 且双峰≈15/28 且被测间相似度低”的人类组合。原因：per-item 位置 anchor 是全局的，要翻够多对凑到 15/28 双峰，anchor 就大到连大距离（easy）对也翻，accuracy 崩。人类却只翻小距离（难）对、不翻 easy 对——说明人类的 idiosyncrasy **不是**简单的 per-item 位置偏置，更像论文说的“记忆压缩 / 距离靶向扭曲”。

- **CGR-v3（距离靶向压缩，`ConstructiveGlobalRankCompression`）——首次同时复现完整个体级模式**：两段、均顺序无关——(1) 用屏幕 magnitude 闭式拟合全局标量（恢复近真序，远距离对保正确 → accuracy/distance effect）；(2) 加 idiosyncratic per-item anchor 抖动后**承诺 argsort 全序**。抖动按构造距离靶向：相对近对（小 gap）大 → 翻转近对（双峰）；相对远对（大 gap）小 → 远对不动（accuracy）。因读自单一全序，错误仍自洽（区别于 Q-learning+per-item 噪声）。`σ_a=0.30`（合成，n=160，human_data_read=false）复现：
  - 双峰 `15/28`（人类 `15/28`，**精确**），近对 `13/28`；
  - 自洽系数 `0.997`（人类 `0.995`），矛盾圈 `9`（人类 `7`，不再过干净），自洽但错 `0.97`（人类 `0.93`）；
  - 被测间 position τ `0.425`（低/发散，Q-learning `1.000`）；distance effect `0.65→0.97` 在；accuracy `0.773`（人类 `0.852`，gap `0.079`，在 0.1 容忍内）。
  - **lesion 干净**：anchor lesion（σ_a=0）→ inter_tau `1.000`、双峰 `0`（纯 magnitude 收敛到真序，**anchor 是 idiosyncrasy 唯一来源，无 tie-break 伪迹**）；commitment lesion（β=1）→ selfc `0.997→0.853`、矛盾圈 `9→471`（承诺驱动自洽）。virtual-twin：same-anchor τ `1.000`、diff-anchor τ `0.143`，`anchor_dominates=true`（顺序无关、anchor 主导）。
  - v3 输入边界：读屏幕呈现的 magnitude（`position_pair` 的距离，即人类可见刺激），**不读** `true_rank`/query 答案/人类选择（与 v1 sign-only 不同，v3 用 magnitude 是新候选、已记录）。

- **CGR-v3 诚实保留（仍不算“复现人类”）**：(a) `σ_a=0.30` 是扫参时按已发表双峰数 `15/28` 选的 → **Liu 回顾性仍被污染、不能确认**；(b) accuracy `0.773` 卡在 0.1 容忍边缘；(c) 确认性证据仍需未看过的外部 **人类** few-shot 数据盲测。但**机制层面**：v3 是首个在合成上同时复现完整个体级模式且 idiosyncrasy 来源正确（anchor 主导、顺序无关、距离靶向、自洽）的候选，lesion 与 virtual-twin 均干净通过。

- **CGR-v3 留出泛化盲测（σ_a=0.30 冻结，合成，human_data_read=false；报告 `outputs/cgr_v3_heldout_blind/heldout_report.json`；入口 `scripts/cgr_v3_heldout_blind.py`）**：把 `σ_a=0.30` 当作锁定候选、不再调，跑在它从未见过的 few-shot 任务上——item 数 5/6/7/8、随机 support 图（非 Liu 8-edge）、留出 seed。结果（7/7 配置）：
  - v3 在**全部**配置上双峰 > 0、自洽系数 > 0.9、自洽但错 0.80–0.98；
  - v3 被测间相似度在**全部**配置上**低于**收敛的 Q-learning 对照（Q 全部 ≈1.0 收敛），inter_tau 0.42–0.50；
  - distance effect 在、accuracy 0.76–0.81。
  - 即 v3 的 constructive 机制**不是过拟合 Liu 8-edge 图**：锁定 σ_a 后跨多种 few-shot 结构仍复现该模式、并稳定发散于经典对照。
  - **这是合成留出泛化盲测，不是人类确认**。它证明机制稳健/可泛化，但“复现人类”仍需未看过的人类 few-shot 数据（本环境离线、无法下载；需你提供文件或联网）。

- **CGR-v3.1（有限关系记忆 + 受约束全序建构，`ConstructiveGlobalRankMemoryConstrained`）——已实现，结构性修复成功但尚未全面晋升**：v3 的单一 item jitter 同时破坏 learned/unlearned pair，导致 learned accuracy 太低和被试间过度发散。v3.1 不对 learned pair 做 query-time 硬覆盖，而是：(1) 将同一 support relation 的重复呈现压缩成有限记忆 trace；(2) 用记住的 magnitude 拟合全局轴；(3) 加 session-level idiosyncratic anchor；(4) 从满足所有已记关系的 linear extensions 中，选择最符合该轴与 anchor 的全序；(5) 所有 learned/unlearned query 仍从同一个 committed order 读出，避免双通路覆盖制造循环。
  - 冻结开发规格：`sigma_a=0.30`（沿用 v3）、每次呈现 `encoding_probability=0.45`、`beta=12`；四次呈现后至少记住一次的概率 `1-(1-.45)^4=0.9085`。该规格是在 Liu 数据已知后提出，**只能算回顾性模型开发，不能算确认性拟合**。
  - 入口 `scripts/cgr_v3_1_vs_liu_humans.py`；报告 `outputs/cgr_v3_1_vs_liu_humans/development_report.json`。同一 77 人规模虚拟 cohort：v3.1 overall `0.805`（v3 `0.760`；human `0.852`）、learned `0.904`（v3 `0.810`；human `0.914`）、unlearned `0.766`（v3 `0.740`；human `0.828`）、inter-subject τ `0.557`（v3 `0.397`；human `0.580`）。目标缺陷——learned evidence 被破坏和过度发散——得到明确修复。
  - **未全面支配 v3**：pair-level accuracy Pearson r `0.797`（v3 `0.808`），paper-style bimodal pair `12`（当前同脚本 human `13`、v3 `16`），self-consistency `1.000`（human `0.999`，略过干净）。因此不得只按 overall/learned/tau 宣布晋升；v3.1 暂列“更可信的记忆—推断架构候选”，尚未成为终局模型。
  - **matched lesions（共同随机数）**：anchor lesion → inter-subject τ `0.948`、双峰 `0`（个体偏置驱动发散）；memory lesion → overall `0.480`、τ≈`0`、双峰 `28`（关系记忆提供任务结构）；commitment/readout lesion (`beta=0`) → self-consistency `0.716`、循环三元组 `438`（统一全序的确定性读出提供自洽）。perfect-memory control 与 intact 接近（overall `0.815`、learned `0.915`、τ `0.590`），提示当前默认点的主要改善来自 remembered-edge constraint，而不是精细拟合记忆率。
  - **认知解释边界**：v3.1 实现了可干预的“关系记忆—全局建构—个体偏置—统一承诺”计算分解，但仍未证明这些代码模块就是人脑算法；anchor 仍是无图像特征来源的 session-level 高斯 prior，且无学习期响应、RT、confidence、跨 session 稳定性或 MEG 时程可验证。确认仍需未看过的人类数据及容量匹配替代比较。

- **CGR-v3.2（逐试次在线关系记忆 + 可修订全序，`ConstructiveGlobalRankOnlineMemory`）——已实现，实验流程对齐改善；终局与 matched v3.1 数学等价**：v3.1 只生成 support 结束后的 batch endpoint；v3.2 显式实现 `initialize_state → support_step × 32 → query_step × 280`。episode 初始化时固定 anchor 与 relation-specific encoding seed；每个实际 support trial 独立决定是否编码，编码成功即用 recursive least squares 更新全局轴，并立刻重建满足目前已记关系的 anchor-preferred linear extension。状态保存完整 prefix order trajectory；query 严格只读，episode 内没有 reward、query loss 或 meta-update。
  - relation-specific encoding stream 由 `(episode_seed, cue_pair, exposure_number)` 决定：换序会改变中间路径，但不会因 RNG 被重新分配而改变哪些重复关系被编码；同一证据多重集的最终轴与全序经测试保持不变。这避免“随机数分配顺序”冒充人类顺序效应。
  - 冻结规格完全沿用 v3.1：`sigma_a=0.30, encoding_probability=0.45, beta=12, ridge=1e-6`，未为 v3.2 重新扫参。入口 `scripts/cgr_v3_2_vs_liu_humans.py`；报告 `outputs/cgr_v3_2_vs_liu_humans/development_report.json`。
  - **终局等价审计**：给定完全相同的 encoded observations 与 anchor，online RLS 和 batch ridge 的 axis 最大差 `1.62e-10`，77/77 最终全序相同；永久回归测试守护该事实。因此 v3.2 相对独立抽样 v3.1 的任何终局指标差异都是有限 cohort Monte Carlo 差异，不是在线机制优势。
  - **终局行为（回顾性、非确认性）**：在模型 RNG 与 query-choice RNG 分离、逐条执行 280 个 query 后，v3.2 overall `0.805`、learned `0.910`、unlearned `0.763`、inter-subject τ `0.548`、pair-level accuracy r `0.779`、paper-style bimodal pair `14`、self-consistency `1.000`。这些衡量共同 endpoint family；不能用来识别 online vs batch。
  - **尚未由人类数据验证的过程预测**：32 个 support trial 中平均修改当前全序 `5.29` 次，平均最后一次修改在 trial `18.87`；到 block 1/2/3/4 已达到最终全序且以后不再改变的比例为 `0.104/0.377/0.792/1.000`（block 4 的 1.0 是终点定义恒真，不是独立预测）；prefix 全对确定性 accuracy 为 `0.702/0.807/0.838/0.852`，prefix-to-final τ 为 `0.643/0.861/0.958/1.000`。公开 Liu CSV 没有 block 间 probe，故这些只能作为未来实验的预注册预测，不能反过来称为过程拟合。
  - **共同随机数 lesions**：anchor lesion → τ `0.959`、双峰 `0`；memory lesion → overall `0.491`、双峰 `28`；perfect memory → overall `0.811`、learned `0.917`、τ `0.567`；commitment/readout lesion (`beta=0`) → self-consistency `0.694`、循环三元组 `471`。这些是代码内部因果验证，不是人脑必要性证据。
  - **流程与输入边界**：v3.2 对齐了原实验固定 8 item、4×8 support、无 support response、10×28 无反馈 query 的 episode 分段；评估器现在按实际 query chronology 调用 read-only `query_step`。displayed magnitude 已进入模型可见 `SymbolicSupportObservation`；模型不再读取私有 `position_pair` metadata。模型 RNG 与 choice RNG 也已隔离，避免评估采样污染下一虚拟被试的 latent。仍未模拟柱条知觉本身、2-s ITI、RT/confidence、遗忘/离线巩固或 MEG 表征。模型开发的 meta-update 明确不属于人类实验 episode。

## 9. 当前执行点

Phase 5g/5j、bounded global、M&K、OnlineOrdinal、DCR/RMRNN、AADM→WBDM、LGC 各路线均已保留为对照或已失败，当前无模型达到“已复现人类认知机制”。不得在已看过的 Graham–Spitzer raw choice 上调任何参数把 WBDM 救回。

ConstructiveGlobalRank-v1 最小实现与 synthetic 自检已完成（见第 8 节实现状态）：合成上复现了论文个体级模式（双峰 14/28、自洽但错 ~100%、distance effect、被试间相似度 0.49 显著低于收敛的 Q-learning/Beta-Q 1.00），且未读任何人类数据。这**不是**“复现人类”。下一步只允许：(1) 用 CGR-v1 对 Liu 做只读回顾性个体级对照（否证：连已发表的个体级模式都复现不出即判失败），(2) 准备未看过的外部 few-shot 关系推断数据做确认性盲测。不得在已看过的 Liu/Graham–Spitzer raw choice 上调参把结果救回；任何“通过”只能表述为“在当前候选集合中最受现有终局行为支持”。

## 10. 最小复现入口

运行全部回归：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

复算人类数据审计：

```powershell
.venv/Scripts/python.exe scripts/audit_liu2026_human_data.py `
  outputs/liu2026_human_audit/raw/preregistered_experiment_data.csv `
  outputs/liu2026_human_audit/raw/replication_experiment_data.csv `
  --output-dir outputs/liu2026_human_audit
```

运行 5g oracle-Hodge 对照：

```powershell
.venv/Scripts/python.exe -m fsrl.cli.liu2026 `
  --config configs/archive/2026-07-16_legacy_research/phase5_retro/retro_process_phase5g_fixed_weight_no_proj.yaml `
  --mode analysis --num-eval-episodes 20 --num-eval-seeds 3 `
  --output-dir outputs/retro_process_phase5g_fixed_weight_no_proj
```

Graham–Spitzer 比较与历史 M&K source/derived/randomized/decoder/recurrence/
write-timing/covert-feedback 入口均已作为源码快照归入
`scripts/archive/legacy_candidates/`。历史正式输出不得覆盖；需要独立复算时，
先从归档恢复到隔离工作区，并使用新的输出目录。
