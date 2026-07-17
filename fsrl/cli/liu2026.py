"""Liu 2026 专用 CLI 入口。"""

import argparse
from dataclasses import fields
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import yaml


def _bootstrap_requested_device(argv: list[str]) -> None:
    """Select the device before importing modules that allocate tensors."""
    requested = None
    if "--device" in argv:
        index = argv.index("--device")
        if index + 1 < len(argv):
            requested = argv[index + 1]
    elif "--config" in argv:
        index = argv.index("--config")
        if index + 1 < len(argv):
            path = Path(argv[index + 1])
            if path.exists():
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    if isinstance(raw.get("device"), str):
                        requested = raw["device"]
                    else:
                        for values in raw.values():
                            if isinstance(values, dict) and isinstance(
                                values.get("device"), str
                            ):
                                requested = values["device"]
    if requested is not None:
        os.environ["FSRL_DEVICE"] = requested


_bootstrap_requested_device(sys.argv[1:])

import numpy as np
import torch

from fsrl.analysis.pipeline import run_full_analysis
from fsrl.config import Liu2026Config, ModelType
from fsrl.device import set_seed
from fsrl.episode.types import EpisodeRecord, TestResponse
from fsrl.training.liu2026_loop import eval_liu2026, train_liu2026


def create_parser():
    parser = argparse.ArgumentParser(
        description="FSRL x Liu 2026: Few-shot ranking meta-learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径；文件中 common/phase/analysis 段提供默认值，CLI 参数可覆盖",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="plastic_rnn",
        choices=[m.value for m in ModelType],
        help="Model type",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/liu2026")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--nbiter", type=int, default=30000)
    parser.add_argument("--pe", type=int, default=101)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None, help="训练学习率（覆盖配置）")
    parser.add_argument("--device", type=str, default=None)

    # Liu 2026 特定参数
    parser.add_argument("--support-blocks", type=int, default=4)
    parser.add_argument("--query-blocks", type=int, default=10)
    parser.add_argument(
        "--supervision-drop",
        type=int,
        default=0,
        help="Number of support pairs to drop per subject (0 = all see same)",
    )
    parser.add_argument(
        "--support-order-file",
        type=str,
        default=None,
        help="JSON 文件路径，自定义 support phase 的 position-pair 顺序（诊断实验用）",
    )
    parser.add_argument(
        "--support-drop-pairs",
        type=str,
        default=None,
        help="逗号分隔的索引，从 config.support_pairs 中删除指定 pair 后构造 support_order（诊断实验用）",
    )
    parser.add_argument(
        "--baux-test",
        type=float,
        default=0.0,
        help=(
            "Legacy runner query-loss weight. Human no-feedback constrains "
            "within-episode inputs; it does not forbid a synthetic outer-loop "
            "meta-learning target"
        ),
    )
    parser.add_argument(
        "--meta-sign-runner",
        action="store_true",
        help=(
            "Use the per-subject sign-only RetroModulRNN meta-learning runner; "
            "synthetic query targets are outer-loop loss only"
        ),
    )

    # 个体差异参数
    parser.add_argument("--pw-init-std", type=float, default=0.0)
    parser.add_argument("--encoding-noise", type=float, default=0.0)
    parser.add_argument(
        "--choice-temperature",
        type=float,
        default=0.0,
        help="决策温度异质性（query softmax 温度的对数正态标准差）",
    )
    parser.add_argument("--subject-emb-dim", type=int, default=0)

    # RetroLatentRank 混合模型参数
    parser.add_argument(
        "--retro-context-dim",
        type=int,
        default=None,
        help="RNN hidden 投影到的 context 维度（None 默认 hs//2）",
    )
    parser.add_argument(
        "--retro-detach-per-trial",
        action="store_true",
        default=True,
        help="每 trial 后 detach et/pw 防止可塑性图爆炸",
    )
    parser.add_argument(
        "--retro-no-detach",
        dest="retro_detach_per_trial",
        action="store_false",
        help="关闭每 trial detach（仅用于小实验调试）",
    )
    parser.add_argument(
        "--retro-support-aux-weight",
        type=float,
        default=0.0,
        help="support-phase teacher-sign 辅助损失权重",
    )
    parser.add_argument(
        "--retro-rnn-lr-scale",
        type=float,
        default=1.0,
        help="RNN 参数学习率缩放",
    )
    parser.add_argument(
        "--retro-freeze-rnn",
        action="store_true",
        help="冻结 RNN，只训练 context_proj + LatentRank",
    )

    # LatentRank 排序码本参数
    parser.add_argument(
        "--num-rank-modes",
        type=int,
        default=0,
        help="排序码本大小（>1 启用多样化全局排序；0 = 原行为）",
    )
    parser.add_argument(
        "--rank-mode-temperature",
        type=float,
        default=1.0,
        help="mode selector softmax 温度",
    )
    parser.add_argument(
        "--rank-diversity-weight",
        type=float,
        default=0.0,
        help="码本多样性正则权重（鼓励不同 mode 对应不同排序）",
    )
    parser.add_argument(
        "--rank-adjustment-scale",
        type=float,
        default=1.0,
        help="support-derived rank adjustment 缩放系数",
    )
    parser.add_argument(
        "--use-stochastic-rank",
        action="store_true",
        help="每个 episode 用 Gumbel-STE 硬采样一个排序 mode",
    )
    parser.add_argument(
        "--eval-stochastic-mode-sample",
        action="store_true",
        help="eval 时按 mode selector 分布对每个被试分类采样一个 mode",
    )
    parser.add_argument(
        "--rank-gumbel-scale",
        type=float,
        default=0.0,
        help="rank scores 上的 Gumbel 扰动幅度",
    )
    parser.add_argument(
        "--rank-score-noise-std",
        type=float,
        default=0.0,
        help="eval 时对 rank scores 加 per-subject 高斯噪声的标准差",
    )
    parser.add_argument(
        "--rank-score-noise-position-weight",
        type=float,
        default=0.0,
        help="rank score noise 位置加权幅度：0=不加权；>0 时中间 rank 的 cue 噪声更大，两端更小",
    )
    parser.add_argument(
        "--rank-distribution",
        type=str,
        default="gaussian_score",
        choices=["gaussian_score", "mallows", "plackett_luce"],
        help="eval 时使用的排列分布：gaussian_score 保持当前高斯噪声；"
        "mallows 在线性扩展空间上按 Kendall-tau 距离采样；"
        "plackett_luce 用 item strengths 顺序采样并投影到线性扩展",
    )
    parser.add_argument(
        "--mallows-phi",
        type=float,
        default=0.0,
        help="Mallows 分布集中度 phi（越大越集中在中心 mode），0 等价于 gaussian_score",
    )
    parser.add_argument(
        "--plackett-luce-scale",
        type=float,
        default=1.0,
        help="Plackett-Luce 温度参数（越小越接近中心 mode 的排序）",
    )
    parser.add_argument(
        "--no-mallows-anchor-mode0",
        action="store_true",
        help="默认 mode 0（真实排序）跳过 Mallows/PL 扰动；设置此项则也参与扰动",
    )
    parser.add_argument(
        "--mallows-reference",
        type=str,
        default="mode",
        choices=["mode", "true_rank"],
        help="Mallows 参考中心：mode 围绕被试选定的 mode；true_rank 围绕真实排序",
    )
    parser.add_argument(
        "--mallows-mix-weight",
        type=float,
        default=0.0,
        help="Mallows 中心以多大概率替换为真实排序（0=纯 mode，1=纯 true_rank；0–1 为混合）",
    )
    parser.add_argument(
        "--per-subject-position-precision",
        action="store_true",
        help="位置加权噪声的幅度从 subject embedding 推导，实现 per-subject 异质",
    )
    parser.add_argument(
        "--freeze-subject-embedding",
        action="store_true",
        help="冻结 subject embedding 为随机初始值（增强跨被试 mode 异质性）",
    )
    parser.add_argument(
        "--support-consistency-weight",
        type=float,
        default=0.0,
        help="support-consistency loss 权重（惩罚与 support pairs 冲突的 mode）",
    )
    parser.add_argument(
        "--use-hard-ranking-eval",
        action="store_true",
        help="eval 时使用离散 hard ranking 而非连续 scores",
    )
    parser.add_argument(
        "--hard-ranking-noise-prob",
        type=float,
        default=0.0,
        help="hard ranking 后随机翻转 pairwise 决策的概率（模拟人类不确定性）",
    )
    parser.add_argument(
        "--hard-ranking-noise-on-learned-pairs",
        action="store_true",
        help="噪声也作用于已学习的 support pairs（默认仅对未学习 pair 加噪声）",
    )
    parser.add_argument(
        "--hard-ranking-learned-noise-prob",
        type=float,
        default=None,
        help="已学习 pair 的独立翻转概率（默认与 --hard-ranking-noise-prob 相同）",
    )
    parser.add_argument(
        "--hard-ranking-noise-type",
        type=str,
        default="independent",
        choices=["independent", "distance_dependent", "systematic"],
        help="hard ranking 噪声类型：independent 逐 trial 独立翻转；"
        "distance_dependent/systematic 按真实秩差衰减并在同被试同 pair 上保持一致",
    )
    parser.add_argument(
        "--hard-ranking-noise-distance-scale",
        type=float,
        default=1.0,
        help="distance-dependent 噪声的衰减尺度（越大远距 pair 也越多噪声）",
    )
    parser.add_argument(
        "--use-probabilistic-ranking-eval",
        action="store_true",
        help="eval 时使用连续 rank scores + BTL 逐 trial 采样，替代 hard ranking",
    )
    parser.add_argument(
        "--pairwise-choice-temperature",
        type=float,
        default=None,
        help="BTL pairwise 选择温度（None 使用模型学习到的 temperature）",
    )
    parser.add_argument(
        "--block-level-rank-noise-std",
        type=float,
        default=0.0,
        help="每个 test block 初对 rank scores 加高斯漂移，模拟记忆/注意波动",
    )
    parser.add_argument(
        "--probabilistic-pair-flip-prob",
        type=float,
        default=0.0,
        help="probabilistic eval 中逐 trial 独立翻转 pairwise 决策的概率（引入传递性违反）",
    )
    parser.add_argument(
        "--pairwise-distance-noise-alpha",
        type=float,
        default=0.0,
        help="pair-dependent BTL temperature 强度：temp_eff = temp * (1 + alpha * exp(-beta*d))，d=|pos_i-pos_j|",
    )
    parser.add_argument(
        "--pairwise-distance-noise-beta",
        type=float,
        default=1.0,
        help="pair-dependent temperature 衰减尺度（beta 越大，远距 pair 越接近基础温度）",
    )
    parser.add_argument(
        "--local-drift-window-size",
        type=int,
        default=0,
        help="局部 block drift 窗口大小（1-nbcues）；0=全局漂移，>0 时每个 block 只扰动随机连续窗口",
    )
    parser.add_argument(
        "--use-bayesian-posterior-eval",
        action="store_true",
        help="eval 时用 support evidence 在线推断排序后验并 per-block 采样（替代 codebook+噪声）",
    )
    parser.add_argument(
        "--bayesian-posterior-phi",
        type=float,
        default=1.0,
        help="贝叶斯后验的 Mallows 先验集中度 phi",
    )
    parser.add_argument(
        "--bayesian-posterior-tau",
        type=float,
        default=0.3,
        help="support pair 似然温度 tau（越大对 support 证据越不信任）",
    )
    parser.add_argument(
        "--bayesian-samples-per-block",
        type=int,
        default=1,
        help="每个 block 从后验采样几个排列；>1 时取平均以降低方差",
    )
    parser.add_argument(
        "--support-observation-noise-prob",
        type=float,
        default=0.0,
        help="Phase 4：每个被试对 support pair 观测符号的随机翻转概率（0=使用教师符号）",
    )
    parser.add_argument(
        "--randomize-true-rank",
        action="store_true",
        help="每个 episode 独立采样随机真实 ranking（防止模型记忆固定 query 答案）",
    )
    parser.add_argument(
        "--diversity-use-kendall",
        action="store_true",
        help="diversity loss 使用 Kendall tau 而非余弦相似度",
    )

    # 线性扩展码本机制
    parser.add_argument(
        "--linear-extension-init",
        action="store_true",
        help="用 support 偏序的线性扩展初始化码本",
    )
    parser.add_argument(
        "--anchor-true-rank",
        action="store_true",
        help="mode 0 固定为真实排序并冻结",
    )
    parser.add_argument(
        "--anchor-target-prob",
        type=float,
        default=0.07,
        help="期望选择 mode 0 的被试比例",
    )
    parser.add_argument(
        "--mode0-bias",
        type=float,
        default=0.0,
        help="mode 0 的 logit 偏置（正数提高 correct-ranking 被试比例）",
    )
    parser.add_argument(
        "--mode-usage-weight",
        type=float,
        default=0.0,
        help="mode usage KL loss 权重",
    )
    parser.add_argument(
        "--mode-entropy-weight",
        type=float,
        default=0.0,
        help="mode 使用熵正则权重（负熵，越大越鼓励分散）",
    )
    parser.add_argument(
        "--support-consistency-margin",
        type=float,
        default=0.1,
        help="support consistency margin",
    )
    parser.add_argument(
        "--project-to-support",
        action="store_true",
        help="每个优化步后将码本投影到 support 约束",
    )
    parser.add_argument(
        "--asymmetric-diversity",
        action="store_true",
        help="diversity loss 排除 mode 0",
    )
    parser.add_argument(
        "--freeze-codebook",
        action="store_true",
        help="冻结 rank_codebook（配合线性扩展初始化使用）",
    )
    parser.add_argument(
        "--freeze-mode-selector",
        action="store_true",
        help="冻结 mode selector（被试-mode 映射由随机初始化固定）",
    )

    # === 过程模型改造：mode selector 与 support evidence 约束 ===
    parser.add_argument(
        "--mode-selector-input",
        type=str,
        default="context_only",
        choices=["context_only", "context_plus_subject_prior", "context_gated_subject"],
        help="mode selector 输入来源：context_only 只用 support evidence；"
        "context_plus_subject_prior 拼接 subject embedding 作为先验；"
        "context_gated_subject 用 context 门控 subject prior",
    )
    parser.add_argument(
        "--subject-embedding-for-mode-selector",
        action="store_true",
        help="允许 subject embedding 进入 mode selector（旧行为，过程模型目标下应为 False）",
    )
    parser.add_argument(
        "--selected-mode-support-consistency-weight",
        type=float,
        default=0.0,
        help="惩罚当前 episode 选中 mode 分布违反 support pairs 的损失权重",
    )
    parser.add_argument(
        "--mode-subject-independence-weight",
        type=float,
        default=0.0,
        help="惩罚 mode 选择可被 subject ID 预测的正则权重",
    )
    parser.add_argument(
        "--use-sequential-rank-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 trial-level sequential rank update",
    )
    parser.add_argument(
        "--rank-update-cell-hidden-dim",
        type=int,
        default=64,
        help="RankUpdateCell 隐层维度",
    )
    parser.add_argument(
        "--learn-rank-update-noise",
        action="store_true",
        help="学习 rank update 的逐 cue 噪声标准差",
    )
    parser.add_argument(
        "--rank-update-noise-init",
        type=float,
        default=0.01,
        help="rank update 噪声初始标准差",
    )
    parser.add_argument(
        "--rank-update-local-mask",
        action="store_true",
        help="RankUpdateCell 只更新当前 support trial 涉及的两个 cue（局部更新归纳偏置）",
    )
    parser.add_argument(
        "--use-pairwise-preference-accumulator",
        action="store_true",
        help="启用 pairwise preference accumulator + HodgeRank 头（无码本时的过程性排序推理）",
    )
    parser.add_argument(
        "--pairwise-preference-hidden-dim",
        type=int,
        default=64,
        help="Pairwise preference MLP 隐层维度",
    )
    parser.add_argument(
        "--hodgerank-eps",
        type=float,
        default=0.01,
        help="HodgeRank 图 Laplacian 正则化项",
    )
    parser.add_argument(
        "--pairwise-preference-nonnegative",
        action="store_true",
        help="Pairwise preference MLP 输出强制非负，evidence 方向由 teacher_sign 决定",
    )
    parser.add_argument(
        "--project-rank-to-support",
        action="store_true",
        help="HodgeRank 输出后投影到 support 约束（Route B）",
    )
    parser.add_argument(
        "--pairwise-preference-fixed-weight",
        type=float,
        default=0.0,
        help="固定每个 support trial 的 preference 幅度；>0 时忽略 MLP",
    )

    # === 过程模型诊断：组件 lesion / 干预 ===
    parser.add_argument(
        "--lesion-mode-selector-uniform",
        action="store_true",
        help="强制 mode selector 输出均匀分布（测试假设选择过程的必要性）",
    )
    parser.add_argument(
        "--lesion-single-mode",
        action="store_true",
        help="强制只使用 mode 0（真实排序），测试排序假设库的必要性",
    )
    parser.add_argument(
        "--lesion-retro-plasticity",
        action="store_true",
        help="关闭 RetroModulRNN 的 pw 更新（测试快速可塑性的必要性）",
    )
    parser.add_argument(
        "--lesion-pairwise-accumulator-off",
        action="store_true",
        help="关闭 pairwise preference accumulator / HodgeRank 路径（测试过程性排序推理的必要性）",
    )
    parser.add_argument(
        "--lesion-sequential-update-off",
        action="store_true",
        help="保留 sequential 模块与 checkpoint，只切断其行为路径",
    )
    parser.add_argument(
        "--zero-subject-embedding",
        action="store_true",
        help="把 subject embedding 置零（测试个体差异来源）",
    )
    parser.add_argument(
        "--lesion-reset-pw-after-support",
        action="store_true",
        help="support phase 后重置 plastic weights（测试 pw 记忆必要性）",
    )

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "analysis", "full"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="train 模式：从指定 checkpoint 加载参数后继续训练",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=None,
        help="评估 episode 数（默认使用配置值）",
    )
    parser.add_argument(
        "--num-eval-seeds",
        type=int,
        default=None,
        help="独立评估随机种子数",
    )
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="显式允许 checkpoint 缺键/多键；默认严格加载",
    )

    return parser


def build_config(args, yaml_data: dict | None = None, phase_key: str | None = None, arg_dests: set | None = None) -> Liu2026Config:
    config = Liu2026Config(
        model_type=ModelType(args.model),
        bs=args.bs,
        nbiter=args.nbiter,
        pe=args.pe,
        save_every=args.save_every,
        warmup=args.warmup,
        rngseed=args.seed,
        lr=args.lr if args.lr is not None else 1e-4,
        support_blocks=args.support_blocks,
        query_blocks=args.query_blocks,
        supervision_drop=args.supervision_drop,
        baux_test=args.baux_test,
        use_meta_sign_runner=args.meta_sign_runner,
        pw_init_std=args.pw_init_std,
        encoding_noise_std=args.encoding_noise,
        choice_temperature_heterogeneity=args.choice_temperature,
        subject_embedding_dim=args.subject_emb_dim,
        num_rank_modes=args.num_rank_modes,
        rank_mode_temperature=args.rank_mode_temperature,
        rank_diversity_weight=args.rank_diversity_weight,
        rank_adjustment_scale=args.rank_adjustment_scale,
        use_stochastic_rank=args.use_stochastic_rank,
        eval_stochastic_mode_sample=args.eval_stochastic_mode_sample,
        rank_gumbel_scale=args.rank_gumbel_scale,
        rank_score_noise_std=args.rank_score_noise_std,
        rank_score_noise_position_weight=args.rank_score_noise_position_weight,
        rank_distribution=args.rank_distribution,
        mallows_phi=args.mallows_phi,
        plackett_luce_scale=args.plackett_luce_scale,
        mallows_anchor_mode0=not args.no_mallows_anchor_mode0,
        mallows_reference=args.mallows_reference,
        mallows_mix_weight=args.mallows_mix_weight,
        per_subject_position_precision=args.per_subject_position_precision,
        freeze_subject_embedding=args.freeze_subject_embedding,
        support_consistency_weight=args.support_consistency_weight,
        use_hard_ranking_eval=args.use_hard_ranking_eval,
        hard_ranking_noise_prob=args.hard_ranking_noise_prob,
        hard_ranking_noise_on_learned_pairs=args.hard_ranking_noise_on_learned_pairs,
        hard_ranking_noise_type=args.hard_ranking_noise_type,
        hard_ranking_noise_distance_scale=args.hard_ranking_noise_distance_scale,
        hard_ranking_learned_noise_prob=args.hard_ranking_learned_noise_prob,
        use_probabilistic_ranking_eval=args.use_probabilistic_ranking_eval,
        pairwise_choice_temperature=args.pairwise_choice_temperature,
        block_level_rank_noise_std=args.block_level_rank_noise_std,
        probabilistic_pair_flip_prob=args.probabilistic_pair_flip_prob,
        pairwise_distance_noise_alpha=args.pairwise_distance_noise_alpha,
        pairwise_distance_noise_beta=args.pairwise_distance_noise_beta,
        local_drift_window_size=args.local_drift_window_size,
        use_bayesian_posterior_eval=args.use_bayesian_posterior_eval,
        bayesian_posterior_phi=args.bayesian_posterior_phi,
        bayesian_posterior_tau=args.bayesian_posterior_tau,
        bayesian_samples_per_block=args.bayesian_samples_per_block,
        support_observation_noise_prob=args.support_observation_noise_prob,
        randomize_true_rank=args.randomize_true_rank,
        diversity_use_kendall=args.diversity_use_kendall,
        linear_extension_init=args.linear_extension_init,
        anchor_true_rank=args.anchor_true_rank,
        anchor_target_prob=args.anchor_target_prob,
        mode0_bias=args.mode0_bias,
        mode_usage_weight=args.mode_usage_weight,
        mode_entropy_weight=args.mode_entropy_weight,
        support_consistency_margin=args.support_consistency_margin,
        project_to_support=args.project_to_support,
        asymmetric_diversity=args.asymmetric_diversity,
        freeze_codebook=args.freeze_codebook,
        freeze_mode_selector=args.freeze_mode_selector,
        retro_context_dim=args.retro_context_dim,
        retro_detach_per_trial=args.retro_detach_per_trial,
        retro_support_aux_weight=args.retro_support_aux_weight,
        retro_rnn_lr_scale=args.retro_rnn_lr_scale,
        retro_freeze_rnn=args.retro_freeze_rnn,
        mode_selector_input=args.mode_selector_input,
        subject_embedding_for_mode_selector=args.subject_embedding_for_mode_selector,
        selected_mode_support_consistency_weight=args.selected_mode_support_consistency_weight,
        mode_subject_independence_weight=args.mode_subject_independence_weight,
        use_sequential_rank_update=args.use_sequential_rank_update,
        rank_update_cell_hidden_dim=args.rank_update_cell_hidden_dim,
        learn_rank_update_noise=args.learn_rank_update_noise,
        rank_update_noise_init=args.rank_update_noise_init,
        rank_update_local_mask=args.rank_update_local_mask,
        use_pairwise_preference_accumulator=args.use_pairwise_preference_accumulator,
        pairwise_preference_hidden_dim=args.pairwise_preference_hidden_dim,
        hodgerank_eps=args.hodgerank_eps,
        pairwise_preference_nonnegative=args.pairwise_preference_nonnegative,
        project_rank_to_support=args.project_rank_to_support,
        pairwise_preference_fixed_weight=args.pairwise_preference_fixed_weight,
        lesion_mode_selector_uniform=args.lesion_mode_selector_uniform,
        lesion_single_mode=args.lesion_single_mode,
        lesion_retro_plasticity=args.lesion_retro_plasticity,
        lesion_pairwise_accumulator_off=args.lesion_pairwise_accumulator_off,
        lesion_sequential_update_off=args.lesion_sequential_update_off,
        zero_subject_embedding=args.zero_subject_embedding,
        lesion_reset_pw_after_support=args.lesion_reset_pw_after_support,
        output_dir=args.output_dir,
        allow_partial_checkpoint=args.allow_partial_checkpoint,
    )
    if args.device is not None:
        config.device = args.device
    if args.num_eval_episodes is not None:
        config.num_eval_episodes = args.num_eval_episodes
    if args.num_eval_seeds is not None:
        config.num_eval_seeds = args.num_eval_seeds

    # 应用 YAML 中未被 argparse 暴露的配置字段（如 support_pairs、query_feedback 等）
    if yaml_data is not None and phase_key is not None:
        arg_dests = arg_dests or set()
        for section in ("task", "model", "training", phase_key):
            section_data = yaml_data.get(section)
            if not isinstance(section_data, dict):
                continue
            for key, value in section_data.items():
                # key 本身或映射后的 argparse dest 已被 CLI 处理，则跳过
                if key in arg_dests or _YAML_KEY_TO_ARG.get(key, key) in arg_dests:
                    continue
                if hasattr(config, key):
                    setattr(config, key, value)
    return config


def _cue_to_position_map(record: EpisodeRecord) -> dict[int, int]:
    """Return cue -> ordinal-position mapping, validating the episode ranking."""
    rank = [int(cue) for cue in record.true_rank]
    expected = set(range(record.nbcues))
    if len(rank) != record.nbcues or set(rank) != expected:
        raise ValueError(
            f"Invalid true_rank for {record.nbcues} cues: {record.true_rank!r}"
        )
    return {cue: position for position, cue in enumerate(rank)}


def _cue_to_position_map_for_subject(
    record: EpisodeRecord, batch_index: int
) -> dict[int, int]:
    """Return the rank map for one batch element, with legacy fallback."""

    if batch_index not in record.subject_true_ranks:
        return _cue_to_position_map(record)
    rank = [int(cue) for cue in record.subject_true_ranks[batch_index]]
    expected = set(range(record.nbcues))
    if len(rank) != record.nbcues or set(rank) != expected:
        raise ValueError(
            f"Invalid subject_true_ranks[{batch_index}] for {record.nbcues} "
            f"cues: {rank!r}"
        )
    return {cue: position for position, cue in enumerate(rank)}


def _align_pair_to_positions(
    pair: tuple[int, int] | list[int], cue_to_position: dict[int, int]
) -> tuple[int, int]:
    """Map a cue-ID pair to a canonical ordinal-position pair."""
    cue_i, cue_j = int(pair[0]), int(pair[1])
    pos_i, pos_j = cue_to_position[cue_i], cue_to_position[cue_j]
    return (pos_i, pos_j) if pos_i < pos_j else (pos_j, pos_i)


def merge_episode_records(records: list[EpisodeRecord]) -> EpisodeRecord:
    """Merge evaluation episodes in a common ordinal-position coordinate system.

    With ``randomize_true_rank=True``, cue ID 0 can occupy a different ordinal
    position in every episode.  Pooling raw cue-ID pairs would therefore mix
    different psychological comparisons (for example, an adjacent pair in one
    episode with a distant pair in another).  Before pooling, this function maps
    every response, supervision pair, and query pair from cue IDs to that
    episode's true ordinal positions.  The merged record consequently has the
    identity ranking ``[0, ..., nbcues - 1]``.

    ``TestResponse.action`` remains the original presentation-relative button
    press.  ``pair`` and ``prefers_i_over_j`` are the aligned analysis fields.
    """
    if not records:
        raise ValueError("No records to merge")

    nbcues = records[0].nbcues
    if any(record.nbcues != nbcues for record in records):
        raise ValueError("Cannot merge records with different nbcues values")

    all_responses: list[TestResponse] = []
    subject_true_ranks: dict[int, list[int]] = {}
    subject_modes: dict[int, int] = {}
    subject_support_pairs: dict[int, list[tuple[int, int]]] = {}
    aligned_supervision_set: list[tuple[int, int]] | None = None
    aligned_query_set: list[tuple[int, int]] | None = None
    first_support_order: list[tuple[int, int, int]] = []
    first_dropped_pairs: list[tuple[int, int]] = []
    current_offset = 0
    identity_rank = list(range(nbcues))

    for record_index, record in enumerate(records):
        cue_to_position = _cue_to_position_map(record)
        record_supervision = sorted(
            {
                _align_pair_to_positions(item[:2], cue_to_position)
                for item in record.supervision_set
            }
        )
        record_query = sorted(
            {
                _align_pair_to_positions(pair, cue_to_position)
                for pair in record.query_set
            }
        )
        if aligned_supervision_set is None:
            aligned_supervision_set = record_supervision
            aligned_query_set = record_query
            first_support_order = [
                (
                    cue_to_position[int(cue_i)],
                    cue_to_position[int(cue_j)],
                    int(sign),
                )
                for cue_i, cue_j, sign in record.support_order
            ]
            first_dropped_pairs = sorted(
                {
                    _align_pair_to_positions(pair, cue_to_position)
                    for pair in record.support_dropped_pairs
                }
            )
        elif (
            record_supervision != aligned_supervision_set
            or record_query != aligned_query_set
        ):
            raise ValueError(
                "Episodes do not share the same supervision/query sets after "
                f"ordinal alignment (record {record_index})"
            )

        for resp in record.test_responses:
            new_batch = resp.batch_index + current_offset
            cue_i, cue_j = int(resp.pair[0]), int(resp.pair[1])
            subject_cue_to_position = _cue_to_position_map_for_subject(
                record, int(resp.batch_index)
            )
            pos_i = subject_cue_to_position[cue_i]
            pos_j = subject_cue_to_position[cue_j]
            aligned_pair = (
                (pos_i, pos_j) if pos_i < pos_j else (pos_j, pos_i)
            )
            aligned_preference = (
                resp.prefers_i_over_j
                if pos_i < pos_j
                else not resp.prefers_i_over_j
            )
            all_responses.append(
                TestResponse(
                    block_id=resp.block_id,
                    pair=aligned_pair,
                    action=resp.action,
                    prefers_i_over_j=aligned_preference,
                    correct=resp.correct,
                    batch_index=new_batch,
                    confidence=resp.confidence,
                    rt_proxy=resp.rt_proxy,
                )
            )
            subject_true_ranks[new_batch] = identity_rank.copy()
        for old_batch, mode in record.subject_modes.items():
            subject_modes[old_batch + current_offset] = mode
        for old_batch, pairs in record.subject_support_pairs.items():
            subject_cue_to_position = _cue_to_position_map_for_subject(
                record, int(old_batch)
            )
            subject_support_pairs[old_batch + current_offset] = sorted(
                {
                    _align_pair_to_positions(pair, subject_cue_to_position)
                    for pair in pairs
                }
            )
        record_subjects = {
            int(resp.batch_index) for resp in record.test_responses
        } | {int(batch) for batch in record.subject_true_ranks} | {
            int(batch) for batch in record.subject_modes
        }
        for old_batch in record_subjects:
            subject_true_ranks[old_batch + current_offset] = identity_rank.copy()
        max_batch = max(record_subjects, default=-1)
        current_offset += max_batch + 1

    # 准确率按总体重新计算
    test_acc = float(np.mean([r.correct for r in all_responses])) if all_responses else 0.0

    return EpisodeRecord(
        nbcues=nbcues,
        supervision_set=aligned_supervision_set or [],
        query_set=aligned_query_set or [],
        test_responses=all_responses,
        true_rank=identity_rank,
        subject_true_ranks=subject_true_ranks,
        subject_modes=subject_modes,
        subject_support_pairs=subject_support_pairs,
        support_order=first_support_order,
        support_dropped_pairs=first_dropped_pairs,
        provenance={
            "evaluation_units": "episode_batch_records",
            "n_source_episodes": len(records),
            "source_runs": [record.provenance for record in records],
        },
        test_perf=test_acc,
        test_perf_adjacent=None,
        test_perf_nonadjacent=None,
    )


def print_analysis_summary(report):
    print("\n--- Liu 2026 Analysis Results ---")
    print(f"Overall accuracy: {report.test_accuracy:.3f}")
    print(f"Supervised accuracy: {report.acc_supervised:.3f}")
    print(f"Unsupervised accuracy: {report.acc_unsupervised:.3f}")
    print(
        f"Serial position effect: F({report.nbcues - 1},*) = "
        f"{report.serial_position_anova_f:.2f}, "
        f"GG epsilon = {report.serial_position_anova_epsilon_gg:.3f}, "
        f"p_GG = {report.serial_position_anova_p:.4f}"
    )
    print(
        f"Distance effect: slope = {report.distance_effect_slope:.4f}, "
        f"p = {report.distance_effect_p:.4f}"
    )
    n_query_pairs = report.nbcues * (report.nbcues - 1) // 2
    print(
        f"Bimodal pairs: {report.n_pair_bimodal}/{len(report.pair_beta_fits)} "
        f"fitted ({n_query_pairs} query pairs total)"
    )
    print(
        "Mean per-record self-consistency coefficient: "
        f"{report.self_consistency_coefficient:.3f}"
    )
    print(
        "Rank-analysis records: "
        f"{report.n_subjects_rank_analysis} "
        f"(excluded below accuracy={report.n_subjects_excluded_below_accuracy}, "
        "all-pairs-majority-correct="
        f"{report.n_subjects_excluded_all_pairs_majority_correct})"
    )
    print(
        "Perfectly consistent rank-analysis records: "
        f"{report.n_subjects_perfectly_consistent}/"
        f"{report.n_subjects_rank_analysis}"
    )
    print(f"Mean inter-subject tau (all): {report.mean_inter_subject_tau:.3f}")
    print(f"Mean inter-subject tau (consistent only): {report.mean_inter_subject_tau_consistent:.3f}")
    print(
        "Mean pairwise response consistency (either direction): "
        f"{report.mean_pairwise_response_consistency:.3f}"
    )
    print(
        f"Consistent error pair ratio (>=80% error): "
        f"{report.consistent_error_pair_ratio:.3f}"
    )
    print(
        f"Consistent pair ratio either way (>=80%): "
        f"{report.consistent_pair_ratio_either_way:.3f}"
    )
    print(f"Subjects with consistent error (>=80%): {report.n_subjects_consistent_error_80}")
    print(f"Subjects with any consistent pair (>=80%): {report.n_subjects_consistent_any}")
    print(f"Subjects with wrong rank: {report.n_subjects_with_wrong_rank}")
    print(f"Self-consistent wrong rank: {report.n_subjects_self_consistent_wrong}")
    if report.mean_confidence is not None:
        print(f"Mean choice confidence: {report.mean_confidence:.3f}")
    if report.mean_rt_proxy is not None:
        print(f"Mean RT proxy: {report.mean_rt_proxy:.3f}")
    if report.mode_support_consistency is not None:
        msc = report.mode_support_consistency
        print(
            f"Support consistency: actual={msc['mean_actual_consistency']:.3f} "
            f"(random={msc['mean_random_consistency']:.3f}, "
            f"delta={msc['delta']:+.3f}, p={msc['p_value']:.4f})"
        )
    if report.error_dynamics is not None:
        ed = report.error_dynamics
        print(
            f"Error dynamics: drift_better={ed['n_drift_better']}/{ed['n_subjects']} "
            f"({ed['proportion_drift_better']:.1%}), "
            f"mean AR(1) slope={ed['mean_ar1_slope']:.3f}"
        )


# YAML / config 字段名 -> argparse dest 的映射（不完全一致时）
_YAML_KEY_TO_ARG = {
    "model_type": "model",
    "subject_embedding_dim": "subject_emb_dim",
    "encoding_noise_std": "encoding_noise",
    "choice_temperature_heterogeneity": "choice_temperature",
}


def _flatten_yaml_for_args(yaml_data: dict, phase_key: str) -> dict:
    """把 YAML 的 common/phase 段展平为 argparse 可用的默认值字典。

    也支持由训练脚本保存的扁平 config.json（顶层键与 argparse dest 同名）。
    """
    defaults: dict = {}
    # 嵌套 YAML 结构
    for section in ("task", "model", "training"):
        if isinstance(yaml_data.get(section), dict):
            for key, value in yaml_data[section].items():
                arg_key = _YAML_KEY_TO_ARG.get(key, key)
                defaults[arg_key] = value
    if isinstance(yaml_data.get(phase_key), dict):
        for key, value in yaml_data[phase_key].items():
            arg_key = _YAML_KEY_TO_ARG.get(key, key)
            defaults[arg_key] = value
    # 扁平 JSON 结构（训练保存的 config.json）：当没有嵌套 section 且存在 model_type 等键时
    has_nested = any(isinstance(yaml_data.get(s), dict) for s in ("task", "model", "training"))
    if not has_nested:
        for key, value in yaml_data.items():
            arg_key = _YAML_KEY_TO_ARG.get(key, key)
            if arg_key not in defaults:
                defaults[arg_key] = value
    return defaults


def _validate_yaml_schema(parser, yaml_data: dict) -> None:
    """Fail fast on misspelled or dead YAML keys instead of silently ignoring them."""
    if not isinstance(yaml_data, dict):
        parser.error("配置文件顶层必须是 mapping")
    config_keys = {field.name for field in fields(Liu2026Config)}
    cli_keys = {action.dest for action in parser._actions}
    known_keys = config_keys | cli_keys | set(_YAML_KEY_TO_ARG) | {
        "model_type",
        "seed",
    }
    nested_sections = {
        "task",
        "model",
        "training",
        "phase1_frozen",
        "phase2_unfrozen",
        "analysis",
    }
    has_nested = any(section in yaml_data for section in nested_sections)
    if has_nested:
        unknown_sections = set(yaml_data) - nested_sections - {"experiment", "seed"}
        if unknown_sections:
            parser.error(
                "未知配置段: " + ", ".join(sorted(unknown_sections))
            )
        for section in nested_sections:
            values = yaml_data.get(section)
            if values is None:
                continue
            if not isinstance(values, dict):
                parser.error(f"配置段 {section} 必须是 mapping")
            unknown = {
                key
                for key in values
                if key not in known_keys
                and _YAML_KEY_TO_ARG.get(key, key) not in known_keys
            }
            if unknown:
                parser.error(
                    f"配置段 {section} 含未知字段: " + ", ".join(sorted(unknown))
                )
    else:
        unknown = {
            key
            for key in yaml_data
            if key not in known_keys
            and _YAML_KEY_TO_ARG.get(key, key) not in known_keys
        }
        if unknown:
            parser.error("配置含未知字段: " + ", ".join(sorted(unknown)))


def _determine_phase_key(mode: str, resume: str | None) -> str:
    if mode in ("analysis", "eval"):
        return "analysis"
    if mode == "train" and resume is not None:
        return "phase2_unfrozen"
    return "phase1_frozen"


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        list((project_root / "fsrl").rglob("*.py"))
        + list((project_root / "scripts").rglob("*.py"))
        + list((project_root / "tests").rglob("*.py"))
    )
    for path in files:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _analysis_provenance(config, config_path: Path | None, checkpoint) -> dict:
    project_root = Path(__file__).resolve().parents[2]
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_commit = None
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    try:
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        git_dirty = None
    return {
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": _file_sha256(config_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_tree_sha256": _source_tree_sha256(project_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(config.device),
        "argv": sys.argv[1:],
    }


def main():
    parser = create_parser()

    # 第一遍：读取 --config、--mode、--resume，用于选择 YAML 中的阶段段
    partial, _ = parser.parse_known_args()
    yaml_data = None
    config_path = None
    if partial.config is not None:
        config_path = Path(partial.config)
        if not config_path.exists():
            parser.error(f"配置文件不存在: {config_path}")
        yaml_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        _validate_yaml_schema(parser, yaml_data)
        phase_key = _determine_phase_key(partial.mode, partial.resume)
        defaults = _flatten_yaml_for_args(yaml_data, phase_key)
        parser.set_defaults(**defaults)

    args = parser.parse_args()

    if args.use_hard_ranking_eval and args.use_probabilistic_ranking_eval:
        parser.error(
            "--use-hard-ranking-eval 与 --use-probabilistic-ranking-eval 不能同时启用"
        )
    if args.use_bayesian_posterior_eval and (
        args.use_hard_ranking_eval or args.use_probabilistic_ranking_eval
    ):
        parser.error(
            "--use-bayesian-posterior-eval 与 hard/probabilistic ranking 不能同时启用"
        )

    phase_key = _determine_phase_key(args.mode, args.resume) if yaml_data is not None else None
    arg_dests = {action.dest for action in parser._actions}
    config = build_config(args, yaml_data=yaml_data, phase_key=phase_key, arg_dests=arg_dests)
    try:
        config.validate(mode=args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.rngseed)
    np.random.seed(config.rngseed)
    torch.manual_seed(config.rngseed)

    if args.mode in ("train", "full"):
        print(f"=== Training {config.model_type.value} ===")
        train_liu2026(config, output_dir, resume_path=args.resume)

    if args.mode in ("eval", "analysis", "full"):
        checkpoint = (
            args.checkpoint
            if args.checkpoint is not None
            else (None if config.analytic_pairwise_control else output_dir / "net.dat")
        )
        print(f"=== Evaluating {config.model_type.value} on Liu 2026 ===")

        # 构造自定义 support_order（诊断实验用）
        support_order = None
        if args.support_order_file is not None:
            support_order_path = Path(args.support_order_file)
            if not support_order_path.exists():
                parser.error(f"support order 文件不存在: {support_order_path}")
            support_order = json.loads(support_order_path.read_text(encoding="utf-8"))
        elif args.support_drop_pairs is not None:
            drop_idxs = {int(x.strip()) for x in args.support_drop_pairs.split(",")}
            base_pairs = list(config.support_pairs)
            kept_pairs = [p for i, p in enumerate(base_pairs) if i not in drop_idxs]
            if not kept_pairs:
                parser.error("--support-drop-pairs 不能删除所有 support pairs")
            # 保持 4 blocks，每 block 包含剩余的 pairs
            support_order = kept_pairs * config.support_blocks

        records = eval_liu2026(config, checkpoint, support_order=support_order)

        print("--- Per-episode true ranks ---")
        for idx, record in enumerate(records):
            print(f"  eval episode {idx}: true_rank={record.true_rank}")

        merged_record = merge_episode_records(records)
        merged_record.provenance.update(
            _analysis_provenance(config, config_path, checkpoint)
        )

        if args.mode in ("analysis", "full"):
            print("=== Running analysis ===")
            analysis_dir = output_dir / "analysis"
            report = run_full_analysis(merged_record, analysis_dir)
            print_analysis_summary(report)


if __name__ == "__main__":
    main()
