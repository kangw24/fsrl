"""
独立分析入口：加载模型 → 跑 eval episode → 响应结构分析 → 输出报告与图表。

用法：
    python run_analysis.py --model-path net.dat --batch-size 64
    python run_analysis.py --model-path new/net.dat --batch-size 64 \\
        --persistent-pw --subject-pw-path new/subject_pw.pt
"""

import argparse
from pathlib import Path

from simple_neo_backup import ROOT_DIR, TrainConfig, run_analyze_mode, set_seed


def main():
    ap = argparse.ArgumentParser(
        description="运行 eval episode 并执行响应结构分析（HodgeRank 等）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--output-dir", default=str(ROOT_DIR))
    ap.add_argument("--nbcues", type=int, default=8)
    ap.add_argument("--learn-blocks", type=int, default=4)
    ap.add_argument("--test-blocks", type=int, default=10)
    ap.add_argument("--supervision-size", type=int, default=8)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--analysis-dir", default=None)
    ap.add_argument(
        "--pw-init-std",
        type=float,
        default=0.0,
        help="无 subject_pw.pt 时的初值 std；加载 checkpoint 后仅用于填充额外 batch 槽位",
    )
    ap.add_argument(
        "--persistent-pw",
        action="store_true",
        help="无 subject_pw.pt 时按 persistent 规则初始化；有 checkpoint 时自动加载",
    )
    ap.add_argument(
        "--pw-episode-jitter",
        type=float,
        default=0.0,
        help="episode 初在 pw 上加 N(0, jitter) 扰动（严格复现训练态时请设 0）",
    )
    ap.add_argument(
        "--subject-pw-path",
        default=None,
        help="虚拟被试慢权重 subject_pw.pt（默认与 --model-path 同目录自动查找）",
    )
    ap.add_argument(
        "--liu-minimal",
        action="store_true",
        help="分析时假定 persistent_pw；无 subject_pw 时 pw_init_std=0.08",
    )
    args = ap.parse_args()

    if args.liu_minimal:
        args.persistent_pw = True
        if args.pw_init_std == 0.0:
            args.pw_init_std = 0.08

    config = TrainConfig(
        rngseed=args.seed,
        bs=args.batch_size,
        nbcues=args.nbcues,
        nb_learn_blocks=args.learn_blocks,
        nb_test_blocks=args.test_blocks,
        supervision_size=args.supervision_size,
        pw_init_std=args.pw_init_std,
        persistent_pw=args.persistent_pw,
        pw_episode_jitter=args.pw_episode_jitter,
    )
    set_seed(config.rngseed)

    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_dir / "analysis"
    run_analyze_mode(
        config,
        args.model_path,
        analysis_dir,
        subject_pw_path=args.subject_pw_path,
    )


if __name__ == "__main__":
    main()
