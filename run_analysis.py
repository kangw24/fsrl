"""
独立分析入口：加载模型 → 跑 eval episode → 响应结构分析 → 输出报告与图表。

用法：
    python run_analysis.py --model-path net.dat --batch-size 64
"""

from pathlib import Path

from simple_neo import TrainConfig, set_seed
from simple_neo import parse_args as neo_parse_args
from simple_neo import run_analyze_mode


def main():
    parser = neo_parse_args.__wrapped__ if hasattr(neo_parse_args, "__wrapped__") else None
    # 复用 simple_neo 的 CLI，但本脚本始终进入分析模式
    import argparse
    from simple_neo import ROOT_DIR

    ap = argparse.ArgumentParser(
        description="运行 eval episode 并执行响应结构分析（HodgeRank 等）。",
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
    args = ap.parse_args()

    config = TrainConfig(
        rngseed=args.seed,
        bs=args.batch_size,
        nbcues=args.nbcues,
        nb_learn_blocks=args.learn_blocks,
        nb_test_blocks=args.test_blocks,
        supervision_size=args.supervision_size,
    )
    set_seed(config.rngseed)

    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else output_dir / "analysis"
    run_analyze_mode(config, args.model_path, analysis_dir)


if __name__ == "__main__":
    main()
