import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import stats

from fsrl.analysis.liu_effects import rank_positions
from fsrl.analysis.matrix import display_response_matrix
from fsrl.analysis.types import AnalysisReport
from fsrl.device import log

ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
)

_BETA_COLORS = {
    "bimodal": "#55a868",
    "high_accuracy": "#c44e52",
    "unimodal": "#aaaaaa",
    "other": "#4c72b0",
}


def configure_matplotlib_chinese() -> str | None:
    """选用系统里第一个可用的中文字体，避免 DejaVu Sans 缺字警告。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    for font in font_manager.fontManager.ttflist:
        if any(
            key in font.name
            for key in ("YaHei", "SimHei", "PingFang", "Noto Sans CJK", "WenQuanYi", "Heiti")
        ):
            matplotlib.rcParams["font.sans-serif"] = [font.name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font.name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


configure_matplotlib_chinese()


def _sorted_int_keys(curve: dict) -> list[int]:
    return sorted(int(k) for k in curve.keys())


def _curve_xy(curve: dict) -> tuple[list[int], list[float]]:
    xs = _sorted_int_keys(curve)
    ys = []
    for x in xs:
        if x in curve:
            ys.append(float(curve[x]))
        else:
            ys.append(float(curve[str(x)]))
    return xs, ys


def _beta_profile_legend():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_BETA_COLORS[k],
            label=k,
            markersize=8,
        )
        for k in ("bimodal", "high_accuracy", "unimodal", "other")
    ]


def _pick_exemplar_pair(pair_beta_fits: list[dict]) -> dict:
    bimodal = [f for f in pair_beta_fits if f["profile"] == "bimodal"]
    pool = bimodal or pair_beta_fits

    def _spread(fit: dict) -> float:
        accs = fit.get("subject_accuracies", [])
        if len(accs) < 2:
            return 0.0
        return float(np.std(accs))

    return max(pool, key=_spread)


def _plot_per_subject_rankings(report: AnalysisReport, output_dir) -> None:
    items = report.per_subject_rankings
    if not items:
        return

    nbcues = report.nbcues
    x = np.arange(nbcues)
    true_pos = [report.true_rank.index(i) for i in range(nbcues)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        x,
        true_pos,
        "o-",
        label="真实排序",
        color="green",
        lw=2.5,
        markersize=8,
        zorder=3,
    )
    for item in items:
        rank = item["subjective_rank"]
        subj_pos = [rank.index(i) for i in range(nbcues)]
        tau = item["kendall_tau_vs_true"]
        color = "#c44e52" if tau < 0.999 else "#4c72b0"
        ax.plot(
            x,
            subj_pos,
            "-",
            color=color,
            alpha=0.45,
            lw=1.0,
            zorder=1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ALPHABET[:nbcues])
    ax.set_ylabel("秩位置（越小越强）")
    ax.set_title(
        "逐被试 HodgeRank 主观排序\n"
        f"错序={report.n_subjects_with_wrong_rank}, "
        f"自洽错序={report.n_subjects_self_consistent_wrong}, "
        f"被试间 tau_mean={report.mean_inter_subject_tau:.2f}"
    )
    ax.invert_yaxis()
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="green", lw=2.5, marker="o", label="真实排序"),
        Line2D([0], [0], color="#4c72b0", lw=1.5, label="被试=真序"),
        Line2D([0], [0], color="#c44e52", lw=1.5, label="被试≠真序"),
    ]
    ax.legend(handles=legend_items, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "per_subject_rankings.png", dpi=200)
    plt.close(fig)


def _plot_liu_extended(report: AnalysisReport, output_dir) -> None:
    nbcues = report.nbcues
    pos_labels = [f"P{p}" for p in range(nbcues)]

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, style, color in (
        ("all", "-o", "#4c72b0"),
        ("supervised", "--s", "#c44e52"),
        ("unsupervised", "--^", "#55a868"),
    ):
        curve = report.serial_position_acc.get(key, {})
        if not curve:
            continue
        xs, ys = _curve_xy(curve)
        label = {"all": "全部", "supervised": "已监督 S", "unsupervised": "未监督 Q\\S"}[key]
        ax.plot(xs, ys, style, color=color, label=label, markersize=5)
    ax.set_xticks(range(nbcues))
    ax.set_xticklabels(pos_labels)
    ax.set_xlabel("秩位置（P0=最强）")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1)
    ax.set_title("Serial position effect (Fig 1F)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "serial_position_effect.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, style, color in (
        ("all", "-o", "#4c72b0"),
        ("supervised", "--s", "#c44e52"),
        ("unsupervised", "--^", "#55a868"),
    ):
        curve = report.distance_acc.get(key, {})
        if not curve:
            continue
        xs, ys = _curve_xy(curve)
        label = {"all": "全部", "supervised": "已监督 S", "unsupervised": "未监督 Q\\S"}[key]
        ax.plot(xs, ys, style, color=color, label=label, markersize=5)
    ax.set_xlabel("秩差 |rank(i)-rank(j)|")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1)
    ax.set_title("Symbolic distance effect (Fig 1G)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_effect.png", dpi=200)
    plt.close(fig)

    if report.pair_beta_fits:
        fig, ax = plt.subplots(figsize=(5, 5))
        for fit in report.pair_beta_fits:
            ax.scatter(
                fit["alpha"],
                fit["beta"],
                c=_BETA_COLORS.get(fit["profile"], "#4c72b0"),
                s=40,
                alpha=0.85,
            )
        ax.axvline(1.0, color="k", lw=0.5, ls="--")
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Beta α")
        ax.set_ylabel("Beta β")
        ax.set_title(
            f"Pair-level Beta 拟合 (n={len(report.pair_beta_fits)})\n"
            f"双峰={report.n_pair_bimodal}, 高准确={report.n_pair_high_accuracy}, "
            f"单峰={report.n_pair_unimodal}"
        )
        ax.legend(handles=_beta_profile_legend(), fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "pair_beta_fits.png", dpi=200)
        plt.close(fig)

    if report.pair_beta_fits:
        exemplar = _pick_exemplar_pair(report.pair_beta_fits)
        accs = exemplar.get("subject_accuracies", [])
        if len(accs) >= 3:
            fig, ax = plt.subplots(figsize=(4, 3))
            n_bins = min(12, max(4, len(accs) // 2))
            ax.hist(accs, bins=n_bins, range=(0, 1), color="#55a868", alpha=0.75)
            alpha, beta_p = exemplar["alpha"], exemplar["beta"]
            xs = np.linspace(0.001, 0.999, 200)
            pdf = stats.beta.pdf(xs, alpha, beta_p)
            bin_width = 1.0 / n_bins
            ax.plot(
                xs,
                pdf * len(accs) * bin_width,
                color="#c44e52",
                lw=1.5,
                label=f"Beta({alpha:.1f},{beta_p:.1f})",
            )
            ax.axvline(exemplar["mean"], color="k", ls="--", label=f"均值={exemplar['mean']:.2f}")
            ax.set_xlabel("被试准确率")
            ax.set_ylabel("计数")
            ax.set_title(
                f"跨被试分布: pair {exemplar['pair']} [{exemplar['profile']}]"
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "exemplar_pair_subject_hist.png", dpi=200)
            plt.close(fig)

    if report.subject_beta_fits:
        fig, ax = plt.subplots(figsize=(5, 5))
        for fit in report.subject_beta_fits:
            ax.scatter(
                fit["alpha"],
                fit["beta"],
                c=_BETA_COLORS.get(fit["profile"], "#4c72b0"),
                s=40,
                alpha=0.85,
            )
        ax.axvline(1.0, color="k", lw=0.5, ls="--")
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Beta α")
        ax.set_ylabel("Beta β")
        ax.set_title(
            f"被试级 Beta 拟合 (n={report.n_subjects}, 双峰={report.n_subject_bimodal})"
        )
        ax.legend(handles=_beta_profile_legend(), fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / "subject_beta_fits.png", dpi=200)
        plt.close(fig)


def plot_analysis_report(report: AnalysisReport, output_dir) -> None:
    """生成分析图表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    nbcues = report.nbcues
    labels = ALPHABET[:nbcues]

    fig, ax = plt.subplots(figsize=(6, 5))
    R_disp = display_response_matrix(report.response_matrix)
    im = ax.imshow(R_disp, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(nbcues))
    ax.set_yticks(range(nbcues))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Pairwise 响应矩阵 R\n上三角 P(i优于j)，下三角 P(j优于i)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_R.png", dpi=200)
    fig.savefig(output_dir / "heatmap_R.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    names = ["S (已监督)", "Q\\S (未监督)"]
    accs = [report.acc_supervised, report.acc_unsupervised]
    ax.bar(names, accs, color=["#4c72b0", "#dd8452"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("准确率")
    ax.set_title("已监督 vs 未监督 pair")
    fig.tight_layout()
    fig.savefig(output_dir / "acc_S_vs_QminusS.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    distances, error_rates = [], []
    pos_map = rank_positions(report.true_rank)
    for key, err in report.error_by_pair.items():
        i, j = map(int, key.split("-"))
        distances.append(abs(pos_map[i] - pos_map[j]))
        error_rates.append(err)
    if distances:
        ax.scatter(distances, error_rates, alpha=0.6)
        ax.set_xlabel("秩差 |rank(i)-rank(j)|")
        ax.set_ylabel("错误率")
        ax.set_title("错误率 vs symbolic distance")
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_distance.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(nbcues)
    true_pos = [report.true_rank.index(i) for i in range(nbcues)]
    subj_pos = [report.subjective_rank.index(i) for i in range(nbcues)]
    ax.plot(x, true_pos, "o-", label="真实排序", color="green", zorder=1)
    ax.plot(
        x,
        subj_pos,
        "s--",
        label="HodgeRank 主观排序",
        color="orange",
        markersize=7,
        zorder=2,
    )
    if true_pos == subj_pos:
        ax.text(
            0.02,
            0.02,
            "两曲线完全重合",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("秩位置（越小越强）")
    ax.set_title(
        f"排序对比 (Kendall τ={report.kendall_tau_vs_true:.3f}, "
        f"curl={report.curl_ratio:.3f})"
    )
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "true_vs_subjective_rank.png", dpi=200)
    plt.close(fig)

    _plot_liu_extended(report, output_dir)
    _plot_per_subject_rankings(report, output_dir)

    log(f"[analysis] 图表已保存至 {output_dir}")
