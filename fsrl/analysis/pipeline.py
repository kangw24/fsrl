from pathlib import Path

import numpy as np

from fsrl.analysis.hodge import (
    detect_circular_triads,
    hodge_rank_scores,
    subjective_rank_from_scores,
)
from fsrl.analysis.liu_effects import analyze_liu_behavioral_effects
from fsrl.analysis.matrix import (
    analyze_error_patterns,
    build_response_matrix,
    split_supervised_unsupervised,
)
from fsrl.analysis.error_dynamics import (
    analyze_error_dynamics,
    analyze_pair_error_dynamics,
)
from fsrl.analysis.mode_support_consistency import (
    analyze_mode_support_consistency,
)
from fsrl.analysis.per_subject import analyze_per_subject_rankings, kendall_tau, spearman_rho
from fsrl.analysis.plots import plot_analysis_report
from fsrl.analysis.types import AnalysisReport
from fsrl.device import log
from fsrl.episode.types import EpisodeRecord


def analyze_episode(
    record: EpisodeRecord,
    curl_threshold: float = 0.15,
    exclude_all_pairs_majority_correct: bool = True,
) -> AnalysisReport:
    """对单次 episode 记录运行完整分析链。"""
    nbcues = record.nbcues
    responses = record.test_responses

    R = build_response_matrix(responses, nbcues)
    acc_s, acc_u, n_s, n_u = split_supervised_unsupervised(
        responses, record.supervision_set, record.query_set
    )
    error_by_pair, stable_errors = analyze_error_patterns(responses, nbcues)
    triads = detect_circular_triads(R, nbcues)
    scores, curl_ratio = hodge_rank_scores(R, nbcues)
    subj_rank = subjective_rank_from_scores(scores)
    true_rank = list(record.true_rank)

    tau = kendall_tau(true_rank, subj_rank)
    rho = spearman_rho(true_rank, subj_rank)
    test_acc = float(np.mean([r.correct for r in responses])) if responses else float("nan")

    # 置信度 / RT 代理（为后续认知 grounding 提供可相关变量）
    confidences = [r.confidence for r in responses if r.confidence is not None]
    rt_proxies = [r.rt_proxy for r in responses if r.rt_proxy is not None]
    mean_confidence = float(np.mean(confidences)) if confidences else None
    mean_rt_proxy = float(np.mean(rt_proxies)) if rt_proxies else None

    mode_support = analyze_mode_support_consistency(record)
    mean_support_consistency = mode_support["mean_actual_consistency"]

    error_dynamics = analyze_error_dynamics(record)
    pair_error_dynamics = analyze_pair_error_dynamics(record)

    liu = analyze_liu_behavioral_effects(
        responses, true_rank, record.supervision_set, record.query_set
    )
    per_subj = analyze_per_subject_rankings(
        responses,
        nbcues,
        true_rank,
        subject_true_ranks=getattr(record, "subject_true_ranks", None),
        exclude_all_pairs_majority_correct=exclude_all_pairs_majority_correct,
    )

    # 论文口径是先逐参与者计算自洽系数，再在参与者间汇总；聚合响应矩阵
    # 会把相反的个体循环互相抵消，因此不能作为群体自洽性指标。
    per_subject_consistency = [
        item["self_consistency_coefficient"]
        for item in per_subj["per_subject_rankings"]
    ]
    self_consistency = (
        float(np.mean(per_subject_consistency))
        if per_subject_consistency
        else float("nan")
    )
    subject_cycle_counts = [
        item["n_circular_triads"]
        for item in per_subj["per_subject_rankings"]
    ]

    # 论文口径：仅 perfectly self-consistent 被试间计算 tau
    consistent_items = [
        item
        for item in per_subj["per_subject_rankings"]
        if item["globally_self_consistent"]
    ]
    n_subjects_perfectly_consistent = len(consistent_items)
    if n_subjects_perfectly_consistent >= 2:
        consistent_ranks = [item["subjective_rank"] for item in consistent_items]
        pairwise_taus = []
        for i in range(len(consistent_ranks)):
            for j in range(i + 1, len(consistent_ranks)):
                pairwise_taus.append(
                    kendall_tau(consistent_ranks[i], consistent_ranks[j])
                )
        mean_inter_subject_tau_consistent = float(np.mean(pairwise_taus))
    else:
        mean_inter_subject_tau_consistent = float("nan")

    return AnalysisReport(
        nbcues=nbcues,
        response_matrix=R,
        acc_supervised=acc_s,
        acc_unsupervised=acc_u,
        n_supervised_pairs=n_s,
        n_unsupervised_pairs=n_u,
        error_by_pair=error_by_pair,
        stable_errors=stable_errors,
        circular_triads=triads,
        n_circular_triads=len(triads),
        mean_subject_circular_triads=(
            float(np.mean(subject_cycle_counts))
            if subject_cycle_counts
            else float("nan")
        ),
        total_subject_circular_triads=int(sum(subject_cycle_counts)),
        curl_ratio=curl_ratio,
        global_consistency=curl_ratio < curl_threshold,
        subjective_scores=scores,
        subjective_rank=subj_rank,
        true_rank=true_rank,
        kendall_tau_vs_true=tau,
        spearman_rho_vs_true=rho,
        test_accuracy=test_acc,
        n_subjects=liu["n_subjects"],
        serial_position_acc=liu["serial_position_acc"],
        distance_acc=liu["distance_acc"],
        pair_beta_fits=liu["pair_beta_fits"],
        subject_beta_fits=liu["subject_beta_fits"],
        beta_analysis_n_subjects=liu["beta_analysis_n_subjects"],
        beta_excluded_all_pairs_majority_correct=liu[
            "beta_excluded_all_pairs_majority_correct"
        ],
        beta_boundary_rule=liu["beta_boundary_rule"],
        beta_sensitivity=liu["beta_sensitivity"],
        n_pair_bimodal=liu["n_pair_bimodal"],
        n_pair_high_accuracy=liu["n_pair_high_accuracy"],
        n_pair_unimodal=liu["n_pair_unimodal"],
        n_subject_bimodal=liu["n_subject_bimodal"],
        mean_error_consistency=liu["mean_error_consistency"],
        mean_pairwise_response_consistency=liu[
            "mean_pairwise_response_consistency"
        ],
        n_subjects_consistent_error_80=liu["n_subjects_consistent_error_80"],
        consistent_error_pair_ratio=liu["consistent_error_pair_ratio"],
        consistent_pair_ratio_either_way=liu["consistent_pair_ratio_either_way"],
        n_subjects_consistent_any=liu["n_subjects_consistent_any"],
        per_subject_rankings=per_subj["per_subject_rankings"],
        mean_per_subject_tau_vs_true=per_subj["mean_per_subject_tau_vs_true"],
        mean_inter_subject_tau=per_subj["mean_inter_subject_tau"],
        n_subjects_with_wrong_rank=per_subj["n_subjects_with_wrong_rank"],
        n_subjects_self_consistent_wrong=per_subj["n_subjects_self_consistent_wrong"],
        n_subjects_rank_analysis=per_subj["n_subjects_rank_analysis"],
        n_subjects_excluded_all_pairs_majority_correct=per_subj[
            "n_subjects_excluded_all_pairs_majority_correct"
        ],
        n_subjects_excluded_below_accuracy=per_subj[
            "n_subjects_excluded_below_accuracy"
        ],
        self_consistency_coefficient=self_consistency,
        n_subjects_perfectly_consistent=n_subjects_perfectly_consistent,
        mean_inter_subject_tau_consistent=mean_inter_subject_tau_consistent,
        serial_position_anova_f=liu["serial_position_anova_f"],
        serial_position_anova_p=liu["serial_position_anova_p"],
        serial_position_anova_epsilon_gg=liu[
            "serial_position_anova_epsilon_gg"
        ],
        distance_effect_slope=liu["distance_effect_slope"],
        distance_effect_p=liu["distance_effect_p"],
        distance_effect_n_subjects=liu["distance_effect_n_subjects"],
        mean_confidence=mean_confidence,
        mean_rt_proxy=mean_rt_proxy,
        mode_support_consistency=mode_support,
        mean_support_consistency=mean_support_consistency,
        error_dynamics=error_dynamics,
        pair_error_dynamics=pair_error_dynamics,
        provenance=getattr(record, "provenance", {}),
    )


def save_analysis_report(report: AnalysisReport, output_dir: Path) -> Path:
    """保存 JSON 报告。"""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_json_dict(), f, indent=2, ensure_ascii=False)
    log(f"[analysis] 已保存报告: {path}")
    return path


def save_raw_responses(record: EpisodeRecord, output_dir: Path) -> Path:
    """Persist trial-level responses so every aggregate can be recomputed."""
    import json
    from dataclasses import asdict

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "responses.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for response in record.test_responses:
            handle.write(json.dumps(asdict(response), ensure_ascii=False) + "\n")
    return path


def run_full_analysis(record: EpisodeRecord, output_dir: Path) -> AnalysisReport:
    """分析总入口：计算 + 保存报告 + 作图。"""
    log("[analysis] 开始响应结构分析...")
    save_raw_responses(record, output_dir)
    report = analyze_episode(record)
    save_analysis_report(report, output_dir)
    plot_analysis_report(report, output_dir)
    log(
        f"[analysis] 完成: test_acc={report.test_accuracy:.3f}, "
        f"acc_S={report.acc_supervised:.3f}, acc_Q\\S={report.acc_unsupervised:.3f}, "
        f"triads={report.n_circular_triads}, self_consistency={report.self_consistency_coefficient:.3f}, "
        f"curl={report.curl_ratio:.3f}, "
        f"Kendall τ={report.kendall_tau_vs_true:.3f}, "
        f"n_subj={report.n_subjects}, pair_bimodal={report.n_pair_bimodal}, "
        f"err_consistency={report.mean_error_consistency:.3f}, "
        f"wrong_rank={report.n_subjects_with_wrong_rank}, "
        f"inter_subj_τ={report.mean_inter_subject_tau:.3f}, "
        f"perfectly_consistent={report.n_subjects_perfectly_consistent}, "
        f"inter_subj_τ_consistent={report.mean_inter_subject_tau_consistent:.3f}, "
        f"serial_pos_F={report.serial_position_anova_f:.2f}, "
        f"distance_slope={report.distance_effect_slope:.4f}, "
        f"mean_confidence={(f'{report.mean_confidence:.3f}' if report.mean_confidence is not None else 'n/a')}, "
        f"mean_support_consistency={(f'{report.mean_support_consistency:.3f}' if report.mean_support_consistency is not None else 'n/a')}"
    )
    if report.error_dynamics is not None:
        log(
            f"[analysis] error_dynamics: drift_better={report.error_dynamics['n_drift_better']}/"
            f"{report.error_dynamics['n_subjects']} "
            f"({report.error_dynamics['proportion_drift_better']:.1%})"
        )
    return report
