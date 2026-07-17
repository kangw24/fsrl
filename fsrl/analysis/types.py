from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class AnalysisReport:
    """单次 episode 分析结果。"""

    nbcues: int
    response_matrix: np.ndarray
    acc_supervised: float
    acc_unsupervised: float
    n_supervised_pairs: int
    n_unsupervised_pairs: int
    error_by_pair: dict[str, float]
    stable_errors: list[str]
    circular_triads: list[list[int]]
    n_circular_triads: int
    curl_ratio: float
    global_consistency: bool
    subjective_scores: np.ndarray
    subjective_rank: list[int]
    true_rank: list[int]
    kendall_tau_vs_true: float
    spearman_rho_vs_true: float
    test_accuracy: float
    mean_subject_circular_triads: float = float("nan")
    total_subject_circular_triads: int = 0
    n_subjects: int = 0
    serial_position_acc: dict = field(default_factory=dict)
    distance_acc: dict = field(default_factory=dict)
    pair_beta_fits: list[dict] = field(default_factory=list)
    subject_beta_fits: list[dict] = field(default_factory=list)
    beta_analysis_n_subjects: int = 0
    beta_excluded_all_pairs_majority_correct: int = 0
    beta_boundary_rule: str = ""
    beta_sensitivity: dict = field(default_factory=dict)
    n_pair_bimodal: int = 0
    n_pair_high_accuracy: int = 0
    n_pair_unimodal: int = 0
    n_subject_bimodal: int = 0
    mean_error_consistency: float = float("nan")
    mean_pairwise_response_consistency: float = float("nan")
    n_subjects_consistent_error_80: int = 0
    consistent_error_pair_ratio: float = float("nan")
    consistent_pair_ratio_either_way: float = float("nan")
    n_subjects_consistent_any: int = 0
    per_subject_rankings: list = field(default_factory=list)
    mean_per_subject_tau_vs_true: float = float("nan")
    mean_inter_subject_tau: float = float("nan")
    n_subjects_with_wrong_rank: int = 0
    n_subjects_self_consistent_wrong: int = 0
    n_subjects_rank_analysis: int = 0
    n_subjects_excluded_all_pairs_majority_correct: int = 0
    n_subjects_excluded_below_accuracy: int = 0
    self_consistency_coefficient: float = float("nan")
    n_subjects_perfectly_consistent: int = 0
    mean_inter_subject_tau_consistent: float = float("nan")
    serial_position_anova_f: float = float("nan")
    serial_position_anova_p: float = float("nan")
    serial_position_anova_epsilon_gg: float = float("nan")
    distance_effect_slope: float = float("nan")
    distance_effect_p: float = float("nan")
    distance_effect_n_subjects: int = 0
    mean_confidence: float | None = None
    mean_rt_proxy: float | None = None
    mode_support_consistency: dict | None = None
    mean_support_consistency: float | None = None
    error_dynamics: dict | None = None
    pair_error_dynamics: dict | None = None
    provenance: dict = field(default_factory=dict)

    def to_json_dict(self):
        """转为可 JSON 序列化的字典。"""
        data = asdict(self)
        data["response_matrix"] = self.response_matrix.tolist()
        data["subjective_scores"] = self.subjective_scores.tolist()
        return json_sanitize(data)


def json_sanitize(obj):
    """把 numpy 标量/数组递归转为原生 Python 类型，供 json.dump 使用。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    return obj
