from pathlib import Path

import numpy as np
import torch

from fsrl.device import DEVICE, ROOT_DIR, log
from fsrl.training.io import load_torch_payload, resolve_model_path


def init_subject_rank(config) -> torch.Tensor | None:
    """persistent_pw + constructive rank 时，跨 episode 保留的私有 rank 初态。"""
    if not config.use_constructive_rank:
        return None
    if config.construct_rank_init_std > 0:
        return config.construct_rank_init_std * torch.randn(
            config.bs, config.nbcues, device=DEVICE
        )
    return torch.zeros(config.bs, config.nbcues, device=DEVICE)


def resolve_episode_construct_rank(config, subject_rank=None) -> torch.Tensor | None:
    """决定本 episode 各 batch 的起始私有 rank。"""
    if not config.use_constructive_rank:
        return None
    if config.persistent_rank and subject_rank is not None:
        rank = subject_rank.detach().clone()
    else:
        rank = init_subject_rank(config)
    if rank is not None and config.rank_episode_jitter > 0:
        rank = rank + config.rank_episode_jitter * torch.randn_like(rank)
    return rank


def update_constructive_rank(rank, cue_pairs, teacher_signal, delta, config) -> None:
    """学习 trial 末：按呈现顺序与 teacher 信号更新私有 rank。"""
    if rank is None:
        return
    step = float(config.construct_rank_lr)
    stim1 = torch.tensor([p[0] for p in cue_pairs], device=rank.device, dtype=torch.long)
    stim2 = torch.tensor([p[1] for p in cue_pairs], device=rank.device, dtype=torch.long)
    sign = torch.from_numpy(teacher_signal).to(rank.device)
    batch_idx = torch.arange(rank.shape[0], device=rank.device)
    pair_delta = torch.abs(stim1 - stim2).float()
    if config.nbcues > 1:
        step_vec = step * (1.0 + pair_delta / (config.nbcues - 1))
    else:
        step_vec = torch.full((config.bs,), step, device=rank.device)
    rank[batch_idx, stim1] = rank[batch_idx, stim1] + step_vec * sign
    rank[batch_idx, stim2] = rank[batch_idx, stim2] - step_vec * sign
    noise_std = float(config.construct_rank_update_noise)
    if noise_std > 0:
        rank[batch_idx, stim1] = (
            rank[batch_idx, stim1] + noise_std * torch.randn(config.bs, device=rank.device)
        )
        rank[batch_idx, stim2] = (
            rank[batch_idx, stim2] + noise_std * torch.randn(config.bs, device=rank.device)
        )


def subject_pair_rng(config, batch_index: int, stim1: int, stim2: int) -> np.random.RandomState:
    """每 (被试, pair) 固定 RNG：同一 pair 跨 test block 决策稳定（促双峰）。"""
    i, j = min(stim1, stim2), max(stim1, stim2)
    base = config.rngseed if config.rngseed >= 0 else 0
    seed = (base + 1) * 1_000_003 + batch_index * 1_009 + i * 101 + j
    return np.random.RandomState(int(seed % (2**31 - 1)))


def rank_choice_probability(rank_diff: float, config) -> float:
    temp = max(float(config.construct_rank_temperature), 1e-6)
    return float(torch.sigmoid(torch.tensor(rank_diff / temp, device=DEVICE)).item())


def sample_constructive_actions(rank, cue_pairs, y_probs, config) -> np.ndarray:
    """测试决策：rank 比较 + 可选温度与 per-(被试,pair) 固定随机阈值。"""
    actions = np.zeros(config.bs, dtype=np.int32)
    mix = float(config.construct_rank_mix)
    use_fixed = bool(config.construct_rank_fixed_pair_rng)
    for batch_index in range(config.bs):
        stim1, stim2 = int(cue_pairs[batch_index][0]), int(cue_pairs[batch_index][1])
        rank_diff = float(rank[batch_index, stim1] - rank[batch_index, stim2])
        rank_p = rank_choice_probability(rank_diff, config)
        if mix >= 1.0:
            p = rank_p
        else:
            net_p = float(y_probs[batch_index, 1])
            p = mix * rank_p + (1.0 - mix) * net_p

        if use_fixed:
            u = subject_pair_rng(config, batch_index, stim1, stim2).random()
            actions[batch_index] = 1 if u < p else 0
        elif mix >= 1.0 and config.construct_rank_temperature >= 1.0:
            actions[batch_index] = 1 if rank_diff > 0 else 0
        else:
            actions[batch_index] = 1 if np.random.random() < p else 0
    return actions


def subject_rank_paths(output_dir: Path, rngseed: int) -> list[Path]:
    output_dir = Path(output_dir)
    paths = [output_dir / "subject_rank.pt"]
    if rngseed >= 0:
        paths.append(output_dir / f"subject_rankAE{rngseed}.pt")
    return paths


def save_subject_rank(output_dir: Path, config, subject_rank: torch.Tensor) -> None:
    if subject_rank is None or not config.use_constructive_rank:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject_rank": subject_rank.detach().cpu(),
        "train_bs": int(config.bs),
        "nbcues": int(config.nbcues),
        "rngseed": int(config.rngseed),
        "construct_rank_init_std": float(config.construct_rank_init_std),
        "rank_episode_jitter": float(config.rank_episode_jitter),
        "construct_rank_update_noise": float(config.construct_rank_update_noise),
        "construct_rank_temperature": float(config.construct_rank_temperature),
        "construct_rank_fixed_pair_rng": bool(config.construct_rank_fixed_pair_rng),
        "persistent_rank": bool(config.persistent_rank),
        "use_constructive_rank": True,
    }
    for path in subject_rank_paths(output_dir, config.rngseed):
        torch.save(payload, path)
    log(f"[save] subject_rank shape={tuple(subject_rank.shape)} → {output_dir}")


def resolve_subject_rank_path(model_path=None, subject_rank_path=None) -> Path | None:
    if subject_rank_path:
        path = Path(subject_rank_path)
        return path if path.is_absolute() else ROOT_DIR / path
    if model_path:
        model = resolve_model_path(model_path)
        candidate = model.parent / "subject_rank.pt"
        if candidate.exists():
            return candidate
    return None


def load_subject_rank(path: Path, config) -> torch.Tensor:
    payload = load_torch_payload(path)
    if isinstance(payload, dict):
        rank = payload["subject_rank"].to(DEVICE)
        train_bs = int(payload.get("train_bs", rank.shape[0]))
        saved_nbcues = int(payload.get("nbcues", rank.shape[1]))
    else:
        rank = payload.to(DEVICE)
        train_bs = rank.shape[0]
        saved_nbcues = rank.shape[1]
    if saved_nbcues != config.nbcues:
        raise ValueError(
            f"subject_rank nbcues={saved_nbcues} 与当前 nbcues={config.nbcues} 不一致: {path}"
        )
    eval_bs = config.bs
    if eval_bs == train_bs:
        return rank
    if eval_bs < train_bs:
        log(f"[subject_rank] eval bs={eval_bs} < train bs={train_bs}，使用前 {eval_bs} 个被试")
        return rank[:eval_bs].clone()
    if config.construct_rank_init_std > 0:
        pad = config.construct_rank_init_std * torch.randn(
            eval_bs - train_bs, config.nbcues, device=DEVICE
        )
    else:
        pad = torch.zeros(eval_bs - train_bs, config.nbcues, device=DEVICE)
    log(
        f"[subject_rank] eval bs={eval_bs} > train bs={train_bs}，"
        f"额外 {eval_bs - train_bs} 个槽位使用新初值"
    )
    return torch.cat([rank, pad], dim=0)


def prepare_eval_subject_rank(
    config, model_path=None, subject_rank_path=None
) -> torch.Tensor | None:
    if not config.use_constructive_rank:
        return None
    if not config.persistent_rank:
        log("[subject_rank] persistent_rank=False，eval 使用 init_subject_rank()")
        return init_subject_rank(config)
    path = resolve_subject_rank_path(model_path, subject_rank_path)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"未找到 subject_rank 文件: {path}")
        rank = load_subject_rank(path, config)
        log(f"[subject_rank] 已加载 {path} (shape={tuple(rank.shape)})")
        return rank
    if config.persistent_pw:
        log("[subject_rank] 未找到 checkpoint，使用 init_subject_rank()")
        return init_subject_rank(config)
    return None
