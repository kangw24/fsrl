"""Liu 2026 专用训练与评估循环。"""

import time
from enum import Enum
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy as np
import torch
from tqdm.auto import tqdm

from fsrl.device import DEVICE, log, set_seed
from fsrl.episode.liu2026 import run_liu2026_episode, run_liu2026_eval_episode
from fsrl.episode.liu2026_meta_sign import (
    run_meta_sign_episode,
    run_meta_sign_eval_episode,
)
from fsrl.episode.liu2026_leaky import (
    run_leaky_accumulator_episode,
    run_leaky_accumulator_eval_episode,
)
from fsrl.episode.liu2026_online_ordinal import (
    run_online_ordinal_episode,
    run_online_ordinal_eval_episode,
)
from fsrl.episode.types import EpisodeStats
from fsrl.model.latent_rank import LatentRankMetaLearner
from fsrl.model.q_learning import QLearningBaseline
from fsrl.model.retro_latent_rank import RetroLatentRank
from fsrl.model.retro_modul_rnn import RetroModulRNN
from fsrl.model.vanilla_rnn import VanillaRNN
from fsrl.model.leaky_rank_accumulator import LeakyRankAccumulator
from fsrl.model.online_ordinal import OnlineOrdinalPredictionError
from fsrl.training.io import load_model_state, resolve_model_path


def build_model(config):
    """根据 model_type 构建模型。"""
    requested = torch.device(config.device)
    if requested != DEVICE:
        raise ValueError(
            f"config.device={requested} but runtime device is {DEVICE}. "
            "For the CLI, pass --device before model construction; for the Python "
            "API, set FSRL_DEVICE before importing fsrl."
        )
    if config.model_type.value == "q_learning":
        return QLearningBaseline(config).to(DEVICE)
    if config.model_type.value == "vanilla_rnn":
        return VanillaRNN(config.to_model_dict()).to(DEVICE)
    if config.model_type.value == "plastic_rnn":
        return RetroModulRNN(config.to_model_dict()).to(DEVICE)
    if config.model_type.value == "latent_rank":
        return LatentRankMetaLearner(config).to(DEVICE)
    if config.model_type.value == "retro_latent_rank":
        return RetroLatentRank(config).to(DEVICE)
    if config.model_type.value == "leaky_accumulator":
        return LeakyRankAccumulator(config).to(DEVICE)
    if config.model_type.value == "online_ordinal":
        return OnlineOrdinalPredictionError(config).to(DEVICE)
    raise ValueError(f"Unknown model_type: {config.model_type}")


def load_model(config, model_path=None):
    """加载已训练模型（eval / 分析用）。"""
    if getattr(config, "analytic_pairwise_control", False) and model_path is not None:
        raise ValueError(
            "analytic_pairwise_control must not load a checkpoint; its behavior "
            "must be fully specified by the analytic rule and config"
        )
    if model_path is None and not getattr(config, "analytic_pairwise_control", False):
        raise ValueError("A checkpoint is required for non-analytic models")
    if model_path is None:
        log("[model] analytic control: using configured rules without a checkpoint")
        net = build_model(config)
        net.eval()
        return net
    path = resolve_model_path(model_path)
    log(f"[model] 从 {path} 加载参数")
    net = build_model(config)
    strict = not getattr(config, "allow_partial_checkpoint", False)
    missing, unexpected = net.load_state_dict(load_model_state(path), strict=strict)
    if missing:
        log(f"[model] 警告：state_dict 缺失以下键（将使用默认初始化）：{missing}")
    if unexpected:
        log(f"[model] 警告：state_dict 包含以下未期望的键：{unexpected}")
    net.eval()
    return net


def print_episode_summary(config, episode_index, stats, start_time):
    elapsed = time.time() - start_time
    log(f"Episode {episode_index} ====")
    log(f"Time spent on last {config.pe} iters: {elapsed:.2f}s")
    log(f"Mean loss: {stats.loss_value:.6f}")
    log(f"Backward objective: {stats.loss_objective:.6f}")
    if (
        stats.test_perf_learned is not None
        and stats.test_perf_nonlearned is not None
    ):
        log(
            "Query performance: {:.3f} | learned: {:.3f} | non-learned: {:.3f}".format(
                stats.test_perf if stats.test_perf is not None else 0.0,
                stats.test_perf_learned,
                stats.test_perf_nonlearned,
            )
        )
    else:
        log(
            "Query performance: {:.3f}".format(
                stats.test_perf if stats.test_perf is not None else 0.0
            )
        )
    pw = stats.final_pw
    if pw.numel() > 1:
        log(f"mean-abs fast state: {float(torch.mean(torch.abs(pw))):.6f}")


def train_liu2026(config, output_dir, trace_steps=False, resume_path=None):
    """Liu 2026 meta-training loop。"""
    config.validate(mode="train")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.rngseed)
    net = build_model(config)
    resume_payload = None

    if resume_path is not None:
        resume_path = Path(resume_path)
        log(f"[setup] Resuming from checkpoint: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model_state = (
            resume_payload["model_state"]
            if isinstance(resume_payload, dict) and "model_state" in resume_payload
            else resume_payload
        )
        missing, unexpected = net.load_state_dict(
            model_state,
            strict=not getattr(config, "allow_partial_checkpoint", False),
        )
        if missing:
            log(f"[setup] 警告：checkpoint 缺失以下键（将使用默认初始化）：{missing}")
        if unexpected:
            log(f"[setup] 警告：checkpoint 包含以下未期望的键：{unexpected}")

    # ``lpw`` is the learning rate for the plasticity coefficient alpha.  Keep
    # it separate from slow weights so the config field has executable meaning.
    if config.model_type.value == "retro_latent_rank":
        rnn_slow_params = []
        rnn_alpha_params = []
        other_params = []
        for name, p in net.named_parameters():
            if p.requires_grad and name == "rnn.alpha":
                rnn_alpha_params.append(p)
            elif p.requires_grad and name.startswith("rnn."):
                rnn_slow_params.append(p)
            elif p.requires_grad:
                other_params.append(p)

        if config.retro_freeze_rnn:
            for p in rnn_slow_params + rnn_alpha_params:
                p.requires_grad = False
            log("[setup] RNN encoder frozen; training context_proj + LatentRank only")
            param_groups = [{"params": other_params, "lr": config.lr}]
        else:
            param_groups = [
                {"params": other_params, "lr": config.lr},
                {
                    "params": rnn_slow_params,
                    "lr": config.lr * config.retro_rnn_lr_scale,
                },
                {
                    "params": rnn_alpha_params,
                    "lr": config.lpw * config.retro_rnn_lr_scale,
                },
            ]
            log(
                f"[setup] RNN lr scale: {config.retro_rnn_lr_scale}; "
                f"RNN lr: {config.lr * config.retro_rnn_lr_scale:.2e}; "
                f"alpha lr: {config.lpw * config.retro_rnn_lr_scale:.2e}"
            )
        optimizer = torch.optim.Adam(
            [group for group in param_groups if group["params"]],
            eps=config.eps,
            weight_decay=config.l2,
        )
    elif config.model_type.value == "plastic_rnn":
        alpha_params = []
        other_params = []
        for name, parameter in net.named_parameters():
            if not parameter.requires_grad:
                continue
            (alpha_params if name == "alpha" else other_params).append(parameter)
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": config.lr},
                {"params": alpha_params, "lr": config.lpw},
            ],
            eps=config.eps,
            weight_decay=config.l2,
        )
    else:
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.l2
        )
    start_episode = 0
    test_rewards = []
    if isinstance(resume_payload, dict) and "model_state" in resume_payload:
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        start_episode = int(resume_payload.get("next_episode_index", 0))
        test_rewards = list(resume_payload.get("test_rewards", []))
        if "numpy_rng_state" in resume_payload:
            np.random.set_state(resume_payload["numpy_rng_state"])
        if "torch_rng_state" in resume_payload:
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_states"])
        log(f"[setup] Restored optimizer/RNG state; next episode={start_episode}")
    elif resume_payload is not None:
        log(
            "[setup] Legacy state_dict resume: optimizer and RNG state were not "
            "available, so continuation is not bitwise reproducible"
        )

    log(f"[setup] Device: {DEVICE}")
    if (
        getattr(config, "use_meta_sign_runner", False)
        or config.model_type.value in {"leaky_accumulator", "online_ordinal"}
    ):
        task_description = (
            f"meta-train cues={config.meta_train_min_cues}..{config.meta_train_max_cues}; "
            f"support blocks={config.meta_train_min_support_blocks}.."
            f"{config.meta_train_max_support_blocks}; query fraction>="
            f"{config.meta_train_min_query_fraction:.2f}"
        )
    else:
        task_description = (
            f"support={config.num_support_trials}; query={config.num_query_trials}"
        )
    log(
        f"[setup] Model: {config.model_type.value}; batch size: {config.bs}; "
        f"episodes: {config.nbiter}; {task_description}; "
        f"baux_test={config.baux_test}; lpw={config.lpw}; "
        f"warmup={config.warmup}; output: {output_dir}"
    )
    log(f"[setup] Parameter shapes: {[x.size() for x in net.parameters()]}")

    trace_start = time.time()
    episode_iter = tqdm(
        range(start_episode, config.nbiter),
        desc="liu2026 training episodes",
        unit="episode",
        dynamic_ncols=True,
        file=__import__("sys").stdout,
    )

    for episode_index in episode_iter:
        should_print_summary = episode_index % config.pe == 0
        print_trace = trace_steps and should_print_summary

        net.train()
        optimizer.zero_grad()
        if config.model_type.value == "leaky_accumulator":
            leaky_result = run_leaky_accumulator_episode(
                config, net, task_regime="meta_train"
            )
            leaky_result.loss.backward()
            stats = EpisodeStats(
                loss=leaky_result.loss.detach(),
                loss_value=float(leaky_result.loss.detach().cpu()),
                loss_objective=float(leaky_result.loss.detach().cpu()),
                test_reward_mean=leaky_result.accuracy,
                nbtesttrials=int(leaky_result.targets.numel()),
                test_perf=leaky_result.accuracy,
                test_perf_adjacent=None,
                test_perf_nonadjacent=None,
                final_pw=leaky_result.evidence_state.detach(),
                final_subject_rank=None,
                test_perf_learned=None,
                test_perf_nonlearned=None,
            )
        elif config.model_type.value == "online_ordinal":
            ordinal_result = run_online_ordinal_episode(
                config, net, task_regime="meta_train"
            )
            ordinal_result.loss.backward()
            stats = EpisodeStats(
                loss=ordinal_result.loss.detach(),
                loss_value=float(ordinal_result.loss.detach().cpu()),
                loss_objective=float(ordinal_result.loss.detach().cpu()),
                test_reward_mean=ordinal_result.accuracy,
                nbtesttrials=int(ordinal_result.targets.numel()),
                test_perf=ordinal_result.accuracy,
                test_perf_adjacent=None,
                test_perf_nonadjacent=None,
                final_pw=ordinal_result.score_state.detach(),
                final_subject_rank=None,
                test_perf_learned=None,
                test_perf_nonlearned=None,
            )
        elif getattr(config, "use_meta_sign_runner", False):
            meta_result = run_meta_sign_episode(
                config, net, task_regime="meta_train"
            )
            meta_result.loss.backward()
            stats = EpisodeStats(
                loss=meta_result.loss.detach(),
                loss_value=float(meta_result.loss.detach().cpu()),
                loss_objective=float(meta_result.loss.detach().cpu()),
                test_reward_mean=meta_result.accuracy,
                nbtesttrials=int(meta_result.targets.numel()),
                test_perf=meta_result.accuracy,
                test_perf_adjacent=None,
                test_perf_nonadjacent=None,
                final_pw=meta_result.support_plastic_weights.detach(),
                final_subject_rank=None,
                test_perf_learned=None,
                test_perf_nonlearned=None,
            )
        else:
            stats = run_liu2026_episode(config, net, print_trace=print_trace)

        torch.nn.utils.clip_grad_norm_(net.parameters(), config.gc)
        if episode_index >= config.warmup:
            optimizer.step()
            if hasattr(net, "project_codebook_support"):
                net.project_codebook_support()

        test_rewards.append(stats.test_reward_mean)
        if should_print_summary:
            print_episode_summary(config, episode_index, stats, trace_start)
            trace_start = time.time()

        if episode_index % config.save_every == 0 and episode_index > 0:
            save_liu2026_checkpoint(
                config, net, output_dir, test_rewards, episode_index, optimizer
            )

    final_episode = max(start_episode - 1, config.nbiter - 1)
    save_liu2026_checkpoint(
        config, net, output_dir, test_rewards, final_episode, optimizer
    )
    return net


def save_liu2026_checkpoint(
    config, net, output_dir, test_rewards, episode_index=None, optimizer=None
):
    """保存 Liu 2026 模型 checkpoint。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if episode_index is not None:
        torch.save(net.state_dict(), output_dir / f"net_ep{episode_index:04d}.dat")
    torch.save(net.state_dict(), output_dir / f"netAE{config.rngseed}.dat")
    torch.save(net.state_dict(), output_dir / "net.dat")
    if optimizer is not None:
        torch.save(
            {
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "next_episode_index": (
                    int(episode_index) + 1 if episode_index is not None else 0
                ),
                "test_rewards": list(test_rewards),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            output_dir / "training_state.pt",
        )
    with open(output_dir / f"tAE{config.rngseed}.txt", "w") as thefile:
        for item in test_rewards[::10]:
            thefile.write(f"{item}\n")

    # 保存配置供后续批量评估使用
    config_path = output_dir / "config.json"
    try:
        cfg_dict = {}
        for k, v in vars(config).items():
            if isinstance(v, Enum):
                cfg_dict[k] = v.value
            elif isinstance(v, (str, int, float, bool, type(None), list, tuple, dict)):
                cfg_dict[k] = v
            else:
                cfg_dict[k] = str(v)
        with open(config_path, "w") as f:
            json.dump(cfg_dict, f, indent=2)
    except Exception as e:
        log(f"[save] Warning: could not save config.json: {e}")

    checkpoint_hashes = {}
    for filename in (f"netAE{config.rngseed}.dat", "net.dat"):
        path = output_dir / filename
        checkpoint_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    if episode_index is not None:
        filename = f"net_ep{episode_index:04d}.dat"
        path = output_dir / filename
        checkpoint_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_commit = None
    manifest = {
        "checkpoint_sha256": checkpoint_hashes,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "episode_index": episode_index,
        "git_commit": git_commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
    }
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    log(f"[save] Wrote checkpoint, manifest, and test-reward log to {output_dir}")


def eval_liu2026(config, model_path=None, num_episodes=None, support_order=None):
    """评估 Liu 2026 模型，返回 EpisodeRecord 列表。"""
    if num_episodes is None:
        num_episodes = config.num_eval_episodes

    net = load_model(config, model_path)
    records = []
    num_seeds = max(1, int(getattr(config, "num_eval_seeds", 1)))

    for seed_offset in range(num_seeds):
        eval_seed = int(config.rngseed) + seed_offset
        rng = np.random.RandomState(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        for ep_idx in range(num_episodes):
            if config.model_type.value == "leaky_accumulator":
                if support_order is not None:
                    raise ValueError(
                        "custom support_order is not yet defined for the per-subject "
                        "leaky accumulator runner"
                    )
                record = run_leaky_accumulator_eval_episode(config, net, rng=rng)
            elif config.model_type.value == "online_ordinal":
                if support_order is not None:
                    raise ValueError(
                        "custom support_order is not yet defined for the per-subject "
                        "online ordinal runner"
                    )
                record = run_online_ordinal_eval_episode(config, net, rng=rng)
            elif getattr(config, "use_meta_sign_runner", False):
                if support_order is not None:
                    raise ValueError(
                        "custom support_order is not yet defined for the per-subject "
                        "MetaSign runner"
                    )
                record = run_meta_sign_eval_episode(config, net, rng=rng)
            else:
                record = run_liu2026_eval_episode(
                    config, net, rng=rng, support_order=support_order
                )
            record.provenance.update(
                {
                    "eval_seed": eval_seed,
                    "episode_within_seed": ep_idx,
                }
            )
            records.append(record)

    return records
