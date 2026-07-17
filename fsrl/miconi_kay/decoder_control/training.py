"""Fixed-budget, frozen-core training for the decoder-capacity control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from fsrl.miconi_kay.decoder_control.protocol import (
    DECODER_CONTROL_TRAINABLE_PARAMETERS,
    DECODER_CONTROL_TRAINING_CONFIGURATION,
    decoder_control_evaluation_plan,
    decoder_control_training_plan,
)
from fsrl.miconi_kay.randomized.curriculum import (
    RandomizedLinkedCurriculumConfig,
    run_randomized_linked_episode,
)
from fsrl.miconi_kay.randomized.training import randomized_tree_sha256
from fsrl.miconi_kay.source.checkpoint import (
    PUBLIC_CHECKPOINT_SHA256,
    SOURCE_STATE_KEYS,
    load_source_checkpoint_strict,
    sha256_file,
    source_tree_sha256,
    validate_source_state_dict,
    verify_public_checkpoint_identity,
)
from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.train import set_source_seed


@dataclass(frozen=True)
class DecoderControlFineTuneConfig:
    episodes: int = 1000
    learning_rate: float = 1e-5
    seed: int = 104
    print_every: int = 25

    def validate(self) -> None:
        if self.episodes <= 0 or self.learning_rate <= 0:
            raise ValueError("episodes and learning rate must be positive")
        if self.seed < 0 or self.print_every <= 0:
            raise ValueError("seed must be non-negative and print interval positive")


@dataclass(frozen=True)
class DecoderControlSummary:
    episode_index: int
    loss_value: float
    item_count: int
    bridge_trials: int
    query_accuracy: float
    mean_correct_choice_probability: float
    mean_absolute_post_bridge_fast_weight: float
    trainable_gradient_norm_before_clip: float


def decoder_control_tree_sha256() -> str:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py"), key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _forbidden_training_inputs() -> list[str]:
    return [
        "Liu graph",
        "Liu human choices",
        "Liu endpoint metrics",
        "participant identifiers",
        "teacher_sign",
        "direct correct-action target",
        "unchosen-action reward",
    ]


def _observed_training_configuration(
    source: MiconiKaySourceConfig,
    curriculum: RandomizedLinkedCurriculumConfig,
    fine_tune: DecoderControlFineTuneConfig,
) -> dict[str, object]:
    return {
        "episodes": fine_tune.episodes,
        "learning_rate": fine_tune.learning_rate,
        "seed": fine_tune.seed,
        "batch_size": source.batch_size,
        "hidden_size": source.hidden_size,
        "cue_size": source.cue_size,
        "source_min_items": source.min_items,
        "source_max_items": source.max_items,
        "trial_steps": source.trial_steps,
        "response_step": source.response_step,
        "item_counts": list(curriculum.item_counts),
        "component_trials": curriculum.component_trials,
        "minimum_bridge_trials": curriculum.minimum_bridge_trials,
        "maximum_bridge_trials": curriculum.maximum_bridge_trials,
        "query_branches": curriculum.query_branches,
        "query_loss_multiplier": curriculum.query_loss_multiplier,
        "trainable_parameters": list(DECODER_CONTROL_TRAINABLE_PARAMETERS),
    }


def register_decoder_control_protocol(
    output_dir: str | Path,
    *,
    base_checkpoint: Path,
    source_config: MiconiKaySourceConfig,
    curriculum_config: RandomizedLinkedCurriculumConfig,
    fine_tune_config: DecoderControlFineTuneConfig,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty registered run directory: {directory}"
        )
    if 8 in curriculum_config.item_counts:
        raise ValueError("formal 8-item challenge cannot enter decoder training")
    if curriculum_config.maximum_bridge_trials >= 4:
        raise ValueError("formal four-bridge duration cannot enter decoder training")
    observed = _observed_training_configuration(
        source_config, curriculum_config, fine_tune_config
    )
    registration = {
        "registration_kind": "prospective_local_code_registration",
        "limitation": "not externally timestamped or independently registered",
        "registration_moment": "written before optimizer construction and first step",
        "claim_ceiling": (
            "frozen-core decoder capacity diagnostic only; not source parity, "
            "Liu transfer, or evidence of a human mechanism"
        ),
        "selection_rule": "fixed final episode; no early stopping or Liu metric",
        "training_objective_audit": {
            "outer_objective": "sampled-action REINFORCE actor-critic",
            "online_feedback": "chosen-action-contingent signed reward only",
            "direct_correct_action_cross_entropy": False,
            "counterfactual_unchosen_action_reward_used": False,
            "exhaustive_challenge_pairs_enumerated": False,
            "exact_eight_item_four_bridge_condition_held_out": True,
        },
        "parameter_intervention": {
            "trainable_parameters": list(DECODER_CONTROL_TRAINABLE_PARAMETERS),
            "all_other_source_parameters": "requires_grad=False and bitwise audited",
        },
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "source_config": source_config.to_manifest_dict(),
        "curriculum_config": asdict(curriculum_config),
        "fine_tune_config": asdict(fine_tune_config),
        "registered_training_plan": decoder_control_training_plan(),
        "observed_training_configuration": observed,
        "is_formal_training_configuration": (
            observed == DECODER_CONTROL_TRAINING_CONFIGURATION
        ),
        "evaluation_plan": decoder_control_evaluation_plan(),
        "source_tree_sha256": source_tree_sha256(),
        "randomized_tree_sha256": randomized_tree_sha256(),
        "decoder_control_tree_sha256": decoder_control_tree_sha256(),
        "forbidden_training_inputs": _forbidden_training_inputs(),
    }
    path = directory / "protocol_registration.json"
    path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _parameter_audit(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, object]:
    trainable = set(DECODER_CONTROL_TRAINABLE_PARAMETERS)
    frozen = sorted(set(before) - trainable)
    frozen_unchanged = {
        name: bool(torch.equal(before[name], after[name])) for name in frozen
    }
    changed_trainable = sorted(
        name for name in trainable if not torch.equal(before[name], after[name])
    )
    return {
        "trainable_parameter_names": sorted(trainable),
        "frozen_parameter_names": frozen,
        "frozen_bitwise_unchanged": all(frozen_unchanged.values()),
        "frozen_unchanged_by_parameter": frozen_unchanged,
        "changed_trainable_parameters": changed_trainable,
        "before_sha256": {name: _tensor_sha256(before[name]) for name in sorted(before)},
        "after_sha256": {name: _tensor_sha256(after[name]) for name in sorted(after)},
    }


def validate_decoder_control_manifest(
    checkpoint: str | Path, manifest_path: str | Path | None
) -> dict[str, object]:
    """Validate the complete frozen-core provenance before any evaluation."""

    checkpoint_path = Path(checkpoint)
    if manifest_path is None or not Path(manifest_path).is_file():
        raise ValueError("decoder control checkpoint requires --active-manifest")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("decoder control checkpoint SHA-256 does not match manifest")
    if manifest.get("base_checkpoint_sha256") != PUBLIC_CHECKPOINT_SHA256[
        "net_active.dat"
    ]:
        raise ValueError("decoder control was not based on public active")
    if manifest.get("selection_rule") != (
        "fixed final episode; no early stopping or Liu metric"
    ):
        raise ValueError("decoder control selection rule is not admissible")
    registration_path = path.parent / "protocol_registration.json"
    if not registration_path.is_file():
        raise ValueError("decoder control lacks pre-optimization registration")
    if manifest.get("protocol_registration_sha256") != sha256_file(
        registration_path
    ):
        raise ValueError("decoder control registration SHA-256 mismatch")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if manifest.get("prospective_registration") != registration:
        raise ValueError("decoder manifest does not embed registration exactly")
    for key in ("source_config", "curriculum_config", "fine_tune_config"):
        if manifest.get(key) != registration.get(key):
            raise ValueError(f"decoder manifest and registration disagree: {key}")
    if registration.get("registered_training_plan") != decoder_control_training_plan():
        raise ValueError("decoder training plan differs from locked protocol")
    if registration.get("evaluation_plan") != decoder_control_evaluation_plan():
        raise ValueError("decoder evaluation plan differs from locked protocol")
    if registration.get("is_formal_training_configuration") is not True:
        raise ValueError("decoder checkpoint did not use formal training parameters")
    expected_objective = {
        "outer_objective": "sampled-action REINFORCE actor-critic",
        "online_feedback": "chosen-action-contingent signed reward only",
        "direct_correct_action_cross_entropy": False,
        "counterfactual_unchosen_action_reward_used": False,
        "exhaustive_challenge_pairs_enumerated": False,
        "exact_eight_item_four_bridge_condition_held_out": True,
    }
    if registration.get("training_objective_audit") != expected_objective:
        raise ValueError("decoder control objective differs from reward-only design")
    current_hashes = {
        "source_tree_sha256": source_tree_sha256(),
        "randomized_tree_sha256": randomized_tree_sha256(),
        "decoder_control_tree_sha256": decoder_control_tree_sha256(),
    }
    for key, value in current_hashes.items():
        if registration.get(key) != value or manifest.get(key) != value:
            raise ValueError(f"decoder control code identity is stale: {key}")
    audit = manifest.get("parameter_audit")
    if not isinstance(audit, dict):
        raise ValueError("decoder control manifest lacks parameter audit")
    trainable = set(DECODER_CONTROL_TRAINABLE_PARAMETERS)
    frozen = set(SOURCE_STATE_KEYS) - trainable
    if set(audit.get("trainable_parameter_names", [])) != trainable:
        raise ValueError("decoder control trainable-parameter audit is wrong")
    if set(audit.get("frozen_parameter_names", [])) != frozen:
        raise ValueError("decoder control frozen-parameter audit is wrong")
    if audit.get("frozen_bitwise_unchanged") is not True:
        raise ValueError("decoder control changed a frozen core parameter")
    before_hashes = audit.get("before_sha256", {})
    after_hashes = audit.get("after_sha256", {})
    if any(before_hashes.get(name) != after_hashes.get(name) for name in frozen):
        raise ValueError("decoder control frozen hashes differ before and after")
    if not set(audit.get("changed_trainable_parameters", [])).issubset(trainable):
        raise ValueError("decoder audit reports a changed non-readout parameter")
    return manifest


def train_decoder_control(
    *,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    source_config: MiconiKaySourceConfig,
    curriculum_config: RandomizedLinkedCurriculumConfig,
    fine_tune_config: DecoderControlFineTuneConfig,
    device: torch.device | str | None = None,
    require_public_active: bool = True,
) -> tuple[MiconiKayRetroModulRNN, list[DecoderControlSummary], Path]:
    source_config.validate()
    curriculum_config.validate()
    fine_tune_config.validate()
    checkpoint_path = Path(base_checkpoint)
    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_source_seed(fine_tune_config.seed)
    model = MiconiKayRetroModulRNN(source_config.to_model_dict()).to(resolved_device)
    load_source_checkpoint_strict(model, checkpoint_path)
    if require_public_active:
        verify_public_checkpoint_identity(checkpoint_path, "active")
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    parameter_map = dict(model.named_parameters())
    if set(DECODER_CONTROL_TRAINABLE_PARAMETERS) - set(parameter_map):
        raise ValueError("registered decoder parameters are absent from source model")
    for name, parameter in parameter_map.items():
        parameter.requires_grad_(name in DECODER_CONTROL_TRAINABLE_PARAMETERS)
    trainable_parameters = [
        parameter_map[name] for name in DECODER_CONTROL_TRAINABLE_PARAMETERS
    ]

    set_source_seed(fine_tune_config.seed)
    registration_path = register_decoder_control_protocol(
        output_dir,
        base_checkpoint=checkpoint_path,
        source_config=source_config,
        curriculum_config=curriculum_config,
        fine_tune_config=fine_tune_config,
    )
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=fine_tune_config.learning_rate,
        eps=source_config.adam_epsilon,
        weight_decay=source_config.weight_decay,
    )
    history: list[DecoderControlSummary] = []
    for episode_index in range(fine_tune_config.episodes):
        optimizer.zero_grad(set_to_none=True)
        stats = run_randomized_linked_episode(source_config, curriculum_config, model)
        stats.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, source_config.gradient_clip
        )
        optimizer.step()
        row = DecoderControlSummary(
            episode_index=episode_index,
            loss_value=stats.loss_value,
            item_count=stats.item_count,
            bridge_trials=stats.bridge_trials,
            query_accuracy=stats.query_accuracy,
            mean_correct_choice_probability=stats.mean_correct_choice_probability,
            mean_absolute_post_bridge_fast_weight=(
                stats.mean_absolute_post_bridge_fast_weight
            ),
            trainable_gradient_norm_before_clip=float(gradient_norm.detach().cpu()),
        )
        history.append(row)
        if (
            episode_index == 0
            or episode_index == fine_tune_config.episodes - 1
            or (episode_index + 1) % fine_tune_config.print_every == 0
        ):
            print(
                "decoder-capacity-control-v1 "
                f"episode={episode_index + 1}/{fine_tune_config.episodes} "
                f"items={row.item_count} bridge={row.bridge_trials} "
                f"loss={row.loss_value:.6f} query_acc={row.query_accuracy:.4f} "
                f"query_p={row.mean_correct_choice_probability:.4f}"
            )

    after = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    validate_source_state_dict(after)
    audit = _parameter_audit(before, after)
    if audit["frozen_bitwise_unchanged"] is not True:
        raise RuntimeError("a frozen source-core parameter changed during decoder control")

    directory = Path(output_dir)
    checkpoint = directory / "net.dat"
    torch.save(model.state_dict(), checkpoint)
    history_path = directory / "training_history.json"
    history_path.write_text(
        json.dumps([asdict(row) for row in history], indent=2) + "\n",
        encoding="utf-8",
    )
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    manifest = {
        "claim_status": (
            "frozen-core decoder capacity diagnostic only; not source parity, "
            "not Liu transfer, and not a human-mechanism result"
        ),
        "selection_rule": "fixed final episode; no early stopping or Liu metric",
        "base_checkpoint": str(checkpoint_path.resolve()),
        "base_checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_config": source_config.to_manifest_dict(),
        "curriculum_config": asdict(curriculum_config),
        "fine_tune_config": asdict(fine_tune_config),
        "source_tree_sha256": source_tree_sha256(),
        "randomized_tree_sha256": randomized_tree_sha256(),
        "decoder_control_tree_sha256": decoder_control_tree_sha256(),
        "history_sha256": sha256_file(history_path),
        "protocol_registration": str(registration_path.resolve()),
        "protocol_registration_sha256": sha256_file(registration_path),
        "prospective_registration": registration,
        "parameter_audit": audit,
        "forbidden_training_inputs": _forbidden_training_inputs(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return model, history, checkpoint
