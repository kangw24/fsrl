"""Fixed-budget source-only fine-tuning for the linked-list curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from fsrl.miconi_kay.derived.linked_curriculum import (
    LinkedCurriculumConfig,
    run_linked_curriculum_episode,
)
from fsrl.miconi_kay.derived.protocol import (
    LINKED_SOURCE_TRAINING_CONFIGURATION,
    linked_source_evaluation_plan,
    linked_source_training_plan,
)
from fsrl.miconi_kay.source.checkpoint import (
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
class LinkedFineTuneConfig:
    episodes: int = 500
    learning_rate: float = 1e-5
    seed: int = 101
    print_every: int = 25

    def validate(self) -> None:
        if self.episodes <= 0:
            raise ValueError("fine-tuning episode count must be positive")
        if self.learning_rate <= 0:
            raise ValueError("fine-tuning learning rate must be positive")
        if self.seed < 0 or self.print_every <= 0:
            raise ValueError("seed must be non-negative and print interval positive")


@dataclass(frozen=True)
class LinkedFineTuneSummary:
    episode_index: int
    loss_value: float
    query_accuracy: float
    minimum_pair_accuracy: float
    mean_absolute_post_bridge_fast_weight: float
    gradient_norm_before_clip: float


def derived_tree_sha256() -> str:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py"), key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _forbidden_training_inputs() -> list[str]:
    return [
        "Liu graph",
        "Liu human choices",
        "Liu endpoint metrics",
        "participant identifiers",
        "teacher_sign",
    ]


def register_linked_finetune_protocol(
    output_dir: str | Path,
    *,
    base_checkpoint: Path,
    source_config: MiconiKaySourceConfig,
    curriculum_config: LinkedCurriculumConfig,
    fine_tune_config: LinkedFineTuneConfig,
) -> Path:
    """Lock the training/evaluation plan before the first optimizer update."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty registered run directory: {directory}"
        )
    observed_training_configuration = {
        "episodes": fine_tune_config.episodes,
        "learning_rate": fine_tune_config.learning_rate,
        "seed": fine_tune_config.seed,
        "batch_size": source_config.batch_size,
        "component_trials": curriculum_config.component_trials,
        "bridge_trials": curriculum_config.bridge_trials,
        "query_loss_multiplier": curriculum_config.query_loss_multiplier,
    }
    registration = {
        "registration_kind": "prospective_local_code_registration",
        "limitation": "not externally timestamped or independently registered",
        "registration_moment": "written before optimizer construction and first step",
        "claim_ceiling": (
            "M&K-derived source mechanism candidate; not source parity, Liu "
            "transfer, or evidence of a human mechanism"
        ),
        "selection_rule": "fixed final episode; no early stopping or Liu metric",
        "training_objective_audit": {
            "outer_objective": "sampled-action REINFORCE actor-critic",
            "online_feedback": "chosen-action-contingent signed reward only",
            "direct_correct_action_cross_entropy": False,
            "counterfactual_unchosen_action_reward_used": False,
            "all_sixteen_cross_pairs_equal_weight": True,
        },
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "source_config": source_config.to_manifest_dict(),
        "curriculum_config": asdict(curriculum_config),
        "fine_tune_config": asdict(fine_tune_config),
        "registered_training_plan": linked_source_training_plan(),
        "observed_training_configuration": observed_training_configuration,
        "is_formal_training_configuration": (
            observed_training_configuration
            == LINKED_SOURCE_TRAINING_CONFIGURATION
        ),
        "evaluation_plan": linked_source_evaluation_plan(),
        "source_tree_sha256": source_tree_sha256(),
        "derived_tree_sha256": derived_tree_sha256(),
        "forbidden_training_inputs": _forbidden_training_inputs(),
    }
    path = directory / "protocol_registration.json"
    path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def save_linked_finetune_checkpoint(
    model: MiconiKayRetroModulRNN,
    output_dir: str | Path,
    *,
    base_checkpoint: Path,
    source_config: MiconiKaySourceConfig,
    curriculum_config: LinkedCurriculumConfig,
    fine_tune_config: LinkedFineTuneConfig,
    history: list[LinkedFineTuneSummary],
    registration_path: Path,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    validate_source_state_dict(state)
    checkpoint = directory / "net.dat"
    torch.save(state, checkpoint)
    history_path = directory / "training_history.json"
    history_path.write_text(
        json.dumps([asdict(row) for row in history], indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "claim_status": (
            "M&K-derived source-only linked-list fine-tune; not source parity, "
            "not Liu transfer, and not a human-mechanism result"
        ),
        "selection_rule": "fixed final episode; no early stopping or Liu metric",
        "protocol_registration": str(registration_path.resolve()),
        "protocol_registration_sha256": sha256_file(registration_path),
        "prospective_registration": json.loads(
            registration_path.read_text(encoding="utf-8")
        ),
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_config": source_config.to_manifest_dict(),
        "curriculum_config": asdict(curriculum_config),
        "fine_tune_config": asdict(fine_tune_config),
        "source_tree_sha256": source_tree_sha256(),
        "derived_tree_sha256": derived_tree_sha256(),
        "history_sha256": sha256_file(history_path),
        "forbidden_training_inputs": _forbidden_training_inputs(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def train_linked_finetune(
    *,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    source_config: MiconiKaySourceConfig,
    curriculum_config: LinkedCurriculumConfig,
    fine_tune_config: LinkedFineTuneConfig,
    device: torch.device | str | None = None,
    require_public_active: bool = True,
) -> tuple[MiconiKayRetroModulRNN, list[LinkedFineTuneSummary], Path]:
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
    # Model construction consumes Torch RNG even though its weights are then
    # replaced.  Reset here so training trajectories depend only on the stated
    # fine-tuning seed.
    set_source_seed(fine_tune_config.seed)
    registration_path = register_linked_finetune_protocol(
        output_dir,
        base_checkpoint=checkpoint_path,
        source_config=source_config,
        curriculum_config=curriculum_config,
        fine_tune_config=fine_tune_config,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=fine_tune_config.learning_rate,
        eps=source_config.adam_epsilon,
        weight_decay=source_config.weight_decay,
    )
    history: list[LinkedFineTuneSummary] = []
    for episode_index in range(fine_tune_config.episodes):
        optimizer.zero_grad(set_to_none=True)
        stats = run_linked_curriculum_episode(
            source_config, curriculum_config, model
        )
        stats.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), source_config.gradient_clip
        )
        optimizer.step()
        summary = LinkedFineTuneSummary(
            episode_index=episode_index,
            loss_value=stats.loss_value,
            query_accuracy=stats.query_accuracy,
            minimum_pair_accuracy=stats.minimum_pair_accuracy,
            mean_absolute_post_bridge_fast_weight=(
                stats.mean_absolute_post_bridge_fast_weight
            ),
            gradient_norm_before_clip=float(gradient_norm.detach().cpu()),
        )
        history.append(summary)
        if (
            episode_index == 0
            or episode_index == fine_tune_config.episodes - 1
            or (episode_index + 1) % fine_tune_config.print_every == 0
        ):
            print(
                "linked-source-v2 "
                f"episode={episode_index + 1}/{fine_tune_config.episodes} "
                f"loss={summary.loss_value:.6f} "
                f"query_acc={summary.query_accuracy:.4f} "
                f"min_pair={summary.minimum_pair_accuracy:.4f}"
            )

    output_checkpoint = save_linked_finetune_checkpoint(
        model,
        output_dir,
        base_checkpoint=checkpoint_path,
        source_config=source_config,
        curriculum_config=curriculum_config,
        fine_tune_config=fine_tune_config,
        history=history,
        registration_path=registration_path,
    )
    return model, history, output_checkpoint
