"""Explicit M&K-derived variants that are not source-reproduction code."""

from fsrl.miconi_kay.derived.linked_curriculum import (
    LinkedCurriculumConfig,
    LinkedCurriculumStats,
    run_linked_curriculum_episode,
)

__all__ = [
    "LinkedCurriculumConfig",
    "LinkedCurriculumStats",
    "run_linked_curriculum_episode",
]
