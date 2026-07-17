"""Mechanism-faithful source-task implementation for Miconi & Kay.

This package follows the readable reproduction at VeriTas-arch/fsrl commit
``afd31c923c1d77bef56c1d551f062179b55c487f``.  It is kept separate from any
Liu adapters so source-task parity can be tested independently.
"""

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.episode import EpisodeStats, EpisodeTrace, run_episode
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN

__all__ = [
    "EpisodeStats",
    "EpisodeTrace",
    "MiconiKayRetroModulRNN",
    "MiconiKaySourceConfig",
    "run_episode",
]
