"""Miconi & Kay source reproduction and task-transfer experiments.

The source implementation is deliberately isolated from the Liu task.  Code
under :mod:`fsrl.miconi_kay.source` must not import Liu-specific modules.
"""

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN

__all__ = ["MiconiKaySourceConfig", "MiconiKayRetroModulRNN"]
