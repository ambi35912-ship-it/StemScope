from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

STEMS = ("vocals", "drums", "bass", "other")


class Separator(ABC):
    """Write four named WAV stems into an existing job directory."""

    name: str = "Custom separator"

    @property
    def configuration(self) -> dict:
        """Parameters recorded in benchmark manifests."""
        return {}

    @abstractmethod
    def separate(self, samples: np.ndarray, sample_rate: int, output_dir: Path) -> dict[str, Path]:
        """Return stem names mapped to generated paths; raise on failure."""
        raise NotImplementedError
