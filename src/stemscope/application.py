"""Compose available services; UI and processing depend only on common contracts."""

from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.demucs import DemucsSeparator
from stemscope.separators.openunmix import OpenUnmixSeparator
from stemscope.service import SeparationService


def create_services(settings: Settings) -> dict[str, SeparationService]:
    """Create lightweight adapters; each loads weights only on first use."""
    adapters = (DemucsSeparator(settings.device), OpenUnmixSeparator(settings.device))
    return {adapter.name: SeparationService(adapter, settings) for adapter in adapters}


def choose_model(path, choice: str, settings: Settings) -> tuple[str, str]:
    """Resolve manual or experimental automatic selection before running one separator."""
    from stemscope.selection.inference import AUTO_CHOICES, recommend

    if choice in AUTO_CHOICES:
        if settings.device != "cpu":
            raise StemScopeError(
                "Experimental Auto modes were evaluated on CPU. Choose a model manually for another device."
            )
        return recommend(
            path,
            settings.output_dir / "phase6-selector" / "selector.json",
            choice.removeprefix("Auto: "),
        )
    return choice, ""
