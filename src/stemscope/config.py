import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Local storage and execution settings."""

    output_dir: Path
    device: str = "cpu"
    max_bytes: int = 200 * 1024 * 1024
    max_seconds: int = 1200
    pilot_dir: Path = Path("data/musdb7-pilot")

    @classmethod
    def from_env(cls) -> "Settings":
        config_path = os.environ.get("STEMSCOPE_CONFIG")
        values = {}
        if config_path:
            values = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if not isinstance(values, dict) or set(values) - {
                "output_dir",
                "device",
                "max_bytes",
                "max_seconds",
                "pilot_dir",
            }:
                raise ValueError("Configuration must be an object with known Settings keys.")
        for key in ("output_dir", "device", "max_bytes", "max_seconds", "pilot_dir"):
            override = os.environ.get("STEMSCOPE_" + key.upper())
            if override is not None:
                values[key] = override
        for key in ("max_bytes", "max_seconds"):
            if key in values:
                values[key] = int(values[key])
                if values[key] <= 0:
                    raise ValueError(f"{key} must be positive.")
        values["output_dir"] = Path(values.get("output_dir", "outputs")).resolve()
        values["pilot_dir"] = Path(values.get("pilot_dir", "data/musdb7-pilot")).resolve()
        return cls(**values)
