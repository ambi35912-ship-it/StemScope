from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf

from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator


class DemucsSeparator(Separator):
    """Lazy, reusable htdemucs model; serialize access to its mutable device state."""

    name = "Demucs (htdemucs)"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self._lock = Lock()

    @property
    def configuration(self) -> dict:
        return {"checkpoint": "htdemucs", "shifts": 0, "overlap": 0.25, "device": self.device}

    def separate(self, samples: np.ndarray, sample_rate: int, output_dir: Path) -> dict[str, Path]:
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.audio import convert_audio
            from demucs.pretrained import get_model
        except (ImportError, OSError) as exc:
            raise StemScopeError(
                "Demucs dependencies are unavailable. Install the documented environment."
            ) from exc
        try:
            with self._lock, torch.inference_mode():
                if self._model is None:
                    self._model = get_model("htdemucs").eval()
                model = self._model
                if set(model.sources) != set(STEMS):
                    raise StemScopeError("The model does not provide the four required stems.")
                wav = convert_audio(
                    torch.from_numpy(samples.T.copy()),
                    sample_rate,
                    model.samplerate,
                    model.audio_channels,
                )
                reference = wav.mean(0)
                mean, scale = reference.mean(), reference.std()
                # Silence and anti-phase stereo must never cause division by zero.
                if not torch.isfinite(scale) or scale < 1e-8:
                    scale = wav.std().clamp_min(1e-8)
                separated = apply_model(
                    model,
                    ((wav - mean) / scale)[None],
                    device=self.device,
                    shifts=0,
                    split=True,
                    overlap=0.25,
                    progress=False,
                )[0]
                separated = (separated * scale + mean).cpu()
                paths = {}
                for name, audio in zip(model.sources, separated):
                    if not torch.isfinite(audio).all():
                        raise StemScopeError("Demucs produced invalid audio samples.")
                    path = output_dir / f"{name}.wav"
                    # Float WAV preserves relative stem gain without individual normalization.
                    sf.write(path, audio.T.numpy(), model.samplerate, subtype="FLOAT")
                    paths[name] = path
                return paths
        except StemScopeError:
            raise
        except Exception as exc:
            raise StemScopeError(
                "Demucs failed. Check model download access, device, memory, and logs."
            ) from exc
