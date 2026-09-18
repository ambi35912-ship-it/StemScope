"""Open-Unmix integration with reusable weights and bounded inference excerpts."""

from contextlib import ExitStack
from pathlib import Path
from threading import Lock

import numpy as np
import soundfile as sf

from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator


class OpenUnmixSeparator(Separator):
    """Run umxhq in 15-second cores with one second of surrounding context.

    Crop each contextual estimate to its core and stream it into the output WAVs.
    This bounds inference tensors while preserving exact duration and stereo gain.
    Chunking limits the recurrent model's context; outputs may differ from full-track inference.
    """

    name = "Open-Unmix (umxhq)"

    def __init__(self, device: str = "cpu", chunk_seconds: int = 15) -> None:
        if chunk_seconds < 1:
            raise ValueError("chunk_seconds must be at least 1")
        self.device = device
        self.chunk_seconds = chunk_seconds
        self._model = None
        self._lock = Lock()

    @property
    def configuration(self) -> dict:
        return {
            "checkpoint": "umxhq",
            "niter": 1,
            "chunk_seconds": self.chunk_seconds,
            "context_seconds": 1,
            "device": self.device,
        }

    def separate(self, samples: np.ndarray, sample_rate: int, output_dir: Path) -> dict[str, Path]:
        try:
            import torch
            from openunmix.utils import load_separator
            from torchaudio.functional import resample
        except (ImportError, OSError) as exc:
            raise StemScopeError(
                "Open-Unmix dependencies are unavailable. Install the documented environment."
            ) from exc
        try:
            with self._lock, torch.inference_mode(), ExitStack() as stack:
                if self._model is None:
                    model = load_separator(
                        model_str_or_path="umxhq",
                        targets=list(STEMS),
                        device=self.device,
                        niter=1,
                        residual=False,
                        wiener_win_len=300,
                        pretrained=True,
                        filterbank="torch",
                    )
                    model.eval()
                    self._model = model
                model = self._model
                rate = int(model.sample_rate)
                audio = torch.from_numpy(samples.T.copy())
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)
                if sample_rate != rate:
                    audio = resample(audio, sample_rate, rate)
                length = audio.shape[-1]
                core_length = self.chunk_seconds * rate
                context = rate
                paths = {name: output_dir / f"{name}.wav" for name in STEMS}
                writers = {
                    name: stack.enter_context(
                        sf.SoundFile(path, mode="w", samplerate=rate, channels=2, subtype="FLOAT")
                    )
                    for name, path in paths.items()
                }
                for start in range(0, length, core_length):
                    end = min(start + core_length, length)
                    left, right = max(0, start - context), min(length, end + context)
                    excerpt = audio[:, left:right]
                    # Reflection padding in the model STFT needs more than 2,048 samples.
                    if excerpt.shape[-1] < 4096:
                        excerpt = torch.nn.functional.pad(excerpt, (0, 4096 - excerpt.shape[-1]))
                    estimates = model.to_dict(model(excerpt[None].to(self.device)))
                    if set(estimates) != set(STEMS):
                        raise StemScopeError("Open-Unmix did not produce all four expected stems.")
                    for name, estimate in estimates.items():
                        if estimate.ndim != 3 or estimate.shape[:2] != (1, 2):
                            raise StemScopeError("Open-Unmix returned an invalid stem shape.")
                        core = estimate[0, :, start - left : end - left].cpu()
                        if core.shape[-1] != end - start or not torch.isfinite(core).all():
                            raise StemScopeError("Open-Unmix produced incomplete or invalid audio.")
                        writers[name].write(core.T.numpy())
                return paths
        except StemScopeError:
            raise
        except Exception as exc:
            raise StemScopeError(
                "Open-Unmix failed. Check model download access, device, memory, and logs."
            ) from exc
