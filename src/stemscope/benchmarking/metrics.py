"""Full-track, zero-mean SI-SDR with explicit undefined-result handling."""

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from stemscope.errors import StemScopeError


@dataclass(frozen=True)
class Score:
    db: float | None
    status: str


def align_estimate(
    estimate: np.ndarray, rate: int, reference: np.ndarray, reference_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the estimate; permit only one sample of rounding, never time shifts."""
    if rate != reference_rate:
        factor = gcd(rate, reference_rate)
        estimate = resample_poly(estimate, reference_rate // factor, rate // factor, axis=0)
    if reference.shape[1] == 1 and estimate.shape[1] == 2:
        reference = np.repeat(reference, 2, axis=1)
    if estimate.shape[1] != reference.shape[1]:
        raise StemScopeError("Reference and estimate channel layouts do not match.")
    if abs(len(estimate) - len(reference)) > 1:
        raise StemScopeError(
            "Reference and estimate durations do not match; no automatic alignment is applied."
        )
    length = min(len(estimate), len(reference))
    return estimate[:length], reference[:length]


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> Score:
    """One scale factor over concatenated channels, after per-channel mean removal.

    Return no numeric value for undefined cases or mathematical infinities. The
    explicit status prevents silence from receiving a fabricated high score.
    """
    if reference.shape != estimate.shape or reference.ndim != 2 or not reference.size:
        raise StemScopeError("SI-SDR requires matching nonempty frames-by-channels arrays.")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise StemScopeError("Cannot score non-finite audio samples.")
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    ref -= ref.mean(axis=0, keepdims=True)
    est -= est.mean(axis=0, keepdims=True)
    ref_energy = float(np.sum(ref * ref))
    est_energy = float(np.sum(est * est))
    if ref_energy / ref.size < 1e-20:
        return Score(None, "silent_reference")
    if est_energy / est.size < 1e-20:
        return Score(None, "silent_estimate")
    target = ref * (float(np.sum(est * ref)) / ref_energy)
    target_energy = float(np.sum(target * target))
    error_energy = float(np.sum((est - target) ** 2))
    if target_energy <= np.finfo(float).eps * est_energy:
        return Score(None, "zero_projection_negative_infinity")
    if error_energy <= np.finfo(float).eps * target_energy:
        return Score(None, "perfect_positive_infinity")
    return Score(float(10 * np.log10(target_energy / error_energy)), "ok")
