"""
pifeat — Zero-dependency, native Rust EEG and biosignal feature extraction engine.

pip install pifeat   # zero Python runtime dependencies!

Accepts any object implementing the CPython Buffer Protocol (e.g. ``array.array('d')``,
``memoryview``, ``bytearray``, or ``numpy.ndarray``) in both ``channels_first``
(``[channels, samples]``) and ``samples_first`` (``[samples, channels]``) layouts.
"""

from .pifeat import (
    FLAG_FLATLINE,
    FLAG_NON_FINITE,
    FLAG_SEQ_GAP,
    FLAG_VALID,
    NUM_FEATURES_PER_CHANNEL,
    FeatureExtractor,
    FeatureFrame,
    StreamingFeatureEngine,
    __version__,
    compute_band_powers,
    compute_hjorth,
    compute_line_length,
    compute_permutation_entropy,
    compute_rms,
    compute_variance,
    compute_welch_psd,
    compute_zero_crossing_rate,
)

FEATURE_NAMES: list[str] = FeatureExtractor.feature_names()

__all__ = [
    "__version__",
    "NUM_FEATURES_PER_CHANNEL",
    "FEATURE_NAMES",
    "FLAG_VALID",
    "FLAG_NON_FINITE",
    "FLAG_FLATLINE",
    "FLAG_SEQ_GAP",
    "FeatureFrame",
    "FeatureExtractor",
    "StreamingFeatureEngine",
    "compute_rms",
    "compute_variance",
    "compute_line_length",
    "compute_zero_crossing_rate",
    "compute_hjorth",
    "compute_permutation_entropy",
    "compute_welch_psd",
    "compute_band_powers",
]
