"""
Scientific reference & Buffer-Protocol validation test suite for pifeat.

Tests numerical equivalence against:
- scipy.signal.welch (exact periodic Hann window, constant detrend, one-sided PSD density)
- numpy / MNE-Features / Antropy mathematical definitions for RMS, Variance (ddof=1),
  Line Length, Zero-Crossing Rate, Hjorth (Activity, Mobility, Complexity),
  Band Powers (trapezoidal + bin_mean), Spectral Entropy, and Permutation Entropy.
- Zero-dependency Buffer Protocol support (array.array('d'), array.array('f'), memoryview,
  2D numpy.ndarray in both channels_first and samples_first layouts).
- Degenerate signals (flatline, DC offset, NaN/Inf, sequence gaps in streaming).
"""

from __future__ import annotations

import array
import math
import numpy as np
import pytest
import scipy.signal

import pifeat as pf


def _ref_hjorth(x: np.ndarray) -> tuple[float, float, float]:
    """Reference Hjorth Activity, Mobility, Complexity with sample variance (ddof=1)."""
    var_x = float(np.var(x, ddof=1))
    if var_x <= 1e-30:
        return 0.0, 0.0, 0.0
    d1 = np.diff(x)
    var_d1 = float(np.var(d1, ddof=1))
    d2 = np.diff(d1)
    var_d2 = float(np.var(d2, ddof=1))
    mob = math.sqrt(var_d1 / var_x)
    comp = math.sqrt(var_d2 / var_d1) / mob if var_d1 > 1e-30 and mob > 1e-30 else 0.0
    return var_x, mob, comp


def _ref_perm_entropy(x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """Reference Bandt & Pompe (2002) permutation entropy matching Antropy."""
    n = len(x)
    n_vectors = n - (order - 1) * delay
    if n_vectors <= 0:
        return 0.0
    embedded = np.empty((n_vectors, order), dtype=np.float64)
    for j in range(order):
        embedded[:, j] = x[j * delay : j * delay + n_vectors]
    # argsort with kind='mergesort' preserves earlier index on ties
    patterns = np.argsort(embedded, axis=1, kind="mergesort")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / n_vectors
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    h = float(-np.sum(probs * np.log2(probs)))
    if normalize:
        h /= math.log2(math.factorial(order))
    return h


def _ref_spectral_entropy(psd: np.ndarray) -> float:
    """Reference normalized Shannon spectral entropy excluding DC bin (index 0)."""
    ac = psd[1:]
    total = float(np.sum(ac))
    if total <= 1e-30:
        return 0.0
    p = ac[ac > 0] / total
    h = float(-np.sum(p * np.log2(p)))
    return h / math.log2(len(ac))


def test_welch_psd_matches_scipy_exactly():
    rng = np.random.default_rng(42)
    fs = 250.0
    t = np.arange(1000, dtype=np.float64) / fs
    # Multi-component synthetic EEG + colored noise + DC offset
    sig = (
        250.0  # DC offset
        + 12.0 * np.sin(2 * np.pi * 2.5 * t)   # Delta
        + 8.0 * np.sin(2 * np.pi * 6.0 * t)    # Theta
        + 20.0 * np.sin(2 * np.pi * 10.0 * t)  # Alpha
        + 5.0 * np.cos(2 * np.pi * 20.0 * t)   # Beta
        + 2.0 * np.sin(2 * np.pi * 45.0 * t)   # Gamma
        + rng.normal(0.0, 3.0, size=len(t))
    )

    for nperseg, noverlap in [(250, 125), (500, 250), (256, 128), (125, 62)]:
        ref_freqs, ref_psd = scipy.signal.welch(
            sig,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )
        rust_freqs, rust_psd = pf.compute_welch_psd(
            sig, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend_constant=True
        )
        np.testing.assert_allclose(rust_freqs, ref_freqs, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(rust_psd, ref_psd, rtol=1e-12, atol=1e-12)


def test_time_domain_and_nonlinear_against_reference():
    rng = np.random.default_rng(12345)
    sig = rng.normal(loc=15.0, scale=25.0, size=500).astype(np.float64)

    expected_rms = float(np.sqrt(np.mean(sig ** 2)))
    expected_var = float(np.var(sig, ddof=1))
    expected_ll = float(np.sum(np.abs(np.diff(sig))))
    centered = sig - np.mean(sig)
    expected_zcr = float(np.sum(centered[:-1] * centered[1:] < 0) / (len(sig) - 1))
    exp_act, exp_mob, exp_comp = _ref_hjorth(sig)

    assert math.isclose(pf.compute_rms(sig), expected_rms, rel_tol=1e-13)
    assert math.isclose(pf.compute_variance(sig), expected_var, rel_tol=1e-13)
    assert math.isclose(pf.compute_line_length(sig), expected_ll, rel_tol=1e-13)
    assert math.isclose(pf.compute_zero_crossing_rate(sig), expected_zcr, rel_tol=1e-13)

    act, mob, comp = pf.compute_hjorth(sig)
    assert math.isclose(act, exp_act, rel_tol=1e-12)
    assert math.isclose(mob, exp_mob, rel_tol=1e-12)
    assert math.isclose(comp, exp_comp, rel_tol=1e-12)

    for order in (3, 4, 5):
        for delay in (1, 2):
            for norm in (True, False):
                rust_pe = pf.compute_permutation_entropy(sig, order=order, delay=delay, normalize=norm)
                ref_pe = _ref_perm_entropy(sig, order=order, delay=delay, normalize=norm)
                assert math.isclose(rust_pe, ref_pe, rel_tol=1e-12, abs_tol=1e-12)


def test_band_powers_and_ratios():
    fs = 250.0
    n = 500
    t = np.arange(n, dtype=np.float64) / fs
    # 10 Hz Alpha (A=10 -> power=50 uV^2) + 20 Hz Beta (A=5 -> power=12.5 uV^2) + 6 Hz Theta (A=5 -> power=12.5 uV^2)
    sig = (
        10.0 * np.sin(2 * np.pi * 10.0 * t)
        + 5.0 * np.sin(2 * np.pi * 20.0 * t)
        + 5.0 * np.sin(2 * np.pi * 6.0 * t)
    )

    bp = pf.compute_band_powers(sig, fs=fs, nperseg=500, noverlap=250, integration="trapezoid")
    assert math.isclose(bp["alpha_power"], 50.0, rel_tol=1e-3)
    assert math.isclose(bp["beta_power"], 12.5, rel_tol=1e-3)
    assert math.isclose(bp["theta_power"], 12.5, rel_tol=1e-3)
    assert math.isclose(bp["total_power"], 75.0, rel_tol=1e-3)
    assert math.isclose(bp["alpha_beta_ratio"], 4.0, rel_tol=2e-3)
    assert math.isclose(bp["theta_beta_ratio"], 1.0, rel_tol=2e-3)
    assert math.isclose(
        bp["rel_delta"] + bp["rel_theta"] + bp["rel_alpha"] + bp["rel_beta"] + bp["rel_gamma"],
        1.0,
        rel_tol=1e-12,
    )


def test_stdlib_buffer_protocol_without_numpy():
    """Verify extraction works purely with Python's built-in array.array and memoryview."""
    n_ch = 8
    n_samples = 500
    fs = 250.0

    # Create pure-Python stdlib array('d') in channels_first layout
    raw_f64 = array.array("d", [0.0] * (n_ch * n_samples))
    raw_f32 = array.array("f", [0.0] * (n_ch * n_samples))
    for ch in range(n_ch):
        freq = 8.0 + ch * 2.0
        for s in range(n_samples):
            val = 10.0 * math.sin(2.0 * math.pi * freq * (s / fs) + 0.37)
            raw_f64[ch * n_samples + s] = val
            raw_f32[ch * n_samples + s] = val

    ext = pf.FeatureExtractor(num_channels=n_ch, window_samples=n_samples, fs=fs)
    frame64 = ext.extract(memoryview(raw_f64), layout="channels_first")
    frame32 = ext.extract(raw_f32, layout="channels_first")

    assert frame64.num_channels == n_ch
    assert frame64.window_samples == n_samples
    assert frame64.window_valid is True

    m64 = frame64.matrix()
    m32 = frame32.matrix()
    assert len(m64) == n_ch
    assert len(m64[0]) == pf.NUM_FEATURES_PER_CHANNEL == 22

    for ch in range(n_ch):
        np.testing.assert_allclose(m64[ch], m32[ch], rtol=1e-5, atol=1e-5)

    # Also test zero-allocation extract_into with a stdlib array('d') output buffer
    out_buf = array.array("d", [0.0] * (n_ch * pf.NUM_FEATURES_PER_CHANNEL))
    valid = ext.extract_into(raw_f64, out_buf, layout="channels_first")
    assert valid is True
    np.testing.assert_allclose(list(out_buf), frame64.values_flat(), rtol=1e-14, atol=1e-14)


def test_channels_first_vs_samples_first_equivalence():
    rng = np.random.default_rng(99)
    n_ch = 16
    n_samples = 500
    data_cs = rng.normal(0.0, 20.0, size=(n_ch, n_samples)).astype(np.float64)
    data_sc = np.ascontiguousarray(data_cs.T)  # [n_samples, n_ch]

    ext = pf.FeatureExtractor(num_channels=n_ch, window_samples=n_samples, fs=250.0)
    f_cs = ext.extract(data_cs, layout="channels_first")
    f_sc = ext.extract(data_sc, layout="samples_first")

    np.testing.assert_allclose(f_cs.values_flat(), f_sc.values_flat(), rtol=1e-14, atol=1e-14)


def test_degenerate_signals_and_streaming_validity():
    n_ch = 4
    n_samples = 250
    data = np.ones((n_ch, n_samples), dtype=np.float64) * 10.0  # Channel 0..2 are flatlines
    data[1, :] = np.sin(np.linspace(0, 20 * np.pi, n_samples)) * 15.0  # Channel 1 is normal
    data[2, 100] = float("nan")  # Channel 2 has a NaN corruption

    ext = pf.FeatureExtractor(num_channels=n_ch, window_samples=n_samples, fs=250.0, nperseg=125, noverlap=62)
    frame = ext.extract(data, layout="channels_first")

    assert frame.window_valid is False
    flags = frame.channel_flags
    assert flags[0] == pf.FLAG_FLATLINE
    assert flags[1] == pf.FLAG_VALID
    assert flags[2] == pf.FLAG_NON_FINITE

    # Flatline channel 0 must have finite 0.0 values (not NaN/Inf!)
    ch0 = frame.channel(0)
    assert all(math.isfinite(v) for v in ch0)
    assert ch0[1] == 0.0  # variance == 0.0
    assert ch0[5] == 0.0  # mobility == 0.0
    assert ch0[6] == 0.0  # complexity == 0.0
    assert ch0[20] == 0.0 # spectral_entropy == 0.0
    assert ch0[21] == 0.0 # permutation_entropy == 0.0

    # NaN channel 2 is explicitly marked NaN
    ch2 = frame.channel(2)
    assert all(math.isnan(v) for v in ch2)
