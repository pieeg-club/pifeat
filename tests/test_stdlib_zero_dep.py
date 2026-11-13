"""
Zero-dependency pure-stdlib validation suite for pifeat.

Designed to run in bare Python environments (e.g., Raspberry Pi / arm32v7 Docker
containers) WITHOUT numpy, scipy, or pytest installed, while also being discoverable
by pytest.

Run standalone:
    python tests/test_stdlib_zero_dep.py
"""

from __future__ import annotations

import array
import math
import pifeat as pf


def test_stdlib_feature_extractor_and_buffer_protocol() -> None:
    n_ch = 8
    n_samples = 500
    fs = 250.0

    raw_f64 = array.array("d", [0.0] * (n_ch * n_samples))
    raw_f32 = array.array("f", [0.0] * (n_ch * n_samples))
    for ch in range(n_ch):
        freq = 8.0 + ch * 2.0
        for s in range(n_samples):
            t = s / fs
            val = 12.0 * math.sin(2.0 * math.pi * freq * t) + 4.0 * math.cos(2.0 * math.pi * 20.0 * t)
            raw_f64[ch * n_samples + s] = val
            raw_f32[ch * n_samples + s] = val

    ext = pf.FeatureExtractor(num_channels=n_ch, window_samples=n_samples, fs=fs)
    frame64 = ext.extract(memoryview(raw_f64), layout="channels_first")
    frame32 = ext.extract(raw_f32, layout="channels_first")

    assert frame64.num_channels == n_ch
    assert frame64.window_samples == n_samples
    assert frame64.window_valid is True
    assert frame64.channel_flags == [pf.FLAG_VALID] * n_ch

    m64 = frame64.matrix()
    m32 = frame32.matrix()
    assert len(m64) == n_ch
    assert len(m64[0]) == pf.NUM_FEATURES_PER_CHANNEL == 22

    for ch in range(n_ch):
        for f_idx in range(pf.NUM_FEATURES_PER_CHANNEL):
            assert math.isfinite(m64[ch][f_idx])
            assert math.isclose(m64[ch][f_idx], m32[ch][f_idx], rel_tol=1e-4, abs_tol=1e-4)

    # Zero-allocation extract_into with stdlib array('d')
    out_buf = array.array("d", [0.0] * (n_ch * pf.NUM_FEATURES_PER_CHANNEL))
    valid = ext.extract_into(raw_f64, out_buf, layout="channels_first")
    assert valid is True
    flat = frame64.values_flat()
    for a, b in zip(out_buf, flat):
        assert math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14)


def test_stdlib_standalone_dsp_functions() -> None:
    fs = 250.0
    n = 500
    # 10 Hz Alpha (A=10 -> power=50) + 20 Hz Beta (A=5 -> power=12.5)
    sig = array.array(
        "d",
        [
            10.0 * math.sin(2.0 * math.pi * 10.0 * (i / fs))
            + 5.0 * math.sin(2.0 * math.pi * 20.0 * (i / fs))
            for i in range(n)
        ],
    )

    rms = pf.compute_rms(sig)
    var = pf.compute_variance(sig)
    ll = pf.compute_line_length(sig)
    zcr = pf.compute_zero_crossing_rate(sig)
    act, mob, comp = pf.compute_hjorth(sig)
    pe = pf.compute_permutation_entropy(sig, order=3, delay=1, normalize=True)
    freqs, psd = pf.compute_welch_psd(sig, fs=fs, nperseg=500, noverlap=250)
    bp = pf.compute_band_powers(sig, fs=fs, nperseg=500, noverlap=250, integration="trapezoid")

    assert math.isclose(rms, math.sqrt(50.0 + 12.5), rel_tol=1e-3)
    assert math.isclose(var, act, rel_tol=1e-12)
    assert ll > 0.0
    assert 0.0 < zcr < 1.0
    assert mob > 0.0 and comp > 0.0
    assert 0.0 < pe <= 1.0
    assert len(freqs) == len(psd) == 251
    assert math.isclose(bp["alpha_power"], 50.0, rel_tol=1e-3)
    assert math.isclose(bp["beta_power"], 12.5, rel_tol=1e-3)
    assert math.isclose(bp["alpha_beta_ratio"], 4.0, rel_tol=2e-3)


def test_stdlib_streaming_engine_and_gap_detection() -> None:
    n_ch = 4
    n_win = 250
    hop = 125
    fs = 250.0
    eng = pf.StreamingFeatureEngine(
        num_channels=n_ch,
        window_samples=n_win,
        hop_samples=hop,
        fs=fs,
        nperseg=125,
        noverlap=62,
    )

    emitted = 0
    for seq in range(1, n_win + 1):
        t = seq / fs
        sample = [10.0 * math.sin(2.0 * math.pi * (8.0 + c) * t) for c in range(n_ch)]
        frame = eng.push_sample(seq, t, sample)
        if frame is not None:
            emitted += 1
            assert frame.window_valid is True
            assert frame.missing_samples == 0

    assert emitted == 1
    assert eng.is_ready is True

    # Introduce a sequence gap (skip seq = 251)
    for seq in range(252, 252 + hop):
        t = seq / fs
        sample = [10.0 * math.sin(2.0 * math.pi * (8.0 + c) * t) for c in range(n_ch)]
        frame = eng.push_sample(seq, t, sample)
        if frame is not None:
            assert frame.window_valid is False
            assert frame.missing_samples == 1
            assert all((flag & pf.FLAG_SEQ_GAP) != 0 for flag in frame.channel_flags)


if __name__ == "__main__":
    test_stdlib_feature_extractor_and_buffer_protocol()
    test_stdlib_standalone_dsp_functions()
    test_stdlib_streaming_engine_and_gap_detection()
    print("OK: All zero-dependency stdlib validation tests passed (version=" + pf.__version__ + ")")
