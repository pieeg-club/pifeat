<div align="center">

<img width="auto" height="320" alt="pifeat logo" src="https://github.com/user-attachments/assets/da4a95cf-b85b-4382-a949-c1733430cd2f" />

# `pifeat`

**Zero-Dependency, Sub-Millisecond EEG & Biosignal Feature Extraction Engine in Rust + Python**

[![CI](https://github.com/pieeg-club/pifeat/actions/workflows/ci.yml/badge.svg)](https://github.com/pieeg-club/pifeat/actions/workflows/ci.yml)
[![Release](https://github.com/pieeg-club/pifeat/actions/workflows/release.yml/badge.svg)](https://github.com/pieeg-club/pifeat/actions/workflows/release.yml)
[![PyPI](https://img.shields.io/pypi/v/pifeat.svg?color=blue)](https://pypi.org/project/pifeat/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pifeat.svg)](https://pypi.org/project/pifeat/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](#why-pifeat)
[![Powered by Rust](https://img.shields.io/badge/engine-Rust%20%2B%20PyO3-orange.svg)](https://github.com/pieeg-club/pifeat)
[![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Raspberry%20Pi-informational.svg)](#release-and-wheel-distribution)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3%2B-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

```bash
pip install pifeat
```

[Why `pifeat`?](#-why-pifeat) • [Quickstart](#-developer-quickstart) • [22 Canonical Features](#-the-22-features-per-channel) • [Benchmarks](#-benchmarks-100x170x-faster) • [API Reference](#-api-reference)

</div>

---

> **Early development / experimental / no warranties.** APIs, behavior, and packaging may change without notice. Research software provided as-is, without warranty of any kind. Not a medical device and not intended for clinical, diagnostic, or safety-critical use. See `LICENSE` for full terms.

## ⚡ Why `pifeat`?

Building real-time Brain-Computer Interfaces (BCI), neurofeedback loops, or edge ML pipelines on devices like a **Raspberry Pi** or embedded Linux usually forces a painful trade-off: pull in hundreds of megabytes of `scipy` / `mne` / `numba` / `openblas` dependencies—or write fragile DSP glue code by hand.

**`pifeat` solves this in a single `~1.1 MB` native wheel:**

- **Zero Runtime Dependencies**: No `numpy`, `scipy`, `mne`, `numba`, or C/Fortran BLAS required at runtime. Works out of the box with Python's built-in `array.array`, `memoryview`, `bytearray`, **or** `numpy.ndarray` via the **CPython Buffer Protocol**.
- **100x–173x Faster than SciPy + NumPy**: Extracts all 22 time, frequency (Welch PSD), and nonlinear features across 16 channels (500 samples @ 250 Hz) in **~146 µs** with the GIL released.
- **Zero Steady-State Allocations**: Both the Rust core and `extractor.extract_into(...)` perform **0 heap allocations** on the real-time hot path—eliminating GC pauses and latency spikes.
- **Drop-In Scientific Equivalence**: Matches `scipy.signal.welch` (`window='hann'`, `detrend='constant'`, `scaling='density'`) to **$< 10^{-12}$ relative error**, with compensated two-pass variance to survive large uncalibrated ADC DC offsets ($10^5\text{--}10^6\ \mu\text{V}$).
- **Hardware-Agnostic**: Works with **OpenBCI, BrainFlow, LSL (Lab Streaming Layer), MNE, PiEEG**, or any raw ADC/SPI stream.

---

## 🚀 Developer Quickstart

### 30-Second Copy-Paste Example (Pure Python — Zero Dependencies)

```python
import array, math
import pifeat as pf

# 1. Generate 2 seconds (500 samples @ 250 Hz) of synthetic 8-channel EEG (10 Hz Alpha + 20 Hz Beta)
n_ch, n_samples, fs = 8, 500, 250.0
raw = array.array("d", [
    15.0 * math.sin(2 * math.pi * 10.0 * (s / fs)) + 5.0 * math.cos(2 * math.pi * 20.0 * (s / fs))
    for _ in range(n_ch) for s in range(n_samples)
])

# 2. Extract all 22 features per channel in ~70 microseconds
extractor = pf.FeatureExtractor(num_channels=n_ch, window_samples=n_samples, fs=fs)
frame = extractor.extract(raw, layout="channels_first")

print("Features per channel:", len(pf.FEATURE_NAMES))        # 22
print("Ch0 Alpha Power (uV^2):", frame.to_dict()["alpha_power"][0])
print("Ch0 Alpha/Beta Ratio:  ", frame.to_dict()["alpha_beta_ratio"][0])
```

### 1. Batch Multichannel Extraction (`numpy.ndarray`, `array.array`, or `memoryview`)

Accepts contiguous `float64` (`'d'`) or `float32` (`'f'`) buffers in either `channels_first` (`[channels, samples]`) or `samples_first` (`[samples, channels]`) layout:

```python
import array
import pifeat as pf

extractor = pf.FeatureExtractor(
    num_channels=16,
    window_samples=500,     # 2.0 s window at 250 Hz
    fs=250.0,
    nperseg=250,            # 1.0 s Welch segment (1.0 Hz bin spacing)
    noverlap=125,           # 50% overlap (3 Welch segments per window)
    pe_order=3,             # Permutation entropy embedding dimension
    pe_delay=1,             # Permutation entropy sample delay
    integration="trapezoid" # "trapezoid" (uV^2) or "bin_mean" (uV^2/Hz)
)

# Pass a 2D numpy.ndarray (shape [16, 500]), memoryview, or stdlib array.array('d'|'f')
raw_window = array.array("d", [0.0] * (16 * 500))

frame = extractor.extract(raw_window, layout="channels_first")
if frame.window_valid:
    matrix = frame.matrix()          # list[list[float]], shape [16][22]
    ch0 = frame.channel(0)           # list[float], length 22
    by_name = frame.to_dict()        # dict[str, list[float]]

# Zero-allocation hot path: writes directly into a preallocated writable f64 buffer (array or np.ndarray)
out_buf = array.array("d", [0.0] * (16 * pf.NUM_FEATURES_PER_CHANNEL))
valid = extractor.extract_into(raw_window, out_buf, layout="channels_first")
```

### 2. Continuous Real-Time Streaming (`StreamingFeatureEngine`)

`StreamingFeatureEngine` maintains per-channel circular buffers, tracks monotonic sequence numbers and timestamps, detects dropped upstream packets, and **caches overlapping Welch segment periodograms** so each hop only computes the newly completed segment FFT:

```python
import pifeat as pf

stream = pf.StreamingFeatureEngine(
    num_channels=16,
    window_samples=500,     # 2.0 s sliding window
    hop_samples=125,        # Emit a FeatureFrame every 125 samples (500 ms at 250 Hz)
    fs=250.0,
    nperseg=250,
    noverlap=125,
    pe_order=3,
    pe_delay=1,
    integration="trapezoid",
)

# Push sample-by-sample from your hardware loop / LSL / BrainFlow / SPI reader
for seq, timestamp, channels in acquisition_stream:
    frame = stream.push_sample(seq, timestamp, channels)
    if frame is not None:
        # Inspect packet-loss & signal-quality flags before feeding your ML model
        if not frame.window_valid:
            print(f"Warning: {frame.missing_samples} dropped samples or invalid channel (flags={frame.channel_flags})")
        features = frame.matrix()

# Or push a contiguous chunk of samples at once ([samples, channels] or [channels, samples])
frames = stream.push_block(block_buf, start_seq=1, start_timestamp=0.0, layout="samples_first")
```

### 3. Standalone 1D DSP Functions

Need just one metric on a 1D signal (`list[float]`, `array.array`, or 1D `numpy.ndarray`)?

```python
import pifeat as pf

rms = pf.compute_rms(signal_1d)
var = pf.compute_variance(signal_1d)
ll  = pf.compute_line_length(signal_1d)
zcr = pf.compute_zero_crossing_rate(signal_1d)
activity, mobility, complexity = pf.compute_hjorth(signal_1d)
pe  = pf.compute_permutation_entropy(signal_1d, order=3, delay=1, normalize=True)

# Exact scipy.signal.welch equivalent (returns frequencies and one-sided PSD in uV^2/Hz)
freqs, psd = pf.compute_welch_psd(signal_1d, fs=250.0, nperseg=250, noverlap=125)

# Dict of absolute powers, relative powers, ratios, and spectral entropy
bands = pf.compute_band_powers(signal_1d, fs=250.0, nperseg=250, noverlap=125)
```

### 4. Using `pifeat` in Pure Rust (`Cargo.toml`)

`pifeat` can also be embedded directly in Rust firmware, Tauri apps, or native daemons without Python:

```toml
[dependencies]
pifeat = { version = "0.1", default-features = false }
```

---

## 🧠 The 22 Features Per Channel

Each channel produces `NUM_FEATURES_PER_CHANNEL = 22` features in the canonical order given by `pf.FEATURE_NAMES`:

| Index | Feature Name | Domain | Units / Range | Definition |
|---:|---|---|---|---|
| 0 | `rms` | Time | $\mu\text{V}$ | $\sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x_n^2}$ |
| 1 | `variance` | Time | $\mu\text{V}^2$ | Sample variance (`ddof = 1`), two-pass compensated: $\frac{1}{N-1}\sum (x_n - \mu)^2$ |
| 2 | `line_length` | Time | $\mu\text{V}$ | $\sum_{n=1}^{N-1} \|x_n - x_{n-1}\|$ (Esteller et al., 2001) |
| 3 | `zero_crossing_rate` | Time | $[0, 1]$ | Mean-centered sign-change rate across $N - 1$ transitions |
| 4 | `hjorth_activity` | Time | $\mu\text{V}^2$ | $\text{var}(x)$ (`ddof = 1`, Hjorth, 1970) |
| 5 | `hjorth_mobility` | Time | rad/sample | $\sqrt{\text{var}(d_1) / \text{var}(x)}$, where $d_1[n] = x_n - x_{n-1}$ |
| 6 | `hjorth_complexity` | Time | dimensionless | $\frac{\text{Mobility}(d_1)}{\text{Mobility}(x)} = \sqrt{\frac{\text{var}(d_2)/\text{var}(d_1)}{\text{var}(d_1)/\text{var}(x)}}$ |
| 7 | `delta_power` | Frequency | $\mu\text{V}^2$ | Integrated Welch PSD over $[0.5, 4.0]\text{ Hz}$ |
| 8 | `theta_power` | Frequency | $\mu\text{V}^2$ | Integrated Welch PSD over $[4.0, 8.0]\text{ Hz}$ |
| 9 | `alpha_power` | Frequency | $\mu\text{V}^2$ | Integrated Welch PSD over $[8.0, 13.0]\text{ Hz}$ |
| 10 | `beta_power` | Frequency | $\mu\text{V}^2$ | Integrated Welch PSD over $[13.0, 30.0]\text{ Hz}$ |
| 11 | `gamma_power` | Frequency | $\mu\text{V}^2$ | Integrated Welch PSD over $[30.0, \min(100.0, f_s/2)]\text{ Hz}$ |
| 12 | `total_power` | Frequency | $\mu\text{V}^2$ | Sum of absolute powers across Delta, Theta, Alpha, Beta, and Gamma |
| 13..17 | `rel_delta` .. `rel_gamma` | Frequency | $[0, 1]$ | Band power divided by `total_power` |
| 18 | `theta_beta_ratio` | Frequency | dimensionless | `theta_power / beta_power` (guarded against noise floor) |
| 19 | `alpha_beta_ratio` | Frequency | dimensionless | `alpha_power / beta_power` (guarded against noise floor) |
| 20 | `spectral_entropy` | Frequency | $[0, 1]$ | Normalized Shannon entropy of AC Welch PSD bins ($k \ge 1$) |
| 21 | `permutation_entropy` | Nonlinear | $[0, 1]$ | Normalized ordinal-pattern Shannon entropy (Bandt & Pompe, 2002) |

### Numerical and Scientific Guarantees

- **Welch PSD Equivalence (`scipy.signal.welch`)**: Uses a periodic Hann window $w[n] = 0.5 - 0.5\cos(2\pi n / L)$ (`fftbins=True`), per-segment constant mean detrending (`detrend='constant'`), and one-sided power spectral density scaling $U = f_s \sum w[n]^2$ (`scaling='density'`). Matches `scipy.signal.welch` to $< 10^{-12}$ relative error.
- **Band Integration Modes**:
  - `"trapezoid"` (default): Composite trapezoidal integration with linear interpolation at exact band boundaries ($[f_{\text{lo}}, f_{\text{hi}}]$), preserving Parseval's theorem in $\mu\text{V}^2$.
  - `"bin_mean"`: Arithmetic mean of PSD bins in $[f_{\text{lo}}, f_{\text{hi}})$ in $\mu\text{V}^2/\text{Hz}$ (matches legacy `pieeg_server.spectral`).
- **Compensated Variance and Single-Pass Hjorth**: Two-pass compensated summation prevents catastrophic cancellation when large uncalibrated ADC DC offsets ($10^5\text{--}10^6\ \mu\text{V}$) sit atop microvolt EEG signals. Hjorth first- and second-difference means are evaluated in closed form from boundary samples, requiring zero intermediate allocations.
- **Band Ratio Noise-Floor Guard**: Denominator band power is guarded by `max(1e-12, total_power * 1e-9)` so band-limited signals do not produce arbitrary $10^{16}\text{--}10^{29}$ ratios from floating-point FFT floor differences between `f32` and `f64`.
- **Degenerate Signals and Validity Flags**:
  - Flatlines ($\text{var}(x) \le 10^{-24}$) never produce `NaN` or `Inf`; Mobility, Complexity, Spectral Entropy, and Permutation Entropy return `0.0` and set `FLAG_FLATLINE` (`1 << 1`) in `frame.channel_flags`.
  - Non-finite inputs (`NaN` / `Inf`) set `FLAG_NON_FINITE` (`1 << 0`) on the affected channel and `frame.window_valid = False`.
  - Sequence number gaps in streaming mode set `FLAG_SEQ_GAP` (`1 << 2`), record the exact missing count in `frame.missing_samples`, and set `frame.window_valid = False`.

---

## 📖 API Reference

### `FeatureFrame`

| Member | Type | Description |
|---|---|---|
| `num_channels` | `int` | Number of channels in the frame |
| `window_samples` | `int` | Number of samples per channel in the window |
| `timestamp_start` | `float` | Timestamp of the oldest sample in the window (streaming) |
| `timestamp_end` | `float` | Timestamp of the newest sample in the window (streaming) |
| `seq_start` | `int` | Sequence index of the oldest sample in the window |
| `seq_end` | `int` | Sequence index of the newest sample in the window |
| `missing_samples` | `int` | Count of dropped samples detected inside the window |
| `window_valid` | `bool` | `True` when `missing_samples == 0` and all channels are finite |
| `channel_flags` | `list[int]` | Per-channel bitmask of `FLAG_VALID` (`0`), `FLAG_NON_FINITE` (`1`), `FLAG_FLATLINE` (`2`), `FLAG_SEQ_GAP` (`4`) |
| `matrix()` | `list[list[float]]` | `[num_channels][22]` feature matrix |
| `values_flat()` | `list[float]` | Flat row-major `[num_channels * 22]` feature list |
| `channel(ch)` | `list[float]` | 22-element feature vector for channel `ch` |
| `to_dict()` | `dict` | Maps each feature name to a `list[float]` across channels plus metadata |

---

## 📊 Benchmarks (100x–170x Faster)

Measured at $f_s = 250\text{ Hz}$ extracting all 22 features per channel (time-domain, Welch PSD, band powers, ratios, spectral entropy, and permutation entropy). Steady-state heap allocations in Rust are verified via a custom `#[global_allocator]` counter in `tests/bench_native.rs`.

### Native Rust Latency and Steady-State Allocations

| Channels | Window (`@ 250 Hz`) | Layout | Batch Median ($\mu\text{s}$) | Batch p95 ($\mu\text{s}$) | Batch p99 ($\mu\text{s}$) | Streaming Hop Median ($\mu\text{s}$) | Steady-State Heap Allocs |
|---:|---|---|---:|---:|---:|---:|---:|
| 8 | 250 (1.0 s) | `[C, N]` / `[N, C]` | 37.0 / 38.4 | 40.5 / 65.8 | 62.8 / 67.7 | 40.2 | 0 |
| 8 | 500 (2.0 s) | `[C, N]` / `[N, C]` | 71.8 / 75.9 | 114.9 / 122.0 | 122.7 / 127.7 | 63.5 | 0 |
| 16 | 250 (1.0 s) | `[C, N]` / `[N, C]` | 83.3 / 82.0 | 129.9 / 135.3 | 166.4 / 139.5 | 92.0 | 0 |
| 16 | 500 (2.0 s) | `[C, N]` / `[N, C]` | 146.2 / 154.2 | 155.3 / 177.7 | 189.1 / 258.1 | 125.7 | 0 |
| 16 | 1000 (4.0 s) | `[C, N]` / `[N, C]` | 261.7 / 277.1 | 382.2 / 291.5 | 430.5 / 329.8 | 188.2 | 0 |
| 24 | 250 (1.0 s) | `[C, N]` / `[N, C]` | 114.6 / 120.5 | 120.5 / 121.5 | 157.6 / 135.1 | 116.5 | 0 |
| 24 | 500 (2.0 s) | `[C, N]` / `[N, C]` | 219.2 / 231.9 | 340.0 / 346.5 | 368.8 / 405.8 | 192.9 | 0 |
| 24 | 1000 (4.0 s) | `[C, N]` / `[N, C]` | 401.4 / 426.5 | 429.0 / 636.2 | 590.7 / 678.1 | 279.9 | 0 |

### Python API Overhead vs. Reference SciPy / NumPy (`benches/bench_python_vs_native.py`)

| Channels | Window | `extract_into` (Med / p95 / p99 $\mu\text{s}$) | `extract()` (Med / p95 / p99 $\mu\text{s}$) | SciPy + NumPy Median ($\mu\text{s}$) | Speedup |
|---:|---|---:|---:|---:|---:|
| 8 | 250 (1.0 s) | 40.2 / 41.9 / 70.6 | 38.2 / 38.8 / 42.3 | 6,961.9 | 173x |
| 8 | 500 (2.0 s) | 82.6 / 84.0 / 133.7 | 80.0 / 130.0 / 136.6 | 9,066.6 | 110x |
| 16 | 250 (1.0 s) | 85.5 / 86.3 / 92.1 | 82.3 / 83.3 / 86.7 | 14,762.4 | 173x |
| 16 | 500 (2.0 s) | 168.8 / 169.6 / 193.9 | 166.6 / 169.5 / 212.8 | 18,328.4 | 109x |
| 24 | 500 (2.0 s) | 254.8 / 510.1 / 552.7 | 348.4 / 416.5 / 546.3 | 27,512.8 | 108x |
| 24 | 1000 (4.0 s) | 467.8 / 688.4 / 821.3 | 462.9 / 481.1 / 558.9 | 41,161.8 | 88x |

- **State footprint**: `< 96 KB` for a 16-channel, 500-sample `StreamingFeatureEngine`.
- **Binary size**: `~1.18 MB` stripped native extension with zero external shared library dependencies.

---

## Build and Test from Source

Requires Rust (`rustup`) and `maturin`:

```bash
pip install maturin
cd pifeat
maturin develop --release
```

Run the pure-Rust unit tests and zero-allocation benchmark suite:

```bash
cargo test --no-default-features --release
cargo test --no-default-features --release --test bench_native -- --nocapture
```

Run the Python scientific validation suite and benchmark (requires `pytest`, `numpy`, `scipy` for reference comparison only):

```bash
pytest tests/test_scientific_validation.py -v
python benches/bench_python_vs_native.py
```

## Release and Wheel Distribution

`.github/workflows/release.yml` builds precompiled wheels via `PyO3/maturin-action@v1` for:

- **Linux**: `x86_64`, `aarch64` (Raspberry Pi 4/5 64-bit), `armv7` (Raspberry Pi OS 32-bit)
- **macOS**: `x86_64`, `aarch64`
- **Windows**: `x64`
- **Python**: CPython 3.10, 3.11, 3.12, 3.13 + source distribution (`sdist`)

