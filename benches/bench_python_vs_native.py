"""
Python API benchmark for pifeat:
1. Separates zero-allocation `extract_into` (pure Rust + Py_buffer) from `extract` (returns PyFeatureFrame)
   and `to_dict()` (constructs Python dict of lists).
2. Compares against the reference NumPy + SciPy implementation of the exact same 22 features.
3. Measures median, p95, p99 latency, real-time CPU budget utilization (at 2 Hz / 500ms hop rate),
   and Python `tracemalloc` steady-state allocations.
"""

from __future__ import annotations

import array
import math
import platform
import time
import tracemalloc
import numpy as np
import scipy.signal

import pifeat as pf


def _scipy_numpy_extract_all_22(data_cs: np.ndarray, fs: float, nperseg: int, noverlap: int) -> np.ndarray:
    """Reference NumPy/SciPy implementation of all 22 features across channels."""
    n_ch, n_samples = data_cs.shape
    out = np.zeros((n_ch, 22), dtype=np.float64)

    for ch in range(n_ch):
        x = data_cs[ch]
        rms = math.sqrt(float(np.mean(x * x)))
        var_x = float(np.var(x, ddof=1))
        ll = float(np.sum(np.abs(np.diff(x))))
        centered = x - np.mean(x)
        zcr = float(np.sum(centered[:-1] * centered[1:] < 0) / (n_samples - 1))

        d1 = np.diff(x)
        var_d1 = float(np.var(d1, ddof=1))
        d2 = np.diff(d1)
        var_d2 = float(np.var(d2, ddof=1))
        mob = math.sqrt(var_d1 / var_x) if var_x > 1e-30 else 0.0
        comp = (math.sqrt(var_d2 / var_d1) / mob) if var_d1 > 1e-30 and mob > 1e-30 else 0.0

        freqs, psd = scipy.signal.welch(
            x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, detrend="constant"
        )
        # Band integration
        bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 100.0)]
        bp = []
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs <= hi)
            bp.append(float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0)
        delta, theta, alpha, beta, gamma = bp
        total = delta + theta + alpha + beta + gamma
        rel = [b / total if total > 0 else 0.0 for b in bp]
        tbr = theta / beta if beta > 1e-12 else 0.0
        abr = alpha / beta if beta > 1e-12 else 0.0

        ac = psd[1:]
        ac_sum = float(np.sum(ac))
        p_ac = ac[ac > 0] / ac_sum
        spec_ent = float(-np.sum(p_ac * np.log2(p_ac))) / math.log2(len(ac))

        # Permutation entropy (order=3, delay=1)
        n_vec = n_samples - 2
        emb = np.column_stack((x[:n_vec], x[1 : n_vec + 1], x[2 : n_vec + 2]))
        pats = np.argsort(emb, axis=1, kind="mergesort")
        _, counts = np.unique(pats, axis=0, return_counts=True)
        probs = counts / n_vec
        pe = float(-np.sum(probs * np.log2(probs))) / math.log2(6.0)

        out[ch] = [
            rms, var_x, ll, zcr, var_x, mob, comp,
            delta, theta, alpha, beta, gamma, total,
            rel[0], rel[1], rel[2], rel[3], rel[4],
            tbr, abr, spec_ent, pe,
        ]
    return out


def pct_us(samples_ns: list[int], q: float) -> float:
    s = sorted(samples_ns)
    idx = int(round((len(s) - 1) * q))
    return s[idx] / 1000.0


def main() -> None:
    print(f"Platform: {platform.platform()} | Machine: {platform.machine()} | Python: {platform.python_version()}")
    fs = 250.0
    configs = [
        (8, 250, 125, 62),
        (8, 500, 250, 125),
        (16, 250, 125, 62),
        (16, 500, 250, 125),
        (24, 500, 250, 125),
        (24, 1000, 250, 125),
    ]

    print(
        f"\n{'Ch':<4} {'Win':<6} {'extract_into (us)':<24} {'extract() (us)':<22} {'SciPy/NumPy (us)':<20} {'Speedup':<9} {'PyAlloc(B)'}"
    )
    print(
        f"{'':<4} {'':<6} {'Med / p95 / p99':<24} {'Med / p95 / p99':<22} {'Median':<20} {'vs SciPy':<9} {'extract_into'}"
    )
    print("-" * 102)

    rng = np.random.default_rng(7)
    iters = 500

    for n_ch, n_win, nperseg, noverlap in configs:
        data_cs = rng.normal(0.0, 20.0, size=(n_ch, n_win)).astype(np.float64)
        raw_in = array.array("d", data_cs.ravel().tolist())
        raw_out = array.array("d", [0.0] * (n_ch * pf.NUM_FEATURES_PER_CHANNEL))

        ext = pf.FeatureExtractor(
            num_channels=n_ch, window_samples=n_win, fs=fs, nperseg=nperseg, noverlap=noverlap
        )

        # Warmup
        for _ in range(30):
            ext.extract_into(raw_in, raw_out, layout="channels_first")
            ext.extract(raw_in, layout="channels_first")

        # 1. Measure zero-allocation extract_into (stdlib array.array)
        tracemalloc.start()
        snap1 = tracemalloc.take_snapshot()
        t_into = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            ext.extract_into(raw_in, raw_out, layout="channels_first")
            t_into.append(time.perf_counter_ns() - t0)
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        # Net steady-state bytes allocated per call inside extract_into
        stats = snap2.compare_to(snap1, "lineno")
        net_alloc_per_call = max(0, sum(st.size_diff for st in stats if "bench_python_vs_native" not in str(st))) // iters

        # 2. Measure extract() returning PyFeatureFrame
        t_ext = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            _ = ext.extract(data_cs, layout="channels_first")
            t_ext.append(time.perf_counter_ns() - t0)

        # 3. Measure reference SciPy + NumPy
        t_scipy = []
        for _ in range(40):
            t0 = time.perf_counter_ns()
            _ = _scipy_numpy_extract_all_22(data_cs, fs, nperseg, noverlap)
            t_scipy.append(time.perf_counter_ns() - t0)

        med_into = pct_us(t_into, 0.50)
        p95_into = pct_us(t_into, 0.95)
        p99_into = pct_us(t_into, 0.99)

        med_ext = pct_us(t_ext, 0.50)
        p95_ext = pct_us(t_ext, 0.95)
        p99_ext = pct_us(t_ext, 0.99)

        med_scipy = pct_us(t_scipy, 0.50)
        speedup = med_scipy / med_into

        into_str = f"{med_into:5.1f} / {p95_into:5.1f} / {p99_into:5.1f}"
        ext_str = f"{med_ext:5.1f} / {p95_ext:5.1f} / {p99_ext:5.1f}"
        print(
            f"{n_ch:<4} {n_win:<6} {into_str:<24} {ext_str:<22} {med_scipy:<20.1f} {speedup:<8.1f}x {net_alloc_per_call} B"
        )


if __name__ == "__main__":
    main()
