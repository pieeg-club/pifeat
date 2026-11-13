//! Frequency-domain EEG features: Welch PSD, canonical band powers, band ratios, and spectral entropy.
//!
//! ## Mathematical Specification (matching `scipy.signal.welch` and MNE-Python)
//!
//! Given a real signal $x[0 \dots N-1]$ sampled at $f_s$ Hz, segment length $L = \text{nperseg}$,
//! and overlap $O = \text{noverlap}$ (step size $S = L - O$):
//!
//! 1. **Periodic Hann window** (`fftbins=True` in SciPy):
//!    $$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{L}\right), \quad n = 0 \dots L-1$$
//!    Window power normalization factor:
//!    $$U = f_s \sum_{n=0}^{L-1} w[n]^2$$
//!
//! 2. **Per-segment modified periodogram**:
//!    For each segment $m \in \{0 \dots M-1\}$ starting at $m \cdot S$:
//!    - Constant detrend (`detrend='constant'`): subtract segment mean $\bar{x}_m = \frac{1}{L}\sum_{n=0}^{L-1} x_m[n]$.
//!    - Apply window: $y_m[n] = (x_m[n] - \bar{x}_m) \cdot w[n]$.
//!    - Compute real-to-complex FFT $Y_m[k]$ for $k = 0 \dots \lfloor L/2 \rfloor$.
//!    - One-sided PSD scaling (`scaling='density'`, units $\mu\text{V}^2/\text{Hz}$):
//!      $$P_m[k] = \frac{|Y_m[k]|^2}{U} \cdot \begin{cases} 1 & \text{if } k = 0 \text{ or } (k = L/2 \text{ when } L \text{ is even}) \\ 2 & \text{otherwise} \end{cases}$$
//!
//! 3. **Welch average**:
//!    $$P[k] = \frac{1}{M} \sum_{m=0}^{M-1} P_m[k]$$
//!    with frequency axis $f[k] = k \cdot \frac{f_s}{L} = k \cdot \Delta f$.
//!
//! 4. **Band-power integration**:
//!    Supports two explicit, documented integration modes:
//!    - [`BandIntegration::Trapezoidal`] (default, physical power in $\mu\text{V}^2$, matching YASA / MNE):
//!      Integrates $P(f)$ over $[f_{\text{lo}}, f_{\text{hi}}]$ via linear interpolation of band edges
//!      and composite trapezoidal rule across interior frequency bins. By Parseval's theorem,
//!      the sum of contiguous bands equals total AC signal variance.
//!    - [`BandIntegration::BinMean`] (legacy `pieeg-server` mode, average density in $\mu\text{V}^2/\text{Hz}$):
//!      Mean of $P[k]$ for bins $f_{\text{lo}} \le f[k] < f_{\text{hi}}$.
//!
//! 5. **Normalized Spectral Entropy** (Inouye et al., 1991; Antropy `spectral_entropy`):
//!    Let $\hat{p}_k = \frac{P[k]}{\sum_{j=1}^{K_{\max}} P[j]}$ over the active frequency range
//!    (excluding DC bin $k = 0$). Then:
//!    $$H_s = -\frac{1}{\log_2(N_{\text{bins}})} \sum_{k : \hat{p}_k > 0} \hat{p}_k \log_2(\hat{p}_k) \in [0, 1]$$

use std::f64::consts::PI;
use std::sync::Arc;

use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};

const EPSILON: f64 = 1e-24;
/// Minimum denominator band power ($\mu\text{V}^2$) to compute band ratios (`theta/beta`, `alpha/beta`)
/// without dividing by floating-point FFT noise floor ($10^{-28}$ in `f64` vs $10^{-15}$ in `f32`).
const RATIO_MIN_DENOM_POWER: f64 = 1e-12;

/// Canonical EEG frequency bands in Hz.
pub const BAND_DELTA: (f64, f64) = (0.5, 4.0);
pub const BAND_THETA: (f64, f64) = (4.0, 8.0);
pub const BAND_ALPHA: (f64, f64) = (8.0, 13.0);
pub const BAND_BETA: (f64, f64) = (13.0, 30.0);
pub const BAND_GAMMA: (f64, f64) = (30.0, 100.0);

/// How PSD is reduced over a frequency band $[f_{\text{lo}}, f_{\text{hi}}]$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandIntegration {
    /// Composite trapezoidal integration with linear interpolation at exact band boundaries.
    /// Produces true integrated band power in $\mu\text{V}^2$.
    Trapezoidal,
    /// Arithmetic mean of PSD bins in $[f_{\text{lo}}, f_{\text{hi}})$.
    /// Produces mean power spectral density in $\mu\text{V}^2/\text{Hz}$ (matches legacy `pieeg_server.spectral`).
    BinMean,
}

/// Complete frequency-domain feature set for one channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralFeatures {
    pub delta_power: f64,
    pub theta_power: f64,
    pub alpha_power: f64,
    pub beta_power: f64,
    pub gamma_power: f64,
    pub total_power: f64,
    pub rel_delta: f64,
    pub rel_theta: f64,
    pub rel_alpha: f64,
    pub rel_beta: f64,
    pub rel_gamma: f64,
    pub theta_beta_ratio: f64,
    pub alpha_beta_ratio: f64,
    pub spectral_entropy: f64,
}

/// Reusable, zero-allocation Welch PSD & spectral feature workspace.
///
/// Owns a pre-planned `RealToComplex<f64>` FFT, precomputed periodic Hann window,
/// frequency axis, and scratch buffers so steady-state extraction performs zero heap allocations.
pub struct WelchWorkspace {
    nperseg: usize,
    noverlap: usize,
    fs: f64,
    detrend_constant: bool,
    window: Vec<f64>,
    norm_factor: f64,
    freqs: Vec<f64>,
    r2c: Arc<dyn RealToComplex<f64>>,
    time_scratch: Vec<f64>,
    spec_scratch: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    psd_buf: Vec<f64>,
}

impl WelchWorkspace {
    /// Create a reusable Welch PSD workspace.
    ///
    /// - `nperseg`: FFT segment length (must be $\ge 4$).
    /// - `noverlap`: overlap in samples (must be $< \text{nperseg}$).
    /// - `fs`: sampling frequency in Hz (must be $> 0$).
    /// - `detrend_constant`: if `true`, subtracts each segment's mean before windowing (`scipy.signal.welch` default).
    pub fn new(nperseg: usize, noverlap: usize, fs: f64, detrend_constant: bool) -> Self {
        let nperseg = nperseg.max(4);
        let noverlap = noverlap.min(nperseg - 1);
        let fs = if fs > 0.0 { fs } else { 250.0 };

        // Periodic Hann window: w[n] = 0.5 - 0.5 * cos(2 * pi * n / L)
        // Matches scipy.signal.get_window('hann', L, fftbins=True)
        let window: Vec<f64> = (0..nperseg)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * (n as f64) / (nperseg as f64)).cos())
            .collect();

        let win_energy: f64 = window.iter().map(|&w| w * w).sum();
        let norm_factor = fs * win_energy;

        let n_bins = nperseg / 2 + 1;
        let df = fs / (nperseg as f64);
        let freqs: Vec<f64> = (0..n_bins).map(|k| (k as f64) * df).collect();

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(nperseg);
        let time_scratch = r2c.make_input_vec();
        let spec_scratch = r2c.make_output_vec();
        let fft_scratch = r2c.make_scratch_vec();
        let psd_buf = vec![0.0; n_bins];

        Self {
            nperseg,
            noverlap,
            fs,
            detrend_constant,
            window,
            norm_factor,
            freqs,
            r2c,
            time_scratch,
            spec_scratch,
            fft_scratch,
            psd_buf,
        }
    }

    #[inline]
    pub fn nperseg(&self) -> usize {
        self.nperseg
    }

    #[inline]
    pub fn noverlap(&self) -> usize {
        self.noverlap
    }

    #[inline]
    pub fn fs(&self) -> f64 {
        self.fs
    }

    #[inline]
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    #[inline]
    pub fn psd(&self) -> &[f64] {
        &self.psd_buf
    }

    /// Compute one-sided Welch PSD of `signal` into `self.psd_buf` and return a slice to it.
    ///
    /// Performs **zero heap allocations**.
    /// If `signal.len() < self.nperseg`, zero-pads the single available segment after detrending
    /// and scales by the active window energy.
    pub fn compute_psd(&mut self, signal: &[f64]) -> &[f64] {
        let n = signal.len();
        let l = self.nperseg;
        let n_bins = self.psd_buf.len();
        self.psd_buf.fill(0.0);

        if n == 0 {
            return &self.psd_buf;
        }

        let step = l - self.noverlap;
        let has_nyquist = l % 2 == 0;
        let last_interior = if has_nyquist { n_bins - 2 } else { n_bins - 1 };

        if n < l {
            // Short signal fallback: detrend available samples, window with first n samples of Hann, zero-pad rest
            let mean = if self.detrend_constant {
                signal.iter().sum::<f64>() / (n as f64)
            } else {
                0.0
            };
            let mut active_win_energy = 0.0;
            for i in 0..n {
                let w = 0.5 - 0.5 * (2.0 * PI * (i as f64) / (n as f64)).cos();
                active_win_energy += w * w;
                self.time_scratch[i] = (signal[i] - mean) * w;
            }
            for i in n..l {
                self.time_scratch[i] = 0.0;
            }
            let norm = (self.fs * active_win_energy).max(EPSILON);
            let _ = self.r2c.process_with_scratch(
                &mut self.time_scratch,
                &mut self.spec_scratch,
                &mut self.fft_scratch,
            );
            for k in 0..n_bins {
                let c = self.spec_scratch[k];
                let mag_sq = c.re * c.re + c.im * c.im;
                let scale = if k >= 1 && k <= last_interior { 2.0 } else { 1.0 };
                self.psd_buf[k] = (mag_sq * scale) / norm;
            }
            return &self.psd_buf;
        }

        let n_segments = (n - l) / step + 1;
        for seg in 0..n_segments {
            let start = seg * step;
            let slice = &signal[start..start + l];

            let mean = if self.detrend_constant {
                slice.iter().sum::<f64>() / (l as f64)
            } else {
                0.0
            };

            for i in 0..l {
                self.time_scratch[i] = (slice[i] - mean) * self.window[i];
            }

            let _ = self.r2c.process_with_scratch(
                &mut self.time_scratch,
                &mut self.spec_scratch,
                &mut self.fft_scratch,
            );

            for k in 0..n_bins {
                let c = self.spec_scratch[k];
                self.psd_buf[k] += c.re * c.re + c.im * c.im;
            }
        }

        let inv_norm = 1.0 / ((n_segments as f64) * self.norm_factor);
        self.psd_buf[0] *= inv_norm;
        for k in 1..=last_interior {
            self.psd_buf[k] *= 2.0 * inv_norm;
        }
        if has_nyquist {
            self.psd_buf[n_bins - 1] *= inv_norm;
        }

        &self.psd_buf
    }

    /// Compute a single modified periodogram segment of length `nperseg` into `out_psd`.
    /// Used by the streaming engine to cache per-hop FFT segments and avoid recomputing overlapping windows!
    #[inline]
    pub fn compute_single_segment_psd(&mut self, segment: &[f64], out_psd: &mut [f64]) {
        debug_assert_eq!(segment.len(), self.nperseg);
        debug_assert_eq!(out_psd.len(), self.psd_buf.len());

        let l = self.nperseg;
        let n_bins = out_psd.len();
        let has_nyquist = l % 2 == 0;
        let last_interior = if has_nyquist { n_bins - 2 } else { n_bins - 1 };

        let mean = if self.detrend_constant {
            segment.iter().sum::<f64>() / (l as f64)
        } else {
            0.0
        };

        for i in 0..l {
            self.time_scratch[i] = (segment[i] - mean) * self.window[i];
        }

        let _ = self.r2c.process_with_scratch(
            &mut self.time_scratch,
            &mut self.spec_scratch,
            &mut self.fft_scratch,
        );

        let inv_norm = 1.0 / self.norm_factor;
        let c0 = self.spec_scratch[0];
        out_psd[0] = (c0.re * c0.re + c0.im * c0.im) * inv_norm;

        for k in 1..=last_interior {
            let c = self.spec_scratch[k];
            out_psd[k] = (c.re * c.re + c.im * c.im) * (2.0 * inv_norm);
        }
        if has_nyquist {
            let cn = self.spec_scratch[n_bins - 1];
            out_psd[n_bins - 1] = (cn.re * cn.re + cn.im * cn.im) * inv_norm;
        }
    }

    /// Extract all spectral features from a signal window in one call.
    #[inline]
    pub fn extract_spectral_features(
        &mut self,
        signal: &[f64],
        integration: BandIntegration,
    ) -> SpectralFeatures {
        self.compute_psd(signal);
        compute_features_from_psd(&self.freqs, &self.psd_buf, self.fs, integration)
    }
}

/// Integrate or average PSD over $[f_{\text{lo}}, f_{\text{hi}}]$, automatically clamping $f_{\text{hi}}$ to Nyquist $f_s / 2$.
#[inline]
pub fn band_power(
    freqs: &[f64],
    psd: &[f64],
    f_lo: f64,
    f_hi: f64,
    integration: BandIntegration,
) -> f64 {
    if freqs.len() < 2 || freqs.len() != psd.len() {
        return 0.0;
    }
    let f_min = freqs[0];
    let f_max = *freqs.last().unwrap();
    let lo = f_lo.max(f_min);
    let hi = f_hi.min(f_max);
    if hi <= lo {
        return 0.0;
    }

    match integration {
        BandIntegration::BinMean => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for (&f, &p) in freqs.iter().zip(psd.iter()) {
                if f >= lo && f < hi {
                    sum += p;
                    count += 1;
                }
            }
            if count > 0 {
                sum / (count as f64)
            } else {
                0.0
            }
        }
        BandIntegration::Trapezoidal => {
            let df = freqs[1] - freqs[0];
            if df <= 0.0 {
                return 0.0;
            }
            // Interpolate PSD at exact lo and hi boundaries for sub-bin accuracy,
            // plus composite trapezoid over all interior grid points.
            let p_at = |target_f: f64| -> f64 {
                let pos = (target_f - f_min) / df;
                let idx = (pos.floor() as usize).min(freqs.len() - 2);
                let frac = (pos - (idx as f64)).clamp(0.0, 1.0);
                psd[idx] + frac * (psd[idx + 1] - psd[idx])
            };

            let first_interior = ((lo - f_min) / df).ceil() as usize;
            let last_interior = ((hi - f_min) / df).floor() as usize;

            if first_interior > last_interior {
                // Entire band lies within a single frequency bin interval
                let p_lo = p_at(lo);
                let p_hi = p_at(hi);
                return 0.5 * (p_lo + p_hi) * (hi - lo);
            }

            let mut area = 0.0;
            // Left partial trapezoid: [lo, freqs[first_interior]]
            let f_first = freqs[first_interior];
            if f_first > lo {
                let p_lo = p_at(lo);
                area += 0.5 * (p_lo + psd[first_interior]) * (f_first - lo);
            }

            // Interior full bins: [first_interior .. last_interior]
            for k in first_interior..last_interior {
                area += 0.5 * (psd[k] + psd[k + 1]) * (freqs[k + 1] - freqs[k]);
            }

            // Right partial trapezoid: [freqs[last_interior], hi]
            let f_last = freqs[last_interior];
            if hi > f_last {
                let p_hi = p_at(hi);
                area += 0.5 * (psd[last_interior] + p_hi) * (hi - f_last);
            }

            area
        }
    }
}

/// Compute normalized Shannon spectral entropy in $[0, 1]$ from a PSD array.
///
/// Excludes DC bin ($k = 0$) so remaining DC offsets do not collapse entropy toward 0.
/// Returns $0.0$ for a flat/zero-power signal and $1.0$ for uniform white noise.
#[inline]
pub fn spectral_entropy_from_psd(psd: &[f64]) -> f64 {
    if psd.len() <= 2 {
        return 0.0;
    }
    // Exclude DC bin (index 0)
    let ac_psd = &psd[1..];
    let n_bins = ac_psd.len();
    let total: f64 = ac_psd.iter().copied().filter(|&p| p > 0.0).sum();
    if total <= EPSILON {
        return 0.0;
    }

    let mut h = 0.0;
    for &p in ac_psd {
        if p > EPSILON {
            let prob = p / total;
            h -= prob * prob.log2();
        }
    }
    let max_h = (n_bins as f64).log2();
    if max_h <= EPSILON {
        0.0
    } else {
        (h / max_h).clamp(0.0, 1.0)
    }
}

/// Compute all canonical band powers, relative powers, ratios, and spectral entropy from a precomputed PSD.
#[inline]
pub fn compute_features_from_psd(
    freqs: &[f64],
    psd: &[f64],
    fs: f64,
    integration: BandIntegration,
) -> SpectralFeatures {
    let nyquist = 0.5 * fs;
    let delta = band_power(freqs, psd, BAND_DELTA.0, BAND_DELTA.1.min(nyquist), integration);
    let theta = band_power(freqs, psd, BAND_THETA.0, BAND_THETA.1.min(nyquist), integration);
    let alpha = band_power(freqs, psd, BAND_ALPHA.0, BAND_ALPHA.1.min(nyquist), integration);
    let beta = band_power(freqs, psd, BAND_BETA.0, BAND_BETA.1.min(nyquist), integration);
    let gamma = band_power(freqs, psd, BAND_GAMMA.0, BAND_GAMMA.1.min(nyquist), integration);

    let total = delta + theta + alpha + beta + gamma;
    let (rel_delta, rel_theta, rel_alpha, rel_beta, rel_gamma) = if total > EPSILON {
        (
            delta / total,
            theta / total,
            alpha / total,
            beta / total,
            gamma / total,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let ratio_floor = RATIO_MIN_DENOM_POWER.max(total * 1e-9);
    let theta_beta_ratio = if beta > ratio_floor { theta / beta } else { 0.0 };
    let alpha_beta_ratio = if beta > ratio_floor { alpha / beta } else { 0.0 };
    let spec_ent = spectral_entropy_from_psd(psd);

    SpectralFeatures {
        delta_power: delta,
        theta_power: theta,
        alpha_power: alpha,
        beta_power: beta,
        gamma_power: gamma,
        total_power: total,
        rel_delta,
        rel_theta,
        rel_alpha,
        rel_beta,
        rel_gamma,
        theta_beta_ratio,
        alpha_beta_ratio,
        spectral_entropy: spec_ent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parseval_sine_wave_band_power() {
        // 10 Hz alpha sine wave, amplitude A = 10 uV -> theoretical power = A^2 / 2 = 50 uV^2
        let fs = 250.0;
        let nperseg = 500; // 2.0s window -> exact 0.5 Hz bin spacing
        let amp = 10.0;
        let freq = 10.0;
        let sig: Vec<f64> = (0..1000)
            .map(|i| amp * (2.0 * PI * freq * (i as f64) / fs).sin())
            .collect();

        let mut ws = WelchWorkspace::new(nperseg, 250, fs, true);
        let feat = ws.extract_spectral_features(&sig, BandIntegration::Trapezoidal);

        // Total power in Alpha (8-13 Hz) should be ~50.0 uV^2 (within 0.1% due to Hann mainlobe inside [8, 13] Hz)
        assert!(
            (feat.alpha_power - 50.0).abs() < 0.05,
            "Expected ~50.0 uV^2, got {}",
            feat.alpha_power
        );
        assert!(feat.rel_alpha > 0.999);
        assert!(feat.delta_power < 1e-4);
        assert!(feat.theta_power < 1e-4);
        assert!(feat.beta_power < 1e-4);
        assert!(feat.gamma_power < 1e-4);
    }

    #[test]
    fn test_spectral_entropy_bounds() {
        // Pure sine at exact bin center -> concentrated in ~3 Hann bins -> very low spectral entropy
        let fs = 250.0;
        let n = 500;
        let sine: Vec<f64> = (0..n)
            .map(|i| 10.0 * (2.0 * PI * 10.0 * (i as f64) / fs).sin())
            .collect();
        let mut ws = WelchWorkspace::new(250, 125, fs, true);
        let f_sine = ws.extract_spectral_features(&sine, BandIntegration::Trapezoidal);
        assert!(f_sine.spectral_entropy > 0.0 && f_sine.spectral_entropy < 0.25);

        // Flat signal -> 0.0 (never NaN)
        let flat = vec![5.0; 500];
        let f_flat = ws.extract_spectral_features(&flat, BandIntegration::Trapezoidal);
        assert_eq!(f_flat.spectral_entropy, 0.0);
        assert_eq!(f_flat.total_power, 0.0);
        assert_eq!(f_flat.theta_beta_ratio, 0.0);
    }
}
