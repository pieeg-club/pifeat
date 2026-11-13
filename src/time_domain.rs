//! Time-domain EEG features.
//!
//! Mathematical definitions:
//! - **Mean**: $\mu = \frac{1}{N} \sum_{n=0}^{N-1} x_n$
//! - **RMS**: $\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x_n^2}$
//! - **Variance** (sample variance, Bessel-corrected, `ddof = 1`):
//!   $\sigma^2 = \frac{1}{N - 1} \sum_{n=0}^{N-1} (x_n - \mu)^2$
//! - **Line Length** (Esteller et al., 2001):
//!   $L = \sum_{n=1}^{N-1} |x_n - x_{n-1}|$
//! - **Zero-Crossing Rate** (around window mean $\mu$ to be invariant to DC offset):
//!   $\text{ZCR} = \frac{1}{N - 1} \sum_{n=1}^{N-1} \mathbb{I}\big((x_n - \mu)(x_{n-1} - \mu) < 0\big)$
//! - **Hjorth Parameters** (Hjorth, 1970, *EEG and Clinical Neurophysiology*):
//!   Let $d_1[n] = (x_n - x_{n-1}) \cdot f_s$ for $n = 1 \dots N-1$ and
//!   $d_2[n] = (d_1[n] - d_1[n-1]) \cdot f_s$ for $n = 1 \dots N-2$.
//!   Note: in the dimensionless ratios Mobility and Complexity, multiplying by $f_s$
//!   cancels in Complexity and scales Mobility into $\text{s}^{-1}$ (or remains
//!   rad/sample if unscaled; we follow the standard MNE-Features / Antropy convention
//!   using first differences $d_1[n] = x_n - x_{n-1}$, where variances use population or
//!   sample estimator consistently). Specifically, with zero-mean derivative energy
//!   or sample variance $\text{var}(\cdot)$:
//!   - $\text{Activity} = \text{var}(x)$ (with `ddof = 1`, matching sample variance)
//!   - $\text{Mobility} = \sqrt{\frac{\text{var}(d_1)}{\text{var}(x)}}$
//!   - $\text{Complexity} = \frac{\text{Mobility}(d_1)}{\text{Mobility}(x)} = \sqrt{\frac{\text{var}(d_2) / \text{var}(d_1)}{\text{var}(d_1) / \text{var}(x)}}$
//!
//! Degenerate / flat-line handling:
//! - If $N < 2$ or signal has non-finite values, returns `NaN` or `0.0` with validity flag set by caller.
//! - If $\text{var}(x) \le \varepsilon$ (flat signal, $\varepsilon = 10^{-30}$),
//!   Activity = $0.0$, Mobility = $0.0$, Complexity = $0.0$ (never `NaN` or `Inf` on a valid flat EEG line).

const EPSILON: f64 = 1e-30;

/// Bundle of time-domain features for a single channel window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeDomainFeatures {
    pub rms: f64,
    pub variance: f64,
    pub line_length: f64,
    pub zero_crossing_rate: f64,
    pub hjorth_activity: f64,
    pub hjorth_mobility: f64,
    pub hjorth_complexity: f64,
}

///Hjorth parameters: (activity, mobility, complexity).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HjorthParams {
    pub activity: f64,
    pub mobility: f64,
    pub complexity: f64,
}

/// Compute Root Mean Square (RMS) of a signal window.
#[inline]
pub fn rms(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
    (sum_sq / (n as f64)).sqrt()
}

/// Compute sample variance (`ddof = 1`) using two-pass compensated summation for numerical stability.
#[inline]
pub fn variance(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = signal.iter().sum::<f64>() / nf;
    let mut sum_sq = 0.0;
    let mut sum_err = 0.0;
    for &x in signal {
        let d = x - mean;
        sum_sq += d * d;
        sum_err += d;
    }
    // Two-pass correction: reduces roundoff error when DC offset >> AC amplitude
    let corrected = (sum_sq - (sum_err * sum_err) / nf).max(0.0);
    corrected / ((n - 1) as f64)
}

/// Compute total line length $\sum_{n=1}^{N-1} |x_n - x_{n-1}|$.
#[inline]
pub fn line_length(signal: &[f64]) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }
    signal
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum()
}

/// Compute mean-centered Zero-Crossing Rate in $[0, 1]$.
///
/// Mean-centering ensures DC offset on EEG amplifiers does not suppress zero crossings.
/// A strict sign change or exact touch transitions across the mean are counted cleanly
/// by tracking the last non-zero sign.
#[inline]
pub fn zero_crossing_rate(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }
    let mean = signal.iter().sum::<f64>() / (n as f64);
    let mut crossings = 0usize;
    let mut prev_diff = signal[0] - mean;

    for &x in &signal[1..] {
        let diff = x - mean;
        if (diff > 0.0 && prev_diff < 0.0) || (diff < 0.0 && prev_diff > 0.0) {
            crossings += 1;
            prev_diff = diff;
        } else if diff != 0.0 {
            prev_diff = diff;
        }
    }
    (crossings as f64) / ((n - 1) as f64)
}

/// Compute Hjorth Activity, Mobility, and Complexity in a single fused pass without heap allocations.
///
/// Uses sample variance (`ddof = 1`) for $x$, first difference $d_1[n] = x_n - x_{n-1}$,
/// and second difference $d_2[n] = x_n - 2x_{n-1} + x_{n-2}$.
#[inline]
pub fn hjorth_parameters(signal: &[f64]) -> HjorthParams {
    let n = signal.len();
    if n < 3 {
        let act = variance(signal);
        return HjorthParams {
            activity: act,
            mobility: 0.0,
            complexity: 0.0,
        };
    }

    let var_x = variance(signal);
    if var_x <= EPSILON {
        return HjorthParams {
            activity: 0.0,
            mobility: 0.0,
            complexity: 0.0,
        };
    }

    // First difference d1: length n - 1
    // Mean of d1 is simply (signal[n-1] - signal[0]) / (n - 1)
    let n1 = (n - 1) as f64;
    let mean_d1 = (signal[n - 1] - signal[0]) / n1;

    // Second difference d2: length n - 2
    // Mean of d2 is ((signal[n-1] - signal[n-2]) - (signal[1] - signal[0])) / (n - 2)
    let n2 = (n - 2) as f64;
    let mean_d2 = ((signal[n - 1] - signal[n - 2]) - (signal[1] - signal[0])) / n2;

    let mut sum_sq_d1 = 0.0;
    let mut sum_sq_d2 = 0.0;

    let mut prev_d1 = signal[1] - signal[0];
    let diff1_0 = prev_d1 - mean_d1;
    sum_sq_d1 += diff1_0 * diff1_0;

    for i in 2..n {
        let d1 = signal[i] - signal[i - 1];
        let c1 = d1 - mean_d1;
        sum_sq_d1 += c1 * c1;

        let d2 = d1 - prev_d1;
        let c2 = d2 - mean_d2;
        sum_sq_d2 += c2 * c2;

        prev_d1 = d1;
    }

    let var_d1 = sum_sq_d1 / ((n - 2) as f64).max(1.0);
    let var_d2 = if n > 3 {
        sum_sq_d2 / ((n - 3) as f64)
    } else {
        sum_sq_d2
    };

    let mobility = (var_d1 / var_x).sqrt();
    let complexity = if var_d1 <= EPSILON || mobility <= EPSILON {
        0.0
    } else {
        let mob_d1 = (var_d2 / var_d1).sqrt();
        mob_d1 / mobility
    };

    HjorthParams {
        activity: var_x,
        mobility,
        complexity,
    }
}

/// Compute all time-domain features in two cache-friendly passes over `signal` with zero heap allocations.
#[inline]
pub fn extract_time_domain(signal: &[f64]) -> TimeDomainFeatures {
    let n = signal.len();
    if n == 0 {
        return TimeDomainFeatures {
            rms: 0.0,
            variance: 0.0,
            line_length: 0.0,
            zero_crossing_rate: 0.0,
            hjorth_activity: 0.0,
            hjorth_mobility: 0.0,
            hjorth_complexity: 0.0,
        };
    }

    let nf = n as f64;
    let mut sum_x = 0.0;
    let mut sum_x2 = 0.0;
    let mut ll = 0.0;

    sum_x += signal[0];
    sum_x2 += signal[0] * signal[0];

    for i in 1..n {
        let x = signal[i];
        sum_x += x;
        sum_x2 += x * x;
        ll += (x - signal[i - 1]).abs();
    }

    let mean = sum_x / nf;
    let rms_val = (sum_x2 / nf).sqrt();

    if n < 2 {
        return TimeDomainFeatures {
            rms: rms_val,
            variance: 0.0,
            line_length: 0.0,
            zero_crossing_rate: 0.0,
            hjorth_activity: 0.0,
            hjorth_mobility: 0.0,
            hjorth_complexity: 0.0,
        };
    }

    let n1 = (n - 1) as f64;
    let mean_d1 = (signal[n - 1] - signal[0]) / n1;
    let mean_d2 = if n >= 3 {
        ((signal[n - 1] - signal[n - 2]) - (signal[1] - signal[0])) / ((n - 2) as f64)
    } else {
        0.0
    };

    let mut sum_sq_x = 0.0;
    let mut sum_err_x = 0.0;
    let mut sum_sq_d1 = 0.0;
    let mut sum_sq_d2 = 0.0;
    let mut crossings = 0usize;

    let d0 = signal[0] - mean;
    sum_sq_x += d0 * d0;
    sum_err_x += d0;
    let mut prev_centered = d0;
    let mut prev_d1 = 0.0;

    for i in 1..n {
        let x = signal[i];
        let centered = x - mean;
        sum_sq_x += centered * centered;
        sum_err_x += centered;

        if (centered > 0.0 && prev_centered < 0.0) || (centered < 0.0 && prev_centered > 0.0) {
            crossings += 1;
            prev_centered = centered;
        } else if centered != 0.0 {
            prev_centered = centered;
        }

        let d1 = x - signal[i - 1];
        let c1 = d1 - mean_d1;
        sum_sq_d1 += c1 * c1;

        if i >= 2 {
            let d2 = d1 - prev_d1;
            let c2 = d2 - mean_d2;
            sum_sq_d2 += c2 * c2;
        }
        prev_d1 = d1;
    }

    let corrected_sq_x = (sum_sq_x - (sum_err_x * sum_err_x) / nf).max(0.0);
    let var_x = corrected_sq_x / n1;
    let zcr = (crossings as f64) / n1;

    if n < 3 || var_x <= EPSILON {
        return TimeDomainFeatures {
            rms: rms_val,
            variance: if var_x <= EPSILON { 0.0 } else { var_x },
            line_length: ll,
            zero_crossing_rate: zcr,
            hjorth_activity: if var_x <= EPSILON { 0.0 } else { var_x },
            hjorth_mobility: 0.0,
            hjorth_complexity: 0.0,
        };
    }

    let var_d1 = sum_sq_d1 / ((n - 2) as f64);
    let var_d2 = if n > 3 {
        sum_sq_d2 / ((n - 3) as f64)
    } else {
        sum_sq_d2
    };

    let mobility = (var_d1 / var_x).sqrt();
    let complexity = if var_d1 <= EPSILON || mobility <= EPSILON {
        0.0
    } else {
        let mob_d1 = (var_d2 / var_d1).sqrt();
        mob_d1 / mobility
    };

    TimeDomainFeatures {
        rms: rms_val,
        variance: var_x,
        line_length: ll,
        zero_crossing_rate: zcr,
        hjorth_activity: var_x,
        hjorth_mobility: mobility,
        hjorth_complexity: complexity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_sine_rms_and_hjorth() {
        // Pure 10 Hz sine wave, amplitude A = 20 uV, fs = 250 Hz, 4 seconds (1000 samples)
        let fs = 250.0;
        let freq = 10.0;
        let amp = 20.0;
        let n = 1000;
        let sig: Vec<f64> = (0..n)
            .map(|i| amp * (2.0 * PI * freq * (i as f64) / fs).sin())
            .collect();

        let td = extract_time_domain(&sig);

        // Exact RMS of integer number of periods of sine wave is A / sqrt(2)
        let expected_rms = amp / 2.0_f64.sqrt();
        assert!((td.rms - expected_rms).abs() < 1e-12);

        // Sample variance = (N / (N - 1)) * (A^2 / 2)
        let expected_var = (n as f64 / (n - 1) as f64) * (amp * amp / 2.0);
        assert!((td.variance - expected_var).abs() < 1e-10);
        assert!((td.hjorth_activity - expected_var).abs() < 1e-10);

        // For a pure sinusoid x[n] = A sin(omega * n), first difference has amplitude 2*sin(omega/2)*A
        // so discrete mobility ≈ 2 * sin(pi * f / fs) and complexity ≈ 1.0
        let omega = 2.0 * PI * freq / fs;
        let expected_mob = 2.0 * (omega / 2.0).sin();
        assert!(
            (td.hjorth_mobility - expected_mob).abs() < 1e-3,
            "mobility={} vs expected={}",
            td.hjorth_mobility,
            expected_mob
        );
        assert!(
            (td.hjorth_complexity - 1.0).abs() < 2e-3,
            "complexity={}",
            td.hjorth_complexity
        );

        let hj = hjorth_parameters(&sig);
        assert!((td.hjorth_activity - hj.activity).abs() < 1e-12);
        assert!((td.hjorth_mobility - hj.mobility).abs() < 1e-12);
        assert!((td.hjorth_complexity - hj.complexity).abs() < 1e-12);
    }

    #[test]
    fn test_flat_signal_degenerate() {
        let sig = vec![42.5; 512];
        let td = extract_time_domain(&sig);
        assert!((td.rms - 42.5).abs() < 1e-12);
        assert_eq!(td.variance, 0.0);
        assert_eq!(td.line_length, 0.0);
        assert_eq!(td.zero_crossing_rate, 0.0);
        assert_eq!(td.hjorth_activity, 0.0);
        assert_eq!(td.hjorth_mobility, 0.0);
        assert_eq!(td.hjorth_complexity, 0.0);
    }

    #[test]
    fn test_large_dc_offset_numerical_stability() {
        // 1e6 uV DC offset (e.g. uncalibrated raw ADC baseline) + tiny 1 uV alternating signal
        let n = 500;
        let sig: Vec<f64> = (0..n)
            .map(|i| 1_000_000.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let td = extract_time_domain(&sig);
        let expected_var = (n as f64) / ((n - 1) as f64); // 1.0^2 * N/(N-1)
        assert!((td.variance - expected_var).abs() < 1e-9);
        assert!((td.zero_crossing_rate - 1.0).abs() < 1e-12);
    }
}
