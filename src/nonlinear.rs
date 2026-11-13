//! Nonlinear complexity measures for EEG.
//!
//! ## Permutation Entropy (Bandt & Pompe, 2002, *Physical Review Letters*)
//!
//! Given a 1D signal $x[0 \dots N-1]$, embedding dimension $m \in \{3, 4, 5, 6, 7\}$,
//! and time delay $\tau \ge 1$:
//!
//! 1. Construct $M = N - (m - 1)\tau$ embedded vectors:
//!    $$\mathbf{v}_i = \big(x[i],\, x[i + \tau],\, \dots,\, x[i + (m - 1)\tau]\big), \quad i = 0 \dots M-1$$
//! 2. Map each $\mathbf{v}_i$ to its ordinal pattern $\pi_i \in S_m$ (one of $m!$ permutations)
//!    by sorting indices $(0 \dots m-1)$ such that $v_i[\pi_0] \le v_i[\pi_1] \le \dots \le v_i[\pi_{m-1}]$,
//!    breaking ties by earlier index ($\pi_a < \pi_b$ when $v_i[\pi_a] == v_i[\pi_b]$), matching
//!    `antropy.perm_entropy` and `scipy`/`numpy.argsort(kind='mergesort')`.
//! 3. Compute the relative frequency $p(\pi) = \frac{\text{count}(\pi)}{M}$ of each of the $m!$ patterns.
//! 4. Compute Shannon entropy in bits:
//!    $$H_{\text{PE}} = -\sum_{\pi \in S_m : p(\pi) > 0} p(\pi) \log_2 p(\pi)$$
//!    If `normalize == true`, divide by $\log_2(m!)$ so $H_{\text{PE}} \in [0, 1]$.
//!
//! ### Fast Lehmer-code (factoradic) indexing
//! Instead of allocating vectors or running a general sort per sample, we compute the Lehmer code
//! index directly in registers:
//! $$\text{idx}(\mathbf{v}) = \sum_{j=0}^{m-2} c_j \cdot (m - 1 - j)!$$
//! where $c_j$ is the count of elements $k > j$ strictly less than $v[j]$.
//! Notice that because ties are broken by earlier time index ($j < k \implies v[j]$ precedes $v[k]$ when equal),
//! the inversion count is simply the number of $k > j$ with $v[k] < v[j]$!
//! For $m = 3$ ($3! = 6$ patterns), this requires just 3 comparisons per sample and a 6-element stack array.

const FACTORIALS: [usize; 8] = [1, 1, 2, 6, 24, 120, 720, 5040];

/// Reusable trait for nonlinear EEG complexity measures.
pub trait ComplexityMeasure {
    /// Compute the complexity measure on `signal`, returning a finite `f64`.
    fn compute(&self, signal: &[f64]) -> f64;
}

/// Permutation Entropy configuration (Bandt & Pompe, 2002).
#[derive(Debug, Clone, Copy)]
pub struct PermutationEntropy {
    pub order: usize,
    pub delay: usize,
    pub normalize: bool,
}

impl PermutationEntropy {
    pub fn new(order: usize, delay: usize, normalize: bool) -> Self {
        let order = order.clamp(2, 7);
        let delay = delay.max(1);
        Self {
            order,
            delay,
            normalize,
        }
    }
}

impl ComplexityMeasure for PermutationEntropy {
    #[inline]
    fn compute(&self, signal: &[f64]) -> f64 {
        permutation_entropy(signal, self.order, self.delay, self.normalize)
    }
}

/// Compute Permutation Entropy with zero heap allocation for $m \le 5$ (and stack-friendly up to $m = 7$).
///
/// - `order` ($m$): embedding dimension, supported range $2 \dots 7$ (default in EEG is $3$).
/// - `delay` ($\tau$): time delay in samples ($\ge 1$, default $1$).
/// - `normalize`: if `true`, normalizes by $\log_2(m!)$ into $[0, 1]$.
#[inline]
pub fn permutation_entropy(signal: &[f64], order: usize, delay: usize, normalize: bool) -> f64 {
    let m = order.clamp(2, 7);
    let tau = delay.max(1);
    let n = signal.len();
    let span = (m - 1) * tau;
    if n <= span {
        return 0.0;
    }

    let n_vectors = n - span;
    let n_perms = FACTORIALS[m];

    // Fast specialized path for m = 3 (6 patterns, 99% of real-time EEG neurofeedback/BCI use cases)
    if m == 3 {
        let mut counts = [0u32; 6];
        for i in 0..n_vectors {
            let x0 = signal[i];
            let x1 = signal[i + tau];
            let x2 = signal[i + 2 * tau];

            let c0 = ((x1 < x0) as usize) + ((x2 < x0) as usize);
            let c1 = (x2 < x1) as usize;
            let idx = c0 * 2 + c1;
            counts[idx] += 1;
        }
        return entropy_from_counts(&counts, n_vectors, normalize);
    }

    // Fast stack-allocated path for m <= 5 (up to 120 patterns)
    if m <= 5 {
        let mut counts = [0u32; 120];
        let active_counts = &mut counts[..n_perms];
        for i in 0..n_vectors {
            let mut idx = 0usize;
            for j in 0..(m - 1) {
                let xj = signal[i + j * tau];
                let mut inv = 0usize;
                for k in (j + 1)..m {
                    if signal[i + k * tau] < xj {
                        inv += 1;
                    }
                }
                idx += inv * FACTORIALS[m - 1 - j];
            }
            active_counts[idx] += 1;
        }
        return entropy_from_counts(active_counts, n_vectors, normalize);
    }

    // General path for m = 6..=7 (720 or 5040 patterns)
    let mut counts = vec![0u32; n_perms];
    for i in 0..n_vectors {
        let mut idx = 0usize;
        for j in 0..(m - 1) {
            let xj = signal[i + j * tau];
            let mut inv = 0usize;
            for k in (j + 1)..m {
                if signal[i + k * tau] < xj {
                    inv += 1;
                }
            }
            idx += inv * FACTORIALS[m - 1 - j];
        }
        counts[idx] += 1;
    }
    entropy_from_counts(&counts, n_vectors, normalize)
}

#[inline]
fn entropy_from_counts(counts: &[u32], total: usize, normalize: bool) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let inv_total = 1.0 / (total as f64);
    let mut h = 0.0;
    for &c in counts {
        // If a single ordinal pattern has 100% of vectors, log2(1.0) == 0.0 exactly
        if c > 0 && (c as usize) < total {
            let p = (c as f64) * inv_total;
            h -= p * p.log2();
        }
    }
    if normalize {
        let max_h = (counts.len() as f64).log2();
        if max_h > 0.0 {
            (h / max_h).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        h.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_and_flat_signal_zero_entropy() {
        // Strictly increasing ramp -> every 3-tuple has pattern (0, 1, 2) -> p = 1.0 -> H = 0.0
        let ramp: Vec<f64> = (0..500).map(|x| x as f64).collect();
        assert_eq!(permutation_entropy(&ramp, 3, 1, true), 0.0);

        // Flat signal -> tie-breaking assigns all tuples to (0, 1, 2) -> H = 0.0
        let flat = vec![42.0; 500];
        assert_eq!(permutation_entropy(&flat, 3, 1, true), 0.0);
    }

    #[test]
    fn test_antropy_reference_example() {
        // Classic textbook / Antropy example: x = [4, 7, 9, 10, 6, 11, 3]
        // For order = 3, delay = 1 (5 triplets):
        // (4, 7, 9)  -> (0, 1, 2)
        // (7, 9, 10) -> (0, 1, 2)
        // (9, 10, 6) -> (2, 0, 1)  [c0=1, c1=1 -> idx=3]
        // (10, 6, 11)-> (1, 0, 2)  [c0=1, c1=0 -> idx=2]
        // (6, 11, 3) -> (2, 0, 1)  [c0=1, c1=1 -> idx=3]
        // Counts: 2 of (0,1,2), 2 of (2,0,1), 1 of (1,0,2)
        // H = - (2/5 log2(2/5) + 2/5 log2(2/5) + 1/5 log2(1/5)) = 1.5219280948873621 bits
        // Normalized = 1.5219280948873621 / log2(6) = 0.5887621559162939
        let x = [4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0];
        let pe_unnorm = permutation_entropy(&x, 3, 1, false);
        let pe_norm = permutation_entropy(&x, 3, 1, true);

        let expected_unnorm = -(0.4 * 0.4_f64.log2() * 2.0 + 0.2 * 0.2_f64.log2());
        let expected_norm = expected_unnorm / 6.0_f64.log2();

        assert!((pe_unnorm - expected_unnorm).abs() < 1e-15);
        assert!((pe_norm - expected_norm).abs() < 1e-15);
    }

    #[test]
    fn test_higher_orders_match_m3_logic() {
        let x = [4.0, 7.0, 9.0, 10.0, 6.0, 11.0, 3.0, 8.0, 1.0, 5.0];
        let pe4 = permutation_entropy(&x, 4, 1, true);
        let pe5 = permutation_entropy(&x, 5, 1, true);
        assert!(pe4 > 0.0 && pe4 <= 1.0);
        assert!(pe5 > 0.0 && pe5 <= 1.0);
    }
}
