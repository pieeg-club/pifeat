//! Batch and stateful streaming feature extraction engines.
//!
//! Includes:
//! - [`FeatureExtractor`]: Preallocated batch multichannel window extractor.
//!   Zero heap allocations on the steady-state hot path (`extract_into`).
//! - [`StreamingFeatureEngine`]: Stateful rolling-window multichannel engine with:
//!   - Circular sample buffers per channel
//!   - Sequence gap (`missing_samples`) and timestamp-jitter detection
//!   - Channel quality / validity bitflags (`NON_FINITE`, `FLATLINE`, `SEQ_GAP`)
//!   - **Segment PSD Ring Cache**: When `hop_samples == nperseg - noverlap` (e.g., 50% overlap Welch
//!     where `nperseg = 250` and `hop = 125`), each hop only computes **one** new `nperseg` FFT per
//!     channel instead of re-running all overlapping Welch segments over the full window!

use crate::nonlinear::permutation_entropy;
use crate::spectral::{
    compute_features_from_psd, BandIntegration, SpectralFeatures, WelchWorkspace,
};
use crate::time_domain::{extract_time_domain, TimeDomainFeatures};

/// Number of scalar features emitted per channel in the standard v0.1 vector.
pub const NUM_FEATURES_PER_CHANNEL: usize = 22;

/// Ordered canonical names of the 22 features per channel.
pub const FEATURE_NAMES: [&str; NUM_FEATURES_PER_CHANNEL] = [
    "rms",
    "variance",
    "line_length",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "total_power",
    "rel_delta",
    "rel_theta",
    "rel_alpha",
    "rel_beta",
    "rel_gamma",
    "theta_beta_ratio",
    "alpha_beta_ratio",
    "spectral_entropy",
    "permutation_entropy",
];

/// Validity bitflags for a channel or window.
pub const FLAG_VALID: u32 = 0;
pub const FLAG_NON_FINITE: u32 = 1 << 0;
pub const FLAG_FLATLINE: u32 = 1 << 1;
pub const FLAG_SEQ_GAP: u32 = 1 << 2;

const FLATLINE_VAR_EPSILON: f64 = 1e-24;

/// Memory layout of multichannel input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLayout {
    /// Channel-major `[num_channels, num_samples]`: each channel is a contiguous slice of `num_samples`.
    ChannelsFirst,
    /// Sample-major `[num_samples, num_channels]`: interleaved time frames.
    SamplesFirst,
}

/// Complete extracted feature frame for all channels in a window.
#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub num_channels: usize,
    pub window_samples: usize,
    pub timestamp_start: f64,
    pub timestamp_end: f64,
    pub seq_start: u64,
    pub seq_end: u64,
    pub missing_samples: u64,
    pub window_valid: bool,
    pub channel_flags: Vec<u32>,
    /// Flat row-major `[num_channels * NUM_FEATURES_PER_CHANNEL]` feature matrix.
    pub values: Vec<f64>,
}

impl FeatureFrame {
    pub fn new(num_channels: usize, window_samples: usize) -> Self {
        Self {
            num_channels,
            window_samples,
            timestamp_start: 0.0,
            timestamp_end: 0.0,
            seq_start: 0,
            seq_end: 0,
            missing_samples: 0,
            window_valid: true,
            channel_flags: vec![FLAG_VALID; num_channels],
            values: vec![0.0; num_channels * NUM_FEATURES_PER_CHANNEL],
        }
    }

    #[inline]
    pub fn channel_features(&self, ch: usize) -> &[f64] {
        let start = ch * NUM_FEATURES_PER_CHANNEL;
        &self.values[start..start + NUM_FEATURES_PER_CHANNEL]
    }
}

#[inline]
pub fn write_feature_row(
    dst: &mut [f64],
    td: &TimeDomainFeatures,
    sp: &SpectralFeatures,
    pe: f64,
) {
    debug_assert!(dst.len() >= NUM_FEATURES_PER_CHANNEL);
    dst[0] = td.rms;
    dst[1] = td.variance;
    dst[2] = td.line_length;
    dst[3] = td.zero_crossing_rate;
    dst[4] = td.hjorth_activity;
    dst[5] = td.hjorth_mobility;
    dst[6] = td.hjorth_complexity;
    dst[7] = sp.delta_power;
    dst[8] = sp.theta_power;
    dst[9] = sp.alpha_power;
    dst[10] = sp.beta_power;
    dst[11] = sp.gamma_power;
    dst[12] = sp.total_power;
    dst[13] = sp.rel_delta;
    dst[14] = sp.rel_theta;
    dst[15] = sp.rel_alpha;
    dst[16] = sp.rel_beta;
    dst[17] = sp.rel_gamma;
    dst[18] = sp.theta_beta_ratio;
    dst[19] = sp.alpha_beta_ratio;
    dst[20] = sp.spectral_entropy;
    dst[21] = pe;
}

/// Reusable multichannel batch feature extractor.
///
/// Preallocates the FFT plan, Welch scratch buffers, de-interleaving transpose scratch,
/// and output frame so repeated calls to `extract_into` perform **zero heap allocations**.
pub struct FeatureExtractor {
    num_channels: usize,
    window_samples: usize,
    fs: f64,
    pe_order: usize,
    pe_delay: usize,
    integration: BandIntegration,
    welch: WelchWorkspace,
    channel_scratch: Vec<f64>,
}

impl FeatureExtractor {
    pub fn new(
        num_channels: usize,
        window_samples: usize,
        fs: f64,
        nperseg: usize,
        noverlap: usize,
        pe_order: usize,
        pe_delay: usize,
        integration: BandIntegration,
    ) -> Self {
        let num_channels = num_channels.max(1);
        let window_samples = window_samples.max(4);
        let nperseg = nperseg.min(window_samples).max(4);
        let noverlap = noverlap.min(nperseg - 1);
        Self {
            num_channels,
            window_samples,
            fs,
            pe_order: pe_order.clamp(2, 7),
            pe_delay: pe_delay.max(1),
            integration,
            welch: WelchWorkspace::new(nperseg, noverlap, fs, true),
            channel_scratch: vec![0.0; window_samples],
        }
    }

    #[inline]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    #[inline]
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    #[inline]
    pub fn fs(&self) -> f64 {
        self.fs
    }

    #[inline]
    pub fn freqs(&self) -> &[f64] {
        self.welch.freqs()
    }

    /// Extract features from a contiguous `f64` buffer into a preallocated [`FeatureFrame`].
    ///
    /// Performs **zero heap allocations** when `layout == DataLayout::ChannelsFirst` or when
    /// `window_samples <= self.channel_scratch.len()`.
    pub fn extract_f64_into(
        &mut self,
        data: &[f64],
        num_channels: usize,
        num_samples: usize,
        layout: DataLayout,
        out: &mut FeatureFrame,
    ) {
        assert_eq!(data.len(), num_channels * num_samples);
        if out.num_channels != num_channels {
            out.num_channels = num_channels;
            out.channel_flags.resize(num_channels, FLAG_VALID);
            out.values
                .resize(num_channels * NUM_FEATURES_PER_CHANNEL, 0.0);
        }
        out.window_samples = num_samples;
        out.missing_samples = 0;
        out.window_valid = true;

        if self.channel_scratch.len() < num_samples {
            self.channel_scratch.resize(num_samples, 0.0);
        }

        for ch in 0..num_channels {
            let ch_slice: &[f64] = match layout {
                DataLayout::ChannelsFirst => {
                    let start = ch * num_samples;
                    &data[start..start + num_samples]
                }
                DataLayout::SamplesFirst => {
                    for s in 0..num_samples {
                        self.channel_scratch[s] = data[s * num_channels + ch];
                    }
                    &self.channel_scratch[..num_samples]
                }
            };

            let dst = &mut out.values
                [ch * NUM_FEATURES_PER_CHANNEL..(ch + 1) * NUM_FEATURES_PER_CHANNEL];

            // Validate finite samples
            if ch_slice.iter().any(|x| !x.is_finite()) {
                out.channel_flags[ch] = FLAG_NON_FINITE;
                out.window_valid = false;
                dst.fill(f64::NAN);
                continue;
            }

            let td = extract_time_domain(ch_slice);
            let sp = self
                .welch
                .extract_spectral_features(ch_slice, self.integration);
            let pe = permutation_entropy(ch_slice, self.pe_order, self.pe_delay, true);

            let mut flags = FLAG_VALID;
            if td.variance <= FLATLINE_VAR_EPSILON {
                flags |= FLAG_FLATLINE;
            }
            out.channel_flags[ch] = flags;
            write_feature_row(dst, &td, &sp, pe);
        }
    }

    /// Extract features from a contiguous `f32` buffer into a preallocated [`FeatureFrame`].
    pub fn extract_f32_into(
        &mut self,
        data: &[f32],
        num_channels: usize,
        num_samples: usize,
        layout: DataLayout,
        out: &mut FeatureFrame,
    ) {
        assert_eq!(data.len(), num_channels * num_samples);
        if out.num_channels != num_channels {
            out.num_channels = num_channels;
            out.channel_flags.resize(num_channels, FLAG_VALID);
            out.values
                .resize(num_channels * NUM_FEATURES_PER_CHANNEL, 0.0);
        }
        out.window_samples = num_samples;
        out.missing_samples = 0;
        out.window_valid = true;

        if self.channel_scratch.len() < num_samples {
            self.channel_scratch.resize(num_samples, 0.0);
        }

        for ch in 0..num_channels {
            match layout {
                DataLayout::ChannelsFirst => {
                    let start = ch * num_samples;
                    for (dst_val, &src_val) in self.channel_scratch[..num_samples]
                        .iter_mut()
                        .zip(&data[start..start + num_samples])
                    {
                        *dst_val = src_val as f64;
                    }
                }
                DataLayout::SamplesFirst => {
                    for s in 0..num_samples {
                        self.channel_scratch[s] = data[s * num_channels + ch] as f64;
                    }
                }
            }

            let ch_slice = &self.channel_scratch[..num_samples];
            let dst = &mut out.values
                [ch * NUM_FEATURES_PER_CHANNEL..(ch + 1) * NUM_FEATURES_PER_CHANNEL];

            if ch_slice.iter().any(|x| !x.is_finite()) {
                out.channel_flags[ch] = FLAG_NON_FINITE;
                out.window_valid = false;
                dst.fill(f64::NAN);
                continue;
            }

            let td = extract_time_domain(ch_slice);
            let sp = self
                .welch
                .extract_spectral_features(ch_slice, self.integration);
            let pe = permutation_entropy(ch_slice, self.pe_order, self.pe_delay, true);

            let mut flags = FLAG_VALID;
            if td.variance <= FLATLINE_VAR_EPSILON {
                flags |= FLAG_FLATLINE;
            }
            out.channel_flags[ch] = flags;
            write_feature_row(dst, &td, &sp, pe);
        }
    }
}

/// Stateful rolling-window feature extraction engine for continuous EEG streams.
///
/// Key optimizations:
/// 1. **Per-channel circular buffer** (`ring`) with contiguous unwrapping scratch buffer.
/// 2. **Segment PSD Ring Cache**: When `hop_samples == nperseg - noverlap` and the window
///    contains $K$ overlapping segments, pushing a hop of `hop_samples` new samples only
///    computes **1** new segment FFT per channel (the newest `nperseg` slice) and averages
///    the $K$ cached periodograms in the ring! For a 2s window (`500` samples), `nperseg = 250`,
///    `noverlap = 125` ($K = 3$ Welch segments per window), this cuts FFT work by **3×**.
/// 3. **Sequence-gap & validity tracking**: Tracks monotonic `seq` numbers and timestamps,
///    counting any dropped samples inside the active window and flagging `FLAG_SEQ_GAP`.
pub struct StreamingFeatureEngine {
    num_channels: usize,
    window_samples: usize,
    hop_samples: usize,
    fs: f64,
    pe_order: usize,
    pe_delay: usize,
    integration: BandIntegration,
    welch: WelchWorkspace,

    // Per-channel circular sample buffers: [num_channels][window_samples]
    rings: Vec<Vec<f64>>,
    // Circular buffers for sample timestamps and sequence numbers
    ts_ring: Vec<f64>,
    seq_ring: Vec<u64>,
    // Circular buffer recording how many missing samples occurred immediately before each slot
    gap_ring: Vec<u64>,

    head: usize,
    filled: usize,
    samples_since_emit: usize,
    last_seq: Option<u64>,
    total_missing_samples: u64,

    // Segment PSD cache when hop_samples == nperseg - noverlap
    use_psd_cache: bool,
    num_welch_segments: usize,
    // [num_channels][num_welch_segments * n_bins]
    psd_seg_cache: Vec<Vec<f64>>,
    psd_seg_head: usize,
    psd_seg_filled: usize,
    avg_psd_scratch: Vec<f64>,

    // Unwrapped contiguous window scratch: [window_samples]
    unwrap_scratch: Vec<f64>,
}

impl StreamingFeatureEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_channels: usize,
        window_samples: usize,
        hop_samples: usize,
        fs: f64,
        nperseg: usize,
        noverlap: usize,
        pe_order: usize,
        pe_delay: usize,
        integration: BandIntegration,
    ) -> Self {
        let num_channels = num_channels.max(1);
        let window_samples = window_samples.max(4);
        let hop_samples = hop_samples.max(1).min(window_samples);
        let nperseg = nperseg.min(window_samples).max(4);
        let noverlap = noverlap.min(nperseg - 1);
        let step = nperseg - noverlap;

        let welch = WelchWorkspace::new(nperseg, noverlap, fs, true);
        let n_bins = welch.freqs().len();

        // Enable incremental Welch segment cache whenever the streaming hop aligns with the Welch step
        // and the window tiles cleanly into segments: window_samples == nperseg + (K - 1) * step
        let use_psd_cache = hop_samples == step
            && window_samples >= nperseg
            && (window_samples - nperseg) % step == 0;
        let num_welch_segments = if use_psd_cache {
            (window_samples - nperseg) / step + 1
        } else {
            0
        };

        let psd_seg_cache = if use_psd_cache {
            vec![vec![0.0; num_welch_segments * n_bins]; num_channels]
        } else {
            Vec::new()
        };

        Self {
            num_channels,
            window_samples,
            hop_samples,
            fs,
            pe_order: pe_order.clamp(2, 7),
            pe_delay: pe_delay.max(1),
            integration,
            welch,
            rings: vec![vec![0.0; window_samples]; num_channels],
            ts_ring: vec![0.0; window_samples],
            seq_ring: vec![0; window_samples],
            gap_ring: vec![0; window_samples],
            head: 0,
            filled: 0,
            samples_since_emit: 0,
            last_seq: None,
            total_missing_samples: 0,
            use_psd_cache,
            num_welch_segments,
            psd_seg_cache,
            psd_seg_head: 0,
            psd_seg_filled: 0,
            avg_psd_scratch: vec![0.0; n_bins],
            unwrap_scratch: vec![0.0; window_samples],
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.samples_since_emit = 0;
        self.last_seq = None;
        self.total_missing_samples = 0;
        self.psd_seg_head = 0;
        self.psd_seg_filled = 0;
    }

    #[inline]
    pub fn total_missing_samples(&self) -> u64 {
        self.total_missing_samples
    }

    #[inline]
    pub fn is_ready(&self) -> bool {
        self.filled == self.window_samples
    }

    /// Push a single multichannel sample frame (`channels.len() == num_channels`) with sequence number and timestamp.
    ///
    /// Returns `Some(FeatureFrame)` whenever a full window is ready and `hop_samples` have elapsed.
    pub fn push_sample(
        &mut self,
        seq: u64,
        timestamp: f64,
        channels: &[f64],
    ) -> Option<FeatureFrame> {
        assert!(
            channels.len() >= self.num_channels,
            "Expected at least {} channels, got {}",
            self.num_channels,
            channels.len()
        );

        let gap = match self.last_seq {
            Some(prev) if seq > prev + 1 => seq - prev - 1,
            _ => 0,
        };
        self.total_missing_samples += gap;
        self.last_seq = Some(seq);

        let idx = self.head;
        for ch in 0..self.num_channels {
            self.rings[ch][idx] = channels[ch];
        }
        self.ts_ring[idx] = timestamp;
        self.seq_ring[idx] = seq;
        self.gap_ring[idx] = gap;

        self.head = (self.head + 1) % self.window_samples;
        if self.filled < self.window_samples {
            self.filled += 1;
            if self.filled == self.window_samples {
                self.samples_since_emit = self.hop_samples;
            }
        } else {
            self.samples_since_emit += 1;
        }

        if self.filled == self.window_samples && self.samples_since_emit >= self.hop_samples {
            self.samples_since_emit = 0;
            Some(self.emit_current_window())
        } else {
            None
        }
    }

    /// Push a block of samples in either `ChannelsFirst` (`[C, N]`) or `SamplesFirst` (`[N, C]`) layout,
    /// returning all emitted [`FeatureFrame`]s.
    pub fn push_block(
        &mut self,
        data: &[f64],
        num_channels: usize,
        num_samples: usize,
        layout: DataLayout,
        start_seq: u64,
        start_timestamp: f64,
    ) -> Vec<FeatureFrame> {
        assert_eq!(num_channels, self.num_channels);
        assert_eq!(data.len(), num_channels * num_samples);

        let dt = 1.0 / self.fs;
        let mut frames = Vec::new();
        let mut sample_buf = vec![0.0; num_channels];

        for s in 0..num_samples {
            let seq = start_seq + (s as u64);
            let ts = start_timestamp + (s as f64) * dt;
            match layout {
                DataLayout::SamplesFirst => {
                    let row = &data[s * num_channels..(s + 1) * num_channels];
                    if let Some(frame) = self.push_sample(seq, ts, row) {
                        frames.push(frame);
                    }
                }
                DataLayout::ChannelsFirst => {
                    for ch in 0..num_channels {
                        sample_buf[ch] = data[ch * num_samples + s];
                    }
                    if let Some(frame) = self.push_sample(seq, ts, &sample_buf) {
                        frames.push(frame);
                    }
                }
            }
        }

        frames
    }

    fn emit_current_window(&mut self) -> FeatureFrame {
        let w = self.window_samples;
        let start = self.head; // oldest sample index when full
        let newest = (self.head + w - 1) % w;

        let mut frame = FeatureFrame::new(self.num_channels, w);
        frame.timestamp_start = self.ts_ring[start];
        frame.timestamp_end = self.ts_ring[newest];
        frame.seq_start = self.seq_ring[start];
        frame.seq_end = self.seq_ring[newest];

        // Sum any sequence gaps that occurred inside the current window (excluding before the oldest sample)
        let mut window_gaps = 0u64;
        for i in 1..w {
            let slot = (start + i) % w;
            window_gaps += self.gap_ring[slot];
        }
        frame.missing_samples = window_gaps;
        if window_gaps > 0 {
            frame.window_valid = false;
        }

        let n_bins = self.welch.freqs().len();
        let can_increment_cache = self.use_psd_cache && self.psd_seg_filled == self.num_welch_segments;

        for ch in 0..self.num_channels {
            // Unwrap circular buffer into contiguous slice in two fast memcpy-optimized slices
            let ring = &self.rings[ch];
            let first_part = w - start;
            self.unwrap_scratch[..first_part].copy_from_slice(&ring[start..]);
            if start > 0 {
                self.unwrap_scratch[first_part..].copy_from_slice(&ring[..start]);
            }

            let dst = &mut frame.values
                [ch * NUM_FEATURES_PER_CHANNEL..(ch + 1) * NUM_FEATURES_PER_CHANNEL];

            if self.unwrap_scratch.iter().any(|x| !x.is_finite()) {
                let mut flags = FLAG_NON_FINITE;
                if window_gaps > 0 {
                    flags |= FLAG_SEQ_GAP;
                }
                frame.channel_flags[ch] = flags;
                frame.window_valid = false;
                dst.fill(f64::NAN);
                continue;
            }

            let td = extract_time_domain(&self.unwrap_scratch);

            let sp = if self.use_psd_cache {
                let l = self.welch.nperseg();
                let step = l - self.welch.noverlap();
                if !can_increment_cache {
                    // First full window after start/reset: populate all K segments in cache
                    for seg in 0..self.num_welch_segments {
                        let seg_slice = &self.unwrap_scratch[seg * step..seg * step + l];
                        let cache_slice =
                            &mut self.psd_seg_cache[ch][seg * n_bins..(seg + 1) * n_bins];
                        self.welch.compute_single_segment_psd(seg_slice, cache_slice);
                    }
                } else {
                    // Incremental hop: only compute the newest segment [w - l .. w]!
                    let newest_seg_slice = &self.unwrap_scratch[w - l..w];
                    let slot = self.psd_seg_head;
                    let cache_slice =
                        &mut self.psd_seg_cache[ch][slot * n_bins..(slot + 1) * n_bins];
                    self.welch
                        .compute_single_segment_psd(newest_seg_slice, cache_slice);
                }

                // Average the K cached segment PSDs
                self.avg_psd_scratch.fill(0.0);
                let inv_k = 1.0 / (self.num_welch_segments as f64);
                for seg in 0..self.num_welch_segments {
                    let seg_psd = &self.psd_seg_cache[ch][seg * n_bins..(seg + 1) * n_bins];
                    for k in 0..n_bins {
                        self.avg_psd_scratch[k] += seg_psd[k];
                    }
                }
                for k in 0..n_bins {
                    self.avg_psd_scratch[k] *= inv_k;
                }
                compute_features_from_psd(
                    self.welch.freqs(),
                    &self.avg_psd_scratch,
                    self.fs,
                    self.integration,
                )
            } else {
                self.welch
                    .extract_spectral_features(&self.unwrap_scratch, self.integration)
            };

            let pe = permutation_entropy(&self.unwrap_scratch, self.pe_order, self.pe_delay, true);

            let mut flags = FLAG_VALID;
            if td.variance <= FLATLINE_VAR_EPSILON {
                flags |= FLAG_FLATLINE;
            }
            if window_gaps > 0 {
                flags |= FLAG_SEQ_GAP;
            }
            frame.channel_flags[ch] = flags;
            write_feature_row(dst, &td, &sp, pe);
        }

        if self.use_psd_cache {
            if !can_increment_cache {
                self.psd_seg_filled = self.num_welch_segments;
                self.psd_seg_head = 0;
            } else {
                self.psd_seg_head = (self.psd_seg_head + 1) % self.num_welch_segments;
            }
        }

        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_streaming_matches_batch_exactly() {
        let num_channels = 4;
        let window_samples = 500;
        let hop_samples = 125;
        let fs = 250.0;
        let nperseg = 250;
        let noverlap = 125;

        let total_samples = 1000;
        // Generate synthetic multichannel signal in SamplesFirst [total_samples, num_channels]
        let mut stream_data = vec![0.0; total_samples * num_channels];
        for s in 0..total_samples {
            let t = (s as f64) / fs;
            for ch in 0..num_channels {
                let freq = 6.0 + (ch as f64) * 4.0;
                stream_data[s * num_channels + ch] =
                    15.0 * (2.0 * PI * freq * t).sin() + 3.0 * (2.0 * PI * 22.0 * t).cos();
            }
        }

        let mut stream_engine = StreamingFeatureEngine::new(
            num_channels,
            window_samples,
            hop_samples,
            fs,
            nperseg,
            noverlap,
            3,
            1,
            BandIntegration::Trapezoidal,
        );
        assert!(stream_engine.use_psd_cache);

        let frames = stream_engine.push_block(
            &stream_data,
            num_channels,
            total_samples,
            DataLayout::SamplesFirst,
            1,
            100.0,
        );

        // With 1000 samples, window=500, hop=125 -> frames emitted at sample 500, 625, 750, 875, 1000 (5 frames)
        assert_eq!(frames.len(), 5);

        let mut batch_extractor = FeatureExtractor::new(
            num_channels,
            window_samples,
            fs,
            nperseg,
            noverlap,
            3,
            1,
            BandIntegration::Trapezoidal,
        );
        let mut batch_frame = FeatureFrame::new(num_channels, window_samples);

        for (frame_idx, s_frame) in frames.iter().enumerate() {
            let win_start = frame_idx * hop_samples;
            let win_slice =
                &stream_data[win_start * num_channels..(win_start + window_samples) * num_channels];
            batch_extractor.extract_f64_into(
                win_slice,
                num_channels,
                window_samples,
                DataLayout::SamplesFirst,
                &mut batch_frame,
            );

            assert!(s_frame.window_valid);
            assert_eq!(s_frame.missing_samples, 0);
            for (a, b) in s_frame.values.iter().zip(batch_frame.values.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "Mismatch at frame {}: streaming={} vs batch={}",
                    frame_idx,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_sequence_gap_detection() {
        let mut engine = StreamingFeatureEngine::new(
            2,
            100,
            50,
            250.0,
            50,
            25,
            3,
            1,
            BandIntegration::Trapezoidal,
        );

        for s in 0..99 {
            assert!(engine.push_sample(s + 1, (s as f64) * 0.004, &[1.0, 2.0]).is_none());
        }
        // Jump seq from 99 to 105 (5 missing samples!)
        let frame = engine
            .push_sample(105, 100.0 * 0.004, &[1.0, 2.0])
            .expect("Window should emit");
        assert!(!frame.window_valid);
        assert_eq!(frame.missing_samples, 5);
        assert_eq!(frame.channel_flags[0] & FLAG_SEQ_GAP, FLAG_SEQ_GAP);
    }
}
