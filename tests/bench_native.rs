//! Native Rust benchmark & zero-allocation verification suite for `pieeg-features`.
//!
//! Run with:
//! ```bash
//! cargo test --no-default-features --release --test bench_native -- --nocapture
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use pieeg_features::engine::{
    DataLayout, FeatureExtractor, FeatureFrame, StreamingFeatureEngine,
};
use pieeg_features::spectral::BandIntegration;

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_alloc_counters() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
}

fn read_alloc_counters() -> (usize, usize) {
    (
        ALLOC_COUNT.load(Ordering::SeqCst),
        ALLOC_BYTES.load(Ordering::SeqCst),
    )
}

fn percentile_us(sorted_ns: &[u128], p: f64) -> f64 {
    let idx = (((sorted_ns.len() - 1) as f64) * p).round() as usize;
    (sorted_ns[idx] as f64) / 1000.0
}

#[test]
fn verify_zero_steady_state_allocations_and_benchmark() {
    let fs = 250.0;
    let configs = [
        (8usize, 250usize, 125usize, 62usize),   // 8 ch, 1.0s window
        (8usize, 500usize, 250usize, 125usize),  // 8 ch, 2.0s window
        (16usize, 250usize, 125usize, 62usize),  // 16 ch, 1.0s window
        (16usize, 500usize, 250usize, 125usize), // 16 ch, 2.0s window (PiEEG-16 default)
        (16usize, 1000usize, 250usize, 125usize),// 16 ch, 4.0s window
        (24usize, 250usize, 125usize, 62usize),  // 24 ch, 1.0s window
        (24usize, 500usize, 250usize, 125usize), // 24 ch, 2.0s window (PiEEG-24 default)
        (24usize, 1000usize, 250usize, 125usize),// 24 ch, 4.0s window
    ];

    println!("\n==========================================================================================");
    println!("  pieeg-features Native Rust Benchmark (fs = 250 Hz, all 22 features/ch)");
    println!("==========================================================================================");
    println!(
        "{:<8} {:<10} {:<10} {:<12} {:<12} {:<12} {:<12} {:<10}",
        "Ch", "Win (smp)", "Layout", "Median (us)", "p95 (us)", "p99 (us)", "StreamHop(us)", "Allocs/call"
    );
    println!("------------------------------------------------------------------------------------------");

    let iterations = 1000usize;

    for &(n_ch, n_win, nperseg, noverlap) in &configs {
        // Generate synthetic EEG window
        let mut data_cs = vec![0.0f64; n_ch * n_win];
        let mut data_sc = vec![0.0f64; n_ch * n_win];
        for ch in 0..n_ch {
            let f1 = 6.0 + (ch as f64) * 1.5;
            for s in 0..n_win {
                let t = (s as f64) / fs;
                let v = 15.0 * (2.0 * PI * f1 * t).sin()
                    + 8.0 * (2.0 * PI * 10.5 * t).cos()
                    + 4.0 * (2.0 * PI * 22.0 * t).sin();
                data_cs[ch * n_win + s] = v;
                data_sc[s * n_ch + ch] = v;
            }
        }

        let mut extractor = FeatureExtractor::new(
            n_ch,
            n_win,
            fs,
            nperseg,
            noverlap,
            3,
            1,
            BandIntegration::Trapezoidal,
        );
        let mut out_frame = FeatureFrame::new(n_ch, n_win);

        // Warmup
        for _ in 0..50 {
            extractor.extract_f64_into(&data_cs, n_ch, n_win, DataLayout::ChannelsFirst, &mut out_frame);
        }

        // Measure steady-state allocations & latency for ChannelsFirst
        let mut times_cs = Vec::with_capacity(iterations);
        reset_alloc_counters();
        for _ in 0..iterations {
            let t0 = Instant::now();
            extractor.extract_f64_into(&data_cs, n_ch, n_win, DataLayout::ChannelsFirst, &mut out_frame);
            times_cs.push(t0.elapsed().as_nanos());
        }
        let (alloc_cnt_cs, alloc_bytes_cs) = read_alloc_counters();
        assert_eq!(
            alloc_cnt_cs, 0,
            "Expected 0 heap allocations in steady-state extract_f64_into (ChannelsFirst), got {} ({} bytes)",
            alloc_cnt_cs, alloc_bytes_cs
        );

        // Measure SamplesFirst
        let mut times_sc = Vec::with_capacity(iterations);
        reset_alloc_counters();
        for _ in 0..iterations {
            let t0 = Instant::now();
            extractor.extract_f64_into(&data_sc, n_ch, n_win, DataLayout::SamplesFirst, &mut out_frame);
            times_sc.push(t0.elapsed().as_nanos());
        }
        let (alloc_cnt_sc, alloc_bytes_sc) = read_alloc_counters();
        assert_eq!(
            alloc_cnt_sc, 0,
            "Expected 0 heap allocations in steady-state extract_f64_into (SamplesFirst), got {} ({} bytes)",
            alloc_cnt_sc, alloc_bytes_sc
        );

        // Measure incremental StreamingFeatureEngine per-hop latency (hop = nperseg - noverlap)
        let hop = nperseg - noverlap;
        let mut stream_eng = StreamingFeatureEngine::new(
            n_ch,
            n_win,
            hop,
            fs,
            nperseg,
            noverlap,
            3,
            1,
            BandIntegration::Trapezoidal,
        );
        // Prime the streaming engine with n_win samples
        let mut seq = 1u64;
        for s in 0..n_win {
            let _ = stream_eng.push_sample(seq, (seq as f64) / fs, &data_sc[s * n_ch..(s + 1) * n_ch]);
            seq += 1;
        }
        let mut times_stream = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            for s in 0..hop {
                let row = &data_sc[(s % n_win) * n_ch..((s % n_win) + 1) * n_ch];
                let _ = stream_eng.push_sample(seq, (seq as f64) / fs, row);
                seq += 1;
            }
            times_stream.push(t0.elapsed().as_nanos());
        }

        times_cs.sort_unstable();
        times_sc.sort_unstable();
        times_stream.sort_unstable();

        println!(
            "{:<8} {:<10} {:<10} {:<12.1} {:<12.1} {:<12.1} {:<12.1} {:<10}",
            n_ch,
            n_win,
            "[C, N]",
            percentile_us(&times_cs, 0.50),
            percentile_us(&times_cs, 0.95),
            percentile_us(&times_cs, 0.99),
            percentile_us(&times_stream, 0.50),
            alloc_cnt_cs
        );
        println!(
            "{:<8} {:<10} {:<10} {:<12.1} {:<12.1} {:<12.1} {:<12.1} {:<10}",
            n_ch,
            n_win,
            "[N, C]",
            percentile_us(&times_sc, 0.50),
            percentile_us(&times_sc, 0.95),
            percentile_us(&times_sc, 0.99),
            percentile_us(&times_stream, 0.50),
            alloc_cnt_sc
        );
    }
    println!("==========================================================================================\n");
}
