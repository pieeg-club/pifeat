//! `pieeg-features` — Zero-dependency, high-performance EEG feature extraction engine.
//!
//! Works standalone as a pure-Rust library (`default-features = false`) and as a
//! native Python extension (`features = ["python"]`) using CPython's Buffer Protocol.

pub mod engine;
pub mod nonlinear;
#[cfg(feature = "python")]
mod py_buffer;
pub mod spectral;
pub mod time_domain;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

#[cfg(feature = "python")]
use crate::engine::{
    DataLayout, FeatureExtractor, FeatureFrame, StreamingFeatureEngine, FEATURE_NAMES,
    FLAG_FLATLINE, FLAG_NON_FINITE, FLAG_SEQ_GAP, FLAG_VALID, NUM_FEATURES_PER_CHANNEL,
};
#[cfg(feature = "python")]
use crate::py_buffer::{BorrowedSignalBuffer, PyContiguousBuffer};
#[cfg(feature = "python")]
use crate::spectral::{BandIntegration, WelchWorkspace};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "python")]
fn parse_layout(layout: &str) -> PyResult<DataLayout> {
    match layout.trim().to_ascii_lowercase().as_str() {
        "channels_first" | "c_s" | "ch_first" | "[channels, samples]" => {
            Ok(DataLayout::ChannelsFirst)
        }
        "samples_first" | "s_c" | "time_first" | "[samples, channels]" => {
            Ok(DataLayout::SamplesFirst)
        }
        other => Err(PyValueError::new_err(format!(
            "Invalid layout '{other}'. Expected 'channels_first' or 'samples_first'."
        ))),
    }
}

#[cfg(feature = "python")]
fn parse_integration(mode: &str) -> PyResult<BandIntegration> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "trapezoid" | "trapezoidal" | "trapz" | "integral" => Ok(BandIntegration::Trapezoidal),
        "mean" | "bin_mean" | "legacy" => Ok(BandIntegration::BinMean),
        other => Err(PyValueError::new_err(format!(
            "Invalid band integration mode '{other}'. Expected 'trapezoid' or 'bin_mean'."
        ))),
    }
}

#[cfg(feature = "python")]
fn extract_1d_f64_signal(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(buf) = PyContiguousBuffer::acquire(obj) {
        let decoded = buf.decode(Some(1), DataLayout::ChannelsFirst)?;
        return Ok(match decoded {
            BorrowedSignalBuffer::F64 { slice, .. } => slice.to_vec(),
            BorrowedSignalBuffer::F32 { slice, .. } => slice.iter().map(|&v| v as f64).collect(),
        });
    }
    obj.extract::<Vec<f64>>()
}

/// Python wrapper for an extracted multichannel [`FeatureFrame`].
#[cfg(feature = "python")]
#[pyclass(name = "FeatureFrame")]
#[derive(Clone)]
pub struct PyFeatureFrame {
    inner: FeatureFrame,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFeatureFrame {
    #[getter]
    fn num_channels(&self) -> usize {
        self.inner.num_channels
    }

    #[getter]
    fn window_samples(&self) -> usize {
        self.inner.window_samples
    }

    #[getter]
    fn timestamp_start(&self) -> f64 {
        self.inner.timestamp_start
    }

    #[getter]
    fn timestamp_end(&self) -> f64 {
        self.inner.timestamp_end
    }

    #[getter]
    fn seq_start(&self) -> u64 {
        self.inner.seq_start
    }

    #[getter]
    fn seq_end(&self) -> u64 {
        self.inner.seq_end
    }

    #[getter]
    fn missing_samples(&self) -> u64 {
        self.inner.missing_samples
    }

    #[getter]
    fn window_valid(&self) -> bool {
        self.inner.window_valid
    }

    #[getter]
    fn channel_flags(&self) -> Vec<u32> {
        self.inner.channel_flags.clone()
    }

    /// Return the flat row-major `[num_channels * 22]` feature values.
    fn values_flat(&self) -> Vec<f64> {
        self.inner.values.clone()
    }

    /// Return `[num_channels][22]` nested list of feature values.
    fn matrix(&self) -> Vec<Vec<f64>> {
        (0..self.inner.num_channels)
            .map(|ch| self.inner.channel_features(ch).to_vec())
            .collect()
    }

    /// Return the 22 features for channel `ch` as a list.
    fn channel(&self, ch: usize) -> PyResult<Vec<f64>> {
        if ch >= self.inner.num_channels {
            return Err(PyValueError::new_err(format!(
                "Channel index {ch} out of range (num_channels = {})",
                self.inner.num_channels
            )));
        }
        Ok(self.inner.channel_features(ch).to_vec())
    }

    /// Return a dict mapping each feature name to a list of per-channel values `[ch0, ch1, ...]`.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        let n_ch = self.inner.num_channels;
        for (f_idx, &name) in FEATURE_NAMES.iter().enumerate() {
            let col: Vec<f64> = (0..n_ch)
                .map(|ch| self.inner.values[ch * NUM_FEATURES_PER_CHANNEL + f_idx])
                .collect();
            dict.set_item(name, col)?;
        }
        dict.set_item("window_valid", self.inner.window_valid)?;
        dict.set_item("missing_samples", self.inner.missing_samples)?;
        dict.set_item("channel_flags", self.inner.channel_flags.clone())?;
        dict.set_item("timestamp_start", self.inner.timestamp_start)?;
        dict.set_item("timestamp_end", self.inner.timestamp_end)?;
        dict.set_item("seq_start", self.inner.seq_start)?;
        dict.set_item("seq_end", self.inner.seq_end)?;
        Ok(dict.into())
    }
}

/// Reusable multichannel batch feature extractor exposed to Python.
#[cfg(feature = "python")]
#[pyclass(name = "FeatureExtractor")]
pub struct PyFeatureExtractor {
    inner: FeatureExtractor,
    frame_scratch: FeatureFrame,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFeatureExtractor {
    #[new]
    #[pyo3(signature = (
        num_channels=16,
        window_samples=500,
        fs=250.0,
        nperseg=250,
        noverlap=125,
        pe_order=3,
        pe_delay=1,
        integration="trapezoid"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_channels: usize,
        window_samples: usize,
        fs: f64,
        nperseg: usize,
        noverlap: usize,
        pe_order: usize,
        pe_delay: usize,
        integration: &str,
    ) -> PyResult<Self> {
        let integ = parse_integration(integration)?;
        Ok(Self {
            inner: FeatureExtractor::new(
                num_channels,
                window_samples,
                fs,
                nperseg,
                noverlap,
                pe_order,
                pe_delay,
                integ,
            ),
            frame_scratch: FeatureFrame::new(num_channels, window_samples),
        })
    }

    #[getter]
    fn num_channels(&self) -> usize {
        self.inner.num_channels()
    }

    #[getter]
    fn window_samples(&self) -> usize {
        self.inner.window_samples()
    }

    #[getter]
    fn fs(&self) -> f64 {
        self.inner.fs()
    }

    #[getter]
    fn freqs(&self) -> Vec<f64> {
        self.inner.freqs().to_vec()
    }

    #[staticmethod]
    fn feature_names() -> Vec<&'static str> {
        FEATURE_NAMES.to_vec()
    }

    /// Zero-copy extraction from any Python object supporting the Buffer Protocol
    /// (`memoryview`, `array.array('d'|'f')`, `bytearray`, `numpy.ndarray`).
    ///
    /// Releases the GIL during native Rust computation.
    #[pyo3(signature = (buffer, layout="channels_first"))]
    fn extract(&mut self, py: Python<'_>, buffer: &Bound<'_, PyAny>, layout: &str) -> PyResult<PyFeatureFrame> {
        let data_layout = parse_layout(layout)?;
        let expected_ch = self.inner.num_channels();
        let acquired = PyContiguousBuffer::acquire(buffer)?;
        let decoded = acquired.decode(Some(expected_ch), data_layout)?;

        let inner_ptr = &mut self.inner as *mut FeatureExtractor;
        let frame_ptr = &mut self.frame_scratch as *mut FeatureFrame;

        match decoded {
            BorrowedSignalBuffer::F64 {
                slice,
                num_channels,
                num_samples,
                layout,
            } => {
                // SAFETY: `acquired` holds the Py_buffer lock for the duration of `allow_threads`,
                // and `self` is uniquely borrowed by `&mut self`.
                let slice_ptr = slice.as_ptr() as usize;
                let slice_len = slice.len();
                let inner_addr = inner_ptr as usize;
                let frame_addr = frame_ptr as usize;
                py.allow_threads(move || unsafe {
                    let s = std::slice::from_raw_parts(slice_ptr as *const f64, slice_len);
                    let ext = &mut *(inner_addr as *mut FeatureExtractor);
                    let frm = &mut *(frame_addr as *mut FeatureFrame);
                    ext.extract_f64_into(s, num_channels, num_samples, layout, frm);
                });
            }
            BorrowedSignalBuffer::F32 {
                slice,
                num_channels,
                num_samples,
                layout,
            } => {
                let slice_ptr = slice.as_ptr() as usize;
                let slice_len = slice.len();
                let inner_addr = inner_ptr as usize;
                let frame_addr = frame_ptr as usize;
                py.allow_threads(move || unsafe {
                    let s = std::slice::from_raw_parts(slice_ptr as *const f32, slice_len);
                    let ext = &mut *(inner_addr as *mut FeatureExtractor);
                    let frm = &mut *(frame_addr as *mut FeatureFrame);
                    ext.extract_f32_into(s, num_channels, num_samples, layout, frm);
                });
            }
        }

        Ok(PyFeatureFrame {
            inner: self.frame_scratch.clone(),
        })
    }

    /// Zero-copy extraction directly into a caller-provided writable `f64` buffer (`memoryview` / `array('d')` / `ndarray`)
    /// of length `num_channels * 22`. Returns `(window_valid, missing_samples)`.
    ///
    /// Performs **zero allocations across both Rust and Python** on the hot path!
    #[pyo3(signature = (input_buffer, output_buffer, layout="channels_first"))]
    fn extract_into(
        &mut self,
        py: Python<'_>,
        input_buffer: &Bound<'_, PyAny>,
        output_buffer: &Bound<'_, PyAny>,
        layout: &str,
    ) -> PyResult<bool> {
        let data_layout = parse_layout(layout)?;
        let expected_ch = self.inner.num_channels();
        let in_buf = PyContiguousBuffer::acquire(input_buffer)?;
        let decoded = in_buf.decode(Some(expected_ch), data_layout)?;

        // Acquire writable output buffer
        let mut out_view: pyo3::ffi::Py_buffer = unsafe { std::mem::zeroed() };
        let flags = pyo3::ffi::PyBUF_WRITABLE | pyo3::ffi::PyBUF_FORMAT | pyo3::ffi::PyBUF_C_CONTIGUOUS;
        let rc = unsafe { pyo3::ffi::PyObject_GetBuffer(output_buffer.as_ptr(), &mut out_view, flags) };
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
        let out_elems = (out_view.len as usize) / (out_view.itemsize.max(1) as usize);
        let required = expected_ch * NUM_FEATURES_PER_CHANNEL;
        if out_view.itemsize != 8 || out_elems < required {
            unsafe { pyo3::ffi::PyBuffer_Release(&mut out_view) };
            return Err(PyValueError::new_err(format!(
                "output_buffer must be a writable float64 buffer with at least {required} elements"
            )));
        }

        let out_ptr = out_view.buf as usize;
        let inner_addr = (&mut self.inner as *mut FeatureExtractor) as usize;
        let frame_addr = (&mut self.frame_scratch as *mut FeatureFrame) as usize;

        match decoded {
            BorrowedSignalBuffer::F64 {
                slice,
                num_channels,
                num_samples,
                layout,
            } => {
                let slice_ptr = slice.as_ptr() as usize;
                let slice_len = slice.len();
                py.allow_threads(move || unsafe {
                    let s = std::slice::from_raw_parts(slice_ptr as *const f64, slice_len);
                    let ext = &mut *(inner_addr as *mut FeatureExtractor);
                    let frm = &mut *(frame_addr as *mut FeatureFrame);
                    ext.extract_f64_into(s, num_channels, num_samples, layout, frm);
                    let dst = std::slice::from_raw_parts_mut(out_ptr as *mut f64, required);
                    dst.copy_from_slice(&frm.values[..required]);
                });
            }
            BorrowedSignalBuffer::F32 {
                slice,
                num_channels,
                num_samples,
                layout,
            } => {
                let slice_ptr = slice.as_ptr() as usize;
                let slice_len = slice.len();
                py.allow_threads(move || unsafe {
                    let s = std::slice::from_raw_parts(slice_ptr as *const f32, slice_len);
                    let ext = &mut *(inner_addr as *mut FeatureExtractor);
                    let frm = &mut *(frame_addr as *mut FeatureFrame);
                    ext.extract_f32_into(s, num_channels, num_samples, layout, frm);
                    let dst = std::slice::from_raw_parts_mut(out_ptr as *mut f64, required);
                    dst.copy_from_slice(&frm.values[..required]);
                });
            }
        }

        unsafe { pyo3::ffi::PyBuffer_Release(&mut out_view) };
        Ok(self.frame_scratch.window_valid)
    }
}

/// Stateful streaming rolling-window feature engine exposed to Python.
#[cfg(feature = "python")]
#[pyclass(name = "StreamingFeatureEngine")]
pub struct PyStreamingFeatureEngine {
    inner: StreamingFeatureEngine,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyStreamingFeatureEngine {
    #[new]
    #[pyo3(signature = (
        num_channels=16,
        window_samples=500,
        hop_samples=125,
        fs=250.0,
        nperseg=250,
        noverlap=125,
        pe_order=3,
        pe_delay=1,
        integration="trapezoid"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_channels: usize,
        window_samples: usize,
        hop_samples: usize,
        fs: f64,
        nperseg: usize,
        noverlap: usize,
        pe_order: usize,
        pe_delay: usize,
        integration: &str,
    ) -> PyResult<Self> {
        let integ = parse_integration(integration)?;
        Ok(Self {
            inner: StreamingFeatureEngine::new(
                num_channels,
                window_samples,
                hop_samples,
                fs,
                nperseg,
                noverlap,
                pe_order,
                pe_delay,
                integ,
            ),
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    #[getter]
    fn total_missing_samples(&self) -> u64 {
        self.inner.total_missing_samples()
    }

    /// Push one multichannel sample `(seq, timestamp, channels)` into the rolling window.
    /// Returns a `FeatureFrame` when a hop completes, or `None`.
    fn push_sample(
        &mut self,
        seq: u64,
        timestamp: f64,
        channels: Vec<f64>,
    ) -> PyResult<Option<PyFeatureFrame>> {
        Ok(self
            .inner
            .push_sample(seq, timestamp, &channels)
            .map(|inner| PyFeatureFrame { inner }))
    }

    /// Push a contiguous block buffer of samples and return a list of emitted `FeatureFrame`s.
    #[pyo3(signature = (buffer, start_seq=1, start_timestamp=0.0, layout="samples_first"))]
    fn push_block(
        &mut self,
        py: Python<'_>,
        buffer: &Bound<'_, PyAny>,
        start_seq: u64,
        start_timestamp: f64,
        layout: &str,
    ) -> PyResult<Vec<PyFeatureFrame>> {
        let data_layout = parse_layout(layout)?;
        let acquired = PyContiguousBuffer::acquire(buffer)?;
        // Retrieve num_channels by reading the engine field via decoded buffer check
        let decoded = acquired.decode(None, data_layout)?;

        let frames = match decoded {
            BorrowedSignalBuffer::F64 {
                slice,
                num_channels,
                num_samples,
                layout,
            } => {
                let slice_ptr = slice.as_ptr() as usize;
                let slice_len = slice.len();
                let inner_addr = (&mut self.inner as *mut StreamingFeatureEngine) as usize;
                py.allow_threads(move || unsafe {
                    let s = std::slice::from_raw_parts(slice_ptr as *const f64, slice_len);
                    let eng = &mut *(inner_addr as *mut StreamingFeatureEngine);
                    eng.push_block(s, num_channels, num_samples, layout, start_seq, start_timestamp)
                })
            }
            BorrowedSignalBuffer::F32 {
                slice,
                num_channels,
                num_samples,
                layout,
            } => {
                let converted: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
                let inner_addr = (&mut self.inner as *mut StreamingFeatureEngine) as usize;
                py.allow_threads(move || unsafe {
                    let eng = &mut *(inner_addr as *mut StreamingFeatureEngine);
                    eng.push_block(
                        &converted,
                        num_channels,
                        num_samples,
                        layout,
                        start_seq,
                        start_timestamp,
                    )
                })
            }
        };

        Ok(frames
            .into_iter()
            .map(|inner| PyFeatureFrame { inner })
            .collect())
    }
}

// ── Standalone convenience functions for 1D signals ───────────────────────

#[cfg(feature = "python")]
#[pyfunction]
fn compute_rms(signal: &Bound<'_, PyAny>) -> PyResult<f64> {
    let s = extract_1d_f64_signal(signal)?;
    Ok(time_domain::rms(&s))
}

#[cfg(feature = "python")]
#[pyfunction]
fn compute_variance(signal: &Bound<'_, PyAny>) -> PyResult<f64> {
    let s = extract_1d_f64_signal(signal)?;
    Ok(time_domain::variance(&s))
}

#[cfg(feature = "python")]
#[pyfunction]
fn compute_line_length(signal: &Bound<'_, PyAny>) -> PyResult<f64> {
    let s = extract_1d_f64_signal(signal)?;
    Ok(time_domain::line_length(&s))
}

#[cfg(feature = "python")]
#[pyfunction]
fn compute_zero_crossing_rate(signal: &Bound<'_, PyAny>) -> PyResult<f64> {
    let s = extract_1d_f64_signal(signal)?;
    Ok(time_domain::zero_crossing_rate(&s))
}

#[cfg(feature = "python")]
#[pyfunction]
fn compute_hjorth(signal: &Bound<'_, PyAny>) -> PyResult<(f64, f64, f64)> {
    let s = extract_1d_f64_signal(signal)?;
    let h = time_domain::hjorth_parameters(&s);
    Ok((h.activity, h.mobility, h.complexity))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (signal, order=3, delay=1, normalize=true))]
fn compute_permutation_entropy(
    signal: &Bound<'_, PyAny>,
    order: usize,
    delay: usize,
    normalize: bool,
) -> PyResult<f64> {
    let s = extract_1d_f64_signal(signal)?;
    Ok(nonlinear::permutation_entropy(&s, order, delay, normalize))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (signal, fs=250.0, nperseg=250, noverlap=125, detrend_constant=true))]
fn compute_welch_psd(
    signal: &Bound<'_, PyAny>,
    fs: f64,
    nperseg: usize,
    noverlap: usize,
    detrend_constant: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let s = extract_1d_f64_signal(signal)?;
    let mut ws = WelchWorkspace::new(nperseg, noverlap, fs, detrend_constant);
    let psd = ws.compute_psd(&s).to_vec();
    let freqs = ws.freqs().to_vec();
    Ok((freqs, psd))
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (signal, fs=250.0, nperseg=250, noverlap=125, integration="trapezoid"))]
fn compute_band_powers(
    py: Python<'_>,
    signal: &Bound<'_, PyAny>,
    fs: f64,
    nperseg: usize,
    noverlap: usize,
    integration: &str,
) -> PyResult<PyObject> {
    let s = extract_1d_f64_signal(signal)?;
    let integ = parse_integration(integration)?;
    let mut ws = WelchWorkspace::new(nperseg, noverlap, fs, true);
    let sp = ws.extract_spectral_features(&s, integ);

    let dict = PyDict::new_bound(py);
    dict.set_item("delta_power", sp.delta_power)?;
    dict.set_item("theta_power", sp.theta_power)?;
    dict.set_item("alpha_power", sp.alpha_power)?;
    dict.set_item("beta_power", sp.beta_power)?;
    dict.set_item("gamma_power", sp.gamma_power)?;
    dict.set_item("total_power", sp.total_power)?;
    dict.set_item("rel_delta", sp.rel_delta)?;
    dict.set_item("rel_theta", sp.rel_theta)?;
    dict.set_item("rel_alpha", sp.rel_alpha)?;
    dict.set_item("rel_beta", sp.rel_beta)?;
    dict.set_item("rel_gamma", sp.rel_gamma)?;
    dict.set_item("theta_beta_ratio", sp.theta_beta_ratio)?;
    dict.set_item("alpha_beta_ratio", sp.alpha_beta_ratio)?;
    dict.set_item("spectral_entropy", sp.spectral_entropy)?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pymodule]
fn pieeg_features(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add("NUM_FEATURES_PER_CHANNEL", NUM_FEATURES_PER_CHANNEL)?;
    m.add("FLAG_VALID", FLAG_VALID)?;
    m.add("FLAG_NON_FINITE", FLAG_NON_FINITE)?;
    m.add("FLAG_FLATLINE", FLAG_FLATLINE)?;
    m.add("FLAG_SEQ_GAP", FLAG_SEQ_GAP)?;

    m.add_class::<PyFeatureFrame>()?;
    m.add_class::<PyFeatureExtractor>()?;
    m.add_class::<PyStreamingFeatureEngine>()?;

    m.add_function(wrap_pyfunction!(compute_rms, m)?)?;
    m.add_function(wrap_pyfunction!(compute_variance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_line_length, m)?)?;
    m.add_function(wrap_pyfunction!(compute_zero_crossing_rate, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hjorth, m)?)?;
    m.add_function(wrap_pyfunction!(compute_permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(compute_welch_psd, m)?)?;
    m.add_function(wrap_pyfunction!(compute_band_powers, m)?)?;
    Ok(())
}
