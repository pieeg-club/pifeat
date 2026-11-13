//! Zero-copy CPython Buffer Protocol (`Py_buffer`) reader for contiguous `f64` (`"d"`) and `f32` (`"f"`)
//! 1D and 2D buffers (such as `memoryview`, `array.array`, `bytearray`, `bytes`, or `numpy.ndarray`
//! without importing or depending on `numpy` at build or runtime!).

use std::os::raw::c_int;
use std::ptr;

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;

use crate::engine::DataLayout;

/// Decoded shape and element type of a borrowed Python buffer.
pub enum BorrowedSignalBuffer<'a> {
    F64 {
        slice: &'a [f64],
        num_channels: usize,
        num_samples: usize,
        layout: DataLayout,
    },
    F32 {
        slice: &'a [f32],
        num_channels: usize,
        num_samples: usize,
        layout: DataLayout,
    },
}

/// RAII guard for CPython's `Py_buffer` that releases the buffer on `Drop`.
pub struct PyContiguousBuffer {
    view: ffi::Py_buffer,
}

impl PyContiguousBuffer {
    /// Acquire a C-contiguous (`PyBUF_FORMAT | PyBUF_C_CONTIGUOUS`) buffer from `obj`.
    pub fn acquire(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut view: ffi::Py_buffer = unsafe { std::mem::zeroed() };
        let flags: c_int = ffi::PyBUF_FORMAT | ffi::PyBUF_C_CONTIGUOUS;
        let rc = unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags) };
        if rc != 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        if view.buf.is_null() {
            unsafe { ffi::PyBuffer_Release(&mut view) };
            return Err(PyBufferError::new_err("Buffer pointer is null"));
        }
        Ok(Self { view })
    }

    /// Interpret the buffer as a 1D signal (`[num_samples]`) or 2D multichannel matrix
    /// (`[num_channels, num_samples]` for `ChannelsFirst`, or `[num_samples, num_channels]` for `SamplesFirst`).
    ///
    /// If the Python buffer is 1D (like `array.array('d', ...)`), `expected_channels` is used to
    /// infer the second dimension (`total_len / expected_channels`).
    /// If the Python buffer is 2D (like a 2D `memoryview` or `ndarray`), its `shape[0]` and `shape[1]`
    /// are read directly!
    pub fn decode<'a>(
        &'a self,
        expected_channels: Option<usize>,
        layout: DataLayout,
    ) -> PyResult<BorrowedSignalBuffer<'a>> {
        let itemsize = self.view.itemsize as usize;
        if itemsize == 0 {
            return Err(PyBufferError::new_err("Invalid itemsize 0"));
        }
        let total_bytes = self.view.len as usize;
        let total_elems = total_bytes / itemsize;

        let fmt_char = if self.view.format.is_null() {
            b'B'
        } else {
            unsafe {
                let cstr = std::ffi::CStr::from_ptr(self.view.format).to_bytes();
                // Strip standard endianness prefixes ('@', '=', '<', '>') if present
                *cstr
                    .iter()
                    .find(|&&b| b != b'@' && b != b'=' && b != b'<' && b != b'>' && b != b'!')
                    .unwrap_or(&b'B')
            }
        };

        let ndim = self.view.ndim as usize;
        let (num_channels, num_samples) = if ndim == 2 && !self.view.shape.is_null() {
            let d0 = unsafe { (*self.view.shape.add(0)) as usize };
            let d1 = unsafe { (*self.view.shape.add(1)) as usize };
            match layout {
                DataLayout::ChannelsFirst => (d0, d1),
                DataLayout::SamplesFirst => (d1, d0),
            }
        } else if ndim <= 1 {
            let n_ch = expected_channels.unwrap_or(1).max(1);
            if total_elems % n_ch != 0 {
                return Err(PyValueError::new_err(format!(
                    "1D buffer length ({total_elems}) is not divisible by num_channels ({n_ch})"
                )));
            }
            (n_ch, total_elems / n_ch)
        } else {
            return Err(PyValueError::new_err(format!(
                "Expected 1D or 2D buffer, got {ndim}D"
            )));
        };

        if let Some(exp_ch) = expected_channels {
            if num_channels != exp_ch {
                return Err(PyValueError::new_err(format!(
                    "Buffer channel count ({num_channels}) does not match configured num_channels ({exp_ch})"
                )));
            }
        }

        match (fmt_char, itemsize) {
            (b'd', 8) => {
                if (self.view.buf as usize) % std::mem::align_of::<f64>() != 0 {
                    return Err(PyBufferError::new_err(
                        "Buffer pointer is not 8-byte aligned for f64",
                    ));
                }
                let slice =
                    unsafe { std::slice::from_raw_parts(self.view.buf as *const f64, total_elems) };
                Ok(BorrowedSignalBuffer::F64 {
                    slice,
                    num_channels,
                    num_samples,
                    layout,
                })
            }
            (b'f', 4) => {
                if (self.view.buf as usize) % std::mem::align_of::<f32>() != 0 {
                    return Err(PyBufferError::new_err(
                        "Buffer pointer is not 4-byte aligned for f32",
                    ));
                }
                let slice =
                    unsafe { std::slice::from_raw_parts(self.view.buf as *const f32, total_elems) };
                Ok(BorrowedSignalBuffer::F32 {
                    slice,
                    num_channels,
                    num_samples,
                    layout,
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "Unsupported buffer format '{}' (itemsize {}). Expected float64 ('d') or float32 ('f').",
                fmt_char as char, itemsize
            ))),
        }
    }
}

impl Drop for PyContiguousBuffer {
    fn drop(&mut self) {
        if !self.view.buf.is_null() {
            unsafe {
                ffi::PyBuffer_Release(&mut self.view);
            }
            self.view.buf = ptr::null_mut();
        }
    }
}
