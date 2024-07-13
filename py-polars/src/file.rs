use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{
    self, BorrowedCursor, Cursor, ErrorKind, IoSlice, IoSliceMut, Read, Seek, SeekFrom, Write,
};
#[cfg(target_family = "unix")]
use std::os::fd::{FromRawFd, RawFd};
use std::path::PathBuf;

use either::{for_both, Either};
use polars::io::mmap::MmapBytesReader;
use polars_error::{polars_err, polars_warn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::error::PyPolarsErr;
use crate::prelude::resolve_homedir;

pub struct PyFileLikeObject {
    inner: PyObject,
}

/// Wraps a `PyObject`, and implements read, seek, and write for it.
impl PyFileLikeObject {
    /// Creates an instance of a `PyFileLikeObject` from a `PyObject`.
    /// To assert the object has the required methods methods,
    /// instantiate it with `PyFileLikeObject::require`
    pub fn new(object: PyObject) -> Self {
        PyFileLikeObject { inner: object }
    }

    /// Validates that the underlying
    /// python object has a `read`, `write`, and `seek` methods in respect to parameters.
    /// Will return a `TypeError` if object does not have `read`, `seek`, and `write` methods.
    pub fn ensure_requirements(
        object: &Bound<PyAny>,
        read: bool,
        write: bool,
        seek: bool,
    ) -> PyResult<()> {
        if read && object.getattr("read").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .read() method.",
            ));
        }

        if seek && object.getattr("seek").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .seek() method.",
            ));
        }

        if write && object.getattr("write").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .write() method.",
            ));
        }

        Ok(())
    }
}

/// Extracts a string repr from, and returns an IO error to send back to rust.
fn pyerr_to_io_err(e: PyErr) -> io::Error {
    Python::with_gil(|py| {
        let e_as_object: PyObject = e.into_py(py);

        match e_as_object.call_method_bound(py, "__str__", (), None) {
            Ok(repr) => match repr.extract::<String>(py) {
                Ok(s) => io::Error::new(io::ErrorKind::Other, s),
                Err(_e) => io::Error::new(io::ErrorKind::Other, "An unknown error has occurred"),
            },
            Err(_) => io::Error::new(io::ErrorKind::Other, "Err doesn't have __str__"),
        }
    })
}

impl Read for PyFileLikeObject {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method_bound(py, "read", (buf.len(),), None)
                .map_err(pyerr_to_io_err)?;

            let opt_bytes = bytes.downcast_bound::<PyBytes>(py);

            if let Ok(bytes) = opt_bytes {
                buf.write_all(bytes.as_bytes())?;

                bytes.len().map_err(pyerr_to_io_err)
            } else if let Ok(s) = bytes.downcast_bound::<PyString>(py) {
                let s = s.to_cow().map_err(pyerr_to_io_err)?;
                buf.write_all(s.as_bytes())?;
                Ok(s.len())
            } else {
                Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    polars_err!(InvalidOperation: "could not read from input"),
                ))
            }
        })
    }
}

impl Write for PyFileLikeObject {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let pybytes = PyBytes::new_bound(py, buf);

            let number_bytes_written = self
                .inner
                .call_method_bound(py, "write", (pybytes,), None)
                .map_err(pyerr_to_io_err)?;

            number_bytes_written.extract(py).map_err(pyerr_to_io_err)
        })
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        Python::with_gil(|py| {
            self.inner
                .call_method_bound(py, "flush", (), None)
                .map_err(pyerr_to_io_err)?;

            Ok(())
        })
    }
}

impl Seek for PyFileLikeObject {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, io::Error> {
        Python::with_gil(|py| {
            let (whence, offset) = match pos {
                SeekFrom::Start(i) => (0, i as i64),
                SeekFrom::Current(i) => (1, i),
                SeekFrom::End(i) => (2, i),
            };

            let new_position = self
                .inner
                .call_method_bound(py, "seek", (offset, whence), None)
                .map_err(pyerr_to_io_err)?;

            new_position.extract(py).map_err(pyerr_to_io_err)
        })
    }
}

pub struct FileWrapper(Either<File, PyFileLikeObject>);

impl Read for FileWrapper {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        for_both!(&mut self.0, f => f.read(buf))
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        for_both!(&mut self.0, f => f.read_vectored(bufs))
    }

    fn is_read_vectored(&self) -> bool {
        for_both!(&self.0, f => f.is_read_vectored())
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        for_both!(&mut self.0, f => f.read_to_end(buf))
    }

    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        for_both!(&mut self.0, f => f.read_to_string(buf))
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        for_both!(&mut self.0, f => f.read_exact(buf))
    }

    fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> io::Result<()> {
        for_both!(&mut self.0, f => f.read_buf(buf))
    }

    fn read_buf_exact(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        for_both!(&mut self.0, f => f.read_buf_exact(cursor))
    }
}

impl Write for FileWrapper {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for_both!(&mut self.0, f => f.write(buf))
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        for_both!(&mut self.0, f => f.write_vectored(bufs))
    }

    fn is_write_vectored(&self) -> bool {
        for_both!(&self.0, f => f.is_write_vectored())
    }

    fn flush(&mut self) -> io::Result<()> {
        for_both!(&mut self.0, f => f.flush())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        for_both!(&mut self.0, f => f.write_all(buf))
    }

    fn write_all_vectored(&mut self, bufs: &mut [IoSlice<'_>]) -> io::Result<()> {
        for_both!(&mut self.0, f => f.write_all_vectored(bufs))
    }
}

impl Seek for FileWrapper {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        for_both!(&mut self.0, f => f.seek(pos))
    }
}

impl MmapBytesReader for FileWrapper {
    fn to_file(&self) -> Option<&File> {
        if let Either::Left(ref f) = self.0 {
            Some(f)
        } else {
            None
        }
    }
}

impl FileWrapper {
    pub fn new_with_path(
        py_f: Bound<PyAny>,
        write: bool,
    ) -> PyResult<(FileWrapper, Option<PathBuf>)> {
        let py = py_f.py();
        if let Ok(s) = py_f.extract::<Cow<str>>() {
            let file_path = std::path::Path::new(&*s);
            let file_path = resolve_homedir(file_path);
            let f = if write {
                File::create(&file_path)?
            } else {
                polars_utils::open_file(&file_path).map_err(PyPolarsErr::from)?
            };
            Ok((FileWrapper(Either::Left(f)), Some(file_path)))
        } else {
            let io = py.import_bound("io").unwrap();
            let is_utf8_encoding = |py_f: &Bound<PyAny>| -> PyResult<bool> {
                let encoding = py_f.getattr("encoding")?;
                let encoding = encoding.extract::<Cow<str>>()?;
                Ok(encoding.eq_ignore_ascii_case("utf-8") || encoding.eq_ignore_ascii_case("utf8"))
            };
            #[cfg(target_family = "unix")]
            if let Some(fd) = (py_f.is_exact_instance(&io.getattr("FileIO").unwrap())
                || (py_f.is_exact_instance(&io.getattr("BufferedReader").unwrap())
                    || py_f.is_exact_instance(&io.getattr("BufferedWriter").unwrap())
                    || py_f.is_exact_instance(&io.getattr("BufferedRandom").unwrap())
                    || py_f.is_exact_instance(&io.getattr("BufferedRWPair").unwrap())
                    || (py_f.is_exact_instance(&io.getattr("TextIOWrapper").unwrap())
                        && is_utf8_encoding(&py_f)?))
                    && if write {
                        // invalidate read buffer
                        py_f.call_method0("flush").is_ok()
                    } else {
                        // flush write buffer
                        py_f.call_method1("seek", (0, 1)).is_ok()
                    })
            .then(|| {
                py_f.getattr("fileno")
                    .and_then(|fileno| fileno.call0())
                    .and_then(|fileno| fileno.extract::<libc::c_int>())
                    .ok()
            })
            .flatten()
            .map(|fileno| unsafe {
                // `File::from_raw_fd()` takes the ownership of the file descriptor.
                // When the File is dropped, it closes the file descriptor.
                // This is undesired - the Python file object will become invalid.
                // Therefore, we duplicate the file descriptor here.
                // Closing the duplicated file descriptor will not close
                // the original file descriptor;
                // and the status, e.g. stream position, is still shared with
                // the original file descriptor.
                // We use `F_DUPFD_CLOEXEC` here instead of `dup()`
                // because it also sets the `O_CLOEXEC` flag on the duplicated file descriptor,
                // which `dup()` clears.
                // `open()` in both Rust and Python automatically set `O_CLOEXEC` flag;
                // it prevents leaking file descriptors across processes,
                // and we want to be consistent with them.
                // `F_DUPFD_CLOEXEC` is defined in POSIX.1-2008
                // and is present on all alive UNIX(-like) systems.
                libc::fcntl(fileno, libc::F_DUPFD_CLOEXEC, 0)
            })
            .filter(|fileno| *fileno != -1)
            .map(|fileno| fileno as RawFd)
            {
                return Ok((
                    FileWrapper(Either::Left(unsafe { File::from_raw_fd(fd) })),
                    // This works on Linux and BSD with procfs mounted,
                    // otherwise it fails silently.
                    fs::canonicalize(format!("/proc/self/fd/{fd}")).ok(),
                ));
            }

            // BytesIO is relatively fast, and some code relies on it.
            if !py_f.is_exact_instance(&io.getattr("BytesIO").unwrap()) {
                polars_warn!("Polars found a filename. \
                Ensure you pass a path to the file instead of a python file object when possible for best \
                performance.");
            }
            // Unwrap TextIOWrapper
            // Allow subclasses to allow things like pytest.capture.CaptureIO
            let py_f = if py_f
                .is_instance(&io.getattr("TextIOWrapper").unwrap())
                .unwrap_or_default()
            {
                if !is_utf8_encoding(&py_f)? {
                    return Err(PyPolarsErr::from(
                        polars_err!(InvalidOperation: "file encoding is not UTF-8"),
                    )
                    .into());
                }
                // XXX: we have to clear buffer here.
                // Is there a better solution?
                if write {
                    py_f.call_method0("flush")?;
                } else {
                    py_f.call_method1("seek", (0, 1))?;
                }
                py_f.getattr("buffer")?
            } else {
                py_f
            };
            PyFileLikeObject::ensure_requirements(&py_f, !write, write, !write)?;
            let f = PyFileLikeObject::new(py_f.to_object(py));
            Ok((FileWrapper(Either::Right(f)), None))
        }
    }

    pub fn new(py_f: Bound<PyAny>, write: bool) -> PyResult<FileWrapper> {
        FileWrapper::new_with_path(py_f, write).map(|(f, _)| f)
    }

    pub fn is_buffered(&self) -> bool {
        match self.0 {
            Either::Left(_) => false,
            Either::Right(ref py_f) => Python::with_gil(|py| {
                let py_f = py_f.inner.bind(py);
                let io = py.import_bound("io").unwrap();
                py_f.is_instance(&io.getattr("BufferedIOBase").unwrap())
                    .unwrap_or_default()
            }),
        }
    }
}

// XXX: I want to remove the following functions in the future.
// So that we can have a single entrypoint.

/// If the give file-like is a BytesIO, read its contents.
pub fn read_if_bytesio(py_f: Bound<PyAny>) -> Bound<PyAny> {
    if py_f.getattr("read").is_ok() {
        let Ok(bytes) = py_f.call_method0("getvalue") else {
            return py_f;
        };
        if bytes.downcast::<PyBytes>().is_ok() {
            return bytes.clone();
        }
    }
    py_f
}

/// Create reader from PyBytes or a file-like object. To get BytesIO to have
/// better performance, use read_if_bytesio() before calling this.
pub fn get_mmap_bytes_reader<'a>(
    py_f: &'a Bound<'a, PyAny>,
) -> PyResult<Box<dyn MmapBytesReader + 'a>> {
    get_mmap_bytes_reader_and_path(py_f).map(|t| t.0)
}

pub fn get_mmap_bytes_reader_and_path<'a>(
    py_f: &'a Bound<'a, PyAny>,
) -> PyResult<(Box<dyn MmapBytesReader + 'a>, Option<PathBuf>)> {
    // bytes object
    if let Ok(bytes) = py_f.downcast::<PyBytes>() {
        Ok((Box::new(Cursor::new(bytes.as_bytes())), None))
    } else {
        FileWrapper::new_with_path(py_f.clone(), false).map(
            |(f, path)| -> (Box<dyn MmapBytesReader + 'a>, Option<PathBuf>) { (Box::new(f), path) },
        )
    }
}
