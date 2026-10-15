use std::io;
use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

pub mod algorithms;
pub mod errors;
pub mod types;

use algorithms::ecod::{ECOD, ECODModel};
use algorithms::inductive_ecod::{InductiveECOD, InductiveECODModel};
use errors::Error;
use types::{ECODProbaMethod, ECODScoreMethod, Fit, FloatMatrix, FloatVector, Predict};

use crate::types::Persist;

impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        match err {
            // pyo3 maps io::ErrorKind to FileNotFoundError, PermissionError, etc.
            Error::Io { kind } => io::Error::from(kind).into(),
            Error::Compress => PyRuntimeError::new_err("failed to compress model bytes"),
            Error::Decompress => PyRuntimeError::new_err("failed to decompress model bytes"),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}

/// Python wrapper over InductiveECODModel.
#[pyclass(name = "InductiveECOD")]
pub struct PyInductiveECOD {
    estimator: InductiveECOD,
    model: Option<InductiveECODModel<f64>>,
}

/// Python wrapper over ECODModel.
#[pyclass(name = "ECOD")]
pub struct PyECOD {
    estimator: ECOD,
    model: Option<ECODModel<f64>>,
}

impl FromPyObject<'_, '_> for ECODScoreMethod {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let s: PyBackedStr = obj.extract()?;
        match &*s {
            "right" => Ok(Self::RIGHT),
            "left" => Ok(Self::LEFT),
            "auto" => Ok(Self::AUTO),
            "pyod-max" => Ok(Self::PYODMAX),
            "max" => Ok(Self::MAX),
            other => Err(PyValueError::new_err(format!(
                "unknown method {other:?}, expected one of: right, left, auto, pyod-max, max"
            ))),
        }
    }
}

impl FromPyObject<'_, '_> for ECODProbaMethod {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        let s: PyBackedStr = obj.extract()?;
        match &*s {
            "linear" => Ok(Self::LINEAR),
            "unify" => Ok(Self::UNIFY),
            other => Err(PyValueError::new_err(format!(
                "unknown method {other:?}, expected one of: linear, unify"
            ))),
        }
    }
}

#[pymethods]
impl PyInductiveECOD {
    #[new]
    #[pyo3(signature = (contamination = 0.1, method = ECODScoreMethod::MAX))]
    fn new(contamination: f64, method: ECODScoreMethod) -> PyResult<Self> {
        Ok(Self {
            estimator: InductiveECOD::new(contamination, method)?,
            model: None,
        })
    }

    fn fit(mut slf: PyRefMut<'_, Self>, x: FloatMatrix<f64>) -> PyResult<PyRefMut<'_, Self>> {
        let estimator = &slf.estimator;
        let model = slf.py().detach(|| estimator.fit(&x))?;
        slf.model = Some(model);
        Ok(slf)
    }

    #[pyo3(signature = (x))]
    fn decision_function(&self, py: Python<'_>, x: FloatMatrix<f64>) -> PyResult<FloatVector<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.decision_function(&x))?)
    }

    #[pyo3(signature = (x))]
    fn predict(&self, py: Python<'_>, x: FloatMatrix<f64>) -> PyResult<Vec<i32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.predict(&x))?)
    }

    #[pyo3(signature = (x, method = ECODProbaMethod::LINEAR))]
    fn predict_proba(
        &self,
        py: Python<'_>,
        x: FloatMatrix<f64>,
        method: ECODProbaMethod,
    ) -> PyResult<FloatMatrix<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.predict_proba(&x, method))?)
    }

    #[pyo3(signature = (path, *, compression_level = None))]
    fn save(&self, py: Python<'_>, path: PathBuf, compression_level: Option<i32>) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.save(path, compression_level))?)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model: InductiveECODModel<f64> = py.detach(|| InductiveECODModel::load(path))?;
        Ok(Self {
            estimator: InductiveECOD::new(model.contamination(), model.method())?,
            model: Some(model),
        })
    }
}

#[pymethods]
impl PyECOD {
    #[new]
    #[pyo3(signature = (contamination = 0.1, method = ECODScoreMethod::MAX))]
    fn new(contamination: f64, method: ECODScoreMethod) -> PyResult<Self> {
        Ok(Self {
            estimator: ECOD::new(contamination, method)?,
            model: None,
        })
    }

    fn fit(mut slf: PyRefMut<'_, Self>, x: FloatMatrix<f64>) -> PyResult<PyRefMut<'_, Self>> {
        let estimator = &slf.estimator;
        let model = slf.py().detach(|| estimator.fit(&x))?;
        slf.model = Some(model);
        Ok(slf)
    }

    #[pyo3(signature = (x))]
    fn decision_function(&self, py: Python<'_>, x: FloatMatrix<f64>) -> PyResult<FloatVector<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.decision_function(&x))?)
    }

    #[pyo3(signature = (x))]
    fn predict(&self, py: Python<'_>, x: FloatMatrix<f64>) -> PyResult<Vec<i32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.predict(&x))?)
    }

    #[pyo3(signature = (x, method = ECODProbaMethod::LINEAR))]
    fn predict_proba(
        &self,
        py: Python<'_>,
        x: FloatMatrix<f64>,
        method: ECODProbaMethod,
    ) -> PyResult<FloatMatrix<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.predict_proba(&x, method))?)
    }

    #[pyo3(signature = (path, *, compression_level = None))]
    fn save(&self, py: Python<'_>, path: PathBuf, compression_level: Option<i32>) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.save(path, compression_level))?)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model: ECODModel<f64> = py.detach(|| ECODModel::load(path))?;
        Ok(Self {
            estimator: ECOD::new(model.contamination(), model.method())?,
            model: Some(model),
        })
    }
}

/// Rust core of ecod_rs.
#[pymodule]
mod _core {
    #[pymodule_export]
    use super::{PyECOD, PyInductiveECOD};
}
