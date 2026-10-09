use std::io;
use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub mod algorithms;
pub mod errors;
pub mod types;

use algorithms::inductive_ecod::{InductiveECOD, InductiveECODModel};
use errors::Error;
use types::{ECODScoreMethod, Fit, FloatMatrix, FloatVector, Predict};

use crate::types::Persist;

impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        match err {
            // pyo3 maps io::ErrorKind to FileNotFoundError, PermissionError, etc.
            Error::Io { kind } => io::Error::from(kind).into(),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}

/// Python wrapper over InductiveECODModel.
#[pyclass(name = "InductiveECOD")]
pub struct PyInductiveECOD {
    model: Option<InductiveECODModel<f64>>,
}

#[pymethods]
impl PyInductiveECOD {
    #[new]
    fn new() -> Self {
        Self { model: None }
    }

    fn fit(mut slf: PyRefMut<'_, Self>, x: FloatMatrix<f64>) -> PyResult<PyRefMut<'_, Self>> {
        let model = slf.py().detach(|| InductiveECOD.fit(&x))?;
        slf.model = Some(model);
        Ok(slf)
    }

    #[pyo3(signature = (x, method = ECODScoreMethod::MAX))]
    fn decision_function(
        &self,
        py: Python<'_>,
        x: FloatMatrix<f64>,
        method: ECODScoreMethod,
    ) -> PyResult<FloatVector<f64>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.decision_function(&x, method))?)
    }

    #[pyo3(signature = (path))]
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not fitted"))?;
        Ok(py.detach(|| model.save(path))?)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let model: InductiveECODModel<f64> = py.detach(|| InductiveECODModel::load(path))?;
        Ok(Self { model: Some(model) })
    }
}

/// Rust core of fast_ecod.
#[pymodule]
mod _core {
    #[pymodule_export]
    use super::PyInductiveECOD;
    #[pymodule_export]
    use super::types::ECODScoreMethod;
}
