use crate::errors::Error;
use num_traits::Float;
use pyo3::prelude::*;
use std::{fs, path::Path};

/// Enum for each ECOD decision function type
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum ECODScoreMethod {
    RIGHT,
    LEFT,
    AUTO,
    MAX,
}

pub type FloatMatrix<T> = Vec<Vec<T>>;
pub type FloatVector<T> = Vec<T>;

/// Base trait to initialize model parameters
pub trait Fit<T: Float> {
    type Model;
    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error>;
}

/// Base trait to get predictions from model
pub trait Predict<T: Float> {
    fn decision_function(
        &self,
        x: &FloatMatrix<T>,
        method: ECODScoreMethod,
    ) -> Result<FloatVector<T>, Error>;
}

/// Trait to save and load models via blanket impl
pub trait Persist: Sized {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Error>;

    fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        fs::write(path, self.to_bytes())?;
        Ok(())
    }

    fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

impl<T> Persist for T
where
    T: bitcode::Encode + bitcode::DecodeOwned,
{
    fn to_bytes(&self) -> Vec<u8> {
        bitcode::encode(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        bitcode::decode(bytes).map_err(|_| Error::Decode)
    }
}
