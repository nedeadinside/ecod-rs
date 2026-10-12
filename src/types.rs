use crate::errors::Error;
use num_traits::Float;
use std::{fs, path::Path};

/// Enum for each ECOD decision function type
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

pub trait Validate {
    fn validate(&self) -> Result<(), Error>;
}

/// Trait to save and load models via blanket impl
pub trait Persist: Sized {
    fn to_bytes(&self, compression_level: i32) -> Result<Vec<u8>, Error>;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Error>;

    fn save(&self, path: impl AsRef<Path>, compression_level: Option<i32>) -> Result<(), Error> {
        fs::write(
            path,
            self.to_bytes(compression_level.unwrap_or(zstd::DEFAULT_COMPRESSION_LEVEL))?,
        )?;
        Ok(())
    }

    fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

impl<T> Persist for T
where
    T: bitcode::Encode + bitcode::DecodeOwned + Validate,
{
    fn to_bytes(&self, compression_level: i32) -> Result<Vec<u8>, Error> {
        let model_bytes = bitcode::encode(self);
        zstd::bulk::compress(&model_bytes, compression_level).map_err(|_| Error::Compress)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let raw = zstd::decode_all(bytes).map_err(|_| Error::Decompress)?;
        let model = bitcode::decode::<Self>(&raw).map_err(|_| Error::Decode)?;
        model.validate()?;
        Ok(model)
    }
}
