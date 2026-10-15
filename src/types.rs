use crate::errors::Error;
use num_traits::{Float, NumCast};
use std::{fs, path::Path};

/// Enum for each ECOD decision function type
#[derive(Clone, Copy, PartialEq, bitcode::Encode, bitcode::Decode)]
pub enum ECODScoreMethod {
    RIGHT,
    LEFT,
    AUTO,
    MAX,
}

/// Enum for each predict_proba conversion type
#[derive(Clone, Copy, PartialEq)]
pub enum ECODProbaMethod {
    LINEAR,
    UNIFY,
}

pub type FloatMatrix<T> = Vec<Vec<T>>;
pub type FloatVector<T> = Vec<T>;
pub type IntVector = Vec<i32>;

#[derive(Clone, Copy, bitcode::Encode, bitcode::Decode)]
pub struct TrainScores<T> {
    pub min: T,
    pub max: T,
    pub mean: T,
    pub std: T,
}

impl<T: Float> TrainScores<T> {
    pub fn new(train_scores: FloatVector<T>) -> TrainScores<T> {
        let n: T = NumCast::from(train_scores.len()).unwrap();

        let min = train_scores.iter().copied().fold(T::infinity(), Float::min);
        let max = train_scores
            .iter()
            .copied()
            .fold(T::neg_infinity(), Float::max);
        let mean = train_scores
            .iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x)
            / n;
        let std = train_scores.iter().copied().fold(T::zero(), |acc, x| {
            let d = x - mean;
            acc + d * d
        }) / n;
        // Population std (ddof = 0), as numpy.std used by PyOD
        let std = std.sqrt();

        TrainScores {
            min: min,
            max: max,
            mean: mean,
            std: std,
        }
    }
}

/// Base trait to initialize model parameters
pub trait Fit<T: Float> {
    type Model;
    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error>;
}

/// Base trait to get predictions from model
pub trait Predict<T: Float> {
    fn internal_decision_function(
        &self,
        x: &FloatMatrix<T>,
    ) -> Result<(FloatVector<T>, TrainScores<T>, T), Error>;

    fn decision_function(&self, x: &FloatMatrix<T>) -> Result<FloatVector<T>, Error>;

    fn predict(&self, x: &FloatMatrix<T>) -> Result<Vec<i32>, Error> {
        let (result, _, threshold) = self.internal_decision_function(x)?;
        Ok(result
            .iter()
            .map(|&v| if v > threshold { 1 } else { 0 })
            .collect())
    }

    fn predict_proba(
        &self,
        x: &FloatMatrix<T>,
        method: ECODProbaMethod,
    ) -> Result<FloatMatrix<T>, Error> {
        let (result, train_score, _) = self.internal_decision_function(x)?;

        let outlier: FloatVector<T> = match method {
            ECODProbaMethod::LINEAR => {
                // Zero range is replaced by 1, as sklearn MinMaxScaler does
                let range = train_score.max - train_score.min;
                let range = if range == T::zero() { T::one() } else { range };
                result
                    .iter()
                    .map(|&s| ((s - train_score.min) / range).max(T::zero()).min(T::one()))
                    .collect()
            }
            ECODProbaMethod::UNIFY => {
                let scale = train_score.std * T::from(2.0).unwrap().sqrt();
                // Zero std gives NaN Float::max returns the non-NaN operand, so it maps to 0
                result
                    .iter()
                    .map(|&s| {
                        let z = ((s - train_score.mean) / scale).to_f64().unwrap();
                        T::from(libm::erf(z)).unwrap().max(T::zero()).min(T::one())
                    })
                    .collect()
            }
        };
        let inlier = outlier.iter().map(|&p| T::one() - p).collect();

        Ok(vec![inlier, outlier])
    }
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
