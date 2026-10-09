use num_traits::float::{Float, TotalOrder};

use super::common::{
    compute_score, compute_skewness, left_tail, right_tail, validate_finity, validate_shape,
};
use crate::errors::{CorruptionKind, Error};
use crate::types::{ECODScoreMethod, Fit, FloatMatrix, FloatVector, Predict, Validate};

pub struct InductiveECOD;

#[derive(bitcode::Encode, bitcode::Decode)]
pub struct InductiveECODModel<T> {
    data: FloatMatrix<T>,
    skewnesses: FloatVector<T>,
}

impl<T: Float + TotalOrder> Fit<T> for InductiveECOD {
    type Model = InductiveECODModel<T>;

    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error> {
        validate_finity(x)?;

        Ok(InductiveECODModel {
            data: x
                .iter()
                .map(|inner| {
                    let mut v = inner.clone();
                    v.sort_by(|a, b| a.total_cmp(b));
                    v
                })
                .collect(),
            skewnesses: x.iter().map(|vector| compute_skewness(vector)).collect(),
        })
    }
}

impl<T: Float + TotalOrder> Predict<T> for InductiveECODModel<T> {
    fn decision_function(
        &self,
        x: &FloatMatrix<T>,
        method: ECODScoreMethod,
    ) -> Result<FloatVector<T>, Error> {
        validate_shape(&self.data, x)?;
        validate_finity(x)?;

        // Consts
        let denominator = T::from(self.data.first().map_or(0, |col| col.len()) + 1).unwrap();
        let row_count = x.first().map_or(0, |x| x.len());

        let mut res: FloatVector<T> = vec![T::zero(); row_count];

        for ((col, sorted), &skewness) in x.iter().zip(&self.data).zip(&self.skewnesses) {
            let right = |v: &T| right_tail(sorted, *v, denominator);
            let left = |v: &T| left_tail(sorted, *v, denominator);

            match method {
                ECODScoreMethod::RIGHT => compute_score(&mut res, col.iter().map(right)),
                ECODScoreMethod::LEFT => compute_score(&mut res, col.iter().map(left)),
                ECODScoreMethod::AUTO if skewness >= T::zero() => {
                    compute_score(&mut res, col.iter().map(right))
                }
                ECODScoreMethod::AUTO => compute_score(&mut res, col.iter().map(left)),
                // Per-feature auto is either left or right, so min(left, right) covers it
                ECODScoreMethod::MAX => {
                    compute_score(&mut res, col.iter().map(|v| left(v).min(right(v))))
                }
            }
        }
        Ok(res)
    }
}

impl<T: Float> Validate for InductiveECODModel<T> {
    fn validate(&self) -> Result<(), Error> {
        let corrupted = |kind| Error::CorruptedModel { kind };

        if self.data.len() != self.skewnesses.len() {
            return Err(corrupted(CorruptionKind::SkewnessCountMismatch));
        }
        validate_finity(&self.data).map_err(|e| {
            corrupted(match e {
                Error::EmptyInput => CorruptionKind::Empty,
                Error::RaggedRows => CorruptionKind::RaggedFeatures,
                _ => CorruptionKind::NonFiniteData,
            })
        })?;
        if self.skewnesses.iter().any(|v| !v.is_finite()) {
            return Err(corrupted(CorruptionKind::NonFiniteSkewness));
        }
        if self.data.iter().any(|v| !v.is_sorted()) {
            return Err(corrupted(CorruptionKind::UnsortedFeature));
        }
        Ok(())
    }
}
