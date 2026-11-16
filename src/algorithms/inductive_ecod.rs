use num_traits::float::{Float, TotalOrder};
use rayon::prelude::*;

use super::common::{
    compute_score, compute_skewnesses, compute_threshold, par_rows, ranks, validate_contamination,
    validate_finite, validate_finity, validate_shape,
};
use crate::errors::{CorruptionKind, Error};
use crate::matrix::Matrix;
use crate::types::{
    ECODScoreMethod, Fit, FloatMatrix, FloatVector, Predict, TrainScores, Validate,
};

/// Outlier scores of x against sorted train features, row chunks in parallel
fn decision_scores<T: Float + TotalOrder + Send + Sync>(
    data: &FloatMatrix<T>,
    skewnesses: &FloatVector<T>,
    method: ECODScoreMethod,
    x: &Matrix<'_, T>,
) -> FloatVector<T> {
    par_rows(x, |chunk| chunk_scores(data, skewnesses, method, chunk))
}

fn chunk_scores<T: Float + TotalOrder>(
    data: &FloatMatrix<T>,
    skewnesses: &FloatVector<T>,
    method: ECODScoreMethod,
    x: &Matrix<'_, T>,
) -> FloatVector<T> {
    // Consts
    let denominator = T::from(data.first().map_or(0, |col| col.len()) + 1).unwrap();
    let row_count = x.nrows();

    let mut res: FloatVector<T> = vec![T::zero(); row_count];
    // Paper max(L, R, A) = A + max(0, L - A, R - A): res holds A, gains hold the differences
    let gains_len = if method == ECODScoreMethod::MAX {
        row_count
    } else {
        0
    };
    let mut left_gain: FloatVector<T> = vec![T::zero(); gains_len];
    let mut right_gain: FloatVector<T> = vec![T::zero(); gains_len];

    let mut buf = Vec::with_capacity(row_count);
    let mut rk = Vec::with_capacity(row_count);

    for ((col, sorted), &skewness) in x.cols().zip(data).zip(skewnesses) {
        ranks(col, sorted, &mut buf, &mut rk);
        let n = sorted.len();

        let tails = rk.iter().map(|&(lt, le)| {
            (
                T::from(le + 1).unwrap() / denominator,
                T::from(n - lt + 1).unwrap() / denominator,
            )
        });
        let left = tails.clone().map(|(l, _)| l);
        let right = tails.clone().map(|(_, r)| r);

        match method {
            ECODScoreMethod::RIGHT => compute_score(&mut res, right),
            ECODScoreMethod::LEFT => compute_score(&mut res, left),
            ECODScoreMethod::AUTO if skewness >= T::zero() => compute_score(&mut res, right),
            ECODScoreMethod::AUTO => compute_score(&mut res, left),
            // Per-feature auto is either left or right, so min(left, right) covers it
            ECODScoreMethod::PYODMAX => compute_score(&mut res, tails.map(|(l, r)| l.min(r))),
            // Tails are positive: counts start from 1
            ECODScoreMethod::MAX if skewness >= T::zero() => {
                for ((s, g), (l, r)) in res.iter_mut().zip(&mut left_gain).zip(tails) {
                    *s = *s - r.ln();
                    *g = *g + (r / l).ln();
                }
            }
            ECODScoreMethod::MAX => {
                for ((s, g), (l, r)) in res.iter_mut().zip(&mut right_gain).zip(tails) {
                    *s = *s - l.ln();
                    *g = *g + (l / r).ln();
                }
            }
        }
    }
    for ((s, &l), &r) in res.iter_mut().zip(&left_gain).zip(&right_gain) {
        *s = *s + l.max(r).max(T::zero());
    }
    res
}

pub struct InductiveECOD {
    contamination: f64,
    method: ECODScoreMethod,
}

impl InductiveECOD {
    /// Creates estimator, contamination must be in (0, 0.5] as in PyOD
    pub fn new(contamination: f64, method: ECODScoreMethod) -> Result<Self, Error> {
        validate_contamination(contamination)?;
        Ok(InductiveECOD {
            contamination,
            method,
        })
    }
}

#[derive(bitcode::Encode, bitcode::Decode)]
pub struct InductiveECODModel<T> {
    data: FloatMatrix<T>,
    skewnesses: FloatVector<T>,
    contamination: f64,
    method: ECODScoreMethod,
    threshold: T,
    train_score: TrainScores<T>,
}

impl<T: Float + TotalOrder + Send + Sync> InductiveECODModel<T> {
    fn new(
        x: &Matrix<'_, T>,
        contamination: f64,
        method: ECODScoreMethod,
    ) -> InductiveECODModel<T> {
        let cols: Vec<_> = x.cols().collect();
        let data: Vec<Vec<T>> = cols
            .par_iter()
            .map(|col| {
                let mut v: Vec<T> = col.clone().collect();
                v.sort_unstable_by(|a, b| a.total_cmp(b));
                v
            })
            .collect();
        let skewnesses = compute_skewnesses(cols.par_iter().cloned(), method);
        let train_scores = decision_scores(&data, &skewnesses, method, x);

        let threshold = compute_threshold(&train_scores, 100.0 * (1.0 - contamination));

        InductiveECODModel {
            data: data,
            skewnesses: skewnesses,
            contamination,
            method: method,
            threshold: threshold,
            train_score: TrainScores::new(train_scores),
        }
    }
}

impl<T> InductiveECODModel<T> {
    pub fn contamination(&self) -> f64 {
        self.contamination
    }

    pub fn method(&self) -> ECODScoreMethod {
        self.method
    }
}

impl<T: Float + TotalOrder + Send + Sync> Fit<T> for InductiveECOD {
    type Model = InductiveECODModel<T>;

    fn fit(&self, x: &Matrix<'_, T>) -> Result<Self::Model, Error> {
        validate_finite(x)?;

        Ok(InductiveECODModel::new(x, self.contamination, self.method))
    }
}

impl<T: Float + TotalOrder + Send + Sync> Predict<T> for InductiveECODModel<T> {
    fn internal_decision_function(
        &self,
        x: &Matrix<'_, T>,
    ) -> Result<(FloatVector<T>, TrainScores<T>, T), Error> {
        validate_shape(&self.data, x)?;
        validate_finite(x)?;

        let res = decision_scores(&self.data, &self.skewnesses, self.method, x);
        Ok((res, self.train_score, self.threshold))
    }

    fn decision_function(&self, x: &Matrix<'_, T>) -> Result<FloatVector<T>, Error> {
        validate_shape(&self.data, x)?;
        validate_finite(x)?;

        let res = decision_scores(&self.data, &self.skewnesses, self.method, x);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Persist;

    fn fitted() -> InductiveECODModel<f64> {
        InductiveECOD::new(0.1, ECODScoreMethod::MAX)
            .unwrap()
            .fit(&Matrix::from_rows(vec![vec![3.0, -1.0], vec![1.0, 5.0], vec![2.0, 0.5]]).unwrap())
            .unwrap()
    }

    #[test]
    fn new_validates_contamination_range() {
        for value in [0.0, -0.1, 0.51, f64::NAN] {
            assert!(matches!(
                InductiveECOD::new(value, ECODScoreMethod::MAX),
                Err(Error::InvalidContamination { .. })
            ));
        }
        assert!(InductiveECOD::new(0.5, ECODScoreMethod::MAX).is_ok());
    }

    fn assert_corrupted(model: InductiveECODModel<f64>, kind: CorruptionKind) {
        assert_eq!(model.validate(), Err(Error::CorruptedModel { kind }));
    }

    #[test]
    fn validate_accepts_fitted_model() {
        assert_eq!(fitted().validate(), Ok(()));
    }

    #[test]
    fn validate_rejects_skewness_count_mismatch() {
        let mut m = fitted();
        m.skewnesses.pop();
        assert_corrupted(m, CorruptionKind::SkewnessCountMismatch);
    }

    #[test]
    fn validate_rejects_empty_model() {
        let mut m = fitted();
        m.data.clear();
        m.skewnesses.clear();
        assert_corrupted(m, CorruptionKind::Empty);
    }

    #[test]
    fn validate_rejects_ragged_features() {
        let mut m = fitted();
        m.data[1].pop();
        assert_corrupted(m, CorruptionKind::RaggedFeatures);
    }

    #[test]
    fn validate_rejects_non_finite_data() {
        let mut m = fitted();
        m.data[0][0] = f64::NAN;
        assert_corrupted(m, CorruptionKind::NonFiniteData);
    }

    #[test]
    fn validate_rejects_non_finite_skewness() {
        let mut m = fitted();
        m.skewnesses[0] = f64::INFINITY;
        assert_corrupted(m, CorruptionKind::NonFiniteSkewness);
    }

    #[test]
    fn validate_rejects_unsorted_feature() {
        let mut m = fitted();
        m.data[0].reverse();
        assert_corrupted(m, CorruptionKind::UnsortedFeature);
    }

    #[test]
    fn bytes_roundtrip_keeps_model() -> Result<(), Box<dyn std::error::Error>> {
        let m = fitted();
        let restored =
            InductiveECODModel::<f64>::from_bytes(&m.to_bytes(zstd::DEFAULT_COMPRESSION_LEVEL)?)?;
        assert_eq!(restored.data, m.data);
        assert_eq!(restored.skewnesses, m.skewnesses);
        Ok(())
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(InductiveECODModel::<f64>::from_bytes(b"garbage").is_err());
    }
}
