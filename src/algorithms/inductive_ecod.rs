use num_traits::float::{Float, TotalOrder};

use super::common::{
    compute_score, compute_skewnesses, compute_threshold, validate_contamination, validate_finity,
    validate_shape,
};
use crate::errors::{CorruptionKind, Error};
use crate::types::{
    ECODScoreMethod, Fit, FloatMatrix, FloatVector, Predict, TrainScores, Validate,
};

/// Right tail probability
pub fn right_tail<T: Float>(sorted: &FloatVector<T>, value: T, denominator: T) -> T {
    T::from(sorted.len() - sorted.partition_point(|v| *v < value) + 1).unwrap() / denominator
}

/// Left tail probability
pub fn left_tail<T: Float>(sorted: &FloatVector<T>, value: T, denominator: T) -> T {
    T::from(sorted.partition_point(|v| *v <= value) + 1).unwrap() / denominator
}

/// Left and right tail probabilities, second search runs on the prefix only
pub fn tails<T: Float>(sorted: &FloatVector<T>, value: T, denominator: T) -> (T, T) {
    let le = sorted.partition_point(|v| *v <= value);
    let lt = sorted[..le].partition_point(|v| *v < value);
    (
        T::from(le + 1).unwrap() / denominator,
        T::from(sorted.len() - lt + 1).unwrap() / denominator,
    )
}

/// Outlier scores of x against sorted train features
fn decision_scores<T: Float>(
    data: &FloatMatrix<T>,
    skewnesses: &FloatVector<T>,
    method: ECODScoreMethod,
    x: &FloatMatrix<T>,
) -> FloatVector<T> {
    // Consts
    let denominator = T::from(data.first().map_or(0, |col| col.len()) + 1).unwrap();
    let row_count = x.first().map_or(0, |x| x.len());

    let mut res: FloatVector<T> = vec![T::zero(); row_count];
    // Paper max(L, R, A) = A + max(0, L - A, R - A): res holds A, gains hold the differences
    let gains_len = if method == ECODScoreMethod::MAX {
        row_count
    } else {
        0
    };
    let mut left_gain: FloatVector<T> = vec![T::zero(); gains_len];
    let mut right_gain: FloatVector<T> = vec![T::zero(); gains_len];

    for ((col, sorted), &skewness) in x.iter().zip(data).zip(skewnesses) {
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
            ECODScoreMethod::PYODMAX => compute_score(
                &mut res,
                col.iter().map(|v| {
                    let (l, r) = tails(sorted, *v, denominator);
                    l.min(r)
                }),
            ),
            // Tails are positive: counts start from 1
            ECODScoreMethod::MAX if skewness >= T::zero() => {
                for ((s, g), v) in res.iter_mut().zip(&mut left_gain).zip(col) {
                    let (l, r) = tails(sorted, *v, denominator);
                    *s = *s - r.ln();
                    *g = *g + (r / l).ln();
                }
            }
            ECODScoreMethod::MAX => {
                for ((s, g), v) in res.iter_mut().zip(&mut right_gain).zip(col) {
                    let (l, r) = tails(sorted, *v, denominator);
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

impl<T: Float + TotalOrder> InductiveECODModel<T> {
    fn new(
        x: &FloatMatrix<T>,
        contamination: f64,
        method: ECODScoreMethod,
    ) -> InductiveECODModel<T> {
        let data: Vec<Vec<T>> = x
            .iter()
            .map(|inner| {
                let mut v = inner.clone();
                v.sort_by(|a, b| a.total_cmp(b));
                v
            })
            .collect();
        let skewnesses = compute_skewnesses(x, method);
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

impl<T: Float + TotalOrder> Fit<T> for InductiveECOD {
    type Model = InductiveECODModel<T>;

    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error> {
        validate_finity(x)?;

        Ok(InductiveECODModel::new(x, self.contamination, self.method))
    }
}

impl<T: Float + TotalOrder> Predict<T> for InductiveECODModel<T> {
    fn internal_decision_function(
        &self,
        x: &FloatMatrix<T>,
    ) -> Result<(FloatVector<T>, TrainScores<T>, T), Error> {
        validate_shape(&self.data, x)?;
        validate_finity(x)?;

        let res = decision_scores(&self.data, &self.skewnesses, self.method, x);
        Ok((res, self.train_score, self.threshold))
    }

    fn decision_function(&self, x: &FloatMatrix<T>) -> Result<FloatVector<T>, Error> {
        validate_shape(&self.data, x)?;
        validate_finity(x)?;

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

    const EPS: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPS,
            "expected {expected}, got {actual}"
        );
    }

    // Tails
    #[test]
    fn right_tail_counts_values_ge_plus_one() {
        let sorted: FloatVector<f64> = vec![1.0, 2.0, 2.0, 3.0];
        let d = 5.0;
        assert_close(right_tail(&sorted, 0.0, d), 5.0 / d); // 4 + 1
        assert_close(right_tail(&sorted, 2.0, d), 4.0 / d); // 2,2,3 + 1
        assert_close(right_tail(&sorted, 10.0, d), 1.0 / d); // 0 + 1
    }

    #[test]
    fn left_tail_counts_values_le_plus_one() {
        let sorted: FloatVector<f64> = vec![1.0, 2.0, 2.0, 3.0];
        let d = 5.0;
        assert_close(left_tail(&sorted, 0.0, d), 1.0 / d); // 0 + 1
        assert_close(left_tail(&sorted, 2.0, d), 4.0 / d); // 1,2,2 + 1
        assert_close(left_tail(&sorted, 10.0, d), 5.0 / d); // 4 + 1
    }

    fn fitted() -> InductiveECODModel<f64> {
        InductiveECOD::new(0.1, ECODScoreMethod::MAX)
            .unwrap()
            .fit(&vec![vec![3.0, 1.0, 2.0], vec![-1.0, 5.0, 0.5]])
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
