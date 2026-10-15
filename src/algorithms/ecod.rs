use num_traits::float::{Float, TotalOrder};

use super::common::{
    compute_score, compute_skewness, compute_threshold, validate_contamination, validate_finity,
    validate_shape,
};
use crate::errors::{CorruptionKind, Error};
use crate::types::{
    ECODScoreMethod, Fit, FloatMatrix, FloatVector, Predict, TrainScores, Validate,
};

/// Right tail probability
pub fn right_tail<T: Float>(sorted: &FloatVector<T>, value: T, denominator: T) -> T {
    T::from(sorted.len() - sorted.partition_point(|v| *v < value)).unwrap() / denominator
}

/// Left tail probability
pub fn left_tail<T: Float>(sorted: &FloatVector<T>, value: T, denominator: T) -> T {
    T::from(sorted.partition_point(|v| *v <= value)).unwrap() / denominator
}
/// Function that merges a sorted feature matrix with an unsorted one.
fn merge<T: Float + TotalOrder>(
    a: &FloatMatrix<T>,
    unsorted_data: &FloatMatrix<T>,
) -> FloatMatrix<T> {
    let mut res: Vec<Vec<T>> = vec![];

    for (acol, bcol) in a.iter().zip(unsorted_data) {
        let mut b = bcol.clone();
        b.sort_by(|l, r| l.total_cmp(r));

        let mut res_col = Vec::with_capacity(acol.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < acol.len() && j < b.len() {
            if acol[i].total_cmp(&b[j]).is_le() {
                res_col.push(acol[i]);
                i += 1;
            } else {
                res_col.push(b[j]);
                j += 1;
            }
        }
        res_col.extend_from_slice(&acol[i..]);
        res_col.extend_from_slice(&b[j..]);

        res.push(res_col);
    }

    res
}

/// Outlier scores of x against sorted features that already contain x
fn decision_scores<T: Float>(
    sorted_data: &FloatMatrix<T>,
    skewnesses: &FloatVector<T>,
    method: ECODScoreMethod,
    x: &FloatMatrix<T>,
) -> FloatVector<T> {
    // Consts
    let denominator = T::from(sorted_data.first().map_or(0, |col| col.len())).unwrap();
    let row_count = x.first().map_or(0, |x| x.len());

    let mut res: FloatVector<T> = vec![T::zero(); row_count];

    for ((col, sorted), &skewness) in x.iter().zip(sorted_data).zip(skewnesses) {
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
    res
}

/// Base struct of ECOD model from original article.
pub struct ECOD {
    contamination: f64,
    method: ECODScoreMethod,
}

impl ECOD {
    /// Creates estimator, contamination must be in (0, 0.5] as in PyOD
    pub fn new(contamination: f64, method: ECODScoreMethod) -> Result<Self, Error> {
        validate_contamination(contamination)?;
        Ok(ECOD {
            contamination,
            method,
        })
    }
}

/// Model struct of ECOD model from original article.
#[derive(bitcode::Encode, bitcode::Decode)]
pub struct ECODModel<T> {
    data: FloatMatrix<T>,
    contamination: f64,
    method: ECODScoreMethod,
    threshold: T,
    train_score: TrainScores<T>,
}

impl<T: Float + TotalOrder> ECODModel<T> {
    fn new(x: &FloatMatrix<T>, contamination: f64, method: ECODScoreMethod) -> ECODModel<T> {
        let data: FloatMatrix<T> = x
            .iter()
            .map(|inner| {
                let mut v = inner.clone();
                v.sort_by(|a, b| a.total_cmp(b));
                v
            })
            .collect();
        // Train sample is scored against itself, so data already contains x
        let skewnesses: FloatVector<T> = x.iter().map(|v| compute_skewness(v)).collect();
        let train_scores = decision_scores(&data, &skewnesses, method, x);

        let threshold = compute_threshold(&train_scores, 100.0 * (1.0 - contamination));

        ECODModel {
            data,
            contamination,
            method,
            threshold,
            train_score: TrainScores::new(train_scores),
        }
    }

    /// Scores x against train data merged with x itself
    fn merged_scores(&self, x: &FloatMatrix<T>) -> Result<FloatVector<T>, Error> {
        validate_shape(&self.data, x)?;
        validate_finity(x)?;

        let merged = merge(&self.data, x);
        let skewnesses: FloatVector<T> = merged.iter().map(|v| compute_skewness(v)).collect();
        Ok(decision_scores(&merged, &skewnesses, self.method, x))
    }
}

impl<T> ECODModel<T> {
    pub fn contamination(&self) -> f64 {
        self.contamination
    }

    pub fn method(&self) -> ECODScoreMethod {
        self.method
    }
}

impl<T: Float + TotalOrder> Fit<T> for ECOD {
    type Model = ECODModel<T>;

    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error> {
        validate_finity(x)?;

        Ok(ECODModel::new(x, self.contamination, self.method))
    }
}

impl<T: Float + TotalOrder> Predict<T> for ECODModel<T> {
    fn internal_decision_function(
        &self,
        x: &FloatMatrix<T>,
    ) -> Result<(FloatVector<T>, TrainScores<T>, T), Error> {
        let res = self.merged_scores(x)?;
        Ok((res, self.train_score, self.threshold))
    }

    fn decision_function(&self, x: &FloatMatrix<T>) -> Result<FloatVector<T>, Error> {
        self.merged_scores(x)
    }
}

impl<T: Float> Validate for ECODModel<T> {
    fn validate(&self) -> Result<(), Error> {
        let corrupted = |kind| Error::CorruptedModel { kind };

        validate_finity(&self.data).map_err(|e| {
            corrupted(match e {
                Error::EmptyInput => CorruptionKind::Empty,
                Error::RaggedRows => CorruptionKind::RaggedFeatures,
                _ => CorruptionKind::NonFiniteData,
            })
        })?;
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

    // Tails: `sorted` is the merged sample, so the scored value is already in it
    #[test]
    fn right_tail_counts_values_ge() {
        let sorted: FloatVector<f64> = vec![1.0, 2.0, 2.0, 3.0];
        let d = 4.0;
        assert_close(right_tail(&sorted, 1.0, d), 4.0 / d); // 1,2,2,3
        assert_close(right_tail(&sorted, 2.0, d), 3.0 / d); // 2,2,3
        assert_close(right_tail(&sorted, 3.0, d), 1.0 / d); // 3
    }

    #[test]
    fn left_tail_counts_values_le() {
        let sorted: FloatVector<f64> = vec![1.0, 2.0, 2.0, 3.0];
        let d = 4.0;
        assert_close(left_tail(&sorted, 1.0, d), 1.0 / d); // 1
        assert_close(left_tail(&sorted, 2.0, d), 3.0 / d); // 1,2,2
        assert_close(left_tail(&sorted, 3.0, d), 4.0 / d); // 1,2,2,3
    }

    // Merge
    #[test]
    fn merge_sorts_new_data_and_interleaves_per_column() {
        let a: FloatMatrix<f64> = vec![vec![1.0, 3.0, 5.0], vec![-2.0, 0.0, 2.0]];
        let b: FloatMatrix<f64> = vec![vec![4.0, 3.0], vec![1.0, -3.0]];
        assert_eq!(
            merge(&a, &b),
            vec![
                vec![1.0, 3.0, 3.0, 4.0, 5.0],
                vec![-3.0, -2.0, 0.0, 1.0, 2.0]
            ]
        );
    }

    #[test]
    fn merge_appends_leftovers_from_either_side() {
        let a: FloatMatrix<f64> = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        let b: FloatMatrix<f64> = vec![vec![10.0, 9.0], vec![-1.0, 0.0]];
        assert_eq!(
            merge(&a, &b),
            vec![vec![1.0, 2.0, 9.0, 10.0], vec![-1.0, 0.0, 5.0, 6.0]]
        );
    }

    // Model
    fn fitted() -> ECODModel<f64> {
        ECOD::new(0.1, ECODScoreMethod::MAX)
            .unwrap()
            .fit(&vec![vec![3.0, 1.0, 2.0], vec![-1.0, 5.0, 0.5]])
            .unwrap()
    }

    #[test]
    fn new_validates_contamination_range() {
        for value in [0.0, -0.1, 0.51, f64::NAN] {
            assert!(matches!(
                ECOD::new(value, ECODScoreMethod::MAX),
                Err(Error::InvalidContamination { .. })
            ));
        }
        assert!(ECOD::new(0.5, ECODScoreMethod::MAX).is_ok());
    }

    fn assert_corrupted(model: ECODModel<f64>, kind: CorruptionKind) {
        assert_eq!(model.validate(), Err(Error::CorruptedModel { kind }));
    }

    #[test]
    fn validate_accepts_fitted_model() {
        assert_eq!(fitted().validate(), Ok(()));
    }

    #[test]
    fn validate_rejects_empty_model() {
        let mut m = fitted();
        m.data.clear();
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
    fn validate_rejects_unsorted_feature() {
        let mut m = fitted();
        m.data[0].reverse();
        assert_corrupted(m, CorruptionKind::UnsortedFeature);
    }

    #[test]
    fn bytes_roundtrip_keeps_model() -> Result<(), Box<dyn std::error::Error>> {
        let m = fitted();
        let restored = ECODModel::<f64>::from_bytes(&m.to_bytes(zstd::DEFAULT_COMPRESSION_LEVEL)?)?;
        assert_eq!(restored.data, m.data);
        Ok(())
    }

    #[test]
    fn from_bytes_rejects_garbage() {
        assert!(ECODModel::<f64>::from_bytes(b"garbage").is_err());
    }
}
