use crate::errors::Error;
use crate::types::{FloatMatrix, FloatVector};

use num_traits::Float;

/// Computes the score as the negative sum of natural logarithms of the input values.
pub fn compute_score<T: Float>(scores: &mut FloatVector<T>, probs: impl Iterator<Item = T>) {
    // Inputs are guaranteed to be positive and cannot cause the result
    // to overflow to infinity, so no additional validation is required.
    for (score, p) in scores.iter_mut().zip(probs) {
        if p > T::zero() {
            *score = *score - p.ln();
        }
    }
}

/// Computes skewness of the given vector.
pub fn compute_skewness<T: Float>(vector: &FloatVector<T>) -> T {
    let n = T::from(vector.len()).unwrap();
    let mean = vector.iter().fold(T::zero(), |a, &x| a + x) / n;

    let (sum2, sum3) = vector.iter().fold((T::zero(), T::zero()), |(s2, s3), &x| {
        let d = x - mean;
        let d2 = d * d;
        (s2 + d2, s3 + d2 * d)
    });

    let m2 = sum2 / n;
    let m3 = sum3 / n;
    if m2 == T::zero() {
        return T::zero();
    }
    m3 / (m2 * m2.sqrt())
}

pub fn validate_shape<T: Float>(a: &FloatMatrix<T>, b: &FloatMatrix<T>) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::FeatureMismatch {
            expected: (a.len()),
            got: (b.len()),
        });
    }
    Ok(())
}

/// Validates that the feature matrix is non-empty, rectangular, and fully finite.
pub fn validate_finity<T: Float>(feature_matrix: &FloatMatrix<T>) -> Result<(), Error> {
    let n_rows = match feature_matrix.first() {
        Some(col) if !col.is_empty() => col.len(),
        _ => return Err(Error::EmptyInput),
    };

    for (col, vec) in feature_matrix.iter().enumerate() {
        if vec.len() != n_rows {
            return Err(Error::RaggedRows);
        }
        if let Some(row) = vec.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFinite { row, col });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPS,
            "expected {expected}, got {actual}"
        );
    }

    fn matrix(cols: &[&[f32]]) -> FloatMatrix<f32> {
        cols.iter().map(|c| c.to_vec()).collect()
    }

    // Shape
    #[test]
    fn validate_shape_accepts_same_number_of_features() {
        let a = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_eq!(validate_shape(&a, &a), Ok(()));
    }

    #[test]
    fn validate_shape_rejects_fewer_features() {
        let a = matrix(&[&[1.0], &[2.0], &[3.0]]);
        let b = matrix(&[&[1.0]]);
        assert_eq!(
            validate_shape(&a, &b),
            Err(Error::FeatureMismatch {
                expected: 3,
                got: 1
            })
        );
    }

    #[test]
    fn validate_shape_rejects_more_features() {
        let a = matrix(&[&[1.0]]);
        let b = matrix(&[&[1.0], &[2.0]]);
        assert_eq!(
            validate_shape(&a, &b),
            Err(Error::FeatureMismatch {
                expected: 1,
                got: 2
            })
        );
    }

    #[test]
    fn validate_shape_ignores_number_of_rows() {
        let a = matrix(&[&[1.0, 2.0, 3.0]]);
        let b = matrix(&[&[1.0]]);
        assert_eq!(validate_shape(&a, &b), Ok(()));
    }

    // Finity
    #[test]
    fn validate_finity_rejects_empty_matrix() {
        let m: FloatMatrix<f32> = vec![];
        assert_eq!(validate_finity(&m), Err(Error::EmptyInput));
    }

    #[test]
    fn validate_finity_rejects_empty_first_column() {
        let m: FloatMatrix<f32> = vec![vec![]];
        assert_eq!(validate_finity(&m), Err(Error::EmptyInput));
    }

    #[test]
    fn validate_finity_rejects_ragged_columns() {
        let m = matrix(&[&[1.0, 2.0, 3.0], &[1.0, 2.0]]);
        assert_eq!(validate_finity(&m), Err(Error::RaggedRows));
    }

    #[test]
    fn validate_finity_reports_nan_position() {
        let m = matrix(&[&[1.0, 2.0], &[3.0, 4.0], &[f32::NAN, 5.0]]);
        assert_eq!(
            validate_finity(&m),
            Err(Error::NonFinite { row: 0, col: 2 })
        );
    }

    #[test]
    fn validate_finity_rejects_infinities() {
        for bad in [f32::INFINITY, f32::NEG_INFINITY] {
            let m = matrix(&[&[1.0, bad]]);
            assert_eq!(
                validate_finity(&m),
                Err(Error::NonFinite { row: 1, col: 0 })
            );
        }
    }

    #[test]
    fn validate_finity_accepts_finite_matrix() {
        let m = matrix(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_eq!(validate_finity(&m), Ok(()));
    }

    // Score
    #[test]
    fn compute_score_subtracts_ln_and_skips_non_positive() {
        let mut scores: FloatVector<f64> = vec![0.0, 0.0, 10.0];
        compute_score(&mut scores, [1.0, std::f64::consts::E, 0.0].into_iter());
        assert_close(scores[0], 0.0); // ln(1) = 0
        assert_close(scores[1], -1.0); // ln(e) = 1
        assert_close(scores[2], 10.0); // p = 0 
    }

    #[test]
    fn compute_score_leaves_tail_untouched_when_probs_shorter() {
        let mut scores: FloatVector<f64> = vec![1.0, 2.0];
        compute_score(&mut scores, std::iter::once(std::f64::consts::E));
        assert_close(scores[0], 0.0);
        assert_close(scores[1], 2.0);
    }

    // Skewness
    #[test]
    fn skewness_of_constant_vector_is_zero() {
        assert_close(compute_skewness(&vec![3.0f64; 5]), 0.0);
    }

    #[test]
    fn skewness_of_symmetric_vector_is_zero() {
        assert_close(compute_skewness(&vec![1.0f64, 2.0, 3.0]), 0.0);
    }
}
