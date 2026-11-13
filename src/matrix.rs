use std::borrow::Cow;

use crate::errors::Error;

/// Dense 2-D view over a flat buffer with shape (n_samples, n_features), like a numpy array.
///
/// Data is either owned or borrowed.
pub struct Matrix<'a, T: Clone> {
    data: Cow<'a, [T]>,
    shape: [usize; 2],
    /// Distance in elements between neighbours
    strides: [usize; 2],
}

/// Number of buffer elements a view spans.
///
/// Returns None when the size overflows usize.
pub fn span(shape: [usize; 2], strides: [usize; 2]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    let last_row = (shape[0] - 1).checked_mul(strides[0])?;
    let last_col = (shape[1] - 1).checked_mul(strides[1])?;
    last_row.checked_add(last_col)?.checked_add(1)
}

impl<'a, T: Copy> Matrix<'a, T> {
    /// Views borrowed data with the given element strides.
    pub fn borrowed(data: &'a [T], shape: [usize; 2], strides: [usize; 2]) -> Result<Self, Error> {
        Self::new(Cow::Borrowed(data), shape, strides)
    }

    fn new(data: Cow<'a, [T]>, shape: [usize; 2], strides: [usize; 2]) -> Result<Self, Error> {
        if shape.contains(&0) {
            return Err(Error::EmptyInput);
        }
        assert!(
            strides.iter().all(|&s| s > 0),
            "matrix strides must be positive"
        );
        assert!(
            span(shape, strides).is_some_and(|n| n <= data.len()),
            "matrix view is out of buffer bounds"
        );
        Ok(Matrix {
            data,
            shape,
            strides,
        })
    }

    pub fn nrows(&self) -> usize {
        self.shape[0]
    }

    pub fn ncols(&self) -> usize {
        self.shape[1]
    }

    /// Iterates over feature columns, each column yields its values in row order.
    pub fn cols(&self) -> impl Iterator<Item = impl ExactSizeIterator<Item = T> + Clone> {
        (0..self.shape[1]).map(move |j| {
            self.data[j * self.strides[1]..]
                .iter()
                .step_by(self.strides[0])
                .take(self.shape[0])
                .copied()
        })
    }
}

impl<T: Copy> Matrix<'static, T> {
    /// Takes ownership of C-order data.
    pub fn from_vec(data: Vec<T>, shape: [usize; 2]) -> Result<Self, Error> {
        assert_eq!(
            data.len(),
            shape[0] * shape[1],
            "data length does not match shape"
        );
        Self::new(Cow::Owned(data), shape, [shape[1], 1])
    }

    /// Copies rows of nested vectors into an owned C-order matrix.
    pub fn from_rows(rows: Vec<Vec<T>>) -> Result<Self, Error> {
        let n_cols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != n_cols) {
            return Err(Error::RaggedRows);
        }
        Self::from_vec(rows.concat(), [rows.len(), n_cols])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols(m: &Matrix<'_, f64>) -> Vec<Vec<f64>> {
        m.cols().map(Iterator::collect).collect()
    }

    fn rows() -> Vec<Vec<f64>> {
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    }

    #[test]
    fn from_rows_reads_columns() {
        let m = Matrix::from_rows(rows()).unwrap();
        assert_eq!((m.nrows(), m.ncols()), (2, 3));
        assert_eq!(
            cols(&m),
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
    }

    #[test]
    fn from_rows_rejects_ragged_rows() {
        let r = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(matches!(Matrix::from_rows(r), Err(Error::RaggedRows)));
    }

    #[test]
    fn from_rows_rejects_empty_input() {
        for r in [vec![], vec![vec![]]] {
            assert!(matches!(
                Matrix::<f64>::from_rows(r),
                Err(Error::EmptyInput)
            ));
        }
    }

    #[test]
    fn borrowed_fortran_order_matches_rows() {
        let data = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let m = Matrix::borrowed(&data, [2, 3], [1, 2]).unwrap();
        assert_eq!(cols(&m), cols(&Matrix::from_rows(rows()).unwrap()));
    }

    #[test]
    fn borrowed_view_skips_gaps() {
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let m = Matrix::borrowed(&data, [2, 3], [6, 2]).unwrap();
        assert_eq!(
            cols(&m),
            vec![vec![0.0, 6.0], vec![2.0, 8.0], vec![4.0, 10.0]]
        );
    }

    #[test]
    #[should_panic(expected = "out of buffer bounds")]
    fn borrowed_rejects_view_past_buffer() {
        let data = [1.0; 5];
        let _ = Matrix::borrowed(&data, [2, 3], [3, 1]);
    }

    #[test]
    fn span_counts_first_to_last_element() {
        assert_eq!(span([2, 3], [6, 2]), Some(11));
        assert_eq!(span([0, 3], [3, 1]), Some(0));
        assert_eq!(span([2, 2], [usize::MAX, 1]), None);
    }
}
