#[derive(Debug, PartialEq)]
pub enum Error {
    /// Input matrix X is empty.
    EmptyInput,
    /// Rows of X have different lengths.
    RaggedRows,
    /// Number of features passed to predict does not match the number seen during fit.
    FeatureMismatch {
        /// Number of features seen during fit.
        expected: usize,
        /// Number of features passed to predict.
        got: usize,
    },
    /// Input contains a non-finite value.
    NonFinite {
        /// Row index of the value.
        row: usize,
        /// Column index of the value.
        col: usize,
    },
    Io {
        kind: std::io::ErrorKind,
    },
    Decode,
    /// Decoded model violates fit invariants.
    CorruptedModel {
        /// Which invariant is broken.
        kind: CorruptionKind,
    },
}

/// Invariant broken in a loaded model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum CorruptionKind {
    /// Model has no features or no training values.
    Empty,
    /// Features have different numbers of training values.
    RaggedFeatures,
    /// Training values contain a non-finite value.
    NonFiniteData,
    /// Number of skewnesses differs from the number of features.
    SkewnessCountMismatch,
    /// Skewnesses contain a non-finite value.
    NonFiniteSkewness,
    /// Training values of a feature are not sorted.
    UnsortedFeature,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::EmptyInput => write!(f, "input is empty"),
            Error::RaggedRows => {
                write!(f, "feature matrix has different sizes of rows")
            }
            Error::FeatureMismatch { expected, got } => {
                write!(f, "model fitted on {expected} features, got {got}")
            }
            Error::NonFinite { row, col } => write!(f, "non-finite value at ({row}, {col})"),
            Error::Io { kind } => write!(f, "io error, kind: {kind}"),
            Error::Decode => write!(f, "failed to decode model"),
            Error::CorruptedModel { kind } => write!(f, "model file is corrupted: {kind}"),
        }
    }
}

impl std::fmt::Display for CorruptionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorruptionKind::Empty => write!(f, "no training data"),
            CorruptionKind::RaggedFeatures => {
                write!(f, "features have different numbers of training values")
            }
            CorruptionKind::NonFiniteData => write!(f, "non-finite training value"),
            CorruptionKind::SkewnessCountMismatch => {
                write!(f, "number of skewnesses does not match number of features")
            }
            CorruptionKind::NonFiniteSkewness => write!(f, "non-finite skewness"),
            CorruptionKind::UnsortedFeature => write!(f, "training values are not sorted"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io { kind: e.kind() }
    }
}

impl std::error::Error for Error {}
