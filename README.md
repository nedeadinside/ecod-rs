# ecod-rs

The ECOD outlier detector ([Li et al., 2022](https://arxiv.org/abs/2201.00382)) in Rust, with Python bindings.

## Why

The goal is ECOD with a normal API, which PyOD doesn't have.

In PyOD, `fit` only stores the training data, and `decision_function` concatenates it with the new samples and recomputes everything. So the score of a sample depends on the rest of the batch you pass in. This was reported in [#401](https://github.com/yzhao062/pyod/issues/401) and [#426](https://github.com/yzhao062/pyod/issues/426). Yue Zhao, PyOD's maintainer and an ECOD co-author, answered that this is intended, since outlier detection is transductive. Nothing has changed since.

Here you get both:

- `InductiveECOD` fits the per-feature ECDFs once and scores each sample independently.
- `ECOD` behaves like PyOD.

## Install

Requires Python 3.12+:

```bash
pip install ecod-rs
```

## Usage

```python
import numpy as np
from ecod_rs import InductiveECOD

rng = np.random.default_rng(0)
X_train = rng.normal(size=(1000, 10))
X_test = rng.normal(size=(200, 10))

model = InductiveECOD(contamination=0.1).fit(X_train)

scores = model.decision_function(X_test)  # higher means more anomalous
labels = model.predict(X_test)            # 1 is an outlier
proba = model.predict_proba(X_test)       # shape (n, 2): [inlier, outlier]

model.save("model.bin")
model = InductiveECOD.load("model.bin")
```

`ECOD` has the same interface. Input is a numpy array or a list of lists.

The `method` argument sets the scoring: `"max"` (default) follows the paper, `"pyod-max"` reproduces PyOD's numbers, `"left"`, `"right"` and `"auto"` use a single tail. `predict_proba` takes `"linear"` or `"unify"`, as in PyOD.

## Benchmark

Median speedup over PyOD's ECOD on random data and 47 ADBench datasets, single-threaded:

| | ECOD | InductiveECOD |
|---|---|---|
| fit | 2.7x | 2.7x |
| score | 5.9x | 7.5x |
| fit + score | 4.3x | 4.7x |

Full results are at [nedeadinside.github.io/ecod-rs](https://nedeadinside.github.io/ecod-rs/). To rerun: `uv run --group bench python benchmark/bench.py`.

## Development

```bash
uv sync
uv run pytest
cargo test
```

## License

MIT
