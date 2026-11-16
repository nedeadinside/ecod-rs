# ecod-rs

The ECOD outlier detector ([Li et al., 2022](https://arxiv.org/abs/2201.00382)) in Rust, with Python bindings.

**[Documentation and benchmark](https://nedeadinside.github.io/ecod-rs/)**

## Why

The goal is ECOD with a normal API, which PyOD doesn't have.

In PyOD, `fit` only stores the training data, and `decision_function` concatenates it with the new samples and recomputes everything. So the score of a sample depends on the rest of the batch you pass in. This was reported in [#401](https://github.com/yzhao062/pyod/issues/401) and [#426](https://github.com/yzhao062/pyod/issues/426). Yue Zhao, PyOD's maintainer and an ECOD co-author, answered that this is intended, since outlier detection is transductive. Nothing has changed since.

Here you get both: `InductiveECOD` scores each sample independently, `ECOD` behaves like PyOD.

## Install

```bash
pip install ecod-rs
```

```python
from ecod_rs import InductiveECOD

model = InductiveECOD(contamination=0.1).fit(X_train)
labels = model.predict(X_test)  # 1 is an outlier
```

## Development

```bash
uv sync
uv run pytest
cargo test
```

Benchmark: `uv run --group bench python benchmark/bench.py`.

## License

MIT
