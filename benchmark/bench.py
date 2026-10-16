import csv
import gc
import json
import multiprocessing
import os
import statistics
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from ecod_rs import ECOD, InductiveECOD
from pyod.models.ecod import ECOD as PyodECOD
from sklearn.metrics import roc_auc_score

SIZES = [
    (1_000, 10),
    (10_000, 10),
    (100_000, 10),
    (1_000_000, 10),
    (10_000, 100),
    (100_000, 100),
]
REPEATS = 5
ADBENCH_DIR = Path(__file__).parent / "adbench"
ADBENCH_API = (
    "https://api.github.com/repos/Minqi824/ADBench/contents/adbench/datasets/Classical"
)
OUT_CSV = Path(__file__).parent / "results.csv"
OUT_HTML = Path(__file__).parent.parent / "docs" / "index.html"

LIBS = ["ECOD", "InductiveECOD", "pyod-ECOD"]
PHASES = [
    ("fit", ""),
    ("score", ""),
    ("pipeline", ""),
    ("convert", ""),
    ("save", "default"),
    ("save", "compressed"),
    ("load", "default"),
    ("load", "compressed"),
]
METRICS = ["wall_median_s", "peak_mb", "disk_mb", "auc"]

RS = {"ECOD": ECOD, "InductiveECOD": InductiveECOD}
RS_LEVEL = {"default": 3, "compressed": 19}
PYOD_COMPRESS = {"default": 0, "compressed": 3}

INTRO = f"""<h1>ecod-rs benchmark</h1>
<p><a href="https://github.com/nedeadinside/ecod-rs">ecod-rs</a> is a Rust implementation of the ECOD outlier detector
(<a href="https://arxiv.org/abs/2201.00382">Li et al., 2022</a>) with Python bindings.
Here it is compared with ECOD from <a href="https://github.com/yzhao062/pyod">PyOD</a>.</p>
<p>Data is random normal noise of several sizes (rows x features) and the classical datasets from
<a href="https://github.com/Minqi824/ADBench">ADBench</a>. Each dataset is split 60/40 into train and test.
Every measurement runs in a fresh single-threaded process, times are taken over {REPEATS} runs.</p>
<p>What is measured:</p>
<ul>
<li>fit: training, including conversion from numpy;</li>
<li>score: scoring the test part with a fitted model;</li>
<li>pipeline: fit and score together;</li>
<li>convert: passing data into ecod-rs up to input validation;</li>
<li>save and load: writing and reading a model file. The default and compressed variants are
zstd levels 3 and 19 for ecod-rs, joblib levels 0 and 3 for PyOD.</li>
</ul>
"""


def download_adbench() -> None:
    """
    Download ADBench datasets into ADBENCH_DIR.
    """
    ADBENCH_DIR.mkdir(exist_ok=True)
    with urllib.request.urlopen(ADBENCH_API) as resp:
        files = json.load(resp)
    for f in files:
        path = ADBENCH_DIR / f["name"]
        if path.exists():
            continue
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(f["download_url"], part)
        part.rename(path)
        print("downloaded", f["name"], flush=True)


def load_data(
    source: tuple[int, int] | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Build a 60/40 train/test split.

    :param source: Synthetic size as rows and features, or path to an ADBench npz file.
    :return: Train matrix, test matrix and test labels when known.
    """
    if isinstance(source, tuple):
        x = np.random.default_rng(0).normal(size=source)
        y = None
    else:
        with np.load(source) as f:
            x, y = f["X"].astype(np.float64), f["y"]
        idx = np.random.default_rng(0).permutation(len(x))
        x, y = x[idx], y[idx]

    cut = int(len(x) * 0.6)
    x_train, x_test = x[:cut], x[cut:]
    if y is not None:
        y = y[cut:]
    return np.ascontiguousarray(x_train), np.ascontiguousarray(x_test), y


def dataset_name(source: tuple[int, int] | str) -> str:
    """
    Short label of a data source.

    :param source: Synthetic size or path to an npz file.
    :return: Label for reports.
    """
    return (
        f"{source[0]}x{source[1]}" if isinstance(source, tuple) else Path(source).stem
    )


def fit(lib: str, x: np.ndarray):
    """
    Fit a fresh model on numpy data, including any input conversion.

    :param lib: Library label.
    :param x: Train matrix.
    :return: Fitted model.
    """
    if lib == "pyod-ECOD":
        return PyodECOD(n_jobs=1).fit(x)
    return RS[lib](method="pyod-max").fit(x.T.tolist())


def score(lib: str, model, x: np.ndarray):
    """
    Score numpy data.

    :param lib: Library label.
    :param model: Fitted model.
    :param x: Test matrix.
    :return: Outlier scores.
    """
    return model.decision_function(x if lib == "pyod-ECOD" else x.T.tolist())


def save(lib: str, model, path: Path, variant: str) -> None:
    """
    Save a model.

    :param lib: Library label.
    :param model: Fitted model.
    :param path: Target file.
    :param variant: Compression variant.
    """
    if lib == "pyod-ECOD":
        joblib.dump(model, path, compress=PYOD_COMPRESS[variant])
    else:
        model.save(path, compression_level=RS_LEVEL[variant])


def load(lib: str, path: Path):
    """
    Load a model saved by the same library.

    :param lib: Library label.
    :param path: Model file.
    :return: Loaded model.
    """
    return joblib.load(path) if lib == "pyod-ECOD" else RS[lib].load(path)


def make_action(
    lib: str,
    phase: str,
    variant: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    path: Path,
):
    """
    Prepare state for a phase and return the callable to measure.

    :param lib: Library label.
    :param phase: Phase name.
    :param variant: Compression variant for save and load.
    :param x_train: Train matrix.
    :param x_test: Test matrix.
    :param path: Model file for save and load.
    :return: Zero-argument callable.
    """
    if phase == "fit":
        return lambda: fit(lib, x_train)
    if phase == "pipeline":
        return lambda: score(lib, fit(lib, x_train), x_test)
    if phase == "convert":
        bad = x_train.copy()
        bad[0, 0] = np.nan

        def convert():
            """
            Pass input through the boundary and stop at validation.

            :raises AssertionError: When input with NaN is accepted.
            """
            try:
                RS[lib]().fit(bad.T.tolist())
            except ValueError:
                return
            raise AssertionError("NaN input was accepted")

        return convert

    model = fit(lib, x_train)
    if phase == "score":
        return lambda: score(lib, model, x_test)
    if phase == "save":
        return lambda: save(lib, model, path, variant)
    save(lib, model, path, variant)
    return lambda: load(lib, path)


def proc_mb(field: str) -> float:
    """
    Read a memory field of the current process.

    :param field: Field name from the process status file.
    :raises KeyError: When the field is missing.
    :return: Value in megabytes.
    """
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    raise KeyError(field)


def measure(action) -> tuple[float, float]:
    """
    Run an action once and record its time and memory.

    :param action: Action function.
    :return: Wall seconds and peak memory growth in megabytes.
    """
    gc.collect()
    rss0 = proc_mb("VmRSS")
    Path("/proc/self/clear_refs").write_text("5")
    wall0 = time.perf_counter()
    result = action()
    wall = time.perf_counter() - wall0
    peak = proc_mb("VmHWM") - rss0
    del result
    return wall, peak


def run(lib: str, source: tuple[int, int] | str, phase: str, variant: str) -> dict:
    """
    Benchmark one library on one dataset and phase in the current process.

    :param lib: Library label.
    :param source: Synthetic size or path to an npz file.
    :param phase: Phase name.
    :param variant: Compression variant for save and load.
    :return: Report row.
    """
    x_train, x_test, y_test = load_data(source)
    row = dict.fromkeys(METRICS)
    row |= {
        "dataset": dataset_name(source),
        "lib": lib,
        "phase": phase,
        "variant": variant,
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        action = make_action(lib, phase, variant, x_train, x_test, path)

        _, row["peak_mb"] = measure(action)
        row["wall_median_s"] = statistics.median(
            measure(action)[0] for _ in range(REPEATS)
        )
        if phase == "save":
            row["disk_mb"] = path.stat().st_size / 2**20
        if phase == "score" and y_test is not None:
            row["auc"] = roc_auc_score(y_test, np.nan_to_num(np.asarray(action())))
    return row


def fmt(value: str) -> str:
    """
    Format a metric from the CSV for a table cell.

    :param value: Metric value as CSV text, empty when missing.
    :return: Short text.
    """
    if not value:
        return "-"
    return np.format_float_positional(
        float(value), precision=4, unique=False, fractional=False, trim="-"
    )


def write_csv(rows: list[dict]) -> None:
    """
    Write report rows to CSV.

    :param rows: Report rows.
    """
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "lib", "phase", "variant", *METRICS]
        )
        writer.writeheader()
        writer.writerows(rows)


def html_table(
    head: list[str], groups: dict[str, list[str]], body: list[list[str]]
) -> str:
    """
    Render a table with a two-row header.

    :param head: Leading column names spanning both header rows.
    :param groups: Group titles mapped to their column names.
    :param body: Rows of cell texts.
    :return: HTML table.
    """
    top = "".join(f'<th rowspan="2">{h}</th>' for h in head) + "".join(
        f'<th colspan="{len(cols)}">{g}</th>' for g, cols in groups.items()
    )
    sub = "".join(f"<th>{c}</th>" for cols in groups.values() for c in cols)
    lines = ["<table>", f"<tr>{top}</tr>", f"<tr>{sub}</tr>"]
    lines += ["<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body]
    lines.append("</table>")
    return "\n".join(lines)


def write_html() -> None:
    """
    Build the report page from the results CSV.
    """
    with OUT_CSV.open(newline="") as f:
        rows = {
            (r["dataset"], r["lib"], r["phase"], r["variant"]): r
            for r in csv.DictReader(f)
        }
    present = {key[0] for key in rows}
    synth = [d for d in map(dataset_name, SIZES) if d in present]
    adbench = sorted(present - set(synth), key=lambda d: int(d.split("_")[0]))

    def get(dataset: str, lib: str, phase: str, variant: str, metric: str) -> str:
        """
        Look up a metric in the results.

        :return: Metric as CSV text, empty when missing.
        """
        row = rows.get((dataset, lib, phase, variant))
        return row[metric] if row else ""

    def speedup(dataset: str, lib: str, phase: str, variant: str) -> float | None:
        """
        How many times a library is faster than pyod-ECOD on one measurement.

        :return: Ratio of median times, nothing when either is missing.
        """
        own = get(dataset, lib, phase, variant, "wall_median_s")
        base = get(dataset, "pyod-ECOD", phase, variant, "wall_median_s")
        return float(base) / float(own) if own and base else None

    def timing(dataset: str, lib: str, phase: str, variant: str = "") -> str:
        """
        Median time with the speedup over pyod-ECOD in brackets.

        :return: Cell text.
        """
        text = fmt(get(dataset, lib, phase, variant, "wall_median_s"))
        ratio = None if lib == "pyod-ECOD" else speedup(dataset, lib, phase, variant)
        return text if ratio is None else f"{text} (×{ratio:.1f})"

    def summary(lib: str, phase: str, variant: str) -> str:
        """
        Median and range of the speedup over all datasets.

        :return: Cell text.
        """
        ratios = [r for d in present if (r := speedup(d, lib, phase, variant))]
        if not ratios:
            return "-"
        return (
            f"×{statistics.median(ratios):.1f} ({min(ratios):.1f} to {max(ratios):.1f})"
        )

    def speed(datasets: list[str]) -> str:
        """
        Timing table for fit, score, pipeline and convert.

        :return: HTML table.
        """
        phases = ["fit", "score", "pipeline"]
        return html_table(
            ["dataset"],
            {**{f"{p}, s": LIBS for p in phases}, "convert, s": list(RS)},
            [
                [
                    d,
                    *[timing(d, lib, p) for p in phases for lib in LIBS],
                    *[fmt(get(d, lib, "convert", "", "wall_median_s")) for lib in RS],
                ]
                for d in datasets
            ],
        )

    def memory(datasets: list[str]) -> str:
        """
        Peak memory table for fit and score.

        :return: HTML table.
        """
        phases = ["fit", "score"]
        return html_table(
            ["dataset"],
            {f"{p}, MB": LIBS for p in phases},
            [
                [
                    d,
                    *[
                        fmt(get(d, lib, p, "", "peak_mb"))
                        for p in phases
                        for lib in LIBS
                    ],
                ]
                for d in datasets
            ],
        )

    persistence = html_table(
        ["dataset", "variant"],
        {"save, s": LIBS, "load, s": LIBS, "file, MB": LIBS},
        [
            [
                d,
                v,
                *[timing(d, lib, "save", v) for lib in LIBS],
                *[timing(d, lib, "load", v) for lib in LIBS],
                *[fmt(get(d, lib, "save", v, "disk_mb")) for lib in LIBS],
            ]
            for d in synth
            for v in ("default", "compressed")
        ],
    )
    quality = html_table(
        ["dataset"],
        {"ROC AUC": LIBS},
        [[d, *[fmt(get(d, lib, "score", "", "auc")) for lib in LIBS]] for d in adbench],
    )
    overview = html_table(
        ["phase", "variant"],
        {"times faster than pyod-ECOD": list(RS)},
        [
            [p, v, *[summary(lib, p, v) for lib in RS]]
            for p, v in PHASES
            if p != "convert"
        ],
    )

    parts = [
        '<!doctype html>\n<meta charset="utf-8">\n<title>ecod-rs benchmark</title>',
        "<style>table { border-collapse: collapse; } th, td { border: 1px solid; }</style>",
        INTRO,
        "<h2>Summary</h2>",
        (
            "<p>How many times ecod-rs is faster than pyod-ECOD: median over all "
            f"{len(present)} datasets, the range in brackets. Below 1 means slower.</p>"
        ),
        overview,
        "<h2>Speed</h2>",
        (
            "<p>Median time in seconds. In brackets, how many times faster than "
            "pyod-ECOD. PyOD has no separate convert step.</p>"
        ),
        speed(synth),
        f"<details>\n<summary>ADBench datasets</summary>\n{speed(adbench)}\n</details>",
        "<h2>Memory</h2>",
        "<p>How much the process memory peak grew during the run, in MB.</p>",
        memory(synth),
        f"<details>\n<summary>ADBench datasets</summary>\n{memory(adbench)}\n</details>",
        "<h2>Saving and loading</h2>",
        persistence,
        "<h2>Quality</h2>",
        "<p>ROC AUC on the ADBench test part.</p>",
        quality,
    ]
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    download_adbench()
    os.environ |= {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    sources = list(SIZES) + sorted(map(str, ADBENCH_DIR.glob("*.npz")))
    tasks = [
        (lib, source, phase, variant)
        for source in sources
        for lib in LIBS
        for phase, variant in PHASES
        if not (phase == "convert" and lib == "pyod-ECOD")
    ]

    rows = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, max_tasks_per_child=1
    ) as pool:
        for row in pool.map(run, *zip(*tasks)):
            print(
                row["dataset"],
                row["lib"],
                row["phase"],
                row["variant"],
                f"{row['wall_median_s']:.4g}s",
                flush=True,
            )
            rows.append(row)

    write_csv(rows)
    write_html()
