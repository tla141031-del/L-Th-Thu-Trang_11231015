"""Part 1b - curated storage layout and the storage benchmark.

Raw layer   data/raw/financials/REPDTE=YYYYMMDD.parquet   (exactly what the API returned)
Curated     data/curated/financials/year=YYYY/data.parquet (typed, validated, zstd,
            sorted by CERT then REPDTE inside each year, row-group statistics on)
            data/curated/failures.parquet, institutions.parquet (typed dates)

Why this layout is measured in ``benchmark()``: every candidate format is written
from the same in-memory table and each query runs in a fresh Python process, so
the OS page cache and Python's allocator do not favour the format tested last.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from . import config as C

AMOUNT_COLS = [c for c in C.FIN_FIELDS if c not in ("CERT", "REPDTE", "STALP",
                                                       "ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ")]
RATIO_COLS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]


def _parse_us_date(col: str) -> pl.Expr:
    """FDIC tables mix 'M/D/YYYY' and 'MM/DD/YYYY'; both parse with %m/%d/%Y."""
    return pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias(col)


def read_raw_financials() -> pl.DataFrame:
    files = sorted((C.RAW / "financials").glob("REPDTE=*.parquet"))
    frames = []
    for f in files:
        df = pl.read_parquet(f)
        # old quarters can deliver an all-null column with dtype null/str: unify first
        frames.append(df.select([pl.col(c).cast(pl.Utf8) if c in ("REPDTE", "STALP")
                                 else pl.col(c).cast(pl.Float64, strict=False) for c in C.FIN_FIELDS]))
    return pl.concat(frames, how="vertical")


def typed_financials(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Enforce the schema and record what the casts did (reported in the DQ report)."""
    log = {}
    df = raw.with_columns(
        pl.col("CERT").cast(pl.Int32),
        pl.col("REPDTE").str.strptime(pl.Date, "%Y%m%d"),
        pl.col("STALP").cast(pl.Categorical),
    )
    for c in AMOUNT_COLS:
        non_int = df.filter(pl.col(c).is_not_null() & (pl.col(c) != pl.col(c).round(0))).height
        log[c] = {"non_integer_values": non_int}
        if non_int == 0:      # amounts are whole thousands of USD -> Int64 compresses better
            df = df.with_columns(pl.col(c).cast(pl.Int64))
    df = df.with_columns(pl.col("REPDTE").dt.year().cast(pl.Int16).alias("year"))
    return df.sort(["CERT", "REPDTE"]), log


def write_by_year(fin: pl.DataFrame, out: Path, compression: str = "zstd") -> None:
    """Hive layout year=YYYY/data.parquet; the key lives in the path, not inside the file,
    so pandas/pyarrow, Polars and Spark all read the same schema."""
    for (yr,), part in fin.partition_by("year", as_dict=True, maintain_order=True).items():
        d = out / f"year={yr}"
        d.mkdir(parents=True, exist_ok=True)
        part.drop("year").write_parquet(d / "data.parquet", compression=compression, statistics=True,
                                         row_group_size=50_000)


def build_curated() -> dict:
    raw = read_raw_financials()
    fin, cast_log = typed_financials(raw)
    n_dup = fin.select(pl.struct("CERT", "REPDTE").is_duplicated().sum()).item()
    assert n_dup == 0, f"{n_dup} duplicate (CERT, REPDTE) keys in raw data"
    out = C.CURATED / "financials"
    if out.exists():
        shutil.rmtree(out)
    write_by_year(fin, out)
    fails = (pl.read_parquet(C.RAW / "failures.parquet")
             .with_columns(_parse_us_date("FAILDATE"), pl.col("CERT").cast(pl.Int32, strict=False),
                           pl.col("COST").cast(pl.Float64, strict=False),
                           pl.col("QBFASSET").cast(pl.Float64, strict=False),
                           pl.col("QBFDEP").cast(pl.Float64, strict=False)))
    inst = (pl.read_parquet(C.RAW / "institutions.parquet")
            .with_columns(_parse_us_date("ESTYMD"), _parse_us_date("ENDEFYMD"), _parse_us_date("INSDATE"),
                          pl.col("CERT").cast(pl.Int32), pl.col("ACTIVE").cast(pl.Int8, strict=False),
                          pl.col("CHANGEC1").cast(pl.Int32, strict=False)))
    fails.write_parquet(C.CURATED / "failures.parquet", compression="zstd")
    inst.write_parquet(C.CURATED / "institutions.parquet", compression="zstd")
    schema = [{"column": k, "dtype": str(v)} for k, v in fin.schema.items()]
    info = {"financials_rows": fin.height, "partitions": fin["year"].n_unique(),
            "schema": schema, "cast_log": cast_log,
            "failures_rows": fails.height, "institutions_rows": inst.height}
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (C.TABLES / "storage_curated_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"curated financials: {fin.height:,} rows in {info['partitions']} year partitions")
    return info


def load_panel(columns: list[str] | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(C.CURATED / "financials", hive_partitioning=True)
    return lf.select(columns) if columns else lf


# --------------------------------------------------------------------------- benchmark
FORMATS = {
    "csv": "financials.csv",
    "csv_gzip": "financials.csv.gz",
    "parquet_none": "financials_none.parquet",
    "parquet_snappy": "financials_snappy.parquet",
    "parquet_zstd": "financials_zstd.parquet",
    "parquet_zstd_by_year": "by_year",
}

# Each query is a small Polars program run in a fresh interpreter.
QUERY_CODE = {
    "full_load": "df = {reader}.collect()",
    "agg_2cols": ("df = {reader}.group_by('year').agg(pl.col('ASSET').sum(), pl.col('NETINC').mean())"
                  ".collect()"),
    "filter_1yr_state": ("df = {reader}.filter((pl.col('year') == 2009) & (pl.col('STALP') == 'GA'))"
                         ".select('CERT','REPDTE','ASSET','EQ','NCLNLS').collect()"),
}


def _reader(fmt: str, path: Path) -> str:
    if fmt == "csv":
        return f"pl.scan_csv(r'{path}', try_parse_dates=True)"
    if fmt == "csv_gzip":   # polars cannot scan gzip lazily: it must decompress everything
        return f"pl.read_csv(r'{path}', try_parse_dates=True).lazy()"
    if fmt == "parquet_zstd_by_year":
        return f"pl.scan_parquet(r'{path}', hive_partitioning=True)"
    return f"pl.scan_parquet(r'{path}')"


CHILD = """
import resource, time, polars as pl
t0 = time.perf_counter()
{code}
t = time.perf_counter() - t0
print(t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, df.height, df.width)
"""


def _run_child(code: str) -> tuple[float, float, int, int]:
    """Run one query in a fresh interpreter. Peak memory is sampled from outside with psutil:
    ru_maxrss would be wrong here, because Linux keeps the parent's peak across fork/exec."""
    import threading
    import psutil
    proc = subprocess.Popen([sys.executable, "-c", CHILD.format(code=code)], stdout=subprocess.PIPE, text=True,
                            env={**os.environ, "POLARS_MAX_THREADS": str(os.cpu_count())})
    peak = [0.0]

    def sample():
        try:
            ps = psutil.Process(proc.pid)
            while proc.poll() is None:
                peak[0] = max(peak[0], ps.memory_info().rss / 2 ** 20)
                time.sleep(0.01)
        except psutil.Error:
            pass

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out, _ = proc.communicate()
    th.join()
    if proc.returncode:
        raise RuntimeError(code)
    t, _, h, w = out.split()
    return float(t), peak[0], int(h), int(w)


def _size_mb(p: Path) -> float:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2 ** 20
    return p.stat().st_size / 2 ** 20


def benchmark(repeats: int = 5) -> pl.DataFrame:
    import gzip
    import platform
    work = C.DATA / "bench_storage"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    fin = load_panel().collect().with_columns(pl.col("STALP").cast(pl.Utf8))
    write = {}
    for fmt, name in FORMATS.items():
        p = work / name
        t0 = time.perf_counter()
        if fmt == "csv":
            fin.write_csv(p)
        elif fmt == "csv_gzip":
            with gzip.open(p, "wb", compresslevel=6) as fh:
                fin.write_csv(fh)
        elif fmt == "parquet_zstd_by_year":
            write_by_year(fin, p)
        else:
            fin.write_parquet(p, compression={"none": "uncompressed"}.get(fmt.split("_")[1], fmt.split("_")[1]),
                             statistics=True, row_group_size=50_000)
        write[fmt] = time.perf_counter() - t0
    rows = []
    for fmt, name in FORMATS.items():
        p = work / name
        for q, tmpl in QUERY_CODE.items():
            times, mems = [], []
            for _ in range(repeats):
                t, mb, h, w = _run_child(tmpl.format(reader=_reader(fmt, p)))
                times.append(t)
                mems.append(mb)
            rows.append({"format": fmt, "query": q, "size_mb": _size_mb(p), "write_s": write[fmt],
                         "median_s": statistics.median(times), "min_s": min(times),
                         "peak_rss_mb": statistics.median(mems), "rows_out": h, "cols_out": w})
            print(f"{fmt:22s} {q:18s} {statistics.median(times):7.3f}s  {statistics.median(mems):7.0f} MB")
    res = pl.DataFrame(rows)
    base = res.filter(pl.col("format") == "csv").select("query", pl.col("median_s").alias("csv_s"),
                                                         pl.col("size_mb").alias("csv_mb"))
    res = (res.join(base, on="query")
           .with_columns((pl.col("csv_s") / pl.col("median_s")).alias("speedup_vs_csv"),
                         (pl.col("size_mb") / pl.col("csv_mb")).alias("size_vs_csv"))
           .drop("csv_s", "csv_mb"))
    res.write_csv(C.TABLES / "storage_benchmark.csv")
    env = {"rows": fin.height, "columns": fin.width, "repeats": repeats, "cpu_count": os.cpu_count(),
           "python": platform.python_version(), "polars": pl.__version__, "platform": platform.platform(),
           "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (C.TABLES / "storage_benchmark_env.json").write_text(json.dumps(env, indent=1))
    _plot_storage(res)
    shutil.rmtree(work)
    return res


def _plot_storage(res: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(FORMATS)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    sz = res.filter(pl.col("query") == "full_load").to_pandas().set_index("format").loc[order]
    axes[0].barh(order, sz["size_mb"], color="#4C72B0")
    axes[0].set_title("Size on disk (MB)")
    axes[0].invert_yaxis()
    for ax, q, ttl in zip(axes[1:], QUERY_CODE, ["Full load (s)", "Aggregate 2 columns (s)",
                                                 "Filter 1 year + 1 state (s)"]):
        d = res.filter(pl.col("query") == q).to_pandas().set_index("format").loc[order]
        ax.barh(order, d["median_s"], color="#DD8452")
        ax.set_title(ttl)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_yticklabels([])
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / "storage_benchmark.png", dpi=170)
    plt.close(fig)
