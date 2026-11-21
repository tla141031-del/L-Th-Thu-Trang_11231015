"""Part 3 - one core workload in five engines, benchmarked fairly.

Workload (the heart of the pipeline, identical semantics in every engine):
  1. read 13 columns of the curated Financials panel and the failures list;
  2. join failures by CERT and build the 4-quarter label y;
  3. de-cumulate NETINC, INTINC, EINTEXP (YTD -> quarterly, gap-aware);
  4. five ratios with safe division (null if the denominator <= 0);
  5. per bank: gap-aware 4-quarter change of the NPL ratio and a 4-report
     rolling mean of ROA (min 3 values);
  6. per quarter: percentile rank of the NPL ratio (average ties, nulls ignored);
  7. materialise, then return checksums (rows, per-column sum and non-null count).

Engines: pandas (baseline), Polars eager, Polars lazy (in-memory engine),
Polars lazy with the streaming engine, PySpark local[*]. Every run is a fresh
Python process; peak memory is the resident set of the process tree sampled
every 50 ms (so it includes the Spark JVM). Scale x k stacks k copies of the
panel with shifted CERTs (real values, labelled as replicated).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config as C

COLS = ["CERT", "REPDTE", "STALP", "ASSET", "DEP", "EQ", "LNLSGR", "LNLSNET", "NCLNLS",
        "NETINC", "INTINC", "EINTEXP", "year"]
YTD = ["NETINC", "INTINC", "EINTEXP"]
OUT = ["y", "equity_to_assets", "npl_ratio", "roa_q", "nim_q", "loans_to_deposits",
       "d4_npl_ratio", "roa_mean4", "npl_rank_q"]
ENGINES = ["pandas", "polars_eager", "polars_lazy", "polars_streaming", "pyspark"]
CERT_SHIFT = 1_000_000


# --------------------------------------------------------------------------- shared input
def _stacked_parquet(scale: int) -> Path:
    """x1 = the curated panel itself; x k = k shifted copies written once to one file."""
    import polars as pl
    if scale == 1:
        return C.CURATED / "financials"
    p = C.DATA / "bench_processing" / f"financials_x{scale}.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        base = pl.scan_parquet(C.CURATED / "financials", hive_partitioning=True).select(COLS).collect()
        pl.concat([base.with_columns(pl.col("CERT") + k * CERT_SHIFT) for k in range(scale)]) \
          .write_parquet(p, compression="zstd", statistics=True)
    return p


def _failures_path(scale: int) -> Path:
    import polars as pl
    p = C.DATA / "bench_processing" / f"failures_x{scale}.parquet"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        f = (pl.read_parquet(C.CURATED / "failures.parquet")
             .filter((pl.col("RESTYPE") == "FAILURE") & pl.col("CERT").is_not_null())
             .sort("FAILDATE").unique("CERT", keep="first").select("CERT", pl.col("FAILDATE").alias("fail_date")))
        pl.concat([f.with_columns(pl.col("CERT") + k * CERT_SHIFT) for k in range(scale)]).write_parquet(p)
    return p


def _checks_polars(df) -> dict:
    import polars as pl
    out = {"rows": df.height}
    for c in OUT:
        s = df[c].cast(pl.Float64)
        out[f"{c}__sum"] = float(s.sum() or 0.0)
        out[f"{c}__n"] = int(s.is_not_null().sum())
    return out


# --------------------------------------------------------------------------- Polars
def polars_plan(fin, fails):
    import polars as pl

    def div(a, b):
        return pl.when(b > 0).then(a / b).otherwise(None)

    qi = pl.col("REPDTE").dt.year().cast(pl.Int32) * 4 + pl.col("REPDTE").dt.quarter().cast(pl.Int32)
    x = (fin.select(COLS).join(fails, on="CERT", how="left").sort("CERT", "REPDTE")
         .with_columns(qi.alias("_qi"),
                       pl.col("REPDTE").dt.offset_by("12mo").dt.month_end().alias("_end")))
    prev_ok = (pl.col("_qi") - pl.col("_qi").shift(1).over("CERT")) == 1
    q1 = pl.col("REPDTE").dt.quarter() == 1
    x = x.with_columns(
        ((pl.col("fail_date") > pl.col("REPDTE")) & (pl.col("fail_date") <= pl.col("_end"))).fill_null(False)
        .cast(pl.Int8).alias("y"),
        *[pl.when(q1).then(pl.col(c)).when(prev_ok).then(pl.col(c) - pl.col(c).shift(1).over("CERT"))
          .cast(pl.Float64).alias(f"{c}_q") for c in YTD])
    x = x.with_columns(
        div(pl.col("EQ"), pl.col("ASSET")).alias("equity_to_assets"),
        div(pl.col("NCLNLS"), pl.col("LNLSGR")).alias("npl_ratio"),
        div(4 * pl.col("NETINC_q"), pl.col("ASSET")).alias("roa_q"),
        div(4 * (pl.col("INTINC_q") - pl.col("EINTEXP_q")), pl.col("ASSET")).alias("nim_q"),
        div(pl.col("LNLSNET"), pl.col("DEP")).alias("loans_to_deposits"))
    x = x.with_columns(
        pl.when((pl.col("_qi") - pl.col("_qi").shift(4).over("CERT")) == 4)
        .then(pl.col("npl_ratio") - pl.col("npl_ratio").shift(4).over("CERT")).alias("d4_npl_ratio"),
        pl.col("roa_q").rolling_mean(4, min_samples=3).over("CERT").alias("roa_mean4"),
        (pl.col("npl_ratio").rank("average").over("REPDTE")
         / pl.col("npl_ratio").count().over("REPDTE")).alias("npl_rank_q"))
    return x.select("CERT", "REPDTE", *OUT)


def run_polars(mode: str, scale: int) -> tuple[dict, object]:
    import polars as pl
    src, fp = _stacked_parquet(scale), _failures_path(scale)
    if mode == "polars_eager":       # same expressions, executed step by step on DataFrames
        fin = pl.read_parquet(src, hive_partitioning=True) if scale == 1 else pl.read_parquet(src)
        res = polars_plan(fin, pl.read_parquet(fp))
    else:
        fin = pl.scan_parquet(src, hive_partitioning=True) if scale == 1 else pl.scan_parquet(src)
        lf = polars_plan(fin, pl.scan_parquet(fp))
        res = lf.collect(engine="streaming" if mode == "polars_streaming" else "in-memory")
    return _checks_polars(res), res


# --------------------------------------------------------------------------- pandas
def run_pandas(scale: int) -> dict:
    import numpy as np
    import pandas as pd
    src = _stacked_parquet(scale)
    fin = pd.read_parquet(src, columns=COLS)
    fails = pd.read_parquet(_failures_path(scale))
    fin["STALP"] = fin["STALP"].astype(str)
    x = fin.merge(fails, on="CERT", how="left").sort_values(["CERT", "REPDTE"], kind="stable").reset_index(drop=True)
    x["REPDTE"] = pd.to_datetime(x["REPDTE"])
    x["fail_date"] = pd.to_datetime(x["fail_date"])
    x["_qi"] = x["REPDTE"].dt.year * 4 + x["REPDTE"].dt.quarter
    end = x["REPDTE"] + pd.offsets.MonthEnd(12)
    x["y"] = ((x["fail_date"] > x["REPDTE"]) & (x["fail_date"] <= end)).astype("int8")
    g = x.groupby("CERT", sort=False)
    prev_ok = (x["_qi"] - g["_qi"].shift(1)) == 1
    q1 = x["REPDTE"].dt.quarter == 1
    for c in YTD:
        v = x[c].astype("float64")
        x[f"{c}_q"] = np.where(q1, v, np.where(prev_ok, v - g[c].shift(1).astype("float64"), np.nan))

    def div(a, b):
        a, b = a.astype("float64"), b.astype("float64")
        return (a / b).where(b > 0)

    x["equity_to_assets"] = div(x["EQ"], x["ASSET"])
    x["npl_ratio"] = div(x["NCLNLS"], x["LNLSGR"])
    x["roa_q"] = div(4 * x["NETINC_q"], x["ASSET"])
    x["nim_q"] = div(4 * (x["INTINC_q"] - x["EINTEXP_q"]), x["ASSET"])
    x["loans_to_deposits"] = div(x["LNLSNET"], x["DEP"])
    g = x.groupby("CERT", sort=False)
    ok4 = (x["_qi"] - g["_qi"].shift(4)) == 4
    x["d4_npl_ratio"] = (x["npl_ratio"] - g["npl_ratio"].shift(4)).where(ok4)
    x["roa_mean4"] = g["roa_q"].rolling(4, min_periods=3).mean().reset_index(level=0, drop=True)
    gq = x.groupby("REPDTE")["npl_ratio"]
    x["npl_rank_q"] = gq.rank(method="average") / gq.transform("count")
    out = {"rows": len(x)}
    for c in OUT:
        s = x[c].astype("float64")
        out[f"{c}__sum"] = float(np.nansum(s.to_numpy()))
        out[f"{c}__n"] = int(s.notna().sum())
    return out


# --------------------------------------------------------------------------- PySpark
def spark_session(app: str = "bft65b"):
    from pyspark.sql import SparkSession
    # FDIC_SPARK_MASTER / FDIC_SPARK_SHUFFLE are only set by experiments.py (core and partition scans)
    return (SparkSession.builder.master(os.environ.get("FDIC_SPARK_MASTER", "local[*]")).appName(app)
            .config("spark.driver.memory", "6g")
            .config("spark.sql.shuffle.partitions",
                    os.environ.get("FDIC_SPARK_SHUFFLE", str(2 * (os.cpu_count() or 4))))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def spark_plan(spark, scale: int):
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    src, fp = _stacked_parquet(scale), _failures_path(scale)
    fin = spark.read.parquet(str(src)).select(*COLS)
    fails = spark.read.parquet(str(fp))
    x = (fin.join(F.broadcast(fails), on="CERT", how="left")
         .withColumn("_qi", F.year("REPDTE") * 4 + F.quarter("REPDTE"))
         .withColumn("_end", F.last_day(F.add_months("REPDTE", 12))))
    w = Window.partitionBy("CERT").orderBy("REPDTE")
    prev_ok = (F.col("_qi") - F.lag("_qi", 1).over(w)) == 1
    q1 = F.quarter("REPDTE") == 1
    x = x.withColumn("y", F.coalesce(((F.col("fail_date") > F.col("REPDTE")) &
                                      (F.col("fail_date") <= F.col("_end"))).cast("int"), F.lit(0)))
    for c in YTD:
        x = x.withColumn(f"{c}_q", F.when(q1, F.col(c).cast("double"))
                         .when(prev_ok, (F.col(c) - F.lag(c, 1).over(w)).cast("double")))

    def div(a, b):
        return F.when(b > 0, a / b)

    x = (x.withColumn("equity_to_assets", div(F.col("EQ"), F.col("ASSET")))
         .withColumn("npl_ratio", div(F.col("NCLNLS"), F.col("LNLSGR")))
         .withColumn("roa_q", div(4 * F.col("NETINC_q"), F.col("ASSET")))
         .withColumn("nim_q", div(4 * (F.col("INTINC_q") - F.col("EINTEXP_q")), F.col("ASSET")))
         .withColumn("loans_to_deposits", div(F.col("LNLSNET"), F.col("DEP"))))
    w4 = w.rowsBetween(-3, 0)
    x = (x.withColumn("d4_npl_ratio", F.when((F.col("_qi") - F.lag("_qi", 4).over(w)) == 4,
                                             F.col("npl_ratio") - F.lag("npl_ratio", 4).over(w)))
         .withColumn("roa_mean4", F.when(F.count("roa_q").over(w4) >= 3, F.avg("roa_q").over(w4))))
    # average rank of non-null values inside each quarter (nulls sorted last, then masked)
    wq = Window.partitionBy("REPDTE").orderBy(F.col("npl_ratio").asc_nulls_last())
    ties = F.count(F.lit(1)).over(Window.partitionBy("REPDTE", "npl_ratio"))
    n_q = F.count("npl_ratio").over(Window.partitionBy("REPDTE"))
    x = x.withColumn("npl_rank_q", F.when(F.col("npl_ratio").isNotNull(),
                                          (F.rank().over(wq) + (ties - 1) / 2.0) / n_q))
    return x.select("CERT", "REPDTE", *OUT)


def run_spark(scale: int) -> dict:
    from pyspark.sql import functions as F
    t0 = time.perf_counter()
    spark = spark_session()
    spark.range(1).count()
    startup = time.perf_counter() - t0
    t1 = time.perf_counter()
    res = spark_plan(spark, scale)
    aggs = [F.count(F.lit(1)).alias("rows")]
    for c in OUT:
        aggs += [F.sum(F.col(c).cast("double")).alias(f"{c}__sum"), F.count(c).alias(f"{c}__n")]
    row = res.agg(*aggs).collect()[0].asDict()
    compute = time.perf_counter() - t1
    spark.stop()
    out = {k: (float(v) if k.endswith("__sum") else int(v)) for k, v in row.items()}
    for k in out:
        if k.endswith("__sum") and out[k] is None:
            out[k] = 0.0
    out["_spark_startup_s"] = startup
    out["_spark_compute_s"] = compute
    return out


# --------------------------------------------------------------------------- child process
def child(engine: str, scale: int) -> None:
    t0 = time.perf_counter()
    if engine == "pandas":
        res = run_pandas(scale)
    elif engine == "pyspark":
        res = run_spark(scale)
    elif engine == "polars_noopt":     # same lazy query with every optimiser pass switched off
        import polars as pl
        src, fp = _stacked_parquet(scale), _failures_path(scale)
        fin = pl.scan_parquet(src, hive_partitioning=True) if scale == 1 else pl.scan_parquet(src)
        res = _checks_polars(polars_plan(fin, pl.scan_parquet(fp)).collect(optimizations=pl.QueryOptFlags.none()))
    else:
        res, _ = run_polars(engine, scale)
    res["_seconds"] = time.perf_counter() - t0
    print("RESULT" + json.dumps(res))


def _tree_rss_mb(pid: int) -> float:
    import psutil
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        return sum(q.memory_info().rss for q in procs if q.is_running()) / 2 ** 20
    except Exception:
        return 0.0


def _measure(engine: str, scale: int, env: dict | None = None) -> dict:
    code = (f"import sys; sys.path.insert(0, r'{C.ROOT / 'src'}');"
            f"from fdic_ews.bench_processing import child; child('{engine}', {scale})")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            env={**os.environ, **(env or {})})
    peak = [0.0]

    def sample():
        while proc.poll() is None:
            peak[0] = max(peak[0], _tree_rss_mb(proc.pid))
            time.sleep(0.05)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out, _ = proc.communicate()
    th.join()
    line = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    if not line:
        raise RuntimeError(f"{engine} x{scale} failed")
    res = json.loads(line[0][6:])
    res["_peak_mb"] = peak[0]
    return res


def run(scales=(1, 3), repeats: int = 3) -> None:
    import polars as pl
    rows, ref = [], {}
    for sc in scales:
        _stacked_parquet(sc)
        _failures_path(sc)
        for eng in ENGINES:
            reps = [_measure(eng, sc) for _ in range(repeats)]
            secs = [r["_seconds"] for r in reps]
            chk = {k: v for k, v in reps[0].items() if not k.startswith("_")}
            if eng == "polars_lazy":
                ref[sc] = chk
            rows.append({"scale": sc, "engine": eng, "rows": chk["rows"], "median_s": statistics.median(secs),
                         "min_s": min(secs), "max_s": max(secs),
                         "spark_startup_s": statistics.median([r.get("_spark_startup_s", 0.0) for r in reps]),
                         "peak_mb": statistics.median([r["_peak_mb"] for r in reps]), "checks": chk})
            print(f"x{sc} {eng:17s} {statistics.median(secs):8.2f}s peak {rows[-1]['peak_mb']:7.0f} MB", flush=True)
    out = []
    for r in rows:
        base = ref[r["scale"]]
        chk = r.pop("checks")
        n_mis = sum(1 for k in base if k.endswith("__n") or k == "rows" if base[k] != chk[k])
        rel = max(abs(chk[k] - base[k]) / max(1.0, abs(base[k])) for k in base if k.endswith("__sum"))
        out.append({**r, "count_mismatches": n_mis, "max_rel_diff_sums": rel})
    df = pl.DataFrame(out)
    pd_t = df.filter(pl.col("engine") == "pandas").select("scale", pl.col("median_s").alias("_pd"))
    df = df.join(pd_t, on="scale").with_columns((pl.col("_pd") / pl.col("median_s")).alias("speedup_vs_pandas")).drop("_pd")
    df.write_csv(C.TABLES / "processing_benchmark.csv")
    import platform

    import pandas
    import psutil
    import pyspark
    env = {"cpu_count": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
           "python": platform.python_version(), "polars": pl.__version__, "pandas": pandas.__version__,
           "pyspark": pyspark.__version__, "repeats": repeats,
           "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (C.TABLES / "processing_benchmark_env.json").write_text(json.dumps(env, indent=1))
    # plans for the report
    import polars as pl2
    lf = polars_plan(pl2.scan_parquet(C.CURATED / "financials", hive_partitioning=True),
                     pl2.scan_parquet(_failures_path(1)))
    (C.TABLES / "plan_bench_polars_optimized.txt").write_text(lf.explain(optimized=True))
    (C.TABLES / "plan_bench_polars_naive.txt").write_text(lf.explain(optimized=False))
    _plot(df)


def _plot(df) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    scales = sorted(df["scale"].unique().to_list())
    w = 0.8 / len(scales)
    for k, sc in enumerate(scales):
        d = df.filter(df["scale"] == sc).to_pandas().set_index("engine").loc[ENGINES]
        xs = [j + k * w for j in range(len(ENGINES))]
        axes[0].bar(xs, d["median_s"], w, label=f"x{sc}")
        axes[1].bar(xs, d["peak_mb"] / 1024, w, label=f"x{sc}")
    for ax, t in zip(axes, ["Median wall time (s)", "Peak memory of process tree (GB)"]):
        ax.set_xticks([j + w * (len(scales) - 1) / 2 for j in range(len(ENGINES))], ENGINES, rotation=20, fontsize=8)
        ax.set_title(t)
        ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "processing_benchmark.png", dpi=170)
    plt.close(fig)
