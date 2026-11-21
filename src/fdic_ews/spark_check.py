"""Part 3 - PySpark reproduction of the core steps, row-level parity check and plan.

The benchmark compares checksums; this stage goes further: Spark writes its full
result, Polars computes the same workload, and the two are joined on (CERT, REPDTE)
to count rows whose label differs and the largest absolute difference per feature.
The formatted physical plan is saved for the report.
"""
from __future__ import annotations

import json
import shutil
import time

import polars as pl

from . import config as C
from .bench_processing import OUT, polars_plan, spark_plan, spark_session, _failures_path


def run() -> dict:
    out_dir = C.DATA / "spark_out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t0 = time.perf_counter()
    spark = spark_session("bft65b-parity")
    t_start = time.perf_counter() - t0
    sdf = spark_plan(spark, 1)
    plan = sdf._jdf.queryExecution().explainString(
        spark._jvm.org.apache.spark.sql.execution.ExplainMode.fromString("formatted"))
    (C.TABLES / "plan_spark_formatted.txt").write_text(plan)
    t1 = time.perf_counter()
    sdf.write.parquet(str(out_dir))
    t_spark = time.perf_counter() - t1
    spark.stop()

    t2 = time.perf_counter()
    pol = polars_plan(pl.scan_parquet(C.CURATED / "financials", hive_partitioning=True),
                      pl.scan_parquet(_failures_path(1))).collect()
    t_pol = time.perf_counter() - t2
    spk = pl.read_parquet(out_dir / "*.parquet").with_columns(pl.col("CERT").cast(pl.Int32), pl.col("REPDTE").cast(pl.Date))
    j = pol.join(spk, on=["CERT", "REPDTE"], how="full", suffix="_spark", coalesce=True)
    res = {"rows_polars": pol.height, "rows_spark": spk.height,
           "rows_only_in_one_engine": int(j.filter(pl.col("y").is_null() | pl.col("y_spark").is_null()).height),
           "label_mismatches": int(j.filter(pl.col("y") != pl.col("y_spark")).height),
           "spark_session_start_s": t_start, "spark_compute_write_s": t_spark, "polars_compute_s": t_pol}
    cols = []
    for c in OUT:
        if c == "y":
            continue
        a, b = j[c], j[f"{c}_spark"]
        cols.append({"column": c, "null_disagreements": int((a.is_null() != b.is_null()).sum()),
                     "max_abs_diff": float((a - b).abs().max() or 0.0)})
    res["columns"] = cols
    res["passed"] = (res["rows_only_in_one_engine"] == 0 and res["label_mismatches"] == 0
                     and all(c["null_disagreements"] == 0 and c["max_abs_diff"] < 1e-9 for c in cols))
    (C.TABLES / "spark_parity.json").write_text(json.dumps(res, indent=1))
    pl.DataFrame(cols).write_csv(C.TABLES / "spark_parity_columns.csv")
    print(json.dumps(res, indent=1))
    shutil.rmtree(out_dir)
    return res
