"""Extra measurements behind design choices (run after `spark`, before `model`; development data only).

1. resume_test     - kill a real download mid-way, restart it, and check what is re-fetched.
2. polars_scan     - Polars lazy with 1/2/4 threads and with the optimiser switched off.
3. spark_scan      - PySpark with local[1]/[2]/[4] cores and 8 vs 200 shuffle partitions.
4. feature_signal  - one-feature ROC-AUC and medians for y = 1 / y = 0 on development rows only.
5. layout_effects  - in-memory size of the raw vs typed schema; file size and row groups read
                     for a one-bank query with and without sorting by (CERT, REPDTE).

Each timing is a fresh process (bench_processing._measure), median of 3 runs.
    python run.py experiments
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time

from . import config as C
from .bench_processing import _failures_path, _measure, _stacked_parquet


# --------------------------------------------------------------------------- 1. resumability
def resume_test(start: str = "2022-03-31", end: str = "2023-12-31", kill_after_complete: int = 3) -> dict:
    tmp = C.DATA / "resume_test"
    shutil.rmtree(tmp, ignore_errors=True)
    env = {**os.environ, "FDIC_DATA_DIR": str(tmp)}
    code = (f"import sys; sys.path.insert(0, r'{C.ROOT / 'src'}');"
            f"from fdic_ews import ingest; ingest.download_financials('{start}', '{end}')")
    man_p = tmp / "raw" / "manifest.json"

    def n_complete():
        try:
            return sum(1 for r in json.loads(man_p.read_text())["financials"].values() if r["complete"])
        except Exception:
            return 0

    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    while p.poll() is None and n_complete() < kill_after_complete:
        time.sleep(0.2)
    os.killpg(p.pid, signal.SIGKILL)            # hard kill: no clean-up code can run
    p.wait()
    first_s = time.perf_counter() - t0
    files_after_kill = sorted(x.name for x in (tmp / "raw" / "financials").glob("*"))
    done_at_kill = n_complete()

    def rerun():
        t = time.perf_counter()
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        return time.perf_counter() - t, out.splitlines()[0]

    s2, l2 = rerun()
    s3, l3 = rerun()
    man = json.loads(man_p.read_text())["financials"]
    import hashlib
    ok = all(hashlib.sha256((tmp / "raw" / "financials" / f"REPDTE={q}.parquet").read_bytes()).hexdigest()
             == r["sha256"] for q, r in man.items())
    res = {"quarters": len(man), "killed_after_s": round(first_s, 1), "complete_at_kill": done_at_kill,
           "files_after_kill": len(files_after_kill),
           "partial_tmp_files_after_kill": sum(1 for x in files_after_kill if x.endswith(".tmp")),
           "restart_message": l2, "restart_s": round(s2, 1), "second_restart_message": l3,
           "second_restart_s": round(s3, 1), "all_complete": all(r["complete"] for r in man.values()),
           "all_sha256_match": ok, "rows": sum(r["rows"] for r in man.values())}
    (C.TABLES / "ingest_resume_test.json").write_text(json.dumps(res, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(res)
    return res


# --------------------------------------------------------------------------- 2./3. engine scans
def _rep(engine, scale, env, repeats=3):
    reps = [_measure(engine, scale, env) for _ in range(repeats)]
    return {"median_s": statistics.median(r["_seconds"] for r in reps),
            "peak_mb": statistics.median(r["_peak_mb"] for r in reps),
            "spark_startup_s": statistics.median(r.get("_spark_startup_s", 0.0) for r in reps),
            "rows": reps[0]["rows"], "npl_rank_sum": reps[0]["npl_rank_q__sum"]}


def engine_scans(repeats: int = 3) -> None:
    import polars as pl
    for sc in (1, 3):
        _stacked_parquet(sc), _failures_path(sc)
    rows = []
    for sc in (1, 3):
        for th in (1, 2, 4):
            r = _rep("polars_lazy", sc, {"POLARS_MAX_THREADS": str(th)}, repeats)
            rows.append({"experiment": "polars_threads", "engine": "polars_lazy", "scale": sc, "setting": f"{th} threads", **r})
            print(rows[-1], flush=True)
    r = _rep("polars_noopt", 1, {}, repeats)
    rows.append({"experiment": "polars_optimizer", "engine": "polars_lazy", "scale": 1, "setting": "optimiser off", **r})
    r = _rep("polars_lazy", 1, {}, repeats)
    rows.append({"experiment": "polars_optimizer", "engine": "polars_lazy", "scale": 1, "setting": "optimiser on", **r})
    print(rows[-2:], flush=True)
    for sc in (1, 3):
        for k in (1, 2, 4):
            r = _rep("pyspark", sc, {"FDIC_SPARK_MASTER": f"local[{k}]", "FDIC_SPARK_SHUFFLE": str(2 * k)}, repeats)
            rows.append({"experiment": "spark_cores", "engine": "pyspark", "scale": sc, "setting": f"local[{k}]", **r})
            print(rows[-1], flush=True)
    for parts in (8, 200):
        r = _rep("pyspark", 1, {"FDIC_SPARK_SHUFFLE": str(parts)}, repeats)
        rows.append({"experiment": "spark_shuffle", "engine": "pyspark", "scale": 1, "setting": f"{parts} partitions", **r})
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(C.TABLES / "engine_scans.csv")


# --------------------------------------------------------------------------- 4. feature signal (development only)
def feature_signal() -> None:
    import numpy as np
    import polars as pl
    from scipy.stats import rankdata
    from .features import DICT
    df = (pl.scan_parquet(C.FEATURES / "features.parquet")
          .filter((pl.col("REPDTE") <= pl.lit(C.DEV_END).str.to_date()) & pl.col("label_complete"))
          .select("y", *DICT.keys()).collect())
    y = df["y"].to_numpy()
    out = []
    for feat, meta in DICT.items():
        x = df[feat].cast(pl.Float64).to_numpy()
        m = np.isfinite(x)
        yy, xx = y[m], x[m]
        n1 = int(yy.sum())
        n0 = len(yy) - n1
        auc = (rankdata(xx)[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
        out.append({"feature": feat, "pillar": meta[0], "missing_pct": 100 * (1 - m.mean()),
                    "median_y1": float(np.median(xx[yy == 1])) if n1 else None,
                    "median_y0": float(np.median(xx[yy == 0])) if n0 else None,
                    "roc_auc_oriented": max(auc, 1 - auc), "direction": "higher = riskier" if auc >= 0.5 else "lower = riskier"})
    pl.DataFrame(out).sort("roc_auc_oriented", descending=True).write_csv(C.TABLES / "feature_signal_dev.csv")
    print(pl.DataFrame(out).sort("roc_auc_oriented", descending=True).head(10))


# --------------------------------------------------------------------------- 5. schema and sort order
def layout_effects() -> None:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    import polars as pl
    from .storage import read_raw_financials, typed_financials
    raw = read_raw_financials()
    typed, _ = typed_financials(raw)
    typed = typed.drop("year")
    res = {"rows": typed.height,
           "raw_inmemory_mb": raw.estimated_size("mb"), "typed_inmemory_mb": typed.estimated_size("mb"),
           "stalp_string_mb": raw["STALP"].estimated_size("mb"), "stalp_categorical_mb": typed["STALP"].estimated_size("mb"),
           "repdte_string_mb": raw["REPDTE"].estimated_size("mb"), "repdte_date_mb": typed["REPDTE"].estimated_size("mb")}
    tmp = C.DATA / "layout_test"
    tmp.mkdir(parents=True, exist_ok=True)
    one = int(typed["CERT"].median())
    for name, frame in (("sorted", typed), ("unsorted", typed.sample(fraction=1.0, shuffle=True, seed=C.SEED))):
        f = tmp / f"{name}.parquet"
        frame.write_parquet(f, compression="zstd", statistics=True, row_group_size=50_000)
        md = pq.ParquetFile(f).metadata
        ci = md.schema.to_arrow_schema().get_field_index("CERT")
        groups = sum(1 for g in range(md.num_row_groups)
                     if md.row_group(g).column(ci).statistics.min <= one <= md.row_group(g).column(ci).statistics.max)
        t = time.perf_counter()
        for _ in range(5):
            pl.scan_parquet(f).filter(pl.col("CERT") == one).select("REPDTE", "ASSET", "EQ").collect()
        res[f"{name}_file_mb"] = f.stat().st_size / 2 ** 20
        res[f"{name}_row_groups"] = md.num_row_groups
        res[f"{name}_row_groups_read_one_bank"] = groups
        res[f"{name}_one_bank_query_s"] = (time.perf_counter() - t) / 5
    shutil.rmtree(tmp, ignore_errors=True)
    (C.TABLES / "layout_effects.json").write_text(json.dumps(res, indent=1))
    print(res)


def run() -> None:
    layout_effects()
    feature_signal()
    resume_test()
    engine_scans()
