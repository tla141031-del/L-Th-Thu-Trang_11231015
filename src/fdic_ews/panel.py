"""Part 2b - point-in-time bank-quarter panel with the failure label and exit types.

Output: data/curated/panel.parquet, one row per (CERT, REPDTE) that the label can use.

Rules (all checked in ``checks()``):
* Failures = FDIC failures list rows with RESTYPE == 'FAILURE' and a CERT
  (assistance transactions keep the bank open, so they are not failures).
* y(i, q) = 1  iff  q < FAILDATE_i <= q + 4 quarter-ends.  Mergers are 0.
* Reports dated on/after the failure date are removed.
* ``available_date = REPDTE + 60 days``: the earliest date a supervisor could use the report.
* ``label_complete``: the 4-quarter label window ends on or before the date the
  failures list was downloaded. Modelling uses quarters <= 2025Q2 only.
* exit_type per bank: failure / merger_or_absorption / other_exit / active.
"""
from __future__ import annotations

import json
from datetime import date

import polars as pl

from . import config as C
from .storage import load_panel


def failures_clean() -> tuple[pl.DataFrame, dict]:
    f = pl.read_parquet(C.CURATED / "failures.parquet")
    stats = {"rows_raw": f.height,
             "restype_counts": dict(f.group_by("RESTYPE").len().iter_rows()),
             "missing_cert": int(f["CERT"].null_count())}
    f = f.filter((pl.col("RESTYPE") == "FAILURE") & pl.col("CERT").is_not_null())
    stats["duplicate_cert"] = int(f["CERT"].is_duplicated().sum())
    f = f.sort("FAILDATE").unique("CERT", keep="first", maintain_order=True)
    stats["rows_used"] = f.height
    stats["used_since_1984"] = int(f.filter(pl.col("FAILDATE") > date(1984, 3, 31)).height)
    return f.select("CERT", pl.col("FAILDATE").alias("fail_date"), "COST", "QBFASSET", "RESTYPE1"), stats


def _downloaded_on() -> date:
    man = json.loads((C.RAW / "manifest.json").read_text())
    return date.fromisoformat(man["tables"]["failures"]["downloaded_at"][:10])


def classify_exits(fin: pl.LazyFrame, fails: pl.DataFrame) -> pl.DataFrame:
    inst = pl.read_parquet(C.CURATED / "institutions.parquet").select("CERT", "ACTIVE", "CHANGEC1", "ENDEFYMD")
    last = fin.group_by("CERT").agg(pl.col("REPDTE").max().alias("last_report")).collect()
    ex = (last.join(inst, on="CERT", how="left").join(fails.select("CERT", "fail_date"), on="CERT", how="left")
          .with_columns(
              pl.when(pl.col("fail_date").is_not_null()).then(pl.lit("failure"))
              .when(pl.col("ACTIVE") == 1).then(pl.lit("active"))
              .when(pl.col("CHANGEC1").is_between(200, 299)).then(pl.lit("merger_or_absorption"))
              .otherwise(pl.lit("other_exit")).alias("exit_type")))
    return ex


def build() -> dict:
    fails, fstats = failures_clean()
    fin = load_panel()
    asof = _downloaded_on()
    exits = classify_exits(fin, fails)
    end4 = pl.col("REPDTE").dt.offset_by(f"{3 * C.HORIZON_Q}mo").dt.month_end()
    panel = (fin.join(fails.lazy(), on="CERT", how="left")
             .with_columns(end4.alias("label_window_end"))
             .with_columns(
                 ((pl.col("fail_date") > pl.col("REPDTE")) & (pl.col("fail_date") <= pl.col("label_window_end")))
                 .fill_null(False).cast(pl.Int8).alias("y"),
                 (pl.col("REPDTE") + pl.duration(days=C.PUBLICATION_LAG_DAYS)).alias("available_date"),
                 (pl.col("label_window_end") <= pl.lit(asof)).alias("label_complete"),
             ))
    before = fin.select(pl.len()).collect().item()
    after_fail = panel.filter(pl.col("fail_date").is_not_null() & (pl.col("REPDTE") >= pl.col("fail_date")))
    n_after = after_fail.select(pl.len()).collect().item()
    panel = panel.filter(pl.col("fail_date").is_null() | (pl.col("REPDTE") < pl.col("fail_date")))
    panel = panel.join(exits.lazy().select("CERT", "exit_type"), on="CERT", how="left").collect()
    panel.write_parquet(C.CURATED / "panel.parquet", compression="zstd", statistics=True)

    # ------------------------------------------------------------ evidence tables
    lag = (panel.filter(pl.col("fail_date").is_not_null()).group_by("CERT")
           .agg(pl.col("REPDTE").max(), pl.col("fail_date").first())
           .with_columns(((pl.col("fail_date").dt.year() * 12 + pl.col("fail_date").dt.month()
                           - pl.col("REPDTE").dt.year() * 12 - pl.col("REPDTE").dt.month()) / 3)
                         .ceil().cast(pl.Int32).alias("quarters_last_report_to_failure"))
           .group_by("quarters_last_report_to_failure").len().sort("quarters_last_report_to_failure"))
    lag.write_csv(C.TABLES / "panel_last_report_to_failure.csv")
    ex_tab = (exits.with_columns((pl.col("last_report").dt.year() // 10 * 10).alias("decade"))
              .group_by("exit_type", "decade").len().sort("exit_type", "decade"))
    ex_tab.write_csv(C.TABLES / "panel_exit_types.csv")
    (exits.filter(pl.col("exit_type").is_in(["merger_or_absorption", "other_exit"]))
     .group_by("exit_type", "CHANGEC1").len().sort("len", descending=True).head(25)
     .write_csv(C.TABLES / "panel_exit_codes.csv"))
    by_year = (panel.group_by(pl.col("REPDTE").dt.year().alias("year"))
               .agg(pl.len().alias("rows"), pl.col("CERT").n_unique().alias("banks"),
                    pl.col("y").sum().alias("positive_rows"),
                    pl.col("CERT").filter(pl.col("y") == 1).n_unique().alias("banks_failing_within_4q"))
               .sort("year"))
    fails_year = (fails.group_by(pl.col("fail_date").dt.year().alias("year")).len().rename({"len": "failures"}))
    by_year = by_year.join(fails_year, on="year", how="left").fill_null(0)
    by_year.write_csv(C.TABLES / "panel_label_by_year.csv")
    pos = panel.filter(pl.col("y") == 1)
    info = {
        "failures": fstats, "rows_financials": before, "rows_removed_on_or_after_failure": n_after,
        "rows_panel": panel.height, "positive_rows": pos.height,
        "failed_banks_with_positive_row": pos["CERT"].n_unique(),
        "failed_banks_in_panel": int(panel.filter(pl.col("fail_date").is_not_null())["CERT"].n_unique()),
        "positives_failing_before_available": int(pos.filter(pl.col("fail_date") <= pl.col("available_date")).height),
        "labels_asof": str(asof),
        "exit_banks": dict(exits.group_by("exit_type").len().iter_rows()),
    }
    (C.TABLES / "panel_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))
    checks(panel)
    return info


def checks(panel: pl.DataFrame) -> None:
    """Assertions on the label: the pipeline stops if any is violated."""
    p = panel
    res = []

    def add(name, ok, detail=""):
        res.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    add("(CERT, REPDTE) unique", p.select(pl.struct("CERT", "REPDTE").is_duplicated().sum()).item() == 0)
    pos = p.filter(pl.col("y") == 1)
    add("every positive fails strictly after its report date", (pos["fail_date"] > pos["REPDTE"]).all())
    q_to_fail = ((pos["fail_date"].dt.year() * 4 + (pos["fail_date"].dt.month() - 1) // 3)
                 - (pos["REPDTE"].dt.year() * 4 + (pos["REPDTE"].dt.month() - 1) // 3))
    add("every positive fails within 4 quarters", q_to_fail.is_between(1, 4).all(),
        dict(zip(*q_to_fail.value_counts().sort(q_to_fail.name).to_dict(as_series=False).values())))
    neg_close = p.filter((pl.col("y") == 0) & pl.col("fail_date").is_not_null()
                         & (pl.col("fail_date") <= pl.col("label_window_end"))).height
    add("no negative row has a failure inside its window", neg_close == 0, neg_close)
    add("no row on/after its failure date", p.filter(pl.col("REPDTE") >= pl.col("fail_date")).height == 0)
    add("no positive without a failure (mergers are 0)",
        p.filter((pl.col("y") == 1) & (pl.col("exit_type") != "failure")).height == 0)
    per_bank = pos.group_by("CERT").len()["len"]
    add("at most 4 positive quarters per failed bank", per_bank.max() <= 4, f"mean {per_bank.mean():.2f}")
    pl.DataFrame(res).write_csv(C.TABLES / "panel_checks.csv")
    bad = [r for r in res if not r["passed"]]
    assert not bad, bad
    print(f"panel checks: {len(res)} passed")
