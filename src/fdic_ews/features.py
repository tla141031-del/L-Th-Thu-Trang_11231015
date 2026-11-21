"""Part 3/4 - point-in-time feature table built with the Polars lazy API.

Every feature for bank i at quarter q is a function of reports dated <= q only:
* year-to-date income items are turned into quarterly flows: Q1 value as is,
  later quarters minus the previous report *only if it is exactly one quarter
  earlier* (otherwise null - a gap would mix several quarters);
* lags use ``shift(k).over('CERT')`` after sorting by date, guarded by the
  quarter index so a lag never jumps over a missing quarter;
* rolling statistics use a 4-report window that is nulled when it spans more
  than 4 calendar quarters (gap-aware);
* cross-sectional features (percentile rank within the quarter, state median)
  use only reports of the same quarter, which are published at the same time.

``python run.py features`` writes data/features/features.parquet and
reports/tables/feature_dictionary.csv, then proves the point-in-time property
by recomputing everything on data truncated at a cut-off (look-ahead test).
"""
from __future__ import annotations

import json

import polars as pl

from . import config as C


def _div(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(b > 0).then(a / b).otherwise(None)


QI = (pl.col("REPDTE").dt.year().cast(pl.Int32) * 4 + pl.col("REPDTE").dt.quarter().cast(pl.Int32))

# name -> (pillar, formula text) ; filled by the expressions below
DICT: dict[str, tuple[str, str]] = {
    "equity_to_assets": ("C", "EQ / ASSET"),
    "tier1_leverage": ("C", "RBC1AAJ (%), 0 -> missing"),
    "total_rbc_ratio": ("C", "RBCRWAJ (%), 0 -> missing (not collected before 1990)"),
    "npl_ratio": ("A", "NCLNLS / LNLSGR"),
    "nonaccrual_to_assets": ("A", "NAASSET / ASSET"),
    "pd90_to_assets": ("A", "P9ASSET / ASSET"),
    "ore_to_assets": ("A", "ORE / ASSET"),
    "texas_ratio": ("A", "(NCLNLS + ORE) / (EQ + LNATRES)"),
    "alll_to_loans": ("A", "LNATRES / LNLSGR"),
    "alll_coverage": ("A", "LNATRES / NCLNLS"),
    "nco_rate": ("A", "4 x quarterly NTLNLS / LNLSGR"),
    "provision_rate": ("A", "4 x quarterly ELNATR / LNLSGR"),
    "efficiency_ratio": ("M", "NONIX_q / (INTINC_q - EINTEXP_q + NONII_q)"),
    "nonint_exp_to_assets": ("M", "4 x NONIX_q / ASSET"),
    "roa_q": ("E", "4 x NETINC_q / ASSET (de-cumulated, annualised)"),
    "nim_q": ("E", "4 x (INTINC_q - EINTEXP_q) / ASSET"),
    "nonint_inc_to_assets": ("E", "4 x NONII_q / ASSET"),
    "loans_to_deposits": ("L", "LNLSNET / DEP"),
    "liquid_to_assets": ("L", "(CHBAL + SC) / ASSET"),
    "brokered_to_deposits": ("L", "BRO / DEP"),
    "deposits_to_assets": ("L", "DEP / ASSET"),
    "log_assets": ("S", "ln(ASSET)"),
    "re_loans_share": ("S", "LNRE / LNLSGR"),
    "construction_share": ("S", "LNRECONS / LNLSGR"),
    "ci_loans_share": ("S", "LNCI / LNLSGR"),
    "consumer_share": ("S", "LNCON / LNLSGR"),
    "asset_growth_4q": ("dyn", "ASSET_q / ASSET_{q-4} - 1 (null across gaps)"),
    "loan_growth_4q": ("dyn", "LNLSGR_q / LNLSGR_{q-4} - 1"),
    "d4_equity_to_assets": ("dyn", "equity_to_assets_q - equity_to_assets_{q-4}"),
    "d4_npl_ratio": ("dyn", "npl_ratio_q - npl_ratio_{q-4}"),
    "d1_npl_ratio": ("dyn", "npl_ratio_q - npl_ratio_{q-1}"),
    "d4_texas_ratio": ("dyn", "texas_ratio_q - texas_ratio_{q-4}"),
    "roa_mean4": ("dyn", "mean roa_q over the last 4 reports (gap-aware)"),
    "roa_std4": ("dyn", "std roa_q over the last 4 reports (gap-aware)"),
    "npl_rank_q": ("xs", "percentile rank of npl_ratio among banks in the same quarter"),
    "equity_rank_q": ("xs", "percentile rank of equity_to_assets in the same quarter"),
    "roa_rank_q": ("xs", "percentile rank of roa_q in the same quarter"),
    "state_npl_median": ("xs", "median npl_ratio of the bank's state in the same quarter"),
    "state_d4_npl_median": ("xs", "median d4_npl_ratio of the state in the same quarter"),
    "bank_age_years": ("M", "(REPDTE - ESTYMD) / 365.25 (ESTYMD is a fixed fact, known at q)"),
}
FEATURES = list(DICT)


def build_lazy(panel: pl.LazyFrame, est: pl.LazyFrame | None = None) -> pl.LazyFrame:
    lf = panel.sort("CERT", "REPDTE").with_columns(QI.alias("_qi"))
    prev_ok = (pl.col("_qi") - pl.col("_qi").shift(1).over("CERT")) == 1
    q1 = pl.col("REPDTE").dt.quarter() == 1
    lf = lf.with_columns([
        pl.when(q1).then(pl.col(c)).when(prev_ok).then(pl.col(c) - pl.col(c).shift(1).over("CERT"))
        .otherwise(None).cast(pl.Float64).alias(f"{c}_q") for c in C.YTD_FIELDS])
    A = pl.col("ASSET")
    L = pl.col("LNLSGR")
    lf = lf.with_columns(
        _div(pl.col("EQ"), A).alias("equity_to_assets"),
        pl.when(pl.col("RBC1AAJ") != 0).then(pl.col("RBC1AAJ")).alias("tier1_leverage"),
        pl.when(pl.col("RBCRWAJ") != 0).then(pl.col("RBCRWAJ")).alias("total_rbc_ratio"),
        _div(pl.col("NCLNLS"), L).alias("npl_ratio"),
        _div(pl.col("NAASSET"), A).alias("nonaccrual_to_assets"),
        _div(pl.col("P9ASSET"), A).alias("pd90_to_assets"),
        _div(pl.col("ORE"), A).alias("ore_to_assets"),
        _div(pl.col("NCLNLS") + pl.col("ORE"), pl.col("EQ") + pl.col("LNATRES")).alias("texas_ratio"),
        _div(pl.col("LNATRES"), L).alias("alll_to_loans"),
        _div(pl.col("LNATRES"), pl.col("NCLNLS")).alias("alll_coverage"),
        _div(4 * pl.col("NTLNLS_q"), L).alias("nco_rate"),
        _div(4 * pl.col("ELNATR_q"), L).alias("provision_rate"),
        _div(pl.col("NONIX_q"), pl.col("INTINC_q") - pl.col("EINTEXP_q") + pl.col("NONII_q")).alias("efficiency_ratio"),
        _div(4 * pl.col("NONIX_q"), A).alias("nonint_exp_to_assets"),
        _div(4 * pl.col("NETINC_q"), A).alias("roa_q"),
        _div(4 * (pl.col("INTINC_q") - pl.col("EINTEXP_q")), A).alias("nim_q"),
        _div(4 * pl.col("NONII_q"), A).alias("nonint_inc_to_assets"),
        _div(pl.col("LNLSNET"), pl.col("DEP")).alias("loans_to_deposits"),
        _div(pl.col("CHBAL") + pl.col("SC"), A).alias("liquid_to_assets"),
        _div(pl.col("BRO"), pl.col("DEP")).alias("brokered_to_deposits"),
        _div(pl.col("DEP"), A).alias("deposits_to_assets"),
        pl.when(A > 0).then(A.cast(pl.Float64).log()).alias("log_assets"),
        _div(pl.col("LNRE"), L).alias("re_loans_share"),
        _div(pl.col("LNRECONS"), L).alias("construction_share"),
        _div(pl.col("LNCI"), L).alias("ci_loans_share"),
        _div(pl.col("LNCON"), L).alias("consumer_share"),
    )
    lag4_ok = (pl.col("_qi") - pl.col("_qi").shift(4).over("CERT")) == 4
    lag1_ok = prev_ok

    def d(col: str, k: int, ok: pl.Expr) -> pl.Expr:
        return pl.when(ok).then(pl.col(col) - pl.col(col).shift(k).over("CERT"))

    def g(col: str) -> pl.Expr:
        return pl.when(lag4_ok).then(_div(pl.col(col), pl.col(col).shift(4).over("CERT")) - 1)

    lf = lf.with_columns(
        g("ASSET").alias("asset_growth_4q"),
        g("LNLSGR").alias("loan_growth_4q"),
        d("equity_to_assets", 4, lag4_ok).alias("d4_equity_to_assets"),
        d("npl_ratio", 4, lag4_ok).alias("d4_npl_ratio"),
        d("npl_ratio", 1, lag1_ok).alias("d1_npl_ratio"),
        d("texas_ratio", 4, lag4_ok).alias("d4_texas_ratio"),
        # rolling over the last 4 reports; valid only if they span <= 4 calendar quarters
        pl.when((pl.col("_qi") - pl.col("_qi").shift(3).over("CERT")) == 3)
        .then(pl.col("roa_q").rolling_mean(4, min_samples=3).over("CERT")).alias("roa_mean4"),
        pl.when((pl.col("_qi") - pl.col("_qi").shift(3).over("CERT")) == 3)
        .then(pl.col("roa_q").rolling_std(4, min_samples=3).over("CERT")).alias("roa_std4"),
    )
    # cross-sectional (same quarter only)
    lf = lf.with_columns(
        (pl.col("npl_ratio").rank("average").over("REPDTE") / pl.col("npl_ratio").count().over("REPDTE")).alias("npl_rank_q"),
        (pl.col("equity_to_assets").rank("average").over("REPDTE") / pl.col("equity_to_assets").count().over("REPDTE")).alias("equity_rank_q"),
        (pl.col("roa_q").rank("average").over("REPDTE") / pl.col("roa_q").count().over("REPDTE")).alias("roa_rank_q"),
        pl.col("npl_ratio").median().over(["REPDTE", "STALP"]).alias("state_npl_median"),
        pl.col("d4_npl_ratio").median().over(["REPDTE", "STALP"]).alias("state_d4_npl_median"),
    )
    if est is not None:
        lf = lf.join(est, on="CERT", how="left").with_columns(
            ((pl.col("REPDTE") - pl.col("ESTYMD")).dt.total_days() / 365.25).alias("bank_age_years"))
    else:
        lf = lf.with_columns(pl.lit(None, pl.Float64).alias("bank_age_years"))
    keep = ["CERT", "REPDTE", "STALP", "y", "fail_date", "available_date", "label_complete", "exit_type"]
    return lf.select([c for c in keep] + FEATURES).with_columns(
        [pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in FEATURES])


def _est() -> pl.LazyFrame:
    return pl.scan_parquet(C.CURATED / "institutions.parquet").select("CERT", "ESTYMD")


def build() -> dict:
    panel = pl.scan_parquet(C.CURATED / "panel.parquet")
    plan = build_lazy(panel, _est())
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (C.TABLES / "plan_features_optimized.txt").write_text(plan.explain(optimized=True))
    (C.TABLES / "plan_features_naive.txt").write_text(plan.explain(optimized=False))
    feats = plan.collect()
    C.FEATURES.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(C.FEATURES / "features.parquet", compression="zstd", statistics=True)
    dict_rows = [{"feature": k, "pillar": v[0], "formula": v[1],
                  "null_pct": round(100 * feats[k].null_count() / feats.height, 2)} for k, v in DICT.items()]
    pl.DataFrame(dict_rows).write_csv(C.TABLES / "feature_dictionary.csv")
    look = lookahead_test(panel)
    print(f"features: {feats.height:,} rows x {len(FEATURES)} features; look-ahead test: {look}")
    return look


def lookahead_test(panel: pl.LazyFrame, cutoff: str = "2001-12-31") -> dict:
    """Features of rows <= cutoff must be identical whether or not later rows exist."""
    from datetime import date
    cut = date.fromisoformat(cutoff)
    full = build_lazy(panel, _est()).filter(pl.col("REPDTE") <= cut).collect().sort("CERT", "REPDTE")
    trunc = build_lazy(panel.filter(pl.col("REPDTE") <= cut), _est()).collect().sort("CERT", "REPDTE")
    assert full.height == trunc.height
    diffs = {}
    for c in FEATURES:
        a, b = full[c], trunc[c]
        same_null = bool((a.is_null() == b.is_null()).all())
        md = float((a - b).abs().max() or 0.0)
        diffs[c] = {"max_abs_diff": md, "same_nulls": same_null}
    changed = [c for c, v in diffs.items() if v["max_abs_diff"] > 1e-12 or not v["same_nulls"]]
    res = {"cutoff": cutoff, "rows_compared": full.height, "features": len(FEATURES),
           "features_changed_by_future_rows": changed, "passed": not changed}
    (C.TABLES / "features_lookahead_test.json").write_text(json.dumps(res, indent=1))
    return res
