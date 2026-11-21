"""Part 2a - automated data-quality report.

``python run.py quality`` writes reports/data_quality_report.md plus one CSV per check in
reports/tables/dq_*.csv and figures in reports/figures/dq_*.png. Nothing here changes
the data: the report decides which handling rule the feature pipeline applies, and
every rule that learns a number (clip limits, medians) is fitted later on training rows only.
"""
from __future__ import annotations

import json
from datetime import date

import polars as pl

from . import config as C
from .storage import AMOUNT_COLS, RATIO_COLS, load_panel


def _md(df: pl.DataFrame, floatfmt: int = 2) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"{v:,.{floatfmt}f}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def run() -> dict:
    fin = load_panel().collect()
    sections: list[tuple[str, str]] = []
    summary: dict = {}
    n = fin.height

    # 1. completeness against the API (from the ingest manifest)
    comp = json.loads((C.TABLES / "ingest_completeness.json").read_text())
    sections.append(("1. Completeness vs the API", _md(pl.DataFrame([{
        "quarters": comp["quarters_downloaded"], "rows": comp["financials_rows"],
        "api_total_sum": comp["api_total_sum"], "incomplete_quarters": len(comp["quarters_incomplete"]),
        "duplicate_keys": comp["duplicate_cert_repdte"], "checksum_mismatch": len(comp["checksum_mismatch"])}]))))

    # 2. missingness overall and by year -> structural vs scattered
    cols = AMOUNT_COLS + RATIO_COLS + ["STALP"]
    miss = pl.DataFrame([{"column": c, "missing_pct": 100 * fin[c].null_count() / n} for c in cols])
    by_year = (fin.group_by("year").agg([(pl.col(c).is_null().mean() * 100).alias(c) for c in cols]).sort("year"))
    by_year.write_csv(C.TABLES / "dq_missing_by_year.csv")
    long = by_year.unpivot(index="year", variable_name="column", value_name="pct")
    yrs = (long.group_by("column").agg(
        (pl.col("pct") >= 99).sum().alias("years_fully_missing"),
        ((pl.col("pct") > 0) & (pl.col("pct") < 99)).sum().alias("years_partly_missing"),
        pl.col("year").filter(pl.col("pct") >= 99).max().alias("last_fully_missing_year")))
    miss = miss.join(yrs, on="column").with_columns(
        pl.when(pl.col("years_fully_missing") > 0).then(pl.lit("structural (not collected in some years)"))
        .when(pl.col("missing_pct") > 0.1).then(pl.lit("scattered")).otherwise(pl.lit("negligible"))
        .alias("type")).sort("missing_pct", descending=True)
    # informative missingness: compare banks that fail within 4 quarters with the rest
    panel = pl.read_parquet(C.CURATED / "panel.parquet", columns=["CERT", "REPDTE", "y"])
    fy = fin.join(panel, on=["CERT", "REPDTE"], how="inner")
    inf = pl.DataFrame([{"column": c,
                         "missing_pct_y1": 100 * fy.filter(pl.col("y") == 1)[c].null_count() / max(1, fy.filter(pl.col("y") == 1).height),
                         "missing_pct_y0": 100 * fy.filter(pl.col("y") == 0)[c].null_count() / max(1, fy.filter(pl.col("y") == 0).height)}
                        for c in cols])
    miss = miss.join(inf, on="column")
    miss.write_csv(C.TABLES / "dq_missing.csv")
    sections.append(("2. Missing values (overall, by year, failing vs other banks)",
                     _md(miss.filter(pl.col("missing_pct") > 0.01))))
    _heatmap(by_year)

    # 3. zeros that mean 'not reported'
    zero = (fin.group_by("year").agg([((pl.col(c) == 0).mean() * 100).alias(c) for c in ["RBCRWAJ", "RBC1AAJ", "BRO", "P3ASSET"]])
            .sort("year"))
    zero.write_csv(C.TABLES / "dq_zero_share_by_year.csv")
    zsum = pl.DataFrame([{"column": c, "max_zero_pct_year": float(zero[c].max() or 0),
                          "min_zero_pct_year": float(zero[c].min() or 0),
                          "years_ge_95pct_zero": int((zero[c].fill_null(0) >= 95).sum()),
                          "those_years": ",".join(str(y) for y in zero.filter(pl.col(c) >= 95)["year"].to_list()[:12])}
                         for c in ["RBCRWAJ", "RBC1AAJ", "BRO", "P3ASSET"]])
    zsum.write_csv(C.TABLES / "dq_structural_zeros.csv")
    sections.append(("3. Structural zeros", _md(zsum)))

    # 4. year-to-date income: |NETINC|/ASSET grows with the quarter of the year
    ytd = (fin.filter(pl.col("ASSET") > 0)
           .with_columns(pl.col("REPDTE").dt.quarter().alias("q"))
           .group_by("q").agg([(pl.col(c) / pl.col("ASSET") * 100).abs().median().alias(f"{c}_pct_assets")
                               for c in ["NETINC", "INTINC", "NONIX", "ELNATR"]]).sort("q"))
    ytd.write_csv(C.TABLES / "dq_ytd_evidence.csv")
    sections.append(("4. Year-to-date income items (median |item| / assets, %, by quarter of year)", _md(ytd, 3)))

    # 5. reporting gaps inside a bank's series
    qi = fin.select("CERT", (pl.col("REPDTE").dt.year() * 4 + pl.col("REPDTE").dt.quarter()).alias("qi")).sort("CERT", "qi")
    gaps = qi.with_columns((pl.col("qi") - pl.col("qi").shift(1).over("CERT") - 1).alias("gap")).filter(pl.col("gap") > 0)
    gsum = {"banks_with_internal_gaps": gaps["CERT"].n_unique(), "missing_quarters_inside_series": int(gaps["gap"].sum()),
            "longest_gap_quarters": int(gaps["gap"].max() or 0)}
    sections.append(("5. Gaps inside a bank's reporting series", _md(pl.DataFrame([gsum]))))
    summary["gaps"] = gsum

    # 6. accounting identities and domain rules
    rules = {
        "ASSET <= 0": pl.col("ASSET") <= 0,
        "EQ < 0 (book insolvent)": pl.col("EQ") < 0,
        "DEP > ASSET": pl.col("DEP") > pl.col("ASSET"),
        "EQ > ASSET": pl.col("EQ") > pl.col("ASSET"),
        "|LIAB + EQ - ASSET| > 1% of ASSET": (pl.col("LIAB") + pl.col("EQ") - pl.col("ASSET")).abs() > 0.01 * pl.col("ASSET"),
        "LNLSNET > LNLSGR": pl.col("LNLSNET") > pl.col("LNLSGR"),
        "NCLNLS > LNLSGR": pl.col("NCLNLS") > pl.col("LNLSGR"),
        "LNATRES < 0": pl.col("LNATRES") < 0,
        "LNRE + LNCI + LNCON > 1.05 x LNLSGR": (pl.col("LNRE") + pl.col("LNCI") + pl.col("LNCON")) > 1.05 * pl.col("LNLSGR"),
        "|ROA| > 50%": pl.col("ROA").abs() > 50,
        "RBC1AAJ outside [-100, 100]": ~pl.col("RBC1AAJ").is_between(-100, 100),
    }
    rr = []
    for name, cond in rules.items():
        hit = fy.filter(cond.fill_null(False))
        rr.append({"rule": name, "rows": hit.height, "pct_rows": 100 * hit.height / n, "banks": hit["CERT"].n_unique(),
                   "fail_rate_pct_if_violated": 100 * float(hit["y"].mean() or 0)})
    rr = pl.DataFrame(rr)
    rr.write_csv(C.TABLES / "dq_rules.csv")
    sections.append(("6. Accounting identities and domain rules (fail rate = share of rows with y = 1)", _md(rr, 3)))

    # 7. heavy tails in key ratios, and whether extremes carry signal
    rat = fy.filter(pl.col("ASSET") > 0).with_columns(
        (pl.col("EQ") / pl.col("ASSET")).alias("equity_to_assets"),
        pl.when(pl.col("LNLSGR") > 0).then(pl.col("NCLNLS") / pl.col("LNLSGR")).alias("npl_ratio"),
        pl.when(pl.col("DEP") > 0).then(pl.col("LNLSNET") / pl.col("DEP")).alias("loans_to_deposits"),
        pl.col("ROA").alias("roa_fdic"))
    base = float(fy["y"].mean())
    tails = []
    for c in ["equity_to_assets", "npl_ratio", "loans_to_deposits", "roa_fdic"]:
        s = rat[c].drop_nulls()
        lo, hi = s.quantile(0.005), s.quantile(0.995)
        ext = rat.filter((pl.col(c) < lo) | (pl.col(c) > hi))
        tails.append({"ratio": c, "p0.5": lo, "median": s.median(), "p99.5": hi, "min": s.min(), "max": s.max(),
                      "skew": float(s.skew()), "fail_rate_pct_in_tails": 100 * float(ext["y"].mean()),
                      "fail_rate_pct_overall": 100 * base})
    tails = pl.DataFrame(tails)
    tails.write_csv(C.TABLES / "dq_tails.csv")
    sections.append(("7. Heavy tails (0.5% / 99.5% quantiles) and failure rate in the tails", _md(tails, 3)))

    # 8. failures / institutions tables
    f = pl.read_parquet(C.CURATED / "failures.parquet")
    info = json.loads((C.TABLES / "panel_info.json").read_text())
    fails_used = f.filter((pl.col("RESTYPE") == "FAILURE") & pl.col("CERT").is_not_null())
    since84 = fails_used.filter(pl.col("FAILDATE") > date(1984, 3, 31))
    no_fin = since84.join(fin.select("CERT").unique(), on="CERT", how="anti").height
    ft = pl.DataFrame([
        {"item": "rows in failures table", "value": f.height},
        {"item": "RESTYPE = ASSISTANCE (bank stayed open)", "value": int((f["RESTYPE"] == "ASSISTANCE").sum())},
        {"item": "rows without CERT", "value": int(f["CERT"].null_count())},
        {"item": "failures used (RESTYPE = FAILURE, with CERT)", "value": info["failures"]["rows_used"]},
        {"item": "of which after 1984Q1", "value": since84.height},
        {"item": "after 1984Q1 but no Financials row", "value": no_fin},
        {"item": "Financials rows dated on/after the failure (removed)", "value": info["rows_removed_on_or_after_failure"]},
        {"item": "COST missing for used failures (%)", "value": round(100 * fails_used["COST"].null_count() / fails_used.height, 1)},
    ])
    ft.write_csv(C.TABLES / "dq_failures_table.csv")
    sections.append(("8. Failures table", _md(ft)))
    inst = pl.read_parquet(C.CURATED / "institutions.parquet")
    last_q = fin["REPDTE"].max()
    act = inst.join(fin.filter(pl.col("REPDTE") == last_q).select("CERT").with_columns(pl.lit(1).alias("in_last_q")),
                    on="CERT", how="left")
    snap = pl.DataFrame([
        {"item": "institutions rows", "value": inst.height},
        {"item": "ACTIVE = 1", "value": int((inst["ACTIVE"] == 1).sum())},
        {"item": f"ACTIVE = 1 and reporting in {last_q}", "value": int(act.filter((pl.col("ACTIVE") == 1) & (pl.col("in_last_q") == 1)).height)},
        {"item": "ENDEFYMD = 9999-12-31 (sentinel for 'still open')", "value": int((inst["ENDEFYMD"] == date(9999, 12, 31)).sum())},
        {"item": "ENDEFYMD is a real end date (bank ended)", "value": int((inst["ENDEFYMD"] < date(9999, 12, 31)).sum())},
    ])
    snap.write_csv(C.TABLES / "dq_institutions_snapshot.csv")
    sections.append(("9. Institutions table is a snapshot (ACTIVE, ENDEFYMD, CHANGEC1 are as of download date)", _md(snap)))

    md = ["# Data-quality report (generated by `python run.py quality`)", "",
          f"Financials rows: {n:,}; banks: {fin['CERT'].n_unique():,}; quarters: {fin['REPDTE'].n_unique()}.", ""]
    for title, body in sections:
        md += [f"## {title}", "", body, ""]
    (C.REPORTS / "data_quality_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)[:3000])
    return summary


def _heatmap(by_year: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cols = [c for c in by_year.columns if c != "year" and float(by_year[c].max()) > 0.5]
    m = by_year.select(cols).to_numpy().T
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(cols) + 1.2))
    im = ax.imshow(m, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    yrs = by_year["year"].to_list()
    ticks = np.arange(0, len(yrs), 4)
    ax.set_xticks(ticks, [yrs[t] for t in ticks], fontsize=8)
    fig.colorbar(im, ax=ax, label="% rows missing")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "dq_missing_by_year.png", dpi=170)
    plt.close(fig)
