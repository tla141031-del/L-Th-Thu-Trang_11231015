"""Offline unit tests on tiny hand-made panels (no download needed): ``pytest -q``."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdic_ews.bench_processing import polars_plan  # noqa: E402
from fdic_ews.features import FEATURES, build_lazy  # noqa: E402
from fdic_ews.model import Winsorizer  # noqa: E402


def _qends(start_year, n):
    out, y, k = [], start_year, 0
    while len(out) < n:
        m = (3, 6, 9, 12)[k % 4]
        d = {3: 31, 6: 30, 9: 30, 12: 31}[m]
        out.append(date(y + k // 4, m, d))
        k += 1
    return out


def _panel(cert=1, n=8, start=2000, skip=()):
    q = [d for i, d in enumerate(_qends(start, n)) if i not in skip]
    k = len(q)
    base = dict(CERT=[cert] * k, REPDTE=q, STALP=["GA"] * k,
                ASSET=[1000] * k, DEP=[800] * k, DEPDOM=[800] * k, EQ=[100] * k, BRO=[0] * k, CHBAL=[50] * k,
                SC=[200] * k, LIAB=[900] * k, LNLSGR=[600] * k, LNLSNET=[590] * k, LNRE=[300] * k,
                LNRECONS=[30] * k, LNCI=[200] * k, LNCON=[100] * k, LNATRES=[10] * k, NCLNLS=[12] * k,
                P3ASSET=[5] * k, P9ASSET=[3] * k, NAASSET=[9] * k, ORE=[2] * k, ROA=[1.0] * k, ROE=[10.0] * k,
                NIMY=[3.5] * k, RBC1AAJ=[9.0] * k, RBCRWAJ=[14.0] * k)
    # YTD income: 10 per quarter, cumulated within the calendar year
    ytd = [10 * ((d.month - 1) // 3 + 1) for d in q]
    for c in ("NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"):
        base[c] = ytd
    df = pl.DataFrame(base).with_columns(pl.lit(0, pl.Int8).alias("y"), pl.lit(None, pl.Date).alias("fail_date"),
                                         pl.col("REPDTE").alias("available_date"), pl.lit(True).alias("label_complete"),
                                         pl.lit("active").alias("exit_type"))
    return df


def test_ytd_decumulation_gives_constant_quarterly_flow():
    f = build_lazy(_panel().lazy()).collect()
    assert np.allclose(f["roa_q"].to_numpy(), 4 * 10 / 1000)


def test_gap_makes_flows_and_changes_null():
    f = build_lazy(_panel(n=8, skip=(2,)).lazy()).collect()   # 2000Q3 missing
    row = f.filter(pl.col("REPDTE") == date(2000, 12, 31))
    assert row["roa_q"].item() is None                           # Q4 - Q2 would mix two quarters
    assert row["d1_npl_ratio"].item() is None


def test_features_do_not_look_ahead():
    p = pl.concat([_panel(1, 12), _panel(2, 12)])
    p = p.with_columns(pl.when(pl.col("REPDTE") > date(2001, 6, 30)).then(pl.col("NCLNLS") * 50)
                       .otherwise(pl.col("NCLNLS")).alias("NCLNLS"))
    cut = date(2001, 6, 30)
    full = build_lazy(p.lazy()).filter(pl.col("REPDTE") <= cut).collect().sort("CERT", "REPDTE")
    trunc = build_lazy(p.lazy().filter(pl.col("REPDTE") <= cut)).collect().sort("CERT", "REPDTE")
    for c in FEATURES:
        a, b = full[c], trunc[c]
        assert ((a - b).abs().fill_null(0) < 1e-12).all() and (a.is_null() == b.is_null()).all(), c


def test_label_window_is_open_closed_four_quarters():
    p = _panel(1, 8).select("CERT", "REPDTE", "ASSET", "DEP", "EQ", "LNLSGR", "LNLSNET", "NCLNLS", "NETINC",
                            "INTINC", "EINTEXP", "STALP").with_columns(pl.col("REPDTE").dt.year().alias("year"))
    fails = pl.DataFrame({"CERT": [1], "fail_date": [date(2001, 6, 30)]})
    out = polars_plan(p.lazy(), fails.lazy()).collect().sort("REPDTE")
    y = dict(zip(out["REPDTE"].to_list(), out["y"].to_list()))
    assert y[date(2000, 6, 30)] == 1          # failure exactly 4 quarter-ends later: inside (q, q+4]
    assert y[date(2000, 3, 31)] == 0          # 5 quarters ahead: outside
    assert y[date(2001, 6, 30)] == 0          # failure on the report date: not strictly after


def test_winsorizer_learns_only_from_fit_data():
    tr = np.array([[0.0], [1.0], [2.0], [3.0], [np.nan]])
    w = Winsorizer(q=0.25).fit(tr)
    out = w.transform(np.array([[100.0], [-5.0], [np.nan]]))
    assert out[0, 0] == w.hi_[0] and out[1, 0] == w.lo_[0] and np.isnan(out[2, 0])
