"""Central settings for the whole pipeline.

Every stage imports its constants from here, so a reader can see in one file
which data are used, where they are stored and how the sample is split.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FDIC_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"            # API responses, one Parquet file per quarter (immutable)
CURATED = DATA / "curated"    # typed, validated, partitioned panel
FEATURES = DATA / "features"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"

API_BASE = "https://api.fdic.gov/banks"
PAGE_SIZE = 10_000            # documented maximum rows per call
START_QUARTER = "1984-03-31"  # exam scope
END_QUARTER = "2026-06-30"

# ---------------------------------------------------------------------------
# Field choice (see report section 2.1). Amounts in USD thousands, ratios in %.
# Income-statement items are year-to-date (YTD) in the Call Report.
# ---------------------------------------------------------------------------
FIN_FIELDS = [
    "CERT", "REPDTE", "STALP",
    # balance sheet / size / funding
    "ASSET", "DEP", "DEPDOM", "EQ", "BRO", "CHBAL", "SC", "LIAB",
    # loans and reserves
    "LNLSGR", "LNLSNET", "LNRE", "LNRECONS", "LNCI", "LNCON", "LNATRES",
    # asset quality
    "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "NTLNLS", "ORE",
    # earnings (YTD)
    "NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR",
    # FDIC-computed ratios (annualised, %)
    "ROA", "ROE", "NIMY",
    # regulatory capital ratios (%)
    "RBC1AAJ", "RBCRWAJ",
]
YTD_FIELDS = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]

FAIL_FIELDS = ["CERT", "NAME", "CITYST", "FAILDATE", "FAILYR", "SAVR", "RESTYPE",
               "RESTYPE1", "COST", "QBFASSET", "QBFDEP"]
INST_FIELDS = ["CERT", "NAME", "STALP", "BKCLASS", "CHARTER", "ACTIVE", "ESTYMD",
               "INSDATE", "ENDEFYMD", "CHANGEC1", "REPDTE"]

# Columns that describe the bank's future relative to a report date.
# They may build the label or the exit type, but never a feature.
FUTURE_COLUMNS = {"FAILDATE", "FAILYR", "COST", "RESTYPE", "RESTYPE1", "QBFASSET",
                  "QBFDEP", "ACTIVE", "ENDEFYMD", "CHANGEC1", "exit_type",
                  "exit_date", "fail_date", "y"}

# ---------------------------------------------------------------------------
# Label and split (report section 6.1)
# ---------------------------------------------------------------------------
HORIZON_Q = 4                     # y = 1 if failure in (q, q + 4 quarters]
PUBLICATION_LAG_DAYS = 60         # Call Report usable ~60 days after quarter end
LAST_LABEL_QUARTER = "2025-06-30" # later labels are incomplete
DEV_START, DEV_END = "1984-03-31", "2009-12-31"
GAP_QUARTERS = 4                  # 2010Q1-2010Q4 not used
TEST_START, TEST_END = "2011-03-31", "2025-06-30"
# rolling-origin folds inside development: (train label end, validation start, validation end)
CV_FOLDS = [
    ("1988-12-31", "1990-03-31", "1991-12-31"),
    ("1991-12-31", "1993-03-31", "1996-12-31"),
    ("1996-12-31", "1998-03-31", "2004-12-31"),
    ("2004-12-31", "2006-03-31", "2009-12-31"),
]
SEED = 2026
