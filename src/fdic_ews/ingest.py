"""Part 1a - reproducible, resumable download from the FDIC BankFind API.

Adapted from the course starter script ``fdic_download_starter.py`` (BFT65B course
page; kept unchanged in ``scripts/``). Changes made for this project:

* a field list chosen for CAMELS coverage (``config.FIN_FIELDS``);
* quarters downloaded by a small thread pool (2 workers, polite pauses);
* each quarter is written atomically (temp file + rename), so a crash never
  leaves a half-written file that looks complete;
* a manifest records, per quarter, the API total, rows received, duplicate
  keys, a SHA-256 of the file and the UTC time, so completeness can be audited
  later and an interrupted run resumes where it stopped;
* the "latest" quarter is checked against the API before the run.

Usage: ``python run.py ingest [--start 1984-03-31 --end 2026-06-30]``
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import config as C

_local = threading.local()
_lock = threading.Lock()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "bft65b-student-pipeline/1.0"})
        _local.s = s
    return _local.s


def get_json(endpoint: str, params: dict, retries: int = 7) -> dict:
    url = f"{C.API_BASE}/{endpoint}"
    for k in range(1, retries + 1):
        try:
            r = _session().get(url, params=params, timeout=180)
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()
            msg = f"HTTP {r.status_code}"
        except requests.RequestException as exc:  # network hiccup
            msg = type(exc).__name__
        wait = min(90, 2 ** k)
        print(f"   {endpoint} {params.get('filters', '')}: {msg}, retry {k} in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url} {params}")


def fetch_all(endpoint: str, fields: list[str], filters: str | None = None,
              sort_by: str = "CERT") -> tuple[pd.DataFrame, int]:
    out, offset, total = [], 0, None
    while True:
        p = {"fields": ",".join(fields), "limit": C.PAGE_SIZE, "offset": offset,
             "sort_by": sort_by, "sort_order": "ASC", "format": "json"}
        if filters:
            p["filters"] = filters
        js = get_json(endpoint, p)
        total = int(js["meta"]["total"])
        page = [d["data"] for d in js["data"]]
        out.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    df = pd.DataFrame(out).drop(columns=["ID"], errors="ignore")
    return df, total


def quarter_ends(start: str, end: str) -> list[str]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    res = []
    for y in range(a.year, b.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            q = date(y, m, d)
            if a <= q <= b:
                res.append(q.strftime("%Y%m%d"))
    return res


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_path() -> Path:
    return C.RAW / "manifest.json"


def load_manifest() -> dict:
    p = _manifest_path()
    return json.loads(p.read_text()) if p.exists() else {"financials": {}, "tables": {}}


def _save_manifest(m: dict) -> None:
    tmp = _manifest_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True))
    tmp.replace(_manifest_path())


def _download_quarter(q: str) -> dict:
    t0 = time.time()
    df, total = fetch_all("financials", C.FIN_FIELDS, filters=f"REPDTE:{q}")
    target = C.RAW / "financials" / f"REPDTE={q}.parquet"
    if df.empty:
        return {"quarter": q, "rows": 0, "api_total": total, "complete": False, "note": "empty"}
    for col in C.FIN_FIELDS:      # keep a stable schema even if a field is absent in old years
        if col not in df.columns:
            df[col] = None
    df = df[C.FIN_FIELDS]
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)
    dups = int(df.duplicated(["CERT", "REPDTE"]).sum())
    return {"quarter": q, "rows": len(df), "api_total": total, "duplicate_keys": dups,
            "complete": bool(len(df) == total and dups == 0), "sha256": _sha256(target),
            "seconds": round(time.time() - t0, 1), "downloaded_at": _now()}


def download_financials(start: str = C.START_QUARTER, end: str = C.END_QUARTER,
                        workers: int = 2) -> dict:
    (C.RAW / "financials").mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    todo = [q for q in quarter_ends(start, end)
            if not (man["financials"].get(q, {}).get("complete")
                    and (C.RAW / "financials" / f"REPDTE={q}.parquet").exists())]
    print(f"{len(todo)} quarters to download ({len(quarter_ends(start, end)) - len(todo)} already complete)")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_download_quarter, q): q for q in todo}
        for n, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            with _lock:
                man["financials"][rec["quarter"]] = rec
                _save_manifest(man)
            print(f"[{n}/{len(todo)}] {rec['quarter']}: {rec['rows']:>6,} rows / API {rec['api_total']:,} "
                  f"{'OK' if rec['complete'] else 'INCOMPLETE'} {rec.get('seconds', 0)}s", flush=True)
    return man


def download_tables() -> dict:
    man = load_manifest()
    for name, fields, sort_by in (("failures", C.FAIL_FIELDS, "FAILDATE"),
                                  ("institutions", C.INST_FIELDS, "CERT")):
        df, total = fetch_all(name, fields, sort_by=sort_by)
        target = C.RAW / f"{name}.parquet"
        df.to_parquet(target, index=False)
        man["tables"][name] = {"rows": len(df), "api_total": total, "complete": len(df) == total,
                               "sha256": _sha256(target), "downloaded_at": _now()}
        print(f"{name}: {len(df):,} rows (API total {total:,})")
    _save_manifest(man)
    return man


def verify() -> dict:
    """Completeness check written to reports/tables/ingest_completeness.json."""
    man = load_manifest()
    expected = quarter_ends(C.START_QUARTER, C.END_QUARTER)
    fin = man["financials"]
    missing = [q for q in expected if q not in fin]
    incomplete = [q for q, r in fin.items() if not r["complete"]]
    sha_bad = [q for q, r in fin.items()
               if r.get("sha256") and _sha256(C.RAW / "financials" / f"REPDTE={q}.parquet") != r["sha256"]]
    keys = pd.read_parquet(C.RAW / "financials", columns=["CERT", "REPDTE"])
    times = sorted(r["downloaded_at"] for r in fin.values() if r.get("downloaded_at"))
    res = {
        "quarters_expected": len(expected), "quarters_downloaded": len(fin),
        "quarters_missing": missing, "quarters_incomplete": incomplete, "checksum_mismatch": sha_bad,
        "financials_rows": int(len(keys)), "api_total_sum": int(sum(r["api_total"] for r in fin.values())),
        "distinct_banks": int(keys["CERT"].nunique()),
        "duplicate_cert_repdte": int(keys.duplicated().sum()),
        "failures_rows": man["tables"]["failures"]["rows"],
        "institutions_rows": man["tables"]["institutions"]["rows"],
        "download_first": times[0] if times else None, "download_last": times[-1] if times else None,
        "tables_downloaded_at": {k: v["downloaded_at"] for k, v in man["tables"].items()},
    }
    res["complete"] = not (missing or incomplete or sha_bad or res["duplicate_cert_repdte"])
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (C.TABLES / "ingest_completeness.json").write_text(json.dumps(res, indent=1))
    rows_per_q = pd.DataFrame([{"quarter": q, "rows": r["rows"], "api_total": r["api_total"]}
                               for q, r in sorted(fin.items())])
    rows_per_q.to_csv(C.TABLES / "ingest_rows_per_quarter.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k != "tables_downloaded_at"}, indent=1))
    return res
