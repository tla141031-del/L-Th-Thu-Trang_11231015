# FDIC Bank-Failure Early Warning: a Reproducible Big-Data Pipeline

**BFT65B – Big Data Analytics in Finance, National Economics University – Final individual report**
**Student:** Lã Thị Thu Trang · **Student ID:** 11231015
**Repository:** https://github.com/tla141031-del/L-Th-Thu-Trang_11231015

This repository contains the full pipeline behind the report: acquisition of the public FDIC
BankFind data (Financials 1984Q1–2026Q2, Failures, Institutions), storage, an automated
data-quality report, a point-in-time bank-quarter panel, Polars feature engineering with a
pandas / PySpark benchmark, and a leakage-controlled early-warning model with a cost-based alarm
threshold and a logged evaluation on the test period (scored twice: a preliminary run before the availability fix and the final run; see below). Every number in the report is read from a
file in `reports/` written by this code (`report/build_report.py` generates the report itself).

## Quick start (end to end)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel                    # PySpark builds from source with old setuptools
pip install -r requirements.txt                        # Java 17/21 must be installed for PySpark

python run.py ingest        # ~11 min; resumable: rerun after an interruption, finished quarters are skipped
python run.py all           # verify -> store -> bench_storage -> quality -> panel -> features ->
                            # bench_processing -> spark -> experiments -> model -> test -> mllib -> report  (~1.8 h on 4 CPUs;
                            # model and test are skipped if the frozen choice was already scored, see below)
pytest -q                   # offline unit tests (label window, YTD de-cumulation, gaps, look-ahead)
python scripts/make_notebook.py   # re-executes notebooks/run_pipeline.ipynb with outputs
```

The report (`report/out/BFT65B_report.pdf` / `.docx`) is regenerated from `reports/` by `python run.py report`
(needs LibreOffice for the PDF; the DOCX is written without it).

**Test-period history and locks.** The test period has been scored twice, and both scorings are in
`reports/tables/test_log.jsonl`: a preliminary scoring on 2026-11-20 (PR-AUC 0.554) on a sample that still contained
rows whose report was not yet public, then a logged re-selection on development data after the availability fix, and the
final scoring on 2026-11-21 (PR-AUC 0.427, the result in the report). The final test is therefore not an untouched
holdout; the report says so. The locks below stop any further scoring, also under `all`. `python run.py model` (selection, development rows only)
freezes the choice in `reports/tables/model_choice.json` with a SHA-1 hash, and `python run.py test` scores it on the
test period once and appends the result to `reports/tables/test_log.jsonl`. `python run.py all` checks the log first:
if the frozen choice has already been scored (as in this repository), it **skips `model` and `test`** and keeps the
logged result, so rerunning the pipeline never re-scores the test period. Run on their own, `model` refuses to
re-select after a scoring unless `--reason "..."` is given (the reason is logged), and `test` refuses to score the same
or an edited choice again. To reproduce selection and the test scoring from scratch on a fresh copy, move
`model_choice.json` and `test_log.jsonl` aside first, then run `python run.py model` and `python run.py test` once.

Single stages: `python run.py <stage>`. A trial download: `python run.py ingest --start 2023-03-31 --end 2023-12-31`.
Data go to `data/` (git-ignored; set `FDIC_DATA_DIR` to use another folder). Seeds are fixed (`SEED = 2026`).
The notebook `notebooks/run_pipeline.ipynb` runs the same stages and keeps the outputs.

## Stages and outputs

| Stage | Code | Main outputs |
|---|---|---|
| ingest / verify | `src/fdic_ews/ingest.py` | `data/raw/` (one Parquet per quarter + `manifest.json` with API totals and SHA-256), `ingest_completeness.json` |
| store | `src/fdic_ews/storage.py` | `data/curated/financials/year=YYYY/` (typed, zstd, sorted), `storage_curated_info.json` |
| bench_storage | `storage.py` | `storage_benchmark.csv`, `figures/storage_benchmark.png` |
| panel | `src/fdic_ews/panel.py` | `data/curated/panel.parquet`, label/exit tables, `panel_checks.csv` |
| quality | `src/fdic_ews/quality.py` | `reports/data_quality_report.md`, `dq_*.csv`, `figures/dq_missing_by_year.png` |
| features | `src/fdic_ews/features.py` | `data/features/features.parquet`, `feature_dictionary.csv`, look-ahead test, Polars plans |
| bench_processing | `src/fdic_ews/bench_processing.py` | `processing_benchmark.csv` (pandas, Polars eager/lazy/streaming, PySpark) |
| spark | `src/fdic_ews/spark_check.py` | `spark_parity.json` (row-level match with Polars), `plan_spark_formatted.txt` |
| experiments | `src/fdic_ews/experiments.py` | `ingest_resume_test.json` (kill + restart a real download), `layout_effects.json` (schema, sort order), `engine_scans.csv` (Polars threads / optimiser, Spark cores / shuffle partitions), `feature_signal_dev.csv` (development rows only) |
| model | `src/fdic_ews/model.py` (`select`) | development only: CV tables, costs and `model_cfp_evidence.json`, threshold, frozen `model_choice.json` (SHA-1); refuses to re-select after a test scoring unless `--reason` is given |
| test | `src/fdic_ews/model.py` (`evaluate`) | test scoring of the frozen choice: `model_test_results.csv`, detection tables; refuses a second scoring of the same choice; every event in `test_log.jsonl` |
| mllib | `src/fdic_ews/mllib_demo.py` | `mllib_vs_sklearn.csv` |
| report | `report/build_report.py` | `report/out/BFT65B_report.docx` and `.pdf` |

## Leakage controls (summary)

* Label `y = 1` iff the bank fails in (q, q + 4 quarter-ends]; mergers and other exits are 0;
  failures = FDIC failures list with `RESTYPE = FAILURE` (assistance transactions excluded).
* Features at quarter q use reports dated ≤ q only; the look-ahead test recomputes all features on
  data truncated at a cut-off and finds no difference. Snapshot fields (`ACTIVE`, `ENDEFYMD`,
  `CHANGEC1`) and failure fields are never features.
* Availability: a report is usable from `REPDTE + 60 days`; rows of banks that failed on or before that date are
  removed from development and test (`model.load`).
* Development 1984Q1–2009Q4, gap 2010, test 2011Q1–2025Q2. Rolling-origin CV inside development,
  each fold trained only on rows whose label window had closed (purging).
* Clipping, imputation and scaling are inside a scikit-learn `Pipeline` fitted on training rows only.
* Model, features and threshold are frozen in `model_choice.json` before the test is scored (`run.py test`);
  `test_log.jsonl` records every scoring and re-selection: a preliminary scoring before the availability fix
  (0.554), the re-selection on development data, and the final scoring (0.427).

## Sources and acknowledgements

* Federal Deposit Insurance Corporation. *BankFind Suite API* and field dictionary, https://api.fdic.gov/banks/docs/ (data source).
* BFT65B course. `fdic_download_starter.py` (course page, 2026): kept unchanged in `scripts/`; `src/fdic_ews/ingest.py` is adapted from it (paging, retries, per-quarter files, completeness check).
* Software: Polars, pandas, PyArrow, PySpark / Spark MLlib, scikit-learn, SciPy, matplotlib, python-docx (versions in `requirements.txt`).
* Methods and papers cited in the code and report: Martin (1977); Cole & White (2012); Tukey (1977); Saito & Rehmsmeier (2015); Elkan (2001); López de Prado (2018); Pedregosa et al. (2011); Zaharia et al. (2016); Board of Governors of the Federal Reserve System (2023). The full reference list is in the report.
* **AI tool:** Claude (Anthropic, 2026) was used as a coding and writing assistant to draft code, tests and report text. The author reviewed, ran and checked the work; all numbers come from runs of this code.
* **Other AI tools:** ChatGPT (OpenAI, 2026) and Gemini (Google, 2026) were used only to review drafts of the report against the exam brief and rubric (consistency of numbers, wording, formatting). They wrote no code; every suggested change was checked against the generated tables before it was made.
