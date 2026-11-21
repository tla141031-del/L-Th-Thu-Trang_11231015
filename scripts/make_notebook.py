"""Create and execute notebooks/run_pipeline.ipynb (outputs kept), after `python run.py ingest`.

The notebook reruns the fast stages end to end (verify -> features, Spark parity) and
displays the evidence written by the slower stages (benchmarks, model, MLlib).
    python scripts/make_notebook.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
cells = [
    ("md", "# BFT65B – FDIC early-warning pipeline: executed run\n"
           "Student: Lã Thị Thu Trang (11231015). Every cell calls the same code as `python run.py <stage>`. "
           "Data must first be downloaded with `python run.py ingest` (resumable, ~11 min)."),
    ("code", "import os, sys, json, subprocess\nos.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None\n"
             "sys.path.insert(0, 'src')\nimport polars as pl\npl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(12)\n"
             "def run(stage):\n    out = subprocess.run([sys.executable, 'run.py', stage], capture_output=True, text=True)\n"
             "    print(out.stdout[-2500:]); print(out.stderr[-500:] if out.returncode else '')"),
    ("md", "## Part 1 – completeness and curated storage"),
    ("code", "run('verify')"),
    ("code", "run('store')\npl.read_csv('reports/tables/storage_benchmark.csv').select('format','query','size_mb','median_s','speedup_vs_csv','peak_rss_mb')"),
    ("md", "## Part 2 – point-in-time panel and data-quality report"),
    ("code", "run('panel')\npl.read_csv('reports/tables/panel_checks.csv')"),
    ("code", "run('quality')\nprint(open('reports/data_quality_report.md').read()[:6000])"),
    ("md", "## Part 3/4 – features (Polars lazy), look-ahead test, engines, PySpark parity"),
    ("code", "run('features')\nprint(json.load(open('reports/tables/features_lookahead_test.json')))\n"
             "print(open('reports/tables/plan_bench_polars_optimized.txt').read()[:3000])"),
    ("code", "pl.read_csv('reports/tables/processing_benchmark.csv')"),
    ("code", "run('spark')\nprint(open('reports/tables/plan_spark_formatted.txt').read()[:2500])"),
    ("md", "## Extra measurements behind design choices (results of `python run.py experiments`)"),
    ("code", "print(json.load(open('reports/tables/ingest_resume_test.json')))\nprint(json.load(open('reports/tables/layout_effects.json')))\n"
             "display(pl.read_csv('reports/tables/engine_scans.csv').select('experiment','scale','setting','median_s','peak_mb'))\n"
             "pl.read_csv('reports/tables/feature_signal_dev.csv').head(12)"),
    ("md", "## Part 5 – models (results of `python run.py model`, `python run.py test` and `python run.py mllib`)"),
    ("code", "display(pl.read_csv('reports/tables/model_split.csv'))\ndisplay(pl.read_csv('reports/tables/model_cv_summary.csv'))\n"
             "print(json.load(open('reports/tables/model_choice.json')))\nprint(open('reports/tables/test_log.jsonl').read())"),
    ("code", "display(pl.read_csv('reports/tables/model_test_results.csv'))\ndisplay(pl.read_csv('reports/tables/model_test_detection.csv'))\n"
             "pl.read_csv('reports/tables/mllib_vs_sklearn.csv')"),
]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "md" else nbformat.v4.new_code_cell(s) for k, s in cells]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "run_pipeline.ipynb"
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbformat.write(nb, out)
print("wrote", out)
