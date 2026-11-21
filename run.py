"""Single entry point: ``python run.py <stage>``.

Stages, in order: ingest, verify, store, bench_storage, quality, panel, features,
bench_processing, spark, experiments, model, test, mllib, report.  ``python run.py all`` runs
every stage after ``ingest`` (the download is resumable, so ``ingest`` can simply be rerun).
``all`` never re-scores the test period: if the frozen choice in reports/tables/model_choice.json
has already been scored (test_log.jsonl), it skips ``model`` and ``test`` and keeps the logged result.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

STAGES = ["ingest", "verify", "store", "bench_storage", "panel", "quality", "features",
          "bench_processing", "spark", "experiments", "model", "test", "mllib", "report"]


def run(stage: str, args) -> None:
    if stage == "ingest":
        from fdic_ews import ingest
        ingest.download_financials(args.start, args.end)
        ingest.download_tables()
    elif stage == "verify":
        from fdic_ews import ingest
        ingest.verify()
    elif stage == "store":
        from fdic_ews import storage
        storage.build_curated()
    elif stage == "bench_storage":
        from fdic_ews import storage
        storage.benchmark()
    elif stage == "quality":
        from fdic_ews import quality
        quality.run()
    elif stage == "panel":
        from fdic_ews import panel
        panel.build()
    elif stage == "features":
        from fdic_ews import features
        features.build()
    elif stage == "bench_processing":
        from fdic_ews import bench_processing
        bench_processing.run()
    elif stage == "spark":
        from fdic_ews import spark_check
        spark_check.run()
    elif stage == "experiments":
        from fdic_ews import experiments
        experiments.run()
    elif stage == "model":
        from fdic_ews import model
        model.select(args.reason)
    elif stage == "test":
        from fdic_ews import model
        model.evaluate(args.force)
    elif stage == "mllib":
        from fdic_ews import mllib_demo
        mllib_demo.run()
    elif stage == "report":
        import runpy
        runpy.run_path(str(Path(__file__).resolve().parent / "report" / "build_report.py"), run_name="__main__")


def main() -> None:
    from fdic_ews import config as C
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--start", default=C.START_QUARTER)
    ap.add_argument("--end", default=C.END_QUARTER)
    ap.add_argument("--reason", default=None, help="model: why a choice already used on the test is re-selected")
    ap.add_argument("--force", action="store_true", help="test: score a frozen choice again (logged)")
    a = ap.parse_args()
    for st in (STAGES[1:] if a.stage == "all" else [a.stage]):
        if a.stage == "all" and st in ("model", "test"):
            from fdic_ews import model as M
            if M.frozen_choice_scored():
                print(f"=== {st} skipped: the frozen choice was already scored once (see test_log.jsonl); "
                      f"run `python run.py {st}` on its own to see why it refuses", flush=True)
                continue
        t0 = time.perf_counter()
        print(f"=== {st}", flush=True)
        run(st, a)
        print(f"=== {st} done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
