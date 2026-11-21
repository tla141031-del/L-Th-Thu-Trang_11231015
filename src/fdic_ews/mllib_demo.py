"""Part 5 - short Spark MLlib demonstration.

Same task as the last development CV fold (train: labels known by 2004-12-31,
validate 2006Q1-2009Q4), fitted twice:
  * Spark MLlib Pipeline: Imputer(median) -> VectorAssembler -> StandardScaler ->
    LogisticRegression, and a GBTClassifier on the same vector;
  * scikit-learn: the same logistic regression and a gradient-boosting model.
Everything (medians, scaling) is fitted on the training rows only, inside each Pipeline.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .bench_processing import spark_session
from .model import BASELINE, X, fold_frames, load, make_model

REG = 0.001


def run() -> dict:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    dev, _ = load()
    tr, va = fold_frames(dev, C.CV_FOLDS[-1])
    feats = BASELINE + ["texas_ratio", "tier1_leverage", "d4_npl_ratio", "construction_share",
                        "brokered_to_deposits", "roa_std4", "npl_rank_q", "state_d4_npl_median"]
    cols = ["y"] + feats
    tmp = C.DATA / "mllib"
    tmp.mkdir(parents=True, exist_ok=True)
    tr.select(cols).write_parquet(tmp / "train.parquet")
    va.select(cols).write_parquet(tmp / "valid.parquet")

    t0 = time.perf_counter()
    spark = spark_session("bft65b-mllib")
    start_s = time.perf_counter() - t0
    str_ = spark.read.parquet(str(tmp / "train.parquet")).withColumn("y", F.col("y").cast("double"))
    sva = spark.read.parquet(str(tmp / "valid.parquet")).withColumn("y", F.col("y").cast("double"))
    # clip at training quantiles (same idea as the scikit-learn Winsorizer)
    q = {c: str_.approxQuantile(c, [0.005, 0.995], 1e-4) for c in feats}
    clip = [F.least(F.greatest(F.col(c), F.lit(q[c][0])), F.lit(q[c][1])).alias(c) if q[c] else F.col(c) for c in feats]
    str_, sva = str_.select("y", *clip), sva.select("y", *clip)
    imp = Imputer(inputCols=feats, outputCols=[f"{c}_i" for c in feats], strategy="median")
    vec = VectorAssembler(inputCols=[f"{c}_i" for c in feats], outputCol="v")
    sc = StandardScaler(inputCol="v", outputCol="features", withMean=True, withStd=True)
    out = []
    for name, est in (("logistic", LogisticRegression(labelCol="y", featuresCol="features", regParam=REG, maxIter=300)),
                      ("gbt", GBTClassifier(labelCol="y", featuresCol="features", maxIter=60, maxDepth=4, seed=C.SEED))):
        t1 = time.perf_counter()
        model = Pipeline(stages=[imp, vec, sc, est]).fit(str_)
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        pr = model.transform(sva).select("y", vector_to_array("probability")[1].alias("p")).toPandas()
        pred_s = time.perf_counter() - t2
        out.append({"engine": "spark-mllib", "model": name, "fit_s": fit_s, "predict_s": pred_s,
                    "pr_auc": average_precision_score(pr["y"], pr["p"]), "roc_auc": roc_auc_score(pr["y"], pr["p"])})
        if name == "logistic":
            p_spark_lr = pr["p"].to_numpy()
    spark.stop()

    ytr, yva = tr["y"].to_numpy(), va["y"].to_numpy()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression as SkLR
    from sklearn.pipeline import Pipeline as SkPipe
    from sklearn.preprocessing import StandardScaler as SkScaler
    from .model import Winsorizer
    # same objective as Spark: mean log-loss + regParam/2 ||w||^2  <=>  C = 1 / (regParam * n)
    sk_lr = SkPipe([("clip", Winsorizer()), ("impute", SimpleImputer(strategy="median")),
                    ("scale", SkScaler()), ("clf", SkLR(C=1 / (REG * len(ytr)), max_iter=3000))])
    for name, m in (("logistic", sk_lr),
                    ("gbt", make_model("hgb", iters=60, leaves=16, lr=0.1))):
        t1 = time.perf_counter()
        m.fit(X(tr, feats), ytr)
        fit_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        p = m.predict_proba(X(va, feats))[:, 1]
        out.append({"engine": "scikit-learn", "model": name, "fit_s": fit_s, "predict_s": time.perf_counter() - t2,
                    "pr_auc": average_precision_score(yva, p), "roc_auc": roc_auc_score(yva, p)})
        if name == "logistic":
            p_sk_lr = p
    from scipy.stats import spearmanr
    agree = {"spearman_logistic_probs": float(spearmanr(p_spark_lr, p_sk_lr).statistic),
             "spark_session_start_s": start_s, "train_rows": tr.height, "valid_rows": va.height,
             "features": feats}
    pl.DataFrame(out).write_csv(C.TABLES / "mllib_vs_sklearn.csv")
    (C.TABLES / "mllib_agreement.json").write_text(json.dumps(agree, indent=1))
    print(pl.DataFrame(out), agree)
    return agree
