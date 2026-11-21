"""Part 4b/5 - leak-free preprocessing, model comparison, cost-based threshold, single test.

Order of operations (nothing below step 5 is allowed to influence steps 1-4):
 1. development rows 1984Q1-2009Q4, test rows 2011Q1-2025Q2, 2010 is a 4-quarter gap;
 2. rolling-origin CV inside development: each fold trains only on rows whose
    label window ended before the training cut-off (purging), validates on a later block;
 3. model / feature-set / hyper-parameter choice by mean CV PR-AUC;
 4. alarm threshold = cost minimiser on the pooled out-of-fold CV predictions, with
    costs estimated from the FDIC failures table (development failures only);
 5. the frozen choice is written to reports/tables/model_choice.json (`python run.py model`);
 6. a separate stage (`python run.py test`) refits the frozen choice on all development rows and
    scores the test period. It refuses to run twice for the same frozen choice, and `model`
    refuses to re-select after a test scoring unless a reason is given; both events are logged
    in reports/tables/test_log.jsonl, so every look at the test period is on record.

Availability rule: a decision for quarter q can only be taken once the report is public
(available_date = REPDTE + 60 days). Rows of banks that failed on or before that date cannot
be acted on, so they are removed from development and test alike (load()).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import DICT, FEATURES

BASELINE = ["equity_to_assets", "npl_ratio", "roa_q", "loans_to_deposits"]
GROUPS = {
    "levels": [f for f, (p, _) in DICT.items() if p in ("C", "A", "M", "E", "L", "S")],
    "dynamics": [f for f, (p, _) in DICT.items() if p == "dyn"],
    "cross_section": [f for f, (p, _) in DICT.items() if p == "xs"],
}
FEATURE_SETS = {
    "levels": GROUPS["levels"],
    "levels+dyn": GROUPS["levels"] + GROUPS["dynamics"],
    "all": FEATURES,
}
TOPK = 0.02            # alert budget: top 2% of banks each quarter


# --------------------------------------------------------------------------- preprocessing
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at quantiles learned on the training rows (NaN kept)."""

    def __init__(self, q: float = 0.005):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # a column never observed in the training window (e.g. RBCRWAJ before 1990) is set to a
        # constant 0, so the column count stays fixed and no later value can leak into the fit
        self.observed_ = np.isfinite(X).any(axis=0)
        lo = np.full(X.shape[1], -np.inf)
        hi = np.full(X.shape[1], np.inf)
        if self.observed_.any():
            lo[self.observed_] = np.nanquantile(X[:, self.observed_], self.q, axis=0)
            hi[self.observed_] = np.nanquantile(X[:, self.observed_], 1 - self.q, axis=0)
        self.lo_, self.hi_ = lo, hi
        return self

    def transform(self, X):
        X = np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
        X[:, ~self.observed_] = 0.0
        return X


def make_model(kind: str, **kw) -> Pipeline:
    if kind == "logit":
        return Pipeline([("clip", Winsorizer()), ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=kw.get("C", 1.0), max_iter=2000,
                                                    class_weight=kw.get("class_weight")))])
    if kind == "hgb":
        return Pipeline([("clip", Winsorizer()),
                         ("clf", HistGradientBoostingClassifier(
                             learning_rate=kw.get("lr", 0.06), max_leaf_nodes=kw.get("leaves", 31),
                             min_samples_leaf=kw.get("min_leaf", 200), l2_regularization=1.0,
                             max_iter=kw.get("iters", 300), class_weight=kw.get("class_weight"),
                             early_stopping=False, random_state=C.SEED))])
    raise ValueError(kind)


CANDIDATES = [
    # name, kind, feature set, params
    ("baseline_logit_4", "logit", "baseline", {}),
    ("logit_all", "logit", "all", {"C": 0.1}),
    ("logit_all_balanced", "logit", "all", {"C": 0.1, "class_weight": "balanced"}),
    ("hgb_levels", "hgb", "levels", {}),
    ("hgb_levels+dyn", "hgb", "levels+dyn", {}),
    ("hgb_all", "hgb", "all", {}),
    ("hgb_all_balanced", "hgb", "all", {"class_weight": "balanced"}),
    ("hgb_all_small", "hgb", "all", {"leaves": 15, "lr": 0.05, "min_leaf": 500}),
]


def cols_of(fs: str) -> list[str]:
    return BASELINE if fs == "baseline" else FEATURE_SETS[fs]


# --------------------------------------------------------------------------- metrics
def recall_at_topk(df: pl.DataFrame, score: np.ndarray, k: float = TOPK) -> tuple[float, float]:
    d = df.select("REPDTE", "y").with_columns(pl.Series("s", score))
    d = d.with_columns((pl.col("s").rank("ordinal", descending=True).over("REPDTE")
                        <= (pl.len().over("REPDTE") * k).ceil()).alias("alert"))
    tp = d.filter(pl.col("alert") & (pl.col("y") == 1)).height
    return tp / max(1, int(d["y"].sum())), tp / max(1, int(d["alert"].sum()))


def metrics(df: pl.DataFrame, p: np.ndarray) -> dict:
    y = df["y"].to_numpy()
    r, pr = recall_at_topk(df, p)
    return {"rows": len(y), "positives": int(y.sum()), "base_rate_pct": 100 * y.mean(),
            "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "recall_top2pct": r, "precision_top2pct": pr}


# --------------------------------------------------------------------------- data
USABLE = pl.col("fail_date").is_null() | (pl.col("fail_date") > pl.col("available_date"))


def load(with_excluded: bool = False):
    """Development and test rows whose label is complete and whose report was public before any
    failure (USABLE). with_excluded=True also returns the removed rows per split, for the report."""
    f = pl.read_parquet(C.FEATURES / "features.parquet")
    f = f.filter(pl.col("label_complete") & (pl.col("REPDTE") <= date.fromisoformat(C.LAST_LABEL_QUARTER)))
    if with_excluded:
        ex = f.filter(~USABLE)
        exd = ex.filter(pl.col("REPDTE").is_between(date.fromisoformat(C.DEV_START), date.fromisoformat(C.DEV_END)))
        ext = ex.filter(pl.col("REPDTE").is_between(date.fromisoformat(C.TEST_START), date.fromisoformat(C.TEST_END)))
    f = f.filter(USABLE)
    dev = f.filter(pl.col("REPDTE").is_between(date.fromisoformat(C.DEV_START), date.fromisoformat(C.DEV_END)))
    test = f.filter(pl.col("REPDTE").is_between(date.fromisoformat(C.TEST_START), date.fromisoformat(C.TEST_END)))
    if with_excluded:
        return dev, test, exd, ext
    return dev, test


def bank_reconciliation() -> pl.DataFrame:
    """Where each failed bank with a positive row ends up (labels and dates only; no model, no scores)."""
    f = pl.read_parquet(C.FEATURES / "features.parquet", columns=["CERT", "REPDTE", "y", "label_complete",
                                                                  "fail_date", "available_date"]).filter(pl.col("y") == 1)
    d = date.fromisoformat
    f = f.with_columns(
        pl.when(pl.col("REPDTE") <= d(C.DEV_END)).then(pl.lit("development"))
        .when(pl.col("REPDTE") < d(C.TEST_START)).then(pl.lit("gap"))
        .when((pl.col("REPDTE") <= d(C.TEST_END)) & pl.col("label_complete")).then(pl.lit("test"))
        .otherwise(pl.lit("after")).alias("seg"), USABLE.alias("ok"))
    b = f.group_by("CERT").agg(pl.col("seg").filter(pl.col("ok")).alias("useg"), pl.col("seg").alias("seg"))
    order = ["development: usable positive row", "development: every positive row removed by availability",
             "gap 2010 only (never scored)", "test: usable positive row",
             "test: every positive row removed by availability", "after 2025Q2 only (label window not closed)"]

    def cls(u, s):
        if "development" in u: return order[0]
        if "test" in u: return order[3]
        if "development" in s: return order[1]
        if "test" in s: return order[4]
        if "gap" in s: return order[2]
        return order[5]
    b = b.with_columns(pl.struct("useg", "seg").map_elements(lambda r: cls(r["useg"], r["seg"]),
                                                             return_dtype=pl.Utf8).alias("group"))
    out = (pl.DataFrame({"group": order}).join(b.group_by("group").len().rename({"len": "banks"}), on="group", how="left")
           .fill_null(0))
    out.write_csv(C.TABLES / "model_bank_reconciliation.csv")
    return out


def X(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(cols).to_numpy().astype(float)


def fold_frames(dev: pl.DataFrame, fold) -> tuple[pl.DataFrame, pl.DataFrame]:
    lab_end, v0, v1 = (date.fromisoformat(s) for s in fold)
    tr = dev.filter(pl.col("REPDTE").dt.offset_by("12mo").dt.month_end() <= lab_end)  # label known by lab_end
    va = dev.filter(pl.col("REPDTE").is_between(v0, v1))
    return tr, va


# --------------------------------------------------------------------------- costs
def estimate_costs(alpha: float = 0.2, cfp: float = 50.0) -> dict:
    """Missed failure: share alpha of the FDIC resolution cost is assumed avoidable by early
    action (development failures only). False alarm: cost of one targeted examination (USD k)."""
    f = pl.read_parquet(C.CURATED / "failures.parquet").filter(
        (pl.col("RESTYPE") == "FAILURE") & pl.col("CERT").is_not_null()
        & pl.col("FAILDATE").is_between(date(1984, 4, 1), date.fromisoformat(C.DEV_END)))
    cost = f["COST"].drop_nulls()
    panel = pl.read_parquet(C.CURATED / "panel.parquet", columns=["CERT", "REPDTE", "y", "fail_date", "available_date"]).filter(
        (pl.col("REPDTE") <= date.fromisoformat(C.DEV_END)) & USABLE)
    pos = panel.filter(pl.col("y") == 1).group_by("CERT").len()["len"]
    loss_rate = (f.filter(pl.col("QBFASSET") > 0).select((pl.col("COST") / pl.col("QBFASSET")).median()).item())
    cfn_bank = alpha * float(cost.median())
    rows_per_bank = float(pos.mean())
    return {"failures_dev": f.height, "with_cost": cost.len(), "cost_median_k": float(cost.median()),
            "cost_mean_k": float(cost.mean()), "cost_p25_k": float(cost.quantile(0.25)),
            "cost_p75_k": float(cost.quantile(0.75)), "loss_rate_median": float(loss_rate),
            "alpha": alpha, "cfn_per_bank_k": cfn_bank, "positive_rows_per_failed_bank": rows_per_bank,
            "cfn_per_row_k": cfn_bank / rows_per_bank, "cfp_k": cfp,
            "bayes_threshold": cfp / (cfp + cfn_bank / rows_per_bank)}


# FDIC 2024 Annual Report, Budget section (https://fdic.gov/financial-reports/2024-annual-report-section3):
# 2024 operating budget about USD 3.0 billion, of which about 49% for the Supervision and Consumer
# Protection program. Used only to bound the false-alarm cost, never in the model.
FDIC_BUDGET_2024_K = 3_000_000.0
SUPERVISION_SHARE = 0.49


def cfp_evidence(oof_y: np.ndarray | None = None, oof_p: np.ndarray | None = None, cfn_row: float | None = None) -> dict:
    """Ground the false-alarm cost C_FP in data: (a) FDIC supervision spending per FDIC-supervised
    institution (state non-member banks NM, savings banks SB, state savings associations SL, from the
    institutions table); (b) the break-even C_FP below which the alarm rule beats 'no model' at a
    given recall, from the pooled out-of-fold development predictions."""
    inst = pl.read_parquet(C.CURATED / "institutions.parquet")
    n_sup = inst.filter((pl.col("ACTIVE") == 1) & pl.col("BKCLASS").is_in(["NM", "SB", "SL"])).height
    per_bank = FDIC_BUDGET_2024_K * SUPERVISION_SHARE / n_sup
    out = {"fdic_supervised_active_banks": n_sup, "supervision_budget_k": FDIC_BUDGET_2024_K * SUPERVISION_SHARE,
           "supervision_cost_per_bank_year_k": per_bank}
    if oof_y is not None:
        order = np.argsort(-oof_p)
        tp, fp = np.cumsum(oof_y[order]), np.cumsum(1 - oof_y[order])
        P = oof_y.sum()
        for rec in (0.3, 0.5, 0.7):
            k = int(np.searchsorted(tp, rec * P))
            out[f"breakeven_cfp_k_at_recall_{int(rec * 100)}"] = float(cfn_row * tp[k] / max(1, fp[k]))
    return out


def cost_curve(y: np.ndarray, p: np.ndarray, cfn: float, cfp: float):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    cost = cfn * fn + cfp * fp
    k = int(np.argmin(cost))
    return ps, cost, ps[k], {"tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k]), "cost_k": float(cost[k]),
                             "cost_alert_none": float(cfn * ys.sum()), "cost_alert_all": float(cfp * (1 - ys).sum())}


def at_threshold(y, p, t, cfn, cfp) -> dict:
    a = p >= t
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    fn = int(((~a) & (y == 1)).sum())
    return {"threshold": float(t), "alerts": int(a.sum()), "tp": tp, "fp": fp, "fn": fn,
            "recall": tp / max(1, int(y.sum())), "precision": tp / max(1, int(a.sum())),
            "cost_k": cfn * fn + cfp * fp, "cost_no_model_k": cfn * int(y.sum())}


# --------------------------------------------------------------------------- main
LOG = C.TABLES / "test_log.jsonl"
CHOICE = C.TABLES / "model_choice.json"


def _log() -> list[dict]:
    return [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()] if LOG.exists() else []


def _append_log(rec: dict) -> None:
    with open(LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def select(reason: str | None = None) -> dict:
    """Steps 1-5, development data only. Writes and freezes model_choice.json."""
    if CHOICE.exists():
        old = json.loads(CHOICE.read_text())
        scored = any(r.get("event", "test_scored") == "test_scored" and r["choice_sha1"] == old["sha1"] for r in _log())
        if scored and not reason:
            raise SystemExit("model_choice.json was already used on the test period; re-selecting would let the "
                             "test influence the choice. Pass --reason '...' to re-select on purpose (it is logged).")
    else:
        old, scored = None, False
    dev, test, exd, ext = load(with_excluded=True)
    print(f"dev {dev.height:,} rows ({int(dev['y'].sum())} pos), test {test.height:,} rows ({int(test['y'].sum())} pos)")
    split_tab = []
    for name, df, ex in (("development", dev, exd), ("test", test, ext)):
        split_tab.append({"split": name, "from": str(df["REPDTE"].min()), "to": str(df["REPDTE"].max()),
                          "rows": df.height, "banks": df["CERT"].n_unique(), "positive_rows": int(df["y"].sum()),
                          "failed_banks": int(df.filter(pl.col("y") == 1)["CERT"].n_unique()),
                          "base_rate_pct": 100 * float(df["y"].mean()),
                          "rows_excluded_unavailable": ex.height,
                          "positive_rows_excluded": int(ex["y"].sum()),
                          "failed_banks_only_in_excluded_rows": len(set(ex.filter(pl.col("y") == 1)["CERT"].to_list())
                                                                    - set(df.filter(pl.col("y") == 1)["CERT"].to_list()))})
    pl.DataFrame(split_tab).write_csv(C.TABLES / "model_split.csv")
    bank_reconciliation()

    leakage_proof(dev, test)

    # ---- CV
    fold_rows, oof = [], {}
    for name, kind, fs, params in CANDIDATES:
        cols = cols_of(fs)
        oof[name] = []
        for k, fold in enumerate(C.CV_FOLDS, 1):
            tr, va = fold_frames(dev, fold)
            t0 = time.perf_counter()
            m = make_model(kind, **params).fit(X(tr, cols), tr["y"].to_numpy())
            fit_s = time.perf_counter() - t0
            p = m.predict_proba(X(va, cols))[:, 1]
            oof[name].append((va.select("CERT", "REPDTE", "y"), p))
            fold_rows.append({"model": name, "fold": k, "train_rows": tr.height, "train_pos": int(tr["y"].sum()),
                              "valid": f"{fold[1]}..{fold[2]}", "fit_s": fit_s, **metrics(va, p)})
            print(f"{name:20s} fold {k}: PR-AUC {fold_rows[-1]['pr_auc']:.3f} ({fit_s:.1f}s)", flush=True)
    folds = pl.DataFrame(fold_rows)
    folds.write_csv(C.TABLES / "model_cv_folds.csv")
    cv = (folds.group_by("model").agg(pl.col("pr_auc").mean().alias("pr_auc_mean"),
                                       pl.col("pr_auc").std().alias("pr_auc_sd"),
                                       pl.col("roc_auc").mean().alias("roc_auc_mean"),
                                       pl.col("recall_top2pct").mean().alias("recall_top2pct_mean"),
                                       pl.col("brier").mean().alias("brier_mean"),
                                       pl.col("fit_s").mean().alias("fit_s_mean"))
          .sort("pr_auc_mean", descending=True))
    cv.write_csv(C.TABLES / "model_cv_summary.csv")
    chosen = cv.filter(pl.col("model") != "baseline_logit_4")["model"][0]
    spec = {c[0]: c for c in CANDIDATES}

    # ---- costs and threshold from the pooled out-of-fold predictions (development only)
    costs = estimate_costs()
    (C.TABLES / "model_costs.json").write_text(json.dumps(costs, indent=1))
    thr = {}
    for name in (chosen, "baseline_logit_4"):
        y = np.concatenate([d["y"].to_numpy() for d, _ in oof[name]])
        p = np.concatenate([q for _, q in oof[name]])
        ps, cost, t, info = cost_curve(y, p, costs["cfn_per_row_k"], costs["cfp_k"])
        thr[name] = {"threshold": float(t), **info}
        if name == chosen:
            _plot_cost(ps, cost, t, costs)
    sens = []
    y_c = np.concatenate([d["y"].to_numpy() for d, _ in oof[chosen]])
    p_c = np.concatenate([q for _, q in oof[chosen]])
    for alpha in (0.1, 0.2, 0.4):
        for cfp in (20.0, 50.0, 100.0, 200.0):
            c2 = estimate_costs(alpha, cfp)
            _, _, t, info = cost_curve(y_c, p_c, c2["cfn_per_row_k"], cfp)
            sens.append({"alpha": alpha, "cfp_k": cfp, "cfn_row_k": c2["cfn_per_row_k"],
                         "cfn_over_cfp": c2["cfn_per_row_k"] / cfp, "threshold_dev": float(t),
                         "dev_recall": info["tp"] / max(1, int(y_c.sum())), "dev_alerts": info["tp"] + info["fp"]})

    choice = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "chosen_model": chosen,
              "kind": spec[chosen][1], "feature_set": spec[chosen][2], "params": spec[chosen][3],
              "features": cols_of(spec[chosen][2]), "threshold": thr[chosen]["threshold"],
              "baseline_threshold": thr["baseline_logit_4"]["threshold"],
              "cfn_per_row_k": costs["cfn_per_row_k"], "cfp_k": costs["cfp_k"],
              "dev_end": C.DEV_END, "test": [C.TEST_START, C.TEST_END],
              "sample_rule": "label_complete & (fail_date is null | fail_date > REPDTE + 60 days)",
              "supersedes": old["sha1"] if old else None, "reason": reason}
    choice["sha1"] = hashlib.sha1(json.dumps(choice, sort_keys=True).encode()).hexdigest()
    CHOICE.write_text(json.dumps(choice, indent=1))
    (C.TABLES / "model_threshold_dev.json").write_text(json.dumps(thr, indent=1))
    (C.TABLES / "model_cost_sensitivity_dev.csv").write_text(pl.DataFrame(sens).write_csv())
    (C.TABLES / "model_cfp_evidence.json").write_text(json.dumps(cfp_evidence(y_c, p_c, costs["cfn_per_row_k"]), indent=1))
    if scored:
        _append_log({"event": "reselected", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "old_choice_sha1": old["sha1"], "new_choice_sha1": choice["sha1"], "reason": reason})
    print("frozen:", choice["chosen_model"], round(choice["threshold"], 4), choice["sha1"][:10])

    # ---- importance on the last fold (development only)
    perm_importance(dev, spec[chosen])
    _plot_cv(folds)
    return choice


def frozen_choice_scored() -> bool:
    """True if the current frozen choice has already been scored on the test period."""
    if not CHOICE.exists():
        return False
    sha = json.loads(CHOICE.read_text()).get("sha1")
    return any(r.get("event", "test_scored") == "test_scored" and r.get("choice_sha1") == sha for r in _log())


def evaluate(force: bool = False) -> None:
    """Step 6: refit the frozen choice on all development rows and score the test period once."""
    choice = json.loads(CHOICE.read_text())
    body = {k: v for k, v in choice.items() if k != "sha1"}
    if hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest() != choice["sha1"]:
        raise SystemExit("model_choice.json was edited after it was frozen")
    if any(r.get("event", "test_scored") == "test_scored" and r["choice_sha1"] == choice["sha1"] for r in _log()) and not force:
        raise SystemExit("this frozen choice has already been scored on the test period (see test_log.jsonl)")
    dev, test = load()
    spec = {c[0]: c for c in CANDIDATES}
    chosen = choice["chosen_model"]
    costs = {"cfn_per_row_k": choice["cfn_per_row_k"], "cfp_k": choice["cfp_k"]}
    sens = pl.read_csv(C.TABLES / "model_cost_sensitivity_dev.csv").to_dicts()
    results, preds = [], {}
    for name in (chosen, "baseline_logit_4"):
        _, kind, fs, params = spec[name]
        cols = cols_of(fs)
        m = make_model(kind, **params).fit(X(dev, cols), dev["y"].to_numpy())
        p = m.predict_proba(X(test, cols))[:, 1]
        preds[name] = p
        t = choice["threshold"] if name == chosen else choice["baseline_threshold"]
        y = test["y"].to_numpy()
        res = {"model": name, "chosen": name == chosen, **metrics(test, p),
               **at_threshold(y, p, t, costs["cfn_per_row_k"], costs["cfp_k"])}
        results.append(res)
        if name == chosen:
            import joblib
            (C.DATA / "models").mkdir(parents=True, exist_ok=True)
            joblib.dump(m, C.DATA / "models" / "final_model.joblib")
    pl.DataFrame(results).write_csv(C.TABLES / "model_test_results.csv")
    _append_log({"event": "test_scored", "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "choice_sha1": choice["sha1"], "pr_auc": results[0]["pr_auc"], "forced": force})
    # sensitivity applied to the test (thresholds fixed on development)
    y = test["y"].to_numpy()
    for s in sens:
        r = at_threshold(y, preds[chosen], s["threshold_dev"], s["cfn_row_k"], s["cfp_k"])
        s.update({"test_recall": r["recall"], "test_alerts": r["alerts"], "test_cost_k": r["cost_k"],
                  "test_cost_no_model_k": r["cost_no_model_k"]})
    pl.DataFrame(sens).write_csv(C.TABLES / "model_cost_sensitivity.csv")
    detection(test, preds, choice)
    _plot_pr(test, preds)
    print(pl.DataFrame(results))


def leakage_proof(dev: pl.DataFrame, test: pl.DataFrame) -> None:
    """Fit the preprocessing on the first training fold and show that its learned numbers equal
    the training-window statistics and change when the window changes, i.e. they never see later data."""
    tr, _ = fold_frames(dev, C.CV_FOLDS[0])
    allrows = pl.concat([dev, test])
    m = make_model("logit", C=0.1).fit(X(tr, FEATURES), tr["y"].to_numpy())
    clip, imp = m.named_steps["clip"], m.named_steps["impute"]
    rows = []
    Xtr = clip.transform(X(tr, FEATURES))
    for j, c in enumerate(FEATURES):
        tr_med = float(np.nanmedian(Xtr[:, j])) if np.isfinite(Xtr[:, j]).any() else np.nan
        all_hi = float(np.nanquantile(X(allrows, [c])[:, 0], 0.995))
        rows.append({"feature": c, "learned_median": float(imp.statistics_[j]), "train_median": tr_med,
                     "learned_p99_5": float(clip.hi_[j]), "all_periods_p99_5": all_hi,
                     "rel_shift_if_fitted_on_all_pct": 100 * abs(all_hi - clip.hi_[j]) / max(1e-12, abs(clip.hi_[j]))})
    df = pl.DataFrame(rows)
    df.write_csv(C.TABLES / "leakage_preprocessing.csv")
    ok = bool(np.allclose(df["learned_median"].to_numpy(), df["train_median"].to_numpy(), equal_nan=True))
    # transform of a test row does not depend on other test rows
    Xt = X(test.head(2000), FEATURES)
    a = m[:-1].transform(Xt)[:5]
    b = m[:-1].transform(Xt[:5])
    summary = {"fold1_train_rows": tr.height, "medians_equal_train": ok, "row_wise_transform": bool(np.allclose(a, b)),
               "median_rel_shift_clip_pct": float(df["rel_shift_if_fitted_on_all_pct"].median())}
    (C.TABLES / "leakage_summary.json").write_text(json.dumps(summary, indent=1))
    assert ok and summary["row_wise_transform"]


def perm_importance(dev: pl.DataFrame, spec) -> None:
    from sklearn.inspection import permutation_importance
    name, kind, fs, params = spec
    cols = cols_of(fs)
    tr, va = fold_frames(dev, C.CV_FOLDS[-1])
    m = make_model(kind, **params).fit(X(tr, cols), tr["y"].to_numpy())
    rng = np.random.default_rng(C.SEED)
    idx = np.where(va["y"].to_numpy() == 1)[0]
    neg = np.where(va["y"].to_numpy() == 0)[0]
    sub = np.concatenate([idx, rng.choice(neg, size=min(len(neg), 60_000), replace=False)])
    r = permutation_importance(m, X(va, cols)[sub], va["y"].to_numpy()[sub], scoring="average_precision",
                               n_repeats=5, random_state=C.SEED, n_jobs=1)
    pl.DataFrame({"feature": cols, "pr_auc_drop_mean": r.importances_mean, "pr_auc_drop_sd": r.importances_std}) \
      .sort("pr_auc_drop_mean", descending=True).write_csv(C.TABLES / "model_permutation_importance.csv")


def detection(test: pl.DataFrame, preds: dict, choice: dict) -> None:
    rows, by_year = [], []
    for name, p in preds.items():
        t = choice["threshold"] if name == choice["chosen_model"] else choice["baseline_threshold"]
        d = test.select("CERT", "REPDTE", "y", "fail_date").with_columns(pl.Series("p", p))
        d = d.with_columns((pl.col("p").rank("ordinal", descending=True).over("REPDTE")
                            <= (pl.len().over("REPDTE") * TOPK).ceil()).alias("topk"),
                           (pl.col("p") >= t).alias("alert"))
        fb = (d.filter(pl.col("y") == 1).group_by("CERT")
              .agg(pl.col("fail_date").first(), pl.col("alert").any().alias("det"), pl.col("topk").any().alias("det_topk"),
                   pl.col("REPDTE").filter(pl.col("alert")).min().alias("first_alert")))
        fb = fb.with_columns(((pl.col("fail_date") - pl.col("first_alert")).dt.total_days() / 91.3).alias("lead_q"))
        rows.append({"model": name, "failed_banks": fb.height, "detected_at_threshold": int(fb["det"].sum()),
                     "detected_top2pct": int(fb["det_topk"].sum()),
                     "median_lead_quarters": float(fb["lead_q"].median() or 0)})
        by_year.append(fb.group_by(pl.col("fail_date").dt.year().alias("fail_year"))
                       .agg(pl.len().alias("failed_banks"), pl.col("det").sum().alias("detected"),
                            pl.col("det_topk").sum().alias("detected_top2pct")).with_columns(pl.lit(name).alias("model")))
    pl.DataFrame(rows).write_csv(C.TABLES / "model_test_detection.csv")
    pl.concat(by_year).sort("model", "fail_year").write_csv(C.TABLES / "model_test_detection_by_year.csv")


def _plot_cost(ps, cost, t, costs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(ps, cost / 1000, lw=1.5)
    ax.axvline(t, color="k", ls="--", lw=1, label=f"cost-minimising threshold {t:.3f}")
    ax.axvline(costs["bayes_threshold"], color="grey", ls=":", lw=1, label=f"Bayes threshold {costs['bayes_threshold']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("alarm threshold on predicted probability")
    ax.set_ylabel("expected cost (USD million)")
    ax.legend(fontsize=8)
    ax.set_title("Expected cost on pooled out-of-fold development predictions", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_cost_curve.png", dpi=170)
    plt.close(fig)


def _plot_pr(test, preds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    y = test["y"].to_numpy()
    for name, p in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="grey", ls=":", label="base rate")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_pr_test.png", dpi=170)
    plt.close(fig)


def _plot_cv(folds: pl.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for name in folds["model"].unique(maintain_order=True):
        d = folds.filter(pl.col("model") == name).sort("fold")
        ax.plot(d["fold"], d["pr_auc"], marker="o", label=name)
    ax.set_xticks([1, 2, 3, 4], [f"{a[1][:4]}-{a[2][:4]}" for a in C.CV_FOLDS])
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("validation block")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_cv_folds.png", dpi=170)
    plt.close(fig)


def cfp_evidence_only() -> dict:
    """Recompute model_cfp_evidence.json from the frozen choice (development folds only, no test rows)."""
    choice = json.loads(CHOICE.read_text())
    dev, _ = load()
    spec = {c[0]: c for c in CANDIDATES}[choice["chosen_model"]]
    ys, ps = [], []
    for fold in C.CV_FOLDS:
        tr, va = fold_frames(dev, fold)
        m = make_model(spec[1], **spec[3]).fit(X(tr, cols_of(spec[2])), tr["y"].to_numpy())
        ys.append(va["y"].to_numpy()); ps.append(m.predict_proba(X(va, cols_of(spec[2])))[:, 1])
    ev = cfp_evidence(np.concatenate(ys), np.concatenate(ps), choice["cfn_per_row_k"])
    (C.TABLES / "model_cfp_evidence.json").write_text(json.dumps(ev, indent=1))
    return ev
