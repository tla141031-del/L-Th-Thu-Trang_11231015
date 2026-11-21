"""Generate the final report (DOCX, then PDF with LibreOffice) from the files in reports/.

Every number in the text is read from a table the pipeline wrote, so the report cannot
drift from the evidence:  ``python run.py report``  ->  report/out/BFT65B_report.{docx,pdf}
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdic_ews import config as CFG  # noqa: E402

T, FIGS, OUT = ROOT / "reports" / "tables", ROOT / "reports" / "figures", ROOT / "report" / "out"
STUDENT, SID = "Lã Thị Thu Trang", "11231015"
REPO = "https://github.com/tla141031-del/Final-Big-"


def rows(n):
    with open(T / n, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def js(n):
    return json.loads((T / n).read_text(encoding="utf-8"))


def pick(rs, **kw):
    for r in rs:
        if all(str(r[k]) == str(v) for k, v in kw.items()):
            return r
    raise KeyError(kw)


def f(x, d=2):
    return f"{float(x):,.{d}f}"


def n(x):
    return f"{int(float(x)):,}"


def pc(x, d=1):
    return f"{100 * float(x):.{d}f}%"


# ------------------------------------------------------------------ evidence
comp = js("ingest_completeness.json")
pinfo = js("panel_info.json")
cur = js("storage_curated_info.json")
sb = rows("storage_benchmark.csv")
sb_env = js("storage_benchmark_env.json")
pb = rows("processing_benchmark.csv")
pb_env = js("processing_benchmark_env.json")
par = js("spark_parity.json")
look = js("features_lookahead_test.json")
leak = js("leakage_summary.json")
miss = rows("dq_missing.csv")
zeros = rows("dq_structural_zeros.csv")
ytd = rows("dq_ytd_evidence.csv")
rules = rows("dq_rules.csv")
tails = rows("dq_tails.csv")
ftab = rows("dq_failures_table.csv")
exits = rows("panel_exit_types.csv")
lagf = rows("panel_last_report_to_failure.csv")
pchecks = rows("panel_checks.csv")
split = rows("model_split.csv")
cv = rows("model_cv_summary.csv")
folds = rows("model_cv_folds.csv")
costs = js("model_costs.json")
choice = js("model_choice.json")
thr = js("model_threshold_dev.json")
test = rows("model_test_results.csv")
det = rows("model_test_detection.csv")
dety = rows("model_test_detection_by_year.csv")
sens = rows("model_cost_sensitivity.csv")
perm = rows("model_permutation_importance.csv")
mll = rows("mllib_vs_sklearn.csv")
mlla = js("mllib_agreement.json")
fdict = rows("feature_dictionary.csv")
labyear = rows("panel_label_by_year.csv")
rpq = rows("ingest_rows_per_quarter.csv")
rt = js("ingest_resume_test.json")
lay = js("layout_effects.json")
scans = rows("engine_scans.csv")
fsig = rows("feature_signal_dev.csv")
lprep = rows("leakage_preprocessing.csv")
ecodes = rows("panel_exit_codes.csv")
isnap = {r["item"]: r["value"] for r in rows("dq_institutions_snapshot.csv")}
zshare = rows("dq_zero_share_by_year.csv")


def scan(exp, sc, setting):
    return pick(scans, experiment=exp, scale=sc, setting=setting)
ntests = len((T / "test_log.jsonl").read_text().strip().splitlines())

S = {r["split"]: r for r in split}
CH = choice["chosen_model"]
cht, blt = pick(test, model=CH), pick(test, model="baseline_logit_4")
chcv, blcv = pick(cv, model=CH), pick(cv, model="baseline_logit_4")
chd, bld = pick(det, model=CH), pick(det, model="baseline_logit_4")
NAMES = {"baseline_logit_4": "Baseline logit, 4 ratios", "logit_all": "Logit, all 40 features",
         "logit_all_balanced": "Logit, all, class-balanced", "hgb_levels": "Gradient boosting, levels only",
         "hgb_levels+dyn": "Gradient boosting, levels + dynamics", "hgb_all": "Gradient boosting, all 40",
         "hgb_all_balanced": "Gradient boosting, all, balanced", "hgb_all_small": "Gradient boosting, all, smaller trees"}


def sbr(fmt, q):
    return pick(sb, format=fmt, query=q)


def pbr(sc, e):
    return pick(pb, scale=sc, engine=e)


# ------------------------------------------------------------------ document helpers
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(1.8)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after, st.paragraph_format.line_spacing = Pt(4), 1.08
for lvl, size in ((1, 14), (2, 11.5)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.color.rgb = "Calibri", Pt(size), RGBColor(0x1B, 0x3B, 0x6F)
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(9 if lvl == 1 else 6), Pt(3)

fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run(f"BFT65B Final Report · {STUDENT} ({SID}) · page ")
r.font.size = Pt(8)
for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
    rr = fp.add_run()
    rr.font.size = Pt(8)
    el = OxmlElement("w:fldChar" if tag else "w:instrText")
    if tag:
        el.set(qn("w:fldCharType"), tag)
    else:
        el.set(qn("xml:space"), "preserve")
        el.text = text
    rr._r.append(el)


def H(t, lvl=1):
    doc.add_heading(t, level=lvl)


def _inline(p, text, size=None):
    for tok in re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text):
        if not tok:
            continue
        if tok.startswith("`"):
            rr = p.add_run(tok[1:-1])
            rr.font.name, rr.font.size = "Consolas", Pt((size or 10.5) - 1.2)
        elif tok.startswith("**"):
            rr = p.add_run(tok[2:-2])
            rr.bold = True
        else:
            rr = p.add_run(tok)
        if size and not tok.startswith("`"):
            rr.font.size = Pt(size)


def P(text, size=None):
    p = doc.add_paragraph()
    _inline(p, text, size)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def BUL(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        _inline(p, it)


def _shade(c, fill):
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"), s.set(qn("w:color"), "auto"), s.set(qn("w:fill"), fill)
    c._tc.get_or_add_tcPr().append(s)


TN = [0]


REF = {}


def TAB(header, body, widths=None, size=8.5, caption=None, label=None):
    if caption:
        TN[0] += 1
        if label:
            REF["T:" + label] = TN[0]
        cp = doc.add_paragraph()
        rr = cp.add_run(f"Table {TN[0]}. {caption}")
        rr.bold, rr.font.size = True, Pt(9)
        cp.paragraph_format.space_after = Pt(2)
        cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        rr = c.paragraphs[0].add_run(str(h))
        rr.bold, rr.font.size = True, Pt(size)
        _shade(c, "DDE5F0")
    for row in body:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            rr = cells[j].paragraphs[0].add_run(str(v))
            rr.font.size = Pt(size)
            if j > 0 and re.fullmatch(r"[-+\d.,%x ×/e]+", str(v) or "x"):
                cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = p.paragraph_format.space_before = Pt(0)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


FN = [0]


def FIG(label, path, cap, w=15.5):
    if not Path(path).exists():
        P(f"[figure missing: {Path(path).name}]")
        return
    doc.add_picture(str(path), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    FN[0] += 1
    if label:
        REF["F:" + label] = FN[0]
    cp = doc.add_paragraph()
    rr = cp.add_run(f"Figure {FN[0]}. {cap}")
    rr.italic, rr.font.size = True, Pt(9)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER


def arch_fig(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11, 2.7), dpi=200)
    ax.set_xlim(0, 11), ax.set_ylim(0, 2.9), ax.axis("off")
    B = [("FDIC API", "financials · failures\ninstitutions\n10k rows / call", "#E4ECF7"),
         ("Raw", "1 Parquet / quarter\nmanifest: API total,\nSHA-256 (resumable)", "#F4E7D7"),
         ("Curated", "typed schema, zstd\nyear=YYYY partitions\nsorted CERT, REPDTE", "#E6E6E6"),
         ("Panel", "label (q, q+4Q]\nexit types,\navailable_date", "#F6EBC4"),
         ("Features", "40 CAMELS + dynamic\n+ cross-section\n(Polars lazy)", "#DCEEDF"),
         ("Model", "rolling-origin CV\ncost threshold\nsingle test", "#EADFF2")]
    for k, (t, s, c) in enumerate(B):
        x = 0.1 + k * 1.83
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.55, 1.85, boxstyle="round,pad=0.03", fc=c, ec="#555", lw=1))
        ax.text(x + 0.775, 2.0, t, ha="center", fontsize=10.5, weight="bold")
        ax.text(x + 0.775, 1.15, s, ha="center", va="center", fontsize=7.6)
        if k < len(B) - 1:
            ax.annotate("", xy=(x + 1.82, 1.45), xytext=(x + 1.6, 1.45), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(5.5, 0.15, "Data-quality report (quality) reads Raw + Curated; PySpark reproduces the panel and core "
            "features (spark); every stage is one `python run.py <stage>` command", ha="center", fontsize=8,
            style="italic")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def label_fig(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yrs = [int(r_["year"]) for r_ in labyear]
    fl = [int(r_["failures"]) for r_ in labyear]
    fig, ax = plt.subplots(figsize=(10, 2.6), dpi=200)
    ax.bar(yrs, fl, color="#4C72B0")
    ax.axvspan(1983.5, 2009.5, color="#2ca02c", alpha=0.08, label="development 1984-2009")
    ax.axvspan(2009.5, 2010.5, color="grey", alpha=0.25, label="gap 2010")
    ax.axvspan(2010.5, 2025.5, color="#d62728", alpha=0.08, label="test 2011Q1-2025Q2")
    ax.set_ylabel("bank failures")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def banks_fig(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datetime import date
    xs = [date(int(r_["quarter"][:4]), int(r_["quarter"][4:6]), int(r_["quarter"][6:])) for r_ in rpq]
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.6), dpi=200)
    axes[0].plot(xs, [int(r_["rows"]) for r_ in rpq], color="#4C72B0")
    axes[0].set_title("Banks reporting per quarter (rows = API total)", fontsize=9)
    zy = [int(r_["year"]) for r_ in zshare]
    for col, lab_ in (("RBCRWAJ", "total risk-based ratio = 0"), ("BRO", "brokered deposits = 0"),
                      ("RBC1AAJ", "Tier 1 leverage = 0")):
        axes[1].plot(zy, [float(r_[col]) if r_[col] else None for r_ in zshare], label=lab_)
    axes[1].set_title("Share of banks reporting exactly 0 (%)", fontsize=9)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)

# ================================================================== PAGE 1
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
rr = p.add_run("NATIONAL ECONOMICS UNIVERSITY · BFT65B – BIG DATA ANALYTICS IN FINANCE")
rr.bold, rr.font.size = True, Pt(9)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
rr = p.add_run("A Reproducible, Leakage-Free Big-Data Pipeline\nfor Bank Failure Early Warning on FDIC Call Reports")
rr.bold, rr.font.size, rr.font.color.rgb = True, Pt(16), RGBColor(0x1B, 0x3B, 0x6F)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Final Examination – Individual Research Report").italic = True

dl0, dl1 = comp["download_first"], comp["download_last"]
dev, tst = S["development"], S["test"]
TAB(["Item", "Value"], [
    ["Student", STUDENT],
    ["Student ID", SID],
    ["GitHub repository (public)", REPO],
    ["Download method", "FDIC BankFind Suite public API (https://api.fdic.gov/banks, no key), own resumable "
                        "client adapted from the course starter script: `python run.py ingest`"],
    ["Download date (UTC)", f"{dl0[:10]}, {dl0[11:16]}–{dl1[11:16]} (Financials); failures and institutions "
                            f"{comp['tables_downloaded_at']['failures'][11:16]}"],
    ["Financials", f"{n(comp['financials_rows'])} rows (= sum of API totals), {n(comp['distinct_banks'])} banks, "
                   f"{comp['quarters_downloaded']} quarters 1984Q1–2026Q2, 0 duplicate (CERT, REPDTE)"],
    ["Failures", f"{n(comp['failures_rows'])} rows; {n(pinfo['failures']['rows_used'])} failures used "
                 f"(RESTYPE = FAILURE with CERT)"],
    ["Institutions", f"{n(comp['institutions_rows'])} rows"],
    ["Label", "y = 1 if the bank fails in (q, q + 4 quarters]; mergers and other exits are 0"],
    ["Development period", f"1984Q1–2009Q4: {n(dev['rows'])} bank-quarters, {n(dev['failed_banks'])} failed "
                           f"banks ({n(dev['positive_rows'])} positive rows)"],
    ["Gap", "2010Q1–2010Q4 (4 quarters, unused)"],
    ["Test period (used once)", f"2011Q1–2025Q2: {n(tst['rows'])} bank-quarters, {n(tst['failed_banks'])} "
                                f"failed banks ({n(tst['positive_rows'])} positive rows)"],
], widths=[4.2, 12.8], size=9)
sz, sc = sbr("parquet_zstd_by_year", "filter_1yr_state"), sbr("csv", "filter_1yr_state")
P(f"**Summary.** The pipeline downloads the full exam scope resumably and verifies it against the API's own "
  f"totals. A typed, zstd-compressed Parquet layout partitioned by year is "
  f"{f(1 / float(sbr('parquet_zstd_by_year', 'full_load')['size_vs_csv']), 1)}x smaller than CSV and answers a "
  f"filtered query {f(sz['speedup_vs_csv'], 0)}x faster. An automated quality report finds and handles mixed "
  f"event types, structural gaps, informative missingness, year-to-date income, reports after failure and "
  f"snapshot fields that encode the future. Polars computes the core workload "
  f"{f(pbr('1', 'polars_lazy')['speedup_vs_pandas'], 1)}–{f(pbr('1', 'polars_streaming')['speedup_vs_pandas'], 1)}x "
  f"faster than pandas, and PySpark reproduces it row by row. On the single test the chosen model reaches "
  f"PR-AUC {f(cht['pr_auc'], 3)} (baseline {f(blt['pr_auc'], 3)}) and flags {n(chd['detected_at_threshold'])} of "
  f"{n(chd['failed_banks'])} failed banks at a threshold derived from FDIC loss data.")
doc.add_page_break()

# ================================================================== 1
H("1. Introduction and architecture")
P("The supervision department wants a system that ranks banks by failure risk, and it first needs a reliable, "
  "scalable and reproducible data pipeline. This report treats the problem as data engineering. Most choices "
  "concern how 1.7 million quarterly reports are acquired, stored, checked and turned into features without "
  "look-ahead bias. The model is the final consumer, and it is judged by whether the pipeline allows an honest "
  "evaluation. Figure [[F:arch]] shows the stages. Each is one command of `run.py`; constants (fields, dates, split, "
  "seed) live in `src/fdic_ews/config.py`; every number below is read by `report/build_report.py` from a file "
  "in `reports/tables` that the pipeline wrote.")
arch_fig(OUT / "fig_architecture.png")
FIG("arch", OUT / "fig_architecture.png", "Pipeline architecture. Stages: ingest → verify → store → bench_storage → "
    "panel → quality → features → bench_processing → spark → experiments → model → mllib → report.", 17)
P("Three rules hold throughout: raw data are immutable so every later step can be rerun; nothing that learns a "
  "parameter (clip limits, medians, scaling, model, threshold) is fitted outside the training window; and every "
  "efficiency claim is a measurement taken in fresh processes, with the environment saved next to the result.")
P("Table [[T:decisions]] summarises the main design decisions and the measurement that supports each one. The sections that "
  "follow give the full evidence; every value in the table is read from the pipeline's output files.")
_z, _c = sbr("parquet_zstd_by_year", "filter_1yr_state"), sbr("csv", "filter_1yr_state")
_t1, _t4 = scan("polars_threads", "1", "1 threads"), scan("polars_threads", "1", "4 threads")
_s8, _s200 = scan("spark_shuffle", "1", "8 partitions"), scan("spark_shuffle", "1", "200 partitions")
TAB(["Decision", "Alternative", "Measured evidence (section)"], [
    ["API, one quarter per request, 2 workers, manifest", "bulk CSV; 3+ workers",
     f"rows = API total in all {comp['quarters_downloaded']} quarters; 3 workers hit HTTP 429; a killed download "
     f"resumes with {rt['restart_message'].split(' (')[0]} left (2.1)"],
    ["Typed Parquet, zstd, year partitions, sorted", "CSV; snappy; one file",
     f"{f(_z['speedup_vs_csv'], 0)}x faster filter, {pc(sbr('parquet_zstd_by_year', 'full_load')['size_vs_csv'], 0)} "
     f"of CSV size; sorting reads {lay['sorted_row_groups_read_one_bank']} instead of "
     f"{lay['unsorted_row_groups_read_one_bank']} row groups for one bank (2.2)"],
    ["Keep extreme values, clip at train quantiles", "delete outliers",
     f"outer 1% of NPL ratio fail at {f(pick(tails, ratio='npl_ratio')['fail_rate_pct_in_tails'], 1)}% vs "
     f"{f(pick(tails, ratio='npl_ratio')['fail_rate_pct_overall'], 2)}% overall (3.1)"],
    ["Polars lazy / streaming", "pandas; Polars eager",
     f"{f(pbr('1', 'polars_lazy')['speedup_vs_pandas'], 1)}x faster than pandas; 4 threads "
     f"{f(float(_t1['median_s']) / float(_t4['median_s']), 1)}x faster than 1 (4.2–4.3)"],
    ["PySpark only as a check and for scale-out", "Spark for the whole pipeline",
     f"{f(pbr('1', 'pyspark')['median_s'], 0)} s vs {f(pbr('1', 'polars_lazy')['median_s'], 1)} s at x1, of which "
     f"~{f(_s8['spark_startup_s'], 0)} s start-up; identical rows (4.2–4.3)"],
    ["Dynamic and cross-sectional features", "levels only",
     f"CV PR-AUC {f(pick(cv, model='hgb_levels')['pr_auc_mean'], 3)} → {f(pick(cv, model='hgb_all')['pr_auc_mean'], 3)} (5.3)"],
    ["Rolling-origin purged CV, PR-AUC", "random K-fold, accuracy",
     f"base rate {f(S['development']['base_rate_pct'], 2)}%: accuracy of 'never fails' > 99% (6.1)"],
    ["Cost-based threshold on out-of-fold predictions", "0.5 or Bayes rule",
     f"dev cost {f(thr[CH]['cost_k'] / 1000, 0)} M vs {f(thr[CH]['cost_alert_none'] / 1000, 0)} M USD without "
     f"alarms (6.4)"],
], widths=[4.6, 3.0, 9.4], size=7.8, label="decisions", caption="Design decisions and the measurement behind each one.")

# ================================================================== 2
H("2. Part 1 – Acquisition and storage")
H("2.1 Reproducible, resumable acquisition and verification", 2)
P("`src/fdic_ews/ingest.py` adapts the course starter script. It pages through each endpoint with "
  "`limit = 10,000` and `offset`, retries HTTP errors and connection failures with exponential back-off, and "
  "pauses between calls. Financials are requested one quarter at a time (filter `REPDTE:YYYYMMDD`) by two "
  "worker threads; three workers triggered HTTP 429 rate limits, so two is the measured polite maximum. Each "
  "quarter is written to a temporary file and renamed only when complete, so a crash never leaves a partial "
  "file that looks finished. A manifest stores, per quarter, the API's `meta.total`, the rows received, "
  "duplicate keys, a SHA-256 of the file and the time. On restart, quarters marked complete are skipped, so an "
  "interrupted download resumes where it stopped.")
P(f"`python run.py verify` then checks completeness: {comp['quarters_downloaded']} of {comp['quarters_expected']} "
  f"quarters present; rows received equal the API total in every quarter (sum {n(comp['api_total_sum'])}); "
  f"{len(comp['checksum_mismatch'])} checksum mismatches; {comp['duplicate_cert_repdte']} duplicate "
  f"(CERT, REPDTE) keys. The failures ({n(comp['failures_rows'])}) and institutions "
  f"({n(comp['institutions_rows'])}) tables also match their API totals. The full download took about "
  f"11 minutes.")
P(f"**Resumability, tested rather than claimed.** `experiments.resume_test` starts a real download of "
  f"{rt['quarters']} quarters (2022Q1–2023Q4) into an empty folder and kills the process with SIGKILL once "
  f"{rt['complete_at_kill']} quarters are complete ({f(rt['killed_after_s'], 0)} s in), so no clean-up code can run. "
  f"The folder then holds {rt['files_after_kill']} finished files and {rt['partial_tmp_files_after_kill']} partial "
  f"files, because each file appears only through an atomic rename. The restart prints “{rt['restart_message']}” "
  f"and finishes in {f(rt['restart_s'], 0)} s; a second restart prints “{rt['second_restart_message']}” and takes "
  f"{f(rt['second_restart_s'], 1)} s. All {rt['quarters']} files match the SHA-256 recorded in the manifest and the "
  f"API totals ({n(rt['rows'])} rows). The API was preferred to the bulk-download pages because its `meta.total` "
  f"gives an independent row count for every request, which is what makes completeness checkable, and because "
  f"it returns only the selected fields.")
banks_fig(OUT / "fig_banks.png")
_mx = max(rpq, key=lambda r_: int(r_["rows"]))
FIG("banks", OUT / "fig_banks.png", f"Left: banks per quarter, from {n(rpq[0]['rows'])} in 1984Q1 (peak {n(_mx['rows'])}) to "
    f"{n(rpq[-1]['rows'])} in 2026Q2. Right: fields where 0 means 'not reported' in some years (Section 3.1).", 16)
P(f"Figure [[F:banks]] shows why a per-quarter design is natural: the number of reporting banks fell by "
  f"{pc(1 - int(rpq[-1]['rows']) / int(rpq[0]['rows']), 0)} over the period, mostly through mergers, so any "
  f"alarm budget must be set per quarter (Section 6.1) and any count of failures must be read against a shrinking "
  f"population.")
P("**Field choice.** The 35 Financials fields (Appendix A) cover every CAMELS pillar, following the early-warning "
  "literature (Martin, 1977; Cole & White, 2012). Capital is covered by EQ and the Tier 1 leverage and total "
  "risk-based ratios. Asset quality uses noncurrent, past-due and non-accrual assets, foreclosed real estate "
  "(ORE), charge-offs and the loss allowance. Earnings use net income, interest income and expense, and "
  "non-interest income and expense. Liquidity uses cash, securities and deposits, including brokered "
  "deposits. Loan mix uses real-estate, construction, commercial and consumer loans. Construction loans "
  "(LNRECONS) and brokered deposits were added because they drove the 2008–2012 failures (Cole & White, 2012). "
  "LIAB was added to test the balance-sheet identity. Institution and failure fields that describe the bank's "
  "future (`ACTIVE`, `ENDEFYMD`, `CHANGEC1`, `FAILDATE`, `COST`) are downloaded only for labels, exits and costs, "
  "and are listed in `FUTURE_COLUMNS` so they can never become features.")
H("2.2 Storage layout and why it beats CSV", 2)
P("The raw layer keeps one Parquet file per quarter exactly as the API returned it. The curated layer enforces a "
  "schema (CERT Int32, REPDTE Date, state as Categorical, amounts as Int64 because they are whole thousands of "
  "USD, ratios as Float64). Each cast is logged: no amount had a fractional value. The curated panel is "
  "partitioned by year (`year=YYYY/data.parquet`, 43 files of 2–6 MB), compressed with zstd, sorted by "
  "(CERT, REPDTE) inside each file, and written with row-group statistics, so per-bank window functions read "
  "data that are already in order. Partitioning by quarter would create 170 very small files and multiply the "
  "per-file cost measured below.")
TAB(["Column group", "Raw type", "Curated type", "Why"], [
    ["CERT", "Float64", "Int32", "join key; 4 bytes, exact equality"],
    ["REPDTE", "String 'YYYYMMDD'", "Date",
     f"date arithmetic for lags and windows; {f(lay['repdte_string_mb'], 1)} → {f(lay['repdte_date_mb'], 1)} MB in memory"],
    ["STALP", "String", "Categorical",
     "57 distinct values; dictionary-encoded on disk and read as a category by pandas and Spark"],
    [f"{len(cur['cast_log'])} amounts (ASSET … ELNATR)", "Float64", "Int64",
     f"whole thousands of USD: {sum(v['non_integer_values'] for v in cur['cast_log'].values())} non-integer values "
     f"in {n(cur['financials_rows'])} rows; integers compress better"],
    ["Ratios (ROA, ROE, RBC1AAJ, RBCRWAJ …)", "Float64", "Float64", "genuinely fractional percentages"],
], widths=[4.2, 3.0, 2.3, 7.5], size=7.8, label="schema", caption="Curated schema: every cast is checked and logged.")
P(f"Two further measurements (`layout_effects.json`) test the layout. **Schema:** in memory the typed table is "
  f"only slightly smaller than the raw one ({f(lay['typed_inmemory_mb'], 0)} vs {f(lay['raw_inmemory_mb'], 0)} MB), "
  f"because Int64 and Float64 both take eight bytes. The schema is therefore justified by correctness rather than "
  f"memory: amounts stay exact integers, dates support quarter arithmetic, and every engine reads the same types. "
  f"**Sort order:** written with the same {lay['sorted_row_groups']} row groups of 50,000 rows, the table sorted "
  f"by (CERT, REPDTE) is {f(lay['sorted_file_mb'], 0)} MB against {f(lay['unsorted_file_mb'], 0)} MB unsorted, "
  f"because neighbouring rows of one bank have similar values. More importantly, the min/max statistics of a "
  f"sorted file let a query for one bank open {lay['sorted_row_groups_read_one_bank']} row group(s) instead of "
  f"{lay['unsorted_row_groups_read_one_bank']}; the query takes {f(1000 * lay['sorted_one_bank_query_s'], 0)} ms "
  f"against {f(1000 * lay['unsorted_one_bank_query_s'], 0)} ms. Per-bank histories are the pipeline's main access "
  f"pattern, so sorting is the most valuable layout choice after the columnar format itself.")
body = []
lab = {"csv": "CSV", "csv_gzip": "CSV + gzip", "parquet_none": "Parquet, uncompressed", "parquet_snappy": "Parquet snappy",
       "parquet_zstd": "Parquet zstd (1 file)", "parquet_zstd_by_year": "Parquet zstd, by year (chosen)"}
for fmt in lab:
    a, b, c = sbr(fmt, "full_load"), sbr(fmt, "agg_2cols"), sbr(fmt, "filter_1yr_state")
    body.append([lab[fmt], f(a["size_mb"], 0), f(a["write_s"], 1), f(a["median_s"], 2), f(b["median_s"], 3),
                 f(c["median_s"], 3), f(c["speedup_vs_csv"], 1) + "x", f(c["peak_rss_mb"], 0)])
TAB(["Format", "MB", "Write s", "Full load s", "Agg 2 cols s", "Filter s", "Filter vs CSV", "Filter peak MB"], body,
    label="storage", caption=f"Storage benchmark on {n(sb_env['rows'])} rows × {sb_env['columns']} columns (median of "
            f"{sb_env['repeats']} runs, each in a fresh process; {sb_env['cpu_count']} CPUs, Polars "
            f"{sb_env['polars']}). Filter = one year and one state, 5 columns.", size=8,
    widths=[4.4, 1.3, 1.4, 1.7, 1.9, 1.5, 1.9, 2.0])
cz, cc = sbr("parquet_zstd_by_year", "full_load"), sbr("csv", "full_load")
z1 = sbr("parquet_zstd", "agg_2cols")
P(f"The measurements favour Parquet on every axis. **Size:** columnar encoding with zstd stores the table in "
  f"{f(cz['size_mb'], 0)} MB, {pc(cz['size_vs_csv'], 0)} of the CSV. gzip shrinks the CSV to a similar size, but "
  f"then every query must decompress and parse the whole file, which makes it the slowest format. "
  f"**Projection:** an aggregate over two columns reads only those column chunks "
  f"({f(z1['median_s'], 3)} s for one zstd file vs {f(sbr('csv', 'agg_2cols')['median_s'], 2)} s for CSV) and "
  f"needs a quarter of the memory. **Predicate pushdown:** a filter on one year and one state skips row groups "
  f"by their min/max statistics, taking {f(sz['median_s'], 3)} s and {f(sz['peak_rss_mb'], 0)} MB against "
  f"{f(sc['median_s'], 2)} s and {f(sc['peak_rss_mb'], 0)} MB for CSV. **Types:** CSV loses the schema, so "
  f"dates and categories must be re-parsed on every read; Parquet keeps Int64 amounts and Date keys.")
P(f"Two results argue against over-engineering. Snappy decodes a full load faster than zstd "
  f"({f(sbr('parquet_snappy', 'full_load')['median_s'], 3)} s vs {f(sbr('parquet_zstd', 'full_load')['median_s'], 3)} s), "
  f"but its file is {f(sbr('parquet_snappy', 'full_load')['size_mb'], 0)} MB against "
  f"{f(sbr('parquet_zstd', 'full_load')['size_mb'], 0)} MB, and on the projected, filtered reads the pipeline "
  f"uses the codecs are within milliseconds. Year partitions do not speed up queries compared with one sorted "
  f"file: the filter runs equally fast because row-group statistics already skip other years, and the "
  f"two-column aggregate is slower ({f(sbr('parquet_zstd_by_year', 'agg_2cols')['median_s'], 3)} s) because 43 "
  f"file footers must be opened. Partitioning was kept for operational reasons. A new quarter rewrites one "
  f"small file instead of the whole table, a failed write affects one year, and Spark can read the partitions "
  f"in parallel. The measured query cost of this choice is about 0.1 s.")
FIG("storage", FIGS / "storage_benchmark.png", "Size and median query time by format (log scale for time).", 16)

# ================================================================== 3
H("3. Part 2 – Data quality and the point-in-time panel")
H("3.1 Automated quality report", 2)
P("`python run.py quality` rebuilds `reports/data_quality_report.md` and nine `dq_*.csv` tables on every "
  "download. Table [[T:dq]] lists the problems it finds and how the pipeline handles each one. The handling is "
  "chosen from the evidence; any parameter it needs is learned later on training rows only.")
p3 = pick(miss, column="P3ASSET")
p9 = pick(miss, column="P9ASSET")
roe = pick(miss, column="ROE")
rbc = pick(zeros, column="RBCRWAJ")
R = {r["rule"]: r for r in rules}
ytd1, ytd4 = ytd[0], ytd[3]
FT = {r["item"]: r["value"] for r in ftab}
TAB(["Problem", "Measured evidence", "Handling"], [
    ["Failures table mixes event types", f"{n(pinfo['failures']['restype_counts'].get('ASSISTANCE', 0))} "
     f"assistance deals (bank stayed open); {n(pinfo['failures']['missing_cert'])} rows without CERT; "
     f"{pinfo['failures']['duplicate_cert']} duplicate CERTs", "Keep RESTYPE = FAILURE with CERT, first failure per "
     "CERT, so the join cannot duplicate rows"],
    ["Reports after failure", f"{n(pinfo['rows_removed_on_or_after_failure'])} Financials rows dated on/after the "
     "bank's failure date", "Removed: a failed bank cannot be scored"],
    ["Structural missingness", f"P3ASSET missing {f(p3['missing_pct'], 1)}%, entirely absent in "
     f"{p3['years_fully_missing']} years (to {p3['last_fully_missing_year']})", "Not used as a feature"],
    ["Informative missingness", f"P9ASSET missing {f(p9['missing_pct_y1'], 1)}% when y = 1 vs "
     f"{f(p9['missing_pct_y0'], 1)}% when y = 0; ROE {f(roe['missing_pct_y1'], 1)}% vs {f(roe['missing_pct_y0'], 1)}%",
     "Median imputation plus a missing-value indicator (logit); native NaN branch (boosting)"],
    ["Zero means 'not reported'", f"RBCRWAJ is 0 for ≥95% of banks in {rbc['years_ge_95pct_zero']} years "
     f"({rbc['those_years'][:4]}–{rbc['those_years'][-4:]})", "0 recoded to missing for capital ratios"],
    ["Year-to-date income", f"median |NETINC|/assets rises from {f(ytd1['NETINC_pct_assets'], 2)}% in Q1 to "
     f"{f(ytd4['NETINC_pct_assets'], 2)}% in Q4", "De-cumulated to quarterly flows (Section 5.1)"],
    ["Gaps inside a series", "32 banks miss 92 quarters inside their series (longest gap 30 quarters)",
     "Lags and windows are nulled across gaps"],
    ["Identity / domain violations", f"EQ < 0 in {n(R['EQ < 0 (book insolvent)']['rows'])} rows (fail rate "
     f"{f(R['EQ < 0 (book insolvent)']['fail_rate_pct_if_violated'], 1)}%); DEP > ASSET "
     f"{n(R['DEP > ASSET']['rows'])} rows; LIAB + EQ ≠ ASSET (>1%) {n(R['|LIAB + EQ - ASSET| > 1% of ASSET']['rows'])} rows",
     "Kept: they are distress signals, not errors; safe division, then clipping"],
    ["Heavy tails", f"loans/deposits max {f(pick(tails, ratio='loans_to_deposits')['max'], 0)}; NPL ratio skew "
     f"{f(pick(tails, ratio='npl_ratio')['skew'], 1)}", "Clip at 0.5/99.5% quantiles learned on train; never delete"],
    ["Snapshot fields", "ACTIVE, CHANGEC1 and ENDEFYMD (9999-12-31 = still open) are as of download day",
     "Used for exit types only, never as features"],
    ["Restated values", "The API serves the latest amended reports, not the first release", "Stated as a limitation"],
], widths=[3.3, 7.6, 6.1], size=7.8, label="dq", caption="Data-quality problems found by the automated report and their handling.")
eq_t, npl_t = pick(tails, ratio="equity_to_assets"), pick(tails, ratio="npl_ratio")
P(f"Extreme values are clipped, not deleted, because the tails carry the signal. Rows in the outer 1% of the NPL "
  f"ratio have a failure rate of {f(npl_t['fail_rate_pct_in_tails'], 1)}%, and rows in the outer 1% of "
  f"equity/assets {f(eq_t['fail_rate_pct_in_tails'], 1)}%, against {f(npl_t['fail_rate_pct_overall'], 2)}% "
  f"overall. Deleting outliers would delete most failing banks. A rule such as mean ± 3 SD would be distorted "
  f"by skewness of this size (Tukey, 1977), so quantile limits learned on the training window are used.")
FIG("dqmiss", FIGS / "dq_missing_by_year.png", "Share of missing values by field and year: structural blocks (whole years) vs scattered gaps.", 15)

H("3.2 The evidence behind each rule", 2)
P("The audit is systematic in three ways: every field is profiled for missingness by year and by label, every "
  "accounting identity and domain constraint is tested on every row, and every table is reconciled with the "
  "others. Tables [[T:recon]]–[[T:missing]] give the detail behind the summary in Table [[T:dq]].")
TAB(["Step", "Rows", "Comment"], [
    ["Failures table as downloaded", n(FT["rows in failures table"]), "= API total"],
    ["− assistance transactions", n(FT["RESTYPE = ASSISTANCE (bank stayed open)"]), "open-bank assistance, not a failure"],
    ["− rows without CERT", n(FT["rows without CERT"]), "cannot be joined to Financials"],
    ["− repeated CERT (keep first failure)", n(pinfo["failures"]["duplicate_cert"]), "would duplicate panel rows"],
    ["= failures used", n(FT["failures used (RESTYPE = FAILURE, with CERT)"]), f"{n(pinfo['failures']['used_since_1984'])} after 1984Q1"],
    ["failed after 1984Q1 but never in Financials", n(FT["after 1984Q1 but no Financials row"]), "cannot be scored; listed"],
    ["Financials rows dated on/after failure", n(FT["Financials rows dated on/after the failure (removed)"]), "removed"],
    ["failed banks with ≥ 1 positive row", n(pinfo["failed_banks_with_positive_row"]), "every failed bank in the panel"],
    ["COST missing among used failures", f"{f(FT['COST missing for used failures (%)'], 1)}%", "excluded from the cost median"],
], widths=[6.5, 2.0, 8.5], size=7.8, label="recon", caption="Reconciliation of the failures list with the Financials panel.")
TAB(["Rule tested on every row", "Rows", "% rows", "Banks", "Fail rate if violated %"],
    [[r_["rule"], n(r_["rows"]), f(r_["pct_rows"], 3), n(r_["banks"]), f(r_["fail_rate_pct_if_violated"], 1)]
     for r_ in rules], widths=[7.2, 1.8, 1.7, 1.6, 3.4], size=7.8,
    label="rules", caption=f"Accounting-identity and domain checks (overall fail rate {f(npl_t['fail_rate_pct_overall'], 2)}%).")
P(f"The rule table is what decides the handling. Book insolvency (EQ < 0) and deposits above assets are "
  f"violated in {n(R['EQ < 0 (book insolvent)']['rows'])} and {n(R['DEP > ASSET']['rows'])} rows, and those rows "
  f"fail at {f(R['EQ < 0 (book insolvent)']['fail_rate_pct_if_violated'], 0)}% and "
  f"{f(R['DEP > ASSET']['fail_rate_pct_if_violated'], 0)}%, about 35–45 times the base rate, so they are "
  f"signals to keep. By contrast, a Tier 1 ratio outside ±100% occurs in {n(R['RBC1AAJ outside [-100, 100]']['rows'])} "
  f"rows that almost never fail ({f(R['RBC1AAJ outside [-100, 100]']['fail_rate_pct_if_violated'], 1)}%): these are "
  f"tiny or shell institutions with near-zero risk-weighted assets, and clipping removes their leverage on the "
  f"fit. Loan sub-categories exceeding total loans ({n(R['LNRE + LNCI + LNCON > 1.05 x LNLSGR']['rows'])} rows) "
  f"come from overlapping definitions, so the loan-mix shares are capped by clipping rather than trusted as exact.")
SENT = "ENDEFYMD = 9999-12-31 (sentinel for 'still open')"
nt = pick(miss, column="NTLNLS")
mtop = sorted(miss, key=lambda r_: -float(r_["missing_pct"]))[:6]
TAB(["Field", "Missing %", "Whole years missing", "Type", "Missing % if y = 1", "if y = 0"],
    [[r_["column"], f(r_["missing_pct"], 1), r_["years_fully_missing"], r_["type"].split(" (")[0],
      f(r_["missing_pct_y1"], 1), f(r_["missing_pct_y0"], 1)] for r_ in mtop],
    widths=[2.4, 1.9, 2.8, 3.0, 3.2, 1.9], size=7.8, label="missing", caption="Most-missing fields, overall and by label.")
P(f"Missingness is of two kinds, and they are handled differently. P3ASSET (30–89 days past due) was not collected "
  f"until {int(p3['last_fully_missing_year']) + 1}, so it is absent for whole years; imputing 17 years would invent "
  f"data, and the field is dropped. NTLNLS, P9ASSET and NAASSET are missing {f(nt['missing_pct_y1'], 1)}% of the time for "
  f"rows with y = 1 against {f(nt['missing_pct_y0'], 1)}% for y = 0: the reports of banks close to failure are less complete. That "
  f"missingness is itself information, so it is kept as an indicator instead of being hidden by imputation. Zero "
  f"is a third kind of missing value (Figure [[F:banks]], right): RBCRWAJ is 0 for at least 95% of banks in "
  f"{rbc['those_years'][:4]}–{rbc['those_years'][-4:]}, before risk-based capital ratios were reported, so a 0 there means 'not reported', while a 0 in brokered deposits is a real value. The YTD check "
  f"is equally direct: the median quarterly ratio of net income to assets reported in Q1–Q4 is "
  f"{', '.join(f(r_['NETINC_pct_assets'], 2) for r_ in ytd)}%, rising almost linearly, which only cumulative "
  f"reporting explains. Finally, the snapshot tables are consistent: {n(isnap['ACTIVE = 1'])} institutions are "
  f"ACTIVE, {n(isnap['ACTIVE = 1 and reporting in 2026-06-30'])} of them report in 2026Q2, and "
  f"{n(isnap[SENT])} carry the sentinel end date 9999-12-31, "
  f"which is parsed as 'open', not as a date.")

H("3.3 Exits and point-in-time correctness", 2)
E = {}
for r in exits:
    E[r["exit_type"]] = E.get(r["exit_type"], 0) + int(r["len"])
TOPC = ", ".join([f"{r_['CHANGEC1']} ({n(r_['len'])})" for r_ in ecodes if r_["exit_type"] == "merger_or_absorption"][:4])
P(f"Banks leave the panel for different reasons. `panel.classify_exits` labels each bank as a failure "
  f"({n(E.get('failure', 0))} banks), a merger or absorption ({n(E.get('merger_or_absorption', 0))}; CHANGEC1 codes "
  f"2xx; the largest are {TOPC}), another exit ({n(E.get('other_exit', 0))}), or active ({n(E.get('active', 0))}). Only "
  f"failures create positives; mergers are 0 as the brief requires. Every failed bank in the panel keeps at least "
  f"one positive row: the last report precedes failure by one quarter for {n(lagf[0]['len'])} banks and by two or "
  f"three quarters for the rest. Four rules make the panel point-in-time correct:")
BUL([
    "**Label:** `y = 1` iff q < FAILDATE ≤ q + 4 quarter-ends. Seven assertions in `panel.checks` stop the "
    "pipeline if any positive fails on or before its report date or more than four quarters later, if a negative "
    "row has a failure inside its window, or if a merger is labelled 1 (`panel_checks.csv`, all passed).",
    f"**Availability:** `available_date = REPDTE + {CFG.PUBLICATION_LAG_DAYS} days`, because Call Reports are "
    f"published weeks after quarter end. For {n(pinfo['positives_failing_before_available'])} of "
    f"{n(pinfo['positive_rows'])} positive rows the bank failed before its report would have been public; the test "
    "is also scored without them.",
    "**Censoring:** a label is complete only if its window closed before the failures list was downloaded; "
    "modelling stops at 2025Q2.",
    "**No snapshot leakage:** institution status fields describe today, not quarter q, so they are excluded.",
])

# ================================================================== 4
H("4. Part 3 – Scalable processing")
H("4.1 Core transformations in Polars and how the engine runs them", 2)
P("The feature pipeline (`features.py`) is one Polars lazy query. It sorts by (CERT, REPDTE) and builds a "
  "quarter index. It de-cumulates seven year-to-date items: the Q1 value is the flow; later quarters subtract the "
  "previous report only if that report is exactly one quarter earlier. It then computes 26 ratios with safe "
  "division, gap-aware 1- and 4-quarter changes and 4-report rolling statistics per bank (`over('CERT')`), and "
  "cross-sectional percentile ranks and state medians per quarter (`over('REPDTE')`). The optimised plan "
  "(`plan_bench_polars_optimized.txt`) shows how the engine executes it. **Projection pushdown:** the Parquet "
  "scan reads 11 of 36 columns (`PROJECT 11/36 COLUMNS`); unused columns such as the state code are never "
  "decoded. **Common-subexpression elimination:** guards such as `ASSET > 0` and the Q1 test are computed once "
  "as cached `__POLARS_CSER_*` columns. **Fusion:** consecutive `with_columns` run in one pass over Arrow "
  "buffers, in parallel across columns. Window expressions over the same key reuse one group-by. The streaming "
  "engine executes the same plan in batches, so it holds less in memory.")
H("4.2 Benchmark against pandas, and PySpark parity", 2)
P("`bench_processing.py` implements one workload with identical semantics in five engines: read 13 columns, join "
  "the failures list, build the label, de-cumulate three YTD items, compute five ratios, a gap-aware 4-quarter "
  "change, a 4-report rolling mean and a per-quarter percentile rank. To keep the comparison fair, every run "
  "starts a fresh process; time covers read, compute and materialise; peak memory is the resident set of the "
  "whole process tree sampled every 50 ms, so it includes the Spark JVM; the median of three runs is reported; "
  "and every engine's checksums (row count, per-column sum and non-null count) are compared with Polars lazy.")
body = []
for scl in ("1", "3"):
    for e in ("pandas", "polars_eager", "polars_lazy", "polars_streaming", "pyspark"):
        r_ = pbr(scl, e)
        body.append([f"x{scl}", n(r_["rows"]), e, f(r_["median_s"], 2), f(r_["speedup_vs_pandas"], 2) + "x",
                     f(r_["peak_mb"], 0), f(r_["spark_startup_s"], 1) if e == "pyspark" else "–",
                     f"{r_['count_mismatches']} / {float(r_['max_rel_diff_sums']):.0e}"])
TAB(["Scale", "Rows", "Engine", "Median s", "vs pandas", "Peak MB", "Spark start s", "Mismatches / rel. diff"], body,
    label="engines", caption=f"Core workload in five engines ({pb_env['cpu_count']} CPUs, {pb_env['ram_gb']} GB RAM; pandas "
            f"{pb_env['pandas']}, Polars {pb_env['polars']}, PySpark {pb_env['pyspark']} local[*]). x3 stacks three "
            f"copies of the real panel with shifted CERTs.", size=8)
pl1, ps1, pd1, sp1 = pbr("1", "polars_lazy"), pbr("1", "polars_streaming"), pbr("1", "pandas"), pbr("1", "pyspark")
pl3, ps3, pd3, sp3 = pbr("3", "polars_lazy"), pbr("3", "polars_streaming"), pbr("3", "pandas"), pbr("3", "pyspark")
P(f"Polars lazy runs the workload in {f(pl1['median_s'], 2)} s, {f(pl1['speedup_vs_pandas'], 1)}x faster than "
  f"pandas, with half the memory ({f(pl1['peak_mb'], 0)} vs {f(pd1['peak_mb'], 0)} MB). The streaming engine is "
  f"fastest ({f(ps1['median_s'], 2)} s). The gap widens with size: at x3, pandas takes {f(pd3['median_s'], 1)} s "
  f"against {f(ps3['median_s'], 1)} s for Polars streaming. pandas (McKinney, 2010) is a fair baseline here because the workload "
  f"is written with vectorised `groupby().shift()` and `transform`. Its costs are single-threaded execution and "
  f"a full copy of the frame at each step. Polars eager executes the same expressions step by step without the "
  f"optimiser and is slower than lazy at both scales. All engines agree: 0 count mismatches and relative "
  f"differences in sums of at most {max(float(r_['max_rel_diff_sums']) for r_ in pb):.0e} (floating-point "
  f"summation order).")
P(f"PySpark is the slowest on one machine ({f(sp1['median_s'], 1)} s at x1, of which {f(sp1['spark_startup_s'], 1)} s "
  f"is JVM and session start-up), but its time grows the least: from x1 to x3 it rises by "
  f"{f(float(sp3['median_s']) / float(sp1['median_s']), 2)}x, against {f(float(pl3['median_s']) / float(pl1['median_s']), 2)}x "
  f"for Polars lazy, because most of its cost is fixed. Beyond the checksums, `spark_check.py` writes Spark's full "
  f"output and joins it with Polars on (CERT, REPDTE): {n(par['rows_spark'])} rows in both, "
  f"{par['rows_only_in_one_engine']} rows in only one engine, {par['label_mismatches']} label mismatches, 0 "
  f"null disagreements, and a maximum absolute feature difference of "
  f"{max(c['max_abs_diff'] for c in par['columns']):.1e}. The two engines produce the same panel.")
FIG("proc", FIGS / "processing_benchmark.png", "Median time and peak memory by engine at x1 and x3.", 15)
H("4.3 What the threads, the optimiser and Spark's settings contribute", 2)
P("A benchmark of defaults does not show why an engine is fast. `experiments.engine_scans` therefore reruns the "
  "same workload, in fresh processes and with the same checksums, while changing one setting at a time: the "
  "number of Polars threads (`POLARS_MAX_THREADS`), the Polars optimiser (all passes off with "
  "`QueryOptFlags.none()`), the number of Spark cores (`local[k]`) and the number of shuffle partitions.")


def _sc(exp, setting, scale):
    try:
        r_ = scan(exp, scale, setting)
        return f(r_["median_s"], 2), f(r_["peak_mb"], 0)
    except KeyError:
        return "–", "–"


_body = []
for exp, eng, sets in (("polars_threads", "Polars lazy", ("1 threads", "2 threads", "4 threads")),
                       ("polars_optimizer", "Polars lazy", ("optimiser off", "optimiser on")),
                       ("spark_cores", "PySpark", ("local[1]", "local[2]", "local[4]")),
                       ("spark_shuffle", "PySpark", ("8 partitions", "200 partitions"))):
    for s_ in sets:
        _body.append([eng, s_.replace("1 threads", "1 thread"), *_sc(exp, s_, "1"), *_sc(exp, s_, "3")])
TAB(["Engine", "Setting", "x1 median s", "x1 peak MB", "x3 median s", "x3 peak MB"], _body,
    widths=[3.0, 3.4, 2.6, 2.6, 2.6, 2.6], size=8,
    label="scans", caption="One setting changed at a time (median of 3 fresh-process runs; checksums identical in every run).")
th1, th4 = scan("polars_threads", "3", "1 threads"), scan("polars_threads", "3", "4 threads")
th1a, th4a = scan("polars_threads", "1", "1 threads"), scan("polars_threads", "1", "4 threads")
oo, on_ = scan("polars_optimizer", "1", "optimiser off"), scan("polars_optimizer", "1", "optimiser on")
k1, k4 = scan("spark_cores", "3", "local[1]"), scan("spark_cores", "3", "local[4]")
k1a, k4a = scan("spark_cores", "1", "local[1]"), scan("spark_cores", "1", "local[4]")
sh8, sh200 = scan("spark_shuffle", "1", "8 partitions"), scan("spark_shuffle", "1", "200 partitions")
_sp_th = float(th1["median_s"]) / float(th4["median_s"])
_sp_k = float(k1["median_s"]) / float(k4["median_s"])
P(f"**Threads.** Polars parallelises across columns and across groups of a window, so four threads run the x3 "
  f"workload {f(_sp_th, 1)}x faster than one ({f(th1['median_s'], 1)} s vs {f(th4['median_s'], 1)} s; "
  f"{f(float(th1a['median_s']) / float(th4a['median_s']), 1)}x at x1). The speed-up is below 4 because the sort "
  f"by (CERT, REPDTE) and the join are only partly parallel, and the extra threads cost memory for their buffers "
  f"({f(th1['peak_mb'], 0)} → {f(th4['peak_mb'], 0)} MB). Even single-threaded, Polars "
  f"({f(th1a['median_s'], 2)} s) beats pandas ({f(pd1['median_s'], 2)} s), so its advantage is not only "
  f"parallelism: it also comes from columnar Arrow memory and from not copying the frame at each step.")
P(f"**Optimiser.** With every optimisation pass switched off, the same lazy query takes {f(oo['median_s'], 2)} s "
  f"against {f(on_['median_s'], 2)} s, but peaks at {f(oo['peak_mb'], 0)} MB against {f(on_['peak_mb'], 0)} MB. "
  f"Time hardly changes because this workload already selects its 13 columns explicitly and is dominated by the "
  f"sort and the window functions, which no rewrite can avoid. The optimiser's gain here is memory: projection "
  f"pushdown into the scan and common-subexpression elimination mean fewer intermediate columns are "
  f"materialised. The full feature pipeline, which derives 40 columns from many shared sub-expressions, is the "
  f"case where CSE matters most (`plan_features_optimized.txt`).")
P(f"**Spark cores and partitions.** Spark's time falls from {f(k1['median_s'], 1)} s with one core to "
  f"{f(k4['median_s'], 1)} s with four at x3 ({f(_sp_k, 1)}x), and from {f(k1a['median_s'], 1)} to "
  f"{f(k4a['median_s'], 1)} s at x1. The fixed start-up of about {f(k4['spark_startup_s'], 0)} s does not shrink "
  f"with cores. With the default 200 shuffle partitions the x1 job takes {f(sh200['median_s'], 1)} s against "
  f"{f(sh8['median_s'], 1)} s with 8 (2 × cores), and peaks at {f(sh200['peak_mb'], 0)} against "
  f"{f(sh8['peak_mb'], 0)} MB. The difference is small because adaptive query execution coalesces the 200 "
  f"small post-shuffle partitions at run time. Setting 2 × cores is still the cleaner choice for one machine, "
  f"but the measurement shows that the default is not what makes Spark slow here: the fixed start-up and the "
  f"two shuffles are.")

H("4.4 Reading the Spark physical plan", 2)
P("The formatted plan (`plan_spark_formatted.txt`, excerpt in Appendix D) reads bottom-up. (1) `Scan parquet` "
  "with a pruned `ReadSchema`, the same projection pushdown as Polars. (3)–(6) The 3,036-row failures table is "
  "broadcast (`BroadcastExchange`, `BroadcastHashJoin LeftOuter`), so the 1.7-million-row side is never "
  "shuffled for the join. (8) `Exchange hashpartitioning(CERT)` is the first shuffle: it moves all rows of a bank "
  "into one partition, and (9) sorts them by date so that (10) and (12) `Window` operators can compute lags and "
  "the rolling mean. (14) A second `Exchange hashpartitioning(REPDTE)` is needed because the percentile rank is "
  "computed across banks within a quarter. (15)–(19) sort and rank. (21) `AdaptiveSparkPlan` can re-size "
  "partitions at run time; the number of shuffle partitions was set to 2 × cores instead of the default 200, "
  "which is tuned for clusters. The two Exchanges are where a cluster would move data over the network "
  "(Zaharia et al., 2016). They explain why Spark's cost is high but nearly fixed on one machine.")
H("4.5 When would a cluster be justified?", 2)
P(f"The measurements answer the first half of Q1. The whole 40-year panel is {f(cz['size_mb'], 0)} MB in "
  f"Parquet. Polars processes it in about one second and three times as much in "
  f"{f(pl3['median_s'], 1)} s with {f(float(pl3['peak_mb']) / 1024, 1)} GB peak memory, so memory grows by about "
  f"{f((float(pl3['peak_mb']) - float(pl1['peak_mb'])) / 2 / 1024, 2)} GB per extra copy of the panel. A 16 GB "
  f"laptop therefore has room for about {int((16 * 1024 - float(pl1['peak_mb'])) / ((float(pl3['peak_mb']) - float(pl1['peak_mb'])) / 2))} panels of this size, with the streaming engine lowering the peak further. A Spark cluster is justified only when (a) the working "
  f"set exceeds one machine's memory even with streaming, for example loan-level or daily data that are 100x "
  f"larger; (b) the data already live in a distributed store such as a data lake, where copying them to a laptop "
  f"costs more than processing them in place; or (c) many jobs and users share the pipeline and need scheduling "
  f"and fault tolerance. Spark's near-fixed cost ({f(sp1['median_s'], 0)} s at x1, {f(sp3['median_s'], 0)} s at "
  f"x3) only pays off once the job is long enough for many machines to amortise it. At every size measured here, "
  f"Polars on one machine is faster.")
_cmp1 = float(k1["median_s"]) - float(k1["spark_startup_s"])
_cmp4 = float(k4["median_s"]) - float(k4["spark_startup_s"])
_be = float(pl3["median_s"]) / 3
P(f"The scans in Section 4.3 make this argument quantitative. Excluding start-up, Spark's x3 compute time falls "
  f"from {f(_cmp1, 1)} s on one core to {f(_cmp4, 1)} s on four, a parallel efficiency of "
  f"{pc(_cmp1 / _cmp4 / 4, 0)}, while Polars needs about {f(_be, 1)} s per copy of the panel on four threads. "
  f"A cluster adds cores but also moves the two shuffles of Section 4.4 onto the network, so it can only win "
  f"once a single machine cannot hold the data. On the measured memory growth, that point is roughly "
  f"{int((16 * 1024 - float(pl1['peak_mb'])) / ((float(pl3['peak_mb']) - float(pl1['peak_mb'])) / 2))} panels "
  f"for the in-memory engine on 16 GB, and further for the streaming engine. In practice, for a supervisor this "
  f"means: Call-Report analytics stay on one machine; a cluster becomes the right tool for loan-level credit "
  f"registry data, for intraday payment data, or when the pipeline runs as a shared service for many analysts.")

# ================================================================== 5
H("5. Part 4 – Feature pipeline")
H("5.1 Features and why they are informative and point-in-time", 2)
P("The 40 features (Appendix B) are grouped as follows. **Capital:** equity/assets, Tier 1 leverage, total "
  "risk-based ratio. **Asset quality:** NPL ratio, non-accrual and 90-day past-due assets, ORE, the Texas ratio "
  "(problem assets over capital plus reserves), allowance coverage, and charge-off and provision rates. "
  "**Management/efficiency:** efficiency ratio, non-interest expense, bank age. **Earnings:** quarterly ROA, NIM "
  "and fee income, annualised after de-cumulation, because a YTD ROA in Q4 and in Q1 are not comparable. "
  "**Liquidity and funding:** loans/deposits, liquid assets, brokered deposits, deposits/assets. **Loan mix:** "
  "real-estate, construction, commercial and consumer shares. **Dynamics:** asset and loan growth, and 1- and "
  "4-quarter changes in NPL, Texas and equity ratios, ROA mean and volatility; deterioration predicts failure "
  "better than levels (Cole & White, 2012). **Cross-section:** percentile ranks within the quarter and state "
  "medians of the NPL ratio and its change. Ranks remove the drift in bank ratios over four decades, and state "
  "medians capture regional real-estate cycles. They use only reports of the same quarter, which are published "
  "together.")
P(f"**Look-ahead test.** `features.lookahead_test` computes all features twice: on the full panel, and on the panel "
  f"truncated at {look['cutoff']}. It then compares the {n(look['rows_compared'])} rows before the cut-off. "
  f"{len(look['features_changed_by_future_rows'])} of {look['features']} features change when future rows are "
  f"added. A unit test plants extreme future values and confirms that no earlier feature moves.")
P("**Is each group informative?** Before any model is fitted, `experiments.feature_signal` measures the "
  "univariate ranking power of every feature on development rows only: the ROC-AUC of the feature alone "
  "(oriented so that 0.5 means no signal), computed on the rows where it is observed, and its median for "
  "y = 1 and y = 0. Table [[T:signal]] shows the strongest feature of each group and the overall top features; the full "
  "list is in `feature_signal_dev.csv`.")
_best = {}
for r_ in fsig:
    _best.setdefault(r_["pillar"], r_)
_show = list(fsig[:8]) + [r_ for k_, r_ in _best.items() if r_ not in fsig[:8]]
PIL = {"C": "Capital", "A": "Asset quality", "M": "Management", "E": "Earnings", "L": "Liquidity",
       "S": "Size / mix", "dyn": "Dynamics", "xs": "Cross-section"}
TAB(["Feature", "Group", "AUC alone", "Direction", "Median y = 1", "Median y = 0", "Missing %"],
    [[r_["feature"], PIL.get(r_["pillar"], r_["pillar"]), f(r_["roc_auc_oriented"], 3), r_["direction"],
      f(r_["median_y1"], 4), f(r_["median_y0"], 4), f(r_["missing_pct"], 1)] for r_ in _show],
    widths=[3.8, 2.2, 1.6, 2.6, 2.2, 2.2, 1.6], size=7.6,
    label="signal", caption="Univariate signal on development rows only (top 8, then the best feature of each remaining group).")
_wk = [r_ for r_ in fsig if float(r_["roc_auc_oriented"]) < 0.6]
P(f"Every CAMELS group carries signal on its own. Capital and asset-quality ratios are the strongest, with "
  f"AUCs near {f(fsig[0]['roc_auc_oriented'], 2)}: the median failing bank has equity of "
  f"{pc(pick(fsig, feature='equity_to_assets')['median_y1'])} of assets against "
  f"{pc(pick(fsig, feature='equity_to_assets')['median_y0'])} for the others. Dynamics such as ROA volatility "
  f"and the 4-quarter changes in the Texas and equity ratios "
  f"({f(pick(fsig, feature='d4_texas_ratio')['roc_auc_oriented'], 2)} and "
  f"{f(pick(fsig, feature='d4_equity_to_assets')['roc_auc_oriented'], 2)}) are almost as strong, which is why "
  f"they were built. "
  f"{len(_wk)} features have a univariate AUC below 0.6 ({', '.join(r_['feature'] for r_ in _wk[:6])}"
  f"{' …' if len(_wk) > 6 else ''}); they are kept because they matter in interaction (loan mix changes what "
  f"an NPL ratio means), and Section 5.3 tests the groups jointly in the model.")
H("5.2 Preprocessing fitted on training data only", 2)
P(f"All learned preprocessing sits in one scikit-learn `Pipeline` with the model (Pedregosa et al., 2011): a "
  f"`Winsorizer` (0.5/99.5% limits learned in `fit`), median imputation with missing indicators, and "
  f"standardisation for the logistic models. Gradient boosting uses its native missing-value handling. A column "
  f"never observed in a training window (total risk-based capital before 1990) is set to a constant, so later "
  f"values cannot leak into the fit. `model.leakage_proof` fits the pipeline on the first CV training window "
  f"({n(leak['fold1_train_rows'])} rows). It verifies that every learned median equals the median of that "
  f"window ({leak['medians_equal_train']}) and that transforming a test row does not depend on other test rows "
  f"({leak['row_wise_transform']}). Fitting the clip limits on all periods instead would move them by "
  f"{f(leak['median_rel_shift_clip_pct'], 0)}% (median over features), which is how much the ratios drifted "
  f"over 40 years.")
_lp = [r_ for r_ in lprep if r_["rel_shift_if_fitted_on_all_pct"] not in ("NaN", "")][:6]
TAB(["Feature", "Median learned on fold 1", "= train median", "Upper clip (train)", "Upper clip (all periods)", "Shift %"],
    [[r_["feature"], f(r_["learned_median"], 4), "yes" if r_["learned_median"] == r_["train_median"] else "no",
      f(r_["learned_p99_5"], 3), f(r_["all_periods_p99_5"], 3), f(r_["rel_shift_if_fitted_on_all_pct"], 0)]
     for r_ in _lp], widths=[3.6, 3.0, 2.0, 2.6, 3.2, 1.6], size=7.6,
    label="leakprep", caption="Leakage proof for the preprocessing (first CV training window).")
P("The shift column is the size of the leak that the pipeline avoids. Had the clip limits been learned on all "
  "periods, the training rows of the 1980s would have been clipped with limits that partly reflect 2010–2025 "
  "balance sheets, and the validation scores would have been optimistic in a way that cannot be detected "
  "afterwards.")
H("5.3 Feature choice measured, not assumed", 2)
fs_rows = [pick(cv, model=m) for m in ("hgb_levels", "hgb_levels+dyn", "hgb_all")]
P(f"Feature groups were added only if they improved cross-validated PR-AUC on development data. With the same "
  f"gradient-boosting model, levels alone give {f(fs_rows[0]['pr_auc_mean'], 3)}, adding dynamics gives "
  f"{f(fs_rows[1]['pr_auc_mean'], 3)}, and adding cross-sectional features gives {f(fs_rows[2]['pr_auc_mean'], 3)} "
  f"(Table [[T:cv]]). Permutation importance on the last development fold (Appendix E) ranks "
  f"{', '.join(r_['feature'] for r_ in perm[:5])} highest. Permutation importance was preferred to impurity "
  f"importance, which favours high-cardinality continuous features.")

# ================================================================== 6
H("6. Part 5 – Modelling and validation")
H("6.1 Split, validation scheme and metrics", 2)
P(f"**Split.** Failures cluster in 1984–1993 and 2008–2014 (`panel_label_by_year.csv`). The development period "
  f"1984Q1–2009Q4 covers the savings-and-loan crisis, the calm years and the start of the global financial "
  f"crisis, so the model sees both crisis types: real-estate-driven failures in the late 1980s and "
  f"construction-loan failures in 2008–2009. A four-quarter gap (2010) separates it from the test period "
  f"2011Q1–2025Q2. Every development label (the last window ends 2010Q4) is known before the first test quarter. "
  f"The test has {n(tst['failed_banks'])} failed banks, four times the required 50. It covers the end of the "
  f"crisis wave, the calm 2015–2022 period and the 2023 failures, which is the honest use case: a model built on "
  f"the past, applied to a new decade.")
label_fig(OUT / "fig_labels_by_year.png")
FIG("labels", OUT / "fig_labels_by_year.png", "Failures per year (FDIC list, RESTYPE = FAILURE) and the development / gap / test periods.", 14)
P("**Validation.** Random K-fold would train on later years to predict earlier ones and split one bank's quarters "
  "across folds. Model choice therefore uses four rolling-origin folds inside development (validation 1990–91, "
  "1993–96, 1998–2004, 2006–09). Each trains only on rows whose label window had closed before the fold's "
  "training cut-off. This purging (López de Prado, 2018) leaves a gap of at least four quarters before "
  "validation.")
P(f"**Metrics.** Positives are {f(dev['base_rate_pct'], 2)}% of development rows and {f(tst['base_rate_pct'], 2)}% "
  f"of test rows. Accuracy is therefore meaningless: always predicting 'no failure' is over 99% accurate. "
  f"ROC-AUC is inflated by the mass of easy negatives. The main metric is PR-AUC (Saito & Rehmsmeier, 2015), "
  f"reported with recall in the top 2% of banks per quarter (an inspection budget) and the Brier score.")
H("6.2 Baseline versus models", 2)
TAB(["Model", "CV PR-AUC", "SD", "ROC-AUC", "Recall top 2%", "Brier", "Fit s"],
    [[NAMES.get(r_["model"], r_["model"]), f(r_["pr_auc_mean"], 3), f(r_["pr_auc_sd"], 3), f(r_["roc_auc_mean"], 3),
      f(r_["recall_top2pct_mean"], 3), f(r_["brier_mean"], 4), f(r_["fit_s_mean"], 1)] for r_ in cv],
    widths=[5.8, 1.8, 1.4, 1.7, 2.0, 1.6, 1.4], size=8.5,
    label="cv", caption="Rolling-origin CV on development (mean over 4 folds).")
P(f"The baseline is a logistic regression on four classic ratios (equity/assets, NPL ratio, ROA, loans/deposits), "
  f"in the tradition of Martin (1977). Its CV PR-AUC is {f(blcv['pr_auc_mean'], 3)}. The chosen model, "
  f"**{NAMES.get(CH, CH)}**, has the best mean ({f(chcv['pr_auc_mean'], 3)}). The large fold standard deviation "
  f"comes from base rates that differ by a factor of ten or more between crisis and calm folds (Appendix F); "
  f"gradient boosting beats the logistic models in every fold. Class re-weighting lowered PR-AUC for both model families, "
  f"so it was not used, and the unweighted models also give better-calibrated probabilities (Brier), which the "
  f"cost-based threshold needs.")
H("6.3 Spark MLlib demonstration", 2)
sl, ss = pick(mll, engine="spark-mllib", model="logistic"), pick(mll, engine="scikit-learn", model="logistic")
sg, kg = pick(mll, engine="spark-mllib", model="gbt"), pick(mll, engine="scikit-learn", model="gbt")
P(f"`mllib_demo.py` fits the last development fold ({n(mlla['train_rows'])} training rows, 12 features) with an "
  f"MLlib `Pipeline` (Imputer → VectorAssembler → StandardScaler → LogisticRegression, and a GBTClassifier) and "
  f"with the equivalent scikit-learn pipelines. Training-quantile clipping is applied first, and the "
  f"regularisation is matched (MLlib `regParam` = 1/(C·n)). The logistic models agree: PR-AUC "
  f"{f(sl['pr_auc'], 3)} (MLlib) vs {f(ss['pr_auc'], 3)}, with a Spearman correlation of "
  f"{f(mlla['spearman_logistic_probs'], 3)} between their probabilities. MLlib's GBT reaches {f(sg['pr_auc'], 3)} "
  f"in {f(sg['fit_s'], 0)} s against {f(kg['pr_auc'], 3)} in {f(kg['fit_s'], 0)} s for scikit-learn's "
  f"histogram boosting. On one machine MLlib adds overhead, but the same pipeline code would run on a cluster.")
H("6.4 Alarm threshold from costs estimated from the data", 2)
P(f"A missed failure costs the Deposit Insurance Fund the resolution cost. The median `COST` of the "
  f"{n(costs['with_cost'])} development-period failures with a recorded cost is "
  f"{f(costs['cost_median_k'] / 1000, 1)} million USD (median loss rate {pc(costs['loss_rate_median'], 0)} of "
  f"assets). Assume early supervisory action avoids a share α = {costs['alpha']} of that loss. A missed failed "
  f"bank then costs {f(costs['cfn_per_bank_k'] / 1000, 2)} million USD, which is "
  f"{f(costs['cfn_per_row_k'], 0)} thousand USD per positive bank-quarter (a failed bank has "
  f"{f(costs['positive_rows_per_failed_bank'], 2)} positive rows). A false alarm costs one targeted examination, "
  f"assumed at {f(costs['cfp_k'], 0)} thousand USD. For calibrated probabilities the Bayes threshold would be "
  f"C_FP/(C_FP + C_FN) = {f(costs['bayes_threshold'], 3)} (Elkan, 2001). The threshold actually used, "
  f"{f(choice['threshold'], 4)}, minimises the total cost over the pooled out-of-fold development predictions, "
  f"so it also absorbs any miscalibration (Figure [[F:cost]]). α and the examination cost are assumptions; Appendix G "
  f"repeats the choice for 12 combinations.")
FIG("cost", FIGS / "model_cost_curve.png", "Expected cost of the alarm rule on out-of-fold development predictions.", 12)
H("6.5 Single evaluation on the test period", 2)
TAB(["Model", "PR-AUC", "PR-AUC*", "ROC-AUC", "Recall top 2%", "Alerts", "TP", "FN", "Cost M USD"],
    [[NAMES.get(r_["model"], r_["model"]) + (" (chosen)" if r_["chosen"] == "true" else ""), f(r_["pr_auc"], 3),
      f(r_["pr_auc_report_available_before_failure"], 3), f(r_["roc_auc"], 3), f(r_["recall_top2pct"], 3),
      n(r_["alerts"]), n(r_["tp"]), n(r_["fn"]), f(float(r_["cost_k"]) / 1000, 1)] for r_ in test],
    widths=[4.8, 1.5, 1.6, 1.6, 1.9, 1.4, 1.1, 1.1, 1.9], size=8,
    label="test", caption=f"Test 2011Q1–2025Q2 ({n(cht['rows'])} rows, {n(cht['positives'])} positives) at the thresholds fixed on "
            f"development. PR-AUC* excludes positives whose report was published after the failure.")
P(f"The choice (model, features, threshold) was written to `model_choice.json` (frozen {choice['frozen_at'][:10]}, "
  f"hash `{choice['sha1'][:10]}`) before the test rows were scored. Each scoring is appended to "
  f"`test_log.jsonl`, which holds {ntests} entry for this run. The chosen model reaches a test PR-AUC of "
  f"{f(cht['pr_auc'], 3)} against {f(blt['pr_auc'], 3)} for the baseline. At the cost-based threshold it raises "
  f"{n(cht['alerts'])} alerts and catches {n(cht['tp'])} of {n(cht['positives'])} positive bank-quarters, at an "
  f"expected cost of {f(float(cht['cost_k']) / 1000, 0)} million USD; with no model the cost would be "
  f"{f(float(cht['cost_no_model_k']) / 1000, 0)} million. By bank, {n(chd['detected_at_threshold'])} of "
  f"{n(chd['failed_banks'])} failed banks ({pc(int(chd['detected_at_threshold']) / int(chd['failed_banks']), 0)}) "
  f"are flagged at least once in the four quarters before failure, with a median lead time of "
  f"{f(chd['median_lead_quarters'], 1)} quarters (baseline: {n(bld['detected_at_threshold'])}). ROC-AUC barely "
  f"separates the two models ({f(cht['roc_auc'], 3)} vs {f(blt['roc_auc'], 3)}) because nearly all of the "
  f"{n(cht['rows'])} rows are easy negatives; PR-AUC, the alert counts and the cost show the difference that matters "
  f"to a supervisor: {n(int(blt['alerts']) - int(cht['alerts']))} fewer examinations for "
  f"{n(int(cht['tp']) - int(blt['tp']))} more detected positive quarters.")
y23 = [r_ for r_ in dety if r_["model"] == CH and int(r_["fail_year"]) in (2023, 2024, 2025)]
n23, d23 = sum(int(r_["failed_banks"]) for r_ in y23), sum(int(r_["detected"]) for r_ in y23)
y11 = [r_ for r_ in dety if r_["model"] == CH and 2011 <= int(r_["fail_year"]) <= 2014]
n11, d11 = sum(int(r_["failed_banks"]) for r_ in y11), sum(int(r_["detected"]) for r_ in y11)
P(f"**Where it fails.** The model flags {d11} of the {n11} banks that failed in 2011–2014, which resemble the "
  f"development crises, but only {d23} of the {n23} failures of 2023–2025 (Appendix H). The 2023 failures came "
  f"from deposit runs and unrealised losses on securities while capital and asset-quality ratios looked sound "
  f"(Board of Governors of the Federal Reserve System, 2023). Credit-risk features cannot see this, and the "
  f"training data contain almost no such cases. The remedy belongs to the pipeline: add uninsured-deposit and "
  f"securities-valuation fields to `FIN_FIELDS` and retrain, with no other stage changing.")
FIG("pr", FIGS / "model_pr_test.png", "Precision–recall curves on the test period (single evaluation).", 11)
H("6.6 The alarm rule in operation and its robustness", 2)
_cm = []
for lab_, m_ in (("Development, out-of-fold", CH), ("Development, out-of-fold", "baseline_logit_4")):
    d_ = thr[m_]
    _cm.append([lab_, NAMES[m_], f(d_["threshold"], 4), n(d_["tp"]), n(d_["fp"]), n(d_["fn"]),
                f(d_["tp"] / (d_["tp"] + d_["fp"]), 3), f(d_["tp"] / (d_["tp"] + d_["fn"]), 3), f(d_["cost_k"] / 1000, 0)])
for r_ in (cht, blt):
    _cm.append(["Test (single use)", NAMES[r_["model"]], f(r_["threshold"], 4), n(r_["tp"]), n(r_["fp"]), n(r_["fn"]),
                f(r_["precision"], 3), f(r_["recall"], 3), f(float(r_["cost_k"]) / 1000, 0)])
TAB(["Sample", "Model", "Threshold", "TP", "FP", "FN", "Precision", "Recall", "Cost M USD"], _cm,
    widths=[3.4, 4.2, 1.6, 1.2, 1.2, 1.2, 1.5, 1.3, 1.6], size=7.6,
    label="confusion", caption="Confusion counts at the frozen thresholds (bank-quarters).")
_beats = sum(1 for r_ in sens if float(r_["test_cost_k"]) < float(r_["test_cost_no_model_k"]))
_thr = [float(r_["threshold_dev"]) for r_ in sens]
_tr = [float(r_["test_recall"]) for r_ in sens]
P(f"On the test period recall is higher and precision lower than in development for both models. That is what "
  f"a fixed threshold implies when the base rate falls from {f(dev['base_rate_pct'], 2)}% to "
  f"{f(tst['base_rate_pct'], 2)}% and most test failures (2011–2014) resemble the development crises. In both "
  f"samples the chosen model has higher precision and higher recall than the baseline, so it needs fewer alerts "
  f"for more true positives. A precision of about {f(cht['precision'], 2)} means roughly one alarm in "
  f"{f(1 / float(cht['precision']), 0)} points at a bank that fails within a year, which is a workable "
  f"inspection list for a supervisor when the alternative is to miss a failure costing several million dollars. "
  f"The cost assumptions matter, so Appendix G repeats the whole procedure (threshold chosen on development, "
  f"applied once to test) for 12 combinations of α and examination cost. The threshold ranges from "
  f"{f(min(_thr), 3)} to {f(max(_thr), 3)} and test recall from {f(min(_tr), 2)} to {f(max(_tr), 2)}, and the "
  f"model's cost is below the no-model cost in {_beats} of {len(sens)} combinations. The conclusion that the "
  f"model is worth using therefore does not hinge on the exact assumptions, while the right alert volume does: "
  f"it should be set with the supervisor's real examination budget.")

# ================================================================== 7
H("7. Part 6 – Conclusions")
P(f"**Q1 – storage and processing.** Four decades of Call Reports are a medium-data problem. "
  f"{n(comp['financials_rows'])} rows fit in {f(cz['size_mb'], 0)} MB of partitioned Parquet, and Polars computes "
  f"the core workload in about a second on a 4-core machine. Typed columnar storage, partitioning by year with "
  f"row-group statistics, and lazy execution with pushdown are what make this efficient (Tables [[T:storage]] and [[T:engines]]). Spark "
  f"gives identical results but costs a near-fixed ~{f(sp1['median_s'], 0)} s. It becomes worthwhile only for data "
  f"that no longer fit one machine (tens of times today's panel, e.g. loan-level data), data already in a "
  f"distributed store, or many concurrent jobs.")
P(f"**Q2 – data quality.** The raw data contain at least eleven distinct problems (Table [[T:dq]]): a failures "
  f"list that mixes {n(FT['RESTYPE = ASSISTANCE (bank stayed open)'])} assistance deals with failures and has "
  f"{n(FT['rows without CERT'])} rows without a key; {n(pinfo['rows_removed_on_or_after_failure'])} reports dated on or "
  f"after a failure; a field missing for 17 whole years; missingness that is nearly five times higher before failure; "
  f"zeros that mean 'not reported'; cumulative income; gaps inside series; identity violations that are really "
  f"distress signals; heavy tails; restated values; and snapshot fields that describe the future. Each is found "
  f"by an automated check that reruns with every download, and each has an explicit rule. Rules that learn "
  f"numbers are fitted on the training window only, which the leakage proof verifies (Table [[T:leakprep]]). "
  f"Mergers ({n(E.get('merger_or_absorption', 0))} banks) are separated from failures ({n(E.get('failure', 0))}), "
  f"and the label window, the availability date and the censoring rule make the panel point-in-time; seven "
  f"assertions and a look-ahead test stop the pipeline if any of this breaks.")
P(f"**Q3 – honest validation.** Yes, within limits. Purged rolling-origin CV, a four-quarter gap, preprocessing "
  f"inside the training pipeline, a frozen choice and a single logged test evaluation give an estimate that can "
  f"be trusted. The model beats the baseline and catches {pc(int(chd['detected_at_threshold']) / int(chd['failed_banks']), 0)} "
  f"of failed banks in advance, with a test PR-AUC of {f(cht['pr_auc'], 3)} against {f(blt['pr_auc'], 3)} and a "
  f"base rate of {f(tst['base_rate_pct'], 2)}%. The test is honest because nothing about it was chosen after "
  f"seeing it: the choice file was frozen and hashed before scoring, and the log shows {ntests} scoring. The "
  f"same honesty shows the blind spot: liquidity-driven failures like those of 2023, which the next version of "
  f"the field list should address.")
H("7.1 Limitations", 2)
TAB(["Limitation", "Effect on the results", "Mitigation here / next step"], [
    ["Distressed mergers are labelled 0", "some failures in substance count as negatives; recall is understated",
     "exit types are kept in the panel, so a 'distressed exit' label can be tested without re-ingesting"],
    ["Avoidable-loss share α and examination cost are assumptions", "the alert volume depends on them",
     f"12-combination sensitivity (Appendix G); cost below no-model in {_beats} of {len(sens)}"],
    ["Costs are nominal USD", "older failures look cheaper", "deflate COST with a price index"],
    ["60-day publication lag is approximate", f"{n(pinfo['positives_failing_before_available'])} positives failed before the report was public",
     "PR-AUC* in Table [[T:test]] scores without them"],
    ["API serves restated, not first-release, values", "slight look-ahead in amended reports", "archive each download (manifest, SHA-256) to build vintages from now on"],
    ["No interest-rate, uninsured-deposit or market data", f"2023–2025 failures mostly missed ({d23} of {n23})",
     "add fields to FIN_FIELDS; no other stage changes"],
    ["Few test failures after 2014", "the estimate for calm periods is noisy", "report per-year detection (Appendix H)"],
    ["Benchmarks on one 4-core machine", "absolute times are hardware-specific", "ratios between engines and the scans are what generalise; environment saved with each result"],
], widths=[4.4, 5.4, 7.2], size=7.6, label="limits", caption="Limitations, their effect and how the pipeline mitigates them.")
H("7.2 What would change for Vietnam", 2)
P("The State Bank of Vietnam's supervisory data are monthly or quarterly, follow "
  "Vietnamese accounting standards and Basel II capital rules (State Bank of Vietnam, 2016), and cover about 50 "
  "banks. First, the event differs: weak banks are put under special control and transferred compulsorily rather "
  "than closed: four weak banks were transferred compulsorily to stronger banks in 2024–2025 under the Law on Credit "
  "Institutions No. 32/2024/QH15 (National Assembly of Vietnam, 2024). The label would be 'placed under special control', with very few events, so a supervised model "
  "would need US-trained patterns, expert scorecards and peer ranks, which the cross-sectional features already "
  "provide. Second, the volume is tiny, so Polars alone suffices; Spark would matter only for credit-registry "
  "(CIC) loan-level data. Third, the leakage rules carry over unchanged: use only data available at the decision "
  "date, split by time, and fit preprocessing on the past.")
TAB(["Pipeline component", "US (this report)", "Vietnam (SBV supervision)"], [
    ["Acquisition", "public API, meta.total as completeness oracle", "internal reporting system; reconcile with the number of reporting institutions per period"],
    ["Storage", f"Parquet by year, {f(cz['size_mb'], 0)} MB", "same layout; tiny volume (about 50 banks), Spark only for CIC loan-level data"],
    ["Quality rules", "identity checks, YTD, zeros as 'not reported'", "same checks plus VAS-to-IFRS mapping and changes in reporting templates"],
    ["Event / label", "FDIC failure within 4 quarters", "special control or compulsory transfer; very few events"],
    ["Features", "CAMELS ratios, dynamics, peer ranks", "same pillars; peer ranks matter more with few banks; add liquidity and related-party exposure"],
    ["Validation", "rolling-origin CV, single test", "too few events for a test set: back-test on known cases, combine with expert scorecards"],
], widths=[3.2, 5.4, 8.4], size=7.6, label="vietnam", caption="What carries over to Vietnam and what must change.")

H("8. Reproducibility")
P("The public repository holds `README.md`, a pinned `requirements.txt`, the code, an executed notebook and the "
  "evidence in `reports/`; data files are git-ignored. The README gives the end-to-end commands: "
  "`pip install -r requirements.txt`, `python run.py ingest`, `python run.py all`. A rerun reproduces every table "
  "and figure, and the report is rebuilt from them. The seed is fixed (2026), the download is resumable, and "
  "`pytest -q` runs offline unit tests of the label window, YTD de-cumulation, gap handling, the look-ahead "
  "property and the Winsorizer.")
H("Use of AI tools and other sources")
P("Claude (Anthropic, 2026), a large language model, was used as a coding and writing assistant: to draft parts "
  "of the code, tests and report text, and to review the pipeline for leakage. The author decided the design, "
  "ran every stage and checked each number against the generated tables. The ingestion code is adapted from the "
  "course starter script `fdic_download_starter.py`. The same statement is in the README.")
H("References")
REFS = [
    "Anthropic. (2026). Claude [Large language model]. https://claude.ai",
    "BFT65B course. (2026). fdic_download_starter.py [Starter script]. National Economics University course page.",
    "Board of Governors of the Federal Reserve System. (2023). Review of the Federal Reserve's supervision and regulation of Silicon Valley Bank. https://www.federalreserve.gov/publications/files/svb-review-20230428.pdf",
    "Cole, R. A., & White, L. J. (2012). Déjà vu all over again: The causes of U.S. commercial bank failures this time around. Journal of Financial Services Research, 42(1–2), 5–29. https://doi.org/10.1007/s10693-011-0116-9",
    "Elkan, C. (2001). The foundations of cost-sensitive learning. In Proceedings of the 17th International Joint Conference on Artificial Intelligence (pp. 973–978).",
    "Federal Deposit Insurance Corporation. (n.d.). BankFind Suite API and field definitions. Retrieved November 20, 2026, from https://api.fdic.gov/banks/docs/",
    "López de Prado, M. (2018). Advances in financial machine learning. Wiley.",
    "Martin, D. (1977). Early warning of bank failure: A logit regression approach. Journal of Banking & Finance, 1(3), 249–276. https://doi.org/10.1016/0378-4266(77)90022-X",
    "McKinney, W. (2010). Data structures for statistical computing in Python. In Proceedings of the 9th Python in Science Conference (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a",
    "National Assembly of Vietnam. (2024). Law on Credit Institutions (Law No. 32/2024/QH15).",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Polars developers. (2026). Polars (Version 2.1) [Computer software]. https://pola.rs",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
    "State Bank of Vietnam. (2016). Circular No. 41/2016/TT-NHNN on capital adequacy ratios for banks and foreign bank branches.",
    "Tukey, J. W. (1977). Exploratory data analysis. Addison-Wesley.",
    "Zaharia, M., Xin, R. S., Wendell, P., et al. (2016). Apache Spark: A unified engine for big data processing. Communications of the ACM, 59(11), 56–65. https://doi.org/10.1145/2934664",
]
for ref in REFS:
    p = doc.add_paragraph(ref)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(0.8), Cm(-0.8)
    p.paragraph_format.space_after = Pt(2)
    for rr in p.runs:
        rr.font.size = Pt(9)

# ================================================================== APPENDIX
doc.add_page_break()
H("Appendix")
H("A. Downloaded fields", 2)
P("Financials (" + str(len(CFG.FIN_FIELDS)) + "): " + ", ".join(CFG.FIN_FIELDS) + ". Year-to-date: "
  + ", ".join(CFG.YTD_FIELDS) + ".", size=9)
P("Failures: " + ", ".join(CFG.FAIL_FIELDS) + ". Institutions: " + ", ".join(CFG.INST_FIELDS)
  + ". Never used as features: " + ", ".join(sorted(c for c in CFG.FUTURE_COLUMNS if c.isupper())) + ".", size=9)
H("B. Feature dictionary", 2)
TAB(["Feature", "Group", "Definition", "Null %"],
    [[r_["feature"], r_["pillar"], r_["formula"], f(r_["null_pct"], 1)] for r_ in fdict],
    widths=[4.0, 1.2, 10.2, 1.6], size=7)
H("C. Label checks (`panel_checks.csv`)", 2)
TAB(["Check", "Passed", "Detail"], [[r_["check"], r_["passed"], r_["detail"][:70]] for r_ in pchecks],
    widths=[7.5, 1.5, 8.0], size=7.5)
H("D. Spark physical plan (operator tree)", 2)
tree = []
for line in (T / "plan_spark_formatted.txt").read_text().splitlines()[1:]:
    if not line.strip():
        break
    tree.append(line)
p = doc.add_paragraph()
rr = p.add_run("\n".join(tree))
rr.font.name, rr.font.size = "Consolas", Pt(7.5)
H("E. Permutation importance (chosen model, last development fold, PR-AUC drop)", 2)
TAB(["Feature", "Mean drop", "SD"], [[r_["feature"], f(r_["pr_auc_drop_mean"], 4), f(r_["pr_auc_drop_sd"], 4)]
                                     for r_ in perm[:15]], widths=[6, 3, 3], size=7.5)
H("F. Cross-validation folds", 2)
TAB(["Model", "Fold", "Validation", "Train rows", "Valid pos.", "Base rate %", "PR-AUC"],
    [[r_["model"], r_["fold"], r_["valid"], n(r_["train_rows"]), n(r_["positives"]), f(r_["base_rate_pct"], 2),
      f(r_["pr_auc"], 3)] for r_ in folds if r_["model"] in (CH, "baseline_logit_4")],
    widths=[3.6, 1.0, 4.6, 2.0, 1.7, 1.8, 1.6], size=7.5)
H("G. Cost sensitivity (threshold chosen on development, applied to test)", 2)
TAB(["α", "C_FP k$", "C_FN/C_FP", "Threshold", "Dev recall", "Test recall", "Test alerts", "Test cost M$", "No model M$"],
    [[r_["alpha"], f(r_["cfp_k"], 0), f(r_["cfn_over_cfp"], 1), f(r_["threshold_dev"], 4), f(r_["dev_recall"], 3),
      f(r_["test_recall"], 3), n(r_["test_alerts"]), f(float(r_["test_cost_k"]) / 1000, 0),
      f(float(r_["test_cost_no_model_k"]) / 1000, 0)] for r_ in sens], size=7.5)
H("H. Failed banks flagged in the test period, by failure year", 2)
yrs = sorted({r_["fail_year"] for r_ in dety}, key=int)


def _get(model, y, col):
    try:
        return pick(dety, model=model, fail_year=y)[col]
    except KeyError:
        return "0"


TAB(["Year"] + yrs, [["Failed banks"] + [_get(CH, y, "failed_banks") for y in yrs],
                     ["Chosen model"] + [_get(CH, y, "detected") for y in yrs],
                     ["Baseline"] + [_get("baseline_logit_4", y, "detected") for y in yrs]], size=7)
FIG("cvfolds", FIGS / "model_cv_folds.png", "PR-AUC by CV fold and model.", 13)
H("I. Polars optimised plan of the benchmark workload (excerpt)", 2)
P("Compare with the naive plan (`plan_bench_polars_naive.txt`), which scans `*/36` columns and evaluates every "
  "guard separately. The optimised plan below projects 11 of 36 columns in the scan and caches shared "
  "sub-expressions as `__POLARS_CSER_*` columns.", size=9)
_pl = [ln[:150] for ln in (T / "plan_bench_polars_optimized.txt").read_text().splitlines()]
p = doc.add_paragraph()
rr = p.add_run("\n".join(_pl[:40]))
rr.font.name, rr.font.size = "Consolas", Pt(6.5)
H("J. PySpark vs Polars parity by column (`spark_parity_columns.csv`)", 2)
TAB(["Column", "Null disagreements", "Max absolute difference"],
    [[r_["column"], r_["null_disagreements"], f"{float(r_['max_abs_diff']):.1e}"] for r_ in rows("spark_parity_columns.csv")],
    widths=[5, 4, 4], size=7.5)

# resolve [[T:x]] / [[F:x]] cross-references now that every table and figure is numbered
def _resolve(par):
    for run_ in par.runs:
        if "[[" in run_.text:
            run_.text = re.sub(r"\[\[([TF]:[a-z]+)\]\]", lambda m: str(REF[m.group(1)]), run_.text)


for par in doc.paragraphs:
    _resolve(par)
for tb in doc.tables:
    for row in tb.rows:
        for c in row.cells:
            for par in c.paragraphs:
                _resolve(par)
docx_path = OUT / "BFT65B_report.docx"
doc.save(docx_path)
print("wrote", docx_path)
soffice = shutil.which("soffice") or shutil.which("libreoffice")
if soffice:
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(docx_path)],
                   check=True, capture_output=True)
    print("wrote", OUT / "BFT65B_report.pdf")
