# FDP-WEEKEND-Challenge
#!/usr/bin/env python3
"""
Student Marks Analysis Toolkit
==============================
Pipeline: generate/load CSV -> clean -> NumPy computations -> groupby analysis
-> statistics & hypothesis tests -> charts -> regression model -> saved report.

Run:  python student_marks_toolkit.py
Optional flags: --input data.csv  --output outputs  --seed 42  --n 150

Dependencies: pandas, numpy, matplotlib, scipy
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend: charts are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PASS_MARK = 40          # final marks (out of 100) needed to pass
ATTENDANCE_CUTOFF = 75  # typical minimum attendance requirement (%)
NUMERIC_COLS = ["attendance", "internal_marks", "final_marks", "study_hours"]
DEPARTMENTS = ["CSE", "ECE", "MECH", "CIVIL", "EEE"]


# --------------------------------------------------------------------------- #
# 1. Dataset creation / loading
# --------------------------------------------------------------------------- #
def generate_dataset(path: str, n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible synthetic dataset (with deliberate data problems)
    and save it as a raw CSV.

    Problems injected on purpose so the cleaning step has real work to do:
      * ~8% missing values in numeric columns, a few missing departments
      * 10 duplicated rows
      * inconsistent department text (case / whitespace)
      * attendance stored as text like "82%" in some rows
      * a few impossible values (attendance > 100, negative marks)
    """
    rng = np.random.default_rng(seed)
    dept_effect = {"CSE": 4, "ECE": 1, "MECH": -2, "CIVIL": -3, "EEE": 0}

    dept = rng.choice(DEPARTMENTS, size=n)
    attendance = np.clip(rng.normal(80, 12, n), 40, 100).round(1)
    study_hours = np.clip(rng.normal(3.5, 1.3, n), 0.5, 8).round(1)
    internal = np.clip(10 + 0.30 * attendance + 1.5 * study_hours
                       + rng.normal(0, 4, n), 5, 50).round(1)
    final = (-14 + 0.45 * attendance + 0.80 * internal + 2.0 * study_hours
             + np.array([dept_effect[d] for d in dept]) + rng.normal(0, 9, n))
    final = np.clip(final, 0, 100).round(1)

    df = pd.DataFrame({
        "student_id": np.arange(1001, 1001 + n),
        "department": dept,
        "gender": rng.choice(["M", "F"], size=n),
        "attendance": attendance,
        "internal_marks": internal,
        "study_hours": study_hours,
        "final_marks": final,
    })

    # ---- inject problems ----
    df["attendance"] = df["attendance"].astype(object)
    for i in rng.choice(n, 12, replace=False):          # text with % sign
        df.at[i, "attendance"] = f"{df.at[i, 'attendance']}%"
    for col in NUMERIC_COLS:                            # missing numerics
        df.loc[rng.choice(n, 8, replace=False), col] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "department"] = np.nan
    messy = rng.choice(n, 20, replace=False)            # messy department text
    df.loc[messy[:10], "department"] = df.loc[messy[:10], "department"].str.lower()
    df.loc[messy[10:], "department"] = " " + df.loc[messy[10:], "department"] + " "
    df.at[5, "attendance"] = 130.0                      # impossible values
    df.at[17, "internal_marks"] = -8.0
    dupes = df.sample(10, random_state=seed)            # duplicate rows
    df = pd.concat([df, dupes], ignore_index=True)

    df.to_csv(path, index=False)
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV with pandas and validate that required columns exist."""
    df = pd.read_csv(path)
    required = {"student_id", "department", "attendance", "internal_marks", "final_marks"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
def snapshot(df: pd.DataFrame) -> dict:
    """Counts used for the before/after comparison."""
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_student_ids": int(df["student_id"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }


def clean_data(df: pd.DataFrame):
    """Clean the raw frame. Returns (clean_df, before_counts, after_counts, log)."""
    before = snapshot(df)
    log = []
    df = df.copy()

    # --- fix data types / text formats ---
    df["attendance"] = (df["attendance"].astype(str)
                        .str.replace("%", "", regex=False).str.strip())
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["department"] = df["department"].astype("string").str.strip().str.upper()
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("string").str.strip().str.upper()
    df["student_id"] = pd.to_numeric(df["student_id"], errors="coerce").astype("Int64")
    log.append("Converted numeric columns (removed '%' signs), standardised text case/whitespace.")

    # --- invalid values -> NaN (then handled as missing) ---
    invalid = (~df["attendance"].between(0, 100)) & df["attendance"].notna()
    invalid_int = (~df["internal_marks"].between(0, 50)) & df["internal_marks"].notna()
    invalid_fin = (~df["final_marks"].between(0, 100)) & df["final_marks"].notna()
    n_invalid = int(invalid.sum() + invalid_int.sum() + invalid_fin.sum())
    df.loc[invalid, "attendance"] = np.nan
    df.loc[invalid_int, "internal_marks"] = np.nan
    df.loc[invalid_fin, "final_marks"] = np.nan
    log.append(f"Set {n_invalid} out-of-range values to missing (attendance 0-100, internal 0-50, final 0-100).")

    # --- duplicates ---
    n0 = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="student_id", keep="first")
    log.append(f"Removed {n0 - len(df)} duplicate rows (exact duplicates + repeated student_id).")

    # --- missing values ---
    n0 = len(df)
    df = df.dropna(subset=["department", "final_marks", "student_id"])
    log.append(f"Dropped {n0 - len(df)} rows missing department / final_marks / student_id "
               "(cannot be reliably imputed).")
    for col in ["attendance", "internal_marks", "study_hours"]:
        n_missing = int(df[col].isna().sum())
        df[col] = df[col].fillna(df.groupby("department")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
        log.append(f"Imputed {n_missing} missing '{col}' values with the department median.")

    df = df.reset_index(drop=True)
    after = snapshot(df)
    assert after["missing_values"] - int(df["gender"].isna().sum()) == 0, "Unexpected NaNs remain"
    return df, before, after, log


# --------------------------------------------------------------------------- #
# 3. NumPy computations (vectorised, no Python loops)
# --------------------------------------------------------------------------- #
def numpy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features using vectorised NumPy operations."""
    df = df.copy()
    final = df["final_marks"].to_numpy(dtype=float)
    att = df["attendance"].to_numpy(dtype=float)

    # 1) z-score of final marks
    z = (final - final.mean()) / final.std(ddof=1)
    df["final_zscore"] = z.round(3)

    # 2) min-max normalisation of attendance
    df["attendance_norm"] = ((att - att.min()) / (att.max() - att.min())).round(3)

    # 3) percentile rank of each student (fraction of students scoring <= them)
    sorted_final = np.sort(final)
    df["percentile_rank"] = (np.searchsorted(sorted_final, final, side="right")
                             / len(final) * 100).round(1)

    # 4) outlier flag using the IQR rule
    q1, q3 = np.percentile(final, [25, 75])
    iqr = q3 - q1
    df["is_outlier"] = np.where((final < q1 - 1.5 * iqr) | (final > q3 + 1.5 * iqr), True, False)

    # 5) grade bands via np.select
    conds = [final >= 85, final >= 70, final >= 55, final >= PASS_MARK]
    df["grade"] = np.select(conds, ["A", "B", "C", "D"], default="F")
    df["passed"] = final >= PASS_MARK
    return df


# --------------------------------------------------------------------------- #
# 4. Group-wise analysis
# --------------------------------------------------------------------------- #
def department_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("department")
    out = g.agg(students=("student_id", "count"),
                avg_attendance=("attendance", "mean"),
                avg_internal=("internal_marks", "mean"),
                avg_final=("final_marks", "mean"),
                median_final=("final_marks", "median"),
                std_final=("final_marks", "std"),
                pass_percentage=("passed", lambda s: s.mean() * 100)).round(2)
    return out.sort_values("avg_final", ascending=False)


def attendance_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(df["attendance"], [0, 60, 75, 90, 100],
                   labels=["<=60", "60-75", "75-90", ">90"], include_lowest=True)
    return (df.groupby(bands, observed=True)
              .agg(students=("student_id", "count"),
                   avg_final=("final_marks", "mean"),
                   pass_percentage=("passed", lambda s: s.mean() * 100)).round(2))


# --------------------------------------------------------------------------- #
# 5. Statistics, correlation, hypothesis tests, probability
# --------------------------------------------------------------------------- #
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["attendance", "internal_marks", "study_hours", "final_marks"]
    d = pd.DataFrame({"mean": df[cols].mean(), "median": df[cols].median(),
                      "std": df[cols].std(), "min": df[cols].min(), "max": df[cols].max()})
    return d.round(2)


def statistical_tests(df: pd.DataFrame) -> dict:
    res = {}
    # Pearson correlation (and its p-value) between each predictor and final marks
    for col in ["attendance", "internal_marks", "study_hours"]:
        r, p = stats.pearsonr(df[col], df["final_marks"])
        res[f"corr_{col}"] = (round(r, 3), p)

    # Hypothesis test: do students with good attendance score higher? (Welch t-test)
    good = df.loc[df["attendance"] >= ATTENDANCE_CUTOFF, "final_marks"]
    low = df.loc[df["attendance"] < ATTENDANCE_CUTOFF, "final_marks"]
    t, p = stats.ttest_ind(good, low, equal_var=False)
    res["ttest"] = {"t": round(t, 3), "p": p, "mean_good": good.mean(),
                    "mean_low": low.mean(), "n_good": len(good), "n_low": len(low)}

    # One-way ANOVA: do departments differ in mean final marks?
    groups = [g["final_marks"].to_numpy() for _, g in df.groupby("department")]
    f, p_anova = stats.f_oneway(*groups)
    res["anova"] = {"F": round(f, 3), "p": p_anova}

    # Probability calculations (empirical)
    res["p_pass"] = df["passed"].mean()
    res["p_pass_given_low_att"] = df.loc[df["attendance"] < ATTENDANCE_CUTOFF, "passed"].mean()
    res["p_pass_given_good_att"] = df.loc[df["attendance"] >= ATTENDANCE_CUTOFF, "passed"].mean()
    return res


def interpret(res: dict) -> list:
    """One-line interpretations of the statistical results."""
    lines = []
    t = res["ttest"]
    verdict = "significant" if t["p"] < 0.05 else "not significant"
    lines.append(f"t-test: students with attendance >= {ATTENDANCE_CUTOFF}% average "
                 f"{t['mean_good']:.1f} vs {t['mean_low']:.1f} marks (p = {t['p']:.4g}); "
                 f"difference is {verdict} at the 5% level.")
    a = res["anova"]
    lines.append("ANOVA: department means " +
                 ("differ significantly" if a["p"] < 0.05 else "do not differ significantly") +
                 f" (F = {a['F']}, p = {a['p']:.4g}).")
    r, _ = res["corr_attendance"]
    lines.append(f"Correlation: attendance vs final marks r = {r} "
                 f"({'strong' if abs(r) > 0.6 else 'moderate' if abs(r) > 0.3 else 'weak'} positive relationship).")
    lines.append(f"Probability: P(pass) = {res['p_pass']:.2%}; P(pass | attendance < {ATTENDANCE_CUTOFF}%) = "
                 f"{res['p_pass_given_low_att']:.2%} vs {res['p_pass_given_good_att']:.2%} when attendance is higher.")
    return lines


# --------------------------------------------------------------------------- #
# 6. Predictive model (NumPy least squares) with train/test evaluation
# --------------------------------------------------------------------------- #
def fit_regression(df: pd.DataFrame, seed: int = 42) -> dict:
    """Predict final marks from attendance, internal marks and study hours using
    OLS (np.linalg.lstsq) and evaluate on a held-out 20% test set."""
    feats = ["attendance", "internal_marks", "study_hours"]
    X = df[feats].to_numpy(dtype=float)
    y = df["final_marks"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = idx[:cut], idx[cut:]

    A_tr = np.column_stack([np.ones(len(tr)), X[tr]])
    coef, *_ = np.linalg.lstsq(A_tr, y[tr], rcond=None)
    pred = np.column_stack([np.ones(len(te)), X[te]]) @ coef

    resid = y[te] - pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
    return {"features": feats, "coefficients": dict(zip(["intercept"] + feats, coef.round(3))),
            "r2": round(1 - ss_res / ss_tot, 3),
            "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            "mae": round(float(np.mean(np.abs(resid))), 3),
            "n_train": len(tr), "n_test": len(te)}


# --------------------------------------------------------------------------- #
# 7. Charts
# --------------------------------------------------------------------------- #
def make_charts(df: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> list:
    paths = []

    def save(fig, name):
        p = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    # 1. Histogram of final marks
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["final_marks"], bins=15, color="#4C72B0", edgecolor="white")
    ax.axvline(df["final_marks"].mean(), color="red", ls="--", label="Mean")
    ax.axvline(PASS_MARK, color="black", ls=":", label=f"Pass mark ({PASS_MARK})")
    ax.set(title="Distribution of Final Marks", xlabel="Final marks (out of 100)", ylabel="Number of students")
    ax.legend()
    save(fig, "01_hist_final_marks.png")

    # 2. Bar chart: pass percentage by department
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ps = summary["pass_percentage"].sort_values(ascending=False)
    bars = ax.bar(ps.index, ps.values, color="#55A868")
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set(title="Pass Percentage by Department", xlabel="Department", ylabel="Pass percentage (%)", ylim=(0, 110))
    save(fig, "02_bar_pass_pct_department.png")

    # 3. Scatter + regression line: attendance vs final marks
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["attendance"], df["final_marks"], alpha=0.65, color="#C44E52")
    m, c = np.polyfit(df["attendance"], df["final_marks"], 1)
    xs = np.linspace(df["attendance"].min(), df["attendance"].max(), 100)
    ax.plot(xs, m * xs + c, color="black", label=f"Trend: y = {m:.2f}x + {c:.1f}")
    ax.set(title="Attendance vs Final Marks", xlabel="Attendance (%)", ylabel="Final marks (out of 100)")
    ax.legend()
    save(fig, "03_scatter_attendance_vs_final.png")

    # 4. Box plot: final marks by department
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = sorted(df["department"].unique())
    ax.boxplot([df.loc[df["department"] == d, "final_marks"] for d in order],
               patch_artist=True, boxprops=dict(facecolor="#8172B2", alpha=0.7))
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.set(title="Final Marks by Department", xlabel="Department", ylabel="Final marks (out of 100)")
    save(fig, "04_box_final_by_department.png")

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    cols = ["attendance", "internal_marks", "study_hours", "final_marks"]
    corr = df[cols].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set(title="Correlation Matrix", xlabel="Variable", ylabel="Variable")
    save(fig, "05_heatmap_correlation.png")
    return paths


# --------------------------------------------------------------------------- #
# 8. Report
# --------------------------------------------------------------------------- #
def write_report(path, before, after, log, desc, summary, band, res, interp, model):
    top, bottom = summary.index[0], summary.index[-1]
    L = ["STUDENT MARKS - FINDINGS SUMMARY", "=" * 40, "",
         "DATA CLEANING (before -> after)"]
    for k in before:
        L.append(f"  {k:<22}: {before[k]:>4} -> {after[k]:>4}")
    L += ["", "Cleaning steps:"] + [f"  - {s}" for s in log]
    L += ["", "DESCRIPTIVE STATISTICS", desc.to_string(),
          "", "DEPARTMENT SUMMARY", summary.to_string(),
          "", "ATTENDANCE BAND SUMMARY", band.to_string(),
          "", "STATISTICAL FINDINGS"] + [f"  - {s}" for s in interp]
    L += ["", "REGRESSION MODEL (final ~ attendance + internal + study_hours)",
          f"  Coefficients: {model['coefficients']}",
          f"  Test R2 = {model['r2']}, RMSE = {model['rmse']}, MAE = {model['mae']} "
          f"(train n={model['n_train']}, test n={model['n_test']})",
          "", "KEY TAKEAWAYS",
          f"  1. {top} has the highest average final marks ({summary.loc[top, 'avg_final']}); "
          f"{bottom} the lowest ({summary.loc[bottom, 'avg_final']}).",
          f"  2. Overall pass rate is {res['p_pass']:.1%}.",
          "  3. Attendance and internal marks are the strongest predictors of final marks."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# 9. Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Student marks analysis toolkit")
    ap.add_argument("--input", default="student_marks_raw.csv", help="raw CSV path")
    ap.add_argument("--output", default="outputs", help="output directory")
    ap.add_argument("--n", type=int, default=150, help="students to generate if CSV absent")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    a = ap.parse_args(argv)

    os.makedirs(a.output, exist_ok=True)
    if not os.path.exists(a.input):
        print(f"[1/7] '{a.input}' not found -> generating synthetic dataset (n={a.n}, seed={a.seed})")
        generate_dataset(a.input, a.n, a.seed)
    else:
        print(f"[1/7] Using existing dataset '{a.input}'")

    raw = load_data(a.input)
    print(f"[2/7] Loaded raw data: {raw.shape[0]} rows x {raw.shape[1]} columns")

    clean, before, after, log = clean_data(raw)
    print("[3/7] Cleaning (before -> after):")
    for k in before:
        print(f"      {k:<22}: {before[k]:>4} -> {after[k]:>4}")
    for s in log:
        print("      -", s)

    clean = numpy_features(clean)
    clean.to_csv(os.path.join(a.output, "student_marks_clean.csv"), index=False)

    summary = department_summary(clean)
    band = attendance_band_summary(clean)
    desc = descriptive_stats(clean)
    print("[4/7] Department summary:\n" + summary.to_string())
    print("      Descriptive statistics:\n" + desc.to_string())

    res = statistical_tests(clean)
    interp = interpret(res)
    print("[5/7] Statistical findings:")
    for s in interp:
        print("      -", s)

    model = fit_regression(clean, a.seed)
    print(f"[6/7] Regression: test R2={model['r2']}, RMSE={model['rmse']}, MAE={model['mae']}")

    charts = make_charts(clean, summary, a.output)
    report_path = os.path.join(a.output, "findings_summary.txt")
    write_report(report_path, before, after, log, desc, summary, band, res, interp, model)
    print(f"[7/7] Saved {len(charts)} charts, cleaned CSV and report to '{a.output}/'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
