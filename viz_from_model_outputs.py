

# Expected files in the input directory are the outputs for RQ3 model script:
# - test_set_predictions_slim.csv
# - test_set_predictions_full.csv
# - cross_hospital_metrics.csv

# Outputs (PNG + a summary CSV) go to:
#  C:\Users\dckee\OneDrive\Documents\DAT490\model_cross_hospital\viz_out


from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# config dir
# ==========================
BASE_DIR = Path(r"C:\Users\dckee\OneDrive\Documents\DAT490\model_cross_hospital")
SLIM_CSV = BASE_DIR / "test_set_predictions_slim.csv"
FULL_CSV = BASE_DIR / "test_set_predictions_full.csv"
CROSS_CSV = BASE_DIR / "cross_hospital_metrics.csv"
OUT_DIR = BASE_DIR / "viz_out"


# ==========================
# Helpers
# ==========================
def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_fig(outdir: Path, filename: str, dpi: int = 180) -> Path:
    outpath = outdir / filename
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    return outpath

def to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def filter_positive_pairs(y_true: pd.Series, y_pred: pd.Series):
    yt = to_numeric(y_true)
    yp = to_numeric(y_pred)
    m = yt.notna() & yp.notna() & (yt > 0) & (yp > 0)
    return yt[m], yp[m]

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


# ==========================
# Plotting
# ==========================
def plot_pred_vs_actual(df: pd.DataFrame, outdir: Path, pred_col: str, tag: str, sample_max: int = 100_000):
    if "true_negotiated_rate" not in df.columns or pred_col not in df.columns:
        print(f"[WARN] Missing columns for pred-vs-actual: true_negotiated_rate or {pred_col}")
        return

    yt, yp = filter_positive_pairs(df["true_negotiated_rate"], df[pred_col])
    if len(yt) == 0:
        print(f"[WARN] No valid positive pairs for {pred_col}. Skipping.")
        return

    # Downsample for readability/speed
    if len(yt) > sample_max:
        idx = np.random.RandomState(7).choice(np.arange(len(yt)), size=sample_max, replace=False)
        yt = yt.iloc[idx]
        yp = yp.iloc[idx]

    y_true = yt.to_numpy()
    y_pred = yp.to_numpy()

    _r2 = r2_score(y_true, y_pred)
    _mae = mae(y_true, y_pred)
    _rmse = rmse(y_true, y_pred)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=3, alpha=0.25)

    # y=x reference line
    minv = max(1e-6, float(min(y_true.min(), y_pred.min())))
    maxv = float(max(y_true.max(), y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv], linewidth=1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual negotiated rate (log)")
    plt.ylabel("Predicted negotiated rate (log)")
    plt.title(f"{tag}: Predicted vs Actual | R²={_r2:.3f}  MAE={_mae:,.0f}  RMSE={_rmse:,.0f}")

    fname = f"{tag.lower().replace(' ', '_')}_pred_vs_actual_{pred_col}.png"
    save_fig(outdir, fname)

def plot_log_residual_hist(df: pd.DataFrame, outdir: Path, pred_col: str, tag: str):
    if "true_negotiated_rate" not in df.columns or pred_col not in df.columns:
        print(f"[WARN] Missing columns for residuals: true_negotiated_rate or {pred_col}")
        return

    yt, yp = filter_positive_pairs(df["true_negotiated_rate"], df[pred_col])
    if len(yt) == 0:
        print(f"[WARN] No valid positive pairs for {pred_col}. Skipping residual hist.")
        return

    res = np.log10(yp.to_numpy()) - np.log10(yt.to_numpy())

    plt.figure(figsize=(10, 6))
    plt.hist(res, bins=80)
    plt.xlabel("log10(predicted) - log10(actual)")
    plt.ylabel("Count")
    plt.title(f"{tag}: Residuals in log10 space ({pred_col})")

    fname = f"{tag.lower().replace(' ', '_')}_residual_hist_log10_{pred_col}.png"
    save_fig(outdir, fname)

def plot_error_by_group(df: pd.DataFrame, outdir: Path, group_col: str, tag: str, top_n: int = 12):
    needed = {"true_negotiated_rate", "pred_catboost", "pred_random_forest", group_col}
    if not needed.issubset(df.columns):
        print(f"[WARN] Missing columns for group plot '{group_col}': {needed - set(df.columns)}")
        return

    base = df.copy()
    base["true_negotiated_rate"] = to_numeric(base["true_negotiated_rate"])
    base["pred_catboost"] = to_numeric(base["pred_catboost"])
    base["pred_random_forest"] = to_numeric(base["pred_random_forest"])
    base = base.dropna(subset=["true_negotiated_rate", "pred_catboost", "pred_random_forest"])

    base["ae_cb"] = (base["pred_catboost"] - base["true_negotiated_rate"]).abs()
    base["ae_rf"] = (base["pred_random_forest"] - base["true_negotiated_rate"]).abs()

    agg = (
        base.groupby(group_col)
        .agg(
            rows=("true_negotiated_rate", "size"),
            MAE_CB=("ae_cb", "mean"),
            MedAE_CB=("ae_cb", "median"),
            MAE_RF=("ae_rf", "mean"),
            MedAE_RF=("ae_rf", "median"),
        )
        .sort_values("rows", ascending=False)
        .head(top_n)
        .reset_index()
    )

    x = np.arange(len(agg))
    width = 0.35

    # MAE plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, agg["MAE_CB"], width, label="CatBoost MAE")
    plt.bar(x + width / 2, agg["MAE_RF"], width, label="RandomForest MAE")
    plt.xticks(x, agg[group_col].astype(str), rotation=30, ha="right")
    plt.ylabel("MAE (absolute dollars)")
    plt.title(f"{tag}: MAE by {group_col} (Top {top_n} by volume)")
    plt.legend()
    save_fig(outdir, f"{tag.lower().replace(' ', '_')}_mae_by_{group_col}.png")

    # MedianAE plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, agg["MedAE_CB"], width, label="CatBoost MedianAE")
    plt.bar(x + width / 2, agg["MedAE_RF"], width, label="RandomForest MedianAE")
    plt.xticks(x, agg[group_col].astype(str), rotation=30, ha="right")
    plt.ylabel("Median Absolute Error (absolute dollars)")
    plt.title(f"{tag}: MedianAE by {group_col} (Top {top_n} by volume)")
    plt.legend()
    save_fig(outdir, f"{tag.lower().replace(' ', '_')}_medae_by_{group_col}.png")

def plot_better_model_share(df: pd.DataFrame, outdir: Path, tag: str):
    if "better_model" not in df.columns:
        print("[WARN] 'better_model' column missing. Skipping better-model share plot.")
        return

    counts = df["better_model"].astype(str).value_counts()
    labels = counts.index.tolist()
    vals = counts.values.astype(float)

    plt.figure(figsize=(7, 5))
    plt.bar(labels, vals)
    plt.ylabel("Row count")
    plt.title(f"{tag}: Which model wins more rows? (better_model)")
    save_fig(outdir, f"{tag.lower().replace(' ', '_')}_better_model_share.png")

def plot_cross_hospital_metrics(cross: pd.DataFrame, outdir: Path):
    needed = {"model", "MAE", "RMSE", "R2", "MedAE"}
    if not needed.issubset(cross.columns):
        print(f"[WARN] cross_hospital_metrics.csv missing columns: {needed - set(cross.columns)}")
        return

    cross = cross.copy()
    for c in ["MAE", "RMSE", "R2", "MedAE"]:
        cross[c] = to_numeric(cross[c])

    # MAE / RMSE / MedAE
    for metric in ["MAE", "RMSE", "MedAE"]:
        plt.figure(figsize=(9, 5))
        plt.bar(cross["model"].astype(str), cross[metric])
        plt.ylabel(metric)
        plt.title(f"Cross-Hospital Performance: {metric} by model")
        save_fig(outdir, f"cross_hospital_{metric.lower()}_by_model.png")

    # R2 (can be negative)
    plt.figure(figsize=(9, 5))
    plt.bar(cross["model"].astype(str), cross["R2"])
    plt.axhline(0, linewidth=1)
    plt.ylabel("R²")
    plt.title("Cross-Hospital Performance: R-squared by model (negative = worse than mean baseline)")
    save_fig(outdir, "cross_hospital_r2_by_model.png")

def plot_top_worst(df: pd.DataFrame, outdir: Path, pred_col: str, tag: str, top_n: int = 25):
    if "true_negotiated_rate" not in df.columns or pred_col not in df.columns:
        print(f"[WARN] Missing columns for worst-predictions: true_negotiated_rate or {pred_col}")
        return

    base = df.copy()
    base["true_negotiated_rate"] = to_numeric(base["true_negotiated_rate"])
    base[pred_col] = to_numeric(base[pred_col])
    base = base.dropna(subset=["true_negotiated_rate", pred_col])

    base["abs_error"] = (base[pred_col] - base["true_negotiated_rate"]).abs()
    base = base.sort_values("abs_error", ascending=False).head(top_n).copy()

    label_cols = [c for c in ["hospital_name", "cpt_code", "payer", "plan", "setting", "billing_class"] if c in base.columns]

    def row_label(r):
        parts = []
        for c in label_cols[:4]:
            parts.append(f"{c.split('_')[0]}={str(r[c])[:28]}")
        return " | ".join(parts) if parts else f"row={r.name}"

    labels = base.apply(row_label, axis=1).tolist()
    vals = base["abs_error"].to_numpy()

    plt.figure(figsize=(12, 8))
    y = np.arange(len(vals))
    plt.barh(y, vals)
    plt.yticks(y, labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute error ($)")
    plt.title(f"{tag}: Top {top_n} worst predictions by absolute error ({pred_col})")
    save_fig(outdir, f"{tag.lower().replace(' ', '_')}_top_{top_n}_worst_{pred_col}.png")


# ==========================
# Main
# ==========================
def main():
    outdir = ensure_outdir(OUT_DIR)

    # Load files
    if not SLIM_CSV.exists():
        raise FileNotFoundError(f"Missing: {SLIM_CSV}")
    if not FULL_CSV.exists():
        raise FileNotFoundError(f"Missing: {FULL_CSV}")
    if not CROSS_CSV.exists():
        raise FileNotFoundError(f"Missing: {CROSS_CSV}")

    slim = pd.read_csv(SLIM_CSV, low_memory=False)
    full = pd.read_csv(FULL_CSV, low_memory=False)
    cross = pd.read_csv(CROSS_CSV, low_memory=False)

    TAG_SLIM = "Test Set (Slim)"
    TAG_FULL = "Test Set (Full)"

    # 1) Predicted vs Actual + Residuals (uses slim)
    for pred_col in ["pred_catboost", "pred_random_forest"]:
        plot_pred_vs_actual(slim, outdir, pred_col, TAG_SLIM)
        plot_log_residual_hist(slim, outdir, pred_col, TAG_SLIM)

    # 2) Error by groups (use slim)
    plot_error_by_group(slim, outdir, group_col="hospital_name", tag=TAG_SLIM, top_n=12)
    plot_error_by_group(slim, outdir, group_col="setting", tag=TAG_SLIM, top_n=10)
    plot_error_by_group(slim, outdir, group_col="billing_class", tag=TAG_SLIM, top_n=10)

    # 3) Winner share
    plot_better_model_share(slim, outdir, TAG_SLIM)

    # 4) Cross hospital aggregate metrics
    plot_cross_hospital_metrics(cross, outdir)

    # 5) Worst predictions
    for pred_col in ["pred_catboost", "pred_random_forest"]:
        plot_top_worst(full, outdir, pred_col, TAG_FULL, top_n=25)

    # 6) Write an overall metrics CSV (from slim)
    base = slim.copy()
    needed = {"true_negotiated_rate", "pred_catboost", "pred_random_forest"}
    if needed.issubset(base.columns):
        base["true_negotiated_rate"] = to_numeric(base["true_negotiated_rate"])
        base["pred_catboost"] = to_numeric(base["pred_catboost"])
        base["pred_random_forest"] = to_numeric(base["pred_random_forest"])
        base = base.dropna(subset=list(needed))

        y = base["true_negotiated_rate"].to_numpy()
        cb = base["pred_catboost"].to_numpy()
        rf = base["pred_random_forest"].to_numpy()

        summary = pd.DataFrame([
            {"model": "CatBoost", "MAE": mae(y, cb), "RMSE": rmse(y, cb), "R2": r2_score(y, cb), "MedAE": medae(y, cb)},
            {"model": "RandomForest", "MAE": mae(y, rf), "RMSE": rmse(y, rf), "R2": r2_score(y, rf), "MedAE": medae(y, rf)},
        ])
        summary_path = outdir / "test_set_overall_metrics_from_slim.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[INFO] Wrote: {summary_path}")
    else:
        print(f"[WARN] Can't write summary metrics; missing columns: {needed - set(base.columns)}")

    print(f"[DONE] Figures saved to: {outdir}")


if __name__ == "__main__":
    main()


