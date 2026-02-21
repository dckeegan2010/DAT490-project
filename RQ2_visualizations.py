

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# CONFIG
# ==========================
BASE_DIR = Path(r"C:\Users\dckee\OneDrive\Documents\DAT490\model_compare_hash_rf")
PRED_CSV = BASE_DIR / "predictions_side_by_side_with_ids.csv"
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

def mae(ae: np.ndarray) -> float:
    return float(np.mean(ae))

def medae(ae: np.ndarray) -> float:
    return float(np.median(ae))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")


# ==========================
# Plot functions
# ==========================
def plot_pred_vs_actual(df: pd.DataFrame, outdir: Path, actual_col: str, pred_col: str, title: str, fname: str, sample_max: int = 120_000):
    yt, yp = filter_positive_pairs(df[actual_col], df[pred_col])
    if len(yt) == 0:
        print(f"[WARN] No valid positive pairs for {pred_col}. Skipping {fname}")
        return

    if len(yt) > sample_max:
        idx = np.random.RandomState(7).choice(np.arange(len(yt)), size=sample_max, replace=False)
        yt = yt.iloc[idx]
        yp = yp.iloc[idx]

    y_true = yt.to_numpy()
    y_pred = yp.to_numpy()

    _r2 = r2_score(y_true, y_pred)
    _rmse = rmse(y_true, y_pred)

    # MAE from abs error column if present, else compute
    _mae = float(np.mean(np.abs(y_pred - y_true)))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=3, alpha=0.25)

    minv = max(1e-6, float(min(y_true.min(), y_pred.min())))
    maxv = float(max(y_true.max(), y_pred.max()))
    plt.plot([minv, maxv], [minv, maxv], linewidth=1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Actual negotiated rate (log)")
    plt.ylabel("Predicted negotiated rate (log)")
    plt.title(f"{title}\nR²={_r2:.3f}  MAE={_mae:,.0f}  RMSE={_rmse:,.0f}")

    save_fig(outdir, fname)

def plot_residual_hist_log10(df: pd.DataFrame, outdir: Path, actual_col: str, pred_col: str, title: str, fname: str):
    yt, yp = filter_positive_pairs(df[actual_col], df[pred_col])
    if len(yt) == 0:
        print(f"[WARN] No valid positive pairs for {pred_col}. Skipping {fname}")
        return

    res = np.log10(yp.to_numpy()) - np.log10(yt.to_numpy())

    plt.figure(figsize=(10, 6))
    plt.hist(res, bins=80)
    plt.xlabel("log10(predicted) - log10(actual)")
    plt.ylabel("Count")
    plt.title(title)

    save_fig(outdir, fname)

def plot_winner_share(df: pd.DataFrame, outdir: Path, fname: str):
    if "winner" not in df.columns:
        print("[WARN] winner column missing; skipping winner-share plot.")
        return

    counts = df["winner"].astype(str).value_counts()
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index.tolist(), counts.values.astype(float))
    plt.ylabel("Row count")
    plt.title("UPenn: Which model wins more rows? (lower absolute error)")
    save_fig(outdir, fname)

def plot_error_by_setting(df: pd.DataFrame, outdir: Path, fname: str, top_n: int = 10):
    needed = {"setting", "abs_err_rf", "abs_err_cb"}
    if not needed.issubset(df.columns):
        print(f"[WARN] Missing columns for error-by-setting plot: {needed - set(df.columns)}")
        return

    base = df.copy()
    base["abs_err_rf"] = to_numeric(base["abs_err_rf"])
    base["abs_err_cb"] = to_numeric(base["abs_err_cb"])
    base = base.dropna(subset=["setting", "abs_err_rf", "abs_err_cb"])

    agg = (
        base.groupby("setting")
            .agg(rows=("setting", "size"),
                 MAE_RF=("abs_err_rf", "mean"),
                 MedAE_RF=("abs_err_rf", "median"),
                 MAE_CB=("abs_err_cb", "mean"),
                 MedAE_CB=("abs_err_cb", "median"))
            .sort_values("rows", ascending=False)
            .head(top_n)
            .reset_index()
    )

    x = np.arange(len(agg))
    width = 0.35

    # MAE
    plt.figure(figsize=(11, 6))
    plt.bar(x - width/2, agg["MAE_CB"], width, label="CatBoost MAE")
    plt.bar(x + width/2, agg["MAE_RF"], width, label="RF_hash MAE")
    plt.xticks(x, agg["setting"].astype(str), rotation=25, ha="right")
    plt.ylabel("MAE (absolute dollars)")
    plt.title("UPenn: MAE by setting")
    plt.legend()
    save_fig(outdir, "upenn_mae_by_setting.png")

    # MedAE
    plt.figure(figsize=(11, 6))
    plt.bar(x - width/2, agg["MedAE_CB"], width, label="CatBoost MedAE")
    plt.bar(x + width/2, agg["MedAE_RF"], width, label="RF_hash MedAE")
    plt.xticks(x, agg["setting"].astype(str), rotation=25, ha="right")
    plt.ylabel("Median Absolute Error (absolute dollars)")
    plt.title("UPenn: MedianAE by setting")
    plt.legend()
    save_fig(outdir, "upenn_medae_by_setting.png")

    # Combine both into a single “sixth” figure requested: grouped MAE + MedAE in one plot
    # (Useful as a single summary chart)
    plt.figure(figsize=(12, 6))
    plt.plot(x, agg["MAE_CB"], marker="o", label="CatBoost MAE")
    plt.plot(x, agg["MAE_RF"], marker="o", label="RF_hash MAE")
    plt.plot(x, agg["MedAE_CB"], marker="s", label="CatBoost MedAE")
    plt.plot(x, agg["MedAE_RF"], marker="s", label="RF_hash MedAE")
    plt.xticks(x, agg["setting"].astype(str), rotation=25, ha="right")
    plt.ylabel("Error (absolute dollars)")
    plt.title("UPenn: Error by setting (MAE and MedianAE)")
    plt.legend()
    save_fig(outdir, fname)

def plot_pct_error_box(df: pd.DataFrame, outdir: Path, fname: str, clip: float = 5.0):

    base = df.copy()
    base["pct_err_rf"] = to_numeric(base["pct_err_rf"])
    base["pct_err_cb"] = to_numeric(base["pct_err_cb"])
    base = base.dropna(subset=["pct_err_rf", "pct_err_cb"])

    # clip extreme percent errors for readability (e.g., +/- 500% if clip=5.0)
    base["pct_err_rf_clip"] = base["pct_err_rf"].clip(-clip, clip)
    base["pct_err_cb_clip"] = base["pct_err_cb"].clip(-clip, clip)

    plt.figure(figsize=(8, 6))
    plt.boxplot([base["pct_err_cb_clip"].to_numpy(), base["pct_err_rf_clip"].to_numpy()],
                labels=["CatBoost", "RF_hash"],
                showfliers=False)
    plt.ylabel(f"Percent error (clipped to ±{int(clip*100)}%)")
    plt.title("UPenn: Percent error distribution (clipped, outliers hidden)")
    save_fig(outdir, fname)


# ==========================
# Main 
# ==========================
def main():
    outdir = ensure_outdir(OUT_DIR)

    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Missing: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV, low_memory=False)

    # Ensure numeric where needed
    for c in ["true_rate", "pred_random_forest_hash", "pred_catboost", "abs_err_rf", "abs_err_cb", "pct_err_rf", "pct_err_cb"]:
        if c in df.columns:
            df[c] = to_numeric(df[c])

    # 1) Pred vs Actual — CatBoost
    plot_pred_vs_actual(
        df, outdir,
        actual_col="true_rate",
        pred_col="pred_catboost",
        title="UPenn: Predicted vs Actual (CatBoost)",
        fname="01_upenn_pred_vs_actual_catboost.png"
    )

    # 2) Pred vs Actual — RF_hash
    plot_pred_vs_actual(
        df, outdir,
        actual_col="true_rate",
        pred_col="pred_random_forest_hash",
        title="UPenn: Predicted vs Actual (RF_hash)",
        fname="02_upenn_pred_vs_actual_rf_hash.png"
    )

    # 3) Residual hist log10 — CatBoost
    plot_residual_hist_log10(
        df, outdir,
        actual_col="true_rate",
        pred_col="pred_catboost",
        title="UPenn: Residuals in log10 space (CatBoost)",
        fname="03_upenn_residual_hist_log10_catboost.png"
    )

    # 4) Residual hist log10 — RF_hash
    plot_residual_hist_log10(
        df, outdir,
        actual_col="true_rate",
        pred_col="pred_random_forest_hash",
        title="UPenn: Residuals in log10 space (RF_hash)",
        fname="04_upenn_residual_hist_log10_rf_hash.png"
    )

    # 5) Winner share
    plot_winner_share(df, outdir, fname="05_upenn_winner_share.png")

    # 6) Percent error distribution (clipped boxplot)  
    plot_pct_error_box(df, outdir, fname="06_upenn_pct_error_boxplot.png", clip=5.0)

    # Alternative #6:
    plot_error_by_setting(df, outdir, fname="06_upenn_error_by_setting_summary.png")

    print(f"[DONE] Saved visuals to: {outdir}")


if __name__ == "__main__":
    main()
