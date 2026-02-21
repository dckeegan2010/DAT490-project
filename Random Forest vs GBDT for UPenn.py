"""
model comparison script:
1) RandomForestRegressor with HASH ENCODING for categoricals
2) CatBoostRegressor (GBDT) with native categoricals
3) Compare metrics + save side-by-side predictions

"""

# =========================
# CONFIG — EDIT THESE
# =========================

DATA_PATH = r"C:\Users\dckee\OneDrive\Documents\DAT490\price_files\UPenn_pricing_long.csv"
OUTDIR = r"C:\Users\dckee\OneDrive\Documents\DAT490\model_compare_hash_rf"

TARGET_COL = "negotiated_rate"

# ===================LOOK HERE======================
# ===================LOOK HERE======================
# ===================LOOK HERE======================

CB_USE_GPU = True  # set True only if NVIDIA CUDA GPU is ready

# ^^^^^^^^^^^^^^^^^^^LOOK HERE^^^^^^^^^^^^^^^^^^^^^^
# ===================LOOK HERE======================
# ===================LOOK HERE======================
# ===================LOOK HERE======================

# Categoricals used by both models
CATEGORICAL_COLS = [
    "cpt_code",
    "payer",
  # "hospital",        # include if present, remove if not in file, WILL NEED FOR CROSS HOSPITAL TEST
    "setting",
    "billing_class",
    "plan",
    "modifiers",
]

# Numeric inputs, optional, used by both models if present, but not in other hospital data
NUMERIC_COLS = [
    "standard_charge_gross",
    "standard_charge_discounted_cash",
]

TEST_SIZE = 0.10
RANDOM_STATE = 67 # lol I am a damn child

USE_LOG_TARGET = True

'''
==================================== 
Random Forest (hash-encoded) params
====================================
'''
RF_TREES = 150                 # start small, increase if runtime OK
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"
RF_N_JOBS = -1

# Hashing dimension: 2**16=65k, 2**17=131k, 2**18=262k
HASH_N_FEATURES = 2**16

# ---- CatBoost (GBDT) params ----
CB_ITER = 1500
CB_DEPTH = 8
CB_LR = 0.05
CB_VERBOSE_EVERY = 200





# =========================
# IMPORTS
# =========================

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor


# =========================
# HELPERS
# =========================

def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    # MAPE: avoid divide-by-zero by excluding true_rate == 0
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = np.nan

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape,          
        "MAPE_pct": mape * 100 if np.isfinite(mape) else np.nan,  # added (%)
        "n": int(len(y_true)),
        "n_mape": int(mask.sum()),  # how many rows used in MAPE
    }


def ensure_dir(path_str: str) -> Path:
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_token_dicts(X, feature_names=None):
    """
    Convert 2D input (DataFrame or ndarray) into list-of-dicts for FeatureHasher.
    """
    # If  DataFrame, get column names directly
    if hasattr(X, "columns"):
        cols = list(X.columns)
        values = X.to_numpy()
    else:
        # Otherwise its ndarray, need feature_names
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is not a DataFrame.")
        cols = feature_names
        values = X

    records = []
    for row in values:
        d = {}
        for c, v in zip(cols, row):
            if v is None:
                continue
            # handle missing values (NaN)
            if isinstance(v, float) and np.isnan(v):
                continue
            d[f"{c}={v}"] = 1
        records.append(d)
    return records



# =========================
# LOAD + CLEAN
# =========================

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Clean target
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()
print("Rows after cleaning:", len(df))

# Feature presence check
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]

if len(CATEGORICAL_COLS) == 0 and len(NUMERIC_COLS) == 0:
    raise RuntimeError("No usable feature columns found. Check CATEGORICAL_COLS/NUMERIC_COLS names.")

print("Categorical:", CATEGORICAL_COLS)
print("Numeric:", NUMERIC_COLS)

# y
y_raw = df[TARGET_COL].values
y_model = np.log1p(y_raw) if USE_LOG_TARGET else y_raw
print("Target transform:", "log1p" if USE_LOG_TARGET else "none")

X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()

# =========================
# SPLIT same split for both models
# =========================

idx = np.arange(len(df))
idx_train, idx_test = train_test_split(
    idx,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

X_train = X.iloc[idx_train].copy()
X_test = X.iloc[idx_test].copy()
y_train = y_model[idx_train]
y_test = y_model[idx_test]

print("Train rows:", len(X_train))
print("Test rows :", len(X_test))


# ===================================
# 1. RANDOM FOREST with HASH ENCODING
# ===================================

print("\nBuilding hashed-feature RandomForest pipeline...")

# Categorical -> impute -> dict tokens -> FeatureHasher (sparse)
cat_hash_pipe = Pipeline([
    ("select_cat", FunctionTransformer(lambda d: d[CATEGORICAL_COLS], validate=False)),
    ("impute_cat", SimpleImputer(strategy="most_frequent")),
    ("to_dict", FunctionTransformer(lambda X: to_token_dicts(X, feature_names=CATEGORICAL_COLS), validate=False)),
    ("hasher", FeatureHasher(n_features=HASH_N_FEATURES, input_type="dict")),
])

# Numeric -> impute (kept dense)
num_pipe = Pipeline([
    ("select_num", FunctionTransformer(lambda d: d[NUMERIC_COLS], validate=False)),
    ("impute_num", SimpleImputer(strategy="median")),
])

# ColumnTransformer combines hashed sparse + numeric dense
# Note: output will typically be sparse if hashing is used
preprocess = ColumnTransformer(
    transformers=[
        ("cat_hash", cat_hash_pipe, CATEGORICAL_COLS),
        ("num", num_pipe, NUMERIC_COLS),
    ],
    remainder="drop"
)

rf_model = RandomForestRegressor(
    n_estimators=RF_TREES,
    max_depth=RF_MAX_DEPTH,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    max_features=RF_MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=RF_N_JOBS
)

rf_pipe = Pipeline([
    ("prep", preprocess),
    ("rf", rf_model)
])

print("Training RandomForest (hash-encoded)...")
rf_pipe.fit(X_train, y_train)

print("Predicting RandomForest...")
rf_pred = rf_pipe.predict(X_test)

if USE_LOG_TARGET:
    y_true_dollars = np.expm1(y_test)
    rf_pred_dollars = np.expm1(rf_pred)
else:
    y_true_dollars = y_test
    rf_pred_dollars = rf_pred

rf_metrics = evaluate_predictions(y_true_dollars, rf_pred_dollars)
print("RF metrics:", rf_metrics)


# =========================
# 2. CATBOOST (GBDT)
# =========================

print("\nTraining CatBoost (GBDT)...")

# CatBoost needs categoricals as strings with no NaN
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for c in CATEGORICAL_COLS:
    X_train_cb[c] = X_train_cb[c].astype("string").fillna("MISSING")
    X_test_cb[c] = X_test_cb[c].astype("string").fillna("MISSING")

cat_feature_indices = [X_train_cb.columns.get_loc(c) for c in CATEGORICAL_COLS]

cb_params = dict(
    iterations=CB_ITER,
    depth=CB_DEPTH,
    learning_rate=CB_LR,
    loss_function="RMSE",
    random_seed=RANDOM_STATE,
    verbose=CB_VERBOSE_EVERY,
)
if CB_USE_GPU:
    cb_params["task_type"] = "GPU"

cb_model = CatBoostRegressor(**cb_params)

cb_model.fit(
    X_train_cb, y_train,
    cat_features=cat_feature_indices,
    eval_set=(X_test_cb, y_test),
    use_best_model=True
)

print("Predicting CatBoost...")
cb_pred = cb_model.predict(X_test_cb)

if USE_LOG_TARGET:
    cb_pred_dollars = np.expm1(cb_pred)
else:
    cb_pred_dollars = cb_pred

cb_metrics = evaluate_predictions(y_true_dollars, cb_pred_dollars)
print("CatBoost metrics:", cb_metrics)


# =========================
# 3. COMPARE + SAVE OUTPUTS
# =========================

outdir = ensure_dir(OUTDIR)

# Metrics table
metrics_df = pd.DataFrame([
    {"model": "RandomForest_hash", **rf_metrics, "hash_n_features": HASH_N_FEATURES, "rf_trees": RF_TREES},
    {"model": "CatBoost_GBDT", **cb_metrics, "iterations": CB_ITER, "depth": CB_DEPTH},
]).sort_values(["MAE", "RMSE"], ascending=True)

metrics_df.to_csv(outdir / "model_metrics_comparison.csv", index=False)

# Readable predictions table from the original test rows 
# Include identifiers to groupby payer / CPT / plan, etc
id_cols = [c for c in ["cpt_code", "payer", "hospital", "setting", "billing_class", "plan", "modifiers"] if c in df.columns]

preds_out = df.loc[idx_test, id_cols].copy()

# Add true rate + predictions
preds_out["true_rate"] = y_true_dollars
preds_out["pred_random_forest_hash"] = rf_pred_dollars
preds_out["pred_catboost"] = cb_pred_dollars

# Errors
preds_out["abs_err_rf"] = np.abs(preds_out["true_rate"] - preds_out["pred_random_forest_hash"])
preds_out["abs_err_cb"] = np.abs(preds_out["true_rate"] - preds_out["pred_catboost"])

preds_out["pct_err_rf"] = preds_out["abs_err_rf"] / np.maximum(preds_out["true_rate"], 1e-9)
preds_out["pct_err_cb"] = preds_out["abs_err_cb"] / np.maximum(preds_out["true_rate"], 1e-9)

# which model was closer on each row, winner winner chicken dinner!
preds_out["winner"] = np.where(preds_out["abs_err_cb"] < preds_out["abs_err_rf"], "catboost", "random_forest")

preds_out.to_csv(outdir / "predictions_side_by_side_with_ids.csv", index=False)

# Save CatBoost model
cb_model.save_model(str(outdir / "catboost_model.cbm"))

print("\n=== SUMMARY (lower MAE/RMSE is better) ===")
print(metrics_df)

print("\nSaved metrics :", outdir / "model_metrics_comparison.csv")
print("Saved preds   :", outdir / "predictions_side_by_side_with_ids.csv")
print("Saved model   :", outdir / "catboost_model.cbm")
print("Done.")


"""
Notes for me

TUNING TIPS from 401:
- Reduce RF_TREES to 80–150 for iteration, then scale up
- Reduce HASH_N_FEATURES to 2**16 if memory is low
- Increase RF_MIN_SAMPLES_LEAF (like 5 or 10) to speed training and reduce overfitting
"""
