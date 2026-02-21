# -*- coding: utf-8 -*-
"""
Cross-hospital model:
Train: UPenn + Temple + Cooper
Test : Jefferson

Uses enriched combined file with:
- hospital_name
- total_cases
- market_share
- ACS demographic columns
"""

# =========================
# CONFIG — EDIT THESE
# =========================

DATA_PATH = r"C:\Users\dckee\OneDrive\Documents\DAT490\price_files\ALL_HOSPITALS_pricing_long_TOP10_with_marketshare_demographics.csv"
OUTDIR = r"C:\Users\dckee\OneDrive\Documents\DAT490\model_cross_hospital"

TARGET_COL = "negotiated_rate"

TRAIN_HOSPITALS = ["UPenn", "Temple", "Cooper"]
TEST_HOSPITALS  = ["Jefferson"]

CB_USE_GPU = True
USE_LOG_TARGET = True
RANDOM_STATE = 67

# ============
# FEATURES
# =============

CATEGORICAL_COLS = [
    "cpt_code",
    "payer",
    "setting",
    "billing_class",
    "plan",
    "modifiers",
]

 
NUMERIC_COLS = [
    "standard_charge_gross",
    "standard_charge_discounted_cash",

    # new market structure fields
    "total_cases",
    "market_share",

    # new demographic fields
    "total_population",
    "median_household_income",
    "pct_below_poverty",
    "pct_white_alone",
    "pct_black_alone",
    "pct_asian_alone",
    "pct_hispanic_any_race",
]

# =============
# MODEL PARAMS 
# =============

RF_TREES = 200
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = "sqrt"
RF_N_JOBS = -1
HASH_N_FEATURES = 2**17

CB_ITER = 3000
CB_DEPTH = 8
CB_LR = 0.05
CB_VERBOSE_EVERY = 200

# ========
# IMPORTS
# ========

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

# =========
# HELPERS
# =========

def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def to_token_dicts(X, feature_names=None):
    if hasattr(X, "columns"):
        cols = list(X.columns)
        values = X.to_numpy()
    else:
        cols = feature_names
        values = X

    recs = []
    for row in values:
        d = {}
        for c, v in zip(cols, row):
            if pd.notna(v):
                d[f"{c}={v}"] = 1
        recs.append(d)
    return recs

# =========================
# LOAD
# =========================

print("Loading combined data. Here we go!!!...")

# =========
# fix for bad rows
# ==========

bad_lines = []

def bad_line_handler(bad_line):
    bad_lines.append(bad_line)
    return None  # skip

print("Loading combined data (skip malformed rows)...")
df = pd.read_csv(
    DATA_PATH,
    engine="python",
    on_bad_lines=bad_line_handler,
  # low_memory=False 
)

print("Loaded rows:", len(df))
print("Bad lines skipped:", len(bad_lines))

    
# =========================================================



df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL, "hospital_name"]).copy()

# keep only columns that exist
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df.columns]

print("Categorical:", CATEGORICAL_COLS)
print("Numeric:", NUMERIC_COLS)


print("Unique hospital_name values (first 50):")
print(df["hospital_name"].astype(str).str.strip().value_counts().head(50))


# =========================
# TRAIN / TEST SPLIT BY HOSPITAL
# =========================

train_df = df[df["hospital_name"].isin(TRAIN_HOSPITALS)].copy()
test_df  = df[df["hospital_name"].isin(TEST_HOSPITALS)].copy()

print("Train hospitals:", train_df["hospital_name"].value_counts().to_dict())
print("Test hospitals :", test_df["hospital_name"].value_counts().to_dict())

X_train = train_df[CATEGORICAL_COLS + NUMERIC_COLS]
X_test  = test_df[CATEGORICAL_COLS + NUMERIC_COLS]

y_train_raw = train_df[TARGET_COL].values
y_test_raw  = test_df[TARGET_COL].values

y_train = np.log1p(y_train_raw) if USE_LOG_TARGET else y_train_raw
y_test  = np.log1p(y_test_raw) if USE_LOG_TARGET else y_test_raw

# =========================
# RANDOM FOREST — HASHED
# =========================

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("to_dict", FunctionTransformer(lambda X: to_token_dicts(X, CATEGORICAL_COLS), validate=False)),
    ("hash", FeatureHasher(n_features=HASH_N_FEATURES, input_type="dict")),
])

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
])

prep = ColumnTransformer([
    ("cat", cat_pipe, CATEGORICAL_COLS),
    ("num", num_pipe, NUMERIC_COLS),
])

rf = RandomForestRegressor(
    n_estimators=RF_TREES,
    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
    max_features=RF_MAX_FEATURES,
    n_jobs=RF_N_JOBS,
    random_state=RANDOM_STATE
)

rf_pipe = Pipeline([("prep", prep), ("rf", rf)])

print("\nTraining RandomForest...")
rf_pipe.fit(X_train, y_train)

rf_pred = rf_pipe.predict(X_test)

# =========================
# CATBOOST
# =========================

print("\nTraining CatBoost...")

X_train_cb = X_train.copy()
X_test_cb  = X_test.copy()

for c in CATEGORICAL_COLS:
    X_train_cb[c] = X_train_cb[c].astype("string").fillna("MISSING")
    X_test_cb[c]  = X_test_cb[c].astype("string").fillna("MISSING")

cat_idx = [X_train_cb.columns.get_loc(c) for c in CATEGORICAL_COLS]

cb = CatBoostRegressor(
    iterations=CB_ITER,
    depth=CB_DEPTH,
    learning_rate=CB_LR,
    loss_function="RMSE",
    random_seed=RANDOM_STATE,
    verbose=CB_VERBOSE_EVERY,
    task_type="GPU" if CB_USE_GPU else "CPU"
)

cb.fit(X_train_cb, y_train, cat_features=cat_idx)

cb_pred = cb.predict(X_test_cb)

# =========================
# BACK-TRANSFORM
# =========================

if USE_LOG_TARGET:
    y_true = np.expm1(y_test)
    rf_pred = np.expm1(rf_pred)
    cb_pred = np.expm1(cb_pred)
else:
    y_true = y_test

# =========================
# METRICS
# =========================

rf_m = evaluate_predictions(y_true, rf_pred)
cb_m = evaluate_predictions(y_true, cb_pred)

print("\nRF:", rf_m)
print("CB:", cb_m)

# =========================
# SAVE
# =========================

outdir = ensure_dir(OUTDIR)

pd.DataFrame([
    {"model": "RF_hash", **rf_m},
    {"model": "CatBoost", **cb_m}
]).to_csv(outdir / "cross_hospital_metrics.csv", index=False)

print("\nSaved metrics to", outdir)

# =================================
# EXPORT TEST SET FOR VISUALIZATION 
# =================================


EXPORT_DIR = OUTDIR  # reuse your model output folder
os.makedirs(EXPORT_DIR, exist_ok=True)

print("\nBuilding visualization-ready test dataset...")

# start from original test rows
viz_df = test_df.copy()

#
viz_df["true_negotiated_rate"] = y_true

# add predictions
viz_df["pred_random_forest"] = rf_pred
viz_df["pred_catboost"] = cb_pred

# error metrics per row
viz_df["abs_error_rf"] = np.abs(viz_df["true_negotiated_rate"] - viz_df["pred_random_forest"])
viz_df["abs_error_cb"] = np.abs(viz_df["true_negotiated_rate"] - viz_df["pred_catboost"])

viz_df["pct_error_rf"] = viz_df["abs_error_rf"] / np.maximum(viz_df["true_negotiated_rate"], 1e-9)
viz_df["pct_error_cb"] = viz_df["abs_error_cb"] / np.maximum(viz_df["true_negotiated_rate"], 1e-9)

viz_df["better_model"] = np.where(
    viz_df["abs_error_cb"] < viz_df["abs_error_rf"],
    "catboost",
    "random_forest"
)

# derived fields for charts
if "market_share" in viz_df.columns:
    viz_df["market_share_pct"] = viz_df["market_share"] * 100

viz_df["log_true_rate"] = np.log1p(viz_df["true_negotiated_rate"])
viz_df["log_pred_cb"] = np.log1p(viz_df["pred_catboost"])
viz_df["log_pred_rf"] = np.log1p(viz_df["pred_random_forest"])

# reorder key columns to front if present
front_cols = [
    "hospital_name",
    "proc_code" if "proc_code" in viz_df.columns else "cpt_code",
    "payer",
    "plan",
    "setting",
    "billing_class",
    "true_negotiated_rate",
    "pred_catboost",
    "pred_random_forest",
    "abs_error_cb",
    "abs_error_rf",
    "pct_error_cb",
    "pct_error_rf",
    "better_model",
]

front_cols = [c for c in front_cols if c in viz_df.columns]
other_cols = [c for c in viz_df.columns if c not in front_cols]
viz_df = viz_df[front_cols + other_cols]

# write full file
full_path = os.path.join(EXPORT_DIR, "test_set_predictions_full.csv")
viz_df.to_csv(
    full_path,
    index=False,
    encoding="utf-8",
    lineterminator="\n",
    quoting=csv.QUOTE_MINIMAL,
    escapechar="\\"
)

print("Saved:", full_path)
print("Rows:", len(viz_df))

# ---- also write a slim version for quick viz tools ----
slim_cols = [c for c in front_cols if c in viz_df.columns]
slim_path = os.path.join(EXPORT_DIR, "test_set_predictions_slim.csv")
viz_df[slim_cols].to_csv(slim_path, index=False)

print("Saved:", slim_path)
