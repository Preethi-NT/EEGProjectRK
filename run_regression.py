#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_regression.py

Two modes:

A) SIMPLE (as-is): Use pre-standardized X and precomputed composite y.
   Example:
     python run_regression.py \
       --x data/ml_regression_X_std.csv \
       --y data/ml_regression_y_composite.csv \
       --model rf --cv 5 --outdir outputs_simple

B) RESEARCH-GRADE (fold-wise, no leakage):python run_regression.py --raw_csv data/ml_features_aligned_raw.csv --use_aligned_raw --model svr --cv 5 --outdir outputs_svr
 Start from raw or direction-aligned features.
   We will: direction-align (if needed) -> fit scaler on TRAIN only -> transform -> recompute composite per fold -> train/eval.
   Example (using aligned raw exported by build_composite.py):
     python run_regression.py \
       --raw_csv data/ml_features_aligned_raw.csv \
       --use_aligned_raw \
       --model rf --cv 5 --outdir outputs_foldwise
"""

import argparse
import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

META_COLS = ["subject", "window_idx", "file"]

FEATURES = [
    "alpha_power", "beta_power", "theta_power",
    "alpha_beta_ratio", "theta_beta_ratio",
    "alpha_rel_power", "beta_rel_power", "theta_rel_power",
]

DIRECTION = {
    "alpha_power": -1,
    "beta_power":  1,
    "theta_power": -1,
    "alpha_beta_ratio": -1,
    "theta_beta_ratio": -1,
    "alpha_rel_power": -1,
    "beta_rel_power":  1,
    "theta_rel_power": -1,
}

def parse_args():
    p = argparse.ArgumentParser(description="Train EEG window-wise regression model.")
    # SIMPLE mode inputs
    p.add_argument("--x", help="Path to X CSV (standardized features)")
    p.add_argument("--y", help="Path to y CSV (with 'composite' column)")

    # RESEARCH-GRADE (fold-wise) raw input
    p.add_argument("--raw_csv", default=None,
                   help="Path to raw window-wise feature CSV (original or direction-aligned). If given, fold-wise leakage-free training is used.")
    p.add_argument("--use_aligned_raw", action="store_true",
                   help="Set if --raw_csv is already direction-aligned. If not set, we will apply DIRECTION to FEATURES.")

    p.add_argument("--model", choices=["rf", "enet", "svr"], default="rf", help="Which model to train")
    p.add_argument("--test-size", type=float, default=0.2, help="(SIMPLE mode) Test size fraction for holdout")
    p.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", default="outputs", help="Directory to write artifacts")
    p.add_argument("--no-group", action="store_true", help="(SIMPLE mode) Force random split even if 'subject' exists")
    return p.parse_args()

def load_data_simple(x_path, y_path):
    X = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    # Extract y
    if "composite" in y_df.columns:
        y = y_df["composite"].copy()
    else:
        y = y_df.iloc[:, -1].copy()

    # Identify groups (subjects) if available
    groups = None
    if "subject" in X.columns:
        groups = X["subject"].copy()

    # Drop meta columns from X
    X = X.drop(columns=[c for c in META_COLS if c in X.columns], errors="ignore")
    return X, y, groups

def get_model(which: str, seed: int):
    if which == "rf":
        model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        params = {"n_estimators": 300, "random_state": seed, "n_jobs": -1}
    elif which == "enet":
        model = ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=seed, max_iter=10000)
        params = {"alpha": 0.01, "l1_ratio": 0.2, "random_state": seed, "max_iter": 10000}
    elif which == "svr":
        model = SVR(C=2.0, epsilon=0.05, kernel="rbf")
        params = {"C": 2.0, "epsilon": 0.05, "kernel": "rbf"}
    else:
        raise ValueError("Unknown model")
    return model, params

def direction_align(df_feat: pd.DataFrame) -> pd.DataFrame:
    X = df_feat[FEATURES].astype(float).copy()
    for c in FEATURES:
        X[c] = X[c] * DIRECTION[c]
    return X

def compute_composite_from_Z(Z_df: pd.DataFrame) -> np.ndarray:
    return Z_df.mean(axis=1).values

def group_aware_split(X, y, groups, test_size, seed, force_random=False):
    """SIMPLE mode holdout split."""
    if (groups is not None) and (not force_random):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        groups_train = groups_test = None
    return X_train, X_test, y_train, y_test, groups_train, groups_test

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ================================
    # MODE B: RESEARCH-GRADE (fold-wise, no leakage)
    # ================================
    if args.raw_csv:
        raw = pd.read_csv(args.raw_csv)

        # Determine groups if present
        groups = raw["subject"] if "subject" in raw.columns else None

        # Ensure required columns
        missing = [c for c in FEATURES if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required feature columns in --raw_csv: {missing}")

        # Direction alignment if needed
        if args.use_aligned_raw:
            X_raw = raw[FEATURES].astype(float).copy()
        else:
            X_raw = direction_align(raw)

        # Build CV splits
        if groups is not None:
            splitter = GroupKFold(n_splits=args.cv)
            splits = list(splitter.split(X_raw, np.zeros(len(X_raw)), groups))
        else:
            splitter = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
            splits = list(splitter.split(X_raw, np.zeros(len(X_raw))))

        fold_metrics = []
        y_true_all, y_pred_all = [], []

        for fold_idx, (tr, te) in enumerate(splits, 1):
            # Fit scaler on TRAIN ONLY
            scaler = StandardScaler().fit(X_raw.iloc[tr])

            # Transform train/test
            Z_train = pd.DataFrame(scaler.transform(X_raw.iloc[tr]), columns=FEATURES, index=X_raw.index[tr])
            Z_test  = pd.DataFrame(scaler.transform(X_raw.iloc[te]),  columns=FEATURES, index=X_raw.index[te])

            # Recompute composite y from standardized features
            y_train = compute_composite_from_Z(Z_train)
            y_test  = compute_composite_from_Z(Z_test)

            # Get model and fit
            model, params = get_model(args.model, args.seed)
            model.fit(Z_train, y_train)

            # Predict & score
            y_pred = model.predict(Z_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            fold_metrics.append({"fold": fold_idx, "mse": float(mse), "r2": float(r2)})
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

        # Aggregate CV scores
        r2_mean = float(np.mean([m["r2"] for m in fold_metrics]))
        r2_std  = float(np.std([m["r2"] for m in fold_metrics]))
        mse_mean = float(np.mean([m["mse"] for m in fold_metrics]))
        mse_std  = float(np.std([m["mse"] for m in fold_metrics]))

        # Save artifacts
        pd.DataFrame(fold_metrics).to_csv(os.path.join(args.outdir, "cv_fold_metrics.csv"), index=False)
        pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all}).to_csv(
            os.path.join(args.outdir, "cv_preds_concat.csv"), index=False
        )
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump({
                "mode": "fold-wise",
                "model": args.model,
                "cv": {"folds": args.cv, "r2_mean": r2_mean, "r2_std": r2_std, "mse_mean": mse_mean, "mse_std": mse_std},
                "note": "Leakage-free: direction-align (if needed) + train-only StandardScaler + y recomputed per fold."
            }, f, indent=2)

        print(json.dumps({"r2_mean": r2_mean, "r2_std": r2_std, "mse_mean": mse_mean, "mse_std": mse_std}, indent=2))
        return

    # ================================
    # MODE A: SIMPLE (pre-standardized X + precomputed y)
    # ================================
    if not args.x or not args.y:
        raise SystemExit("In SIMPLE mode, please provide both --x and --y (or use --raw_csv for fold-wise mode).")

    X, y, groups = load_data_simple(args.x, args.y)
    model, params = get_model(args.model, args.seed)

    # Holdout split
    X_train, X_test, y_train, y_test, groups_train, groups_test = group_aware_split(
        X, y, groups, args.test_size, args.seed, force_random=args.no_group
    )

    # Cross-validation on the TRAIN partition (optional sanity CV)
    if groups_train is not None:
        cv = GroupKFold(n_splits=args.cv)
        scores_r2 = cross_val_score(model, X_train, y_train, cv=cv.split(X_train, y_train, groups_train), scoring="r2")
        scores_mse = -cross_val_score(model, X_train, y_train, cv=cv.split(X_train, y_train, groups_train), scoring="neg_mean_squared_error")
    else:
        cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        scores_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        scores_mse = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")

    cv_report = {
        "cv_folds": args.cv,
        "r2_mean": float(np.mean(scores_r2)),
        "r2_std": float(np.std(scores_r2)),
        "mse_mean": float(np.mean(scores_mse)),
        "mse_std": float(np.std(scores_mse)),
    }

    # Fit on train and evaluate on test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save artifacts
    joblib.dump(model, os.path.join(args.outdir, f"regression_model_{args.model}.joblib"))
    pd.DataFrame({"y_true": y_test.reset_index(drop=True), "y_pred": y_pred}).to_csv(
        os.path.join(args.outdir, "test_predictions.csv"), index=False
    )
    metrics = {
        "mode": "simple",
        "model": args.model,
        "params": params,
        "test_mse": float(mse),
        "test_r2": float(r2),
        "cv": cv_report
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importances for RF
    if args.model == "rf":
        fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})\
                .sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(args.outdir, "feature_importances.csv"), index=False)

    with open(os.path.join(args.outdir, "README.txt"), "w") as f:
        f.write(f"""Artifacts written to: {args.outdir}

Files:
- regression_model_{args.model}.joblib : Trained model
- metrics.json                           : Test metrics + CV summary
- test_predictions.csv                   : y_true vs y_pred on test split
- feature_importances.csv                : Only for rf (simple mode)

Notes:
- SIMPLE mode expects already standardized X and precomputed composite y.
- FOLD-WISE mode (--raw_csv) avoids leakage by fitting the scaler on train only and recomputing y per fold.
""")

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
