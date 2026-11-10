"""Leakage-aware Linear Regression baseline for next-active-month scores."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)
FEATURES = ["message_count", "avg_word_count", "avg_message_length", "prior_month_score", "trailing_3_month_score_mean"]
SCORE_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}


def make_forecast_frame(messages: pd.DataFrame, employee_col: str, date_col: str, text_col: str, sentiment_col: str = "sentiment") -> pd.DataFrame:
    """Build features from each observed month for the next observed active month."""
    required = [employee_col, date_col, text_col, sentiment_col]
    missing = [col for col in required if col not in messages.columns]
    if missing:
        raise ValueError(f"Missing required forecasting columns: {missing}")
    data = messages.dropna(subset=[employee_col, date_col, sentiment_col]).copy()
    data = data[data[sentiment_col].isin(SCORE_MAP)].copy()
    if data.empty:
        return pd.DataFrame(columns=[employee_col, "month", *FEATURES, "target_month", "target_score"])
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data = data.dropna(subset=[date_col])
    data["month"] = data[date_col].dt.strftime("%Y-%m")
    data["_word_count"] = data[text_col].fillna("").astype(str).str.split().str.len()
    data["_message_length"] = data[text_col].fillna("").astype(str).str.len()
    data["_score"] = data[sentiment_col].map(SCORE_MAP).astype(int)
    agg = data.groupby([employee_col, "month"], as_index=False).agg(
        monthly_score=("_score", "sum"), message_count=("_score", "size"),
        avg_word_count=("_word_count", "mean"), avg_message_length=("_message_length", "mean"))
    agg["_month_ts"] = pd.to_datetime(agg["month"] + "-01", utc=True)
    agg = agg.sort_values([employee_col, "_month_ts"], kind="stable")
    grouped = agg.groupby(employee_col, sort=False)
    agg["prior_month_score"] = grouped["monthly_score"].shift(1)
    agg["trailing_3_month_score_mean"] = grouped["monthly_score"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    agg["target_month"] = grouped["month"].shift(-1)
    agg["target_score"] = grouped["monthly_score"].shift(-1)
    return agg.dropna(subset=["prior_month_score", "target_month", "target_score"]).reset_index(drop=True)


def evaluate_forecast(frame: pd.DataFrame, employee_col: str, test_fraction: float = 0.2) -> tuple[dict, pd.DataFrame]:
    """Train on earlier target months and evaluate on a later chronological holdout."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    base = {"target": "next observed active-month employee sentiment score", "features": FEATURES,
            "model": "LinearRegression", "train_rows": 0, "test_rows": 0,
            "MAE": None, "MSE": None, "RMSE": None, "R2": None,
            "note": None}
    if frame.empty:
        base["note"] = "No sufficient employee history to create target examples."
        return base, pd.DataFrame(columns=[employee_col, "target_month", "target_score", "predicted_score", "absolute_error"])
    missing = [c for c in [employee_col, "target_month", "target_score", *FEATURES] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")
    months = sorted(frame["target_month"].dropna().unique())
    if len(months) < 2:
        base["note"] = f"Insufficient history for chronological train/test evaluation ({len(months)} target month(s))."
        return base, pd.DataFrame(columns=[employee_col, "target_month", "target_score", "predicted_score", "absolute_error"])
    cutoff_idx = max(1, int(np.floor(len(months) * (1 - test_fraction))))
    cutoff_idx = min(cutoff_idx, len(months) - 1)
    cutoff = months[cutoff_idx]
    train = frame[frame["target_month"] < cutoff].copy()
    test = frame[frame["target_month"] >= cutoff].copy()
    if len(train) < 2 or test.empty:
        base["note"] = f"Insufficient rows after chronological split (train={len(train)}, test={len(test)})."
        return base, pd.DataFrame(columns=[employee_col, "target_month", "target_score", "predicted_score", "absolute_error"])
    model = LinearRegression().fit(train[FEATURES], train["target_score"])
    predictions = model.predict(test[FEATURES])
    actual = test["target_score"].to_numpy()
    mse = float(mean_squared_error(actual, predictions))
    base.update({"train_rows": len(train), "test_rows": len(test), "cutoff_month": str(cutoff),
                 "MAE": float(mean_absolute_error(actual, predictions)), "MSE": mse,
                 "RMSE": float(np.sqrt(mse)),
                 "R2": float(r2_score(actual, predictions)) if len(actual) > 1 else None,
                 "coefficients": dict(zip(FEATURES, model.coef_.astype(float).tolist())),
                 "intercept": float(model.intercept_),
                 "note": "R² omitted because the holdout has one observation." if len(actual) == 1 else "Exploratory chronological holdout; interpret cautiously."})
    predictions_df = test[[employee_col, "target_month", "target_score"]].copy()
    predictions_df["predicted_score"] = predictions
    predictions_df["absolute_error"] = np.abs(actual - predictions)
    logger.info("Evaluated Linear Regression on %d training and %d test rows", len(train), len(test))
    return base, predictions_df
