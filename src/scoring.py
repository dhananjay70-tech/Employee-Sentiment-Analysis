"""Monthly employee sentiment scores and deterministic rankings."""
from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)
SCORE_MAP = {"Positive": 1, "Neutral": 0, "Negative": -1}


def monthly_scores(df: pd.DataFrame, employee_col: str, date_col: str, sentiment_col: str = "sentiment") -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [employee_col, date_col, sentiment_col]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required scoring columns: {missing_cols}")
    if len(set(required)) != len(required):
        raise ValueError("Employee, date, and sentiment columns must be distinct")
    data = df[[employee_col, date_col, sentiment_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data = data.dropna(subset=[employee_col, date_col, sentiment_col]).copy()
    unknown = set(data[sentiment_col].unique()) - set(SCORE_MAP)
    if unknown:
        raise ValueError(f"Unsupported sentiment labels: {sorted(unknown)}")
    data["sentiment_score"] = data[sentiment_col].map(SCORE_MAP).astype(int)
    data["month"] = data[date_col].dt.strftime("%Y-%m")
    data = data.sort_values([employee_col, date_col], kind="stable").reset_index(drop=True)
    data["monthly_running_score"] = data.groupby([employee_col, "month"], sort=False)["sentiment_score"].cumsum()
    monthly = (data.groupby([employee_col, "month"], as_index=False)
               .agg(monthly_score=("sentiment_score", "sum"), message_count=(sentiment_col, "size"),
                    positive_count=(sentiment_col, lambda x: int(x.eq("Positive").sum())),
                    neutral_count=(sentiment_col, lambda x: int(x.eq("Neutral").sum())),
                    negative_count=(sentiment_col, lambda x: int(x.eq("Negative").sum())))
               .sort_values(["month", employee_col], kind="stable").reset_index(drop=True))
    logger.info("Scored %d messages into %d employee-months", len(data), len(monthly))
    return data, monthly


def monthly_rankings(monthly: pd.DataFrame, employee_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"month", employee_col, "monthly_score", "message_count", "positive_count", "neutral_count", "negative_count"}
    missing = sorted(required - set(monthly.columns))
    if missing:
        raise ValueError(f"Missing required ranking columns: {missing}")
    positives, negatives = [], []
    for month, group in monthly.groupby("month", sort=True):
        pos = group[group.monthly_score > 0].sort_values(["monthly_score", employee_col], ascending=[False, True]).head(3).copy()
        neg = group[group.monthly_score < 0].sort_values(["monthly_score", employee_col], ascending=[True, True]).head(3).copy()
        pos.insert(0, "rank", range(1, len(pos) + 1)); neg.insert(0, "rank", range(1, len(neg) + 1))
        positives.append(pos); negatives.append(neg)
    cols = ["rank", "month", employee_col, "monthly_score", "message_count", "positive_count", "neutral_count", "negative_count"]
    positive = pd.concat(positives, ignore_index=True)[cols] if positives else pd.DataFrame(columns=cols)
    negative = pd.concat(negatives, ignore_index=True)[cols] if negatives else pd.DataFrame(columns=cols)
    logger.info("Created monthly rankings for %d months", monthly["month"].nunique())
    return positive, negative
