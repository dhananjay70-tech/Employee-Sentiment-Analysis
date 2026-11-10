"""True rolling 30-day negative-message flight-risk detection."""
from __future__ import annotations
from bisect import bisect_left, bisect_right
import logging
from numbers import Integral
import pandas as pd

logger = logging.getLogger(__name__)

def find_flight_risks(df: pd.DataFrame, employee_col: str, date_col: str, sentiment_col: str = "sentiment", threshold: int = 4, window_days: int = 30) -> pd.DataFrame:
    """Find qualifying rolling intervals containing threshold negative messages.

    Every qualifying minimal interval spans at most window_days elapsed days.
    Overlapping intervals are retained so every actual rolling trigger is auditable.
    """
    required = [employee_col, date_col, sentiment_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required flight-risk columns: {missing}")
    if (not isinstance(threshold, Integral) or isinstance(threshold, bool) or
            not isinstance(window_days, Integral) or isinstance(window_days, bool) or
            threshold < 1 or window_days < 1):
        raise ValueError("threshold and window_days must be positive integers")
    data = df.loc[df[sentiment_col].eq("Negative"), [employee_col, date_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data = data.dropna().sort_values([employee_col, date_col])
    episodes = []
    limit = pd.Timedelta(window_days, unit="D")
    for employee, group in data.groupby(employee_col, sort=True):
        times = group[date_col].tolist()
        for i in range(len(times) - threshold + 1):
            start, end = times[i], times[i + threshold - 1]
            if end - start <= limit:
                # Use sorted timestamps and binary searches to count closed-interval
                # messages, including ties that share the endpoint timestamp.
                count = bisect_right(times, end) - bisect_left(times, start)
                episodes.append({employee_col: employee, "window_start": start, "window_end": end,
                                 "negative_email_count": count, "threshold": threshold, "window_days": window_days})
    logger.info("Found %d qualifying rolling intervals across %d employees", len(episodes), data[employee_col].nunique())
    return pd.DataFrame(episodes, columns=[employee_col, "window_start", "window_end", "negative_email_count", "threshold", "window_days"])
